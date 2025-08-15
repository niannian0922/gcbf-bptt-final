from typing import Any, Dict, Tuple, Optional
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import wandb

from ..policy.bptt_policy import BPTTPolicy
from ..policy.guardian_network import GuardianNetwork


def check_nan(tensor: torch.Tensor, name: str):
    """Checks if a tensor contains NaN or Inf values and raises an error if it does."""
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"NaN or Inf detected in '{name}'")


class SimpleMLPPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class BPTTTrainer:
    def __init__(
        self,
        env: Any,
        trainer_cfg: Dict[str, Any],
        policy_cfg: Dict[str, Any],
        device: torch.device,
        full_config: Dict[str, Any],
    ) -> None:
        self.env = env
        self.device = device
        self.config: Dict[str, Any] = full_config

        # Config-driven hyperparameters (robust to different schema variants)
        training_cfg: Dict[str, Any] = full_config.get("training", {})
        self.horizon: int = int(
            trainer_cfg.get("bptt", {}).get("horizon_length",
            training_cfg.get("horizon_length", 64))
        )
        self.num_steps: int = int(
            trainer_cfg.get("trainer", {}).get("num_steps",
            training_cfg.get("training_steps", 1000))
        )
        self.learning_rate: float = float(
            trainer_cfg.get("optim", {}).get("lr",
            training_cfg.get("learning_rate", 1e-3))
        )

        # Logging / naming
        self.run_name: str = str(full_config.get("run_name", trainer_cfg.get("run_name", "default")))
        self.log_dir: str = str(trainer_cfg.get("log_dir", "logs"))
        self.model_save_path: str = os.path.join(self.log_dir, self.run_name, "models")
        os.makedirs(self.model_save_path, exist_ok=True)

        # Checkpointing interval
        self.save_interval: int = int(training_cfg.get("save_interval", 1000))
        self.global_step: int = 0

        # Get environment dimensions for configuration validation
        obs_dim = int(self.env.observation_shape[0]) if isinstance(self.env.observation_shape, tuple) else int(self.env.observation_shape)
        action_dim = int(self.env.action_shape[0]) if isinstance(self.env.action_shape, tuple) else int(self.env.action_shape)

        # --- START OF __init__ REFACTOR ---
        # Get the full network configuration block
        networks_cfg = self.config.get('networks', {})
        pilot_cfg = networks_cfg.get('pilot_network', {})
        guardian_cfg = networks_cfg.get('guardian_network', {})
        training_cfg = self.config.get('training', {})

        # Initialize single policy with adaptive loss weights
        self.policy = BPTTPolicy(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.max_grad_norm = training_cfg.get('max_grad_norm', 10.0)
        # --- END OF __init__ REFACTOR ---

        # Initialize Weights & Biases
        wandb_config: Dict[str, Any] = full_config.get("wandb", {})
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=self.run_name,
            config=self.config,
            mode=wandb_config.get("mode", "online"),
        )

    def rollout(self, init_state: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = init_state
        cumulative_loss = torch.zeros((), device=self.device)
        sum_goal_distance = torch.zeros((), device=self.device)
        collision_count = torch.zeros((), device=self.device)
        sum_alpha = torch.zeros((), device=self.device)

        for _ in range(self.horizon):
            observations = self.env.get_observation(state)
            check_nan(observations, "observations")

            policy_outputs = self.policy(observations)
            nominal_action = policy_outputs['action']
            alpha = policy_outputs.get('alpha')
            check_nan(nominal_action, "policy_outputs[action]")
            if alpha is not None:
                check_nan(alpha, "policy_outputs[alpha]")

            safe_backup_action = torch.zeros_like(nominal_action)
            if alpha is None:
                final_action = nominal_action
                log_alpha = torch.ones(nominal_action.shape[0], 1, device=self.device)
            else:
                final_action = alpha * nominal_action + (1 - alpha) * safe_backup_action
                log_alpha = alpha

            sum_alpha = sum_alpha + torch.mean(log_alpha)
            check_nan(final_action, "final_action")

            step_result = self.env.step(state, final_action, alpha)
            next_state = step_result.next_state
            check_nan(next_state.position, "next_state.position")
            check_nan(next_state.velocity, "next_state.velocity")

            goal_distances = self.env.get_goal_distance(next_state)
            losses_cfg = self.config.get('losses', {})

            if 'loss_weights' in policy_outputs:
                dynamic_weights = policy_outputs['loss_weights']
                goal_w = dynamic_weights['goal_weight']
                alpha_reg_w = dynamic_weights.get('alpha_reg_weight', 0.0) # Use .get for safety
            else:
                goal_w = losses_cfg.get('goal_weight', 1.0)
                alpha_reg_w = losses_cfg.get('alpha_reg_weight', 0.0)

            track_cost = torch.mean(goal_w * (goal_distances ** 2))
            ctrl_cost = 1e-3 * torch.sum(final_action ** 2)

            is_safe_mask = (step_result.cost == 0).float().detach()
            gated_alpha_reg_loss = torch.mean(alpha_reg_w * is_safe_mask * ((1.0 - log_alpha) ** 2))

            step_loss = track_cost + ctrl_cost + gated_alpha_reg_loss
            check_nan(step_loss, "step_loss")

            cumulative_loss = cumulative_loss + step_loss
            sum_goal_distance = sum_goal_distance + torch.mean(goal_distances)
            collision_count = collision_count + torch.sum((step_result.cost > 0).float())
            state = next_state

        avg_goal_distance = sum_goal_distance / float(self.horizon)
        collision_rate = collision_count / float(self.horizon)
        avg_alpha = sum_alpha / float(self.horizon)
        return cumulative_loss, avg_goal_distance, collision_rate, avg_alpha

    def train(self, num_steps: Optional[int] = None) -> None:
        self.policy.train()
        init_state = self.env.reset()

        max_steps: int = int(num_steps) if num_steps is not None else int(self.num_steps)
        for step_idx in range(max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            total_loss, avg_goal_distance, collision_rate, avg_alpha = self.rollout(init_state)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            wandb.log({
                "loss": float(total_loss.item()),
                "goal_distance": float(avg_goal_distance.item()),
                "collision_rate": float(collision_rate.item()),
                "avg_alpha_confidence": float(avg_alpha.item()),
            }, step=self.global_step)
            
            self.global_step += 1

            if (step_idx + 1) % 10 == 0:
                print(
                    f"Step {step_idx + 1}/{self.num_steps} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"GoalDist: {avg_goal_distance.item():.4f} | "
                    f"CollRate: {collision_rate.item():.4f} | "
                    f"Alpha: {avg_alpha.item():.4f}"
                )

            # Periodic checkpoint saving
            if self.global_step % int(self.save_interval) == 0:
                step_dir = os.path.join(self.model_save_path, str(self.global_step))
                os.makedirs(step_dir, exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(step_dir, "policy.pt"))
                print(f"Checkpoint saved at step {self.global_step} -> {step_dir}")


