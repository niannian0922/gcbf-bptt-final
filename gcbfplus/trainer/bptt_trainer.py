from typing import Any, Dict, Tuple, Optional
import os

import torch
from torch import nn
from torch.optim import Adam
import wandb

from ..policy.bptt_policy import BPTTPolicy


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

        # The policy now requires the full config to find the 'losses' block for the adaptive weight head
        self.policy = BPTTPolicy(self.config).to(self.device)

        self.optimizer = Adam(self.policy.parameters(), lr=self.learning_rate)

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
        
        # Initialize previous goal distance for progress reward calculation
        previous_goal_distance = self.env.get_goal_distance(init_state).detach()
        
        # Initialize intervention tracking for Guardian Protocol
        total_interventions = torch.zeros((), device=self.device)

        for _ in range(self.horizon):
            observations = self.env.get_observation(state)
            if hasattr(observations, "to"):
                observations = observations.to(self.device)

            # 1. Get nominal action and safety confidence from the policy
            policy_outputs = self.policy(observations)
            nominal_action = policy_outputs['action']
            alpha = policy_outputs.get('alpha')

            # --- START OF GUARDIAN PROTOCOL ---
            losses_cfg = self.config.get('losses', {})
            guardian_threshold = losses_cfg.get('guardian_threshold', 0.5)

            # The Guardian's Verdict: Check if the policy's proposed action is unsafe.
            intervention_mask = (alpha < guardian_threshold).float().detach() if alpha is not None else torch.zeros_like(nominal_action[:, :1])

            # Define the guaranteed safe action: a gentle braking maneuver.
            safe_backup_action = -0.5 * state.velocity  # Gently brakes by applying a force opposite to velocity.

            # The Guardian's Action: Override the nominal action if intervention is triggered.
            action_to_execute = (1 - intervention_mask) * nominal_action + intervention_mask * safe_backup_action
            log_alpha = alpha if alpha is not None else torch.ones(nominal_action.shape[0], 1, device=self.device)
            # --- END OF GUARDIAN PROTOCOL ---

            step_result = self.env.step(state, action_to_execute, alpha)
            next_state = step_result.next_state
            
            # --- DYNAMIC LOSS CALCULATION ---
            goal_distances = self.env.get_goal_distance(next_state)
            current_goal_distance = goal_distances
            avg_goal_distance_step = torch.mean(goal_distances)
            sum_goal_distance = sum_goal_distance + avg_goal_distance_step
            
            # --- START OF PROGRESS REWARD LOGIC ---
            losses_cfg = self.config.get('losses', {})
            progress_reward_weight = losses_cfg.get('progress_reward_weight', 0.0)

            # The reward is the reduction in goal distance. A negative value is a penalty for moving away.
            progress = previous_goal_distance - current_goal_distance
            progress_reward = progress_reward_weight * torch.mean(progress)

            # Update for the next iteration
            previous_goal_distance = current_goal_distance.detach()
            # --- END OF PROGRESS REWARD LOGIC ---
            
            # 3. Calculate losses based on the new Guardian protocol
            track_cost = losses_cfg.get('goal_weight', 1.0) * torch.mean(current_goal_distance ** 2)
            ctrl_cost = 1e-3 * torch.sum(action_to_execute ** 2)

            # Gated Alpha Regularization
            safety_gate_threshold = losses_cfg.get('safety_gate_threshold', 0.1)
            is_safe_mask = (step_result.cost == 0).float().detach()
            alpha_reg_loss = losses_cfg.get('alpha_reg_weight', 0.05) * torch.mean(is_safe_mask * ((1.0 - log_alpha) ** 2)) if alpha is not None else 0

            # The Guardian's Penalty (NEW!)
            guardian_intervention_penalty = losses_cfg.get('guardian_intervention_penalty', 10.0)
            intervention_loss = guardian_intervention_penalty * torch.mean(intervention_mask)
            
            # Track total interventions for logging
            total_interventions = total_interventions + torch.sum(intervention_mask)

            # 4. Final step loss
            step_loss = track_cost + ctrl_cost - progress_reward + alpha_reg_loss + intervention_loss
            # --- END OF DYNAMIC LOSS CALCULATION ---

            # Collision metric
            collision_event = (step_result.cost > 0).to(self.device)
            collision_count = collision_count + torch.sum(collision_event.float())

            cumulative_loss = cumulative_loss + step_loss

            # Accumulate alpha for averaging
            sum_alpha = sum_alpha + torch.mean(log_alpha)

            state = next_state

        avg_goal_distance = sum_goal_distance / float(self.horizon)
        collision_rate = collision_count / float(self.horizon)
        avg_alpha = sum_alpha / float(self.horizon)
        intervention_rate = total_interventions / float(self.horizon)
        return cumulative_loss, avg_goal_distance, collision_rate, avg_alpha, intervention_rate

    def train(self, num_steps: Optional[int] = None) -> None:
        self.policy.train()
        init_state = self.env.reset()

        max_steps: int = int(num_steps) if num_steps is not None else int(self.num_steps)
        for step_idx in range(max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            total_loss, avg_goal_distance, collision_rate, avg_alpha, intervention_rate = self.rollout(init_state)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
            self.optimizer.step()

            wandb.log(
                {
                    "loss": float(total_loss.item()),
                    "goal_distance": float(avg_goal_distance.item()),
                    "collision_rate": float(collision_rate.item()),
                    "avg_alpha_confidence": float(avg_alpha.item()),
                    "guardian_intervention_rate": float(intervention_rate.item()),
                },
                step=self.global_step,
            )
            self.global_step += 1

            if (step_idx + 1) % 10 == 0:
                print(
                    f"Step {step_idx + 1}/{self.num_steps} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"GoalDist: {avg_goal_distance.item():.4f} | "
                    f"CollRate: {collision_rate.item():.4f}"
                )

            # Periodic checkpoint saving
            if self.global_step % int(self.save_interval) == 0:
                step_dir = os.path.join(self.model_save_path, str(self.global_step))
                os.makedirs(step_dir, exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(step_dir, "policy.pt"))
                print(f"Checkpoint saved at step {self.global_step} -> {step_dir}")


