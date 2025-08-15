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

        # NEW: We now pass the entire policy config dictionary to the advanced BPTTPolicy
        self.policy = BPTTPolicy(policy_cfg).to(self.device)

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

        for _ in range(self.horizon):
            observations = self.env.get_observation(state)
            if hasattr(observations, "to"):
                observations = observations.to(self.device)

            # --- START OF PROBABILISTIC SHIELD LOGIC ---

            # 1. Get the nominal action and safety confidence (alpha) from our advanced policy
            # The policy now returns a tuple: (action, alpha, margin)
            nominal_action, alpha, _ = self.policy(observations)

            # 2. Define the safe backup action. For now, we use the simplest and most robust option: hovering (zero action).
            safe_backup_action = torch.zeros_like(nominal_action)

            # 3. Check if the policy is configured to predict alpha. If not, default to full trust in the policy.
            if alpha is None:
                final_action = nominal_action
                # Set a default alpha of 1.0 for logging purposes
                log_alpha = torch.ones(nominal_action.shape[0], 1, device=self.device)
            else:
                # 4. Implement the core blending logic of the shield!
                final_action = alpha * nominal_action + (1 - alpha) * safe_backup_action
                log_alpha = alpha

            # --- END OF PROBABILISTIC SHIELD LOGIC ---

            step_result = self.env.step(state, final_action, alpha) # Pass alpha to step for potential use
            next_state = step_result.next_state

            # --- Loss Calculation with new Alpha Regularization ---
            goal_distances = self.env.get_goal_distance(next_state)
            avg_goal_distance_step = torch.mean(goal_distances)
            sum_goal_distance = sum_goal_distance + avg_goal_distance_step

            # Control effort regularization (on the final action)
            ctrl_cost = 1e-3 * torch.sum(final_action ** 2)

            # Tracking cost: squared goal distance
            track_cost = torch.sum(goal_distances ** 2)

            # --- START OF SAFETY-GATED ALPHA REGULARIZATION ---

            # 1. Get the safety gate threshold from the config.
            losses_cfg = self.config.get('losses', {})
            alpha_reg_weight = losses_cfg.get('alpha_reg_weight', 0.0)
            safety_gate_threshold = losses_cfg.get('safety_gate_threshold', 0.0) # Default to 0 (always on) if not specified

            # 2. Determine if the current state is "safe". We use the cost from the step_result.
            # A cost > 0 indicates a collision, which is unsafe.
            # We can also use goal_distance as a proxy for safety near obstacles.
            # For simplicity and robustness, we'll use the collision cost. A cost of 0 means no collision occurred.
            is_safe_mask = (step_result.cost == 0).float().detach() # Use detach to not flow gradients through the mask itself

            # 3. Apply the alpha regularization loss ONLY on the states that are safe.
            gated_alpha_reg_loss = alpha_reg_weight * torch.mean(is_safe_mask * ((1.0 - log_alpha) ** 2))

            # --- END OF SAFETY-GATED ALPHA REGULARIZATION ---

            # Collision metric
            collision_event = (step_result.cost > 0).to(self.device)
            collision_count = collision_count + torch.sum(collision_event.float())

            # Update total loss
            step_loss = track_cost + ctrl_cost + gated_alpha_reg_loss # Added new gated loss term
            cumulative_loss = cumulative_loss + step_loss

            # Accumulate alpha for averaging
            sum_alpha = sum_alpha + torch.mean(log_alpha)

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
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
            self.optimizer.step()

            wandb.log(
                {
                    "loss": float(total_loss.item()),
                    "goal_distance": float(avg_goal_distance.item()),
                    "collision_rate": float(collision_rate.item()),
                    "avg_alpha_confidence": float(avg_alpha.item()), # New metric!
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


