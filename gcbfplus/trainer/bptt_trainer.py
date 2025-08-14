from typing import Any, Dict, Tuple, Optional
import os

import torch
from torch import nn
from torch.optim import Adam
import wandb


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

        # Determine observation and action dimensions from the environment
        if hasattr(self.env, "observation_shape"):
            obs_shape = self.env.observation_shape
            obs_dim = int(obs_shape[0]) if isinstance(obs_shape, tuple) else int(obs_shape)
        else:
            obs_dim = int(getattr(self.env, "state_dim", 4))

        if hasattr(self.env, "action_shape"):
            act_shape = self.env.action_shape
            action_dim = int(act_shape[0]) if isinstance(act_shape, tuple) else int(act_shape)
        else:
            action_dim = int(getattr(self.env, "action_dim", 2))

        # Policy
        hidden_dim: int = int(policy_cfg.get("hidden_dim", 64))
        self.policy = SimpleMLPPolicy(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

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

    def rollout(self, init_state: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = init_state
        cumulative_loss = torch.zeros((), device=self.device)
        sum_goal_distance = torch.zeros((), device=self.device)
        collision_count = torch.zeros((), device=self.device)

        for _ in range(self.horizon):
            observations = self.env.get_observation(state)
            if hasattr(observations, "to"):
                observations = observations.to(self.device)

            action = self.policy(observations)
            step_result = self.env.step(state, action, None)
            next_state = step_result.next_state

            # Goal distance metric
            goal_distances = self.env.get_goal_distance(next_state)
            avg_goal_distance_step = torch.mean(goal_distances)
            sum_goal_distance = sum_goal_distance + avg_goal_distance_step

            # Control effort regularization
            ctrl_cost = 1e-3 * torch.sum(action ** 2)

            # Tracking cost: squared goal distance
            track_cost = torch.sum(goal_distances ** 2)

            # Collision metric
            collision_event = (step_result.cost > 0).to(self.device)
            collision_count = collision_count + torch.sum(collision_event.float())

            step_loss = track_cost + ctrl_cost
            cumulative_loss = cumulative_loss + step_loss

            state = next_state

        avg_goal_distance = sum_goal_distance / float(self.horizon)
        collision_rate = collision_count / float(self.horizon)
        return cumulative_loss, avg_goal_distance, collision_rate

    def train(self, num_steps: Optional[int] = None) -> None:
        self.policy.train()
        init_state = self.env.reset()

        max_steps: int = int(num_steps) if num_steps is not None else int(self.num_steps)
        for step_idx in range(max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            total_loss, avg_goal_distance, collision_rate = self.rollout(init_state)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
            self.optimizer.step()

            wandb.log(
                {
                    "loss": float(total_loss.item()),
                    "goal_distance": float(avg_goal_distance.item()),
                    "collision_rate": float(collision_rate.item()),
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


