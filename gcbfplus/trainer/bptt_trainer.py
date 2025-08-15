from typing import Any, Dict, Tuple, Optional
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import wandb

from ..policy.bptt_policy import BPTTPolicy
from ..policy.guardian_network import GuardianNetwork


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

        # 1. Instantiate the Pilot Network (our refactored BPTTPolicy)
        # We pass the 'pilot_network' sub-config to it
        self.pilot_policy = BPTTPolicy(pilot_cfg).to(self.device)

        # 2. Instantiate the Guardian Network (our new safety expert)
        self.guardian_network = GuardianNetwork(guardian_cfg).to(self.device)

        # 3. Create two separate optimizers
        pilot_lr = training_cfg.get('pilot_lr', 0.001)
        guardian_lr = training_cfg.get('guardian_lr', 0.001)
        self.max_grad_norm = training_cfg.get('max_grad_norm', 1.0)

        self.pilot_optimizer = torch.optim.Adam(self.pilot_policy.parameters(), lr=pilot_lr)
        self.guardian_optimizer = torch.optim.Adam(self.guardian_network.parameters(), lr=guardian_lr)
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
        cumulative_pilot_loss = torch.zeros((), device=self.device)
        cumulative_guardian_loss = torch.zeros((), device=self.device)
        sum_goal_distance = torch.zeros((), device=self.device)
        collision_count = torch.zeros((), device=self.device)

        for _ in range(self.horizon):
            # --- START OF rollout LOOP REFACTOR ---
            
            # 1. Get observation from the environment
            observations = self.env.get_observation(state)
            if hasattr(observations, 'to'):
                observations = observations.to(self.device)

            # --- GEMINI PROTOCOL DECISION PIPELINE ---
            # 2. The Pilot Network proposes a nominal action
            policy_outputs = self.pilot_policy(observations)
            nominal_action = policy_outputs['action']

            # 3. The Guardian Network predicts the safety value 'h'
            predicted_h = self.guardian_network(observations)

            # 4. The QP Arbiter (Placeholder): For now, we use a "pass-through"
            #    This allows us to test the training loop before integrating a complex QP solver.
            safe_action_to_execute = nominal_action
            # --- END OF GEMINI PROTOCOL DECISION PIPELINE ---

            # 5. Execute the action and get the next state
            step_result = self.env.step(state, safe_action_to_execute)  # Removed alpha passing
            next_state = step_result.next_state

            # --- DUAL LOSS CALCULATION ---
            losses_cfg = self.config.get('losses', {})

            # 6. Guardian's Loss (Safety-driven):
            #    We train the Guardian to accurately predict the true safety barrier value.
            h_regression_weight = losses_cfg.get('h_regression_weight', 1.0)
            
            # Compute true barrier function - placeholder implementation
            # For now, we'll use distance to nearest obstacle as true h
            if hasattr(self.env, 'compute_barrier_function'):
                true_h, _ = self.env.compute_barrier_function(state)
            else:
                # Fallback: compute distance to obstacles as barrier function
                if hasattr(state, 'obstacles') and state.obstacles is not None:
                    # Simple distance-based barrier function
                    agent_pos = state.position.unsqueeze(1)  # [batch, 1, 2]
                    obs_positions = state.obstacles[..., :2]  # [batch, n_obs, 2]
                    distances = torch.norm(agent_pos - obs_positions, dim=-1)  # [batch, n_obs]
                    true_h = torch.min(distances, dim=-1)[0].unsqueeze(-1)  # [batch, 1]
                else:
                    # No obstacles, h should be large (safe)
                    true_h = torch.ones_like(predicted_h) * 10.0
            
            guardian_loss = h_regression_weight * F.mse_loss(predicted_h, true_h.detach())

            # 7. Pilot's Loss (Efficiency-driven):
            #    Calculated based on the outcome of the final, safe action.
            current_goal_distance = self.env.get_goal_distance(next_state)
            track_cost = losses_cfg.get('goal_weight', 1.0) * torch.mean(current_goal_distance ** 2)
            # (For now, we omit other pilot losses like jerk for simplicity)
            pilot_loss = track_cost

            # --- END OF DUAL LOSS CALCULATION ---

            # Update cumulative losses for logging
            cumulative_guardian_loss += guardian_loss
            cumulative_pilot_loss += pilot_loss

            # Metrics tracking
            avg_goal_distance_step = torch.mean(current_goal_distance)
            sum_goal_distance = sum_goal_distance + avg_goal_distance_step

            # Collision metric
            collision_event = (step_result.cost > 0).to(self.device)
            collision_count = collision_count + torch.sum(collision_event.float())

            state = next_state
            
            # --- END OF rollout LOOP REFACTOR ---

        avg_goal_distance = sum_goal_distance / float(self.horizon)
        collision_rate = collision_count / float(self.horizon)
        return cumulative_pilot_loss, cumulative_guardian_loss, avg_goal_distance, collision_rate

    def train(self, num_steps: Optional[int] = None) -> None:
        # Set both networks to training mode
        self.pilot_policy.train()
        self.guardian_network.train()
        init_state = self.env.reset()

        max_steps: int = int(num_steps) if num_steps is not None else int(self.num_steps)
        for step_idx in range(max_steps):
            # --- START OF train METHOD REFACTOR ---

            # 1. Perform the rollout to get both losses
            #    (You'll need to update the return values of your rollout function)
            pilot_loss, guardian_loss, avg_goal_distance, collision_rate = self.rollout(init_state)

            # 2. Update the Guardian Network
            self.guardian_optimizer.zero_grad()
            guardian_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.guardian_network.parameters(), self.max_grad_norm)
            self.guardian_optimizer.step()

            # 3. Update the Pilot Network
            #    We detach the guardian_loss as the pilot should not be influenced by the guardian's training signal.
            self.pilot_optimizer.zero_grad()
            pilot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pilot_policy.parameters(), self.max_grad_norm)
            self.pilot_optimizer.step()

            # 4. Update wandb logging
            wandb.log({
                "pilot_loss": float(pilot_loss.item()),
                "guardian_loss": float(guardian_loss.item()),
                "goal_distance": float(avg_goal_distance.item()),
                "collision_rate": float(collision_rate.item()),
            }, step=self.global_step)

            #--- END OF train METHOD REFACTOR ---
            self.global_step += 1

            if (step_idx + 1) % 10 == 0:
                print(
                    f"Step {step_idx + 1}/{self.num_steps} | "
                    f"Pilot Loss: {pilot_loss.item():.4f} | "
                    f"Guardian Loss: {guardian_loss.item():.4f} | "
                    f"GoalDist: {avg_goal_distance.item():.4f} | "
                    f"CollRate: {collision_rate.item():.4f}"
                )

            # Periodic checkpoint saving
            if self.global_step % int(self.save_interval) == 0:
                step_dir = os.path.join(self.model_save_path, str(self.global_step))
                os.makedirs(step_dir, exist_ok=True)
                torch.save(self.pilot_policy.state_dict(), os.path.join(step_dir, "pilot.pt"))
                torch.save(self.guardian_network.state_dict(), os.path.join(step_dir, "guardian.pt"))
                print(f"Dual-network checkpoint saved at step {self.global_step} -> {step_dir}")


