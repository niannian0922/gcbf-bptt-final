from typing import Any, Dict, Tuple, Optional
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import wandb

# --- THE ULTIMATE FIX 1: Import the CORRECT, advanced policy class ---
from ..policy.bptt_policy import BPTTPolicy


def check_nan(tensor: torch.Tensor, name: str):
    """Checks if a tensor contains NaN or Inf values and raises an error if it does."""
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"NaN or Inf detected in '{name}'")


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

        # Config-driven hyperparameters
        training_cfg: Dict[str, Any] = self.config.get("training", {})
        self.horizon: int = int(training_cfg.get("horizon_length", 64))
        self.num_steps: int = int(training_cfg.get("training_steps", 10000))
        self.learning_rate: float = float(training_cfg.get("learning_rate", 1e-3))

        # Logging / naming
        self.run_name: str = str(self.config.get("run_name", "default"))
        self.log_dir: str = str(self.config.get("trainer", {}).get("log_dir", "logs"))
        self.model_save_path: str = os.path.join(self.log_dir, self.run_name, "models")
        os.makedirs(self.model_save_path, exist_ok=True)
        self.save_interval: int = int(training_cfg.get("save_interval", 1000))
        self.global_step: int = 0

        # --- THE ULTIMATE FIX 2: Instantiate the TRUE BPTTPolicy for both Pilot and Guardian ---
        print("Instantiating Pilot Policy (BPTTPolicy)...")
        self.pilot_policy = BPTTPolicy(self.config).to(self.device)
        
        print("Instantiating Guardian Policy (BPTTPolicy)...")
        self.guardian_policy = BPTTPolicy(self.config).to(self.device)

        # Create separate optimizers for each policy
        self.pilot_optimizer = Adam(self.pilot_policy.parameters(), lr=self.learning_rate)
        self.guardian_optimizer = Adam(self.guardian_policy.parameters(), lr=self.learning_rate)

        # Initialize Weights & Biases
        wandb_config: Dict[str, Any] = self.config.get("wandb", {})
        if wandb_config:
            wandb.init(
                project=wandb_config.get("project", "gcbf-bptt-final"),
                entity=wandb_config.get("entity"),
                name=self.run_name,
                config=self.config,
                mode=wandb_config.get("mode", "online"),
            )

    def train(self, num_steps: Optional[int] = None) -> None:
        """
        Main training loop.
        NOTE: This is a simplified placeholder for the dual-policy training logic.
        You will need to implement the actual training steps for pilot and guardian.
        For now, we focus on training only the pilot to ensure the architecture works.
        """
        self.pilot_policy.train()
        self.guardian_policy.train()
        init_state = self.env.reset(batch_size=self.config.get('training', {}).get('batch_size', 32))

        max_steps: int = int(num_steps) if num_steps is not None else int(self.num_steps)
        for step_idx in range(max_steps):
            # --- Placeholder for actual dual-policy training ---
            # For now, we will just train the pilot to verify the architecture.
            self.pilot_optimizer.zero_grad()
            
            # This rollout and loss calculation will need to be adapted for your dual-policy strategy
            total_loss, avg_goal_distance, collision_rate, avg_alpha = self.rollout(init_state, self.pilot_policy)
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.pilot_policy.parameters(), self.config.get('training', {}).get('max_grad_norm', 1.0))
            self.pilot_optimizer.step()
            
            # Logging
            wandb.log({
                "total_loss": float(total_loss.item()),
                "goal_distance": float(avg_goal_distance.item()),
                "collision_rate": float(collision_rate.item()),
                "avg_alpha_confidence": float(avg_alpha.item()),
            }, step=self.global_step)
            
            if (self.global_step + 1) % 10 == 0:
                print(f"Step {self.global_step + 1}/{max_steps} | Loss: {total_loss.item():.4f} | GoalDist: {avg_goal_distance.item():.4f} | CollRate: {collision_rate.item():.4f}")

            # Periodic checkpoint saving
            if (self.global_step + 1) % self.save_interval == 0:
                step_dir = os.path.join(self.model_save_path, str(self.global_step + 1))
                os.makedirs(step_dir, exist_ok=True)
                torch.save(self.pilot_policy.state_dict(), os.path.join(step_dir, "pilot.pt"))
                torch.save(self.guardian_policy.state_dict(), os.path.join(step_dir, "guardian.pt"))
                print(f"Checkpoints saved at step {self.global_step + 1} -> {step_dir}")

            self.global_step += 1
    
    # The rollout function now takes a policy argument to be more flexible
    def rollout(self, init_state: Any, policy_to_rollout: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = init_state
        cumulative_loss = torch.zeros((), device=self.device)
        sum_goal_distance = torch.zeros((), device=self.device)
        collision_count = torch.zeros((), device=self.device)
        sum_alpha = torch.zeros((), device=self.device)

        for _ in range(self.horizon):
            observations = self.env.get_observation(state)
            check_nan(observations, "observations")

            policy_outputs = policy_to_rollout(observations)
            nominal_action = policy_outputs['action']
            alpha = policy_outputs.get('alpha')
            check_nan(nominal_action, "policy_outputs[action]")
            if alpha is not None:
                check_nan(alpha, "policy_outputs[alpha]")

            safe_backup_action = torch.zeros_like(nominal_action)
            if alpha is None:
                final_action = nominal_action
                log_alpha = torch.ones(nominal_action.shape[0], 1, device=self.device) if nominal_action.dim() > 1 else torch.ones(1, device=self.device)
            else:
                final_action = alpha * nominal_action + (1 - alpha) * safe_backup_action
                log_alpha = alpha

            check_nan(final_action, "final_action")
            step_result = self.env.step(state, final_action, alpha)
            next_state = step_result.next_state
            check_nan(next_state.position, "next_state.position")
            check_nan(next_state.velocity, "next_state.velocity")

            goal_distances = self.env.get_goal_distance(next_state)
            sum_goal_distance += torch.mean(goal_distances)
            collision_count += torch.sum((step_result.cost > 0).float())

            losses_cfg = self.config.get('losses', {})
            track_cost = torch.mean(losses_cfg.get('goal_weight', 1.0) * (goal_distances ** 2))
            ctrl_cost = losses_cfg.get('control_weight', 1e-4) * torch.mean(final_action ** 2)
            
            step_loss = track_cost + ctrl_cost
            check_nan(step_loss, "step_loss")
            cumulative_loss += step_loss
            sum_alpha += torch.mean(log_alpha)
            state = next_state

        avg_goal_distance = sum_goal_distance / float(self.horizon)
        collision_rate = collision_count / (float(self.horizon) * state.batch_size)
        avg_alpha = sum_alpha / float(self.horizon)
        
        return cumulative_loss, avg_goal_distance, collision_rate, avg_alpha


