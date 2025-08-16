# gcbfplus/trainer/bptt_trainer.py (决战修正版)

from typing import Any, Dict, Tuple, Optional
import os

import torch
from torch import nn
from torch.optim import Adam
import wandb

# =====================================================================
# ▼▼▼ 核心修复：将所有相对导入 ".." 替换为绝对导入 "gcbfplus." ▼▼▼
# =====================================================================
from gcbfplus.policy.bptt_policy import BPTTPolicy
from gcbfplus.env.single_gcbf_layer import SingleAgentGCBFLayer # <-- 确保类名与文件名匹配
from gcbfplus.dynamics.base_dynamics import BaseDynamics
from gcbfplus.domain.base_domain import BaseDomain
from modules.model import AgileCommand
from gcbfplus.trainer.phoenix_loss import compute_phoenix_loss
# =====================================================================
# ▲▲▲ 修复结束 ▲▲▲
# =====================================================================

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
        self.dynamics: BaseDynamics = self.env.dynamics
        self.domain: BaseDomain = self.env.domain

        training_cfg: Dict[str, Any] = self.config.get("training", {})
        self.horizon: int = int(training_cfg.get("horizon_length", 64))
        self.num_steps: int = int(training_cfg.get("training_steps", 10000))
        self.learning_rate: float = float(training_cfg.get("learning_rate", 1e-3))

        self.run_name: str = str(self.config.get("run_name", "default"))
        self.log_dir: str = str(self.config.get("trainer", {}).get("log_dir", "logs"))
        self.model_save_path: str = os.path.join(self.log_dir, self.run_name, "models")
        os.makedirs(self.model_save_path, exist_ok=True)
        self.save_interval: int = int(training_cfg.get("save_interval", 1000))
        self.global_step: int = 0

        model_type = self.config.get('model_type', 'bptt')
        if model_type == 'phoenix':
            print("Instantiating AgileCommand (Phoenix) model...")
            cbf_config = self.config.get('cbf', {})
            # --- 核心修复：使用正确的类名 SingleAgentGCBFLayer ---
            gcbf_module = SingleAgentGCBFLayer(self.domain, cbf_config, self.dynamics).to(self.device)
            policy_net = BPTTPolicy(self.dynamics, gcbf_module, self.config).to(self.device)
            self.model = AgileCommand(policy_net, gcbf_module).to(self.device)
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            print("Instantiating legacy BPTTPolicy model...")
            cbf_config = self.config.get('cbf', {})
            self.cbf_layer = SingleAgentGCBFLayer(self.domain, cbf_config, self.dynamics).to(self.device)
            self.model = BPTTPolicy(self.dynamics, self.cbf_layer, self.config).to(self.device)
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        wandb_config: Dict[str, Any] = self.config.get("wandb", {})
        if wandb_config and wandb_config.get("use_wandb", False):
            wandb.init(project=wandb_config.get("project"), name=self.run_name, config=self.config)

    def train(self, num_steps: Optional[int] = None) -> None:
        """Main training loop, re-architected for the Phoenix model."""
        self.model.train()
        state = self.env.reset(batch_size=self.config.get('training', {}).get('batch_size', 32))
        max_steps: int = int(num_steps) if num_steps is not None else int(self.num_steps)

        for step_idx in range(max_steps):
            self.optimizer.zero_grad()
            total_loss, loss_dict, state = self.rollout_and_loss_phoenix(state)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('training', {}).get('max_grad_norm', 1.0))
            self.optimizer.step()

            if wandb.run: wandb.log(loss_dict, step=self.global_step)
            if (self.global_step + 1) % 10 == 0:
                 print(f"Step {self.global_step + 1}/{max_steps} | Total Loss: {loss_dict['total_loss']:.4f} | "
                      f"Eff: {loss_dict['efficiency_loss']:.4f} | Corr: {loss_dict['correction_penalty']:.4f} | Jerk: {loss_dict['jerk_loss']:.4f}")
            
            if (self.global_step + 1) % self.save_interval == 0:
                step_dir = os.path.join(self.model_save_path, str(self.global_step + 1))
                os.makedirs(step_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(step_dir, "phoenix.pt"))
                print(f"Phoenix checkpoint saved at step {self.global_step + 1}")
            
            self.global_step += 1

    def rollout_and_loss_phoenix(self, init_state: Any) -> Tuple[torch.Tensor, Dict[str, Any], Any]:
        """Performs a rollout for the Phoenix model and computes the loss."""
        state = init_state
        nominal_actions_list, safe_actions_list, rewards_list = [], [], []

        for _ in range(self.horizon):
            observations = self.env.get_observation(state)
            safe_action, nominal_action = self.model(observations)
            
            dummy_alpha = torch.ones(safe_action.shape[0], 1, device=self.device)
            step_result = self.env.step(state, safe_action, dummy_alpha)
            
            safe_actions_list.append(safe_action)
            nominal_actions_list.append(nominal_action)
            rewards_list.append(step_result.rewards)
            state = step_result.next_state

        nominal_actions_tensor = torch.stack(nominal_actions_list)
        safe_actions_tensor = torch.stack(safe_actions_list)
        
        rewards_tensor_dict = {
            k: torch.stack([dic[k] for dic in rewards_list]) for k in rewards_list[0]
        }
        
        total_loss, loss_dict = compute_phoenix_loss(
            nominal_action=nominal_actions_tensor,
            safe_action=safe_actions_tensor,
            rewards=rewards_tensor_dict,
            cfg=self.config
        )

        return total_loss, loss_dict, state