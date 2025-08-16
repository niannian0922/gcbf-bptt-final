# train_single_agent.py (决战修正版)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# =====================================================================
# ▼▼▼ 核心修复：使用正确的、绝对的导入路径 ▼▼▼
# =====================================================================
from gcbfplus.trainer.bptt_trainer import BPTTTrainer
from gcbfplus.env.single_agent_env import SingleAgentEnv
# =====================================================================
# ▲▲▲ 修复结束 ▲▲▲
# =====================================================================

@hydra.main(config_path="../config", config_name="default_single_agent.yaml")
def main(cfg: DictConfig):
    # Print the full config for debugging
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 核心修复：确保env和trainer都被正确初始化 ---
    # 1. Initialize the environment
    env = SingleAgentEnv(cfg, device=device)

    # 2. Initialize the trainer
    #    (bptt_trainer的__init__已经足够处理所有逻辑，这里无需传入trainer_cfg和policy_cfg)
    trainer = BPTTTrainer(
        env=env,
        trainer_cfg=cfg.trainer,
        policy_cfg=cfg.policy,
        device=device,
        full_config=cfg
    )

    # 3. Start training
    trainer.train()

if __name__ == '__main__':
    main()