# inspect_model.py (最终的、绝对正确的版本)

import torch
import torch.nn as nn
import yaml
import os

# ===================================================================
# 核心修复：我们不再从外部导入复杂的BPTTPolicy。
# 我们直接在这里，根据错误日志，定义一个与我们已训练好的模型100%匹配的、简单的模型结构。
# ===================================================================
class SimpleTrainedPolicy(nn.Module):
    """
    这个模型是根据 "Unexpected key(s) in state_dict" 错误日志，
    为我们已训练好的 policy.pt 文件“量身定制”的身体。
    它的结构是一个名为 'net' 的简单序列网络。
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def inspect_trained_model():
    """
    加载一个训练好的模型权重，并打印出它的结构。
    这个最终版本使用了一个与保存的权重完全匹配的模型结构。
    """
    # --- 1. 定义你的配置文件和模型文件路径 ---
    config_path = "config/final_config.yaml"
    model_path = "logs/final_config/models/10000/policy.pt" # 假设这是你下载的模型

    print(f"正在从 '{model_path}' 加载模型权重...")

    # --- 2. 加载配置，以获取输入输出维度 ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env_cfg = config.get('env', {})
    # 观测维度 = 位置(2) + 速度(2) + 相对目标(2) = 6
    obs_dim = 6 
    action_dim = env_cfg.get('action_dim', 2)
    device = torch.device('cpu')

    # --- 3. 加载模型权重 ---
    saved_state_dict = torch.load(model_path, map_location=device)
    print("模型权重加载成功！")

    # --- 4. 创建一个与“大脑”完全匹配的“身体” ---
    model = SimpleTrainedPolicy(input_dim=obs_dim, output_dim=action_dim).to(device)

    # --- 5. 将加载的权重（大脑）装入新的模型实例（身体） ---
    model.load_state_dict(saved_state_dict)
    model.eval()
    print("权重已成功载入模型！")

    # --- 6. 打印模型结构，一探究竟 ---
    print("\n--- 训练好的模型结构如下：---")
    print(model)
    print("\n--- 模型探查成功！---")


if __name__ == '__main__':
    inspect_trained_model()