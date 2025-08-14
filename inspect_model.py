import torch
import yaml
from gcbfplus.policy.bptt_policy import BPTTPolicy # 导入你的模型定义

def inspect_trained_model():
    """
    加载一个训练好的模型权重，并打印出它的结构，以验证文件是否正确。
    """
    # --- 1. 定义你的配置文件和模型文件路径 ---
    # 注意：请确保这里的路径与你存放文件的实际路径一致
    config_path = "config/final_config.yaml"
    model_path = "logs/final_config/models/10000/policy.pt"

    print(f"正在从 '{model_path}' 加载模型权重...")

    # --- 2. 加载配置，因为创建模型需要它 ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    policy_cfg = config.get('policy', {})
    device = torch.device('cpu') # 我们在本地分析，所以先用CPU即可

    # --- 3. 加载模型权重 ---
    # torch.load() 是我们用来“读取大脑”的唯一正确工具
    saved_state_dict = torch.load(model_path, map_location=device)
    print("模型权重加载成功！")

    # --- 4. 创建一个新的、同样结构的模型实例 ---
    # 我们需要一个“新的身体”，来装载这个“大脑”
    model = BPTTPolicy(policy_cfg).to(device)

    # --- 5. 将加载的权重（大脑）装入新的模型实例（身体） ---
    model.load_state_dict(saved_state_dict)
    model.eval() # 设置为评估模式
    print("权重已成功载入模型！")

    # --- 6. 打印模型结构，一探究竟 ---
    print("\n--- 训练好的模型结构如下：---")
    print(model)
    print("\n--- 模型探查成功！---")


if __name__ == '__main__':
    inspect_trained_model()