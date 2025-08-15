# Policy networks with dynamic alpha prediction for adaptive safety margins

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union, Any


class PerceptionModule(nn.Module):
    """
    感知模块，用于处理传感器输入。
    
    该模块可配置为处理不同类型的输入：
    - 基于视觉的观测（使用CNN处理深度图像）
    - 密集向量观测（使用MLP处理状态向量）
    """
    
    def __init__(self, config: Dict):
        """
        初始化感知模块。
        
        参数:
            config: 配置字典，包含以下键值：
                视觉模式:
                - 'vision_enabled': 是否启用基于视觉的处理
                - 'input_channels': 输入通道数（深度图默认为1）
                - 'conv_channels': CNN通道大小列表
                - 'kernel_sizes': 每个卷积层的核大小列表
                - 'image_size': 输入图像大小（假设为正方形）
                
                状态模式:
                - 'input_dim': 输入维度大小
                - 'hidden_dim': 隐藏维度大小
                - 'num_layers': 隐藏层数量
                - 'activation': 激活函数名称
                - 'use_batch_norm': 是否使用批归一化
        """
        super(PerceptionModule, self).__init__()
        
        # 检查是否启用视觉模式
        self.vision_enabled = config.get('vision_enabled', False)
        hidden_dim = config.get('hidden_dim', 64)
        activation = config.get('activation', 'relu')
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        if self.vision_enabled:
            # 基于视觉的CNN处理
            input_channels = config.get('input_channels', 1)  # 深度图像
            conv_channels = config.get('conv_channels', [32, 64, 128])
            kernel_sizes = config.get('kernel_sizes', [5, 3, 3])
            image_size = config.get('image_size', 64)
            
            # 构建CNN层
            cnn_layers = []
            in_channels = input_channels
            
            for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
                cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                           stride=2, padding=kernel_size//2))
                cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(self.activation)
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            
            # 计算卷积后的尺寸
            # 每个步长为2的卷积层将空间维度减半
            final_size = image_size // (2 ** len(conv_channels))
            cnn_output_size = conv_channels[-1] * final_size * final_size
            
            # 最终MLP获得期望的输出维度
            self.cnn_projection = nn.Sequential(
                nn.Linear(cnn_output_size, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.output_dim = hidden_dim
            
        else:
            # 基于状态的MLP处理（原始实现）
            input_dim = config.get('input_dim', 9)
            num_layers = config.get('num_layers', 2)
            use_batch_norm = config.get('use_batch_norm', False)
            
            # 构建MLP层
            layers = []
            
            # 输入层
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            
            # 隐藏层
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.activation)
            
            self.mlp = nn.Sequential(*layers)
            self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过感知模块处理输入。
        
        参数:
            x: 输入张量 
               - 视觉模式: [batch_size, n_agents, channels, height, width]
               - 状态模式: [batch_size, n_agents, input_dim] 或 [batch_size, input_dim]
            
        Returns:
            处理后的特征 [batch_size, output_dim] 或 [batch_size, n_agents, output_dim]
        """
        original_shape = x.shape
        
        if self.vision_enabled:
            # 处理视觉输入: [batch_size, n_agents, channels, height, width]
            if len(original_shape) == 5:
                batch_size, n_agents, channels, height, width = original_shape
                
                # 重塑为 [batch_size * n_agents, channels, height, width]
                x_flat = x.reshape(batch_size * n_agents, channels, height, width)
                
                # 通过CNN处理
                cnn_features = self.cnn(x_flat)  # [batch_size * n_agents, final_channels, final_h, final_w]
                
                # 展平空间维度
                cnn_flat = cnn_features.view(cnn_features.size(0), -1)  # [batch_size * n_agents, flat_size]
                
                # 投影到期望的输出维度
                features = self.cnn_projection(cnn_flat)  # [batch_size * n_agents, output_dim]
                
                # 重塑回 [batch_size, n_agents, output_dim]
                return features.view(batch_size, n_agents, -1)
            else:
                raise ValueError(f"视觉模式期望5D输入 [batch, agents, channels, height, width]，得到 {original_shape}")
        
        else:
            # 处理基于状态的输入（原始实现）
            if len(original_shape) == 3:
                batch_size, n_agents, input_dim = original_shape
                
                # 重塑为 [batch_size * n_agents, input_dim]
                x_flat = x.reshape(batch_size * n_agents, input_dim)
                
                # 通过MLP处理
                features = self.mlp(x_flat)
                
                # 重塑回 [batch_size, n_agents, output_dim]
                return features.view(batch_size, n_agents, -1)
            else:
                # 简单批处理 [batch_size, input_dim]
                return self.mlp(x)


class MemoryModule(nn.Module):
    """
    记忆模块，用于维护时序状态信息。
    
    使用GRU网络维护智能体的内部状态，支持多智能体场景。
    可选择是否在不同时间步之间保持记忆状态。
    """
    
    def __init__(self, config: Dict):
        """
        初始化记忆模块。
        
        参数:
            config: 包含记忆模块参数的字典
                必需键值:
                - 'input_dim': 输入维度
                - 'hidden_dim': 隐藏状态维度
                可选键值:
                - 'num_layers': GRU层数（默认1）
                - 'dropout': dropout率（默认0.0）
        """
        super(MemoryModule, self).__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0.0)
        
        # 创建GRU单元
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.hidden_state = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：处理输入并更新内部状态。
        
        参数:
            x: 输入张量，形状为[batch_size, n_agents, input_dim]或[batch_size, input_dim]
               
        返回:
            输出张量，形状为[batch_size, n_agents, hidden_dim]或[batch_size, hidden_dim]
        """
        if x.dim() == 3:  # 多智能体情况
            batch_size, n_agents, input_dim = x.shape
            
            # 处理多智能体观测
            # 重塑为 [batch_size * n_agents, 1, input_dim]，因为GRU期望序列长度维度
            x = x.view(batch_size * n_agents, 1, input_dim)
            
            # 初始化或重置隐藏状态（如果需要）
            if self.hidden_state is None or self.hidden_state.size(1) != batch_size * n_agents:
                self.hidden_state = torch.zeros(self.num_layers, batch_size * n_agents, self.hidden_dim, 
                                               device=x.device, dtype=x.dtype)
            
            # 创建新的张量而不是原地修改
            if self.hidden_state.device != x.device:
                self.hidden_state = self.hidden_state.to(x.device)
            
            # 更新隐藏状态
            output, new_hidden = self.gru(x, self.hidden_state)
            
            # 存储新的隐藏状态（不破坏计算图）
            self.hidden_state = new_hidden.detach()
            
            # 重塑回 [batch_size, n_agents, hidden_dim]
            return output.view(batch_size, n_agents, self.hidden_dim)
        else:
            # 简单批处理
            batch_size, input_dim = x.shape
            x = x.view(batch_size, 1, input_dim)
            
            # 初始化或重置隐藏状态（如果需要）
            if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
                self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                                               device=x.device, dtype=x.dtype)
            
            # 创建新的张量而不是原地修改
            if self.hidden_state.device != x.device:
                self.hidden_state = self.hidden_state.to(x.device)
            
            # 更新隐藏状态
            output, new_hidden = self.gru(x, self.hidden_state)
            
            # 存储新的隐藏状态（不破坏计算图）
            self.hidden_state = new_hidden.detach()
            
            return output.squeeze(1)  # 移除序列长度维度
    
    def reset(self) -> None:
        """重置记忆状态。"""
        self.hidden_state = None


class PolicyHeadModule(nn.Module):
    """
    策略头模块 - 纯飞行员网络（双子座协议）。
    
    专注于效率优化的动作生成，不包含安全相关预测。
    安全功能由独立的Guardian Network处理。
    """
    
    def __init__(self, config: Dict):
        """
        初始化策略头模块。
        
        参数:
            config: 包含策略头参数的字典
                必需键值:
                - 'input_dim': 输入特征维度
                - 'output_dim': 动作输出维度
                可选键值:
                - 'hidden_dims': 隐藏层维度列表
                - 'activation': 激活函数名称
                - 'output_activation': 输出层激活函数
                - 'action_scale': 动作缩放因子
                - 'predict_alpha': 是否预测动态alpha（默认True）
                - 'alpha_hidden_dim': alpha网络隐藏层维度
        """
        super(PolicyHeadModule, self).__init__()
        
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.action_scale = config.get('action_scale', 1.0)
        
        # 激活函数
        activation_name = config.get('activation', 'relu')
        self.activation = getattr(nn, activation_name.capitalize())() if hasattr(nn, activation_name.capitalize()) else nn.ReLU()
        
        output_activation = config.get('output_activation', None)
        self.output_activation = getattr(nn, output_activation.capitalize())() if output_activation and hasattr(nn, output_activation.capitalize()) else None
        
        # 移除自适应安全边距配置 - 纯飞行员网络专注于效率优化
        # 安全功能现在由Guardian Network专门处理
        
        # 构建动作预测MLP层
        self.action_layers = nn.ModuleList()
        
        # 动作的隐藏层
        layer_dims = [self.input_dim] + self.hidden_dims
        for i in range(len(layer_dims) - 1):
            self.action_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.action_layers.append(self.activation)
        
        # 动作输出层
        self.action_layers.append(nn.Linear(self.hidden_dims[-1] if self.hidden_dims else self.input_dim, self.output_dim))
        
        self.action_network = nn.Sequential(*self.action_layers)
        
        # 移除alpha预测网络 - 纯飞行员网络不预测安全相关参数
            
        # 移除margin预测网络 - 纯飞行员网络不预测安全相关参数
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播：生成动作（纯飞行员网络，专注于效率）。
        
        参数:
            features: 输入特征，形状为[batch_size, n_agents, input_dim]或[batch_size, input_dim]
               
        返回:
            actions: 动作张量（移除alpha和margin预测，专注于效率优化）
        """
        if features.dim() == 3:  # 多智能体情况
            batch_size, n_agents, input_dim = features.shape
            
            # 处理多智能体特征
            # 重塑为 [batch_size * n_agents, input_dim]
            features_flat = features.view(-1, input_dim)
            
            # 通过动作网络处理
            actions_flat = self.action_network(features_flat)
            
            # 应用输出激活函数
            if self.output_activation is not None:
                actions_flat = self.output_activation(actions_flat)
            
            # 缩放动作（如果需要）
            if self.action_scale != 1.0:
                actions_flat = actions_flat * self.action_scale
            
            # 重塑动作回 [batch_size, n_agents, -1]
            actions = actions_flat.view(batch_size, n_agents, -1)
            
            return actions
        else:
            # 简单批处理 - 单智能体情况
            actions = self.action_network(features)
            
            # 应用输出激活函数
            if self.output_activation is not None:
                actions = self.output_activation(actions)
            
            # 缩放动作（如果需要）
            if self.action_scale != 1.0:
                actions = actions * self.action_scale
                
            return actions


class LossWeightHead(nn.Module):
    """A head that predicts dynamic loss weights based on features."""
    def __init__(self, input_dim: int, weight_config: Dict[str, List[float]]):
        super().__init__()
        self.weight_config = weight_config
        self.weight_keys = list(weight_config.keys())
        self.num_weights = len(self.weight_keys)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_weights),
            nn.Sigmoid()  # Output normalized weights in [0, 1]
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Predict normalized weights
        normalized_weights = self.net(features)
        
        # Scale weights to their defined [min, max] ranges
        scaled_weights = {}
        for i, key in enumerate(self.weight_keys):
            min_val, max_val = self.weight_config[key]
            # Handle both batched and un-batched features
            if features.dim() == 2: # Shape [batch*agents, features]
                scaled_weights[key] = normalized_weights[:, i] * (max_val - min_val) + min_val
            else: # Should not happen, but for safety
                scaled_weights[key] = normalized_weights[i] * (max_val - min_val) + min_val

        return scaled_weights


class BPTTPolicy(nn.Module):
    """
    BPTT飞行员网络 - 双子座协议的效率专家。
    
    结合感知、记忆和策略头模块，专注于效率优化的端到端策略学习。
    安全功能由独立的Guardian Network处理，实现角色分离。
    """
    
    def __init__(self, config: Dict):
        """
        初始化BPTT策略网络。
        
        参数:
            config: 包含策略网络完整配置的字典
        """
        super(BPTTPolicy, self).__init__()
        
        # 存储完整配置（自适应损失权重需要访问根级别的losses）
        self.config = config
        
        # Get the policy-specific configuration block from the main config
        policy_cfg = config.get('policy', {})

        # Extract sub-configurations
        perception_config = policy_cfg.get('perception', {})
        memory_config = policy_cfg.get('memory', {}).copy()  # Use .copy() to avoid modifying the original dict
        policy_head_config = policy_cfg.get('policy_head', {})

        # --- THE FINAL, DEFINITIVE FIX IS HERE ---
        # 1. Build the PerceptionModule first to determine its output dimension.
        self.perception = PerceptionModule(perception_config)

        # 2. Get the master hidden_dim from the top level of the policy config.
        # This ensures a consistent hidden dimension across all modules.
        master_hidden_dim = policy_cfg.get('hidden_dim')
        if master_hidden_dim is None:
            raise ValueError("'hidden_dim' is a required key in the 'policy' configuration block.")

        # 3. Explicitly construct the memory_config with all required keys.
        memory_config['input_dim'] = self.perception.output_dim
        memory_config['hidden_dim'] = master_hidden_dim  # Pass the master hidden_dim down.

        # 4. Now, initialize the MemoryModule with the complete and correct configuration.
        self.memory = MemoryModule(memory_config)

        # 5. Construct the policy_head_config.
        policy_head_config['input_dim'] = self.memory.hidden_dim

        # 6. Initialize the PolicyHeadModule.
        self.policy_head = PolicyHeadModule(policy_head_config)
        
        # NEW: Adaptive loss weights functionality
        self.use_adaptive_loss_weights = policy_cfg.get('use_adaptive_loss_weights', False)
        if self.use_adaptive_loss_weights:
            # Filter for loss ranges defined as lists in the config
            loss_ranges = {k: v for k, v in self.config.get('losses', {}).items() if isinstance(v, list)}
            
            # NEW: Add an assertion for robust error checking
            assert loss_ranges, "Adaptive loss weights enabled, but no loss ranges (e.g., goal_weight: [min, max]) were found in the config's 'losses' section."
            
            self.loss_weight_head = LossWeightHead(self.memory.hidden_dim, loss_ranges)
        else:
            self.loss_weight_head = None
    
    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播：将观测转换为动作（纯飞行员网络，专注于效率优化）。
        
        参数:
            observations: 观测张量
               
        返回:
            字典包含:
            - action: 动作张量（专注于效率优化，安全由Guardian Network处理）
        """
        # 通过感知模块处理
        features = self.perception(observations)
        
        # 通过记忆模块处理
        memory_output = self.memory(features)
        
        # 通过策略头生成动作（纯飞行员网络）
        actions = self.policy_head(memory_output)

        outputs = {
            'action': actions
        }

        if self.use_adaptive_loss_weights and self.loss_weight_head is not None:
            # Flatten memory output for the head: [batch, agents, dim] -> [batch*agents, dim]
            if memory_output.dim() == 3:
                memory_output_flat = memory_output.view(-1, self.memory.hidden_dim)
                dynamic_weights = self.loss_weight_head(memory_output_flat)
                # Reshape weights back: e.g., [batch*agents] -> [batch, agents]
                batch_size, n_agents = memory_output.shape[:2]
                for key in dynamic_weights:
                    dynamic_weights[key] = dynamic_weights[key].view(batch_size, n_agents)
            else: # single-agent case
                dynamic_weights = self.loss_weight_head(memory_output)

            outputs['loss_weights'] = dynamic_weights

        return outputs
    
    def reset(self) -> None:
        """重置策略的内部状态（例如记忆）。"""
        if hasattr(self, 'memory'):
            self.memory.reset()


class EnsemblePolicy(nn.Module):
    """
    集成策略网络。
    
    结合多个策略网络的输出，提供更稳定的动作预测。
    支持简单平均和加权平均两种集成方法。
    """
    
    def __init__(self, config: Dict):
        """
        初始化集成策略网络。
        
        参数:
            config: 包含集成策略配置的字典
                必需键值:
                - 'policies': 策略配置列表
                可选键值:
                - 'ensemble_method': 集成方法（'mean'或'weighted'）
                - 'num_policies': 策略数量
        """
        super(EnsemblePolicy, self).__init__()
        
        # 提取配置参数
        policies_config = config.get('policies', [])
        self.ensemble_method = config.get('ensemble_method', 'mean')
        self.num_policies = config.get('num_policies', len(policies_config))
        
        # 创建策略集成
        self.policies = nn.ModuleList()
        for policy_config in policies_config:
            policy = BPTTPolicy(policy_config)
            self.policies.append(policy)
        
        # 如果使用加权集成，创建权重参数
        if self.ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_policies))
            # 设备将在模型移动到设备时设置
        
        # 存储配置
        self.config = config
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：通过集成策略生成动作和alpha值。
        
        参数:
            observations: 观测张量
               
        返回:
            元组(actions, alpha):
            - actions: 集成后的动作张量
            - alpha: 集成后的alpha值（如果启用）或None
        """
        # 从每个策略获取动作和alpha
        policy_outputs = []
        for policy in self.policies:
            actions, alpha = policy(observations)
            policy_outputs.append((actions, alpha))
        
        # 分离动作和alpha
        actions_list = [output[0] for output in policy_outputs]
        alphas_list = [output[1] for output in policy_outputs if output[1] is not None]
        
        # 堆叠以便组合 [num_policies, batch_size, action_dim/1]
        stacked_actions = torch.stack(actions_list, dim=0)
        stacked_alphas = torch.stack(alphas_list, dim=0) if alphas_list else None
        
        # 基于集成方法组合动作和alpha
        if self.ensemble_method == 'mean':
            # 简单平均
            final_actions = torch.mean(stacked_actions, dim=0)
            final_alpha = torch.mean(stacked_alphas, dim=0) if stacked_alphas is not None else None
        elif self.ensemble_method == 'weighted':
            # 加权平均
            weights = torch.softmax(self.ensemble_weights, dim=0)
            weights = weights.view(-1, 1, 1, 1)  # 广播形状
            final_actions = torch.sum(stacked_actions * weights, dim=0)
            final_alpha = torch.sum(stacked_alphas * weights, dim=0) if stacked_alphas is not None else None
        else:
            # 默认使用均值
            final_actions = torch.mean(stacked_actions, dim=0)
            final_alpha = torch.mean(stacked_alphas, dim=0) if stacked_alphas is not None else None
        
        return final_actions, final_alpha
    
    def reset(self) -> None:
        """重置集成中的所有策略。"""
        for policy in self.policies:
            policy.reset()


def create_policy_from_config(config: Dict) -> nn.Module:
    """
    Factory function to create a policy from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Policy network instance
    """
    policy_type = config.get('type', 'bptt')
    
    if policy_type == 'bptt':
        return BPTTPolicy(config)
    elif policy_type == 'ensemble':
        return EnsemblePolicy(config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}") 