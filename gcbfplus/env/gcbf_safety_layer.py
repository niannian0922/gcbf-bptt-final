import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List, Union
from dataclasses import dataclass

from .multi_agent_env import MultiAgentState


class GCBFSafetyLayer(nn.Module):
    """
    🛡️ PROBABILISTIC SAFETY SHIELD 概率安全防护罩
    
    重构为概率安全防护罩，输出安全信心分数 alpha_safety (0-1范围)。
    不再直接过滤动作，而是作为"明智顾问"，基于CBF值评估安全状况。
    支持自适应安全边距和动态Alpha机制。
    
    核心创新：解耦"安全"和"效率"目标，允许策略网络在安全区域自由探索，
    同时在危险情况下提供安全回退保证。
    """
    
    def __init__(self, config: Dict):
        """
        初始化安全层。
        
        参数:
            config: 包含安全层参数的字典
                必需键值:
                - 'alpha': CBF参数alpha (h_dot + alpha * h >= 0)
                - 'eps': 数值稳定性的小正参数
                - 'safety_margin': 碰撞避免的安全距离边距
                可选键值:
                - 'use_qp': 是否使用QP求解器（否则使用简单投影）
                - 'qp_relaxation_weight': 约束松弛权重
                - 'max_iterations': 求解器的最大迭代次数
        """
        super(GCBFSafetyLayer, self).__init__()
        
        # CBF参数
        self.alpha = config.get('alpha', 1.0)
        self.eps = config.get('eps', 0.02)
        self.safety_margin = config.get('safety_margin', 0.05)
        
        # QP参数
        self.use_qp = config.get('use_qp', True)
        self.qp_relaxation_weight = config.get('qp_relaxation_weight', 10.0)
        self.max_iterations = config.get('max_iterations', 10)
        
        # 注册参数
        self.register_buffer('alpha_tensor', torch.tensor([self.alpha], dtype=torch.float32))
        
        # 🛡️ PROBABILISTIC SAFETY SHIELD: 安全信心分数的超参数
        self.k = config.get('safety_sharpness', 1.0)  # 控制过渡的锐利度
        
    def barrier_function(self, state: MultiAgentState, dynamic_margins: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算智能体间和智能体-障碍物间的屏障函数值。
        
        参数:
            state: 多智能体状态，包含位置、速度和障碍物信息
            
        返回:
            屏障函数值张量 [batch, n_agents, n_constraints]
        """
        batch_size, n_agents, pos_dim = state.positions.shape
        
        # 计算智能体间屏障函数（碰撞避免）
        # 对于每对智能体，h(x) = ||p_i - p_j||^2 - (2r)^2，其中r是智能体半径
        
        # 计算位置间的成对差异
        # 形状: [batch, n_agents, n_agents, pos_dim]
        pos_diff = state.positions.unsqueeze(2) - state.positions.unsqueeze(1)
        
        # 计算平方距离
        # 形状: [batch, n_agents, n_agents]
        dist_squared = torch.sum(pos_diff**2, dim=-1)
        
        # 🚀 CORE INNOVATION: 使用动态安全裕度（如果提供）
        agent_radius = getattr(state, 'agent_radius', 0.05)  # 默认半径
        
        # 从已知在GPU上的张量获取正确的设备信息
        device = dist_squared.device
        
        if dynamic_margins is not None:
            # 使用动态安全裕度：[batch_size, n_agents, 1] -> [batch_size, n_agents, n_agents]
            # 对于智能体i和j的交互，使用两者裕度的平均值
            margins_i = dynamic_margins.unsqueeze(2)  # [batch, n_agents, 1, 1]
            margins_j = dynamic_margins.unsqueeze(1)  # [batch, 1, n_agents, 1]
            avg_margins = (margins_i + margins_j) / 2.0  # [batch, n_agents, n_agents, 1]
            avg_margins = avg_margins.squeeze(-1)  # [batch, n_agents, n_agents]
            
            # 计算动态阈值 (2 * radius + dynamic_margin)^2
            # 关键修复：确保threshold张量在正确的设备上
            threshold = ((2 * agent_radius + avg_margins)**2).to(device)
        else:
            # 使用固定安全裕度
            # 关键修复：确保threshold张量在正确的设备上
            threshold = torch.tensor((2 * agent_radius + self.safety_margin)**2, device=device)
        
        # 创建屏障值: h(x) = dist_squared - threshold
        # 形状: [batch, n_agents, n_agents]
        h_agent = dist_squared - threshold
        
        # 将对角线设置为大值（无自碰撞）
        mask = torch.eye(n_agents, device=h_agent.device, dtype=torch.bool)
        h_agent = h_agent.masked_fill(mask.unsqueeze(0), float('inf'))
        
        # 如果存在障碍物，计算智能体-障碍物屏障函数
        if hasattr(state, 'obstacles') and state.obstacles is not None:
            # 提取障碍物位置和半径
            obstacle_positions = state.obstacles[..., :-1]  # [batch, n_obs, pos_dim]
            obstacle_radii = state.obstacles[..., -1:]     # [batch, n_obs, 1]
            
            # 计算智能体与障碍物位置之间的差异
            # 形状: [batch, n_agents, n_obs, pos_dim]
            agent_obs_diff = state.positions.unsqueeze(2) - obstacle_positions.unsqueeze(1)
            
            # 计算平方距离
            # 形状: [batch, n_agents, n_obs]
            agent_obs_dist_squared = torch.sum(agent_obs_diff**2, dim=-1)
            
            # 计算阈值: (agent_radius + obstacle_radius + margin)^2
            # 形状: [batch, 1, n_obs]
            # 关键修复：确保obs_threshold张量在正确的设备上
            obs_threshold = ((agent_radius + obstacle_radii.squeeze(-1).unsqueeze(1) + self.safety_margin)**2).to(device)
            
            # 创建屏障值: h(x) = dist_squared - threshold
            # 形状: [batch, n_agents, n_obs]
            h_obstacle = agent_obs_dist_squared - obs_threshold
            
            # 组合智能体-智能体和智能体-障碍物约束
            # 形状: [batch, n_agents, n_agents + n_obs]
            h = torch.cat([h_agent, h_obstacle], dim=-1)
        else:
            # 仅智能体-智能体约束
            h = h_agent
            
        return h
    
    def barrier_jacobian(self, state: MultiAgentState) -> torch.Tensor:
        """
        Compute the Jacobian (gradient) of the barrier function with respect to states.
        
        Args:
            state: Current environment state
            
        Returns:
            Jacobian tensor [batch_size, n_agents, n_constraints, state_dim]
        """
        # 我们需要计算梯度，因此启用自动微分
        batch_size = state.batch_size
        n_agents = state.positions.shape[1]
        pos_dim = state.positions.shape[2]
        device = state.positions.device
        
        # 为自动微分创建计算图
        positions = state.positions.clone().requires_grad_(True)
        
        # 计算位置间的成对差异
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        
        # 计算平方距离
        dist_squared = torch.sum(pos_diff**2, dim=3)
        
        # 计算阈值
        threshold = (2 * (self.safety_margin + 0.05))**2
        
        # 创建屏障值: h(x) = dist_squared - threshold
        h_agent_agent = dist_squared - threshold
        
        # 将对角线设置为大值（无自碰撞）
        mask = torch.eye(n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        h_agent_agent = h_agent_agent.masked_fill(mask == 1, 1000.0)
        
        # 获取约束数量
        if state.obstacles is not None:
            n_obs = state.obstacles.shape[1]
            n_constraints = n_agents + n_obs
        else:
            n_constraints = n_agents
        
        # 初始化雅可比矩阵张量
        # 为简化起见，我们只计算相对于位置的梯度
        # 完整版本应包含速度
        jacobian = torch.zeros(batch_size, n_agents, n_constraints, pos_dim*2, device=device)
        
        # 对于每个智能体和每个约束，计算梯度
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    # 计算相对于位置的h_ij梯度
                    # 我们正在计算∂h_ij/∂p_i和∂h_ij/∂p_j
                    grad_outputs = torch.zeros_like(h_agent_agent)
                    grad_outputs[:, i, j] = 1.0
                    
                    # 使用自动微分获取梯度
                    grads = torch.autograd.grad(
                        outputs=h_agent_agent,
                        inputs=positions,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    # 在雅可比矩阵张量中存储梯度
                    # 对于智能体i，约束j（来自智能体j）
                    jacobian[:, i, j, :pos_dim] = grads[:, i]
        
        # 如果有障碍物，计算智能体-障碍物约束的梯度
        if state.obstacles is not None:
            # 提取障碍物位置和半径
            obstacle_positions = state.obstacles[..., :-1]  # [batch, n_obs, pos_dim]
            obstacle_radii = state.obstacles[..., -1:]     # [batch, n_obs, 1]
            
            # 计算智能体与障碍物位置之间的差异
            obs_diff = positions.unsqueeze(2) - obstacle_positions.unsqueeze(1)
            
            # 计算平方距离
            obs_dist_squared = torch.sum(obs_diff**2, dim=3)
            
            # 计算阈值
            obs_threshold = (0.05 + obstacle_radii.squeeze(-1).unsqueeze(1) + self.safety_margin)**2
            
            # 创建屏障值: h(x) = dist_squared - threshold
            h_agent_obs = obs_dist_squared - obs_threshold
            
            # 对于每个智能体和每个障碍物，计算梯度
            for i in range(n_agents):
                for j in range(n_obs):
                    # 计算相对于位置的h_ij梯度
                    grad_outputs = torch.zeros_like(h_agent_obs)
                    grad_outputs[:, i, j] = 1.0
                    
                    # 使用自动微分获取梯度
                    grads = torch.autograd.grad(
                        outputs=h_agent_obs,
                        inputs=positions,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    # 在雅可比矩阵张量中存储梯度
                    # 对于智能体i，约束n_agents+j（来自障碍物j）
                    jacobian[:, i, n_agents+j, :pos_dim] = grads[:, i]
        
        return jacobian
    
    def control_affine_dynamics(self, state: MultiAgentState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算控制仿射动力学矩阵f(x)和g(x)。
        
        对于双积分器系统:
        dx/dt = f(x) + g(x)u
        
        其中f(x) = [vx, vy, 0, 0]^T
        且g(x) = [0, 0; 0, 0; 1/m, 0; 0, 1/m]
        
        参数:
            state: 当前环境状态
            
        返回:
            (f, g)的元组，其中:
            - f: 漂移项 [batch_size, n_agents, state_dim]
            - g: 控制输入项 [batch_size, n_agents, state_dim, action_dim]
        """
        batch_size = state.batch_size
        n_agents = state.positions.shape[1]
        pos_dim = state.positions.shape[2]
        device = state.positions.device
        
        # 对于状态为[x, y, vx, vy]的双积分器，我们有:
        # dx/dt = vx
        # dy/dt = vy
        # dvx/dt = 1/m * fx
        # dvy/dt = 1/m * fy
        
        # 漂移项f(x) = [vx, vy, 0, 0]^T
        f = torch.zeros(batch_size, n_agents, 2*pos_dim, device=device)
        f[:, :, :pos_dim] = state.velocities  # 位置导数 = 速度
        
        # 控制输入项g(x) = [0, 0; 0, 0; 1/m, 0; 0, 1/m]
        g = torch.zeros(batch_size, n_agents, 2*pos_dim, pos_dim, device=device)
        
        # 默认质量 = 1.0 如果未指定
        m = 0.1  # 默认质量
        
        # 设置控制矩阵 - 每个力分量只影响其对应的速度
        for i in range(pos_dim):
            g[:, :, pos_dim+i, i] = 1.0 / m
        
        return f, g
    
    def compute_safety_confidence(
        self, 
        state: MultiAgentState, 
        dynamic_margins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        🛡️ PROBABILISTIC SAFETY SHIELD: 計算安全信心分數
        
        基於屏障函數值計算alpha_safety分數，使用sigmoid函數將h(x)映射到[0,1]範圍。
        
        Args:
            state: 當前環境狀態
            dynamic_margins: 動態安全裕度 [batch_size, n_agents, 1] (可選)
            
        Returns:
            alpha_safety: 安全信心分數 [batch_size, n_agents, 1]
                         0 = 極度危險，1 = 完全安全
        """
        # 計算屏障函數值
        h_val = self.barrier_function(state, dynamic_margins)
        
        # 取最小值（最危險的約束）來代表整體安全狀況
        # h_val: [batch_size, n_agents, n_constraints]
        min_h_val, _ = torch.min(h_val, dim=2)  # [batch_size, n_agents]
        
        # 使用sigmoid函數將h值映射到[0,1]範圍
        # h > 0 對應安全狀態（標準CBF約定）
        # k控制過渡的銳利度
        alpha_safety = torch.sigmoid(self.k * min_h_val)
        
        # 確保輸出形狀為 [batch_size, n_agents, 1]
        alpha_safety = alpha_safety.unsqueeze(-1)
        
        return alpha_safety
    
    def forward(
        self, 
        state: MultiAgentState, 
        raw_action: torch.Tensor, 
        alphas: Optional[torch.Tensor] = None,
        dynamic_margins: Optional[torch.Tensor] = None,
        dynamics_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Apply CBF-based safety filtering to raw actions.
        
        Args:
            state: Current environment state
            raw_action: Raw actions from policy [batch_size, n_agents, action_dim]
            alphas: Dynamic CBF alpha values [batch_size, n_agents, 1] (optional)
            dynamic_margins: Dynamic safety margins [batch_size, n_agents, 1] (optional)
            dynamics_fn: Optional function to compute control-affine dynamics
            
        Returns:
            Safe actions [batch_size, n_agents, action_dim]
        """
        # 🚀 CORE INNOVATION: Compute barrier function values with dynamic margins
        h = self.barrier_function(state, dynamic_margins)
        
        # Compute barrier function Jacobian
        dh_dx = self.barrier_jacobian(state)
        
        # Compute control-affine dynamics
        if dynamics_fn is not None:
            f, g = dynamics_fn(state)
        else:
            f, g = self.control_affine_dynamics(state)
        
        # Compute Lie derivatives
        # L_f h = dh/dx * f(x)
        # L_g h = dh/dx * g(x)
        
        # Compute L_f h (drift term): dh/dx * f(x)
        # [batch, n_agents, n_constraints, state_dim] x [batch, n_agents, state_dim]
        # -> [batch, n_agents, n_constraints]
        L_f_h = torch.sum(dh_dx * f.unsqueeze(2), dim=3)
        
        # Compute L_g h (control term): dh/dx * g(x)
        # [batch, n_agents, n_constraints, state_dim] x [batch, n_agents, state_dim, action_dim]
        # -> [batch, n_agents, n_constraints, action_dim]
        L_g_h = torch.matmul(dh_dx.view(*dh_dx.shape[:-1], 1, -1), 
                            g.view(*g.shape[:-2], -1, g.shape[-1]))
        L_g_h = L_g_h.squeeze(-2)
        
        # CBF constraint: L_f h + L_g h * u + alpha * h >= 0
        # Rearranging: L_g h * u >= -L_f h - alpha * h
        
        # Use dynamic alphas if provided, otherwise use fixed alpha
        if alphas is not None:
            # Ensure alphas are on the same device and have correct shape
            alphas = alphas.to(h.device)
            # Broadcast alphas to match h shape if needed
            # alphas: [batch, n_agents, 1] -> [batch, n_agents, n_constraints]
            alpha_values = alphas.squeeze(-1).unsqueeze(-1).expand_as(h)
        else:
            alpha_values = self.alpha
        
        # Right-hand side of constraint: -L_f h - alpha * h
        rhs = -L_f_h - alpha_values * h
        
        # If using QP solver
        if self.use_qp:
            # Solve quadratic program for each agent
            safe_action = raw_action.clone()
            
            batch_size = state.batch_size
            n_agents = state.positions.shape[1]
            
            # Process each batch and agent separately
            for b in range(batch_size):
                for i in range(n_agents):
                    # Skip if no constraints for this agent
                    if L_g_h[b, i].shape[0] == 0:
                        continue
                    
                    # QP formulation:
                    # min_u 0.5 * (u - u_raw)^T * (u - u_raw)
                    # s.t. L_g_h * u >= rhs
                    
                    # Filter constraints to include only active ones
                    # Active means the constraint is either violated or close to being violated
                    # h < margin or L_f h + L_g h * u_raw + alpha * h < 0
                    active_margin = 0.1
                    constraint_values = h[b, i]
                    constraint_derivatives = L_f_h[b, i] + torch.bmm(L_g_h[b, i].unsqueeze(1), 
                                                                  raw_action[b, i].unsqueeze(-1)).squeeze(-1)
                    
                    # Use appropriate alpha value for this agent
                    agent_alpha = alpha_values[b, i, 0] if alphas is not None else self.alpha
                    active_constraints = (constraint_values < active_margin) | \
                                       (constraint_derivatives + agent_alpha * constraint_values < 0)
                    
                    # Skip if no active constraints
                    if not torch.any(active_constraints):
                        continue
                    
                    # Extract active constraints
                    A = L_g_h[b, i][active_constraints]
                    b_qp = rhs[b, i][active_constraints]
                    
                    # Solve QP using a simple projection method
                    # This is a simplification; a full QP solver would be more robust
                    u = raw_action[b, i]
                    for _ in range(self.max_iterations):
                        # Check constraint violations
                        violations = torch.mm(A, u.unsqueeze(-1)).squeeze(-1) - b_qp
                        if torch.all(violations >= 0):
                            break
                            
                        # Compute projections for violated constraints
                        violated = violations < 0
                        if torch.any(violated):
                            # Update for each violated constraint
                            for j in torch.nonzero(violated):
                                # Compute projection
                                a_j = A[j]
                                b_j = b_qp[j]
                                
                                # Project onto constraint: u -= (a_j^T u - b_j) * a_j / ||a_j||^2
                                a_j_norm = torch.sum(a_j * a_j)
                                if a_j_norm > 1e-6:  # Avoid division by zero
                                    u = u - (torch.dot(a_j, u) - b_j) * a_j / a_j_norm
                    
                    safe_action[b, i] = u
        else:
            # Simpler safety filtering approach for each constraint
            # Instead of solving a QP, we just project the action if constraints are violated
            
            # Check which constraints are violated: L_f h + L_g h * u_raw + alpha * h < 0
            constraint_values = L_f_h + torch.matmul(L_g_h, raw_action.unsqueeze(-1)).squeeze(-1) + alpha_values * h
            violations = constraint_values < 0
            
            # Initialize safe action as raw action
            safe_action = raw_action.clone()
            
            # Process each batch and agent separately
            batch_size = state.batch_size
            n_agents = state.positions.shape[1]
            
            for b in range(batch_size):
                for i in range(n_agents):
                    # Skip if no violations for this agent
                    if not torch.any(violations[b, i]):
                        continue
                    
                    # For each violated constraint, project the action
                    for j in torch.nonzero(violations[b, i]):
                        # Skip if L_g_h is too small (constraint not controllable)
                        a_j = L_g_h[b, i, j]
                        a_j_norm = torch.sum(a_j * a_j)
                        if a_j_norm < 1e-6:
                            continue
                            
                        # Compute minimum value to satisfy constraint
                        b_j = rhs[b, i, j]
                        u = safe_action[b, i]
                        
                        # Project onto constraint: u -= (a_j^T u - b_j) * a_j / ||a_j||^2 if a_j^T u < b_j
                        if torch.dot(a_j, u) < b_j:
                            safe_action[b, i] = u - (torch.dot(a_j, u) - b_j) * a_j / a_j_norm
        
        return safe_action


class GCBFPlusAgent(nn.Module):
    """
    结合策略网络和GCBF安全层的智能体。
    
    该智能体封装了策略网络和安全层，
    提供了生成安全动作的统一接口。
    """
    
    def __init__(self, policy_network: nn.Module, safety_layer: GCBFSafetyLayer, cbf_network: Optional[nn.Module] = None):
        """
        初始化GCBF+智能体。
        
        参数:
            policy_network: 神经网络策略
            safety_layer: 用于动作过滤的CBF安全层
            cbf_network: 可选的学习屏障函数神经网络
        """
        super(GCBFPlusAgent, self).__init__()
        
        self.policy_network = policy_network
        self.safety_layer = safety_layer
        self.cbf_network = cbf_network
        
    def forward(self, state: MultiAgentState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为当前状态生成安全动作。
        
        参数:
            state: 当前环境状态
            
        返回:
            (safe_action, raw_action)的元组:
            - safe_action: 安全过滤后的动作
            - raw_action: 策略网络输出的原始动作
        """
        # 从状态获取观测
        observations = self.get_observations(state)
        
        # 从策略生成原始动作
        raw_action = self.policy_network(observations)
        
        # 应用安全过滤
        safe_action = self.safety_layer(state, raw_action)
        
        return safe_action, raw_action
    
    def get_observations(self, state: MultiAgentState) -> torch.Tensor:
        """
        Extract observations from environment state.
        
        Args:
            state: Current environment state
            
        Returns:
            Observation tensor for the policy network
        """
        # Default implementation: concatenate positions, velocities, and goals
        # This can be overridden for more complex observation spaces
        observations = torch.cat([
            state.positions,
            state.velocities,
            state.goals
        ], dim=2)
        
        return observations 