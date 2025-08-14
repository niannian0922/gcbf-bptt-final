# Multi-agent double integrator environment with safety layers

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .multi_agent_env import MultiAgentEnv, MultiAgentState, StepResult
from ..utils.autograd import apply_gradient_decay, temporal_gradient_decay

# 视觉相关导入
from .vision_renderer import SimpleDepthRenderer, create_simple_renderer


@dataclass
class DoubleIntegratorState(MultiAgentState):
    """双积分器动力学的状态表示。"""
    # 继承MultiAgentState的所有字段


class DoubleIntegratorEnv(MultiAgentEnv):
    """
    多智能体双积分器动力学的可微分环境。
    
    每个智能体具有状态[x, y, vx, vy]和控制输入[fx, fy]。
    动力学遵循双积分器模型：ẍ = f/m。
    支持自适应安全边距和时序梯度衰减机制。
    """
    
    def __init__(self, config: Dict):
        """
        初始化双积分器环境。
        
        参数:
            config: 包含环境参数的字典。
                必需键值:
                - 'num_agents': 智能体数量
                - 'area_size': 正方形环境大小
                - 'dt': 仿真时间步长
                - 'mass': 每个智能体的质量
                - 'car_radius': 每个智能体的半径（用于碰撞检测）
                - 'comm_radius': 通信半径（用于图构建）
                可选键值:
                - 'max_steps': 最大回合长度
                - 'max_force': 最大力的大小
                - 'gradient_decay_rate': 梯度随时间衰减的速率
        """
        super(DoubleIntegratorEnv, self).__init__(config)
        
        # 存储额外参数
        self.mass = config.get('mass', 0.1)
        self.max_force = config.get('max_force', 1.0)
        self.cbf_alpha = config.get('cbf_alpha', 1.0)  # 默认CBF alpha参数
        self.pos_dim = 2  # 2D位置 (x, y)
        self.vel_dim = 2  # 2D速度 (vx, vy)
        self.state_dim = 4  # x, y, vx, vy
        self.action_dim = 2  # fx, fy
        
        # 梯度衰减参数
        training_config = config.get('training', {})
        self.gradient_decay_rate = training_config.get('gradient_decay_rate', 0.95)
        self.use_gradient_decay = self.gradient_decay_rate > 0.0
        self.training = True  # 默认为训练模式
        
        # 基于视觉的观测参数
        vision_config = config.get('vision', {})
        self.use_vision = vision_config.get('enabled', False)
        
        # 如果启用视觉则初始化渲染器
        if self.use_vision:
            renderer_config = {
                'image_size': vision_config.get('image_size', 64),
                'camera_fov': vision_config.get('camera_fov', 90.0),
                'camera_range': vision_config.get('camera_range', 3.0),
                'agent_radius': self.agent_radius,
                'obstacle_base_height': 0.5
            }
            self.depth_renderer = create_simple_renderer(renderer_config)
        
        # 将状态转换矩阵注册为缓冲区
        # 状态转换: x_{t+1} = A * x_t + B * u_t
        A = torch.zeros(self.state_dim, self.state_dim)
        A[0, 0] = 1.0  # x位置
        A[1, 1] = 1.0  # y位置
        A[2, 2] = 1.0  # vx
        A[3, 3] = 1.0  # vy
        A[0, 2] = self.dt  # x += vx * dt
        A[1, 3] = self.dt  # y += vy * dt
        self.register_buffer('A', A)
        
        B = torch.zeros(self.state_dim, self.action_dim)
        B[2, 0] = self.dt / self.mass  # dvx = fx * dt / m
        B[3, 1] = self.dt / self.mass  # dvy = fy * dt / m
        self.register_buffer('B', B)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """获取观测形状: [n_agents, obs_dim] 或视觉模式下的 [n_agents, channels, height, width]"""
        if self.use_vision:
            # 基于视觉: 深度图像
            image_size = self.depth_renderer.image_size
            return (self.num_agents, 1, image_size, image_size)  # 深度图像
        else:
            # 基于状态的观测
            if self.obstacles_config is not None:
                # 包含最近障碍物的位置(2)和半径(1)
                return (self.num_agents, self.state_dim + self.pos_dim + self.pos_dim + 1)
            else:
                return (self.num_agents, self.state_dim + self.pos_dim)  # 状态 + 目标位置

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """获取动作形状: [n_agents, action_dim]"""
        return (self.num_agents, self.action_dim)
    
    def get_action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取有效动作的边界。"""
        lower_bound = -self.max_force * torch.ones(self.action_shape, device=self.device)
        upper_bound = self.max_force * torch.ones(self.action_shape, device=self.device)
        return lower_bound, upper_bound
    
    def render_depth_image(self, state: DoubleIntegratorState, agent_idx: int) -> torch.Tensor:
        """
        Render a depth image from the perspective of a specific agent using the simplified renderer.
        
        Args:
            state: Current environment state
            agent_idx: Index of the agent whose perspective to render from
            
        Returns:
            Depth image tensor [1, H, W] with values in [0, 1]
        """
        if not self.use_vision:
            raise RuntimeError("Vision rendering is not enabled")
        
        # Get agent position and goal
        agent_pos = state.positions[0, agent_idx]  # [2]
        goal_pos = state.goals[0, agent_idx]       # [2]
        
        # Get other agent positions (excluding the viewing agent)
        all_agent_positions = state.positions[0]  # [num_agents, 2]
        other_agents = torch.cat([
            all_agent_positions[:agent_idx],
            all_agent_positions[agent_idx+1:]
        ], dim=0) if len(all_agent_positions) > 1 else torch.empty(0, 2, device=agent_pos.device)
        
        # Render depth image using the simplified renderer
        depth_image = self.depth_renderer.render_depth_from_position(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            other_agents=other_agents,
            obstacles=state.obstacles[0] if state.obstacles is not None else None
        )
        
        # Add some realism with noise (optional)
        depth_image = self.depth_renderer.add_noise_and_realism(depth_image, noise_level=0.01)
        
        return depth_image  # [1, H, W]
    
    def reset(self, batch_size: int = 1, randomize: bool = True) -> DoubleIntegratorState:
        """
        Reset the environment to an initial state.
        
        Args:
            batch_size: Number of parallel environments
            randomize: Whether to randomize initial positions and goals
            
        Returns:
            Initial environment state
        """
        # Create a new random generator on the correct device
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Initialize positions randomly within the environment bounds
        if randomize:
            # 🚀 ENHANCEMENT 4: More challenging initial state randomization
            
            # Generate diverse starting configurations
            config_type = np.random.choice(['corners', 'edges', 'random', 'clustered'], p=[0.3, 0.3, 0.3, 0.1])
            
            if config_type == 'corners':
                # Start agents near corners for more challenging navigation
                corner_margin = 0.2 * self.area_size
                positions = torch.zeros(batch_size, self.num_agents, self.pos_dim, device=device)
                for i in range(self.num_agents):
                    corner = i % 4  # Cycle through 4 corners
                    if corner == 0:  # Bottom-left
                        base_pos = [corner_margin, corner_margin]
                    elif corner == 1:  # Bottom-right
                        base_pos = [self.area_size - corner_margin, corner_margin]
                    elif corner == 2:  # Top-right
                        base_pos = [self.area_size - corner_margin, self.area_size - corner_margin]
                    else:  # Top-left
                        base_pos = [corner_margin, self.area_size - corner_margin]
                    
                    # Add small random offset
                    noise = (torch.rand(batch_size, self.pos_dim, device=device) - 0.5) * 0.1 * self.area_size
                    positions[:, i, :] = torch.tensor(base_pos, device=device) + noise
            
            elif config_type == 'edges':
                # Start agents along edges
                positions = torch.zeros(batch_size, self.num_agents, self.pos_dim, device=device)
                for i in range(self.num_agents):
                    edge = i % 4  # Cycle through 4 edges
                    if edge == 0:  # Bottom edge
                        x = torch.rand(batch_size, device=device) * self.area_size
                        y = torch.rand(batch_size, device=device) * 0.2 * self.area_size
                        positions[:, i, :] = torch.stack([x, y], dim=1)
                    elif edge == 1:  # Right edge
                        x = (0.8 + 0.2 * torch.rand(batch_size, device=device)) * self.area_size
                        y = torch.rand(batch_size, device=device) * self.area_size
                        positions[:, i, :] = torch.stack([x, y], dim=1)
                    elif edge == 2:  # Top edge
                        x = torch.rand(batch_size, device=device) * self.area_size
                        y = (0.8 + 0.2 * torch.rand(batch_size, device=device)) * self.area_size
                        positions[:, i, :] = torch.stack([x, y], dim=1)
                    else:  # Left edge
                        x = torch.rand(batch_size, device=device) * 0.2 * self.area_size
                        y = torch.rand(batch_size, device=device) * self.area_size
                        positions[:, i, :] = torch.stack([x, y], dim=1)
            
            elif config_type == 'clustered':
                # Start agents in clusters for cooperation challenges
                cluster_center = torch.rand(batch_size, 1, self.pos_dim, device=device) * 0.6 * self.area_size + 0.2 * self.area_size
                cluster_radius = 0.15 * self.area_size
                positions = cluster_center + (torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) - 0.5) * cluster_radius
                positions = torch.clamp(positions, 0.05 * self.area_size, 0.95 * self.area_size)
            
            else:  # 'random'
                # Fully random positions with wider distribution
                margin = 0.05 * self.area_size  # Reduced margin for more challenging starts
                range_size = self.area_size - 2 * margin
                positions = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) * range_size + margin
            
            # 🚀 ENHANCEMENT 5: More diverse goal placement strategies
            goal_strategy = np.random.choice(['opposite', 'random', 'crossover', 'center'], p=[0.4, 0.3, 0.2, 0.1])
            
            if goal_strategy == 'opposite':
                # Goals on opposite side (traditional)
                goals = self.area_size - positions + (torch.rand_like(positions) - 0.5) * 0.3 * self.area_size
                goals = torch.clamp(goals, 0.05 * self.area_size, 0.95 * self.area_size)
            
            elif goal_strategy == 'crossover':
                # Agents must cross paths to reach goals
                goals = torch.flip(positions, dims=[1])  # Reverse agent order
                # Add some randomization
                goals += (torch.rand_like(goals) - 0.5) * 0.2 * self.area_size
                goals = torch.clamp(goals, 0.05 * self.area_size, 0.95 * self.area_size)
            
            elif goal_strategy == 'center':
                # Goals clustered toward center, requiring navigation through middle
                center = self.area_size / 2
                center_region = 0.3 * self.area_size
                goals = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) * center_region
                goals += center - center_region / 2
            
            else:  # 'random'
                # Fully random goals
                margin = 0.05 * self.area_size
                range_size = self.area_size - 2 * margin
                goals = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) * range_size + margin
            
            # 🚀 ENHANCEMENT 6: Adaptive minimum distance based on difficulty
            difficulty_factor = np.random.uniform(0.7, 1.3)  # Vary difficulty
            min_goal_dist = 0.25 * self.area_size * difficulty_factor
            
            # Ensure goals are not too close to initial positions
            max_repositioning_attempts = 20
            for attempt in range(max_repositioning_attempts):
                dist = torch.norm(positions - goals, dim=2)
                too_close = dist < min_goal_dist
                
                if not torch.any(too_close):
                    break
                
                # Replace goals that are too close with more strategic placement
                for b in range(batch_size):
                    for a in range(self.num_agents):
                        if too_close[b, a]:
                            # Try to place goal in a challenging but reachable position
                            agent_pos = positions[b, a].cpu().numpy()
                            
                            # Generate candidates in different directions
                            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                            best_goal = None
                            best_score = -1
                            
                            for angle in angles:
                                candidate_distance = min_goal_dist + np.random.uniform(0, 0.3 * self.area_size)
                                candidate_goal = agent_pos + candidate_distance * np.array([np.cos(angle), np.sin(angle)])
                                
                                # Check if candidate is within bounds
                                if (0.05 * self.area_size <= candidate_goal[0] <= 0.95 * self.area_size and
                                    0.05 * self.area_size <= candidate_goal[1] <= 0.95 * self.area_size):
                                    
                                    # Score based on distance from other agents' goals
                                    other_goals = goals[b, :a].cpu().numpy() if a > 0 else np.array([]).reshape(0, 2)
                                    if a < self.num_agents - 1:
                                        other_goals = np.vstack([other_goals, goals[b, a+1:].cpu().numpy()]) if other_goals.size > 0 else goals[b, a+1:].cpu().numpy()
                                    
                                    if other_goals.size > 0:
                                        min_dist_to_others = np.min(np.linalg.norm(other_goals - candidate_goal, axis=1))
                                        score = min_dist_to_others
                                    else:
                                        score = 1.0
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_goal = candidate_goal
                            
                            if best_goal is not None:
                                goals[b, a] = torch.tensor(best_goal, device=device)
                            else:
                                # Fallback: random position far enough
                                margin = 0.1 * self.area_size
                                range_size = self.area_size - 2 * margin
                                goals[b, a] = torch.rand(self.pos_dim, device=device) * range_size + margin
        else:
            # Default initialization: agents in a grid, goals opposite side
            positions = torch.zeros(batch_size, self.num_agents, self.pos_dim, device=device)
            goals = torch.zeros(batch_size, self.num_agents, self.pos_dim, device=device)
            
            # Create a grid arrangement for agents
            grid_size = int(np.ceil(np.sqrt(self.num_agents)))
            spacing = self.area_size / (grid_size + 1)
            
            for i in range(self.num_agents):
                row, col = i // grid_size, i % grid_size
                positions[:, i, 0] = spacing * (col + 1)
                positions[:, i, 1] = spacing * (row + 1)
                
                # Goals on opposite side
                goals[:, i, 0] = self.area_size - positions[0, i, 0]
                goals[:, i, 1] = self.area_size - positions[0, i, 1]
        
        # Initialize velocities as zeros
        velocities = torch.zeros(batch_size, self.num_agents, self.vel_dim, device=device)
        
        # Get obstacles tensor
        obstacles = self.get_obstacle_tensor(batch_size)
        
        # If obstacles exist, ensure agents and goals are not initialized inside obstacles
        if obstacles is not None:
            # Extract obstacle positions and radii
            obstacle_positions = obstacles[..., :-1]  # [batch_size, n_obstacles, pos_dim]
            obstacle_radii = obstacles[..., -1:]     # [batch_size, n_obstacles, 1]
            
            # Ensure agents are not inside obstacles
            for b in range(batch_size):
                for i in range(self.num_agents):
                    # Check distance to all obstacles
                    agent_pos = positions[b, i].unsqueeze(0)  # [1, pos_dim]
                    
                    # Compute distances
                    dists = torch.norm(agent_pos - obstacle_positions[b], dim=1)
                    min_dists = dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                    
                    # If agent is inside any obstacle, move it
                    if torch.any(min_dists < 0):
                        # Try several random positions until we find one that's valid
                        max_tries = 50
                        for _ in range(max_tries):
                            new_pos = torch.rand(self.pos_dim, device=device) * self.area_size
                            new_dists = torch.norm(new_pos - obstacle_positions[b], dim=1)
                            new_min_dists = new_dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                            
                            if torch.all(new_min_dists >= 0):
                                positions[b, i] = new_pos
                                break
            
            # Ensure goals are not inside obstacles
            for b in range(batch_size):
                for i in range(self.num_agents):
                    # Check distance to all obstacles
                    goal_pos = goals[b, i].unsqueeze(0)  # [1, pos_dim]
                    
                    # Compute distances
                    dists = torch.norm(goal_pos - obstacle_positions[b], dim=1)
                    min_dists = dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                    
                    # If goal is inside any obstacle, move it
                    if torch.any(min_dists < 0):
                        # Try several random positions until we find one that's valid
                        max_tries = 50
                        for _ in range(max_tries):
                            new_pos = torch.rand(self.pos_dim, device=device) * self.area_size
                            new_dists = torch.norm(new_pos - obstacle_positions[b], dim=1)
                            new_min_dists = new_dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                            
                            if torch.all(new_min_dists >= 0):
                                goals[b, i] = new_pos
                                break
        
        # Create state object
        state = DoubleIntegratorState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            obstacles=obstacles,  # Include obstacles in state
            batch_size=batch_size,
            step_count=0
        )
        
        return state
    
    def step(self, state: DoubleIntegratorState, action: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> StepResult:
        """
        Take a step in the environment using the given actions and dynamic alpha values.
        
        Args:
            state: Current state
            action: Actions to take [batch_size, n_agents, action_dim]
            alpha: Dynamic CBF alpha values [batch_size, n_agents, 1] (optional)
            
        Returns:
            StepResult containing next_state, reward, cost, done, info
        """
        # Ensure action is on the correct device first
        action = action.to(self.device)
        
        # Ensure alpha is on the correct device if provided
        if alpha is not None:
            alpha = alpha.to(self.device)
        
        # 🛡️ Apply probabilistic safety shield with dynamic margins
        # Note: dynamic_margins would be passed from the policy network if available
        safe_action, alpha_safety = self.apply_safety_layer(state, action, alpha, None)
        
        # Ensure safe_action is also on the correct device
        safe_action = safe_action.to(self.device)
        
        # Clip action to bounds
        # Ensure bounds are on the same device as the state and actions
        lower_bound = torch.tensor([-1.0, -1.0], device=self.device, dtype=safe_action.dtype)
        upper_bound = torch.tensor([1.0, 1.0], device=self.device, dtype=safe_action.dtype)
        safe_action = torch.clamp(safe_action, lower_bound, upper_bound)
        
        # Compute state derivatives using dynamics
        derivatives = self.dynamics(state, safe_action)
        
        # Update positions and velocities using Euler integration with gradient decay
        if self.use_gradient_decay and self.training:
            # Apply gradient decay to stabilize long-horizon BPTT training
            decay_factor = torch.tensor(self.gradient_decay_rate, device=state.positions.device)
            positions_decayed = apply_gradient_decay(state.positions, decay_factor)
            velocities_decayed = apply_gradient_decay(state.velocities, decay_factor)
            
            new_positions = positions_decayed + velocities_decayed * self.dt
            new_velocities = velocities_decayed + (safe_action / self.mass) * self.dt
        else:
            # Standard update without gradient decay
            new_positions = state.positions + state.velocities * self.dt
            new_velocities = state.velocities + (safe_action / self.mass) * self.dt
        
        # Create next state
        next_state = DoubleIntegratorState(
            positions=new_positions,
            velocities=new_velocities,
            goals=state.goals,
            obstacles=state.obstacles,  # Preserve obstacles in next state
            batch_size=state.batch_size,
            step_count=state.step_count + 1
        )
        
        # Compute reward, cost, and done
        reward = self.compute_reward(state, safe_action, next_state)
        cost = self.compute_cost(state, safe_action, next_state)
        done = self.is_done(next_state)
        
        # Prepare info dict
        info = {
            'goal_distance': self.get_goal_distance(next_state),
            'collision': self.check_collision(next_state),
            'action': safe_action,
            'raw_action': action,
            'alpha': alpha if alpha is not None else torch.ones_like(action[..., :1]) * self.cbf_alpha  # Default alpha
        }
        
        return StepResult(next_state, reward, cost, done, info)
    
    def get_observation(self, state: DoubleIntegratorState) -> torch.Tensor:
        """
        Extract observation from environment state.
        
        Returns either state-based or vision-based observations depending on configuration.
        
        State-based observations include:
        - Agent position (x, y), velocity (vx, vy), goal position (gx, gy)
        - Obstacle information (if present)
        
        Vision-based observations include:
        - Depth image from each agent's perspective [n_agents, 1, H, W]
        
        Args:
            state: Current environment state
            
        Returns:
            Observation tensor: 
            - State-based: [batch_size, n_agents, obs_dim]
            - Vision-based: [batch_size, n_agents, 1, H, W]
        """
        batch_size = state.batch_size
        device = state.positions.device
        
        if self.use_vision:
            # Vision-based observations: render depth images for each agent
            vision_obs = []
            
            for agent_idx in range(self.num_agents):
                depth_image = self.render_depth_image(state, agent_idx)  # [1, H, W]
                vision_obs.append(depth_image)
            
            # Stack depth images for all agents
            stacked_obs = torch.stack(vision_obs, dim=0)  # [n_agents, 1, H, W]
            
            # Add batch dimension
            observation = stacked_obs.unsqueeze(0)  # [1, n_agents, 1, H, W]
            
            return observation
        
        else:
            # State-based observations (original implementation)
            # First combine position, velocity and goal information
            base_obs = torch.cat([
                state.positions,                              # positions
                state.velocities,                             # velocities
                state.goals                                   # goals
            ], dim=2)
        
        # If we have obstacles, include information about closest obstacle to each agent
        if state.obstacles is not None:
            # Extract obstacle positions and radii
            obstacle_positions = state.obstacles[..., :-1]  # [batch_size, n_obstacles, pos_dim]
            obstacle_radii = state.obstacles[..., -1:]     # [batch_size, n_obstacles, 1]
            
            # For each agent, find the closest obstacle and include it in observation
            closest_obstacles = torch.zeros(batch_size, self.num_agents, self.pos_dim + 1, device=device)
            
            for b in range(batch_size):
                for i in range(self.num_agents):
                    # Compute distances to all obstacles
                    agent_pos = state.positions[b, i].unsqueeze(0)  # [1, pos_dim]
                    dists = torch.norm(agent_pos - obstacle_positions[b], dim=1)
                    
                    # Find the closest obstacle
                    closest_idx = torch.argmin(dists)
                    
                    # Store position and radius of closest obstacle relative to agent
                    closest_pos = obstacle_positions[b, closest_idx] - state.positions[b, i]
                    closest_rad = obstacle_radii[b, closest_idx]
                    
                    closest_obstacles[b, i, :self.pos_dim] = closest_pos
                    closest_obstacles[b, i, self.pos_dim] = closest_rad
            
            # Combine with base observation
            observation = torch.cat([base_obs, closest_obstacles], dim=2)
        else:
            observation = base_obs
        
        return observation
    
    def render(self, state: DoubleIntegratorState) -> Any:
        """
        Render the environment state. This is a placeholder implementation.
        
        Args:
            state: Environment state to render
            
        Returns:
            Rendering of the environment (matplotlib figure)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Only render the first batch
        batch_idx = 0
        
        # Draw agents as circles
        for i in range(self.num_agents):
            agent_pos = state.positions[batch_idx, i].detach().cpu().numpy()
            agent = Circle(agent_pos, self.agent_radius, color='blue', alpha=0.7)
            ax.add_patch(agent)
            
            # Draw velocity vector
            vel = state.velocities[batch_idx, i].detach().cpu().numpy()
            ax.arrow(agent_pos[0], agent_pos[1], vel[0]*0.1, vel[1]*0.1, 
                     head_width=0.02, head_length=0.03, fc='black', ec='black')
            
            # Draw goal
            goal_pos = state.goals[batch_idx, i].detach().cpu().numpy()
            ax.scatter(goal_pos[0], goal_pos[1], color='green', marker='*', s=100)
        
        # Draw obstacles if present
        if state.obstacles is not None:
            obstacles = state.obstacles[batch_idx].detach().cpu().numpy()
            for obs in obstacles:
                obs_pos = obs[:-1]
                obs_radius = obs[-1]
                obstacle = Circle(obs_pos, obs_radius, color='red', alpha=0.3)
                ax.add_patch(obstacle)
        
        # Set limits and labels
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Double Integrator Environment')
        
        plt.tight_layout()
        
        return fig
    
    def apply_safety_layer(
        self, 
        state: DoubleIntegratorState, 
        raw_action: torch.Tensor, 
        alpha: Optional[torch.Tensor] = None,
        dynamic_margins: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        🛡️ PROBABILISTIC SAFETY SHIELD: 應用概率安全防護罩
        
        使用安全信心分數混合策略動作和安全後備動作，而不是直接過濾動作。
        這解耦了"安全"和"效率"目標。
        
        Args:
            state: Current environment state
            raw_action: Raw action from policy [batch_size, n_agents, action_dim] (策略網絡的積極行動)
            alpha: Dynamic CBF alpha values [batch_size, n_agents, 1] (optional)
            dynamic_margins: Dynamic safety margins [batch_size, n_agents, 1] (optional)
            
        Returns:
            Tuple of (blended_action, alpha_safety):
            - blended_action: 混合後的最終動作 [batch_size, n_agents, action_dim]
            - alpha_safety: 安全信心分數 [batch_size, n_agents, 1] or None
        """
        # Store alpha values for logging and potential future use
        if alpha is not None:
            self._current_alpha = alpha
        
        # If a safety layer is available, use it to compute safety confidence
        if hasattr(self, 'safety_layer') and self.safety_layer is not None:
            try:
                # 🛡️ 計算安全信心分數而不是直接過濾動作
                alpha_safety = self.safety_layer.compute_safety_confidence(state, dynamic_margins)
                
                # 🛡️ 定義超級保守的安全後備動作（懸停，零速度）
                safe_action = torch.zeros_like(raw_action)  # 懸停動作
                
                # 🛡️ 使用安全信心分數混合動作
                # final_action = alpha_safety * policy_output + (1 - alpha_safety) * safe_action
                # alpha_safety接近1：信任策略網絡的積極動作
                # alpha_safety接近0：使用保守的安全動作
                blended_action = alpha_safety * raw_action + (1 - alpha_safety) * safe_action
                
                return blended_action, alpha_safety
                
            except Exception as e:
                # If safety layer fails, fall back to raw action with no confidence score
                print(f"Warning: Safety layer failed: {e}. Using raw action.")
                return raw_action, None
        else:
            # Default implementation: return raw action if no safety layer
            return raw_action, None

    def dynamics(self, state: DoubleIntegratorState, action: torch.Tensor) -> torch.Tensor:
        """
        Apply double integrator dynamics to compute state derivatives.
        
        Args:
            state: Current state
            action: Control inputs [batch_size, n_agents, 2]
            
        Returns:
            State derivatives [batch_size, n_agents, 4]
        """
        # Extract batch size for convenience
        batch_size = state.batch_size
        
        # Reshape state and action for matrix multiplication
        # Combine batch and agent dimensions: [batch_size * n_agents, state_dim/action_dim]
        states_flat = torch.cat([
            state.positions.reshape(batch_size * self.num_agents, self.pos_dim),
            state.velocities.reshape(batch_size * self.num_agents, self.vel_dim)
        ], dim=1)
        actions_flat = action.reshape(batch_size * self.num_agents, self.action_dim)
        
        # Apply dynamics: x_dot = Ax + Bu
        # We don't use the full state transition here, just the derivatives
        acceleration = actions_flat / self.mass
        
        # Create derivatives
        derivatives = torch.cat([
            state.velocities.reshape(batch_size * self.num_agents, self.vel_dim),
            acceleration
        ], dim=1)
        
        # Reshape back to [batch_size, n_agents, state_dim]
        derivatives = derivatives.reshape(batch_size, self.num_agents, self.state_dim)
        
        return derivatives

    def train(self) -> None:
        """将环境设置为训练模式。这会启用梯度衰减。"""
        self.training = True
    
    def eval(self) -> None:
        """将环境设置为评估模式。这会禁用梯度衰减。"""
        self.training = False
