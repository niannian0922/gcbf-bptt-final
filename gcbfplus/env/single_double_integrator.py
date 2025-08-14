"""
Single-agent double integrator environment - Clean and efficient implementation.
Refactored from multi-agent to focus exclusively on single-agent scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .single_agent_env import SingleAgentEnv, SingleAgentState, StepResult
from ..cuda.interface import DifferentiableDynamics
from ..utils.autograd import apply_gradient_decay, temporal_gradient_decay

# Vision-related imports (kept for compatibility but simplified)
from .vision_renderer import SimpleDepthRenderer, create_simple_renderer


@dataclass
class SingleDoubleIntegratorState(SingleAgentState):
    """State representation for single-agent double integrator dynamics."""
    # Inherits all fields from SingleAgentState


class SingleDoubleIntegratorEnv(SingleAgentEnv):
    """
    Single-agent double integrator dynamics in a differentiable environment.
    
    Agent has state [x, y, vx, vy] and control input [fx, fy].
    Dynamics follow double integrator model: ẍ = f/m.
    """
    
    def __init__(self, config: Dict, device: Optional[torch.device] = None):
        """
        Initialize single-agent double integrator environment.
        
        Args:
            config: Environment configuration dictionary.
        """
        super(SingleDoubleIntegratorEnv, self).__init__(config)
        
        # Physical parameters
        self.mass = config.get('mass', 0.1)
        self.max_force = config.get('max_force', 1.0)
        self.cbf_alpha = config.get('cbf_alpha', 1.0)
        
        # State and action dimensions
        self.pos_dim = 2  # 2D position (x, y)
        self.vel_dim = 2  # 2D velocity (vx, vy)
        self.state_dim = 4  # x, y, vx, vy
        self.action_dim = 2  # fx, fy
        
        # Gradient decay parameters
        training_config = config.get('training', {})
        self.gradient_decay_rate = training_config.get('gradient_decay_rate', 0.95)
        self.use_gradient_decay = self.gradient_decay_rate > 0.0
        self.training = True
        
        # Vision-based observation (optional)
        vision_config = config.get('vision', {})
        self.use_vision = vision_config.get('enabled', False)
        
        if self.use_vision:
            renderer_config = {
                'image_size': vision_config.get('image_size', 64),
                'camera_fov': vision_config.get('camera_fov', 90.0),
                'camera_range': vision_config.get('camera_range', 3.0),
                'agent_radius': self.agent_radius,
                'obstacle_base_height': 0.5
            }
            self.depth_renderer = create_simple_renderer(renderer_config)
        
        # Register state transition matrices as buffers
        # State transition: x_{t+1} = A * x_t + B * u_t
        A = torch.zeros(self.state_dim, self.state_dim)
        A[0, 0] = 1.0  # x position
        A[1, 1] = 1.0  # y position
        A[2, 2] = 1.0  # vx
        A[3, 3] = 1.0  # vy
        A[0, 2] = self.dt  # x += vx * dt
        A[1, 3] = self.dt  # y += vy * dt
        self.register_buffer('A', A)
        
        B = torch.zeros(self.state_dim, self.action_dim)
        B[2, 0] = self.dt / self.mass  # dvx = fx * dt / m
        B[3, 1] = self.dt / self.mass  # dvy = fy * dt / m
        self.register_buffer('B', B)

        # Ensure buffers on the intended device if provided
        if device is not None:
            self.to(device)

    # 请将这两个方法添加到 SingleDoubleIntegratorEnv 类的内部

    # 请将这两个方法添加到 SingleDoubleIntegratorEnv 类的内部

    def get_action_bounds(self):
        """Returns the lower and upper bounds of the action space."""
        return self.action_space.low, self.action_space.high

    def render(self, mode='human'):
        """
        Renders the environment.
        NOTE: A full visualization is not implemented for this simple environment.
        This is a placeholder to satisfy the abstract base class requirement.
        """
        pass
    
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Get observation shape: [obs_dim] or [channels, height, width] for vision."""
        if self.use_vision:
            return (1, 64, 64)  # Single depth channel
        else:
            # Standard observation: own state (4) + goal (2) = 6
            return (6,)
    
    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Get action shape."""
        return (self.action_dim,)
    
    # --- 请用这个新版本完整替换你的旧方法 ---
    # --- 请用这个新版本完整替换你的旧方法 ---
    def dynamics(self, state: SingleDoubleIntegratorState, action: torch.Tensor) -> SingleDoubleIntegratorState:
        """
        Computes the next state using the system dynamics.
        This final version correctly handles the SingleDoubleIntegratorState dataclass
        by extracting tensors, performing computation, and creating a new state object.
        """
        action_in = action.reshape(-1, self.action_dim)

        # 最终修复 PART 1: 从 state 对象中正确地提取 .position 和 .velocity 张量
        state_vec = torch.cat([state.position, state.velocity], dim=-1)

        # 确保所有张量都在同一设备上（与缓冲区 self.A/self.B 一致）
        target_device = self.A.device
        state_vec = state_vec.to(target_device)
        action_in = action_in.to(target_device)

        # 调用我们的CUDA函数（或CPU回退），保持精确签名
        new_state_vec = DifferentiableDynamics.apply(state_vec, action_in, self.A, self.B)

        # 最终修复 PART 2: 创建一个全新的 SingleDoubleIntegratorState 对象来存储结果
        # 我们不再尝试 .clone()，而是直接创建一个新的实例
        new_state = SingleDoubleIntegratorState(
            position=new_state_vec[:, :self.pos_dim],
            velocity=new_state_vec[:, self.pos_dim:self.pos_dim + self.vel_dim],
            # 将其他必要的属性从旧的state对象中复制过来
            goal=state.goal,
            obstacles=state.obstacles,
            batch_size=state.batch_size,
            step_count=state.step_count + 1  # 步数加一
        )

        return new_state
    
      
    def step(self, state: SingleAgentState, action: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            state: Current state
            action: Action to take [batch_size, action_dim]
            
        Returns:
            StepResult containing new state, reward, done flag, and info
        """
        # Normalize action to [batch, action_dim] if provided as [batch, n_agents, action_dim]
        if action.dim() == 3:
            action_in = action[:, 0, :]
        else:
            action_in = action

        # Apply dynamics
        new_state = self.dynamics(state, action_in)
        
        # Compute reward (negative distance to goal)
        goal_dist = self.get_goal_distance(new_state)
        reward = -goal_dist
        
        # Check if done
        done = self.is_done(new_state)

        # Compute safety cost
        collision = self.check_obstacle_collisions(new_state)
        cost = collision.float()
        
        # Additional info
        info = {
            'goal_distance': goal_dist,
            'collision': collision,
            'success': goal_dist < self.agent_radius,
            'action': action_in,
            'raw_action': action_in,
            'alpha': alpha if alpha is not None else torch.ones(state.batch_size, 1, device=goal_dist.device) * self.cbf_alpha
        }

        return StepResult(
            next_state=new_state,
            reward=reward,
            cost=cost,
            done=done,
            info=info
        )
    
    def reset(self, batch_size: int = 1, randomize: bool = True) -> SingleAgentState:
        """
        Reset the environment to initial state.
        
        Args:
            batch_size: Number of parallel environments
            randomize: Whether to randomize initial positions
            
        Returns:
            Initial state
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        if randomize:
            # Random initial positions
            positions = torch.rand(batch_size, 2, device=device) * self.area_size
            goals = torch.rand(batch_size, 2, device=device) * self.area_size
            
            # Ensure minimum distance between start and goal
            min_dist = 0.5
            for i in range(batch_size):
                while torch.norm(positions[i] - goals[i]) < min_dist:
                    goals[i] = torch.rand(2, device=device) * self.area_size
        else:
            # Fixed positions for debugging
            positions = torch.tensor([[0.2, 0.2]], device=device).expand(batch_size, -1)
            goals = torch.tensor([[1.8, 1.8]], device=device).expand(batch_size, -1)
        
        # Zero initial velocity
        velocities = torch.zeros(batch_size, 2, device=device)
        
        # Get obstacles
        obstacles = self.get_obstacle_tensor(batch_size)
        
        initial_state = SingleDoubleIntegratorState(
            position=positions,
            velocity=velocities,
            goal=goals,
            obstacles=obstacles,
            batch_size=batch_size,
            step_count=0
        )

        # This ensures the returned state is fully on the correct device
        return initial_state.to(self.A.device)
    
    def get_observation(self, state: SingleAgentState) -> torch.Tensor:
        """
        Get observation for the agent.

        This upgraded method now includes information about the nearest obstacle
        if obstacles are enabled, producing a 9-dimensional observation vector
        to match the policy's expectation.

        Args:
            state: Current state

        Returns:
            Observation tensor [batch_size, obs_dim] (6D or 9D)
        """
        if self.use_vision:
            # Return depth image observation
            return self.render_depth(state)
        else:
            # Base observation: position (2), velocity (2), relative goal (2)
            rel_goal = state.goal - state.position
            base_obs = torch.cat([
                state.position,
                state.velocity,
                rel_goal
            ], dim=-1)

            # If obstacles are present, find the nearest one and append its info
            if state.obstacles is not None:
                batch_size = state.batch_size
                device = state.position.device

                # Extract obstacle positions and radii
                obstacle_positions = state.obstacles[..., :2]  # [batch, n_obs, 2]
                obstacle_radii = state.obstacles[..., 2:]      # [batch, n_obs, 1]

                # For each agent in the batch, find the closest obstacle
                # state.position is [batch, 2], needs unsqueezing for broadcasting
                agent_pos_expanded = state.position.unsqueeze(1) # [batch, 1, 2]

                # Calculate distances to all obstacles
                dists_sq = torch.sum((agent_pos_expanded - obstacle_positions) ** 2, dim=2) # [batch, n_obs]
                
                # Find the index of the closest obstacle for each batch item
                closest_indices = torch.argmin(dists_sq, dim=1) # [batch]

                # Gather the information of the closest obstacles
                # We use gather to select the correct obstacle for each item in the batch
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                closest_obs_pos = obstacle_positions[batch_indices, closest_indices].squeeze(1) # [batch, 2]
                closest_obs_rad = obstacle_radii[batch_indices, closest_indices].squeeze(1) # [batch, 1]

                # Calculate relative position to the closest obstacle
                rel_obs_pos = closest_obs_pos - state.position

                # Concatenate to form the final 9D observation
                # [base_obs (6D), rel_obs_pos (2D), closest_obs_rad (1D)]
                final_obs = torch.cat([base_obs, rel_obs_pos, closest_obs_rad], dim=-1)
                return final_obs
            else:
                # If no obstacles, return the base 6D observation
                return base_obs
    
    def render_depth(self, state: SingleAgentState) -> torch.Tensor:
        """
        Render depth image from agent's perspective.
        
        Args:
            state: Current state
            
        Returns:
            Depth images [batch_size, 1, H, W]
        """
        if not self.use_vision:
            raise ValueError("Vision not enabled in this environment")
        
        batch_size = state.batch_size
        depth_images = []
        
        for i in range(batch_size):
            # Get agent position and velocity (for heading)
            pos = state.position[i].cpu().numpy()
            vel = state.velocity[i].cpu().numpy()
            
            # Calculate heading from velocity or default
            if np.linalg.norm(vel) > 0.01:
                heading = np.arctan2(vel[1], vel[0])
            else:
                heading = 0.0
            
            # Get obstacles for this batch
            if state.obstacles is not None:
                obstacles = state.obstacles[i].cpu().numpy()
            else:
                obstacles = None
            
            # Render depth image
            depth = self.depth_renderer.render(
                agent_pos=pos,
                agent_heading=heading,
                obstacles=obstacles,
                goal_pos=state.goal[i].cpu().numpy()
            )
            
            depth_images.append(torch.from_numpy(depth).unsqueeze(0))
        
        return torch.stack(depth_images).to(state.position.device)