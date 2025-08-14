import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from abc import abstractmethod

from .base_env import BaseEnv, EnvState, StepResult


@dataclass
class MultiAgentState(EnvState):
    """多智能体环境状态信息的容器。"""
    positions: torch.Tensor                  # Shape: [batch_size, n_agents, pos_dim]
    velocities: torch.Tensor                 # Shape: [batch_size, n_agents, vel_dim]
    goals: torch.Tensor                      # Shape: [batch_size, n_agents, pos_dim]
    orientations: Optional[torch.Tensor] = None  # Shape: [batch_size, n_agents, orientation_dim]
    obstacles: Optional[torch.Tensor] = None  # Shape: [batch_size, n_obstacles, pos_dim+1] (positions + radii)
    batch_size: int = field(default=1)
    step_count: int = field(default=0)
    
    @property
    def n_agents(self) -> int:
        return self.positions.shape[1]
    
    @property
    def pos_dim(self) -> int:
        return self.positions.shape[2]
    
    @property
    def state_tensor(self) -> torch.Tensor:
        """为所有智能体创建组合状态张量。"""
        if self.orientations is not None:
            # Include orientations in state
            return torch.cat([
                self.positions,
                self.velocities,
                self.orientations
            ], dim=2)
        else:
            # Just position and velocity
            return torch.cat([
                self.positions,
                self.velocities
            ], dim=2)


class MultiAgentEnv(BaseEnv):
    """
    Abstract base class for multi-agent differentiable environments.
    
    This class extends BaseEnv with functionality specific to multi-agent systems.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multi-agent environment.
        
        Args:
            config: Dictionary containing environment configuration parameters.
                   Must include 'num_agents', 'area_size', 'dt'.
        """
        super(MultiAgentEnv, self).__init__(config)
        
        # Extract common parameters
        self.num_agents = config['num_agents']
        self.area_size = config['area_size']
        self.dt = config['dt']
        self.max_steps = config.get('max_steps', 256)
        self.agent_radius = config.get('agent_radius', 0.05)
        self.comm_radius = config.get('comm_radius', 0.5)
        
        # Set up obstacles
        self.obstacles_config = config.get('obstacles', None)
        self.static_obstacles = None
        
        if self.obstacles_config is not None:
            self._setup_static_obstacles(self.obstacles_config)
    
    def _setup_static_obstacles(self, obstacles_config: Dict):
        """
        Set up static obstacles based on configuration.
        
        Args:
            obstacles_config: Dictionary containing obstacle configuration
                Expected format:
                - 'num_obstacles': Number of obstacles
                - 'positions': List of positions [[x1, y1], [x2, y2], ...]
                - 'radii': List of radii [r1, r2, ...]
                - 'random': If True, generate random obstacles
                - 'random_count': Number of random obstacles (or range [min, max])
                - 'random_min_radius': Minimum radius for random obstacles
                - 'random_max_radius': Maximum radius for random obstacles
                - 'dynamic_count': If True, randomize obstacle count each reset
                - 'count_range': [min_count, max_count] for dynamic obstacle count
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Initialize obstacle storage
        positions = []
        radii = []
        
        # Add explicitly configured obstacles
        if 'positions' in obstacles_config and 'radii' in obstacles_config:
            pos_list = obstacles_config['positions']
            rad_list = obstacles_config['radii']
            
            if len(pos_list) != len(rad_list):
                raise ValueError("Obstacle positions and radii must have the same length")
                
            for pos, rad in zip(pos_list, rad_list):
                positions.append(pos)
                radii.append(rad)
        
        # Generate random obstacles if requested
        if obstacles_config.get('random', False):
            random_count = obstacles_config.get('random_count', 1)
            min_radius = obstacles_config.get('random_min_radius', 0.1)
            max_radius = obstacles_config.get('random_max_radius', 0.3)
            
            for _ in range(random_count):
                # Generate random position within area bounds
                pos = [
                    np.random.uniform(0, self.area_size),
                    np.random.uniform(0, self.area_size)
                ]
                # Generate random radius
                rad = np.random.uniform(min_radius, max_radius)
                
                positions.append(pos)
                radii.append(rad)
        
        # Create a central obstacle if specified
        if obstacles_config.get('center_obstacle', False):
            center_radius = obstacles_config.get('center_radius', 0.3)
            positions.append([self.area_size/2, self.area_size/2])
            radii.append(center_radius)
        
        # Create bottleneck obstacles if specified
        if obstacles_config.get('bottleneck', False):
            # Bottleneck configuration parameters
            gap_width = obstacles_config.get('gap_width', 0.3)
            wall_thickness = obstacles_config.get('wall_thickness', 0.1)
            wall_height = obstacles_config.get('wall_height', 0.8)
            gap_position = obstacles_config.get('gap_position', 0.5)  # 0.5 = center
            
            # Create two walls with a gap in the middle
            # Calculate wall dimensions
            center_y = self.area_size * gap_position
            gap_half_width = gap_width / 2
            wall_half_height = wall_height / 2
            
            # Wall positions (center of each wall segment)
            upper_wall_y = center_y + gap_half_width + wall_half_height
            lower_wall_y = center_y - gap_half_width - wall_half_height
            wall_x = self.area_size / 2
            
            # Create multiple circular obstacles to form wall segments
            obstacle_spacing = obstacles_config.get('obstacle_spacing', 0.08)
            obstacle_radius = obstacles_config.get('obstacle_radius', 0.06)
            
            # Upper wall
            if upper_wall_y - wall_half_height < self.area_size:
                wall_start = max(upper_wall_y - wall_half_height, center_y + gap_half_width)
                wall_end = min(upper_wall_y + wall_half_height, self.area_size)
                
                num_obstacles = int((wall_end - wall_start) / obstacle_spacing) + 1
                for i in range(num_obstacles):
                    y_pos = wall_start + i * obstacle_spacing
                    if y_pos <= wall_end:
                        positions.append([wall_x, y_pos])
                        radii.append(obstacle_radius)
            
            # Lower wall  
            if lower_wall_y + wall_half_height > 0:
                wall_start = max(lower_wall_y - wall_half_height, 0)
                wall_end = min(lower_wall_y + wall_half_height, center_y - gap_half_width)
                
                num_obstacles = int((wall_end - wall_start) / obstacle_spacing) + 1
                for i in range(num_obstacles):
                    y_pos = wall_start + i * obstacle_spacing
                    if y_pos <= wall_end:
                        positions.append([wall_x, y_pos])
                        radii.append(obstacle_radius)
            
        # Convert to tensors
        if positions and radii:
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
            rad_tensor = torch.tensor(radii, dtype=torch.float32, device=device).unsqueeze(1)
            
            # Combine into a single tensor [n_obstacles, pos_dim+1]
            # where last column is the radius
            self.static_obstacles = torch.cat([pos_tensor, rad_tensor], dim=1)
        else:
            self.static_obstacles = None
    
    def _generate_dynamic_obstacles(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """
        Generate dynamic obstacles with enhanced randomization.
        
        Args:
            batch_size: Batch size for the tensor
            
        Returns:
            Dynamic obstacle tensor [batch_size, n_obstacles, pos_dim+1] or None
        """
        if self.obstacles_config is None:
            return None
            
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # 🚀 ENHANCEMENT 1: Dynamic obstacle count randomization
        dynamic_count = self.obstacles_config.get('dynamic_count', False)
        if dynamic_count:
            count_range = self.obstacles_config.get('count_range', [2, 8])
            min_count, max_count = count_range
            num_obstacles = np.random.randint(min_count, max_count + 1)
        else:
            num_obstacles = self.obstacles_config.get('random_count', 3)
        
        if num_obstacles == 0:
            return None
        
        # 🚀 ENHANCEMENT 2: Enhanced obstacle property randomization
        min_radius = self.obstacles_config.get('random_min_radius', 0.08)  # Increased from 0.1
        max_radius = self.obstacles_config.get('random_max_radius', 0.5)   # Increased from 0.3
        
        # Generate obstacles for each batch element
        all_obstacles = []
        for b in range(batch_size):
            positions = []
            radii = []
            
            for _ in range(num_obstacles):
                # 🚀 ENHANCEMENT 3: More diverse obstacle placement
                # Use wider range and avoid clustering
                max_attempts = 50
                for attempt in range(max_attempts):
                    # Generate position with wider area coverage
                    margin = 0.1  # Reduced margin for more challenging placement
                    pos = [
                        np.random.uniform(margin, self.area_size - margin),
                        np.random.uniform(margin, self.area_size - margin)
                    ]
                    
                    # Generate radius with log-normal distribution for more variety
                    radius = np.random.uniform(min_radius, max_radius)
                    
                    # Check distance from existing obstacles to avoid clustering
                    valid_position = True
                    for existing_pos, existing_rad in zip(positions, radii):
                        dist = np.sqrt((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)
                        min_distance = radius + existing_rad + 0.1  # Minimum separation
                        if dist < min_distance:
                            valid_position = False
                            break
                    
                    if valid_position or attempt == max_attempts - 1:
                        positions.append(pos)
                        radii.append(radius)
                        break
            
            if positions:
                # Create obstacle tensor for this batch element
                pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
                rad_tensor = torch.tensor(radii, dtype=torch.float32, device=device).unsqueeze(1)
                obstacles = torch.cat([pos_tensor, rad_tensor], dim=1)
                all_obstacles.append(obstacles)
        
        if all_obstacles:
            # Pad obstacles to same size and stack
            max_obstacles = max(obs.shape[0] for obs in all_obstacles)
            padded_obstacles = []
            
            for obstacles in all_obstacles:
                if obstacles.shape[0] < max_obstacles:
                    # Pad with dummy obstacles (position far away, zero radius)
                    padding_size = max_obstacles - obstacles.shape[0]
                    dummy_obstacles = torch.zeros(padding_size, 3, device=device)
                    dummy_obstacles[:, :2] = -100  # Far away position
                    obstacles = torch.cat([obstacles, dummy_obstacles], dim=0)
                padded_obstacles.append(obstacles)
            
            return torch.stack(padded_obstacles, dim=0)
        
        return None

    def get_obstacle_tensor(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """
        Get obstacle tensor for the environment state.
        
        Args:
            batch_size: Batch size for the tensor
            
        Returns:
            Obstacle tensor [batch_size, n_obstacles, pos_dim+1] or None if no obstacles
        """
        # 🚀 Use dynamic obstacle generation if enabled
        if self.obstacles_config and self.obstacles_config.get('dynamic_count', False):
            return self._generate_dynamic_obstacles(batch_size)
        
        if self.static_obstacles is None:
            return None
        
        # Expand static obstacles to batch dimension
        return self.static_obstacles.unsqueeze(0).expand(batch_size, -1, -1)
        
    def get_pairwise_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between agents.
        
        Args:
            positions: Agent positions tensor [batch_size, n_agents, pos_dim]
            
        Returns:
            Pairwise distances tensor [batch_size, n_agents, n_agents]
        """
        # Compute differences between all pairs of positions
        # This creates a tensor of shape [batch_size, n_agents, n_agents, pos_dim]
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        
        # Compute squared distances
        squared_dist = torch.sum(diff * diff, dim=3)
        
        # Return pairwise distances
        return torch.sqrt(squared_dist + 1e-8)  # Add small epsilon for numerical stability
    
    def check_agent_collisions(self, state: MultiAgentState) -> torch.Tensor:
        """
        Check for collisions between agents.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean tensor [batch_size] indicating if any collision occurred in each batch
        """
        # Compute pairwise distances
        distances = self.get_pairwise_distances(state.positions)
        
        # Compute threshold for collision (sum of agent radii)
        collision_threshold = 2 * self.agent_radius
        
        # Create a mask that excludes self-distances (diagonal elements)
        mask = ~torch.eye(state.n_agents, dtype=torch.bool, device=state.positions.device).unsqueeze(0)
        
        # Check which distances are below threshold
        collisions = (distances < collision_threshold) & mask
        
        # For each batch element, check if any collision occurred
        any_collision = torch.any(collisions, dim=(1, 2))
        
        return any_collision
    
    def check_obstacle_collisions(self, state: MultiAgentState) -> torch.Tensor:
        """
        Check for collisions between agents and obstacles.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean tensor [batch_size] indicating if any collision occurred in each batch
        """
        if state.obstacles is None:
            # No obstacles in the environment
            return torch.zeros(state.batch_size, dtype=torch.bool, device=state.positions.device)
        
        # Extract obstacle positions and radii
        obstacle_positions = state.obstacles[..., :-1]  # [batch_size, n_obstacles, pos_dim]
        obstacle_radii = state.obstacles[..., -1:]     # [batch_size, n_obstacles, 1]
        
        # Reshape for broadcasting
        agent_positions = state.positions.unsqueeze(2)    # [batch_size, n_agents, 1, pos_dim]
        obstacle_positions = obstacle_positions.unsqueeze(1)  # [batch_size, 1, n_obstacles, pos_dim]
        
        # Compute distances between all agents and all obstacles
        diff = agent_positions - obstacle_positions
        distances = torch.sqrt(torch.sum(diff * diff, dim=3) + 1e-8)  # [batch_size, n_agents, n_obstacles]
        
        # Compute collision thresholds
        collision_thresholds = obstacle_radii.squeeze(-1).unsqueeze(1) + self.agent_radius  # [batch_size, 1, n_obstacles]
        
        # Check which distances are below threshold
        collisions = distances < collision_thresholds
        
        # For each batch element, check if any collision occurred
        any_collision = torch.any(collisions, dim=(1, 2))
        
        return any_collision
    
    def check_collision(self, state: MultiAgentState) -> torch.Tensor:
        """
        Check for any collisions (agent-agent or agent-obstacle).
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean tensor [batch_size] indicating if any collision occurred in each batch
        """
        agent_collisions = self.check_agent_collisions(state)
        obstacle_collisions = self.check_obstacle_collisions(state)
        
        return agent_collisions | obstacle_collisions
    
    def get_goal_distance(self, state: MultiAgentState) -> torch.Tensor:
        """
        Calculate distance to goals for all agents.
        
        Args:
            state: Current environment state
            
        Returns:
            Tensor [batch_size] with average distance to goals for each batch
        """
        # Compute differences between positions and goals
        diff = state.positions - state.goals
        
        # Compute distances
        distances = torch.sqrt(torch.sum(diff * diff, dim=2) + 1e-8)
        
        # Average over agents
        avg_distances = torch.mean(distances, dim=1)
        
        return avg_distances
    
    def is_done(self, state: MultiAgentState) -> torch.Tensor:
        """
        Check if episodes are done.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean tensor [batch_size] indicating which episodes are done
        """
        # Episode is done if max steps reached or collision occurred
        max_steps_reached = state.step_count >= self.max_steps
        collisions = self.check_collision(state)
        
        # Also check if all agents reached their goals
        goal_dists = torch.norm(state.positions - state.goals, dim=2)  # [batch_size, n_agents]
        goal_reached = torch.all(goal_dists < self.agent_radius, dim=1)
        
        return collisions | goal_reached | max_steps_reached
    
    def compute_connectivity_graph(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute connectivity graph based on communication radius.
        
        Args:
            positions: Agent positions tensor [batch_size, n_agents, pos_dim]
            
        Returns:
            Tuple of (adjacency matrix, edge features)
            - adjacency matrix: [batch_size, n_agents, n_agents], 1 if agents are connected
            - edge features: [batch_size, n_agents, n_agents, edge_feature_dim]
        """
        # Compute pairwise distances
        distances = self.get_pairwise_distances(positions)
        
        # Compute adjacency matrix based on communication radius
        adjacency = (distances < self.comm_radius) & ~torch.eye(
            self.num_agents, dtype=torch.bool, device=positions.device
        ).unsqueeze(0)
        
        # Compute relative positions for edge features
        rel_positions = positions.unsqueeze(1) - positions.unsqueeze(2)  # [batch_size, n_agents, n_agents, pos_dim]
        
        # Use adjacency matrix to mask edge features
        mask = adjacency.unsqueeze(-1).expand_as(rel_positions)
        edge_features = rel_positions * mask.float()
        
        return adjacency, edge_features

    @abstractmethod
    def dynamics(self, state: MultiAgentState, action: torch.Tensor) -> torch.Tensor:
        """
        Apply system dynamics to compute state derivatives.
        
        Args:
            state: Current state of the environment
            action: Actions to apply [batch_size, n_agents, action_dim]
            
        Returns:
            State derivatives [batch_size, n_agents, state_dim]
        """
        pass 
