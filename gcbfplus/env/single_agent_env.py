"""
Single-agent environment base class - Lightweight and efficient foundation.
This is a complete refactoring for single-agent-only scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from abc import abstractmethod

from .base_env import BaseEnv, EnvState, StepResult


@dataclass
class SingleAgentState(EnvState):
    """Single agent environment state - simplified and efficient."""
    position: torch.Tensor          # Shape: [batch_size, pos_dim]
    velocity: torch.Tensor          # Shape: [batch_size, vel_dim]
    goal: torch.Tensor              # Shape: [batch_size, pos_dim]
    orientation: Optional[torch.Tensor] = None  # Shape: [batch_size, orientation_dim]
    obstacles: Optional[torch.Tensor] = None    # Shape: [batch_size, n_obstacles, pos_dim+1]
    batch_size: int = field(default=1)
    step_count: int = field(default=0)
    
    @property
    def pos_dim(self) -> int:
        return self.position.shape[-1]
    
    @property
    def state_tensor(self) -> torch.Tensor:
        """Create combined state tensor for the single agent."""
        if self.orientation is not None:
            return torch.cat([
                self.position,
                self.velocity,
                self.orientation
            ], dim=-1)
        else:
            return torch.cat([
                self.position,
                self.velocity
            ], dim=-1)


class SingleAgentEnv(BaseEnv):
    """
    Abstract base class for single-agent differentiable environments.
    Optimized for efficiency and simplicity - no multi-agent overhead.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the single-agent environment.
        
        Args:
            config: Dictionary containing environment configuration.
        """
        super(SingleAgentEnv, self).__init__(config)
        
        # Extract parameters - no num_agents needed!
        self.area_size = config['area_size']
        self.dt = config['dt']
        self.max_steps = config.get('max_steps', 256)
        self.agent_radius = config.get('agent_radius', 0.05)
        
        # Obstacle configuration
        self.obstacles_config = config.get('obstacles', None)
        self.static_obstacles = None
        
        if self.obstacles_config is not None:
            self._setup_static_obstacles(self.obstacles_config)
    
    def _setup_static_obstacles(self, obstacles_config: Dict):
        """
        Set up static obstacles based on configuration.
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
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
                pos = [
                    np.random.uniform(0, self.area_size),
                    np.random.uniform(0, self.area_size)
                ]
                rad = np.random.uniform(min_radius, max_radius)
                
                positions.append(pos)
                radii.append(rad)
        
        # Convert to tensors
        if positions and radii:
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
            rad_tensor = torch.tensor(radii, dtype=torch.float32, device=device).unsqueeze(1)
            self.static_obstacles = torch.cat([pos_tensor, rad_tensor], dim=1)
        else:
            self.static_obstacles = None
    
    def _generate_dynamic_obstacles(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """
        Generate dynamic obstacles with randomization for each batch.
        """
        if self.obstacles_config is None:
            return None
            
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Dynamic obstacle count
        dynamic_count = self.obstacles_config.get('dynamic_count', False)
        if dynamic_count:
            count_range = self.obstacles_config.get('count_range', [2, 8])
            min_count, max_count = count_range
            num_obstacles = np.random.randint(min_count, max_count + 1)
        else:
            num_obstacles = self.obstacles_config.get('random_count', 3)
        
        if num_obstacles == 0:
            return None
        
        min_radius = self.obstacles_config.get('random_min_radius', 0.08)
        max_radius = self.obstacles_config.get('random_max_radius', 0.5)
        
        # Generate obstacles for each batch element
        all_obstacles = []
        for b in range(batch_size):
            positions = []
            radii = []
            
            for _ in range(num_obstacles):
                max_attempts = 50
                for attempt in range(max_attempts):
                    margin = 0.1
                    pos = [
                        np.random.uniform(margin, self.area_size - margin),
                        np.random.uniform(margin, self.area_size - margin)
                    ]
                    radius = np.random.uniform(min_radius, max_radius)
                    
                    # Check distance from existing obstacles
                    valid_position = True
                    for existing_pos, existing_rad in zip(positions, radii):
                        dist = np.sqrt((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)
                        min_distance = radius + existing_rad + 0.1
                        if dist < min_distance:
                            valid_position = False
                            break
                    
                    if valid_position or attempt == max_attempts - 1:
                        positions.append(pos)
                        radii.append(radius)
                        break
            
            if positions:
                pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
                rad_tensor = torch.tensor(radii, dtype=torch.float32, device=device).unsqueeze(1)
                obstacles = torch.cat([pos_tensor, rad_tensor], dim=1)
                all_obstacles.append(obstacles)
        
        if all_obstacles:
            # Pad to same size
            max_obstacles = max(obs.shape[0] for obs in all_obstacles)
            padded_obstacles = []
            
            for obstacles in all_obstacles:
                if obstacles.shape[0] < max_obstacles:
                    padding_size = max_obstacles - obstacles.shape[0]
                    dummy_obstacles = torch.zeros(padding_size, 3, device=device)
                    dummy_obstacles[:, :2] = -100  # Far away
                    obstacles = torch.cat([obstacles, dummy_obstacles], dim=0)
                padded_obstacles.append(obstacles)
            
            return torch.stack(padded_obstacles, dim=0)
        
        return None

    def get_obstacle_tensor(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """Get obstacle tensor for the environment state."""
        if self.obstacles_config and self.obstacles_config.get('dynamic_count', False):
            return self._generate_dynamic_obstacles(batch_size)
        
        if self.static_obstacles is None:
            return None
        
        return self.static_obstacles.unsqueeze(0).expand(batch_size, -1, -1)
    
    def check_obstacle_collisions(self, state: SingleAgentState) -> torch.Tensor:
        """
        Check for collisions between the agent and obstacles.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean tensor [batch_size] indicating collision in each batch
        """
        if state.obstacles is None:
            return torch.zeros(state.batch_size, dtype=torch.bool, device=state.position.device)
        
        # Extract obstacle positions and radii
        obstacle_positions = state.obstacles[..., :-1]  # [batch_size, n_obstacles, pos_dim]
        obstacle_radii = state.obstacles[..., -1]       # [batch_size, n_obstacles]
        
        # Compute distances to all obstacles
        # state.position: [batch_size, pos_dim]
        # obstacle_positions: [batch_size, n_obstacles, pos_dim]
        diff = state.position.unsqueeze(1) - obstacle_positions  # [batch_size, n_obstacles, pos_dim]
        distances = torch.sqrt(torch.sum(diff * diff, dim=2) + 1e-8)  # [batch_size, n_obstacles]
        
        # Check collisions
        collision_thresholds = obstacle_radii + self.agent_radius
        collisions = distances < collision_thresholds
        
        # Any collision in each batch
        any_collision = torch.any(collisions, dim=1)
        
        return any_collision
    
    def get_goal_distance(self, state: SingleAgentState) -> torch.Tensor:
        """
        Calculate distance to goal for the agent.
        
        Args:
            state: Current environment state
            
        Returns:
            Tensor [batch_size] with distance to goal
        """
        diff = state.position - state.goal
        distances = torch.sqrt(torch.sum(diff * diff, dim=1) + 1e-8)
        return distances
    
    def is_done(self, state: SingleAgentState) -> torch.Tensor:
        """
        Check if episodes are done.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean tensor [batch_size] indicating which episodes are done
        """
        # Episode done if: max steps, collision, or goal reached
        max_steps_reached = state.step_count >= self.max_steps
        collisions = self.check_obstacle_collisions(state)
        
        goal_dists = self.get_goal_distance(state)
        goal_reached = goal_dists < self.agent_radius
        
        return collisions | goal_reached | max_steps_reached
    
    @abstractmethod
    def dynamics(self, state: SingleAgentState, action: torch.Tensor) -> torch.Tensor:
        """
        Apply system dynamics to compute state derivatives.
        
        Args:
            state: Current state of the environment
            action: Actions to apply [batch_size, action_dim]
            
        Returns:
            State derivatives [batch_size, state_dim]
        """
        pass