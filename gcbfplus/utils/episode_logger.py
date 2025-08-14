"""
Episode Data Logger for Multi-Agent BPTT Training

This module provides comprehensive data logging capabilities for multi-agent episodes,
enabling offline analysis and visualization of agent behavior, safety metrics, and
collaborative dynamics.
"""

import torch
import numpy as np
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EpisodeData:
    """Container for all data collected during a single episode."""
    
    # Time series data - each list contains data for each timestep
    positions: List[np.ndarray] = field(default_factory=list)      # [timestep][batch, n_agents, 3]
    velocities: List[np.ndarray] = field(default_factory=list)     # [timestep][batch, n_agents, 3]
    actions: List[np.ndarray] = field(default_factory=list)        # [timestep][batch, n_agents, action_dim]
    raw_actions: List[np.ndarray] = field(default_factory=list)    # [timestep][batch, n_agents, action_dim]
    alpha_values: List[np.ndarray] = field(default_factory=list)   # [timestep][batch, n_agents, 1]
    h_values: List[np.ndarray] = field(default_factory=list)       # [timestep][batch, n_agents, 1] (CBF values)
    min_distances: List[np.ndarray] = field(default_factory=list)  # [timestep][batch, n_agents]
    goal_distances: List[np.ndarray] = field(default_factory=list) # [timestep][batch, n_agents]
    rewards: List[np.ndarray] = field(default_factory=list)        # [timestep][batch, n_agents]
    costs: List[np.ndarray] = field(default_factory=list)          # [timestep][batch, n_agents]
    
    # Episode metadata
    episode_id: str = ""
    final_status: str = "UNKNOWN"  # "SUCCESS", "COLLISION", "TIMEOUT"
    total_steps: int = 0
    batch_size: int = 1
    n_agents: int = 0
    
    # Environment parameters for reference
    obstacles: Optional[np.ndarray] = None  # [n_obstacles, pos_dim+1] (positions + radii)
    goals: Optional[np.ndarray] = None      # [batch, n_agents, pos_dim]
    safety_radius: float = 0.0
    area_size: float = 0.0


class EpisodeLogger:
    """
    Data logger for recording all critical information during episode execution.
    
    This logger captures:
    - Agent positions and velocities over time
    - Actions (both raw policy output and safety-filtered)
    - CBF values and alpha parameters
    - Safety distances and collision information
    - Rewards and costs
    - Episode outcome and metadata
    """
    
    def __init__(self, log_dir: str = "episode_logs", prefix: str = "episode"):
        """
        Initialize the episode logger.
        
        Args:
            log_dir: Directory to save episode logs
            prefix: Prefix for log filenames
        """
        self.log_dir = log_dir
        self.prefix = prefix
        self.current_episode: Optional[EpisodeData] = None
        self.episode_counter = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def start_episode(self, episode_id: Optional[str] = None, 
                     batch_size: int = 1, n_agents: int = 2,
                     obstacles: Optional[torch.Tensor] = None,
                     goals: Optional[torch.Tensor] = None,
                     safety_radius: float = 0.2,
                     area_size: float = 2.0) -> str:
        """
        Start logging a new episode.
        
        Args:
            episode_id: Unique identifier for this episode
            batch_size: Number of parallel environments
            n_agents: Number of agents per environment
            obstacles: Obstacle positions and radii
            goals: Goal positions for each agent
            safety_radius: Safety radius for collision detection
            area_size: Size of the environment area
            
        Returns:
            episode_id: The episode identifier being used
        """
        if episode_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_id = f"{self.prefix}_{self.episode_counter:03d}_{timestamp}"
        
        self.episode_counter += 1
        
        # Initialize new episode data
        self.current_episode = EpisodeData(
            episode_id=episode_id,
            batch_size=batch_size,
            n_agents=n_agents,
            safety_radius=safety_radius,
            area_size=area_size
        )
        
        # Store environment reference data
        if obstacles is not None:
            self.current_episode.obstacles = self._to_numpy(obstacles)
        if goals is not None:
            self.current_episode.goals = self._to_numpy(goals)
        
        print(f"📊 Started logging episode: {episode_id}")
        return episode_id
    
    def log_step(self, 
                 positions: torch.Tensor,
                 velocities: torch.Tensor,
                 actions: torch.Tensor,
                 raw_actions: Optional[torch.Tensor] = None,
                 alpha_values: Optional[torch.Tensor] = None,
                 h_values: Optional[torch.Tensor] = None,
                 min_distances: Optional[torch.Tensor] = None,
                 goal_distances: Optional[torch.Tensor] = None,
                 rewards: Optional[torch.Tensor] = None,
                 costs: Optional[torch.Tensor] = None) -> None:
        """
        Log data for a single timestep.
        
        Args:
            positions: Agent positions [batch, n_agents, pos_dim]
            velocities: Agent velocities [batch, n_agents, vel_dim]
            actions: Safety-filtered actions [batch, n_agents, action_dim]
            raw_actions: Raw policy actions [batch, n_agents, action_dim]
            alpha_values: CBF alpha parameters [batch, n_agents, 1]
            h_values: CBF barrier function values [batch, n_agents, 1]
            min_distances: Minimum distances to obstacles/agents [batch, n_agents]
            goal_distances: Distances to goals [batch, n_agents]
            rewards: Step rewards [batch, n_agents]
            costs: Step costs (constraint violations) [batch, n_agents]
        """
        if self.current_episode is None:
            raise RuntimeError("Must call start_episode() before logging steps")
        
        # Convert all tensors to numpy and store
        self.current_episode.positions.append(self._to_numpy(positions))
        self.current_episode.velocities.append(self._to_numpy(velocities))
        self.current_episode.actions.append(self._to_numpy(actions))
        
        if raw_actions is not None:
            self.current_episode.raw_actions.append(self._to_numpy(raw_actions))
        else:
            self.current_episode.raw_actions.append(self._to_numpy(actions))  # Fallback
        
        if alpha_values is not None:
            self.current_episode.alpha_values.append(self._to_numpy(alpha_values))
        
        if h_values is not None:
            self.current_episode.h_values.append(self._to_numpy(h_values))
        
        if min_distances is not None:
            self.current_episode.min_distances.append(self._to_numpy(min_distances))
        
        if goal_distances is not None:
            self.current_episode.goal_distances.append(self._to_numpy(goal_distances))
        
        if rewards is not None:
            self.current_episode.rewards.append(self._to_numpy(rewards))
        
        if costs is not None:
            self.current_episode.costs.append(self._to_numpy(costs))
        
        self.current_episode.total_steps += 1
    
    def end_episode(self, final_status: str) -> str:
        """
        End the current episode and save data to file.
        
        Args:
            final_status: Final episode outcome ("SUCCESS", "COLLISION", "TIMEOUT")
            
        Returns:
            filename: Path to the saved data file
        """
        if self.current_episode is None:
            raise RuntimeError("No active episode to end")
        
        self.current_episode.final_status = final_status
        
        # Save to compressed numpy file
        filename = os.path.join(self.log_dir, f"{self.current_episode.episode_id}.npz")
        
        # Prepare data dictionary
        save_data = {
            'episode_id': self.current_episode.episode_id,
            'final_status': final_status,
            'total_steps': self.current_episode.total_steps,
            'batch_size': self.current_episode.batch_size,
            'n_agents': self.current_episode.n_agents,
            'safety_radius': self.current_episode.safety_radius,
            'area_size': self.current_episode.area_size,
        }
        
        # Add time series data (convert lists to arrays)
        if self.current_episode.positions:
            save_data['positions'] = np.array(self.current_episode.positions)
        if self.current_episode.velocities:
            save_data['velocities'] = np.array(self.current_episode.velocities)
        if self.current_episode.actions:
            save_data['actions'] = np.array(self.current_episode.actions)
        if self.current_episode.raw_actions:
            save_data['raw_actions'] = np.array(self.current_episode.raw_actions)
        if self.current_episode.alpha_values:
            save_data['alpha_values'] = np.array(self.current_episode.alpha_values)
        if self.current_episode.h_values:
            save_data['h_values'] = np.array(self.current_episode.h_values)
        if self.current_episode.min_distances:
            save_data['min_distances'] = np.array(self.current_episode.min_distances)
        if self.current_episode.goal_distances:
            save_data['goal_distances'] = np.array(self.current_episode.goal_distances)
        if self.current_episode.rewards:
            save_data['rewards'] = np.array(self.current_episode.rewards)
        if self.current_episode.costs:
            save_data['costs'] = np.array(self.current_episode.costs)
        
        # Add environment reference data
        if self.current_episode.obstacles is not None:
            save_data['obstacles'] = self.current_episode.obstacles
        if self.current_episode.goals is not None:
            save_data['goals'] = self.current_episode.goals
        
        # Save to file
        np.savez_compressed(filename, **save_data)
        
        print(f"💾 Episode data saved: {filename}")
        print(f"   Status: {final_status}, Steps: {self.current_episode.total_steps}")
        
        # Reset current episode
        self.current_episode = None
        
        return filename
    
    @staticmethod
    def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array, handling device and gradient detachment."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)
    
    def get_latest_episode_path(self) -> Optional[str]:
        """Get the path to the most recently saved episode."""
        if self.episode_counter == 0:
            return None
        
        # Find the most recent file in the log directory
        files = [f for f in os.listdir(self.log_dir) if f.startswith(self.prefix) and f.endswith('.npz')]
        if not files:
            return None
        
        files.sort()
        return os.path.join(self.log_dir, files[-1])


# Utility functions for common distance calculations
def compute_min_distances_to_obstacles(positions: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """
    Compute minimum distances from agents to obstacles.
    
    Args:
        positions: Agent positions [batch, n_agents, pos_dim]
        obstacles: Obstacle data [batch, n_obstacles, pos_dim+1] (NEW FORMAT: positions + radii)
        
    Returns:
        min_distances: Minimum distances [batch, n_agents]
    """
    if obstacles is None or len(obstacles) == 0:
        return np.full(positions.shape[:2], float('inf'))
    
    batch_size, n_agents, pos_dim = positions.shape
    
    # 🔧 MODERNIZATION FIX: Handle new obstacle format [batch, n_obstacles, 3]
    if len(obstacles.shape) == 3:
        # New format: [batch_size, n_obstacles, pos_dim+1]
        n_obstacles = obstacles.shape[1]
        
        min_distances = np.full((batch_size, n_agents), float('inf'))
        
        for b in range(batch_size):
            # Extract obstacles for this batch element
            batch_obstacles = obstacles[b]  # [n_obstacles, 3]
            
            # Filter out dummy obstacles (those with zero radius or far-away position)
            valid_mask = (batch_obstacles[:, 2] > 0) & (np.abs(batch_obstacles[:, 0]) < 50) & (np.abs(batch_obstacles[:, 1]) < 50)
            
            if not np.any(valid_mask):
                # No valid obstacles in this batch
                min_distances[b, :] = float('inf')
                continue
                
            valid_obstacles = batch_obstacles[valid_mask]
            obstacle_positions = valid_obstacles[:, :pos_dim]  # [n_valid_obstacles, pos_dim]
            obstacle_radii = valid_obstacles[:, pos_dim]       # [n_valid_obstacles]
            
            for a in range(n_agents):
                agent_pos = positions[b, a]  # [pos_dim]
                
                # Compute distances to all valid obstacles
                distances = np.linalg.norm(obstacle_positions - agent_pos, axis=1) - obstacle_radii
                min_distances[b, a] = np.min(distances)
        
        return min_distances
    
    else:
        # Legacy format: [n_obstacles, pos_dim+1] - maintain backward compatibility
        n_obstacles = obstacles.shape[0]
        
        # Extract obstacle positions and radii
        obstacle_positions = obstacles[:, :pos_dim]  # [n_obstacles, pos_dim]
        obstacle_radii = obstacles[:, pos_dim]       # [n_obstacles]
        
        min_distances = np.full((batch_size, n_agents), float('inf'))
        
        for b in range(batch_size):
            for a in range(n_agents):
                agent_pos = positions[b, a]  # [pos_dim]
                
                # Compute distances to all obstacles
                distances = np.linalg.norm(obstacle_positions - agent_pos, axis=1) - obstacle_radii
                min_distances[b, a] = np.min(distances)
        
        return min_distances


def compute_goal_distances(positions: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """
    Compute distances from agents to their goals.
    
    Args:
        positions: Agent positions [batch, n_agents, pos_dim]
        goals: Goal positions [batch, n_agents, pos_dim]
        
    Returns:
        goal_distances: Distances to goals [batch, n_agents]
    """
    return np.linalg.norm(positions - goals, axis=-1)
