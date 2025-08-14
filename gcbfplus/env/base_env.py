import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass


@dataclass
class EnvState:
    """用于存储环境状态信息的基础数据类。"""
    pass


@dataclass
class StepResult:
    """环境步骤结果的容器。"""
    next_state: EnvState
    reward: torch.Tensor
    cost: torch.Tensor  # For safety constraint violations
    done: torch.Tensor
    info: Dict[str, Any]


class BaseEnv(nn.Module, ABC):
    """
    Abstract base class for differentiable environments.
    
    This class defines the core interface for environments that can be used
    for end-to-end training with backpropagation through time (BPTT).
    All environments must inherit from this class and implement its abstract methods.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the environment.
        
        Args:
            config: Dictionary containing environment configuration parameters.
        """
        super(BaseEnv, self).__init__()
        self.config = config
    
    @property
    def device(self) -> torch.device:
        """获取环境张量存储的设备。"""
        return next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
    
    @abstractmethod
    def reset(self, batch_size: int = 1, randomize: bool = True) -> EnvState:
        """
        Reset the environment to an initial state.
        
        Args:
            batch_size: Number of parallel environments to reset.
            randomize: Whether to randomize the initial state.
            
        Returns:
            The initial state of the environment.
        """
        pass
    
    @abstractmethod
    def step(self, state: EnvState, action: torch.Tensor) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            state: Current state of the environment.
            action: Action to take.
            
        Returns:
            StepResult containing next_state, reward, cost, done, and info.
        """
        pass
    
    @abstractmethod
    def get_observation(self, state: EnvState) -> torch.Tensor:
        """
        Extract observation from environment state.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            Observation tensor.
        """
        pass
    
    @abstractmethod
    def render(self, state: EnvState) -> Any:
        """
        Render the environment state.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            Rendering of the environment (e.g., image tensor, matplotlib figure).
        """
        pass
    
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the observation tensor.
        
        Returns:
            Tuple of integers representing the observation shape.
        """
        pass
    
    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the action tensor.
        
        Returns:
            Tuple of integers representing the action shape.
        """
        pass
    
    @abstractmethod
    def get_action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the bounds of valid actions.
        
        Returns:
            Tuple of (lower_bound, upper_bound) tensors.
        """
        pass
    
    def apply_safety_layer(self, state: EnvState, raw_action: torch.Tensor) -> torch.Tensor:
        """
        Apply safety filtering to raw actions. Default implementation returns raw actions.
        Override this method to implement safety constraints (e.g., GCBF).
        
        Args:
            state: Current state of the environment.
            raw_action: Raw action from policy.
            
        Returns:
            Safe action.
        """
        return raw_action
    
    def get_goal_distance(self, state: EnvState) -> torch.Tensor:
        """
        Calculate distance to goal for reward computation. Default returns zeros.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            Tensor containing distances to goals.
        """
        return torch.zeros(state.batch_size, device=self.device)
    
    def check_collision(self, state: EnvState) -> torch.Tensor:
        """
        Check for collisions in the current state. Default returns zeros.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            Boolean tensor indicating collisions.
        """
        return torch.zeros(state.batch_size, dtype=torch.bool, device=self.device)
    
    def compute_reward(self, state: EnvState, action: torch.Tensor, next_state: EnvState) -> torch.Tensor:
        """
        Compute reward based on state transition. Default uses goal distance.
        
        Args:
            state: Current state.
            action: Action taken.
            next_state: Resulting next state.
            
        Returns:
            Reward tensor.
        """
        # Default reward is negative distance to goal
        return -self.get_goal_distance(next_state)
    
    def compute_cost(self, state: EnvState, action: torch.Tensor, next_state: EnvState) -> torch.Tensor:
        """
        Compute safety constraint violation cost. Default uses collision detection.
        
        Args:
            state: Current state.
            action: Action taken.
            next_state: Resulting next state.
            
        Returns:
            Cost tensor (0 for safe, >0 for unsafe).
        """
        # Default cost is 1.0 for collisions, 0.0 otherwise
        return self.check_collision(next_state).float() 