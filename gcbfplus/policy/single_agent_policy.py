"""
Single-agent policy network - Simplified and efficient.
Refactored for single-agent scenarios with adaptive safety features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class SingleAgentPerceptionModule(nn.Module):
    """
    Perception module for single-agent - processes sensor inputs.
    Supports both vector observations and vision-based inputs.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize perception module.
        
        Args:
            config: Configuration dictionary
        """
        super(SingleAgentPerceptionModule, self).__init__()
        
        self.vision_enabled = config.get('vision_enabled', False)
        hidden_dim = config.get('hidden_dim', 64)
        activation = config.get('activation', 'relu')
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        if self.vision_enabled:
            # Vision-based CNN processing
            input_channels = config.get('input_channels', 1)  # Depth image
            conv_channels = config.get('conv_channels', [32, 64, 128])
            kernel_sizes = config.get('kernel_sizes', [5, 3, 3])
            image_size = config.get('image_size', 64)
            
            # Build CNN layers
            cnn_layers = []
            in_channels = input_channels
            
            for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                           stride=2, padding=kernel_size//2))
                cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(self.activation)
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            
            # Calculate size after convolutions
            final_size = image_size // (2 ** len(conv_channels))
            cnn_output_size = conv_channels[-1] * final_size * final_size
            
            # Project to hidden dimension
            self.cnn_projection = nn.Sequential(
                nn.Linear(cnn_output_size, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.output_dim = hidden_dim
            
        else:
            # State-based MLP processing
            input_dim = config.get('input_dim', 6)  # Default: pos(2) + vel(2) + rel_goal(2)
            num_layers = config.get('num_layers', 2)
            use_batch_norm = config.get('use_batch_norm', False)
            
            # Build MLP layers
            layers = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            
            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.activation)
            
            self.mlp = nn.Sequential(*layers)
            self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through perception module.
        
        Args:
            x: Input tensor
               - Vision mode: [batch_size, channels, height, width]
               - State mode: [batch_size, input_dim]
            
        Returns:
            Processed features [batch_size, output_dim]
        """
        if self.vision_enabled:
            # Process vision input
            cnn_features = self.cnn(x)
            cnn_flat = cnn_features.view(cnn_features.size(0), -1)
            return self.cnn_projection(cnn_flat)
        else:
            # Process state input
            return self.mlp(x)


class SingleAgentPolicyHead(nn.Module):
    """
    Policy head for single-agent - outputs actions and optional safety parameters.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize policy head.
        
        Args:
            config: Configuration dictionary
        """
        super(SingleAgentPolicyHead, self).__init__()
        
        input_dim = config.get('input_dim', 128)
        output_dim = config.get('output_dim', 2)  # Action dimension
        hidden_dims = config.get('hidden_dims', [256])
        activation = config.get('activation', 'relu')
        
        # Select activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        self.action_head = nn.Linear(current_dim, output_dim)
        
        # Optional: predict safety confidence (alpha)
        self.predict_alpha = config.get('predict_alpha', False)
        if self.predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(current_dim, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        
        # Optional: predict adaptive safety margin
        self.predict_margin = config.get('predict_margin', False)
        if self.predict_margin:
            self.margin_head = nn.Sequential(
                nn.Linear(current_dim, 1),
                nn.Sigmoid()  # Will be scaled to [min_margin, max_margin]
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate actions and optional safety parameters.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - 'action': Action tensor [batch_size, action_dim]
                - 'alpha': (optional) Safety confidence [batch_size, 1]
                - 'margin': (optional) Safety margin [batch_size, 1]
        """
        features = self.mlp(x)
        
        outputs = {
            'action': torch.tanh(self.action_head(features))  # Bounded actions
        }
        
        if self.predict_alpha:
            outputs['alpha'] = self.alpha_head(features)
        
        if self.predict_margin:
            outputs['margin'] = self.margin_head(features)
        
        return outputs


class SingleAgentPolicy(nn.Module):
    """
    Complete single-agent policy network.
    Combines perception and policy head for end-to-end control.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize single-agent policy.
        
        Args:
            config: Configuration dictionary
        """
        super(SingleAgentPolicy, self).__init__()
        
        # Extract network configurations
        networks_config = config.get('networks', {})
        policy_config = networks_config.get('policy', {})
        
        # Initialize perception module
        perception_config = policy_config.get('perception', {})
        self.perception = SingleAgentPerceptionModule(perception_config)
        
        # Initialize policy head
        policy_head_config = policy_config.get('policy_head', {})
        policy_head_config['input_dim'] = self.perception.output_dim
        self.policy_head = SingleAgentPolicyHead(policy_head_config)
        
        # Store configuration
        self.config = config
        self.predict_alpha = policy_head_config.get('predict_alpha', False)
        self.predict_margin = policy_head_config.get('predict_margin', False)
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [batch_size, C, H, W]
            
        Returns:
            Dictionary with action and optional safety parameters
        """
        # Process observation through perception
        features = self.perception(obs)
        
        # Generate action and safety parameters
        outputs = self.policy_head(features)
        
        return outputs
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action only (convenience method).
        
        Args:
            obs: Observation tensor
            
        Returns:
            Action tensor [batch_size, action_dim]
        """
        outputs = self.forward(obs)
        return outputs['action']
    
    def get_safety_params(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get safety parameters if available.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Dictionary with available safety parameters
        """
        outputs = self.forward(obs)
        safety_params = {}
        
        if 'alpha' in outputs:
            safety_params['alpha'] = outputs['alpha']
        if 'margin' in outputs:
            safety_params['margin'] = outputs['margin']
        
        return safety_params