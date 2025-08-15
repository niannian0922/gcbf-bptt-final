# gcbfplus/policy/guardian_network.py
import torch
import torch.nn as nn
from typing import Dict

class GuardianNetwork(nn.Module):
    """A network dedicated to predicting the Control Barrier Function value 'h'."""
    
    def __init__(self, config: Dict):
        super().__init__()
        input_dim = config.get('input_dim', 9)
        hidden_dims = config.get('hidden_dims', [128, 128])
        output_dim = config.get('output_dim', 1)
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Guardian Network.
        
        Args:
            observation: Input observation tensor [batch_size, input_dim]
            
        Returns:
            h: Predicted Control Barrier Function value [batch_size, 1]
               We add a ReLU activation at the end to ensure the predicted 'h' 
               is always non-negative, which is consistent with its definition 
               as a distance-based value.
        """
        return torch.relu(self.net(observation))
