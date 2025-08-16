"""
Single-agent GCBF+ safety layer - Simplified and efficient.
Implements Control Barrier Functions for single-agent safety.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class SingleAgentGCBFLayer(nn.Module):
    """
    GCBF+ safety layer for single-agent systems.
    Provides safety-critical control modifications and risk assessment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GCBF safety layer.
        
        Args:
            config: Configuration dictionary containing:
                - alpha: CBF class-K function parameter
                - eps: Small epsilon for numerical stability
                - safety_margin: Base safety margin for obstacles
                - k: Safety sharpness parameter (for probabilistic mode)
                - use_qp: Whether to use QP solver (False for differentiable)
        """
        super(SingleAgentGCBFLayer, self).__init__()
        
        # CBF parameters
        self.alpha = config.get('alpha', 1.0)
        self.eps = config.get('eps', 0.02)
        self.safety_margin = config.get('safety_margin', 0.2)
        self.k = config.get('safety_sharpness', 2.0)  # For probabilistic safety
        
        # Mode selection
        self.use_qp = config.get('use_qp', False)
        self.probabilistic_mode = config.get('probabilistic_mode', True)
        
        # Neural network for learning CBF (optional)
        self.use_learned_cbf = config.get('use_learned_cbf', False)
        if self.use_learned_cbf:
            hidden_dim = config.get('hidden_dim', 64)
            input_dim = config.get('input_dim', 6)  # State dim + relative obstacle info
            
            self.cbf_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def compute_barrier_function(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        obstacles: Optional[torch.Tensor] = None,
        agent_radius: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute control barrier function value and gradients.
        
        Args:
            position: Agent position [batch_size, 2]
            velocity: Agent velocity [batch_size, 2]
            obstacles: Obstacle tensor [batch_size, n_obstacles, 3] (x, y, radius)
            agent_radius: Radius of the agent
            
        Returns:
            Tuple of (h_values, gradients):
                - h_values: CBF values [batch_size, n_obstacles]
                - gradients: CBF gradients [batch_size, n_obstacles, 4] (wrt state)
        """
        batch_size = position.shape[0]
        device = position.device
        
        if obstacles is None:
            # No obstacles - return large positive values (safe)
            h_values = torch.ones(batch_size, 1, device=device) * 1000.0
            gradients = torch.zeros(batch_size, 1, 4, device=device)
            return h_values, gradients
        
        # Extract obstacle positions and radii
        obs_positions = obstacles[..., :2]  # [batch_size, n_obs, 2]
        obs_radii = obstacles[..., 2]       # [batch_size, n_obs]
        
        # Compute relative positions
        rel_pos = position.unsqueeze(1) - obs_positions  # [batch_size, n_obs, 2]
        distances = torch.norm(rel_pos, dim=2)  # [batch_size, n_obs]
        
        # Barrier function: h = ||p - p_obs|| - (r_agent + r_obs + margin)
        min_dist = agent_radius + obs_radii + self.safety_margin
        h_values = distances - min_dist
        
        # Compute gradients
        # ∂h/∂p = (p - p_obs) / ||p - p_obs||
        # ∂h/∂v = 0 (for basic CBF)
        gradients = torch.zeros(batch_size, obstacles.shape[1], 4, device=device)
        
        # Avoid division by zero
        safe_distances = torch.maximum(distances, torch.tensor(1e-6, device=device))
        
        # Position gradients
        gradients[..., :2] = rel_pos / safe_distances.unsqueeze(-1)
        
        return h_values, gradients
    
    def compute_safe_action(
        self,
        state: Dict[str, torch.Tensor],
        nominal_action: torch.Tensor,
        dt: float = 0.05
    ) -> torch.Tensor:
        """
        Compute safe action using CBF constraints.
        
        Args:
            state: Dictionary containing 'position', 'velocity', 'obstacles'
            nominal_action: Proposed action [batch_size, 2]
            dt: Time step
            
        Returns:
            Safe action [batch_size, 2]
        """
        position = state['position']
        velocity = state['velocity']
        obstacles = state.get('obstacles', None)
        
        batch_size = position.shape[0]
        device = position.device
        
        # Compute barrier functions and gradients
        h_values, h_gradients = self.compute_barrier_function(
            position, velocity, obstacles
        )
        
        if obstacles is None or h_values.min() > 0.5:
            # Far from obstacles - return nominal action
            return nominal_action
        
        # For each obstacle, check CBF constraint
        safe_action = nominal_action.clone()
        
        for i in range(h_values.shape[1]):
            h = h_values[:, i]  # [batch_size]
            grad_h = h_gradients[:, i]  # [batch_size, 4]
            
            # Extract velocity gradients (for double integrator, action affects velocity)
            grad_h_v = grad_h[:, 2:4]  # [batch_size, 2]
            
            # CBF constraint: Lfh + Lgh * u >= -alpha * h
            # For double integrator: Lfh = grad_h_p @ v, Lgh = grad_h_v
            grad_h_p = grad_h[:, :2]
            Lfh = torch.sum(grad_h_p * velocity, dim=1)  # [batch_size]
            
            # Check constraint violation
            constraint_value = Lfh + torch.sum(grad_h_v * nominal_action, dim=1)
            min_value = -self.alpha * h
            
            violation = min_value - constraint_value  # Positive if violated
            violated = violation > 0
            
            if violated.any():
                # Project action to satisfy constraint
                # u_safe = u_nominal + lambda * grad_h_v
                # where lambda = max(0, violation / ||grad_h_v||^2)
                grad_norm_sq = torch.sum(grad_h_v ** 2, dim=1) + self.eps
                lambda_cbf = torch.maximum(
                    torch.zeros_like(violation),
                    violation / grad_norm_sq
                ).unsqueeze(1)
                
                # Apply correction only where violated
                correction = lambda_cbf * grad_h_v
                safe_action = torch.where(
                    violated.unsqueeze(1),
                    safe_action + correction,
                    safe_action
                )
        
        return safe_action
    
    def compute_safety_confidence(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute safety confidence score (alpha_safety) for probabilistic shield.
        
        Args:
            state: Dictionary containing 'position', 'velocity', 'obstacles'
            
        Returns:
            Safety confidence in [0, 1] where 1 is safe, 0 is dangerous [batch_size, 1]
        """
        position = state['position']
        velocity = state['velocity']
        obstacles = state.get('obstacles', None)
        
        batch_size = position.shape[0]
        device = position.device
        
        if obstacles is None:
            # No obstacles - fully safe
            return torch.ones(batch_size, 1, device=device)
        
        # Compute barrier function values
        h_values, _ = self.compute_barrier_function(position, velocity, obstacles)
        
        # Take minimum h across all obstacles (most dangerous)
        h_min = torch.min(h_values, dim=1)[0]  # [batch_size]
        
        # Convert to confidence using sigmoid-like function
        # alpha = sigmoid(k * h_min)
        # When h_min > 0 (safe), alpha -> 1
        # When h_min < 0 (unsafe), alpha -> 0
        alpha_safety = torch.sigmoid(self.k * h_min).unsqueeze(1)
        
        return alpha_safety
    
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        nominal_action: torch.Tensor,
        mode: str = 'probabilistic'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through safety layer.
        
        Args:
            state: Current state dictionary
            nominal_action: Proposed action from policy
            mode: 'probabilistic' or 'hard' safety mode
            
        Returns:
            Dictionary containing:
                - 'action': Final safe action
                - 'alpha_safety': Safety confidence (if probabilistic)
                - 'safe_action': Pure safe action from CBF
        """
        # Compute safe action using CBF
        safe_action = self.compute_safe_action(state, nominal_action)
        
        outputs = {'safe_action': safe_action}
        
        if mode == 'probabilistic':
            # Compute safety confidence
            alpha_safety = self.compute_safety_confidence(state)
            outputs['alpha_safety'] = alpha_safety
            
            # Blend actions: action = alpha * nominal + (1-alpha) * safe
            final_action = alpha_safety * nominal_action + (1 - alpha_safety) * safe_action
            outputs['action'] = final_action
        else:
            # Hard safety - always use safe action
            outputs['action'] = safe_action
            outputs['alpha_safety'] = torch.ones_like(nominal_action[:, :1])
        
        return outputs

    def get_cbf_constraints(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the components of the GCBF inequality for a given state.
        Inequality is: cbf_gradient @ u.T >= -cbf_value
        This corresponds to: L_g h(x) @ u.T >= - (L_f h(x) + alpha * h(x))

        Args:
            state (torch.Tensor): The current state of the agent(s).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cbf_value (torch.Tensor): The action-independent term, i.e., L_f h(x) + alpha * h(x).
                - cbf_gradient (torch.Tensor): The action-multiplying term, i.e., L_g h(x).
        """
        # Note: This is a skeleton. The actual implementation will depend on the specifics
        # of our learned GCBF model and the system dynamics. We need to compute the
        # Lie derivatives of the CBF function h(x).

        # Placeholder for h(x) and its derivatives
        h, grad_h = self.cbf_network(state) # Assuming cbf_network gives h and its gradient w.r.t. state

        # Placeholder for dynamics f(x) and g(x)
        # These should come from our differentiable physics model
        fx = self.dynamics_model.f(state)
        gx = self.dynamics_model.g(state)

        # L_f h(x) = grad_h * f(x)
        l_f_h = torch.sum(grad_h * fx, dim=1, keepdim=True)

        # L_g h(x) = grad_h * g(x)
        l_g_h = torch.sum(grad_h * gx, dim=1, keepdim=True) # This might need adjustment based on g(x) dimensions

        # cbf_value = L_f h(x) + alpha * h(x)
        cbf_value = l_f_h + self.alpha * h

        # cbf_gradient = L_g h(x)
        cbf_gradient = l_g_h

        return cbf_value, cbf_gradient