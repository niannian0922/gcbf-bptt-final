import torch
import torch.nn as nn
from modules.safety_layer import DifferentiableSafetyProjection

class AgileCommand(nn.Module):
    """
    Integrates the policy network with a differentiable safety projection layer
    to ensure safe and efficient control.
    """
    def __init__(self, policy_net: nn.Module, gcbf_module: nn.Module):
        """
        Args:
            policy_net: A network that proposes a nominal action given a state.
            gcbf_module: The GCBF module that provides safety constraints
                         via the get_cbf_constraints(state) interface.
        """
        super().__init__()
        self.policy_net = policy_net
        self.gcbf = gcbf_module

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a nominal action and projects it into the safe set.

        Args:
            state: The current environment state.

        Returns:
            (safe_action, nominal_action):
                safe_action: The final action after safety projection.
                nominal_action: The raw action proposed by the policy network.
        """
        # Step 1: The policy network proposes a nominal, potentially unsafe, action.
        nominal_action = self.policy_net(state)

        # Step 2: The GCBF module computes the safety constraint boundaries for the current state.
        cbf_value, cbf_gradient = self.gcbf.get_cbf_constraints(state)

        # Step 3: The differentiable safety layer projects the action onto the safe set.
        safe_action = DifferentiableSafetyProjection.apply(nominal_action, cbf_value, cbf_gradient)

        return safe_action, nominal_action
