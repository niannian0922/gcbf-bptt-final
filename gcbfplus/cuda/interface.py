"""
CUDA Differentiable Dynamics Interface with CPU fallback.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import dynamics_cuda  # Built CUDA extension (optional at runtime)
    _CUDA_AVAILABLE = True
except Exception:
    dynamics_cuda = None
    _CUDA_AVAILABLE = False


class DifferentiableDynamics(torch.autograd.Function):
    """
    Single-step differentiable dynamics with identical CPU/CUDA interface.

    Inputs:
      - current_state: [batch, state_dim]
      - control_action: [batch, action_dim]
      - A: [state_dim, state_dim]
      - B: [state_dim, action_dim]
    Output:
      - next_state: [batch, state_dim]
    """

    @staticmethod
    def forward(ctx, current_state: Tensor, control_action: Tensor, *extra_inputs: Tensor) -> Tensor:
        if len(extra_inputs) < 2:
            raise ValueError("DifferentiableDynamics.forward expects A and B matrices as extra inputs")
        A, B = extra_inputs[:2]
        ctx.num_extras = len(extra_inputs)

        if current_state.dim() != 2 or control_action.dim() != 2:
            raise ValueError("current_state and control_action must be 2D: [batch, dim]")
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("A and B must be 2D matrices")

        # Save for backward (A and B considered constants for gradients)
        ctx.save_for_backward(A, B)

        if _CUDA_AVAILABLE and current_state.is_cuda and control_action.is_cuda and A.is_cuda and B.is_cuda:
            return dynamics_cuda.forward(current_state, control_action, A, B)
        # CPU reference
        return current_state.matmul(A.t()) + control_action.matmul(B.t())

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        A, B = ctx.saved_tensors
        if grad_output.dim() != 2:
            raise ValueError("grad_output must be 2D: [batch, state_dim]")

        if _CUDA_AVAILABLE and A.is_cuda and B.is_cuda and grad_output.is_cuda:
            grad_state, grad_action = dynamics_cuda.backward(grad_output, A, B)
        else:
            grad_state = grad_output.matmul(A)
            grad_action = grad_output.matmul(B)

        extra_nones = tuple(None for _ in range(getattr(ctx, 'num_extras', 0)))
        return (grad_state, grad_action) + extra_nones


