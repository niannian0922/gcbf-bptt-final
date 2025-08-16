# gcbfplus/env/single_gcbf_layer.py (决战修正版)

"""
Single-agent GCBF+ safety layer - Simplified and efficient.
Implements Control Barrier Functions for single-agent safety.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

# =====================================================================
# ▼▼▼ 核心修复：将所有相对导入 ".." 替换为绝对导入 "gcbfplus." ▼▼▼
# =====================================================================
from gcbfplus.dynamics.base_dynamics import BaseDynamics
from gcbfplus.policy.cbf_layer import CBFLayer
from gcbfplus.domain.base_domain import BaseDomain
# =====================================================================
# ▲▲▲ 修复结束 ▲▲▲
# =====================================================================


class SingleAgentGCBFLayer(nn.Module):
    """
    GCBF+ safety layer for single-agent systems.
    Provides safety-critical control modifications and risk assessment.
    """

    def __init__(self, domain: BaseDomain, cfg: dict, dynamics: BaseDynamics):
        """
        Initialize GCBF safety layer.
        """
        super(SingleAgentGCBFLayer, self).__init__()
        
        self.domain = domain
        self.dynamics = dynamics
        self.device = self.domain.device

        self.alpha = cfg.get('alpha', 1.0)
        self.eps = cfg.get('eps', 0.02)
        self.safety_margin = cfg.get('safety_margin', 0.2)
        self.k = cfg.get('safety_sharpness', 2.0)
        
        self.use_qp = cfg.get('use_qp', False)
        self.probabilistic_mode = cfg.get('probabilistic_mode', True)
        
        self.cbf_network = CBFLayer(domain, cfg.get('cbf', {}), dynamics).to(self.device)

    def _get_cbf_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the CBF value h(x) and its gradient ∇h(x).
        """
        h_val, grad_h = self.cbf_network(state)
        return h_val, grad_h

    def _get_cbf_value_probabilistic(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the probabilistic CBF value and gradient.
        """
        h_val, grad_h = self.cbf_network(state)
        return h_val, grad_h

    def _get_safe_action_qp(self, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        """
        Computes the safe action using a QP solver.
        """
        raise NotImplementedError("QP-based safe action is not used in the Phoenix Plan.")

    def _get_safe_action_probabilistic(self, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        """
        Computes the safe action using the probabilistic formulation.
        """
        raise NotImplementedError("This method overlaps with the DifferentiableSafetyProjection layer.")

    def forward(self, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass to get a safe action.
        """
        if self.use_qp:
            return self._get_safe_action_qp(state, nominal_action)
        else:
            return nominal_action

    def get_cbf_constraints(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the components of the GCBF inequality for the Phoenix Plan.
        """
        h_val, grad_h = self._get_cbf_value(state)
        l_f_h = self.cbf_network.l_f_h(state, h_val, grad_h)
        l_g_h = self.cbf_network.l_g_h(state, h_val, grad_h)
        cbf_value = l_f_h.unsqueeze(1) + self.alpha * h_val
        cbf_gradient = l_g_h.squeeze(1) if l_g_h.dim() == 3 else l_g_h
        return cbf_value, cbf_gradient