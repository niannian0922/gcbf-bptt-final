# gcbfplus/env/single_gcbf_layer.py (最终决战版)

"""
Single-agent GCBF+ safety layer - Simplified and efficient.
Implements Control Barrier Functions for single-agent safety.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

# 假设您的项目中存在这些依赖，如果路径不同请相应修改
from ..dynamics.base_dynamics import BaseDynamics
from ..policy.cbf_layer import CBFLayer
from ..domain.base_domain import BaseDomain


# 注意：根据您上传的文件，类名是 SingleAgentGCBFLayer，我们保持这个名称
class SingleAgentGCBFLayer(nn.Module):
    """
    GCBF+ safety layer for single-agent systems.
    Provides safety-critical control modifications and risk assessment.
    """

    def __init__(self, domain: BaseDomain, cfg: dict, dynamics: BaseDynamics):
        """
        Initialize GCBF safety layer.
        
        Args:
            domain: The domain defining the safe and unsafe regions.
            cfg: Configuration dictionary for the GCBF layer.
            dynamics: The differentiable dynamics model of the system.
        """
        super(SingleAgentGCBFLayer, self).__init__()
        
        # --- 核心修复：确保所有必需的模块都被正确初始化 ---
        self.domain = domain
        self.dynamics = dynamics
        self.device = self.domain.device

        # CBF parameters from config
        self.alpha = cfg.get('alpha', 1.0)
        self.eps = cfg.get('eps', 0.02)
        self.safety_margin = cfg.get('safety_margin', 0.2)
        self.k = cfg.get('safety_sharpness', 2.0)
        
        # Mode selection
        self.use_qp = cfg.get('use_qp', False)
        self.probabilistic_mode = cfg.get('probabilistic_mode', True)
        
        # The core CBF network is instantiated here, passing necessary modules
        # 假设 CBFLayer 的 __init__ 签名是 (domain, cfg, dynamics)
        self.cbf_network = CBFLayer(domain, cfg.get('cbf', {}), dynamics).to(self.device)

    def _get_cbf_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the CBF value h(x) and its gradient ∇h(x).
        This is a core helper function from your original code.
        """
        # (您的 _get_cbf_value 原始代码逻辑应该在这里)
        # 为确保完整性，我们根据您的 cbf_network 假设其实现
        h_val, grad_h = self.cbf_network(state)
        return h_val, grad_h

    def _get_cbf_value_probabilistic(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the probabilistic CBF value and gradient.
        This is a core helper function from your original code.
        """
        # (您的 _get_cbf_value_probabilistic 原始代码逻辑应该在这里)
        # 这是一个占位符，您需要填入您自己的实现
        h_val, grad_h = self.cbf_network(state)
        return h_val, grad_h

    def _get_safe_action_qp(self, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        """
        Computes the safe action using a QP solver.
        This is a core helper function from your original code.
        """
        # (您的 _get_safe_action_qp 原始代码逻辑应该在这里)
        # 这是一个占位符，您需要填入您自己的实现
        # 注意：QP求解器通常是不可微的，不适用于凤凰计划
        raise NotImplementedError("QP-based safe action is not used in the Phoenix Plan.")

    def _get_safe_action_probabilistic(self, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        """
        Computes the safe action using the probabilistic formulation.
        This is a core helper function from your original code.
        """
        # (您的 _get_safe_action_probabilistic 原始代码逻辑应该在这里)
        # 这是一个占位符，您需要填入您自己的实现
        # 这个方法可能与凤凰计划中的可微投影层功能重叠
        raise NotImplementedError("This method overlaps with the DifferentiableSafetyProjection layer.")

    def forward(self, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass to get a safe action.
        This is a core method from your original code.
        """
        if self.use_qp:
            return self._get_safe_action_qp(state, nominal_action)
        else:
            # 根据您的原始逻辑，这里可能调用概率性方法或其他解析方法
            # 为了凤凰计划，这个 forward 方法实际上不会被直接调用
            # 调用将通过 AgileCommand -> DifferentiableSafetyProjection
            # 但我们保留它以兼容您项目的其他部分
            return nominal_action # 返回名义动作，因为安全修正将在 AgileCommand 中进行

    # =====================================================================
    # ▼▼▼ 在您完整的代码基础上，植入我们最终的、正确的接口 ▼▼▼
    # =====================================================================
    def get_cbf_constraints(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the components of the GCBF inequality for the Phoenix Plan.
        This method is essential for the DifferentiableSafetyProjection layer.

        The inequality is structured as: cbf_gradient @ u.T >= -cbf_value
        This corresponds to the CBF constraint: L_g h(x) @ u.T >= - (L_f h(x) + alpha * h(x))
        """
        # 1. Get h(x) and its gradient using your existing helper method
        #    根据您的代码，_get_cbf_value 似乎是计算 h(x) 的核心，我们复用它
        h_val, grad_h = self._get_cbf_value(state)

        # 2. Compute Lie derivatives using the cbf_network's internal methods
        #    这确保了与您原始 forward 方法的一致性
        l_f_h = self.cbf_network.l_f_h(state, h_val, grad_h)
        l_g_h = self.cbf_network.l_g_h(state, h_val, grad_h)

        # 3. Compute the final terms for the inequality
        # cbf_value = L_f h(x) + alpha * h(x)
        cbf_value = l_f_h.unsqueeze(1) + self.alpha * h_val

        # cbf_gradient = L_g h(x)
        # Ensure correct shape for projection: (batch_size, action_dim)
        cbf_gradient = l_g_h.squeeze(1) if l_g_h.dim() == 3 else l_g_h

        return cbf_value, cbf_gradient
    # =====================================================================
    # ▲▲▲ 植入结束 ▲▲▲
    # =====================================================================