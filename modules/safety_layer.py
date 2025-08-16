import torch

class DifferentiableSafetyProjection(torch.autograd.Function):
    """
    可微分安全投影层：将违反GCBF约束的动作投影到安全边界，且梯度端到端可微。
    """

    @staticmethod
    def forward(ctx, nominal_action, cbf_value, cbf_gradient):
        """
        Args:
            nominal_action: [batch, action_dim]，原始动作
            cbf_value: [batch, 1]，GCBF值 h(x)
            cbf_gradient: [batch, action_dim]，L_g h(x)（对动作的梯度）

        Returns:
            corrected_action: [batch, action_dim]，投影后的安全动作
        """
        # 投影公式: u_corr = u_nom - (h(x) / ||g||^2) * g
        # 其中g = cbf_gradient, h(x) = cbf_value
        g = cbf_gradient
        h = cbf_value
        g_norm_sq = (g ** 2).sum(dim=1, keepdim=True) + 1e-8  # 防止除零
        correction = (h / g_norm_sq) * g
        corrected_action = nominal_action - correction

        # 保存中间量用于反向传播
        ctx.save_for_backward(g)
        return corrected_action

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：投影的雅可比矩阵 P = I - (g g^T) / ||g||^2
        """
        g, = ctx.saved_tensors
        g_norm_sq = (g ** 2).sum(dim=1, keepdim=True) + 1e-8
        # 计算投影矩阵P
        # grad_output: [batch, action_dim]
        # P: [batch, action_dim, action_dim]
        batch, action_dim = g.shape
        P = torch.eye(action_dim, device=g.device).unsqueeze(0).repeat(batch, 1, 1) \
            - (g.unsqueeze(2) @ g.unsqueeze(1)) / g_norm_sq.unsqueeze(2)
        # 乘以外部梯度
        grad_input = torch.bmm(grad_output.unsqueeze(1), P).squeeze(1)
        # 只对nominal_action传递梯度，其余输入不需要梯度
        return grad_input, None, None
