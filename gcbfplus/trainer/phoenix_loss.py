import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

def compute_phoenix_loss(
    progress_reward: torch.Tensor,
    nominal_action: torch.Tensor,
    safe_action: torch.Tensor,
    jerk: torch.Tensor,
    loss_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Phoenix Plan损失函数：安全由架构保证，损失专注于效率与可学习性。

    Args:
        progress_reward (Tensor): 当前步的进步奖励（越大越好）。
        nominal_action (Tensor): 策略网络输出的原始动作。
        safe_action (Tensor): 经过可微分安全投影后的最终安全动作。
        jerk (Tensor): 当前步的jerk（加加速度）指标。
        loss_weights (dict): 各项损失的权重，包含'"eff_weight"', '"correction_weight"', '"jerk_weight"'。

    Returns:
        total_loss (Tensor): 加权后的总损失。
        loss_dict (dict): 各项损失分量，便于监控与分析。
    """
    # 1. 效率损失：鼓励进步，等于进步奖励的负值
    efficiency_loss = -progress_reward.mean()

    # 2. 修正惩罚：鼓励名义动作本身就安全，MSE损失
    correction_penalty = F.mse_loss(nominal_action, safe_action)

    # 3. 平滑度损失：鼓励轨迹平滑
    jerk_loss = jerk.mean()

    # 权重
    eff_weight = loss_weights.get('eff_weight', 1.0)
    correction_weight = loss_weights.get('correction_weight', 1.0)
    jerk_weight = loss_weights.get('jerk_weight', 1.0)

    # 总损失
    total_loss = (
        eff_weight * efficiency_loss
        + correction_weight * correction_penalty
        + jerk_weight * jerk_loss
    )

    loss_dict = {
        'total_loss': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
        'efficiency_loss': efficiency_loss.item() if hasattr(efficiency_loss, 'item') else float(efficiency_loss),
        'correction_penalty': correction_penalty.item() if hasattr(correction_penalty, 'item') else float(correction_penalty),
        'jerk_loss': jerk_loss.item() if hasattr(jerk_loss, 'item') else float(jerk_loss),
    }
    return total_loss, loss_dict
