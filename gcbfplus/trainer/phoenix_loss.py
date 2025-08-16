# gcbfplus/trainer/phoenix_loss.py (最终版)

import torch

def compute_phoenix_loss(
    nominal_action: torch.Tensor,
    safe_action: torch.Tensor,
    rewards: dict,
    cfg: dict
) -> tuple[torch.Tensor, dict]:
    """
    Computes the loss for the Phoenix model.
    Safety is guaranteed by the architecture, so the loss focuses on efficiency and learning.
    """
    loss_weights = cfg.get('losses', {})
    eff_weight = loss_weights.get('eff_weight', 1.0)
    correction_weight = loss_weights.get('correction_weight', 0.1)
    jerk_weight = loss_weights.get('jerk_weight', 0.05)

    # 1. Efficiency Loss: Encourages the agent to make progress.
    # We want to MAXIMIZE progress_reward, so the loss is its negative.
    progress_reward = rewards.get('progress_reward', torch.tensor(0.0, device=nominal_action.device))
    efficiency_loss = -torch.mean(progress_reward)

    # 2. Correction Penalty: Penalizes the policy for proposing unsafe actions.
    # This is the core of learning to be inherently safe.
    correction_penalty = torch.mean((safe_action - nominal_action) ** 2)

    # 3. Jerk Loss: Encourages smooth trajectories.
    jerk = rewards.get('jerk', torch.tensor(0.0, device=nominal_action.device))
    jerk_loss = torch.mean(jerk)

    # 4. Total Loss: Weighted sum of all components.
    total_loss = (
        eff_weight * efficiency_loss +
        correction_weight * correction_penalty +
        jerk_weight * jerk_loss
    )

    loss_dict = {
        'total_loss': total_loss.item(),
        'efficiency_loss': efficiency_loss.item(),
        'correction_penalty': correction_penalty.item(),
        'jerk_loss': jerk_loss.item(),
    }

    return total_loss, loss_dict