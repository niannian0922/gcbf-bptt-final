# Gradient decay utilities for temporal stability in BPTT

import torch
import torch.nn as nn
from typing import Any


class GDecay(torch.autograd.Function):
    """
    梯度衰减函数，用于BPTT中的梯度稳定。
    
    在前向传播中保持输入不变，在后向传播中应用衰减因子，
    这有助于防止在长时间范围内的梯度爆炸或消失。
    """
    
    @staticmethod
    def forward(ctx, input_tensor, decay_factor):
        """
        前向传播：简单地返回输入张量。
        
        参数:
            input_tensor: 输入张量
            decay_factor: 梯度衰减因子（0-1之间）
            
        返回:
            未修改的输入张量
        """
        # 为后向传播保存衰减因子
        ctx.save_for_backward(decay_factor)
        return input_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        后向传播：对梯度应用衰减因子。
        
        参数:
            grad_output: 来自上游的梯度
            
        返回:
            衰减后的梯度元组
        """
        decay_factor, = ctx.saved_tensors
        
        # 对梯度应用衰减
        decayed_grad = grad_output * decay_factor
        
        # 返回(input_tensor, decay_factor)的梯度
        # decay_factor不需要梯度，因此返回None
        return decayed_grad, None


def apply_gradient_decay(input_tensor: torch.Tensor, decay_factor: torch.Tensor) -> torch.Tensor:
    """
    应用梯度衰减到输入张量。
    
    这是GDecay.apply的便捷包装函数。
    
    参数:
        input_tensor: 要应用衰减的张量
        decay_factor: 衰减因子（0-1之间的标量或张量）
        
    返回:
        在前向传播中不变的张量，但在后向传播中梯度被衰减
        
    示例:
        >>> x = torch.randn(10, requires_grad=True)
        >>> decay_rate = 0.9
        >>> x_decayed = apply_gradient_decay(x, torch.tensor(decay_rate))
        >>> # x_decayed在前向传播中与x相同
        >>> # 但流向x的梯度将被乘以decay_rate
    """
    return GDecay.apply(input_tensor, decay_factor)


def temporal_gradient_decay(
    input_tensor: torch.Tensor, 
    time_step: int, 
    horizon: int, 
    decay_rate: float = 0.95,
    training: bool = True
) -> torch.Tensor:
    """
    根据时间步长应用时间相关的梯度衰减。
    
    在BPTT训练中，较早的时间步长获得更强的衰减，
    这有助于稳定长期依赖关系的学习。
    
    参数:
        input_tensor: 输入张量
        time_step: 当前时间步长（0为第一个）
        horizon: 总时间范围长度
        decay_rate: 基础衰减率
        training: 是否处于训练模式
        
    返回:
        应用了时间相关衰减的张量
    """
    if not training:
        return input_tensor
    
    # 计算时间缩放的衰减因子
    # 较早的时间步长（较小的time_step）获得更强的衰减
    time_factor = (horizon - time_step) / horizon
    effective_decay = decay_rate ** time_factor
    
    # 评估期间或如果decay_rate为0则无衰减
    if not training or decay_rate == 0.0:
        return input_tensor
    
    decay_tensor = torch.tensor(effective_decay, device=input_tensor.device)
    return apply_gradient_decay(input_tensor, decay_tensor) 