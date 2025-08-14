"""
多智能体导航的简化视觉渲染器
该模块提供PyTorch3D的轻量级替代方案，用于生成深度图像。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
import math


class SimpleDepthRenderer(nn.Module):
    """
    简化的深度渲染器，从2D智能体/障碍物位置创建深度图像。
    这是我们导航任务中PyTorch3D的轻量级替代方案。
    """
    
    def __init__(self, config: Dict):
        super(SimpleDepthRenderer, self).__init__()
        
        self.image_size = config.get('image_size', 64)
        self.camera_fov = config.get('camera_fov', 90.0)  # 角度
        self.camera_range = config.get('camera_range', 3.0)
        self.agent_radius = config.get('agent_radius', 0.05)
        self.obstacle_base_height = config.get('obstacle_base_height', 0.5)
        
        # 将视野角度转换为弧度
        self.fov_rad = math.radians(self.camera_fov)
        
        # 为图像创建坐标网格
        self.register_buffer('pixel_coords', self._create_pixel_coordinates())
        
    def _create_pixel_coordinates(self) -> torch.Tensor:
        """为深度图像创建标准化的像素坐标。"""
        # 为图像创建像素坐标 [-1, 1]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size),
            torch.linspace(-1, 1, self.image_size),
            indexing='ij'
        )
        
        # 堆叠以创建 [H, W, 2] 坐标网格
        coords = torch.stack([x_coords, y_coords], dim=2)
        return coords
    
    def render_depth_from_position(
        self, 
        agent_pos: torch.Tensor,      # [2] - 当前智能体位置
        goal_pos: torch.Tensor,       # [2] - 智能体的目标位置  
        other_agents: torch.Tensor,   # [N-1, 2] - 其他智能体位置
        obstacles: Optional[torch.Tensor] = None  # [M, 3] - 障碍物位置和半径
    ) -> torch.Tensor:
        """
        从智能体的视角渲染深度图像。
        
        参数:
            agent_pos: 当前智能体位置 [x, y]
            goal_pos: 智能体的目标位置 [x, y] 
            other_agents: 其他智能体的位置 [N-1, 2]
            obstacles: 障碍物位置和半径 [M, 3]，每行为 [x, y, radius]
            
        返回:
            深度图像 [1, H, W]，值在 [0, 1] 范围内
        """
        device = agent_pos.device
        
        # 计算观察方向（朝向目标）
        view_direction = goal_pos - agent_pos
        view_direction = view_direction / (torch.norm(view_direction) + 1e-8)
        
        # 创建旋转矩阵，将视角与y轴对齐
        cos_theta = view_direction[1]  # y分量
        sin_theta = view_direction[0]  # x分量
        
        rotation_matrix = torch.tensor([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ], device=device, dtype=torch.float32)
        
        # 初始化深度图像（远距离 = 1.0，近距离 = 0.0）
        depth_image = torch.ones(self.image_size, self.image_size, device=device)
        
        # 渲染其他智能体
        if other_agents.numel() > 0:
            depth_image = self._render_objects(
                depth_image, agent_pos, rotation_matrix, other_agents, 
                self.agent_radius, height=0.2
            )
        
        # 渲染障碍物
        if obstacles is not None and obstacles.numel() > 0:
            obstacle_positions = obstacles[:, :2]  # [M, 2]
            obstacle_radii = obstacles[:, 2]       # [M]
            
            for i in range(obstacles.shape[0]):
                single_obstacle = obstacle_positions[i:i+1]  # [1, 2]
                obstacle_radius = obstacle_radii[i].item()
                depth_image = self._render_objects(
                    depth_image, agent_pos, rotation_matrix, single_obstacle,
                    obstacle_radius, height=self.obstacle_base_height
                )
        
        return depth_image.unsqueeze(0)  # 添加通道维度 [1, H, W]
    
    def _render_objects(
        self,
        depth_image: torch.Tensor,
        agent_pos: torch.Tensor,
        rotation_matrix: torch.Tensor,
        object_positions: torch.Tensor,
        object_radius: float,
        height: float
    ) -> torch.Tensor:
        """
        将对象（智能体或障碍物）渲染到深度图像中。
        
        参数:
            depth_image: 当前深度图像 [H, W]
            agent_pos: 观察智能体位置 [2]
            rotation_matrix: 2D旋转矩阵 [2, 2]
            object_positions: 要渲染的对象位置 [N, 2]
            object_radius: 要渲染的对象半径
            height: 对象高度（影响深度强度）
            
        返回:
            更新后的深度图像 [H, W]
        """
        device = depth_image.device
        
        for obj_pos in object_positions:
            # 将对象位置转换为相机坐标系
            relative_pos = obj_pos - agent_pos  # [2]
            
            # 如果对象在智能体后方（相机坐标中y为负），则跳过
            if torch.dot(relative_pos, rotation_matrix[1, :]) < 0:
                continue
                
            # 应用旋转以获得相机坐标中的位置
            cam_pos = torch.matmul(rotation_matrix, relative_pos)  # [2]
            
            # 投影到图像坐标
            # 从相机坐标映射到图像像素坐标
            distance = torch.norm(cam_pos)
            if distance > self.camera_range:
                continue
                
            # 计算角度位置
            angle_x = torch.atan2(cam_pos[0], cam_pos[1])  # 水平角度
            
            # 检查对象是否在视野范围内
            if abs(angle_x) > self.fov_rad / 2:
                continue
            
            # 将角度映射到像素坐标
            pixel_x = (angle_x / (self.fov_rad / 2)) * 0.5  # 标准化到 [-0.5, 0.5]
            pixel_x_int = int((pixel_x + 0.5) * self.image_size)  # 映射到 [0, image_size]
            
            # 根据距离计算对象在像素中的大小
            angular_size = object_radius / (distance + 1e-8)
            pixel_radius = int(angular_size * self.image_size * 0.5)
            pixel_radius = max(1, min(pixel_radius, self.image_size // 4))
            
            # 计算深度值（较近的对象具有较小的深度值）
            depth_value = min(distance / self.camera_range, 1.0)
            # 通过对象高度调制深度以实现视觉区分
            depth_value = depth_value * (1.0 - height * 0.3)
            depth_value = max(0.0, depth_value)
            
            # 渲染圆形对象
            for dy in range(-pixel_radius, pixel_radius + 1):
                for dx in range(-pixel_radius, pixel_radius + 1):
                    if dx*dx + dy*dy <= pixel_radius*pixel_radius:
                        py = self.image_size // 2 + dy
                        px = pixel_x_int + dx
                        
                        if 0 <= py < self.image_size and 0 <= px < self.image_size:
                            # 如果此对象更近，则更新深度
                            depth_image[py, px] = min(depth_image[py, px], depth_value)
        
        return depth_image
    
    def add_noise_and_realism(self, depth_image: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
        """
        为深度图像添加噪声和真实效果。
        
        参数:
            depth_image: 清洁深度图像 [1, H, W]
            noise_level: 要添加的噪声量
            
        返回:
            有噪声的深度图像 [1, H, W]
        """
        # 添加高斯噪声
        noise = torch.randn_like(depth_image) * noise_level
        noisy_depth = torch.clamp(depth_image + noise, 0.0, 1.0)
        
        # 添加轻微模糊以模拟传感器限制
        # 简单的盒式滤波器
        kernel_size = 3
        padding = kernel_size // 2
        
        # 创建简单的平均核
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=depth_image.device) / (kernel_size * kernel_size)
        
        # 应用卷积以实现模糊效果
        blurred = torch.nn.functional.conv2d(
            noisy_depth.unsqueeze(0), kernel, padding=padding
        ).squeeze(0)
        
        return blurred


def create_simple_renderer(config: Dict) -> SimpleDepthRenderer:
    """
    创建简单深度渲染器的工厂函数。
    
    参数:
        config: 配置字典
        
    返回:
        配置好的SimpleDepthRenderer实例
    """
    return SimpleDepthRenderer(config) 