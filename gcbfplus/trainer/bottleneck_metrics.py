#!/usr/bin/env python3

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BottleneckRegion:
    """定义用于指标计算的瓶颈区域"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

@dataclass
class BottleneckMetrics:
    """瓶颈场景分析的综合指标"""
    throughput: float                    # 每秒通过的智能体数量
    velocity_fluctuation: float          # 瓶颈区域内速度的标准差
    total_waiting_time: float           # 等待的总时间（低速状态）
    avg_bottleneck_time: float          # 在瓶颈区域停留的平均时间
    coordination_efficiency: float       # 智能体协调的效率
    collision_rate: float               # 瓶颈区域内的碰撞率
    completion_rate: float              # 到达目标的智能体百分比

class BottleneckAnalyzer:
    """瓶颈专用多智能体协调指标分析器"""
    
    def __init__(self, config: Dict):
        """
        初始化瓶颈分析器。
        
        参数:
            config: 包含bottleneck_metrics部分的配置字典
        """
        bottleneck_config = config.get('bottleneck_metrics', {})
        
        # 瓶颈区域定义
        region_config = bottleneck_config.get('bottleneck_region', {})
        self.bottleneck_region = BottleneckRegion(
            x_min=region_config.get('x_min', 0.9),
            x_max=region_config.get('x_max', 1.1),
            y_min=region_config.get('y_min', 0.8),
            y_max=region_config.get('y_max', 1.2)
        )
        
        # 分析参数
        self.throughput_window = bottleneck_config.get('throughput_window', 1.0)
        self.velocity_threshold = bottleneck_config.get('velocity_threshold', 0.1)
        self.dt = config.get('env', {}).get('dt', 0.03)
        
        # 存储轨迹分析数据
        self.trajectory_data = []
        
    def reset(self):
        """重置分析器以开始新剧集"""
        self.trajectory_data = []
    
    def update(self, positions: torch.Tensor, velocities: torch.Tensor, 
               goals: torch.Tensor, time_step: int):
        """
        使用新的轨迹数据更新分析器。
        
        参数:
            positions: 智能体位置 [batch_size, num_agents, 2]
            velocities: 智能体速度 [batch_size, num_agents, 2] 
            goals: 智能体目标 [batch_size, num_agents, 2]
            time_step: 当前模拟时间步
        """
        batch_size, num_agents, _ = positions.shape
        
        # 转换为numpy进行更简单的处理
        pos_np = positions.detach().cpu().numpy()
        vel_np = velocities.detach().cpu().numpy()
        goals_np = goals.detach().cpu().numpy()
        
        # 存储轨迹数据
        step_data = {
            'time_step': time_step,
            'positions': pos_np,
            'velocities': vel_np,
            'goals': goals_np,
            'time': time_step * self.dt
        }
        
        self.trajectory_data.append(step_data)
    
    def is_in_bottleneck(self, position: np.ndarray) -> bool:
        """检查位置是否在瓶颈区域内"""
        x, y = position[0], position[1]
        return (self.bottleneck_region.x_min <= x <= self.bottleneck_region.x_max and
                self.bottleneck_region.y_min <= y <= self.bottleneck_region.y_max)
    
    def compute_throughput(self) -> float:
        """
        计算吞吐量：每秒通过瓶颈的智能体数量。
        
        返回:
            吞吐率（每秒智能体数量）
        """
        if len(self.trajectory_data) < 2:
            return 0.0
            
        # 跟踪通过瓶颈的智能体
        total_passages = 0
        num_agents = self.trajectory_data[0]['positions'].shape[1]
        
        # 对于每个智能体，检查他们是否通过瓶颈
        for agent_idx in range(num_agents):
            in_bottleneck_history = []
            
            for step_data in self.trajectory_data:
                position = step_data['positions'][0, agent_idx]  # 假设batch_size=1
                in_bottleneck = self.is_in_bottleneck(position)
                in_bottleneck_history.append(in_bottleneck)
            
            # 计算从外部->内部->外部的过渡作为通过
            was_outside = not in_bottleneck_history[0]
            entered_bottleneck = False
            
            for in_bottleneck in in_bottleneck_history[1:]:
                if was_outside and in_bottleneck:
                    entered_bottleneck = True
                elif entered_bottleneck and not in_bottleneck:
                    total_passages += 1
                    entered_bottleneck = False
                    was_outside = True
                else:
                    was_outside = not in_bottleneck
        
        # 计算速率
        total_time = len(self.trajectory_data) * self.dt
        throughput = total_passages / total_time if total_time > 0 else 0.0
        
        return throughput
    
    def compute_velocity_fluctuation(self) -> float:
        """
        计算速度波动：瓶颈区域内速度的标准差。
        
        返回:
            瓶颈区域内速度的标准差
        """
        bottleneck_speeds = []
        
        for step_data in self.trajectory_data:
            positions = step_data['positions'][0]  # 假设batch_size=1
            velocities = step_data['velocities'][0]
            
            for agent_idx in range(positions.shape[0]):
                position = positions[agent_idx]
                velocity = velocities[agent_idx]
                
                if self.is_in_bottleneck(position):
                    speed = np.linalg.norm(velocity)
                    bottleneck_speeds.append(speed)
        
        if len(bottleneck_speeds) == 0:
            return 0.0
            
        return float(np.std(bottleneck_speeds))
    
    def compute_waiting_time(self) -> float:
        """
        计算总等待时间：在瓶颈区域附近速度低于阈值的时间。
        
        返回:
            总等待时间（秒）
        """
        total_waiting_time = 0.0
        
        for step_data in self.trajectory_data:
            positions = step_data['positions'][0]  # 假设batch_size=1
            velocities = step_data['velocities'][0]
            
            for agent_idx in range(positions.shape[0]):
                position = positions[agent_idx]
                velocity = velocities[agent_idx]
                speed = np.linalg.norm(velocity)
                
                # 检查智能体是否接近瓶颈且速度较慢
                near_bottleneck = self.is_near_bottleneck(position)
                if near_bottleneck and speed < self.velocity_threshold:
                    total_waiting_time += self.dt
        
        return total_waiting_time
    
    def is_near_bottleneck(self, position: np.ndarray, margin: float = 0.2) -> bool:
        """检查位置是否接近瓶颈入口"""
        x, y = position[0], position[1]
        expanded_region = BottleneckRegion(
            x_min=self.bottleneck_region.x_min - margin,
            x_max=self.bottleneck_region.x_max + margin,
            y_min=self.bottleneck_region.y_min - margin,
            y_max=self.bottleneck_region.y_max + margin
        )
        return (expanded_region.x_min <= x <= expanded_region.x_max and
                expanded_region.y_min <= y <= expanded_region.y_max)
    
    def compute_avg_bottleneck_time(self) -> float:
        """
        计算每个智能体在瓶颈区域停留的平均时间。
        
        返回:
            平均瓶颈停留时间（秒）
        """
        if len(self.trajectory_data) < 2:
            return 0.0
            
        num_agents = self.trajectory_data[0]['positions'].shape[1]
        total_bottleneck_time = 0.0
        agents_in_bottleneck = 0
        
        for agent_idx in range(num_agents):
            bottleneck_time = 0.0
            
            for step_data in self.trajectory_data:
                position = step_data['positions'][0, agent_idx]
                if self.is_in_bottleneck(position):
                    bottleneck_time += self.dt
            
            if bottleneck_time > 0:
                total_bottleneck_time += bottleneck_time
                agents_in_bottleneck += 1
        
        return total_bottleneck_time / agents_in_bottleneck if agents_in_bottleneck > 0 else 0.0
    
    def compute_coordination_efficiency(self) -> float:
        """
        基于碰撞避免和流畅流动计算协调效率。
        
        返回:
            协调效率得分（0-1，越高越好）
        """
        if len(self.trajectory_data) < 2:
            return 0.0
        
        # 基于以下指标计算效率：
        # 1. 低速度方差（流畅流动）
        # 2. 高吞吐量
        # 3. 低等待时间
        
        velocity_fluctuation = self.compute_velocity_fluctuation()
        throughput = self.compute_throughput()
        waiting_time = self.compute_waiting_time()
        
        # 归一化指标（效率越高，波动越低，等待时间越低，吞吐量越高）
        max_throughput = 2.0  # 预期最大吞吐量用于归一化
        max_waiting = 10.0    # 预期最大等待时间
        max_fluctuation = 1.0 # 预期最大速度波动
        
        throughput_score = min(throughput / max_throughput, 1.0)
        waiting_score = max(0, 1.0 - waiting_time / max_waiting)
        fluctuation_score = max(0, 1.0 - velocity_fluctuation / max_fluctuation)
        
        # 加权组合
        efficiency = 0.4 * throughput_score + 0.3 * waiting_score + 0.3 * fluctuation_score
        
        return efficiency
    
    def compute_collision_rate(self, agent_radius: float = 0.05) -> float:
        """
        计算瓶颈区域内的碰撞率。
        
        参数:
            agent_radius: 每个智能体的半径用于碰撞检测
            
        返回:
            碰撞率（每秒每智能体碰撞次数）
        """
        total_collisions = 0
        total_agent_time_in_bottleneck = 0.0
        
        for step_data in self.trajectory_data:
            positions = step_data['positions'][0]  # 假设batch_size=1
            num_agents = positions.shape[0]
            
            agents_in_bottleneck = []
            for agent_idx in range(num_agents):
                if self.is_in_bottleneck(positions[agent_idx]):
                    agents_in_bottleneck.append(agent_idx)
                    total_agent_time_in_bottleneck += self.dt
            
            # 检查瓶颈内智能体的碰撞
            for i, agent_i in enumerate(agents_in_bottleneck):
                for agent_j in agents_in_bottleneck[i+1:]:
                    pos_i = positions[agent_i]
                    pos_j = positions[agent_j]
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    if distance < 2 * agent_radius:
                        total_collisions += 1
        
        # 每秒每智能体碰撞率
        collision_rate = total_collisions / total_agent_time_in_bottleneck if total_agent_time_in_bottleneck > 0 else 0.0
        
        return collision_rate
    
    def compute_completion_rate(self, goal_threshold: float = 0.1) -> float:
        """
        计算到达目标的智能体百分比。
        
        参数:
            goal_threshold: 到达目标的距离阈值
            
        返回:
            完成率（0-1）
        """
        if len(self.trajectory_data) == 0:
            return 0.0
            
        final_step = self.trajectory_data[-1]
        positions = final_step['positions'][0]  # 假设batch_size=1
        goals = final_step['goals'][0]
        num_agents = positions.shape[0]
        
        agents_at_goal = 0
        for agent_idx in range(num_agents):
            distance_to_goal = np.linalg.norm(positions[agent_idx] - goals[agent_idx])
            if distance_to_goal < goal_threshold:
                agents_at_goal += 1
        
        return agents_at_goal / num_agents
    
    def analyze(self, agent_radius: float = 0.05) -> BottleneckMetrics:
        """
        执行完整的瓶颈分析。
        
        参数:
            agent_radius: 智能体的半径用于碰撞检测
            
        返回:
            包含所有计算指标的BottleneckMetrics对象
        """
        metrics = BottleneckMetrics(
            throughput=self.compute_throughput(),
            velocity_fluctuation=self.compute_velocity_fluctuation(),
            total_waiting_time=self.compute_waiting_time(),
            avg_bottleneck_time=self.compute_avg_bottleneck_time(),
            coordination_efficiency=self.compute_coordination_efficiency(),
            collision_rate=self.compute_collision_rate(agent_radius),
            completion_rate=self.compute_completion_rate()
        )
        
        return metrics 