#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Inference Utility
統一的模型推理工具

解決所有配置加載和張量維度問題
"""

import os
import torch
import yaml
import numpy as np
from typing import Dict, Tuple, Any
import torch.nn as nn
from pathlib import Path

from ..policy import BPTTPolicy
from ..env.multi_agent_env import MultiAgentState


def load_model_and_config(model_dir: str) -> Tuple[nn.Module, Dict]:
    """
    從模型目錄加載策略網絡和配置
    
    Args:
        model_dir: 模型目錄路徑 (e.g., logs/bptt/models/9500)
    
    Returns:
        Tuple[nn.Module, Dict]: (策略網絡, 配置字典)
    """
    print(f"🔍 加載模型和配置: {model_dir}")
    
    # 檢查必要文件
    policy_path = os.path.join(model_dir, "policy.pt")
    config_path = os.path.join(model_dir, "config.pt")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"策略模型文件不存在: {policy_path}")
    
    # 加載配置
    config = None
    if os.path.exists(config_path):
        try:
            config = torch.load(config_path, map_location='cpu', weights_only=False)
            print("✅ 從 config.pt 加載配置成功")
        except Exception as e:
            print(f"⚠️ config.pt 加載失敗: {e}")
    
    # 如果config.pt失敗，嘗試尋找關聯的yaml配置
    if config is None:
        # 查找可能的配置文件
        config_candidates = [
            "config/alpha_medium_obs.yaml",
            "config/simple_collaboration.yaml", 
            "config/collaboration_test.yaml"
        ]
        
        for config_file in config_candidates:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ 從 {config_file} 加載配置成功")
                    break
                except Exception as e:
                    print(f"⚠️ {config_file} 加載失敗: {e}")
    
    # 如果還是沒有配置，使用默認配置
    if config is None:
        print("⚠️ 使用默認配置")
        config = {
            'env': {
                'n_agents': 8,
                'world_size': 4.0,
                'agent_radius': 0.1,
                'goal_radius': 0.2,
                'dt': 0.05,
                'obstacles': {
                    'enabled': True,
                    'num_obstacles': 8,
                    'radius_range': [0.2, 0.4]
                }
            }
        }
    
    # 加載策略模型並推斷架構
    print("🔍 分析策略模型架構...")
    policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=False)
    
    # 從state_dict推斷網絡架構
    perception_weight_key = 'perception.mlp.0.weight'
    if perception_weight_key in policy_state_dict:
        perception_input_dim = policy_state_dict[perception_weight_key].shape[1]
        perception_hidden_dim = policy_state_dict[perception_weight_key].shape[0]
        print(f"📐 推斷的感知模塊架構: input_dim={perception_input_dim}, hidden_dim={perception_hidden_dim}")
    else:
        # 默認架構
        perception_input_dim = 9  # 有障礙物環境
        perception_hidden_dim = 128
        print(f"⚠️ 使用默認感知模塊架構: input_dim={perception_input_dim}, hidden_dim={perception_hidden_dim}")
    
    # 推斷記憶模塊架構
    memory_weight_key = 'memory.rnn.weight_ih_l0'
    if memory_weight_key in policy_state_dict:
        memory_input_dim = policy_state_dict[memory_weight_key].shape[1]
        memory_hidden_dim = policy_state_dict[memory_weight_key].shape[0] // 4  # LSTM有4個gate
        print(f"📐 推斷的記憶模塊架構: input_dim={memory_input_dim}, hidden_dim={memory_hidden_dim}")
    else:
        memory_hidden_dim = 128
        print(f"⚠️ 使用默認記憶模塊架構: hidden_dim={memory_hidden_dim}")
    
    # 推斷策略頭架構
    policy_head_weight_key = 'policy_head.mlp.0.weight'
    if policy_head_weight_key in policy_state_dict:
        policy_head_input_dim = policy_state_dict[policy_head_weight_key].shape[1]
        policy_head_output_key = None
        for key in policy_state_dict.keys():
            if key.startswith('policy_head.mlp.') and key.endswith('.weight'):
                policy_head_output_key = key
        if policy_head_output_key:
            policy_head_output_dim = policy_state_dict[policy_head_output_key].shape[0]
        else:
            policy_head_output_dim = 2  # 默認2D動作
        print(f"📐 推斷的策略頭架構: input_dim={policy_head_input_dim}, output_dim={policy_head_output_dim}")
    else:
        policy_head_output_dim = 2
        print(f"⚠️ 使用默認策略頭架構: output_dim={policy_head_output_dim}")
    
    # 構建策略網絡配置
    policy_config = {
        'perception': {
            'use_vision': False,
            'input_dim': perception_input_dim,
            'hidden_dim': perception_hidden_dim,
            'activation': 'relu'
        },
        'memory': {
            'hidden_dim': memory_hidden_dim,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': policy_head_output_dim,
            'hidden_dim': memory_hidden_dim,  # 通常與memory一致
            'activation': 'relu',
            'predict_alpha': True
        }
    }
    
    # 創建策略網絡
    print("🏗️ 創建策略網絡...")
    policy_network = BPTTPolicy(policy_config)
    
    # 加載權重
    try:
        missing_keys, unexpected_keys = policy_network.load_state_dict(policy_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ 缺失的權重: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ 意外的權重: {unexpected_keys}")
        print("✅ 策略網絡權重加載成功")
    except Exception as e:
        print(f"❌ 權重加載失敗: {e}")
        raise
    
    # 設置為評估模式
    policy_network.eval()
    
    return policy_network, config


def run_simulation_for_visualization(env, policy: nn.Module, steps: int = 300) -> Tuple[list, list]:
    """
    運行仿真生成可視化軌跡
    
    Args:
        env: 環境實例
        policy: 策略網絡
        steps: 仿真步數
    
    Returns:
        Tuple[list, list]: (軌跡列表, 障礙物列表)
    """
    print(f"🎬 開始仿真: {steps} 步")
    
    # 重置環境
    state = env.reset()
    
    # 存儲軌跡
    trajectories = []
    
    # 獲取障礙物信息
    obstacles = []
    if hasattr(env, 'obstacles') and env.obstacles:
        for obs in env.obstacles:
            if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                obstacles.append({
                    'center': obs.center.copy() if hasattr(obs.center, 'copy') else np.array(obs.center),
                    'radius': obs.radius
                })
    
    print(f"📊 環境信息: {env.num_agents} 智能體, {len(obstacles)} 障礙物")
    
    for step in range(steps):
        # 提取位置和速度
        if hasattr(state, 'positions'):
            # MultiAgentState 格式
            positions = state.positions.numpy() if hasattr(state.positions, 'numpy') else np.array(state.positions)
            velocities = state.velocities.numpy() if hasattr(state.velocities, 'numpy') else np.array(state.velocities)
        elif isinstance(state, torch.Tensor):
            # 張量格式
            state_np = state.numpy()
            positions = state_np[:, :2]
            velocities = state_np[:, 2:4] if state_np.shape[1] >= 4 else np.zeros_like(positions)
        else:
            # numpy數組格式
            state_np = np.array(state)
            positions = state_np[:, :2]
            velocities = state_np[:, 2:4] if state_np.shape[1] >= 4 else np.zeros_like(positions)
        
        # 記錄軌跡
        trajectories.append({
            'positions': positions.copy(),
            'velocities': velocities.copy(),
            'step': step
        })
        
        # 獲取觀測
        try:
            obs = env.get_observation(state)
            
            # 確保觀測是正確的張量格式
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)
            
            # 添加批次維度: [n_agents, obs_dim] -> [1, n_agents, obs_dim]
            if obs.dim() == 2:
                obs_tensor = obs.unsqueeze(0)
            else:
                obs_tensor = obs
            
            print(f"Step {step}: obs shape = {obs_tensor.shape}")
            
            # 策略網絡預測
            with torch.no_grad():
                try:
                    result = policy(obs_tensor)
                    
                    # 處理返回值
                    if isinstance(result, tuple):
                        if len(result) >= 2:
                            actions, alpha = result[0], result[1]
                        else:
                            actions = result[0]
                            alpha = None
                    else:
                        actions = result
                        alpha = None
                    
                    # 移除批次維度: [1, n_agents, action_dim] -> [n_agents, action_dim]
                    if actions.dim() == 3:
                        actions = actions.squeeze(0)
                    
                    actions_np = actions.numpy()
                    
                except Exception as policy_error:
                    print(f"⚠️ 策略預測失敗 (step {step}): {policy_error}")
                    # 使用零動作
                    actions_np = np.zeros((env.num_agents, 2))
            
        except Exception as obs_error:
            print(f"⚠️ 觀測獲取失敗 (step {step}): {obs_error}")
            # 使用零動作
            actions_np = np.zeros((env.num_agents, 2))
        
        # 執行動作
        try:
            state = env.step(state, actions_np)
        except Exception as step_error:
            print(f"⚠️ 環境步進失敗 (step {step}): {step_error}")
            break
        
        # 簡單的目標檢查
        if hasattr(env, 'goal_positions') and step % 50 == 0:
            goal_distances = np.linalg.norm(positions - env.goal_positions, axis=1)
            avg_distance = np.mean(goal_distances)
            print(f"Step {step}: 平均目標距離 = {avg_distance:.3f}")
    
    print(f"✅ 仿真完成: 生成 {len(trajectories)} 個軌跡點")
    return trajectories, obstacles