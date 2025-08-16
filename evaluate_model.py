#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for GCBF+BPTT Single Agent.
This version is correctly aligned with the SimpleMLPPolicy used in the BPTTTrainer.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

# --- THE CORE FIX 1: Import the CORRECT, simple policy class ---
# We define it here locally to be 100% certain of the architecture.
class SimpleMLPPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

from gcbfplus.env.single_double_integrator import SingleDoubleIntegratorEnv

@dataclass
class EpisodeResult:
    """Data structure for storing episode evaluation results."""
    success: bool
    collision: bool
    timeout: bool
    completion_time: float
    trajectory_jerk: float
    min_safe_distance: float
    final_goal_distance: float
    episode_length: int

class ModelEvaluator:
    """Comprehensive model evaluator for quantitative analysis."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device, pilot_path: str, guardian_path: str):
        """
        Initialize the model evaluator.
        """
        self.config = config
        self.device = device
        
        self.env = SingleDoubleIntegratorEnv(config.get('env', {}), device=device)
        
        # --- THE CORE FIX 2: Determine dimensions for SimpleMLPPolicy ---
        # The observation space for the simple policy is pos(2) + vel(2) + rel_goal(2) = 6
        obs_dim = 6
        action_dim = self.env.action_dim
        # Get hidden_dim from the 'policy' block, as used by the trainer
        hidden_dim = config.get('policy', {}).get('hidden_dim', 128)

        # Initialize Pilot Policy
        self.pilot_policy = SimpleMLPPolicy(obs_dim, action_dim, hidden_dim).to(device)
        print(f"Loading pilot checkpoint from: {pilot_path}")
        pilot_checkpoint = torch.load(pilot_path, map_location=self.device)
        self.pilot_policy.load_state_dict(pilot_checkpoint)
        print("✓ Pilot model checkpoint loaded successfully")

        # Initialize Guardian Policy
        self.guardian_policy = SimpleMLPPolicy(obs_dim, action_dim, hidden_dim).to(device)
        print(f"Loading guardian checkpoint from: {guardian_path}")
        guardian_checkpoint = torch.load(guardian_path, map_location=self.device)
        self.guardian_policy.load_state_dict(guardian_checkpoint)
        print("✓ Guardian model checkpoint loaded successfully")

        self.pilot_policy.eval()
        self.guardian_policy.eval()
        
        self.episode_results: List[EpisodeResult] = []
    
    def _calculate_trajectory_jerk(self, actions_list: List[torch.Tensor]) -> float:
        """Calculate trajectory jerk from a list of action tensors."""
        if not actions_list or len(actions_list) < 2:
            return 0.0
        actions_tensor = torch.cat(actions_list, dim=0).squeeze(1) # Squeeze the agent dimension
        if actions_tensor.dim() == 1: # Handle case of single action dim
            actions_tensor = actions_tensor.unsqueeze(1)
        if actions_tensor.shape[0] < 2:
            return 0.0
        jerk = torch.diff(actions_tensor, n=1, dim=0)
        jerk_magnitude = torch.norm(jerk, p=2, dim=-1)
        return float(torch.mean(jerk_magnitude).item())

    def _evaluate_single_episode(self, seed: int = None) -> EpisodeResult:
        """Evaluates a single episode using the PILOT policy."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        state = self.env.reset(batch_size=1)
        episode_length = 0
        min_safe_distance = float('inf')
        actions_list = []
        
        while episode_length < self.env.max_steps:
            observations = self.env.get_observation(state)
            
            with torch.no_grad():
                # --- THE CORE FIX 3: Policy call is now simple ---
                actions = self.pilot_policy(observations)
            
            actions_list.append(actions.detach().cpu())
            
            step_result = self.env.step(state, actions, None)
            next_state = step_result.next_state
            
            if state.obstacles is not None and state.obstacles.nelement() > 0:
                rel_pos = state.position.unsqueeze(1) - state.obstacles[..., :2]
                distances = torch.norm(rel_pos, dim=2) - state.obstacles[..., 2] - self.env.agent_radius
                min_dist_this_step = torch.min(distances).item()
                min_safe_distance = min(min_safe_distance, min_dist_this_step)

            if step_result.cost.item() > 0:
                status = 'collision'
                break
            
            if self.env.get_goal_distance(next_state).item() < self.env.agent_radius:
                status = 'success'
                episode_length += 1
                break

            state = next_state
            episode_length += 1
        else:
            status = 'timeout'
            
        final_goal_dist = self.env.get_goal_distance(next_state).item()
        
        return EpisodeResult(
            success=(status == 'success'),
            collision=(status == 'collision'),
            timeout=(status == 'timeout'),
            completion_time=episode_length * self.env.dt,
            trajectory_jerk=self._calculate_trajectory_jerk(actions_list),
            min_safe_distance=min_safe_distance if min_safe_distance != float('inf') else 0.0,
            final_goal_distance=final_goal_dist,
            episode_length=episode_length
        )

    def evaluate(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        eval_config = self.config.get('evaluation', {})
        num_episodes = eval_config.get('num_episodes', 500)
        
        print(f"Starting evaluation of {num_episodes} episodes...")
        
        for episode_idx in range(num_episodes):
            if (episode_idx + 1) % 50 == 0:
                print(f"Progress: {episode_idx + 1}/{num_episodes} episodes completed")
            
            result = self._evaluate_single_episode(seed=episode_idx)
            self.episode_results.append(result)
        
        return self._calculate_aggregate_kpis()

    def _calculate_aggregate_kpis(self) -> Dict[str, Any]:
        """Calculate aggregated KPIs."""
        if not self.episode_results: return {}
        num_episodes = len(self.episode_results)
        success_count = sum(r.success for r in self.episode_results)
        collision_count = sum(r.collision for r in self.episode_results)
        timeout_count = sum(r.timeout for r in self.episode_results)
        successful_episodes = [r for r in self.episode_results if r.success]
        
        return {
            'total_episodes': num_episodes,
            'success_rate': (success_count / num_episodes) * 100,
            'collision_rate': (collision_count / num_episodes) * 100,
            'timeout_rate': (timeout_count / num_episodes) * 100,
            'avg_completion_time': np.mean([r.completion_time for r in successful_episodes]) if successful_episodes else 0.0,
            'avg_trajectory_jerk': np.mean([r.trajectory_jerk for r in successful_episodes]) if successful_episodes else 0.0,
            'avg_min_safe_distance': np.mean([r.min_safe_distance for r in self.episode_results]),
            'success_count': success_count,
            'collision_count': collision_count,
            'timeout_count': timeout_count
        }
    
    def print_final_report(self, kpis: Dict[str, Any]):
        """Print the final evaluation report."""
        print("\n" + "="*80)
        print("                    FINAL EVALUATION REPORT")
        print("="*80)
        print(f"\n📊 EPISODE OUTCOMES:")
        print(f"   Total Episodes: {kpis['total_episodes']}")
        print(f"   Success Rate:   {kpis['success_rate']:.2f}% ({kpis['success_count']} episodes)")
        print(f"   Collision Rate: {kpis['collision_rate']:.2f}% ({kpis['collision_count']} episodes)")
        print(f"   Timeout Rate:   {kpis['timeout_rate']:.2f}% ({kpis['timeout_count']} episodes)")
        print(f"\n🎯 PERFORMANCE METRICS (for successful episodes):")
        print(f"   Average Completion Time: {kpis['avg_completion_time']:.3f} seconds")
        print(f"   Average Trajectory Jerk: {kpis['avg_trajectory_jerk']:.6f}")
        print(f"\n🛡️ SAFETY METRICS (for all episodes):")
        print(f"   Average Min Safe Distance: {kpis['avg_min_safe_distance']:.4f} meters")
        print(f"\n🔧 MODEL CONFIGURATION:")
        print(f"   Pilot Model: {self.config.get('evaluation', {}).get('pilot_path')}")
        print(f"   Guardian Model: {self.config.get('evaluation', {}).get('guardian_path')}")
        print("\n" + "="*80)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained GCBF+BPTT model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--pilot_path", type=str, required=True, help="Path to the trained pilot.pt model file.")
    parser.add_argument("--guardian_path", type=str, required=True, help="Path to the trained guardian.pt model file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Store paths in config for the report
    if 'evaluation' not in config: config['evaluation'] = {}
    config['evaluation']['pilot_path'] = args.pilot_path
    config['evaluation']['guardian_path'] = args.guardian_path
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    try:
        evaluator = ModelEvaluator(config, device, pilot_path=args.pilot_path, guardian_path=args.guardian_path)
        kpis = evaluator.evaluate()
        evaluator.print_final_report(kpis)
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
