#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for GCBF+BPTT Single Agent

This script performs large-scale quantitative evaluation of trained models,
calculating key performance indicators across multiple episodes.
"""

import argparse
import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Import our project modules
from gcbfplus.env.single_double_integrator import SingleDoubleIntegratorEnv
from gcbfplus.trainer.bptt_trainer import SimpleMLPPolicy as BPTTPolicy


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
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary
            device: Device to run evaluation on
        """
        self.config = config
        self.device = device
        
        # Extract evaluation configuration
        eval_config = config.get('evaluation', {})
        self.num_episodes = eval_config.get('num_episodes', 500)
        self.run_name = eval_config.get('run_name', 'final_config')
        self.model_step = eval_config.get('model_step', 10000)
        
        # Initialize environment
        self.env = SingleDoubleIntegratorEnv(config.get('env', {}), device=device)
        
        # Correctly initialize the SimpleMLPPolicy with explicit arguments
        policy_cfg = config.get('policy', {})
        # Based on our inspect_model.py script, the trained model has these dimensions
        input_dim = 6
        output_dim = 2
        hidden_dim = policy_cfg.get('hidden_dim', 128) # Get hidden_dim from config

        self.policy = BPTTPolicy(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
        self.policy.to(device)
        
        # Load trained model checkpoint
        self._load_checkpoint()
        
        # Set policy to evaluation mode
        self.policy.eval()
        
        # Results storage
        self.episode_results: List[EpisodeResult] = []
        
    def _load_checkpoint(self):
        """Load the trained model checkpoint."""
        # Construct checkpoint path
        log_dir = self.config.get('trainer', {}).get('log_dir', 'logs')
        checkpoint_path = Path(log_dir) / self.run_name / 'models' / str(self.model_step) / 'policy.pt'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint)
        print("✓ Model checkpoint loaded successfully")
    
    def _calculate_trajectory_jerk(self, actions: torch.Tensor) -> float:
        """
        Calculate trajectory jerk (rate of change of acceleration).
        
        Args:
            actions: Action tensor of shape [episode_length, action_dim]
            
        Returns:
            Average jerk magnitude
        """
        if actions.shape[0] < 3:
            return 0.0
        
        # Calculate acceleration changes (jerk)
        # actions represent forces, so their differences represent jerk
        jerk = torch.diff(actions, dim=0)
        jerk_magnitude = torch.norm(jerk, dim=1)
        
        return float(torch.mean(jerk_magnitude).item())
    
    def _evaluate_single_episode(self, seed: int = None) -> EpisodeResult:
        """
        Evaluate a single episode.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            EpisodeResult containing all metrics
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Reset environment
        state = self.env.reset()
        episode_length = 0
        trajectory_jerk = 0.0
        min_safe_distance = float('inf')
        
        # Store actions for jerk calculation
        actions_list = []
        
        # Episode loop
        while episode_length < self.env.max_steps:
            # Get observation
            observations = self.env.get_observation(state)
            if hasattr(observations, 'to'):
                observations = observations.to(self.device)
            
            # Get action from policy (no gradients needed for evaluation)
            with torch.no_grad():
                actions = self.policy(observations)
                alpha = None
                dynamic_margins = None
            
            # Store action for jerk calculation
            actions_list.append(actions.detach().cpu())
            
            # Environment step
            step_result = self.env.step(state, actions, None)
            next_state = step_result.next_state
            
            # Calculate safety distance to obstacles
            if hasattr(self.env, 'get_obstacle_distances'):
                obstacle_distances = self.env.get_obstacle_distances(next_state)
                if obstacle_distances is not None:
                    min_dist = torch.min(obstacle_distances).item()
                    min_safe_distance = min(min_safe_distance, min_dist)
            
            # Check for collision
            if step_result.cost > 0:
                return EpisodeResult(
                    success=False,
                    collision=True,
                    timeout=False,
                    completion_time=episode_length * self.env.dt,
                    trajectory_jerk=self._calculate_trajectory_jerk(torch.stack(actions_list)),
                    min_safe_distance=min_safe_distance,
                    final_goal_distance=float(self.env.get_goal_distance(next_state).item()),
                    episode_length=episode_length
                )
            
            # Check for goal reaching
            goal_distance = self.env.get_goal_distance(next_state)
            if goal_distance < 0.1:  # Success threshold
                return EpisodeResult(
                    success=True,
                    collision=False,
                    timeout=False,
                    completion_time=episode_length * self.env.dt,
                    trajectory_jerk=self._calculate_trajectory_jerk(torch.stack(actions_list)),
                    min_safe_distance=min_safe_distance,
                    final_goal_distance=float(goal_distance.item()),
                    episode_length=episode_length
                )
            
            # Update state
            state = next_state
            episode_length += 1
        
        # Episode timeout
        return EpisodeResult(
            success=False,
            collision=False,
            timeout=True,
            completion_time=episode_length * self.env.dt,
            trajectory_jerk=self._calculate_trajectory_jerk(torch.stack(actions_list)),
            min_safe_distance=min_safe_distance,
            final_goal_distance=float(self.env.get_goal_distance(state).item()),
            episode_length=episode_length
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all episodes.
        
        Returns:
            Dictionary containing aggregated evaluation results
        """
        print(f"Starting evaluation of {self.num_episodes} episodes...")
        
        for episode_idx in range(self.num_episodes):
            if (episode_idx + 1) % 50 == 0:
                print(f"Progress: {episode_idx + 1}/{self.num_episodes} episodes completed")
            
            # Evaluate single episode
            result = self._evaluate_single_episode(seed=episode_idx)
            self.episode_results.append(result)
        
        # Calculate aggregated KPIs
        return self._calculate_aggregate_kpis()
    
    def _calculate_aggregate_kpis(self) -> Dict[str, Any]:
        """Calculate aggregated key performance indicators."""
        if not self.episode_results:
            return {}
        
        # Success metrics
        success_count = sum(1 for r in self.episode_results if r.success)
        collision_count = sum(1 for r in self.episode_results if r.collision)
        timeout_count = sum(1 for r in self.episode_results if r.timeout)
        
        success_rate = (success_count / len(self.episode_results)) * 100
        collision_rate = (collision_count / len(self.episode_results)) * 100
        timeout_rate = (timeout_count / len(self.episode_results)) * 100
        
        # Performance metrics (only for successful episodes)
        successful_episodes = [r for r in self.episode_results if r.success]
        
        if successful_episodes:
            avg_completion_time = np.mean([r.completion_time for r in successful_episodes])
            avg_trajectory_jerk = np.mean([r.trajectory_jerk for r in successful_episodes])
        else:
            avg_completion_time = 0.0
            avg_trajectory_jerk = 0.0
        
        # Safety metrics (for all episodes)
        valid_safe_distances = [r.min_safe_distance for r in self.episode_results if r.min_safe_distance != float('inf')]
        avg_min_safe_distance = np.mean(valid_safe_distances) if valid_safe_distances else 0.0
        
        return {
            'total_episodes': len(self.episode_results),
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_completion_time': avg_completion_time,
            'avg_trajectory_jerk': avg_trajectory_jerk,
            'avg_min_safe_distance': avg_min_safe_distance,
            'success_count': success_count,
            'collision_count': collision_count,
            'timeout_count': timeout_count
        }
    
    def print_final_report(self, kpis: Dict[str, Any]):
        """Print comprehensive final evaluation report."""
        print("\n" + "="*80)
        print("                    FINAL EVALUATION REPORT")
        print("="*80)
        
        print(f"\n📊 EPISODE OUTCOMES:")
        print(f"   Total Episodes: {kpis['total_episodes']}")
        print(f"   Success Rate:   {kpis['success_rate']:.2f}% ({kpis['success_count']} episodes)")
        print(f"   Collision Rate: {kpis['collision_rate']:.2f}% ({kpis['collision_count']} episodes)")
        print(f"   Timeout Rate:   {kpis['timeout_rate']:.2f}% ({kpis['timeout_count']} episodes)")
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Average Completion Time: {kpis['avg_completion_time']:.3f} seconds")
        print(f"   Average Trajectory Jerk: {kpis['avg_trajectory_jerk']:.6f}")
        print(f"   Average Min Safe Distance: {kpis['avg_min_safe_distance']:.4f} meters")
        
        print(f"\n🔧 MODEL CONFIGURATION:")
        print(f"   Run Name: {self.run_name}")
        print(f"   Model Step: {self.model_step}")
        print(f"   Device: {self.device}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained GCBF+BPTT model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--model_step", type=int, default=10000, help="Model checkpoint step to load")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override model step if specified
    if 'evaluation' not in config:
        config['evaluation'] = {}
    config['evaluation']['model_step'] = args.model_step
    
    # Set device
    device = torch.device(config.get('device', 'cuda'))
    print(f"Using device: {device}")
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(config, device)
        
        # Run evaluation
        kpis = evaluator.evaluate()
        
        # Print final report
        evaluator.print_final_report(kpis)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
