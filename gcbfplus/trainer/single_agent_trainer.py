"""
Single-agent BPTT trainer - Clean and efficient training loop.
Implements backpropagation through time for single-agent control.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List

from ..env.single_agent_env import SingleAgentEnv, SingleAgentState
from ..env.single_double_integrator import SingleDoubleIntegratorEnv
from ..policy.single_agent_policy import SingleAgentPolicy
from ..env.single_gcbf_layer import SingleAgentGCBFLayer


class SingleAgentBPTTTrainer:
    """
    BPTT trainer for single-agent systems.
    Optimizes policy through differentiable physics simulation.
    """
    
    def __init__(
        self,
        env: SingleAgentEnv,
        policy_network: nn.Module,
        cbf_network: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize single-agent BPTT trainer.
        
        Args:
            env: Single-agent environment
            policy_network: Policy network to train
            cbf_network: Optional CBF safety network
            config: Training configuration
        """
        self.env = env
        self.policy_network = policy_network
        self.cbf_network = cbf_network
        
        # Get device from policy network
        self.device = next(policy_network.parameters()).device
        
        # Configuration
        self.config = config if config is not None else {}
        
        # Extract parameters
        self.run_name = self.config.get('run_name', 'SingleAgent_BPTT')
        self.log_dir = f"logs/{self.run_name}"
        
        # Training parameters
        training_config = self.config.get('training', {})
        self.training_steps = training_config.get('training_steps', 10000)
        self.eval_interval = training_config.get('eval_interval', 100)
        self.save_interval = training_config.get('save_interval', 1000)
        self.horizon_length = training_config.get('horizon_length', 50)
        self.batch_size = training_config.get('batch_size', 32)
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        # Loss weights
        self.goal_weight = training_config.get('goal_weight', 1.0)
        self.safety_weight = training_config.get('safety_weight', 10.0)
        self.control_weight = training_config.get('control_weight', 0.1)
        self.jerk_weight = training_config.get('jerk_weight', 0.05)
        self.alpha_reg_weight = training_config.get('alpha_reg_weight', 0.01)
        
        # Safety parameters
        self.use_probabilistic_shield = training_config.get('use_probabilistic_shield', False)
        self.use_adaptive_margin = training_config.get('use_adaptive_margin', False)
        self.min_safety_margin = training_config.get('min_safety_margin', 0.05)
        self.max_safety_margin = training_config.get('max_safety_margin', 0.15)
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_dir = os.path.join(self.log_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize optimizer
        params = list(self.policy_network.parameters())
        if self.cbf_network is not None:
            params += list(self.cbf_network.parameters())
        
        learning_rate = training_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Learning rate scheduler
        if training_config.get('use_lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.get('lr_step_size', 2000),
                gamma=training_config.get('lr_gamma', 0.5)
            )
        else:
            self.scheduler = None
        
        # Metrics tracking
        self.metrics = {
            'loss': [],
            'goal_distance': [],
            'min_obstacle_distance': [],
            'success_rate': [],
            'collision_rate': []
        }
    
    def compute_loss(
        self,
        trajectory: Dict[str, List[torch.Tensor]],
        final_state: SingleAgentState
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss for a trajectory.
        
        Args:
            trajectory: Dictionary containing trajectory data
            final_state: Final state after rollout
            
        Returns:
            Total loss and component dictionary
        """
        device = self.device
        
        # Extract trajectory data
        positions = torch.stack(trajectory['positions'])  # [T, batch, 2]
        velocities = torch.stack(trajectory['velocities'])  # [T, batch, 2]
        actions = torch.stack(trajectory['actions'])  # [T, batch, 2]
        goals = trajectory['goals'][0]  # [batch, 2]
        
        T, batch_size = positions.shape[:2]
        
        # Goal achievement loss (distance to goal)
        final_position = positions[-1]
        goal_distance = torch.norm(final_position - goals, dim=1)
        goal_loss = goal_distance.mean()
        
        # Safety loss (obstacle collisions)
        safety_loss = torch.zeros(1, device=device)
        if 'min_distances' in trajectory:
            min_distances = torch.stack(trajectory['min_distances'])  # [T, batch]
            # Penalize small distances exponentially
            safety_violations = torch.relu(0.3 - min_distances)  # Activate when < 0.3
            safety_loss = (safety_violations ** 2).mean()
        
        # Control regularization
        control_loss = (actions ** 2).mean()
        
        # Jerk penalty (smoothness)
        jerk_loss = torch.zeros(1, device=device)
        if T > 1:
            action_diff = actions[1:] - actions[:-1]
            jerk_loss = (action_diff ** 2).mean()
        
        # Alpha regularization (encourage confidence in safe areas)
        alpha_loss = torch.zeros(1, device=device)
        if 'alpha_safety' in trajectory:
            alphas = torch.stack(trajectory['alpha_safety'])  # [T, batch, 1]
            # Encourage high alpha when far from obstacles
            if 'min_distances' in trajectory:
                safe_mask = (min_distances > 0.5).float()  # Safe when > 0.5
                alpha_loss = (safe_mask * (1 - alphas.squeeze(-1))).mean()
        
        # Total loss
        total_loss = (
            self.goal_weight * goal_loss +
            self.safety_weight * safety_loss +
            self.control_weight * control_loss +
            self.jerk_weight * jerk_loss +
            self.alpha_reg_weight * alpha_loss
        )
        
        # Return components for logging
        components = {
            'goal_loss': goal_loss,
            'safety_loss': safety_loss,
            'control_loss': control_loss,
            'jerk_loss': jerk_loss,
            'alpha_loss': alpha_loss,
            'total_loss': total_loss
        }
        
        return total_loss, components
    
    def rollout(
        self,
        initial_state: SingleAgentState,
        horizon: int
    ) -> Tuple[SingleAgentState, Dict[str, List[torch.Tensor]]]:
        """
        Perform a differentiable rollout.
        
        Args:
            initial_state: Starting state
            horizon: Number of steps
            
        Returns:
            Final state and trajectory data
        """
        state = initial_state
        trajectory = {
            'positions': [],
            'velocities': [],
            'actions': [],
            'goals': [state.goal],
            'min_distances': [],
            'alpha_safety': []
        }
        
        for t in range(horizon):
            # Get observation
            obs = self.env.get_observation(state)
            
            # Get action from policy
            policy_output = self.policy_network(obs)
            nominal_action = policy_output['action']
            
            # Apply safety layer if available
            if self.cbf_network is not None:
                state_dict = {
                    'position': state.position,
                    'velocity': state.velocity,
                    'obstacles': state.obstacles
                }
                
                safety_output = self.cbf_network(
                    state_dict,
                    nominal_action,
                    mode='probabilistic' if self.use_probabilistic_shield else 'hard'
                )
                
                action = safety_output['action']
                if 'alpha_safety' in safety_output:
                    trajectory['alpha_safety'].append(safety_output['alpha_safety'])
            else:
                action = nominal_action
            
            # Store trajectory data
            trajectory['positions'].append(state.position)
            trajectory['velocities'].append(state.velocity)
            trajectory['actions'].append(action)
            
            # Compute minimum distance to obstacles
            if state.obstacles is not None:
                obs_positions = state.obstacles[..., :2]
                obs_radii = state.obstacles[..., 2]
                rel_pos = state.position.unsqueeze(1) - obs_positions
                distances = torch.norm(rel_pos, dim=2) - obs_radii - self.env.agent_radius
                min_dist = torch.min(distances, dim=1)[0]
                trajectory['min_distances'].append(min_dist)
            
            # Step environment
            # TODO: This dynamics update is the primary candidate for CUDA acceleration in Phase 2.
            state = self.env.dynamics(state, action)
            
            # Check termination
            if self.env.is_done(state).any():
                break
        
        return state, trajectory
    
    def train_step(self, step: int) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of metrics
        """
        # Reset environment with random initialization
        initial_state = self.env.reset(batch_size=self.batch_size, randomize=True)
        
        # Perform rollout
        final_state, trajectory = self.rollout(initial_state, self.horizon_length)
        
        # Compute loss
        loss, loss_components = self.compute_loss(trajectory, final_state)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.max_grad_norm
        )
        if self.cbf_network is not None:
            torch.nn.utils.clip_grad_norm_(
                self.cbf_network.parameters(),
                self.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Extract metrics
        metrics = {k: v.item() for k, v in loss_components.items()}
        
        # Compute success and collision rates
        goal_dist = self.env.get_goal_distance(final_state)
        success = (goal_dist < self.env.agent_radius).float().mean()
        collision = self.env.check_obstacle_collisions(final_state).float().mean()
        
        metrics['success_rate'] = success.item()
        metrics['collision_rate'] = collision.item()
        metrics['avg_goal_distance'] = goal_dist.mean().item()
        
        return metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        self.policy_network.eval()
        if self.cbf_network is not None:
            self.cbf_network.eval()
        
        total_metrics = {
            'success_rate': 0,
            'collision_rate': 0,
            'avg_goal_distance': 0,
            'avg_trajectory_length': 0
        }
        
        with torch.no_grad():
            for _ in range(num_episodes):
                state = self.env.reset(batch_size=1, randomize=True)
                trajectory_length = 0
                
                for t in range(self.env.max_steps):
                    obs = self.env.get_observation(state)
                    action = self.policy_network.get_action(obs)
                    
                    if self.cbf_network is not None:
                        state_dict = {
                            'position': state.position,
                            'velocity': state.velocity,
                            'obstacles': state.obstacles
                        }
                        safety_output = self.cbf_network(state_dict, action)
                        action = safety_output['action']
                    
                    state = self.env.dynamics(state, action)
                    trajectory_length += 1
                    
                    if self.env.is_done(state).any():
                        break
                
                # Update metrics
                goal_dist = self.env.get_goal_distance(state)
                total_metrics['success_rate'] += (goal_dist < self.env.agent_radius).float().mean().item()
                total_metrics['collision_rate'] += self.env.check_obstacle_collisions(state).float().mean().item()
                total_metrics['avg_goal_distance'] += goal_dist.mean().item()
                total_metrics['avg_trajectory_length'] += trajectory_length
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_episodes
        
        self.policy_network.train()
        if self.cbf_network is not None:
            self.cbf_network.train()
        
        return total_metrics
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.model_dir, f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save policy network
        torch.save(
            self.policy_network.state_dict(),
            os.path.join(checkpoint_dir, "policy.pt")
        )
        
        # Save CBF network if available
        if self.cbf_network is not None:
            torch.save(
                self.cbf_network.state_dict(),
                os.path.join(checkpoint_dir, "cbf.pt")
            )
        
        # Save optimizer
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt")
        )
        
        print(f"✅ Checkpoint saved at step {step}")
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting single-agent BPTT training")
        print(f"   Training steps: {self.training_steps}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Horizon length: {self.horizon_length}")
        print(f"   Device: {self.device}")
        
        progress_bar = tqdm(range(self.training_steps), desc="Training")
        
        for step in progress_bar:
            # Training step
            metrics = self.train_step(step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'success': f"{metrics['success_rate']:.2%}",
                'collision': f"{metrics['collision_rate']:.2%}"
            })
            
            # Evaluation
            if (step + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"\n📊 Step {step + 1} Evaluation:")
                print(f"   Success rate: {eval_metrics['success_rate']:.2%}")
                print(f"   Collision rate: {eval_metrics['collision_rate']:.2%}")
                print(f"   Avg goal distance: {eval_metrics['avg_goal_distance']:.3f}")
                print(f"   Avg trajectory length: {eval_metrics['avg_trajectory_length']:.1f}")
            
            # Save checkpoint
            if (step + 1) % self.save_interval == 0:
                self.save_checkpoint(step + 1)
        
        print(f"\n✨ Training completed!")
        self.save_checkpoint(self.training_steps)