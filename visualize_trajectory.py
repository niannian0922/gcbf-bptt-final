#!/usr/bin/env python3
"""
Trajectory Visualization Script for GCBF+BPTT Single Agent

This script provides detailed qualitative analysis of single runs,
generating comprehensive plots for trajectory and safety analysis.
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Import our project modules
from gcbfplus.env.single_double_integrator import SingleDoubleIntegratorEnv
from gcbfplus.trainer.bptt_trainer import SimpleMLPPolicy as BPTTPolicy


class TrajectoryVisualizer:
    """Comprehensive trajectory visualizer for qualitative analysis."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the trajectory visualizer.
        
        Args:
            config: Configuration dictionary
            device: Device to run visualization on
        """
        self.config = config
        self.device = device
        
        # Extract evaluation configuration
        eval_config = config.get('evaluation', {})
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
        
        # Trajectory storage
        self.trajectory_data = {
            'positions': [],
            'velocities': [],
            'actions': [],
            'times': [],
            'goal_distances': [],
            'obstacle_distances': [],
            'alpha_values': [],
            'dynamic_margins': []
        }
        
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
    
    def _extract_state_components(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract position and velocity components from state.
        
        Args:
            state: Environment state
            
        Returns:
            Tuple of (positions, velocities)
        """
        if hasattr(state, 'positions'):
            positions = state.positions
            velocities = state.velocities
        elif hasattr(state, 'state'):
            # Single agent state: [x, y, vx, vy]
            state_tensor = state.state
            positions = state_tensor[:, :2]  # x, y
            velocities = state_tensor[:, 2:]  # vx, vy
        else:
            # Fallback: assume state is a tensor
            state_tensor = state
            positions = state_tensor[:, :2]
            velocities = state_tensor[:, 2:]
        
        return positions, velocities
    
    def run_single_episode(self, seed: int = None) -> bool:
        """
        Run a single episode and collect trajectory data.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            True if episode completed successfully, False otherwise
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Reset environment
        state = self.env.reset()
        episode_length = 0
        
        # Clear previous trajectory data
        for key in self.trajectory_data:
            self.trajectory_data[key].clear()
        
        # Episode loop
        while episode_length < self.env.max_steps:
            # Get observation
            observations = self.env.get_observation(state)
            if hasattr(observations, 'to'):
                observations = observations.to(self.device)
            
            # Get action from policy (no gradients needed for visualization)
            with torch.no_grad():
                actions = self.policy(observations)
                alpha = None
                dynamic_margins = None
            
            # Extract state components
            positions, velocities = self._extract_state_components(state)
            
            # Store trajectory data
            self.trajectory_data['positions'].append(positions.detach().cpu())
            self.trajectory_data['velocities'].append(velocities.detach().cpu())
            self.trajectory_data['actions'].append(actions.detach().cpu())
            self.trajectory_data['times'].append(episode_length * self.env.dt)
            
            # Store goal distance
            goal_distance = self.env.get_goal_distance(state)
            self.trajectory_data['goal_distances'].append(goal_distance.detach().cpu())
            
            # Store obstacle distances
            if hasattr(self.env, 'get_obstacle_distances'):
                obstacle_distances = self.env.get_obstacle_distances(state)
                if obstacle_distances is not None:
                    self.trajectory_data['obstacle_distances'].append(obstacle_distances.detach().cpu())
                else:
                    self.trajectory_data['obstacle_distances'].append(torch.tensor([float('inf')]))
            else:
                self.trajectory_data['obstacle_distances'].append(torch.tensor([float('inf')]))
            
            # Store alpha and dynamic margins
            if alpha is not None:
                self.trajectory_data['alpha_values'].append(alpha.detach().cpu())
            else:
                self.trajectory_data['alpha_values'].append(torch.tensor([1.0]))
            
            if dynamic_margins is not None:
                self.trajectory_data['dynamic_margins'].append(dynamic_margins.detach().cpu())
            else:
                self.trajectory_data['dynamic_margins'].append(torch.tensor([0.1]))
            
            # Environment step
            step_result = self.env.step(state, actions, None)
            next_state = step_result.next_state
            
            # Check for collision
            if step_result.cost > 0:
                print(f"Episode ended with collision at step {episode_length}")
                return False
            
            # Check for goal reaching
            goal_distance = self.env.get_goal_distance(next_state)
            if goal_distance < 0.1:  # Success threshold
                print(f"Episode completed successfully at step {episode_length}")
                return True
            
            # Update state
            state = next_state
            episode_length += 1
        
        print(f"Episode timed out after {episode_length} steps")
        return False
    
    def _get_obstacle_positions(self) -> List[Tuple[float, float]]:
        """Get obstacle positions for visualization."""
        obstacles_config = self.config.get('env', {}).get('obstacles', {})
        
        if not obstacles_config.get('enabled', False):
            return []
        
        # Static obstacles
        positions = obstacles_config.get('positions', [])
        if positions:
            return positions
        
        # Random obstacles (if any were generated)
        # This is a simplified approach - in practice, you might want to store
        # the actual obstacle positions from the environment
        return []
    
    def _get_goal_position(self) -> Tuple[float, float]:
        """Get goal position for visualization."""
        # This is a simplified approach - in practice, you might want to get
        # the actual goal position from the environment state
        return (0.0, 0.0)  # Default goal at origin
    
    def create_comprehensive_plot(self, save_path: str = None):
        """
        Create comprehensive visualization with trajectory and safety plots.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.trajectory_data['positions']:
            print("No trajectory data available. Run an episode first.")
            return
        
        # Convert trajectory data to numpy arrays
        positions = torch.stack(self.trajectory_data['positions']).numpy()
        times = np.array(self.trajectory_data['times'])
        goal_distances = torch.stack(self.trajectory_data['goal_distances']).numpy()
        obstacle_distances = torch.stack(self.trajectory_data['obstacle_distances']).numpy()
        alpha_values = torch.stack(self.trajectory_data['alpha_values']).numpy()
        
        # Extract x, y coordinates
        x_pos = positions[:, 0, 0]  # First agent, x coordinate
        y_pos = positions[:, 0, 1]  # First agent, y coordinate
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 3D Trajectory Plot (X vs Y vs Time)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot3D(x_pos, y_pos, times, 'b-', linewidth=2, label='Drone Trajectory')
        ax1.scatter(x_pos[0], y_pos[0], times[0], c='g', s=100, marker='o', label='Start')
        ax1.scatter(x_pos[-1], y_pos[-1], times[-1], c='r', s=100, marker='*', label='End')
        
        # Add goal position
        goal_x, goal_y = self._get_goal_position()
        ax1.scatter(goal_x, goal_y, 0, c='orange', s=150, marker='s', label='Goal')
        
        # Add obstacles
        obstacle_positions = self._get_obstacle_positions()
        for obs_x, obs_y in obstacle_positions:
            ax1.scatter(obs_x, obs_y, 0, c='black', s=100, marker='x', label='Obstacle')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Time (s)')
        ax1.set_title('3D Trajectory Visualization')
        ax1.legend()
        ax1.grid(True)
        
        # 2D Top-down Trajectory
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(x_pos, y_pos, 'b-', linewidth=2, label='Drone Path')
        ax2.scatter(x_pos[0], y_pos[0], c='g', s=100, marker='o', label='Start')
        ax2.scatter(x_pos[-1], y_pos[-1], c='r', s=100, marker='*', label='End')
        ax2.scatter(goal_x, goal_y, c='orange', s=150, marker='s', label='Goal')
        
        # Add obstacles
        for obs_x, obs_y in obstacle_positions:
            ax2.scatter(obs_x, obs_y, c='black', s=100, marker='x', label='Obstacle')
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('2D Top-down Trajectory')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Safety Distance Plot
        ax3 = fig.add_subplot(2, 2, 3)
        min_obstacle_distances = np.min(obstacle_distances, axis=1)
        safety_margin = self.config.get('gcbf', {}).get('safety_margin', 0.1)
        
        ax3.plot(times, min_obstacle_distances, 'b-', linewidth=2, label='Min Obstacle Distance')
        ax3.axhline(y=safety_margin, color='r', linestyle='--', label=f'Safety Margin ({safety_margin}m)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Collision Threshold')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Distance (m)')
        ax3.set_title('Safety Distance Analysis')
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim(bottom=0)
        
        # Goal Distance and Alpha Plot
        ax4 = fig.add_subplot(2, 2, 4)
        ax4_twin = ax4.twinx()
        
        # Goal distance (left y-axis)
        line1 = ax4.plot(times, goal_distances[:, 0], 'g-', linewidth=2, label='Goal Distance')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Goal Distance (m)', color='g')
        ax4.tick_params(axis='y', labelcolor='g')
        
        # Alpha values (right y-axis)
        line2 = ax4_twin.plot(times, alpha_values[:, 0], 'orange', linewidth=2, label='Dynamic Alpha')
        ax4_twin.set_ylabel('Alpha Value', color='orange')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        
        ax4.set_title('Goal Distance and Dynamic Alpha')
        ax4.grid(True)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        # Adjust layout and add title
        plt.tight_layout()
        fig.suptitle(f'GCBF+BPTT Trajectory Analysis - Run: {self.run_name}, Step: {self.model_step}', 
                     fontsize=16, y=0.98)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Display the plot
        plt.show()
    
    def print_trajectory_summary(self):
        """Print summary statistics of the trajectory."""
        if not self.trajectory_data['positions']:
            print("No trajectory data available.")
            return
        
        positions = torch.stack(self.trajectory_data['positions']).numpy()
        times = np.array(self.trajectory_data['times'])
        goal_distances = torch.stack(self.trajectory_data['goal_distances']).numpy()
        obstacle_distances = torch.stack(self.trajectory_data['obstacle_distances']).numpy()
        alpha_values = torch.stack(self.trajectory_data['alpha_values']).numpy()
        
        # Calculate statistics
        total_distance = np.sum(np.sqrt(np.diff(positions[:, 0, 0])**2 + np.diff(positions[:, 0, 1])**2))
        avg_speed = total_distance / times[-1] if times[-1] > 0 else 0
        min_obstacle_distance = np.min(obstacle_distances)
        final_goal_distance = goal_distances[-1, 0]
        avg_alpha = np.mean(alpha_values)
        
        print("\n" + "="*60)
        print("                    TRAJECTORY SUMMARY")
        print("="*60)
        print(f"Episode Duration:     {times[-1]:.3f} seconds")
        print(f"Total Distance:       {total_distance:.3f} meters")
        print(f"Average Speed:        {avg_speed:.3f} m/s")
        print(f"Final Goal Distance:  {final_goal_distance:.4f} meters")
        print(f"Min Obstacle Distance: {min_obstacle_distance:.4f} meters")
        print(f"Average Alpha:        {avg_alpha:.4f}")
        print(f"Trajectory Points:    {len(positions)}")
        print("="*60)


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize GCBF+BPTT trajectory")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--model_step", type=int, default=10000, help="Model checkpoint step to load")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None, help="Path to save the visualization figure")
    
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
        # Initialize visualizer
        visualizer = TrajectoryVisualizer(config, device)
        
        # Run single episode
        print(f"Running single episode with seed: {args.seed}")
        success = visualizer.run_single_episode(seed=args.seed)
        
        # Print trajectory summary
        visualizer.print_trajectory_summary()
        
        # Create and display visualization
        print("Generating comprehensive visualization...")
        visualizer.create_comprehensive_plot(save_path=args.save)
        
        print("\n✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Visualization failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
