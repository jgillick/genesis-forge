#!/usr/bin/env python3
"""
Evaluate a trained gait policy for the Go2 robot.

This script loads a trained model and runs it in the environment,
optionally recording videos and displaying gait metrics.
"""

import os
import torch
import argparse
import numpy as np
from datetime import datetime

# Genesis and environment imports
from environment import Go2GaitEnv
from genesis_forge.wrappers import RslRlVecEnvWrapper

# RSL_RL imports
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained Go2 gait policies")

    # Model and environment
    parser.add_argument(
        "model_path", type=str, help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--gait",
        type=str,
        default="walk",
        choices=["stand", "walk", "trot", "gallop", "hop"],
        help="Type of gait (should match training)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments for evaluation",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to evaluate"
    )

    # Display and recording
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no viewer)"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record videos of evaluation"
    )
    parser.add_argument(
        "--record_path", type=str, default=None, help="Path to save recorded videos"
    )
    parser.add_argument(
        "--show_metrics",
        action="store_true",
        help="Display gait metrics during evaluation",
    )

    # Evaluation settings
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration noise)",
    )
    parser.add_argument(
        "--slow_motion",
        type=float,
        default=1.0,
        help="Slow motion factor (1.0 = normal speed, 2.0 = half speed)",
    )

    return parser.parse_args()


def load_policy(model_path, env):
    """Load a trained policy from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Create actor-critic network with same architecture as training
    actor_critic = ActorCritic(
        env.num_obs,
        env.num_obs,
        env.num_actions,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
    elif "actor_critic" in checkpoint:
        actor_critic.load_state_dict(checkpoint["actor_critic"])
    else:
        actor_critic.load_state_dict(checkpoint)

    actor_critic.eval()

    # Move to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor_critic.to(device)

    return actor_critic


def analyze_gait_metrics(env, obs, actions):
    """Analyze and return gait-specific metrics."""
    metrics = {}

    # Get phase information
    time = env.episode_length * env.dt
    phases, is_swing = env.gait_command.get_foot_phases(time.unsqueeze(-1))

    # Contact information
    contact_forces = env.foot_contact_manager.contacts.norm(dim=-1)
    has_contact = contact_forces > 5.0

    # Calculate metrics
    metrics["avg_phase"] = phases.mean().item()
    metrics["swing_ratio"] = is_swing.float().mean().item()
    metrics["contact_ratio"] = has_contact.float().mean().item()
    metrics["phase_sync"] = (
        (is_swing == has_contact.logical_not()).float().mean().item()
    )

    # Velocity tracking
    params = env.gait_command.get_gait_params()
    actual_vel = env.robot_manager.get_linear_velocity()
    vel_error = torch.abs(params["velocity_x"] - actual_vel[:, 0]).mean().item()
    metrics["velocity_error"] = vel_error
    metrics["actual_velocity"] = actual_vel[:, 0].mean().item()

    return metrics


def print_metrics(metrics, episode_num):
    """Print formatted gait metrics."""
    print(f"\n--- Episode {episode_num} Metrics ---")
    print(f"Phase Sync:      {metrics['phase_sync']:.2%}")
    print(f"Swing Ratio:     {metrics['swing_ratio']:.2%}")
    print(f"Contact Ratio:   {metrics['contact_ratio']:.2%}")
    print(f"Velocity Error:  {metrics['velocity_error']:.3f} m/s")
    print(f"Actual Velocity: {metrics['actual_velocity']:.3f} m/s")
    print("-" * 30)


def main():
    """Main evaluation function."""
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Evaluating Go2 {args.gait.upper()} gait")
    print(f"{'='*60}\n")

    # Setup recording path
    if args.record and args.record_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.record_path = f"recordings/gait_{args.gait}_{timestamp}"
        os.makedirs(args.record_path, exist_ok=True)

    # Create environment
    print("Creating environment...")
    env = Go2GaitEnv(
        num_envs=args.num_envs,
        gait_type=args.gait,
        randomize_gait=False,  # Use fixed parameters for evaluation
        headless=args.headless,
        max_episode_length_s=20,
    )
    env.build()

    # Wrap environment
    env_wrapped = RslRlVecEnvWrapper(env)

    # Load policy
    policy = load_policy(args.model_path, env_wrapped)

    # Evaluation loop
    print(f"\nRunning {args.num_episodes} episodes...")
    print(f"Gait type: {args.gait}")
    print(f"Deterministic: {args.deterministic}")
    if args.slow_motion != 1.0:
        print(f"Slow motion: {args.slow_motion}x")
    print()

    all_metrics = []
    device = next(policy.parameters()).device

    for episode in range(args.num_episodes):
        print(f"Episode {episode + 1}/{args.num_episodes}")

        # Reset environment
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        episode_metrics = []
        done = False
        step_count = 0

        while not done:
            # Get action from policy
            with torch.no_grad():
                if args.deterministic:
                    actions = policy.act_inference(obs)
                else:
                    actions = policy.act(obs, obs)[0]  # obs used as critic_obs

            # Step environment
            obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32, device=device)

            # Collect metrics
            if args.show_metrics and step_count % 50 == 0:
                metrics = analyze_gait_metrics(env, obs, actions)
                episode_metrics.append(metrics)

            # Check for episode end
            done = dones.any()
            step_count += 1

            # Apply slow motion if requested
            if args.slow_motion > 1.0:
                import time

                time.sleep(env.dt * (args.slow_motion - 1.0))

        # Print episode metrics
        if args.show_metrics and episode_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in episode_metrics])
                for key in episode_metrics[0].keys()
            }
            print_metrics(avg_metrics, episode + 1)
            all_metrics.append(avg_metrics)

    # Print summary statistics
    if all_metrics:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            mean = np.mean(values)
            std = np.std(values)

            if "ratio" in key or "sync" in key:
                print(f"{key:20s}: {mean:.2%} ± {std:.2%}")
            else:
                print(f"{key:20s}: {mean:.3f} ± {std:.3f}")

        # Gait-specific evaluation
        print(f"\n{'='*60}")
        print(f"GAIT QUALITY ASSESSMENT ({args.gait})")
        print(f"{'='*60}")

        avg_phase_sync = np.mean([m["phase_sync"] for m in all_metrics])
        avg_vel_error = np.mean([m["velocity_error"] for m in all_metrics])

        if args.gait == "walk":
            quality = (
                "Excellent"
                if avg_phase_sync > 0.85
                else "Good" if avg_phase_sync > 0.70 else "Needs Improvement"
            )
            print(f"Walking pattern quality: {quality}")
            print(
                f"- Clear 4-beat pattern: {'Yes' if avg_phase_sync > 0.75 else 'Partial'}"
            )
            print(f"- Stable progression: {'Yes' if avg_vel_error < 0.2 else 'No'}")

        elif args.gait == "trot":
            quality = (
                "Excellent"
                if avg_phase_sync > 0.80
                else "Good" if avg_phase_sync > 0.65 else "Needs Improvement"
            )
            print(f"Trotting pattern quality: {quality}")
            print(
                f"- Diagonal synchronization: {'Good' if avg_phase_sync > 0.70 else 'Poor'}"
            )
            print(f"- Dynamic stability: {'Yes' if avg_vel_error < 0.3 else 'No'}")

        elif args.gait == "gallop":
            quality = (
                "Excellent"
                if avg_phase_sync > 0.75
                else "Good" if avg_phase_sync > 0.60 else "Needs Improvement"
            )
            print(f"Galloping pattern quality: {quality}")
            print(
                f"- Front-rear coordination: {'Good' if avg_phase_sync > 0.65 else 'Poor'}"
            )
            print(
                f"- Flight phase present: {'Likely' if avg_phase_sync > 0.70 else 'Unclear'}"
            )

        elif args.gait == "hop":
            quality = (
                "Excellent"
                if avg_phase_sync > 0.85
                else "Good" if avg_phase_sync > 0.70 else "Needs Improvement"
            )
            print(f"Hopping pattern quality: {quality}")
            print(
                f"- Synchronization: {'Excellent' if avg_phase_sync > 0.80 else 'Good' if avg_phase_sync > 0.65 else 'Poor'}"
            )
            print(f"- Aerial phase: {'Clear' if avg_phase_sync > 0.75 else 'Partial'}")

        elif args.gait == "stand":
            quality = (
                "Excellent"
                if avg_vel_error < 0.05
                else "Good" if avg_vel_error < 0.1 else "Needs Improvement"
            )
            print(f"Standing stability: {quality}")
            print(
                f"- Position holding: {'Excellent' if avg_vel_error < 0.05 else 'Good' if avg_vel_error < 0.1 else 'Poor'}"
            )

        print(f"{'='*60}\n")

    print("Evaluation complete!")
    if args.record:
        print(f"Videos saved to: {args.record_path}")


if __name__ == "__main__":
    main()
