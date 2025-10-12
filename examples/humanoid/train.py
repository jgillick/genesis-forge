import os
import copy
import torch
import shutil
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import (
    VideoWrapper,
    RslRlWrapper,
)
from environment import BerkeleyHumanoidEnv

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib").startswith("1."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install install 'rsl-rl-lib>=2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

EXPERIMENT_NAME = "berkeley-humanoid"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=220)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def training_cfg(exp_name: str, max_iterations: int):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    }


def main():
    # Initialize Genesis
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    # Logging directory
    log_base_dir = "./logs"
    experiment_name = args.exp_name
    log_path = os.path.join(log_base_dir, experiment_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Load training configuration and save snapshot of training configs
    cfg = training_cfg(experiment_name, args.max_iterations)
    pickle.dump(
        [cfg],
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    # Create environment
    env = BerkeleyHumanoidEnv(num_envs=args.num_envs, headless=True)

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 2 == 0,
    )

    # Build the environment
    env = RslRlWrapper(env)
    env.build()
    env.reset()

    # Train
    print("💪 Training model...")
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_path, device=gs.device)
    runner.git_status_repos = ["."]
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=False
    )
    env.close()


if __name__ == "__main__":
    main()
