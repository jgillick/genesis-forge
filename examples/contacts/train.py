import os
import copy
import torch
import shutil
import pickle
import argparse
import genesis as gs

from genesis_forge.wrappers import (
    VideoWrapper,
    RslRlWrapper,
)
from environment import Go2CommandDirectionEnv
from rsl_rl.runners import OnPolicyRunner

EXPERIMENT_NAME = "go2-foot-step"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def training_cfg():
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
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
        },
        "seed": 1,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    }


def train(cfg: dict, num_envs: int, log_dir: str, max_iterations: int):
    """
    Train the agent.
    """

    #  Create environment
    env = Go2CommandDirectionEnv(num_envs=num_envs, headless=True)

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_dir, "videos"),
        episode_trigger=lambda episode_id: episode_id % 2 == 0,
    )

    # Build the environment
    env = RslRlWrapper(env)
    env.build()
    env.reset()

    # Setup training runner and train
    print("💪 Training model...")
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir, device=gs.device)
    runner.add_git_repo_to_log(".")
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=False)
    env.close()


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Logging directory
    log_base_dir = "./logs"
    experiment_name = args.exp_name
    log_path = os.path.join(log_base_dir, experiment_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Load training configuration
    cfg = training_cfg()

    # Save config snapshot
    pickle.dump(
        [cfg],
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    # Train agent
    train(cfg, args.num_envs, log_path, args.max_iterations)


if __name__ == "__main__":
    main()
