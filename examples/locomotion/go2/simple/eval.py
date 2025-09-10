import os
import glob
import torch
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import RslRlWrapper
from environment import Go2Env

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if not metadata.version("rsl-rl-lib").startswith("2."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install install 'rsl-rl-lib>=2.3.3'.") from e
from rsl_rl.runners import OnPolicyRunner


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
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
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_latest_model(log_dir: str) -> str:
    """
    Get the last model from the log directory
    """
    model_checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if len(model_checkpoints) == 0:
        print(
            f"Warning: No model files found at '{log_dir}' (you might need to train more)."
        )
        exit(1)
    model_checkpoints.sort()
    return model_checkpoints[-1]


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Logging directory
    log_path = "./logs/go2-walking"

    # Load training configuration
    [cfg] = pickle.load(open(f"{log_path}/cfgs.pkl", "rb"))

    # Setup environment
    env = Go2Env(num_envs=1, headless=False)
    env = RslRlWrapper(env)
    env.build()

    # Eval
    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    resume_path = get_latest_model(log_path)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _rews, _dones, _infos = env.step(actions)


if __name__ == "__main__":
    main()
