import os
import glob
import pickle
import yaml
import subprocess
from datetime import datetime

from skrl.utils.runner.torch import Runner


def load_training_config(
    yaml_path: str, max_iterations: int = None, log_base_dir: str | None = "./logs"
) -> tuple[dict, str]:
    """
    Load the training configuration from the yaml file.

    Args:
        yaml_path: The path to the yaml file.
        max_iterations: The maximum number of iterations.
        log_base_dir: The base directory for the logging directory.

    Returns:
        A tuple containing the training configuration and the logging directory path.
    """
    cfg = Runner.load_cfg_from_yaml(yaml_path)

    # Logging directory
    if log_base_dir is None:
        log_base_dir = "./logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{cfg['agent']['class']}"
    log_path = os.path.join(log_base_dir, experiment_name)

    # Update configuration
    cfg["agent"]["experiment"]["directory"] = log_base_dir
    cfg["agent"]["experiment"]["experiment_name"] = experiment_name
    if max_iterations:
        cfg["trainer"]["timesteps"] = max_iterations * cfg["agent"]["rollouts"]

    return cfg, log_path


def get_latest_checkpoint(log_dir: str) -> str:
    """
    Get the latest checkpoint from the log directory
    """
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    # Best checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_agent.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "agent_*.pt"))
    if len(checkpoint_files) == 0:
        print(
            f"Warning: No checkpoint files found at '{checkpoint_dir}' (you might need to train more)."
        )
        return None
    checkpoint_files.sort()
    return checkpoint_files[-1]


def save_env_snapshots(log_dir: str, cfg: dict):
    """
    Save the environment snapshots to the logging directory.

    Args:
        log_dir: The path to the logging directory.
        cfg: The training configuration.
    """
    snapshot_dir = os.path.join(log_dir, "snapshot")
    os.makedirs(snapshot_dir, exist_ok=False)

    # Training config
    pickle.dump([cfg], open(f"{log_dir}/snapshot/cfg.pkl", "wb"))

    # Git diff
    try:
        result = subprocess.run(
            ["git", "diff"], capture_output=True, text=True, check=True
        )
        diff_file = os.path.join(log_dir, "snapshot/git.patch")
        if result.stdout != "":
            with open(diff_file, "w") as file:
                file.write(result.stdout)
    except:
        pass
