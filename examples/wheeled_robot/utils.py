import pickle
import sys
from pathlib import Path


def get_latest_model(log_dir: str) -> str:
    """
    Get the last model checkpoint from the log directory
    """
    checkpoints = list(Path(log_dir).glob("model_*.pt"))
    if not checkpoints:
        print(f"Error: No model files found at '{log_dir}'.")
        sys.exit(1)
    # Sort by the file with the highest number
    latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    return str(latest)


def save_config_pickle(log_dir: str, cfg) -> None:
    """
    Save the training configuration to the log directory
    """
    cfg_path = Path(log_dir) / "cfgs.pkl"
    with open(cfg_path, "wb") as f:
        pickle.dump([cfg], f)


def load_config_pickle(log_dir: str):
    """
    Load the training configuration from the log directory
    """
    cfg_path = Path(log_dir) / "cfgs.pkl"
    with open(cfg_path, "rb") as f:
        [cfg] = pickle.load(f)
    return cfg
