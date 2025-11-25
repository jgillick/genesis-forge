import os
import glob
import torch
import pickle
import argparse
import genesis as gs
from skrl.utils.runner.torch import Runner

from genesis_forge.wrappers import SkrlEnvWapper
from environment import Go2SimpleEnv

EXPERIMENT_NAME = "go2-simple-skrl"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def get_latest_model(log_dir: str) -> str:
    """
    Get the latest model checkpoint from the log directory
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


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    log_path = f"./logs/{args.exp_name}"
    [cfg] = pickle.load(open(f"{log_path}/cfgs.pkl", "rb"))
    model = get_latest_model(log_path)

    # Setup environment
    env = Go2SimpleEnv(num_envs=1, headless=False)
    env = SkrlEnvWapper(env)
    env.build()

    # Eval
    print(f"🎬 Loading last model: {model}")
    runner = Runner(env, cfg)
    runner.agent.load(model)
    runner.agent.set_running_mode("eval")

    try:
        observation, _infos = env.reset()
        timestep = 0
        while True:
            timestep += 1
            # Get actions from agent
            (actions, _prob, outputs) = runner.agent.act(
                observation, timestep=timestep, timesteps=0
            )
            actions = outputs.get("mean_actions", actions)

            # Perform step
            observation, _rewards, terminated, truncated, _infos = env.step(actions)
            env.render()

            # Check for termination/truncation
            if terminated.any() or truncated.any():
                observation, _infos = env.reset()
                timestep = 0
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if e.message != "Viewer closed.":
            raise e
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
