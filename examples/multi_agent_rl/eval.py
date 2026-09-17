"""Evaluate a trained SKRL MAPPO policy (MASQ-style per-leg agents)."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import genesis as gs
import torch
from env_wrapper import SkrlMasqWrapper
from environment import Go2MasqLocomotionEnv
from models import MasqGaussianPolicy, MasqValue
from skrl.multi_agents.torch.mappo import MAPPO

EXPERIMENT_NAME = "go2-multi-agent"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a MAPPO checkpoint file (*.pt)")
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def get_latest_model(log_dir: str) -> str | None:
    best = os.path.join(log_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best):
        return best
    model_checkpoints = glob.glob(os.path.join(log_dir, "agent_*.pt"))
    if len(model_checkpoints) == 0:
        print(
            f"Warning: No model files found at '{log_dir}' (you might need to train more)."
        )
        sys.exit(1)
    # Sort by the file with the highest number
    sorted_models = sorted(
        model_checkpoints,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    return sorted_models[-1]


def main() -> None:
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    log_path = f"./logs/{args.exp_name}"
    model = args.checkpoint or get_latest_model(log_path)
    with open(f"{log_path}/cfgs.pkl", "rb") as f:
        [cfg] = pickle.load(f)
    print(f"Loading checkpoint: {model}")

    # Setup environment
    core_env = Go2MasqLocomotionEnv(num_envs=1, headless=False)
    env = core_env
    env.build()
    env.reset()

    # Setup SKRL
    wrapped = SkrlMasqWrapper(env)
    agents = list(wrapped.possible_agents)

    obs_space = wrapped.observation_spaces[agents[0]]
    action_space = wrapped.action_spaces[agents[0]]
    state_space = wrapped.state_spaces[agents[0]]

    models: dict[str, dict] = {}
    for uid in agents:
        models[uid] = {
            "policy": MasqGaussianPolicy(obs_space, action_space, clip_actions=False),
            "value": MasqValue(state_space),
        }

    # cfg here only needs to be compatible with the checkpoint modules; learning params won't be used.
    # def for_each_agent(v):
    #     return {uid: v for uid in agents}

    # cfg = MAPPO_CFG(
    #     rollouts=1,
    #     learning_epochs=1,
    #     mini_batches=1,
    #     random_timesteps=0,
    #     learning_starts=0,
    #     learning_rate_scheduler_kwargs=for_each_agent({}),
    #     observation_preprocessor_kwargs=for_each_agent({}),
    #     state_preprocessor_kwargs=for_each_agent({}),
    #     value_preprocessor_kwargs=for_each_agent({}),
    #     experiment=ExperimentCfg(
    #         directory="./logs",
    #         experiment_name=EXPERIMENT_NAME,
    #         write_interval=-1,
    #         checkpoint_interval=-1,
    #     ),
    # )

    agent = MAPPO(
        possible_agents=agents,
        models=models,
        memories=None,
        observation_spaces=wrapped.observation_spaces,
        state_spaces=wrapped.state_spaces,
        action_spaces=wrapped.action_spaces,
        cfg=cfg,
    )
    agent.load(model)
    agent.enable_training_mode(False, apply_to_models=True)

    obs, _info = wrapped.reset()
    cumulative = torch.zeros((1,), dtype=torch.float32, device=gs.device)

    try:
        steps = 1000
        for t in range(steps):
            states = wrapped.state()
            actions, _ = agent.act(obs, states, timestep=t, timesteps=steps)
            obs, rewards, _terminated, _truncated, _info = wrapped.step(actions)
            # team reward is duplicated per agent; take first agent's reward
            r0 = rewards[agents[0]].view(-1)
            cumulative += r0
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if str(e) != "Viewer closed.":
            raise
    except Exception:
        raise
    wrapped.close()


if __name__ == "__main__":
    main()

