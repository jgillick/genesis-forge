"""Fixtures for deployment export tests.

Builds real managers against a stand-in environment, so the parity harness compares
the actual numpy runtime against the actual torch manager code -- no Genesis scene,
no GPU. The stand-in implements only what deployment export reads from a
ManagedEnvironment: manager registration, action slices, num_envs, and dt.
"""

import pytest
import torch

from genesis_forge.managers import ObservationManager, PositionActionManager


class FakeActuatorManager:
    """Minimal actuator manager: DOF indices deliberately differ from column order."""

    def __init__(self, num_envs=4, default_pos=None):
        self.dofs = {"FL_hip": 100, "FL_knee": 101, "FR_hip": 102}
        self._idx_to_col = {idx: col for col, idx in enumerate(self.dofs.values())}
        self.default_dofs_pos = (
            default_pos
            if default_pos is not None
            else torch.tensor([[0.1, 0.2, 0.3]] * num_envs)
        )
        self._lower = torch.tensor([-1.0, -1.5, -2.0])
        self._upper = torch.tensor([1.0, 1.5, 2.0])

    def _cols(self, dofs_idx):
        return [self._idx_to_col[i] for i in dofs_idx]

    def get_dofs_limits(self, dofs_idx):
        cols = self._cols(dofs_idx)
        return self._lower[cols], self._upper[cols]

    def get_deployment_values(self):
        return {
            "joint_names": list(self.dofs.keys()),
            "values": {
                "kp": [50.0, 50.0, 50.0],
                "kv": [0.5, 0.5, 0.5],
                "default_pos": [0.1, 0.2, 0.3],
            },
            "randomized": [],
        }


class FakeManagedEnv:
    """Stands in for ManagedEnvironment wherever deployment export reads from it."""

    def __init__(self, num_envs=4, dt=0.02):
        self.num_envs = num_envs
        self.dt = dt
        self.actions = None
        self.extras = {"episode": {}}
        self.extras_logging_key = "episode"
        self.managers = {
            "action": [],
            "actuator": [],
            "observation": [],
            "command": [],
            "contact": [],
            "entity": [],
            "terrain": [],
        }
        self._action_ranges = []

    def add_manager(self, manager_type, manager):
        self.managers.setdefault(manager_type, []).append(manager)

    @property
    def action_ranges(self):
        return list(self._action_ranges)

    def build(self):
        """Mirrors ManagedEnvironment.build's ordering: actions, then observations."""
        size = 0
        self._action_ranges = []
        for manager in self.managers["action"]:
            manager.build()
            start = size
            size += manager.action_space.shape[0]
            self._action_ranges.append((start, size))
        for manager in self.managers["observation"]:
            manager.build()
        return self


def observation_cfg(num_dofs=3):
    return {
        "gyro": {
            "fn": lambda env: torch.ones((env.num_envs, 3)) * 0.5,
            "scale": 0.25,
            "description": "Body-frame angular velocity",
            "units": "rad/s",
        },
        "dof_pos": {
            "fn": lambda env: torch.ones((env.num_envs, num_dofs)) * 0.2,
        },
    }


@pytest.fixture
def deployable_env():
    """A built environment with one action manager and one observation manager."""
    env = FakeManagedEnv()
    env.actuator_manager = FakeActuatorManager(num_envs=env.num_envs)
    env.managers["actuator"].append(env.actuator_manager)
    env.action_manager = PositionActionManager(
        env, actuator_manager=env.actuator_manager, scale=0.5
    )
    env.observation_manager = ObservationManager(env, cfg=observation_cfg())
    return env.build()


@pytest.fixture
def make_env():
    """Factory for environments with custom observation configs / history."""

    def _build(cfg=None, history_len=None, num_envs=4, dt=0.02):
        env = FakeManagedEnv(num_envs=num_envs, dt=dt)
        env.actuator_manager = FakeActuatorManager(num_envs=num_envs)
        env.managers["actuator"].append(env.actuator_manager)
        env.action_manager = PositionActionManager(
            env, actuator_manager=env.actuator_manager, scale=0.5
        )
        env.observation_manager = ObservationManager(
            env,
            cfg=cfg if cfg is not None else observation_cfg(),
            history_len=history_len,
        )
        return env.build()

    return _build
