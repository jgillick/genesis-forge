"""Shared fixtures for the Genesis Forge test suite.

These tests exercise pure-Python framework behavior (config item dispatch, MDP
function lifecycle) and never build a Genesis scene, so the environment is faked.
"""

from dataclasses import dataclass, field

import genesis as gs
import numpy as np
import pytest
import torch


@dataclass
class FakeEnv:
    """Stands in for a GenesisEnv wherever only the plain attributes are read."""

    num_envs: int = 4
    num_actions: int = 0
    dt: float = 0.02
    actions: object = None
    extras: dict = field(default_factory=lambda: {"episode": {}})
    extras_logging_key: str = "episode"
    episode_length: object = None
    max_episode_length: object = None

    @property
    def all_envs_idx(self) -> torch.Tensor:
        return torch.arange(self.num_envs, dtype=torch.long)


@pytest.fixture
def env() -> FakeEnv:
    return FakeEnv()


@pytest.fixture(autouse=True)
def _fake_genesis_globals(monkeypatch):
    """
    Some functions read `gs.device`/`gs.tc_float`/`gs.tc_int`/`gs.tc_bool` to place or
    type torch buffers. These are only set by a real `gs.init()` -- stub them directly
    instead, which is lighter than `gs.init()` and sufficient since nothing here
    depends on the rest of what init() sets up. `gs.device` is a real `torch.device`
    (not a plain string) since some code reads its `.type` attribute.
    """
    monkeypatch.setattr(gs, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(gs, "tc_float", torch.float32, raising=False)
    monkeypatch.setattr(gs, "tc_int", torch.int32, raising=False)
    monkeypatch.setattr(gs, "tc_bool", torch.bool, raising=False)
    # The float32 epsilon `gs.init()` would set; read by genesis.utils.geom functions
    monkeypatch.setattr(gs, "EPS", float(np.finfo(np.float32).eps), raising=False)
