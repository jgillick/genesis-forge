"""Shared fixtures for the Genesis Forge test suite.

These tests exercise pure-Python framework behavior (config item dispatch, MDP
function lifecycle) and never build a Genesis scene, so the environment is faked.
"""

from dataclasses import dataclass, field

import pytest
import torch
import genesis as gs


@dataclass
class FakeEnv:
    """Stands in for a GenesisEnv wherever only the plain attributes are read."""

    num_envs: int = 4
    dt: float = 0.02
    actions: object = None
    extras: dict = field(default_factory=lambda: {"episode": {}})
    extras_logging_key: str = "episode"
    episode_length: object = None
    max_episode_length: object = None


@pytest.fixture
def env() -> FakeEnv:
    return FakeEnv()


@pytest.fixture(autouse=True)
def _fake_genesis_globals(monkeypatch):
    """
    Some functions read `gs.device`/`gs.tc_float` to place or type torch buffers.
    Both are only set by a real `gs.init()` -- stub them directly instead, which is
    lighter than `gs.init()` and sufficient since nothing here depends on the rest
    of what init() sets up.
    """
    monkeypatch.setattr(gs, "device", "cpu", raising=False)
    monkeypatch.setattr(gs, "tc_float", torch.float32, raising=False)
