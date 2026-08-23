"""Shared fixtures for the Genesis Forge test suite.

These tests exercise pure-Python framework behavior (config item dispatch, MDP
function lifecycle) and never build a Genesis scene, so the environment is faked.
"""

from dataclasses import dataclass, field

import pytest


@dataclass
class FakeEnv:
    """Stands in for a GenesisEnv wherever only the plain attributes are read."""

    num_envs: int = 4
    dt: float = 0.02
    actions: object = None
    extras: dict = field(default_factory=dict)
    episode_length: object = None
    max_episode_length: object = None


@pytest.fixture
def env() -> FakeEnv:
    return FakeEnv()
