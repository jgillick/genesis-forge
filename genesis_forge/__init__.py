"""
Genesis Forge — a modular robotics reinforcement learning framework built on
the Genesis physics simulator. Provides a modular architecture for
constructing Gymnasium-compatible parallel RL environments.

Typical usage::

    from genesis_forge import ManagedEnvironment
    from genesis_forge import managers, mdp

Key exports:
- :class:`GenesisEnv` — base environment class; subclass for full manual control
- :class:`ManagedEnvironment` — config-driven environment; override ``config()``
  to register managers
- :data:`EnvMode` — literal type ``"train" | "eval" | "play"``
"""
from .genesis_env import GenesisEnv, EnvMode
from .managed_env import ManagedEnvironment

__all__ = [
    "GenesisEnv",
    "ManagedEnvironment",
    "EnvMode",
]
