"""
Manager classes for Genesis Forge environments.

Each manager owns one aspect of the RL loop and self-registers with the
environment on construction (via ``super().__init__(env, type="...")``).
Managers are created inside :meth:`ManagedEnvironment.config` and their
``build()``, ``step()``, and ``reset(envs_idx)`` hooks are called
automatically by the environment.

Available managers:
- :class:`RewardManager` — weighted reward components
- :class:`TerminationManager` — episode termination/truncation conditions
- :class:`ObservationManager` — observation space definition
- :class:`PositionActionManager` / :class:`PositionWithinLimitsActionManager` — joint position actions
- :class:`ActuatorManager` — PD controller gains and default joint positions
- :class:`EntityManager` — entity spawning and per-episode resets
- :class:`ContactManager` — per-link contact force tracking
- :class:`TerrainManager` — terrain height queries and bounds
- :class:`CommandManager` / :class:`VelocityCommandManager` — sampled command signals
"""
from .base import BaseManager
from .reward_manager import RewardManager
from .termination_manager import TerminationManager
from .action.position_action_manager import PositionActionManager
from .action.position_within_limits import PositionWithinLimitsActionManager
from .command import (
    CommandManager,
    PositionCommandManager,
    PoseCommandManager,
    VelocityCommandManager,
)
from .contact import ContactManager
from .terrain_manager import TerrainManager
from .entity_manager import EntityManager
from .observation_manager import ObservationManager
from .actuator import ActuatorManager
from .config import (
    MdpFnClass,
    ResetMdpFnClass,
)

__all__ = [
    "BaseManager",
    "RewardManager",
    "TerminationManager",
    "CommandManager",
    "PositionCommandManager",
    "PoseCommandManager",
    "VelocityCommandManager",
    "PositionActionManager",
    "PositionWithinLimitsActionManager",
    "ContactManager",
    "TerrainManager",
    "EntityManager",
    "ObservationManager",
    "MdpFnClass",
    "ResetMdpFnClass",
    "ActuatorManager",
]
