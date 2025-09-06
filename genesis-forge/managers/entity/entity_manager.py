import torch
import inspect
import genesis as gs
from typing import Any

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis.engine.entities import RigidEntity

from .config import EntityResetConfig, ResetConfigFnClass


class EntityManager(BaseManager):
    """
    Provides options for resetting an entity and adding noise and randomization to its state.

    Example::
        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.entity_manager = EntityManager(
                    self,
                    entity_attr="robot",
                    on_reset={
                        "zero_dof": {
                            "fn": reset.zero_all_dofs_velocity,
                        },
                        "rotation": {
                            "fn": reset.set_rotation,
                            "params": {
                                "z": (0, 2 * math.pi),
                            },
                        },
                        "position": {
                            "fn": reset.randomize_terrain_position,
                            "params": {
                                "terrain_manager": self.terrain_manager,
                                "subterrain": self._target_terrain,
                                "height_offset": 0.15,
                            },
                        },
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        entity_attr: str,
        on_reset: dict[str, EntityResetConfig],
    ):
        """
        Initialize the entity manager.

        Args:
            env: The environment instance.
            entity_attr: The attribute name of the environment that the entity is stored in.
        """
        super().__init__(env, type="entity")
        if hasattr(env, "add_entity_manager"):
            env.add_entity_manager(self)

        self.entity: RigidEntity | None = None
        self.on_reset = on_reset
        self._entity_attr = entity_attr

    """
    Operations.
    """

    def build(self):
        """
        Build the entity manager.
        """
        self.entity = getattr(self.env, self._entity_attr)

        # Initialize reset function classes
        for cfg in self.on_reset.values():
            if inspect.isclass(cfg["fn"]):
                self._init_fn_class(cfg)

    def reset(self, envs_idx: list[int] | None = None):
        """
        Call all reset functions
        """
        if not self.enabled:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.env.num_envs, device=gs.device)

        for cfg in self.on_reset.values():
            params = cfg.get("params", {}) or {}

            # Initialize the function class
            if inspect.isclass(cfg["fn"]):
                self._init_fn_class(cfg)

            cfg["fn"](self.env, self.entity, envs_idx, **params)
        return

    """
    Implementation
    """

    def _init_fn_class(self, cfg: EntityResetConfig):
        """Initialize a reset function class"""
        params = cfg.get("params", {}) or {}
        initialized = cfg.get("_initialized", False)
        if initialized:
            return cfg

        cfg["fn"] = cfg["fn"](self.env, self.entity, **params)

        # Clear params so that they cannot be changed after initialization
        cfg["params"] = None

        cfg["_initialized"] = True
        return cfg
