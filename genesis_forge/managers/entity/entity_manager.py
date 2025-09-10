import torch
import inspect
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import (
    transform_by_quat,
    inv_quat,
)

from .config import EntityResetConfig


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

        # Buffers
        self._global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        ).repeat(env.num_envs, 1)
        self._base_pos = torch.zeros(
            (env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self._base_quat = torch.zeros(
            (env.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self._inv_base_quat = torch.zeros_like(self._base_quat)

    """
    Properties
    """

    @property
    def base_pos(self) -> torch.Tensor:
        """
        The position of the entities base link.
        """
        return self._base_pos

    @property
    def base_quat(self) -> torch.Tensor:
        """
        The quaternion of the entity's base link.
        """
        return self._base_quat

    @property
    def inv_base_quat(self) -> torch.Tensor:
        """
        The inverse of the entity's base link quaternion.
        """
        return self._inv_base_quat

    """
    Helpers
    """

    def get_projected_gravity(self) -> torch.Tensor:
        """
        The projected gravity of the entity's base link, in the entity's local frame.
        """
        return transform_by_quat(self._global_gravity, self._inv_base_quat)

    def get_linear_velocity(self) -> torch.Tensor:
        """
        The linear velocity of the entity's base link, in the entity's local frame.
        """
        return transform_by_quat(self.entity.get_vel(), self._inv_base_quat)

    def get_angular_velocity(self) -> torch.Tensor:
        """
        The angular velocity of the entity's base link, in the entity's local frame.
        """
        return transform_by_quat(self.entity.get_ang(), self._inv_base_quat)

    """
    Operations.
    """

    def build(self):
        """
        Build the entity manager.
        """
        self.entity = getattr(self.env, self._entity_attr)
        self._cached_calcs()

        # Initialize reset function classes
        for cfg in self.on_reset.values():
            if inspect.isclass(cfg["fn"]):
                self._init_fn_class(cfg)

    def step(self):
        """
        Run some common shared calculations at each step.
        """
        self._cached_calcs()

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

    def _cached_calcs(self):
        """
        Calculate and cache some common values
        """
        self._base_pos[:] = self.entity.get_pos()
        self._base_quat[:] = self.entity.get_quat()
        self._inv_base_quat = inv_quat(self._base_quat)

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
