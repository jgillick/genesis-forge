import torch
import inspect
import genesis as gs
from typing import TypedDict, Callable, Any

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import (
    transform_by_quat,
    inv_quat,
)
from genesis_forge.managers.config import ConfigItem, ResetMdpFnClass

ResetConfigFn = Callable[[GenesisEnv, RigidEntity, list[int], ...], None]


class EntityResetConfig(TypedDict):
    """Defines an entity reset item."""

    fn: ResetConfigFn | ResetMdpFnClass
    """
    Function, or class function, that will be called on reset.

    Args:
        env: The environment instance.
        entity: The entity instance.
        envs_idx: The environment ids for which the entity is to be reset.
        **params: Additional parameters to pass to the function from the params dictionary.
    """

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    weight: float
    """The weight of the reward item."""


class EntityManager(BaseManager):
    """
    Provides options for resetting an entity and adding noise and randomization to its state.

    Args:
        env: The environment instance.
        entity_attr: The attribute name of the environment that the entity is stored in.
        on_reset: The reset configuration for the entity.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.entity_manager = EntityManager(
                    self,
                    entity_attr="robot",
                    on_reset={
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
        super().__init__(env, type="entity")
        if hasattr(env, "add_entity_manager"):
            env.add_entity_manager(self)

        self.entity: RigidEntity | None = None
        self.on_reset = on_reset
        self._entity_attr = entity_attr

        # Wrap config items
        self.on_reset: dict[str, ConfigItem] = {}
        for name, cfg in on_reset.items():
            self.on_reset[name] = ConfigItem(cfg, env)

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

        # Build reset function classes
        for cfg in self.on_reset.values():
            cfg.build(entity=self.entity)

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

        for name, cfg in self.on_reset.items():
            try:
                cfg.execute(envs_idx)
            except Exception as e:
                print(f"Error resetting entity with config: '{name}'")
                raise e

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
