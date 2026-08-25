from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    transform_by_quat,
)

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.config import ConfigItem, ConfigItemDict, ResetMdpFn

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class ResetConfigFn(Protocol):
    """
    Simple function called during the reset of the entity.

    Args:
        env: the environemnt
        entity: The robot entity that is being reset
        envs_idx: The environment ids for which the entity is to be reset.
        **params: Other args that are provided by the dict "params" value

    Return:
        result: torch.Tensor, shape (n_envs, 1)
    """
    def __call__(self, env: GenesisEnv, entity: RigidEntity, env_ids: list[int], *params: Any, **kwargs: Any) -> None: ...


class EntityResetConfig(ConfigItemDict):
    """Defines an entity reset item."""

    fn: ResetConfigFn | ResetMdpFn
    """
    Function, or class function, that will be called on reset.

    Args:
        env: The environment instance.
        entity: The entity instance.
        envs_idx: The environment ids for which the entity is to be reset.
        **params: Additional parameters to pass to the function from the params dictionary.
    """


class EntityManager(BaseManager):
    """
    Provides options for resetting an entity and adding noise and randomization to its state.

    Args:
        env: The environment instance.
        entity: The entity to manage.
        on_reset: The reset configuration for the entity.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.entity_manager = EntityManager(
                    self,
                    entity=self.robot,
                    on_reset={
                        "position": {
                            "fn": reset.randomize_terrain_position(
                                terrain_manager=self.terrain_manager,
                                subterrain=self._target_terrain,
                                height_offset=0.15,
                            ),
                        },
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        on_reset: dict[str, EntityResetConfig] | None = None,
    ):
        super().__init__(env, type="entity")
        if hasattr(env, "add_entity_manager"):
            env.add_entity_manager(self)

        self.entity: RigidEntity = entity

        # Wrap config items
        on_reset = on_reset if on_reset is not None else {}
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
        self._linear_velocity = torch.zeros(
            (env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self._angular_velocity = torch.zeros_like(self._linear_velocity)
        self._projected_gravity = torch.zeros_like(self._linear_velocity)

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
        Cached once per step -- see `_cached_calcs()`.
        """
        return self._projected_gravity

    def get_linear_velocity(self) -> torch.Tensor:
        """
        The linear velocity of the entity's base link, in the entity's local frame.
        Cached once per step -- see `_cached_calcs()`.
        """
        return self._linear_velocity

    def get_angular_velocity(self) -> torch.Tensor:
        """
        The angular velocity of the entity's base link, in the entity's local frame.
        Cached once per step -- see `_cached_calcs()`.
        """
        return self._angular_velocity

    """
    Operations.
    """

    def build(self):
        """
        Build the entity manager.
        """
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
                cfg.execute(envs_idx=envs_idx)
            except Exception as e:
                print(f"Error resetting entity with config: '{name}'")
                raise e # noqa

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
        self._linear_velocity[:] = transform_by_quat(
            self.entity.get_vel(), self._inv_base_quat
        )
        self._angular_velocity[:] = transform_by_quat(
            self.entity.get_ang(), self._inv_base_quat
        )
        self._projected_gravity[:] = transform_by_quat(
            self._global_gravity, self._inv_base_quat
        )
