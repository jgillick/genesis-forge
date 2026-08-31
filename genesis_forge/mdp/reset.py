from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import genesis as gs
import torch
from genesis.utils.geom import (
    xyz_to_quat,
)

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import ResetMdpFn
from genesis_forge.managers.terrain_manager import TerrainManager
from genesis_forge.utils import links_by_name_pattern

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


XYZRotation = dict[Literal["x", "y", "z"], float | tuple[float, float]]
"""
Define the rotation around the X/Y/Z axes.
The value can either be a distinct value, or a tuple of (min, max) values to randomize within.
"""


@dataclass(kw_only=True, eq=False)
class zero_all_dofs_velocity(ResetMdpFn):
    """
    Zero the velocity of all dofs of the entity.
    """

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: list[int]):
        entity.zero_all_dofs_velocity(envs_idx)


@dataclass(kw_only=True, eq=False)
class set_rotation(ResetMdpFn):
    """
    Set the entity's rotation in either absolute or randomized euler angles.
    If the x/y/z value is a tuple (for example: `(0, 2 * math.pi)`), the rotation will be randomized within that radian range.

    Args:
        x: The x angle or range to set the rotation to.
        y: The y angle or range to set the rotation to.
        z: The z angle or range to set the rotation to.
    """

    x: float | tuple[float, float] = 0
    y: float | tuple[float, float] = 0
    z: float | tuple[float, float] = 0

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: list[int]):
        angle_buffer = torch.zeros((len(envs_idx), 3), device=gs.device)
        if isinstance(self.x, tuple):
            angle_buffer[:, 0].uniform_(*self.x)
        if isinstance(self.y, tuple):
            angle_buffer[:, 1].uniform_(*self.y)
        if isinstance(self.z, tuple):
            angle_buffer[:, 2].uniform_(*self.z)

        # Set angle as quat
        quat = xyz_to_quat(angle_buffer)
        entity.set_quat(quat, envs_idx=envs_idx)


@dataclass(kw_only=True, eq=False)
class position(ResetMdpFn):
    """
    Reset the entity to a fixed position and (optional) rotation

    Args:
        position: The position to set the entity to.
        quat: The quaternion to set the entity to.
        zero_velocity: Whether to zero the velocity of all the entity's dofs.
                       Defaults to True. This is a safety measure after a sudden change in entity pose.
    """

    position: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None
    zero_velocity: bool = True

    def build(self):
        self.reset_pos = torch.tensor(self.position, device=gs.device)
        self._pos_buffer = torch.zeros(
            (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        self.reset_quat = None
        self._quat_buffer = None
        if self.quat is not None:
            self.reset_quat = torch.tensor(self.quat, device=gs.device)
            self._quat_buffer = torch.zeros(
                (self.env.num_envs, 4), device=gs.device, dtype=gs.tc_float
            )

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: list[int]):
        self._pos_buffer[envs_idx] = self.reset_pos
        entity.set_pos(
            self._pos_buffer[envs_idx],
            envs_idx=envs_idx,
            zero_velocity=self.zero_velocity,
        )

        if self.reset_quat is not None:
            self._quat_buffer[envs_idx] = self.reset_quat.reshape(1, -1)
            entity.set_quat(
                self._quat_buffer[envs_idx],
                envs_idx=envs_idx,
                zero_velocity=self.zero_velocity,
            )

@dataclass(kw_only=True, eq=False)
class randomize_terrain_position(ResetMdpFn):
    """
    Place the entity in a random position on the terrain for each environment.

    Args:
        terrain_manager: The terrain manager to use to generate the random position.
        height_offset: The height offset to add to the random position.
        subterrain: The subterrain to generate the random position on.
                    Either a string or a callable that returns a string.
        rotation: The X/Y/Z rotation to set the entity to. Defaults to a random rotation around the z-axis.
                  Set to None to not set a rotation.
        zero_velocity: Whether to zero the velocity of all the entity's dofs.
                       Defaults to True. This is a safety measure after a sudden change in entity pose.
    """

    terrain_manager: TerrainManager
    height_offset: float = 0.1e-3
    subterrain: str | Callable[[], str | None] | None = None
    rotation: XYZRotation | None = field(
        default_factory=lambda: {"z": (0, 2 * math.pi)}
    )
    zero_velocity: bool = True

    def build(self):
        """
        Initialize the buffers
        """
        self._rotation_buffer = torch.zeros(
            (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self._quat_buffer = torch.zeros(
            (self.env.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )

    def define_quat(self, envs_idx: list[int], rotation: XYZRotation):
        """
        Set the rotation quaternion for the given environment ids.
        """
        x = rotation.get("x", 0)
        y = rotation.get("y", 0)
        z = rotation.get("z", 0)
        n_envs = len(envs_idx)

        if isinstance(x, tuple):
            self._rotation_buffer[envs_idx, 0] = torch.empty(
                n_envs, device=gs.device
            ).uniform_(*x)
        if isinstance(y, tuple):
            self._rotation_buffer[envs_idx, 1] = torch.empty(
                n_envs, device=gs.device
            ).uniform_(*y)
        if isinstance(z, tuple):
            self._rotation_buffer[envs_idx, 2] = torch.empty(
                n_envs, device=gs.device
            ).uniform_(*z)

        # Set angle as quat
        self._quat_buffer[envs_idx] = xyz_to_quat(self._rotation_buffer[envs_idx])

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: list[int]):
        # Get the subterrain
        subterrain = self.subterrain
        if subterrain is not None and callable(subterrain):
            subterrain = subterrain()

        # Randomize positions on the terrain
        pos = self.terrain_manager.generate_random_env_pos(
            envs_idx=envs_idx,
            subterrain=subterrain,
            height_offset=self.height_offset,
        )
        entity.set_pos(pos, envs_idx=envs_idx, zero_velocity=self.zero_velocity)

        # Rotation
        if self.rotation is not None:
            self.define_quat(envs_idx, self.rotation)
            entity.set_quat(
                self._quat_buffer[envs_idx],
                envs_idx=envs_idx,
                zero_velocity=self.zero_velocity,
            )


@dataclass(kw_only=True, eq=False)
class randomize_link_mass_shift(ResetMdpFn):
    """
    Randomly add/subtract mass to one or more links of the entity.
    This picks a random value from `mass_range` and passes it to `set_mass_shift` for each environment.

    See: https://genesis-world.readthedocs.io/en/latest/api_reference/entity/rigid_entity/rigid_entity.html#genesis.engine.entities.rigid_entity.rigid_entity.RigidEntity.set_mass_shift

    Args:
        link_name: The name, or regex pattern, of the link(s) to set the mass for.
                   Can also be a list of names/patterns to target multiple sets of links.
        mass_range: The range of the mass that will be added or subtracted from the link(s) on each reset.
    """

    link_name: str | list[str]
    mass_range: tuple[float, float]

    def build(self):
        self._links_idx_local = []
        if self.link_name is None:
            return
        link_names = (
            [self.link_name] if isinstance(self.link_name, str) else self.link_name
        )
        for name in link_names:
            links = links_by_name_pattern(self.entity, name)
            if len(links) == 0:
                raise ValueError(f"No links found with name/pattern '{name}'")
            self._links_idx_local.extend(link.idx_local for link in links)

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: list[int]):
        # Randomize mass
        mass_shift = torch.empty(
            (len(envs_idx), len(self._links_idx_local)), device=gs.device
        ).uniform_(*self.mass_range)

        # Set mass on entity
        entity.set_mass_shift(
            mass_shift,
            links_idx_local=self._links_idx_local,
            envs_idx=envs_idx,
        )
