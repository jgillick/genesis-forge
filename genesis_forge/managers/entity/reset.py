import torch
from typing import Literal
import genesis as gs
from genesis.utils.geom import (
    xyz_to_quat,
)

from genesis.engine.entities import RigidEntity
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.terrain_manager import TerrainManager
from genesis_forge.utils import links_idx_by_name_pattern

from .config import ResetConfigFnClass


def zero_all_dofs_velocity(
    _env: GenesisEnv,
    entity: RigidEntity,
    envs_idx: list[int],
):
    """
    Zero the velocity of all dofs of the entity.
    """
    entity.zero_all_dofs_velocity(envs_idx)


def set_rotation(
    _env: GenesisEnv,
    entity: RigidEntity,
    envs_idx: list[int],
    x: float | tuple[float, float] = 0,
    y: float | tuple[float, float] = 0,
    z: float | tuple[float, float] = 0,
):
    """
    Set the entity's rotation in either absolute or randomized euler angles.
    If the x/y/z value is a tuple (for example: `(0, 2 * math.pi)`), the rotation will be randomized within that radian range.

    Args:
        env: The environment
        entity: The entity to set the rotation of.
        envs_idx: The environment ids to set the rotation for.
        x: The x angle or range to set the rotation to.
        y: The y angle or range to set the rotation to.
        z: The z angle or range to set the rotation to.
    """

    angle_buffer = torch.zeros((len(envs_idx), 3), device=gs.device)
    if isinstance(x, tuple):
        angle_buffer[:, 0].uniform_(*x)
    if isinstance(y, tuple):
        angle_buffer[:, 1].uniform_(*y)
    if isinstance(z, tuple):
        angle_buffer[:, 2].uniform_(*z)

    # Set angle as quat
    quat = xyz_to_quat(angle_buffer)
    entity.set_quat(quat, envs_idx=envs_idx)


class position(ResetConfigFnClass):
    """
    Reset the entity to a fixed position and (optional) rotation

    Args:
        env: The environment
        entity: The entity to set the position of.
        position: The position to set the entity to.
        quat: The quaternion to set the entity to.
        zero_velocity: Whether to zero the velocity of all the entity's dofs.
                       Defaults to True. This is a safety measure after a sudden change in entity pose.
    """

    def __init__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        zero_velocity: bool = True,
    ):
        self.zero_velocity = zero_velocity
        self.reset_pos = torch.tensor(position, device=gs.device)
        self._pos_buffer = torch.zeros(
            (env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        self.reset_quat = None
        self._quat_buffer = None
        if quat is not None:
            self.reset_quat = torch.tensor(quat, device=gs.device)
            self._quat_buffer = torch.zeros(
                (env.num_envs, 4), device=gs.device, dtype=gs.tc_float
            )

    def __call__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        envs_idx: list[int],
    ):
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


def randomize_terrain_position(
    env: GenesisEnv,
    entity: RigidEntity,
    envs_idx: list[int],
    terrain_manager: TerrainManager,
    subterrain: str | None = None,
    height_offset: float = 0.1e-3,
):
    """
    Place the entity in a random position on the terrain for each environment.
    """
    # Randomize positions on the terrain
    pos = terrain_manager.generate_random_env_pos(
        envs_idx=envs_idx,
        subterrain=subterrain,
        height_offset=height_offset,
    )
    entity.set_pos(pos, envs_idx=envs_idx)


class randomize_link_mass_shift(ResetConfigFnClass):
    """
    Randomly add/subtract mass to one or more links of the entity.
    This picks a random value from `add_mass_range` and passes it to `set_mass_shift` for each environment.
    This means that on subsequent calls, the mass can continue to either decrease or increase.
    """

    def __init__(
        self,
        _env: GenesisEnv,
        entity: RigidEntity,
        link_name: str,
        add_mass_range: tuple[float, float] = (-0.2, 0.2),
    ):
        """
        Args:
            env: The environment
            entity: The entity to set the rotation of.
            link_name: The name, or regex pattern, of the link(s) to set the inertial mass for.
            add_mass_range: The range of the mass that can be added or subtracted each reset.
        """
        self.env = _env
        self.add_mass_range = add_mass_range
        self._entity = entity
        self._link_name = link_name
        self._links_idx = []
        self._mass_shift_buffer: torch.tensor | None = None
        self.build()

    def build(self):
        """
        Find the links and initialize buffers
        """
        self._links_idx = []
        self._orig_mass = None
        if self._link_name is not None:
            self._links_idx = links_idx_by_name_pattern(self._entity, self._link_name)
            if len(self._links_idx) > 0:
                self._mass_shift_buffer = torch.zeros(
                    (self.env.num_envs, len(self._links_idx)), device=gs.device
                )

    def __call__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        envs_idx: list[int],
    ):
        """
        Randomly shift the link masses
        """
        # Randomize mass
        self._mass_shift_buffer[envs_idx, :].uniform_(*self.add_mass_range)

        # Set mass on entity
        self._entity.set_mass_shift(
            self._mass_shift_buffer,
            links_idx_local=self._links_idx,
            envs_idx=envs_idx,
        )
