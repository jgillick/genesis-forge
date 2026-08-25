from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from genesis import gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    MdpFn,
    PositionActionManager,
)
from genesis_forge.managers.action.base import BaseActionManager
from genesis_forge.utils import (
    entity_ang_vel,
    entity_lin_vel,
)
from genesis_forge.utils import (
    entity_projected_gravity as get_entity_projected_gravity,
)

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

"""
Entity Observations
"""


@dataclass(kw_only=True, eq=False)
class entity_linear_velocity(MdpFn):
    """
    The linear velocity of the entity's base link, in the entity's local frame.

    Args:
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the observation for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The linear velocity of the entity's base link, in the entity's local frame.
    """

    entity_manager: EntityManager = None
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            return self.entity_manager.get_linear_velocity()
        entity = self.entity if self.entity is not None else env.robot
        return entity_lin_vel(entity)


@dataclass(kw_only=True, eq=False)
class entity_angular_velocity(MdpFn):
    """
    The angular velocity of the entity's base link, in the entity's local frame.

    Args:
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the observation for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The angular velocity of the entity's base link, in the entity's local frame.
    """

    entity_manager: EntityManager = None
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            return self.entity_manager.get_angular_velocity()
        entity = self.entity if self.entity is not None else env.robot
        return entity_ang_vel(entity)


@dataclass(kw_only=True, eq=False)
class entity_projected_gravity(MdpFn):
    """
    The projected gravity of the entity's base link, in the entity's local frame.

    Args:
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the observation for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The projected gravity of the entity's base link, in the entity's local frame.
    """

    entity_manager: EntityManager = None
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            return self.entity_manager.get_projected_gravity()
        entity = self.entity if self.entity is not None else env.robot
        return get_entity_projected_gravity(entity)


"""
Sensor observations
"""


@dataclass(kw_only=True, eq=False)
class read_imu(MdpFn):
    """
    Makes an IMU reading and returns the concatenated linear acceleration and angular velocity readings.

    Args:
        imu: The IMU sensor to read from.

    Example::

        self.imu = gs.sensors.IMU(
            entity_idx=self.robot.idx,
            pos_offset=(0.24, 0.0, 0.0),
            euler_offset=(0.0, 0.0, 0.0),
        )

        ...

        ObservationManager(
            self,
            cfg={
                "imu_sensor": {
                    "fn": observations.read_imu(imu=self.imu),
                },
            }
        )

    Returns:
        torch.Tensor: Shape `(n_envs, 6)` — `[lin_acc_xyz, ang_vel_xyz]` per env.
    """

    imu: gs.sensors.IMU

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        value = self.imu.read()
        return torch.cat([value.lin_acc, value.ang_vel], dim=-1)


"""
DOF/Join observations
"""


@dataclass(kw_only=True, eq=False)
class entity_dofs_position(MdpFn):
    """
    The position of the entity's DOFs.

    Args:
        actuator_manager: The actuator manager for the robot/entity.
                          This bypasses the need for dofs_idx and entity parameters.
        entity: The entity to read DOFs from. Defaults to `env.robot`. This isn't necessary if `actuator_manager` is provided.
        dofs_idx: The indices of the DOFs to get the position of. This isn't necessary if `actuator_manager` is provided.

    Returns:
        torch.Tensor: The position of the entity's DOFs.
    """

    actuator_manager: ActuatorManager = None
    entity: RigidEntity = None
    dofs_idx: list[int] = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.actuator_manager is not None:
            return self.actuator_manager.get_dofs_position()
        entity: RigidEntity = self.entity if self.entity is not None else env.robot
        return entity.get_dofs_position(self.dofs_idx)


@dataclass(kw_only=True, eq=False)
class entity_dofs_velocity(MdpFn):
    """
    The velocity of the entity's DOFs.

    Args:
        action_manager: The action manager for the robot/entity.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to read DOFs from. Defaults to `env.robot`. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the velocity of. This isn't necessary if `action_manager` is provided.

    Returns:
        torch.Tensor: The velocity of the entity's DOFs.
    """

    action_manager: PositionActionManager = None
    entity: RigidEntity = None
    dofs_idx: list[int] = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.action_manager is not None:
            return self.action_manager.get_dofs_velocity()
        entity: RigidEntity = self.entity if self.entity is not None else env.robot
        return entity.get_dofs_velocity(self.dofs_idx)


@dataclass(kw_only=True, eq=False)
class entity_dofs_force(MdpFn):
    """
    The DOF's force being experienced.

    Args:
        actuator_manager: The actuator manager for the robot/entity.
                          This bypasses the need for dofs_idx and entity parameters.
        entity: The entity to read DOFs from. Defaults to `env.robot`. This isn't necessary if `actuator_manager` is provided.
        dofs_idx: The indices of the DOFs to get the force of. This isn't necessary if `actuator_manager` is provided.
        clip_to_max_force: Clip the force to the maximum force defined in the `actuator_manager`.

    Returns:
        torch.Tensor: The force of the entity's DOFs.
    """

    actuator_manager: ActuatorManager = None
    entity: RigidEntity = None
    dofs_idx: list[int] = None
    clip_to_max_force: bool = False

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.actuator_manager is not None:
            return self.actuator_manager.get_dofs_force(
                clip_to_max_force=self.clip_to_max_force
            )
        entity: RigidEntity = self.entity if self.entity is not None else env.robot
        return entity.get_dofs_force(self.dofs_idx)


"""
Actions
"""


@dataclass(kw_only=True, eq=False)
class current_actions(MdpFn):
    """
    The most current step actions.

    Args:
        action_manager: The action manager to source actions from. If not provided,
                        actions are read from `env.actions`.
    """

    action_manager: BaseActionManager | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.action_manager is not None:
            return self.action_manager.raw_actions
        return env.actions


"""
Contacts
"""


@dataclass(kw_only=True, eq=False)
class contact_force(MdpFn):
    """
    Returns the vector norm contact force at each contact point.

    Args:
        contact_manager: The contact manager to check for contact

    Returns:
        torch.Tensor: Shape `(num_envs, num_contacts)`.
    """

    contact_manager: ContactManager

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        return torch.norm(self.contact_manager.contacts[:, :, :], dim=-1)


@dataclass(kw_only=True, eq=False)
class has_contact(MdpFn):
    """
    Return 1 (true) or 0 (false) for each link in the contact manager that meets the contact threshold.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The minimum force necessary for contact detection (default: 1.0)

    Returns:
        1 for each link meeting the contact threshold
    """

    contact_manager: ContactManager
    threshold: float = 1.0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        in_contact = self.contact_manager.contacts.norm(dim=-1) > self.threshold
        return in_contact.float()
