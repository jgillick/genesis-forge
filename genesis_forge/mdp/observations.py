from __future__ import annotations
from genesis import gs
import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    PositionActionManager,
    EntityManager,
    ContactManager,
)
from genesis_forge.utils import entity_lin_vel, entity_ang_vel, entity_projected_gravity
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

"""
Entity Observations
"""


def entity_linear_velocity(
    env: GenesisEnv, entity_manager: EntityManager = None, entity_attr: str = "robot"
) -> torch.Tensor:
    """
    The linear velocity of the entity's base link, in the entity's local frame.

    Args:
        env: The Genesis environment containing the entity
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The linear velocity of the entity's base link, in the entity's local frame.
    """
    if entity_manager is not None:
        return entity_manager.get_linear_velocity()
    entity = getattr(env, entity_attr)
    return entity_lin_vel(entity)


def entity_angular_velocity(
    env: GenesisEnv, entity_manager: EntityManager = None, entity_attr: str = "robot"
) -> torch.Tensor:
    """
    The angular velocity of the entity's base link, in the entity's local frame.

    Args:
        env: The Genesis environment containing the entity
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The angular velocity of the entity's base link, in the entity's local frame.
    """
    if entity_manager is not None:
        return entity_manager.get_angular_velocity()
    entity = getattr(env, entity_attr)
    return entity_ang_vel(entity)


def entity_projected_gravity(
    env: GenesisEnv, entity_manager: EntityManager = None, entity_attr: str = "robot"
) -> torch.Tensor:
    """
    The projected gravity of the entity's base link, in the entity's local frame.

    Args:
        env: The Genesis environment containing the entity
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The projected gravity of the entity's base link, in the entity's local frame.
    """
    if entity_manager is not None:
        return entity_manager.get_projected_gravity()
    entity = getattr(env, entity_attr)
    return entity_projected_gravity(entity)

"""
Sensor observations
"""

def read_imu(env: GenesisEnv, imu: gs.sensors.IMU) -> torch.Tensor:
    """
    Makes an IMU reading and returns the concatenated linear acceleration and angular velocity readings.

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
                    "fn": self.imu_observation,
                },
            }
        )

    Returns:
        torch.Tensor: Shape (n_envs, 6): [lin_acc_xyz, ang_vel_xyz] per env.
    """
    value = imu.read()
    return torch.cat([value.lin_acc, value.ang_vel], dim=-1)

"""
DOF/Join observations
"""


def entity_dofs_position(
    env: GenesisEnv,
    actuator_manager: ActuatorManager = None,
    entity_attr: str = "robot",
    dofs_idx: list[int] = None,
    action_manager: PositionActionManager = None,
) -> torch.Tensor:
    """
    The position of the entity's DOFs.

    Args:
        env: The Genesis environment containing the entity
        actuator_manager: The actuator manager for the robot/entity.
                          This bypasses the need for dofs_idx and entity_attr parameters.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the position of. This isn't necessary if `action_manager` is provided.
        action_manager: (deprecated) The action manager for the robot/entity.
                        This bypasses the need for dofs_idx and entity_attr parameters.

    Returns:
        torch.Tensor: The position of the entity's DOFs.
    """
    if actuator_manager is not None:
        return actuator_manager.get_dofs_position()
    if action_manager is not None:
        return action_manager.get_dofs_position()
    entity: RigidEntity = getattr(env, entity_attr)
    return entity.get_dofs_position(dofs_idx)


def entity_dofs_velocity(
    env: GenesisEnv,
    action_manager: PositionActionManager = None,
    entity_attr: str = "robot",
    dofs_idx: list[int] = None,
) -> torch.Tensor:
    """
    The velocity of the entity's DOFs.

    Args:
        env: The Genesis environment containing the entity
        action_manager: The action manager for the robot/entity.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the velocity of. This isn't necessary if `action_manager` is provided.

    Returns:
        torch.Tensor: The velocity of the entity's DOFs.
    """
    if action_manager is not None:
        return action_manager.get_dofs_velocity()
    entity: RigidEntity = getattr(env, entity_attr)
    return entity.get_dofs_velocity(dofs_idx)


def entity_dofs_force(
    env: GenesisEnv,
    actuator_manager: ActuatorManager = None,
    entity_attr: str = "robot",
    dofs_idx: list[int] = None,
    clip_to_max_force: bool = False,
    action_manager: PositionActionManager = None,
) -> torch.Tensor:
    """
    The DOF's force being experienced.

    Args:
        env: The Genesis environment containing the entity
        actuator_manager: The actuator manager for the robot/entity.
                          This bypasses the need for dofs_idx and entity_attr parameters.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the force of. This isn't necessary if `action_manager` is provided.
        clip_to_max_force: Clip the force to the maximum force defined in the `action_manager`.
        action_manager: (deprecated) The action manager for the robot/entity.

    Returns:
        torch.Tensor: The force of the entity's DOFs.
    """
    if actuator_manager is not None:
        return actuator_manager.get_dofs_force(clip_to_max_force=clip_to_max_force)
    elif action_manager is not None:
        return action_manager.get_dofs_force(clip_to_max_force=clip_to_max_force)
    entity: RigidEntity = getattr(env, entity_attr)
    return entity.get_dofs_force(dofs_idx)


"""
Actions
"""


def current_actions(
    env: GenesisEnv,
    action_manager: PositionActionManager = None,
) -> torch.Tensor:
    """
    The most current step actions.
    """
    if action_manager is not None:
        return action_manager.get_actions()
    return env.actions


"""
Contacts
"""


def contact_force(env: GenesisEnv, contact_manager: ContactManager) -> torch.Tensor:
    """
    Returns the vector norm contact force at each contact point.

    Args:
        env: The Genesis Forge environment
        contact_manager: The contact manager to check for contact

    Returns: tensor of shape (num_envs, num_contacts)
    """
    return torch.norm(contact_manager.contacts[:, :, :], dim=-1)

def has_contact(
    env: GenesisEnv, contact_manager: ContactManager, threshold=1.0
) -> torch.Tensor:
    """
    Return boolean (1/0) for each link in the contact manager that meets the contact threshold.

    Args:
        env: The Genesis Forge environment
        contact_manager: The contact manager to check for contact
        threshold: The minimum force necessary for contact detection (default: 1.0)

    Returns:
        1 for each link meeting the contact threshold
    """
    has_contact = contact_manager.contacts.norm(dim=-1) > threshold
    return has_contact.float()