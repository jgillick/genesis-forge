import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis.engine.entities import RigidEntity
from genesis_forge.managers import (
    PositionActionManager,
    EntityManager,
    ContactManager,
)
from genesis_forge.utils import entity_lin_vel, entity_ang_vel, entity_projected_gravity

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
DOF/Join observations
"""


def entity_dofs_position(
    env: GenesisEnv,
    action_manager: PositionActionManager = None,
    entity_attr: str = "robot",
    dofs_idx: list[int] = None,
) -> torch.Tensor:
    """
    The position of the entity's DOFs.

    Args:
        env: The Genesis environment containing the entity
        action_manager: The action manager for the robot/entity.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the position of. This isn't necessary if `action_manager` is provided.

    Returns:
        torch.Tensor: The position of the entity's DOFs.
    """
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
    action_manager: PositionActionManager = None,
    entity_attr: str = "robot",
    dofs_idx: list[int] = None,
    clip_to_max_force: bool = False,
) -> torch.Tensor:
    """
    The DOF's force being experienced.

    Args:
        env: The Genesis environment containing the entity
        action_manager: The action manager for the robot/entity.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the force of. This isn't necessary if `action_manager` is provided.
        clip_to_max_force: Clip the force to the maximum force defined in the `action_manager`.

    Returns:
        torch.Tensor: The force of the entity's DOFs.
    """
    if action_manager is not None:
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
    Returns the normalized contact force at each contact point.

    Args:
        env: The Genesis Forge environment
        contact_manager: The contact manager to check for contact

    Returns: tensor of shape (num_envs, num_contacts)
    """
    return torch.norm(contact_manager.contacts[:, :, :], dim=-1)
