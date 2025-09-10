"""
Termination functions for the Genesis environment.
Each of these should return a boolean tensor indicating which environments should terminate, in the tensor shape (num_envs,).
"""

import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import entity_projected_gravity
from genesis_forge.managers import (
    ContactManager,
    EntityManager,
)


def timeout(env: GenesisEnv) -> torch.Tensor:
    """
    Terminate the environment if the episode length exceeds the maximum episode length.
    """
    return env.episode_length > env.max_episode_length


def bad_orientation(
    env: GenesisEnv,
    limit_angle: float = 0.7,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Terminate the environment if the robot is tipping over too much.

    This function uses projected gravity to detect when the robot has tilted
    beyond a safe threshold. When the robot is perfectly upright, projected
    gravity should be [0, 0, -1] in the body frame. As the robot tilts,
    the x,y components increase, indicating roll and pitch angles.

    Args:
        env: The Genesis environment containing the robot
        limit_angle: Maximum allowed tilt angle in radians (default: 0.7 ~ 40 degrees)
        entity_manager: The entity manager for the entity.
        entity_attr: The attribute name of the entity in the environment.
                        This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """
    # Get the projected gravity vector in body frame
    projected_gravity = None
    if entity_manager is not None:
        projected_gravity = entity_manager.get_projected_gravity()
    else:
        entity = getattr(env, entity_attr)
        projected_gravity = entity_projected_gravity(entity)

    # Calculate the magnitude of tilt (distance from perfectly upright)
    projected_gravity_xy = projected_gravity[:, :2]
    tilt_magnitude = torch.norm(projected_gravity_xy, dim=1)

    # Convert tilt magnitude to angle
    tilt_angle = torch.asin(torch.clamp(tilt_magnitude, max=0.99))

    # Terminate if tilt angle exceeds the limit
    return tilt_angle > limit_angle


def root_height_below_minimum(
    env: GenesisEnv,
    minimum_height: float = 0.05,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Terminate the environment if the robot's base height falls below a minimum threshold.

    Args:
        env: The Genesis environment containing the robot
        minimum_height: Minimum allowed base height in meters
        entity_manager: The entity manager for the entity.
        entity_attr: The attribute name of the entity in the environment.
                        This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """
    base_pos = None
    if entity_manager is not None:
        base_pos = entity_manager.base_pos
    else:
        entity = getattr(env, entity_attr)
        base_pos = entity.get_pos()
    return base_pos[:, 2] < minimum_height


def has_contact(
    _env: GenesisEnv, contact_manager: ContactManager, threshold=1.0, min_contacts=1
) -> torch.Tensor:
    """
    One or more links in the contact manager are in contact with something.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold, per contact, for contact detection (default: 1.0)
        min_contacts: The minimum number of contacts required to terminate (default: 1)

    Returns:
        True for each environment that has contact
    """
    has_contact = contact_manager.contacts[:, :].norm(dim=-1) > threshold
    return has_contact.sum(dim=1) >= min_contacts


def contact_force(
    _env: GenesisEnv, contact_manager: ContactManager, threshold: float = 1.0
) -> torch.Tensor:
    """
    Terminate if any link in the contact manager is in contact with something with a force greater than the threshold.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0 N)

    Returns:
        The total force for the contact manager for each environment
    """
    force_magnitudes = torch.norm(contact_manager.contacts[:, :, :], dim=-1)
    violated = force_magnitudes > threshold
    return torch.any(violated, dim=1)
