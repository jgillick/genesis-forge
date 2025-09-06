import re
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity

from genesis.utils.geom import (
    transform_by_quat,
    inv_quat,
)

# Pre-allocated gravity tensor for performance
# This avoids creating a new tensor on every function call
_GRAVITY_TENSOR = None

def get_gravity_tensor(num_envs: int) -> torch.Tensor:
    """
    Get and cache the gravity tensor, since this shouldn't change.
    
    Returns:
        torch.Tensor: Gravity tensor of shape (num_envs, 3)
    """
    global _GRAVITY_TENSOR
    if _GRAVITY_TENSOR is None:
        _GRAVITY_TENSOR = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        )
    return _GRAVITY_TENSOR.expand(num_envs, 3)


def entity_lin_vel(entity: RigidEntity) -> torch.Tensor:
    """
    Calculate an entity's linear velocity in its local frame.

    Args:
        entity: The entity to calculate the linear velocity of

    Returns:
        torch.Tensor: Linear velocity in the local frame
    """
    inv_base_quat = inv_quat(entity.get_quat())
    return transform_by_quat(entity.get_vel(), inv_base_quat)


def entity_ang_vel(entity: RigidEntity) -> torch.Tensor:
    """
    Calculate an entity's angular velocity in its local frame.

    Args:
        entity: The entity to calculate the angular velocity of

    Returns:
        torch.Tensor: Angular velocity in the local frame
    """
    inv_base_quat = inv_quat(entity.get_quat())
    return transform_by_quat(entity.get_ang(), inv_base_quat)


def entity_projected_gravity(entity: RigidEntity) -> torch.Tensor:
    """
    Calculate an entity's projected gravity in its local frame.

    Args:
        entity: The entity to calculate the projected gravity of

    Returns:
        torch.Tensor: Projected gravity in the local frame
    """
    inv_base_quat = inv_quat(entity.get_quat())
    global_gravity = get_gravity_tensor(inv_base_quat.shape[0])
    return transform_by_quat(global_gravity, inv_base_quat)


def links_idx_by_name_pattern(entity: RigidEntity, name_re: str) -> list[int]:
    """
    Find a list of entity links by name regex pattern, and return their indices.

    Args:
        entity: The entity to find the links in.
        name_re: The name regex patterns of the links to find.

    Returns:
        List of global link indices.
    """
    links_idx = []
    for link in entity.links:
        if re.match(name_re, link.name):
            links_idx.append(link.idx)
    return links_idx