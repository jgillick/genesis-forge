from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    transform_by_quat,
)

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink

T = TypeVar("T")


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
    gravity = torch.tensor(
        [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
    ).expand(inv_base_quat.shape[0], 3)
    return transform_by_quat(gravity, inv_base_quat)


def name_matches(name: str, pattern: str) -> bool:
    """
    Whether a name matches a pattern, either exactly or as a fully-anchored regex.

    Args:
        name: The name to test.
        pattern: The exact name, or regex, to test it against.

    Returns:
        True if the name matches.

    Raises:
        re.error: If the pattern is not valid regex.
    """
    return name == pattern or re.fullmatch(pattern, name) is not None


def assign_by_pattern(names: list[str], config: dict[str, T]) -> list[T | None]:
    """
    Resolve a `{<name pattern>: "value"}` config against an ordered list of names,
    and return a list with the values assigned to the index of the matching name.

    The first pattern to claim a name wins, so a config can list specific patterns
    before catch-alls and have them take precedence.

    Args:
        names: The names to assign values to, in the order values are returned.
        config: Maps a name pattern to the value to assign to the names it matches.

    Returns:
        One entry per name: the value assigned to it, or None if no pattern matched it.

    Raises:
        RuntimeError: If a pattern claimed no names -- either because it matched
                      nothing, or because an earlier pattern had already claimed
                      everything it matches. Both mean the entry does nothing, which
                      is nearly always a mistake in the configuration.

    Example::

        assign_by_pattern(
            ["hip", "knee_l", "knee_r"],
            {"knee_.*": 30.0, ".*": 50.0}
        )
        # -> [50.0, 30.0, 30.0]
    """
    assigned: list[T | None] = [None] * len(names)
    for pattern, value in config.items():
        found = False
        for i, name in enumerate(names):
            if assigned[i] is None and name_matches(name, pattern):
                assigned[i] = value
                found = True
        if not found:
            unassigned = [n for n, a in zip(names, assigned, strict=True) if a is None]
            raise RuntimeError(
                f"'{pattern}' not found among the unassigned names: {unassigned}"
            )
    return assigned


def links_by_name_pattern(entity: RigidEntity, name_pattern: str) -> list[RigidLink]:
    """
    Find a list of entity links by name regex pattern.

    Args:
        entity: The entity to find the links in.
        name_pattern: The name regex patterns of the links to find.

    Returns:
        List of RigidLink objects.
    """
    return [link for link in entity.links if name_matches(link.name, name_pattern)]
