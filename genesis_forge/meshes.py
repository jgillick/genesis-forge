"""
Trimesh builders for debug visuals.

Each builder returns a `trimesh.Trimesh` already placed in the scene, ready to be drawn
with `scene.draw_debug_mesh(mesh)`.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from numpy.typing import ArrayLike

Color = tuple[float, float, float, float]
"""An RGBA color with components in the range [0, 1]"""


def arrow_mesh(
    pos: ArrayLike,
    vec: ArrayLike,
    radius: float,
    *,
    head_radius_ratio: float = 2.0,
    head_length_ratio: float = 4.0,
    max_head_fraction: float = 0.5,
    sections: int = 24,
    color: Color | None = None,
) -> trimesh.Trimesh:
    """
    An arrow from `pos` along `vec`, with its tip at `pos + vec`.

    Unlike the arrow drawn by `scene.draw_debug_arrow`, the arrowhead is sized from the
    shaft radius rather than the arrow length, so a short arrow keeps a pointed head, and
    the section count is configurable so the arrow looks round rather than blocky.

    Args:
        pos: The position of the base of the arrow, shape (3,)
        vec: The arrow vector: its direction and length, shape (3,). Must not be zero.
        radius: The radius of the shaft
        head_radius_ratio: The base radius of the arrowhead, relative to `radius`
        head_length_ratio: The length of the arrowhead, relative to `radius`
        max_head_fraction: The arrowhead is never longer than this fraction of the arrow length
        sections: The number of sides of the shaft and arrowhead
        color: RGBA color of the arrow, or None to leave the mesh uncolored

    Returns:
        The arrow mesh
    """
    vec = np.asarray(vec, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        raise ValueError("Arrow vector must not be zero")
    if radius <= 0.0:
        raise ValueError(f"Arrow radius must be positive, got {radius}")

    # The arrowhead is sized from the shaft radius, so its width is the same for every
    # arrow. Only its length gives way when the arrow is too short for a full head.
    head_radius = radius * head_radius_ratio
    head_length = min(radius * head_length_ratio, length * max_head_fraction)
    shaft_length = length - head_length

    # Build the arrow along +Z with its base at the origin, then pose it
    head = trimesh.creation.cone(
        radius=head_radius, height=head_length, sections=sections
    )
    head.apply_translation([0.0, 0.0, shaft_length])
    parts = [head]

    if shaft_length > 0.0:
        # The cylinder is centered on the origin; shift it so its base is at z=0
        shaft = trimesh.creation.cylinder(
            radius=radius, height=shaft_length, sections=sections
        )
        shaft.apply_translation([0.0, 0.0, shaft_length / 2.0])
        parts.insert(0, shaft)

    mesh = trimesh.util.concatenate(parts)
    mesh.apply_transform(_z_aligned_pose(pos, vec))
    if color is not None:
        mesh.visual.vertex_colors = color
    return mesh


def arc_arrow_mesh(
    pos: ArrayLike,
    radius: float,
    sweep: float,
    tube_radius: float,
    *,
    start_angle: float = 0.0,
    head_radius_ratio: float = 2.5,
    head_length_ratio: float = 5.0,
    tube_sides: int = 12,
    section_angle: float = math.radians(3.0),
    color: Color | None = None,
) -> trimesh.Trimesh:
    """
    An arc of tube around the vertical axis through `pos`, with an arrowhead at the end.

    The arc lies in the horizontal plane through `pos`. It starts at `start_angle`
    (radians from the world +X axis) and sweeps counter-clockwise (viewed from above) for
    a positive `sweep`, or clockwise for a negative one. The arrowhead sits at the end of
    the sweep and points along the direction of travel.

    Args:
        pos: The center of the arc, shape (3,)
        radius: The radius of the arc, measured to the center of the tube
        sweep: The angle the arc sweeps through, in radians. Must not be zero.
        tube_radius: The radius of the tube
        start_angle: The angle the arc starts from, in radians from the world +X axis
        head_radius_ratio: The base radius of the arrowhead, relative to `tube_radius`
        head_length_ratio: The length of the arrowhead, relative to `tube_radius`
        tube_sides: The number of sides of the tube's cross-section
        section_angle: The angle, in radians, covered by one segment along the arc
        color: RGBA color of the arc, or None to leave the mesh uncolored

    Returns:
        The arc mesh
    """
    if radius <= 0.0:
        raise ValueError(f"Arc radius must be positive, got {radius}")
    if tube_radius <= 0.0:
        raise ValueError(f"Arc tube radius must be positive, got {tube_radius}")
    if sweep == 0.0:
        raise ValueError("Arc sweep must not be zero")

    abs_sweep = abs(sweep)
    direction = math.copysign(1.0, sweep)

    # The tube: a circular cross-section, offset from the vertical axis by the arc radius,
    # revolved around that axis. The last point repeats the first so the profile is closed.
    theta = np.linspace(0.0, 2.0 * np.pi, tube_sides + 1)
    cross_section = np.stack([np.cos(theta), np.sin(theta)], axis=1) * tube_radius
    cross_section[-1] = cross_section[0]
    cross_section[:, 0] += radius
    sections = max(4, math.ceil(abs_sweep / section_angle))
    arc = trimesh.creation.revolve(cross_section, angle=abs_sweep, sections=sections)

    # trimesh revolves counter-clockwise from +X, so the arc covers the angles
    # [0, |sweep|]. A counter-clockwise sweep ends at |sweep|; a clockwise sweep travels
    # from |sweep| back to 0, so its head goes at 0 pointing clockwise.
    head_angle = abs_sweep if direction > 0.0 else 0.0
    head = _arc_head(
        radius,
        head_angle,
        direction,
        head_radius=tube_radius * head_radius_ratio,
        head_length=tube_radius * head_length_ratio,
        sections=tube_sides,
    )
    mesh = trimesh.util.concatenate([arc, head])

    # Rotate the arc so it starts at the start angle. A clockwise arc currently starts at
    # |sweep| and ends at 0, so it is rotated back by the sweep as well.
    yaw = start_angle if direction > 0.0 else start_angle + sweep
    mesh.apply_transform(_yaw_pose(pos, yaw))
    if color is not None:
        mesh.visual.vertex_colors = color
    return mesh


def _arc_head(
    radius: float,
    angle: float,
    direction: float,
    *,
    head_radius: float,
    head_length: float,
    sections: int,
) -> trimesh.Trimesh:
    """
    The arrowhead cone for an arc around the Z axis: its base sits on the arc at `angle`
    (radians from the +X axis) and it points along the arc's tangent, counter-clockwise
    for `direction` +1 or clockwise for -1.
    """
    cone = trimesh.creation.cone(
        radius=head_radius, height=head_length, sections=sections
    )

    # The cone is built along +Z. Build a rotation that maps +Z onto the tangent (its
    # columns are the images of the X, Y, and Z axes) and place it on the arc.
    tangent_x = -math.sin(angle) * direction
    tangent_y = math.cos(angle) * direction
    cone.apply_transform(
        np.array(
            [
                [-tangent_y, 0.0, tangent_x, radius * math.cos(angle)],
                [tangent_x, 0.0, tangent_y, radius * math.sin(angle)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    return cone


def _z_aligned_pose(pos: ArrayLike, direction: ArrayLike) -> np.ndarray:
    """A 4x4 pose that places a mesh at `pos` with its +Z axis aligned to `direction`"""
    direction = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(direction)
    if length == 0.0:
        raise ValueError("Pose direction must not be a zero vector")

    # An orthonormal frame whose Z axis is the direction. The X and Y axes come from the
    # cross product with whichever world axis is farthest from the direction.
    z_axis = direction / length
    helper = (
        np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _yaw_pose(pos: ArrayLike, yaw: float) -> np.ndarray:
    """A 4x4 pose that places a mesh at `pos`, rotated by `yaw` radians about the +Z axis"""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ]
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T
