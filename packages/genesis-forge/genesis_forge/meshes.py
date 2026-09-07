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
    head_radius_ratio: float = 1.5,
    head_length_ratio: float = 4.0,
    max_head_fraction: float = 0.5,
    sections: int = 36,
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
        _set_color(mesh, color)
    return mesh


def flat_arc_arrow_mesh(
    pos: ArrayLike,
    radius: float,
    sweep: float,
    width: float,
    thickness: float,
    *,
    start_angle: float = 0.0,
    head_width_ratio: float = 2.0,
    head_length_ratio: float = 2.0,
    section_angle: float = math.radians(3.0),
    color: Color | None = None,
) -> trimesh.Trimesh:
    """
    A flat arc around the vertical axis through `pos`, with an arrowhead at the end.

    The arc is a two-dimensional band extruded to `thickness` and centered on the
    horizontal plane through `pos`. It starts at `start_angle` (radians from the world
    +X axis) and sweeps counter-clockwise (viewed from above) for a positive `sweep`, or
    clockwise for a negative one. The arrowhead sits at the end of the sweep and points
    along the direction of travel.

    Args:
        pos: The center of the arc, shape (3,)
        radius: The radius of the arc, measured to the center of the band
        sweep: The angle the arc sweeps through, in radians. Must not be zero.
        width: The radial width of the band
        thickness: The vertical thickness of the arc
        start_angle: The angle the arc starts from, in radians from the world +X axis
        head_width_ratio: The width of the arrowhead, relative to `width`. Should be > 1.
        head_length_ratio: The length of the arrowhead, relative to `width`
        section_angle: The angle, in radians, covered by one segment along the arc
        color: RGBA color of the arc, or None to leave the mesh uncolored

    Returns:
        The arc mesh
    """
    if radius <= 0.0:
        raise ValueError(f"Arc radius must be positive, got {radius}")
    if width <= 0.0:
        raise ValueError(f"Arc width must be positive, got {width}")
    if thickness <= 0.0:
        raise ValueError(f"Arc thickness must be positive, got {thickness}")
    if sweep == 0.0:
        raise ValueError("Arc sweep must not be zero")

    abs_sweep = abs(sweep)
    direction = math.copysign(1.0, sweep)

    # The band: outer and inner rims sampled along the angles [0, |sweep|], joined by
    # two triangles per section. Outer rim vertices are 0..n, inner rim n+1..2n+1.
    n = max(4, math.ceil(abs_sweep / section_angle))
    angles = np.linspace(0.0, abs_sweep, n + 1)
    rim = np.column_stack([np.cos(angles), np.sin(angles)])
    outline = np.vstack([rim * (radius + width / 2.0), rim * (radius - width / 2.0)])
    faces = []
    for i in range(n):
        outer, outer_next, inner, inner_next = i, i + 1, n + 1 + i, n + 2 + i
        faces.append([inner, outer, outer_next])
        faces.append([inner, outer_next, inner_next])

    # The head: a triangle across the band at the end of travel, fanned from its tip so
    # it shares the band's end edge. A counter-clockwise sweep ends at |sweep|; a
    # clockwise one travels from |sweep| back to 0, so its head goes at 0.
    if direction > 0.0:
        end_angle, end_outer, end_inner = abs_sweep, n, 2 * n + 1
    else:
        end_angle, end_outer, end_inner = 0.0, 0, n + 1
    radial = np.array([math.cos(end_angle), math.sin(end_angle)])
    tangent = np.array([-radial[1], radial[0]]) * direction
    half_head_w = width * head_width_ratio / 2.0
    head_outer, head_inner, tip = len(outline), len(outline) + 1, len(outline) + 2
    outline = np.vstack(
        [
            outline,
            radial * (radius + half_head_w),
            radial * (radius - half_head_w),
            radial * radius + tangent * width * head_length_ratio,
        ]
    )
    faces += [
        [tip, head_inner, end_inner],
        [tip, end_inner, end_outer],
        [tip, end_outer, head_outer],
    ]
    mesh = _extrude(outline, faces, thickness)

    # Rotate the arc so it starts at the start angle. A clockwise arc currently starts at
    # |sweep| and ends at 0, so it is rotated back by the sweep as well.
    yaw = start_angle if direction > 0.0 else start_angle + sweep
    mesh.apply_transform(_yaw_pose(pos, yaw))
    if color is not None:
        _set_color(mesh, color)
    return mesh


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


def _extrude(outline: np.ndarray, faces: list, thickness: float) -> trimesh.Trimesh:
    """
    Extrude a triangulated 2D outline in the XY plane to `thickness`, centered on the
    plane. Faces are re-wound counter-clockwise so the extrusion's normals face outward.
    """
    faces = np.asarray(faces)
    a, b, c = (outline[faces[:, k]] for k in range(3))
    signed_area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    faces[signed_area < 0.0] = faces[signed_area < 0.0][:, ::-1]
    mesh = trimesh.creation.extrude_triangulation(outline, faces, thickness)
    mesh.apply_translation([0.0, 0.0, -thickness / 2.0])
    return mesh


def _set_color(mesh: trimesh.Trimesh, color: Color) -> None:
    """Color every vertex of `mesh` with one RGBA color"""
    # Assign fresh color visuals rather than setting `mesh.visual.vertex_colors`: the
    # `visual` attribute is typed as ColorVisuals | TextureVisuals, and only the former
    # has vertex colors, so type checkers reject the attribute assignment.
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=color)
