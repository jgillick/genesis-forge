"""
Tests for the trimesh builders in `genesis_forge.meshes`.

These check the geometry of the returned meshes directly and never render anything.
"""

import math

import numpy as np
import pytest
import trimesh

from genesis_forge.meshes import (
    arrow_mesh,
    flat_arc_arrow_mesh,
)

TOL = 1e-6
ORIGIN = (0.0, 0.0, 0.0)


def angles(mesh: trimesh.Trimesh, center=ORIGIN) -> np.ndarray:
    """The angle of each vertex about the vertical axis through `center`, in radians from +X"""
    v = mesh.vertices - np.asarray(center)
    return np.arctan2(v[:, 1], v[:, 0])


def radial(mesh: trimesh.Trimesh, center=ORIGIN) -> np.ndarray:
    """The distance of each vertex from the vertical axis through `center`"""
    v = mesh.vertices - np.asarray(center)
    return np.hypot(v[:, 0], v[:, 1])


"""
arrow_mesh
"""


def test_arrow_is_placed_at_pos_and_points_along_vec():
    pos = np.array([1.0, 2.0, 3.0])
    vec = np.array([0.3, 0.4, 0.0])
    mesh = arrow_mesh(pos, vec, 0.01, head_radius_ratio=2.0)
    # The tip is the vertex farthest along the vector
    along = (mesh.vertices - pos) @ (vec / np.linalg.norm(vec))
    assert along.max() == pytest.approx(0.5, abs=TOL)
    assert along.min() == pytest.approx(0.0, abs=TOL)
    assert np.allclose(mesh.vertices[np.argmax(along)], pos + vec, atol=TOL)
    # Every vertex lies within the head radius of the arrow's axis
    axis_dist = np.linalg.norm(
        (mesh.vertices - pos) - np.outer(along, vec / np.linalg.norm(vec)), axis=1
    )
    assert axis_dist.max() == pytest.approx(0.02, abs=TOL)
    assert mesh.is_watertight


def test_arrow_head_length_comes_from_the_radius_not_the_length():
    mesh = arrow_mesh(ORIGIN, (0.0, 0.0, 1.0), 0.01, head_length_ratio=4.0)
    # The shaft vertices are the ones at the shaft radius; the head starts where they end
    shaft_z = mesh.vertices[np.isclose(radial(mesh), 0.01)][:, 2]
    assert shaft_z.max() == pytest.approx(1.0 - 0.04)


def test_arrow_head_is_clamped_for_short_arrows():
    mesh = arrow_mesh(
        ORIGIN, (0.0, 0.0, 0.02), 0.01, head_length_ratio=4.0, max_head_fraction=0.5
    )
    shaft_z = mesh.vertices[np.isclose(radial(mesh), 0.01)][:, 2]
    assert shaft_z.max() == pytest.approx(0.01)
    assert mesh.vertices[:, 2].max() == pytest.approx(0.02)


def test_arrow_head_radius_does_not_depend_on_the_length():
    for length in (1.0, 0.04, 0.01):
        mesh = arrow_mesh(ORIGIN, (0.0, 0.0, length), 0.01, head_radius_ratio=2.0)
        assert radial(mesh).max() == pytest.approx(0.02)


@pytest.mark.parametrize(
    "vec, radius", [((0.0, 0.0, 0.0), 0.01), ((0.0, 0.0, 0.3), 0.0)]
)
def test_arrow_rejects_degenerate_dimensions(vec, radius):
    with pytest.raises(ValueError):
        arrow_mesh(ORIGIN, vec, radius)


"""
flat_arc_arrow_mesh
"""

WIDTH = 0.02
THICKNESS = 0.004
ARC_RADIUS = 0.2
FLAT_HEAD_ANGLE = math.atan(WIDTH * 2.0 / ARC_RADIUS)
"""The angle the flat arrowhead tip (2 widths long) extends past the end of the arc"""


@pytest.mark.parametrize("sweep", [math.radians(45), -math.radians(45)])
def test_flat_arc_sweeps_from_the_start_angle_in_the_direction_of_the_sign(sweep):
    pos = np.array([1.0, 2.0, 3.0])
    start = math.radians(120)
    mesh = flat_arc_arrow_mesh(
        pos,
        ARC_RADIUS,
        sweep,
        WIDTH,
        THICKNESS,
        start_angle=start,
        head_length_ratio=2.0,
    )
    a = angles(mesh, pos) - start
    a = np.arctan2(np.sin(a), np.cos(a))  # wrap to [-pi, pi]
    if sweep > 0.0:
        assert a.min() == pytest.approx(0.0, abs=TOL)
        assert a.max() == pytest.approx(sweep + FLAT_HEAD_ANGLE, abs=TOL)
    else:
        assert a.max() == pytest.approx(0.0, abs=TOL)
        assert a.min() == pytest.approx(sweep - FLAT_HEAD_ANGLE, abs=TOL)
    # Flat: centered on the plane through pos
    z = mesh.vertices[:, 2]
    assert z.min() == pytest.approx(3.0 - THICKNESS / 2, abs=TOL)
    assert z.max() == pytest.approx(3.0 + THICKNESS / 2, abs=TOL)


def test_flat_arc_band_and_head_widths():
    mesh = flat_arc_arrow_mesh(
        ORIGIN, ARC_RADIUS, math.radians(45), WIDTH, THICKNESS, head_width_ratio=2.5
    )
    r = radial(mesh)
    assert r.min() == pytest.approx(ARC_RADIUS - WIDTH * 2.5 / 2, abs=TOL)
    assert r.max() == pytest.approx(ARC_RADIUS + WIDTH * 2.5 / 2, abs=TOL)
    # The band itself is `WIDTH` wide: its vertices at the start of the arc
    start = mesh.vertices[np.isclose(angles(mesh), 0.0)]
    assert radial(mesh)[np.isclose(angles(mesh), 0.0)].min() == pytest.approx(
        ARC_RADIUS - WIDTH / 2, abs=TOL
    )
    assert len(start) > 0


@pytest.mark.parametrize("sweep", [math.radians(45), -math.radians(45)])
def test_flat_arc_is_closed_with_outward_normals(sweep):
    mesh = flat_arc_arrow_mesh(ORIGIN, ARC_RADIUS, sweep, WIDTH, THICKNESS)
    assert mesh.is_watertight
    assert mesh.volume > 0.0


@pytest.mark.parametrize(
    "radius, sweep, width, thickness",
    [
        (ARC_RADIUS, 0.0, WIDTH, THICKNESS),
        (0.0, 1.0, WIDTH, THICKNESS),
        (ARC_RADIUS, 1.0, 0.0, THICKNESS),
        (ARC_RADIUS, 1.0, WIDTH, 0.0),
    ],
)
def test_flat_arc_rejects_degenerate_dimensions(radius, sweep, width, thickness):
    with pytest.raises(ValueError):
        flat_arc_arrow_mesh(ORIGIN, radius, sweep, width, thickness)


"""
Colors (shared by every builder)
"""


@pytest.mark.parametrize(
    "build",
    [
        lambda color: arrow_mesh(ORIGIN, (0.0, 0.0, 0.3), 0.01, color=color),
        lambda color: flat_arc_arrow_mesh(
            ORIGIN, ARC_RADIUS, math.radians(45), WIDTH, THICKNESS, color=color
        ),
    ],
    ids=["arrow", "flat_arc"],
)
def test_color_is_applied_to_every_vertex_or_left_off(build):
    color = (0.0, 0.5, 0.0, 1.0)
    mesh = build(color)
    assert np.all(mesh.visual.vertex_colors == trimesh.visual.color.to_rgba(color))
    assert build(None).visual.kind is None
