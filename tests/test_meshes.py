"""
Tests for the trimesh builders in `genesis_forge.meshes`.

These check the geometry of the returned meshes directly and never render anything.
"""

import math

import numpy as np
import pytest
import trimesh

from genesis_forge.meshes import arc_arrow_mesh, arrow_mesh

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


def test_arrow_spans_its_base_to_its_tip():
    mesh = arrow_mesh(ORIGIN, (0.0, 0.0, 0.3), 0.01, head_radius_ratio=2.0)
    z = mesh.vertices[:, 2]
    assert z.min() == pytest.approx(0.0)
    assert z.max() == pytest.approx(0.3)
    assert radial(mesh).max() == pytest.approx(0.02)


def test_arrow_is_placed_at_pos_and_points_along_vec():
    pos = np.array([1.0, 2.0, 3.0])
    vec = np.array([0.3, 0.4, 0.0])
    mesh = arrow_mesh(pos, vec, 0.01)
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


def test_arrow_sections_control_the_resolution():
    coarse = arrow_mesh(ORIGIN, (0.0, 0.0, 0.3), 0.01, sections=12)
    fine = arrow_mesh(ORIGIN, (0.0, 0.0, 0.3), 0.01, sections=32)
    assert len(fine.vertices) > len(coarse.vertices)
    assert coarse.is_watertight
    assert fine.is_watertight


def test_arrow_color_is_applied_to_every_vertex():
    color = (0.0, 0.5, 0.0, 1.0)
    mesh = arrow_mesh(ORIGIN, (0.0, 0.0, 0.3), 0.01, color=color)
    expected = trimesh.visual.color.to_rgba(color)
    assert np.all(mesh.visual.vertex_colors == expected)


def test_arrow_without_color_is_left_uncolored():
    mesh = arrow_mesh(ORIGIN, (0.0, 0.0, 0.3), 0.01)
    assert mesh.visual.kind is None


@pytest.mark.parametrize(
    "vec, radius", [((0.0, 0.0, 0.0), 0.01), ((0.0, 0.0, 0.3), 0.0)]
)
def test_arrow_rejects_degenerate_dimensions(vec, radius):
    with pytest.raises(ValueError):
        arrow_mesh(ORIGIN, vec, radius)


"""
arc_arrow_mesh
"""

ARC_RADIUS = 0.2
TUBE_RADIUS = 0.005
HEAD_LENGTH = TUBE_RADIUS * 5.0
HEAD_ANGLE = math.atan(HEAD_LENGTH / ARC_RADIUS)
"""The angle the arrowhead tip extends past the end of the arc"""


def test_arc_positive_sweep_starts_at_x_and_heads_counter_clockwise():
    sweep = math.radians(45)
    mesh = arc_arrow_mesh(ORIGIN, ARC_RADIUS, sweep, TUBE_RADIUS, head_length_ratio=5.0)
    a = angles(mesh)
    assert a.min() == pytest.approx(0.0, abs=TOL)
    assert a.max() == pytest.approx(sweep + HEAD_ANGLE, abs=TOL)
    # The farthest vertex is the cone tip, on the tangent at the end of the arc
    tip = mesh.vertices[np.argmax(a)]
    assert np.hypot(tip[0], tip[1]) == pytest.approx(
        math.hypot(ARC_RADIUS, HEAD_LENGTH), abs=TOL
    )


def test_arc_negative_sweep_starts_at_x_and_heads_clockwise():
    sweep = -math.radians(45)
    mesh = arc_arrow_mesh(ORIGIN, ARC_RADIUS, sweep, TUBE_RADIUS, head_length_ratio=5.0)
    a = angles(mesh)
    assert a.max() == pytest.approx(0.0, abs=TOL)
    assert a.min() == pytest.approx(sweep - HEAD_ANGLE, abs=TOL)


def test_arc_negative_sweep_mirrors_the_positive_sweep():
    sweep = math.radians(30)
    ccw = arc_arrow_mesh(ORIGIN, ARC_RADIUS, sweep, TUBE_RADIUS)
    cw = arc_arrow_mesh(ORIGIN, ARC_RADIUS, -sweep, TUBE_RADIUS)
    mirrored = cw.bounds * np.array([1.0, -1.0, 1.0])
    assert np.allclose(np.sort(mirrored, axis=0), ccw.bounds, atol=TOL)


@pytest.mark.parametrize("sweep", [math.radians(45), -math.radians(45)])
def test_arc_is_centered_on_pos_and_starts_at_the_start_angle(sweep):
    pos = np.array([1.0, 2.0, 3.0])
    start = math.radians(120)
    mesh = arc_arrow_mesh(pos, ARC_RADIUS, sweep, TUBE_RADIUS, start_angle=start)
    # Same shape as the arc at the origin, rotated by the start angle
    a = angles(mesh, pos) - start
    a = np.arctan2(np.sin(a), np.cos(a))  # wrap to [-pi, pi]
    if sweep > 0.0:
        assert a.min() == pytest.approx(0.0, abs=TOL)
        assert a.max() == pytest.approx(sweep + HEAD_ANGLE, abs=TOL)
    else:
        assert a.max() == pytest.approx(0.0, abs=TOL)
        assert a.min() == pytest.approx(sweep - HEAD_ANGLE, abs=TOL)
    assert mesh.vertices[:, 2].mean() == pytest.approx(3.0, abs=TOL)
    assert radial(mesh, pos).min() == pytest.approx(
        ARC_RADIUS - TUBE_RADIUS * 2.5, abs=TOL
    )


def test_arc_stays_near_its_ring():
    mesh = arc_arrow_mesh(
        ORIGIN, ARC_RADIUS, math.radians(60), TUBE_RADIUS, head_radius_ratio=2.5
    )
    head_radius = TUBE_RADIUS * 2.5
    r = radial(mesh)
    assert r.min() >= ARC_RADIUS - head_radius - TOL
    assert r.max() <= ARC_RADIUS + head_radius + TOL
    assert np.abs(mesh.vertices[:, 2]).max() <= head_radius + TOL


def test_arc_resolution_scales_with_the_sweep():
    short = arc_arrow_mesh(ORIGIN, ARC_RADIUS, math.radians(10), TUBE_RADIUS)
    long = arc_arrow_mesh(ORIGIN, ARC_RADIUS, math.radians(45), TUBE_RADIUS)
    assert len(long.vertices) > len(short.vertices)


def test_arc_color_is_applied_to_every_vertex():
    color = (0.0, 0.0, 0.5, 1.0)
    mesh = arc_arrow_mesh(
        ORIGIN, ARC_RADIUS, math.radians(45), TUBE_RADIUS, color=color
    )
    expected = trimesh.visual.color.to_rgba(color)
    assert np.all(mesh.visual.vertex_colors == expected)


@pytest.mark.parametrize(
    "radius, sweep, tube_radius",
    [(ARC_RADIUS, 0.0, TUBE_RADIUS), (0.0, 1.0, TUBE_RADIUS), (ARC_RADIUS, 1.0, 0.0)],
)
def test_arc_rejects_degenerate_dimensions(radius, sweep, tube_radius):
    with pytest.raises(ValueError):
        arc_arrow_mesh(ORIGIN, radius, sweep, tube_radius)
