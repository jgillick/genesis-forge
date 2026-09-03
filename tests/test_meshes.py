"""
Tests for the trimesh builders in `genesis_forge.meshes`.

These check the geometry of the returned meshes directly and never render anything.
"""

import math

import numpy as np
import pytest
import trimesh

from genesis_forge.meshes import arc_arrow_mesh, arrow_mesh, yaw_pose, z_aligned_pose

TOL = 1e-6


def angles(mesh: trimesh.Trimesh) -> np.ndarray:
    """The angle of each vertex about the Z axis, in radians from +X"""
    return np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0])


def radial(mesh: trimesh.Trimesh) -> np.ndarray:
    """The distance of each vertex from the Z axis"""
    return np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])


"""
arrow_mesh
"""


def test_arrow_spans_the_origin_to_its_length():
    mesh = arrow_mesh(0.3, 0.01, head_radius_ratio=2.0)
    z = mesh.vertices[:, 2]
    assert z.min() == pytest.approx(0.0)
    assert z.max() == pytest.approx(0.3)
    assert radial(mesh).max() == pytest.approx(0.02)


def test_arrow_head_length_comes_from_the_radius_not_the_length():
    mesh = arrow_mesh(1.0, 0.01, head_length_ratio=4.0)
    # The shaft vertices are the ones at the shaft radius; the head starts where they end
    shaft_z = mesh.vertices[np.isclose(radial(mesh), 0.01)][:, 2]
    assert shaft_z.max() == pytest.approx(1.0 - 0.04)


def test_arrow_head_is_clamped_for_short_arrows():
    mesh = arrow_mesh(0.02, 0.01, head_length_ratio=4.0, max_head_fraction=0.5)
    shaft_z = mesh.vertices[np.isclose(radial(mesh), 0.01)][:, 2]
    assert shaft_z.max() == pytest.approx(0.01)
    assert mesh.vertices[:, 2].max() == pytest.approx(0.02)


def test_arrow_head_radius_does_not_depend_on_the_length():
    for length in (1.0, 0.04, 0.01):
        mesh = arrow_mesh(length, 0.01, head_radius_ratio=2.0)
        assert radial(mesh).max() == pytest.approx(0.02)


def test_arrow_sections_control_the_resolution():
    coarse = arrow_mesh(0.3, 0.01, sections=12)
    fine = arrow_mesh(0.3, 0.01, sections=32)
    assert len(fine.vertices) > len(coarse.vertices)
    assert coarse.is_watertight
    assert fine.is_watertight


def test_arrow_color_is_applied_to_every_vertex():
    color = (0.0, 0.5, 0.0, 1.0)
    mesh = arrow_mesh(0.3, 0.01, color=color)
    expected = trimesh.visual.color.to_rgba(color)
    assert np.all(mesh.visual.vertex_colors == expected)


def test_arrow_without_color_is_left_uncolored():
    mesh = arrow_mesh(0.3, 0.01)
    assert mesh.visual.kind is None


@pytest.mark.parametrize("length, radius", [(0.0, 0.01), (-0.1, 0.01), (0.3, 0.0)])
def test_arrow_rejects_non_positive_dimensions(length, radius):
    with pytest.raises(ValueError):
        arrow_mesh(length, radius)


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
    mesh = arc_arrow_mesh(ARC_RADIUS, sweep, TUBE_RADIUS, head_length_ratio=5.0)
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
    mesh = arc_arrow_mesh(ARC_RADIUS, sweep, TUBE_RADIUS, head_length_ratio=5.0)
    a = angles(mesh)
    assert a.max() == pytest.approx(0.0, abs=TOL)
    assert a.min() == pytest.approx(sweep - HEAD_ANGLE, abs=TOL)


def test_arc_negative_sweep_mirrors_the_positive_sweep():
    sweep = math.radians(30)
    ccw = arc_arrow_mesh(ARC_RADIUS, sweep, TUBE_RADIUS)
    cw = arc_arrow_mesh(ARC_RADIUS, -sweep, TUBE_RADIUS)
    mirrored = cw.bounds * np.array([1.0, -1.0, 1.0])
    assert np.allclose(np.sort(mirrored, axis=0), ccw.bounds, atol=TOL)


def test_arc_stays_near_its_ring():
    mesh = arc_arrow_mesh(
        ARC_RADIUS, math.radians(60), TUBE_RADIUS, head_radius_ratio=2.5
    )
    head_radius = TUBE_RADIUS * 2.5
    r = radial(mesh)
    assert r.min() >= ARC_RADIUS - head_radius - TOL
    assert r.max() <= ARC_RADIUS + head_radius + TOL
    assert np.abs(mesh.vertices[:, 2]).max() <= head_radius + TOL


def test_arc_resolution_scales_with_the_sweep():
    short = arc_arrow_mesh(ARC_RADIUS, math.radians(10), TUBE_RADIUS)
    long = arc_arrow_mesh(ARC_RADIUS, math.radians(45), TUBE_RADIUS)
    assert len(long.vertices) > len(short.vertices)


def test_arc_color_is_applied_to_every_vertex():
    color = (0.0, 0.0, 0.5, 1.0)
    mesh = arc_arrow_mesh(ARC_RADIUS, math.radians(45), TUBE_RADIUS, color=color)
    expected = trimesh.visual.color.to_rgba(color)
    assert np.all(mesh.visual.vertex_colors == expected)


@pytest.mark.parametrize(
    "radius, sweep, tube_radius",
    [(ARC_RADIUS, 0.0, TUBE_RADIUS), (0.0, 1.0, TUBE_RADIUS), (ARC_RADIUS, 1.0, 0.0)],
)
def test_arc_rejects_degenerate_dimensions(radius, sweep, tube_radius):
    with pytest.raises(ValueError):
        arc_arrow_mesh(radius, sweep, tube_radius)


"""
Pose helpers
"""


@pytest.mark.parametrize(
    "direction",
    [
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_z_aligned_pose_points_z_along_the_direction(direction):
    pos = np.array([1.0, 2.0, 3.0])
    T = z_aligned_pose(pos, direction)
    R = T[:3, :3]
    unit = np.asarray(direction) / np.linalg.norm(direction)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], pos)
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(R[:, 2], unit)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_z_aligned_pose_rejects_a_zero_direction():
    with pytest.raises(ValueError):
        z_aligned_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_yaw_pose_rotates_about_z():
    pos = np.array([1.0, 2.0, 3.0])
    yaw = math.radians(90)
    T = yaw_pose(pos, yaw)
    assert np.allclose(T[:3, 3], pos)
    assert np.allclose(T[:3, 0], [math.cos(yaw), math.sin(yaw), 0.0])
    assert np.allclose(T[:3, 2], [0.0, 0.0, 1.0])
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])
