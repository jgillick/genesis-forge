"""Helpers in genesis_forge.utils.

The name-pattern helpers back the joint/link selection used by every manager that
accepts name patterns, so the exact-vs-regex and first-match-wins rules are pinned here.
"""

import math
import re
from types import SimpleNamespace
from typing import Any

import genesis as gs
import pytest
import torch
from genesis.utils.geom import inv_quat, transform_by_quat, xyz_to_quat

from genesis_forge.utils import (
    apply_external_wrench,
    assign_by_pattern,
    debug_envs_idx,
    entity_ang_vel,
    entity_lin_vel,
    heading_from_quat,
    inf_to_none,
    name_matches,
    to_heading_frame,
)

"""
name_matches
"""


def test_exact_name_matches():
    assert name_matches("wheel1", "wheel1")


def test_non_matching_name_is_rejected():
    assert not name_matches("wheel1", "wheel2")


def test_regex_pattern_matches():
    assert name_matches("wheel3", "wheel[1-4]")
    assert name_matches("TT_Motor-2_axel", "TT_Motor-[1-4]_axel")


def test_pattern_is_fully_anchored():
    """A partial match is not a match -- 'foot' must not select 'foot_link'."""
    assert not name_matches("FL_foot_link", "FL_foot")
    assert not name_matches("prefix_wheel1", "wheel1")


def test_a_valid_regex_name_matches_itself_exactly():
    """`joint(left)` is valid regex (a group), and the exact arm matches it first."""
    assert name_matches("joint(left)", "joint(left)")


"""
assign_by_pattern
"""


def test_assigns_values_to_every_matching_name():
    assert assign_by_pattern(["a1", "a2", "b1"], {"a[0-9]": 10.0, "b1": 20.0}) == [
        10.0,
        10.0,
        20.0,
    ]


def test_first_matching_pattern_wins():
    """Specific patterns can be listed before a catch-all and take precedence."""
    names = ["hip", "knee_l", "knee_r"]
    assert assign_by_pattern(names, {"knee_.*": 30.0, ".*": 50.0}) == [50.0, 30.0, 30.0]


def test_unmatched_names_are_none():
    assert assign_by_pattern(["a", "b"], {"a": 1.0}) == [1.0, None]


def test_a_pattern_matching_nothing_raises():
    with pytest.raises(RuntimeError, match="nonexistent"):
        assign_by_pattern(["a", "b"], {"nonexistent": 1.0})


def test_a_pattern_fully_shadowed_by_an_earlier_one_raises():
    """A shadowed pattern is dead config, so it is reported the same as a typo."""
    with pytest.raises(RuntimeError, match="knee_.*"):
        assign_by_pattern(["knee_l"], {".*": 50.0, "knee_.*": 30.0})


def test_values_can_be_any_type():
    out = assign_by_pattern(["a", "b"], {"a": (1.0, 2.0), "b": (3.0, 4.0)})
    assert out == [(1.0, 2.0), (3.0, 4.0)]


def test_empty_names_with_a_pattern_raises():
    with pytest.raises(RuntimeError):
        assign_by_pattern([], {"a": 1.0})


def test_empty_config_leaves_everything_unassigned():
    assert assign_by_pattern(["a", "b"], {}) == [None, None]


"""
Malformed patterns
"""


def test_an_invalid_regex_pattern_raises():
    """A malformed pattern is a configuration error, and fails loudly."""
    with pytest.raises(re.error):
        name_matches("chassis", "axel(.*")


def test_an_invalid_pattern_still_matches_its_own_literal_name():
    """The exact-name check runs first, so only other names reach the regex."""
    assert name_matches("axel(2", "axel(2")


def test_assign_by_pattern_propagates_an_invalid_pattern():
    with pytest.raises(re.error):
        assign_by_pattern(["chassis", "wheel1"], {"axel(.*": 1.0})


"""
heading_from_quat
"""


def test_heading_from_quat_is_where_the_nose_points():
    """
    The heading is the entity's nose flattened onto the ground, so tilt moves it. The
    euler yaw -- `quat_to_xyz(quat)[:, 2]` -- does not, which is why it isn't used here:
    swap one for the other and a tilted robot's heading error goes wrong.
    """
    # Yawed 40 degrees, and pitched 45 degrees nose-up
    quat = xyz_to_quat(torch.tensor([[0.0, 45.0, 40.0]]), degrees=True)
    nose = transform_by_quat(torch.tensor([[1.0, 0.0, 0.0]]), quat)[0]

    heading = heading_from_quat(quat).item()

    assert math.isclose(heading, math.atan2(nose[1], nose[0]), abs_tol=1e-5)
    # The euler yaw of this orientation is exactly the 40 degrees it was built from; the
    # nose is further round than that, because pitching it up swings it away
    assert heading > math.radians(45)


def test_heading_from_quat_is_the_yaw_for_an_upright_entity():
    quat = torch.tensor([[math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]])

    assert math.isclose(heading_from_quat(quat).item(), math.pi / 4, abs_tol=1e-5)


"""
to_heading_frame
"""


def test_to_heading_frame_keeps_level_motion_level_however_the_entity_tilts():
    """
    Walking level along its heading, pitched and rolled: in the heading frame that's
    straight ahead, and a yaw turn stays a pure yaw turn.
    """
    quat = xyz_to_quat(torch.tensor([[15.0, -25.0, 40.0]]), rpy=True, degrees=True)
    heading = heading_from_quat(quat).item()
    vec = torch.tensor(
        [[2.0 * math.cos(heading), 2.0 * math.sin(heading), 0.0], [0.0, 0.0, 0.7]]
    )

    result = to_heading_frame(vec, quat.expand(2, 4))

    assert torch.allclose(
        result, torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.7]]), atol=1e-5
    )


def test_to_heading_frame_matches_the_base_frame_for_an_upright_entity():
    quat = xyz_to_quat(torch.tensor([[0.0, 0.0, 70.0]]), rpy=True, degrees=True)
    vec = torch.tensor([[0.3, -1.1, 0.4]])

    assert torch.allclose(
        to_heading_frame(vec, quat), transform_by_quat(vec, inv_quat(quat)), atol=1e-5
    )


def test_entity_velocities_reject_an_unknown_frame():
    entity = SimpleNamespace(
        get_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        get_vel=lambda: torch.zeros((1, 3)),
        get_ang=lambda: torch.zeros((1, 3)),
    )

    with pytest.raises(ValueError, match="body"):
        entity_lin_vel(entity, "body")
    with pytest.raises(ValueError, match="body"):
        entity_ang_vel(entity, "body")


"""
inf_to_none -- unbounded values in a deployment bundle
"""


def test_inf_to_none_replaces_either_infinity_with_none():
    values = torch.tensor([1.5, float("inf"), float("-inf"), -2.0])

    assert inf_to_none(values) == [1.5, None, None, -2.0]


def test_inf_to_none_accepts_a_plain_list():
    assert inf_to_none([0.0, float("-inf")]) == [0.0, None]


def test_inf_to_none_returns_plain_floats():
    """So the result serializes, even from a per-joint integer tensor."""
    result = inf_to_none(torch.tensor([1, 2]))

    assert result == [1.0, 2.0]
    assert all(type(value) is float for value in result)


"""
debug_envs_idx
"""


def scene_rendering(rendered_envs_idx) -> Any:
    return SimpleNamespace(
        vis_options=SimpleNamespace(rendered_envs_idx=rendered_envs_idx)
    )


def test_debug_envs_default_to_the_rendered_envs():
    assert debug_envs_idx(scene_rendering([0, 2]), 4, None) == [0, 2]


def test_debug_envs_default_to_every_env_when_rendering_is_not_restricted():
    assert debug_envs_idx(scene_rendering(None), 3, None) == [0, 1, 2]


def test_debug_envs_can_be_some_of_the_rendered_envs():
    assert debug_envs_idx(scene_rendering([0, 2]), 4, [2]) == [2]


def test_debug_envs_that_are_not_rendered_raise():
    with pytest.raises(ValueError, match=r"not rendered: \[1, 3\]"):
        debug_envs_idx(scene_rendering([0, 2]), 4, [0, 1, 3])


def test_debug_envs_can_be_any_env_when_rendering_is_not_restricted():
    assert debug_envs_idx(scene_rendering(None), 4, [3]) == [3]


"""
apply_external_wrench
"""


class LegacySolver:
    """The Genesis < 1.4 API: separate force and torque calls."""

    def __init__(self):
        self.calls = []

    def apply_links_external_force(self, force, links_idx, envs_idx, ref):
        self.calls.append(("force", links_idx, ref))

    def apply_links_external_torque(self, torque, links_idx, envs_idx):
        self.calls.append(("torque", links_idx))


def wrench_entity(solver):
    links = [SimpleNamespace(idx=10), SimpleNamespace(idx=11)]
    return SimpleNamespace(solver=solver, links=links)


def test_apply_external_wrench_requires_a_force_or_torque():
    with pytest.raises(ValueError, match="force or torque"):
        apply_external_wrench(wrench_entity(LegacySolver()), [0], torch.tensor([0]))


def test_apply_external_wrench_on_legacy_genesis_maps_local_links_to_global():
    solver = LegacySolver()
    wrench = torch.zeros((1, 1, 3))
    apply_external_wrench(
        wrench_entity(solver), [1], torch.tensor([0]), force=wrench, torque=wrench
    )
    assert solver.calls == [("force", [11], "link_com"), ("torque", [11])]


def test_apply_external_wrench_on_legacy_genesis_with_only_a_torque():
    solver = LegacySolver()
    apply_external_wrench(
        wrench_entity(solver), [0], torch.tensor([0]), torque=torch.zeros((1, 1, 3))
    )
    assert solver.calls == [("torque", [10])]


def test_apply_external_wrench_on_genesis_1_4_makes_one_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gs, "link_ref_frame", SimpleNamespace(link_COM="com"), raising=False
    )
    solver = SimpleNamespace(
        apply_links_external_wrench=lambda **kwargs: calls.append(kwargs)
    )
    torque = torch.zeros((1, 1, 3))
    apply_external_wrench(wrench_entity(solver), [1], torch.tensor([0]), torque=torque)
    assert len(calls) == 1
    assert calls[0]["force"] is None
    assert calls[0]["torque"] is torque
    assert calls[0]["links_idx"] == [11]
    assert calls[0]["ref"] == "com"
