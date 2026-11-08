"""Helpers in genesis_forge.utils.

The name-pattern helpers back the joint/link selection used by every manager that
accepts name patterns, so the exact-vs-regex and first-match-wins rules are pinned here.
"""

import math
import re

import pytest
import torch
from genesis.utils.geom import transform_by_quat, xyz_to_quat

from genesis_forge.utils import assign_by_pattern, heading_from_quat, name_matches

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
