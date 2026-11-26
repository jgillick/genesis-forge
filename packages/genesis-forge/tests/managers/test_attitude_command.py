"""Behavior of AttitudeCommandManager: the pitch/roll accessors, leveling environments
with level_probability, the level/tilted `when` functions, and the gamepad mapping.

Ranges use equal min/max (e.g. (0.2, 0.2)) wherever a sampled value needs to be
deterministic. No Genesis scene is built.
"""

import pytest
import torch

from genesis_forge.managers import AttitudeCommandManager


class FakeGamepad:
    def __init__(self, axis_values: dict[int, float]):
        self._axis_values = axis_values

    def axis(self, index: int) -> float:
        return self._axis_values[index]


def make(env, pitch=(0.2, 0.2), roll=(-0.1, -0.1), **kwargs):
    return AttitudeCommandManager(env, range={"pitch": pitch, "roll": roll}, **kwargs)


def test_pitch_and_roll_read_their_own_command_column(env):
    mgr = make(env)
    mgr.reset()

    assert mgr.command.shape == (env.num_envs, 2)
    assert torch.allclose(mgr.pitch, torch.full((env.num_envs,), 0.2))
    assert torch.allclose(mgr.roll, torch.full((env.num_envs,), -0.1))


def test_level_probability_of_one_levels_every_env(env):
    mgr = make(env, level_probability=1.0)
    mgr.reset()

    assert torch.equal(mgr.command, torch.zeros(env.num_envs, 2))
    assert mgr.level_envs().all()


def test_level_probability_of_zero_keeps_every_sampled_tilt(env):
    mgr = make(env, level_probability=0.0)
    mgr.reset()

    assert not mgr.level_envs().any()


def test_only_the_resampled_envs_are_leveled(env):
    mgr = make(env)
    mgr.reset()
    mgr.level_probability = 1.0

    mgr.resample_command(torch.tensor([1, 2]))

    assert mgr.level_envs().tolist() == [False, True, True, False]


def test_level_envs_needs_both_angles_within_the_threshold(env):
    mgr = make(env, pitch=(0.0, 0.0), roll=(0.0, 0.0))
    mgr.reset()
    mgr._command[1, 0] = 0.05  # pitched only
    mgr._command[2, 1] = 0.05  # rolled only

    assert mgr.level_envs(threshold=0.01).tolist() == [True, False, False, True]
    assert mgr.level_envs(threshold=0.1).all()


def test_when_level_and_when_tilted_read_this_command(env):
    mgr = make(env)
    mgr.reset()
    mgr._command[0, :] = 0.0

    assert mgr.when_level()(env).tolist() == [True, False, False, False]
    assert mgr.when_tilted()(env).tolist() == [False, True, True, True]


def test_use_gamepad_pushing_forward_pitches_the_nose_down(env):
    """A stick reads -1 pushed forward, which maps to the top of the pitch range."""
    mgr = make(env, pitch=(-0.4, 0.4), roll=(-0.2, 0.2))
    mgr.use_gamepad(FakeGamepad({3: -1.0, 2: 1.0}), pitch_axis=3, roll_axis=2)

    assert torch.allclose(mgr.pitch, torch.full((env.num_envs,), 0.4))
    assert torch.allclose(mgr.roll, torch.full((env.num_envs,), 0.2))


def test_use_gamepad_holds_an_unmapped_angle_level(env):
    mgr = make(env, pitch=(-0.4, 0.4), roll=(-0.2, 0.2))
    mgr.use_gamepad(FakeGamepad({3: 0.5}), pitch_axis=3)

    assert torch.allclose(mgr.pitch, torch.full((env.num_envs,), -0.2))
    assert torch.allclose(mgr.roll, torch.zeros(env.num_envs))


def test_range_setter_rejects_different_keys(env):
    mgr = make(env)
    with pytest.raises(ValueError, match="dict keys"):
        mgr.range = {"pitch": (0.0, 0.1), "yaw": (0.0, 0.1)}
