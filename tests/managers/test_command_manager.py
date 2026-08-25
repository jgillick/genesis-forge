"""Behavior of CommandManager and VelocityCommandManager: range validation,
resampling, the external-controller/gamepad override paths, and (for the velocity
variant) forcing standing environments to a zero command.

Uses a FakeGamepad -- no Genesis scene is built. Ranges are chosen with equal
min/max (e.g. (2.0, 2.0)) wherever a sampled value needs to be deterministic,
since resample_command draws from `torch.uniform_(min, max)`.
"""

import pytest
import torch

from genesis_forge.managers import CommandManager, VelocityCommandManager


class FakeGamepad:
    def __init__(self, axis_values: dict[int, float]):
        self._axis_values = axis_values

    def axis(self, index: int) -> float:
        return self._axis_values[index]


"""
Construction
"""


def test_single_range_allocates_one_command_column(env):
    mgr = CommandManager(env, range=(0.0, 1.0))
    assert mgr.command.shape == (env.num_envs, 1)


def test_dict_range_allocates_one_column_per_key(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0), "b": (-1.0, 1.0)})
    assert mgr.command.shape == (env.num_envs, 2)


def test_resample_time_sec_computes_resample_steps(env):
    mgr = CommandManager(env, range=(0.0, 1.0), resample_time_sec=0.1)
    assert mgr._resample_steps == int(0.1 / env.dt)


"""
range property -- validated replacement
"""


def test_range_setter_accepts_a_valid_replacement(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0)})
    mgr.range = {"a": (-1.0, 2.0)}
    assert mgr.range == {"a": (-1.0, 2.0)}


def test_range_setter_rejects_a_size_change(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0), "b": (0.0, 1.0)})
    with pytest.raises(ValueError, match="shape"):
        mgr.range = {"a": (0.0, 1.0)}


def test_range_setter_rejects_a_type_change(env):
    mgr = CommandManager(env, range=(0.0, 1.0))
    with pytest.raises(ValueError, match="base type"):
        mgr.range = {"a": (0.0, 1.0)}


def test_range_setter_rejects_different_dict_keys(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0)})
    with pytest.raises(ValueError, match="dict keys"):
        mgr.range = {"b": (0.0, 1.0)}


"""
get_command / set_command -- dict ranges only
"""


def test_get_command_returns_the_value_for_a_key(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0), "b": (0.0, 1.0)})
    mgr._command[:, 0] = 5.0
    assert torch.equal(mgr.get_command("a"), torch.full((env.num_envs,), 5.0))


def test_get_command_requires_a_dict_range(env):
    mgr = CommandManager(env, range=(0.0, 1.0))
    with pytest.raises(TypeError, match="not a dict"):
        mgr.get_command("a")


def test_set_command_updates_all_envs_by_default(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0)})
    mgr.set_command("a", 3.0)
    assert torch.equal(mgr.get_command("a"), torch.full((env.num_envs,), 3.0))


def test_set_command_updates_only_the_given_envs(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0)})
    mgr.set_command("a", 3.0, envs_idx=[0])
    result = mgr.get_command("a")
    assert result[0].item() == 3.0
    assert torch.all(result[1:] == 0.0)


def test_get_command_idx_returns_the_keys_position(env):
    mgr = CommandManager(env, range={"a": (0.0, 1.0), "b": (0.0, 1.0)})
    assert mgr.get_command_idx("b") == 1


"""
increment_range
"""


def test_increment_range_single_float_applies_symmetrically(env):
    mgr = CommandManager(env, range={"height": (0.0, 1.0)})
    mgr.increment_range("height", 0.5)
    assert mgr.range["height"] == [-0.5, 1.5]


def test_increment_range_tuple_applies_asymmetrically(env):
    mgr = CommandManager(env, range={"height": (0.0, 1.0)})
    mgr.increment_range("height", (-0.25, 1.0))
    assert mgr.range["height"] == [-0.25, 2.0]


def test_increment_range_respects_the_limit_in_each_direction(env):
    mgr = CommandManager(env, range={"height": (0.0, 1.0)})
    mgr.increment_range("height", (-0.25, 1.0), limit=(-0.1, 1.5))
    assert mgr.range["height"] == [-0.1, 1.5]


def test_increment_range_requires_a_dict_range(env):
    mgr = CommandManager(env, range=(0.0, 1.0))
    with pytest.raises(TypeError, match="non-dict"):
        mgr.increment_range("height", 0.5)


def test_increment_range_raises_for_an_unknown_key(env):
    mgr = CommandManager(env, range={"height": (0.0, 1.0)})
    with pytest.raises(ValueError, match="not found"):
        mgr.increment_range("nonexistent", 0.5)


"""
resample_command / step / reset
"""


def test_resample_command_only_touches_the_given_envs(env):
    mgr = CommandManager(env, range=(2.0, 2.0))  # min == max: deterministic sample
    mgr.resample_command([0, 2])

    result = mgr.command
    assert result[0, 0].item() == 2.0
    assert result[2, 0].item() == 2.0
    assert result[1, 0].item() == 0.0
    assert result[3, 0].item() == 0.0


def test_step_resamples_only_envs_whose_episode_length_hits_the_interval(env):
    env.episode_length = torch.tensor([0, 1, 2, 3])
    mgr = CommandManager(env, range=(2.0, 2.0), resample_time_sec=2 * env.dt)  # resample_steps=2

    mgr.step()

    result = mgr.command
    assert result[0, 0].item() == 2.0  # 0 % 2 == 0
    assert result[2, 0].item() == 2.0  # 2 % 2 == 0
    assert result[1, 0].item() == 0.0
    assert result[3, 0].item() == 0.0


def test_step_is_a_noop_when_disabled(env):
    env.episode_length = torch.zeros(env.num_envs, dtype=torch.long)
    mgr = CommandManager(env, range=(2.0, 2.0))
    mgr.enabled = False

    mgr.step()

    assert torch.all(mgr.command == 0.0)


def test_step_is_a_noop_with_an_external_controller(env):
    env.episode_length = torch.zeros(env.num_envs, dtype=torch.long)
    env.step_count = 0
    mgr = CommandManager(env, range=(2.0, 2.0))
    mgr.use_external_controller(lambda step: torch.full((env.num_envs, 1), 9.0))

    mgr.step()  # must not resample the internal buffer

    assert torch.all(mgr._command == 0.0)
    assert torch.all(mgr.command == 9.0)  # .command still reads through the controller


def test_reset_resamples_every_env_by_default(env):
    mgr = CommandManager(env, range=(2.0, 2.0))
    mgr.reset()
    assert torch.all(mgr.command == 2.0)


def test_reset_is_a_noop_when_disabled(env):
    mgr = CommandManager(env, range=(2.0, 2.0))
    mgr.enabled = False
    mgr.reset()
    assert torch.all(mgr.command == 0.0)


def test_observation_returns_the_command(env):
    mgr = CommandManager(env, range=(2.0, 2.0))
    mgr.reset()
    assert torch.equal(mgr.observation(env), mgr.command)


"""
External controller
"""


def test_command_property_reads_through_the_external_controller(env):
    env.step_count = 0
    mgr = CommandManager(env, range=(0.0, 1.0))
    mgr.use_external_controller(lambda step: torch.full((env.num_envs, 1), 7.0))
    assert torch.all(mgr.command == 7.0)


"""
Gamepad -- axis values are converted into the configured range
"""


def test_use_gamepad_single_axis_maps_to_the_full_range(env):
    env.step_count = 0
    mgr = CommandManager(env, range=(-2.0, 2.0))
    gamepad = FakeGamepad({0: 1.0})  # full-forward
    mgr.use_gamepad(gamepad, range_axis=0)

    assert torch.allclose(mgr.command, torch.full((env.num_envs, 1), 2.0))


def test_use_gamepad_inverts_the_axis_when_requested(env):
    env.step_count = 0
    mgr = CommandManager(env, range=(-2.0, 2.0))
    gamepad = FakeGamepad({0: 1.0})
    mgr.use_gamepad(gamepad, range_axis=0, invert_axis=True)

    assert torch.allclose(mgr.command, torch.full((env.num_envs, 1), -2.0))


def test_use_gamepad_maps_multiple_axes_by_dict(env):
    env.step_count = 0
    mgr = CommandManager(env, range={"a": (-1.0, 1.0), "b": (-1.0, 1.0)})
    gamepad = FakeGamepad({0: 1.0, 1: -1.0})
    mgr.use_gamepad(gamepad, range_axis={"a": 0, "b": 1})

    # get_command() reads the internal buffer directly, which the gamepad override
    # bypasses entirely -- only the `command` property reads through it.
    command = mgr.command
    assert command[0, mgr.get_command_idx("a")].item() == pytest.approx(1.0)
    assert command[0, mgr.get_command_idx("b")].item() == pytest.approx(-1.0)


"""
VelocityCommandManager -- standing environments are forced to a zero command
"""

VELOCITY_RANGE = {
    "lin_vel_x": (3.0, 3.0),  # min == max: deterministic sample
    "lin_vel_y": (3.0, 3.0),
    "ang_vel_z": (3.0, 3.0),
}


def test_velocity_command_forces_standing_envs_to_zero(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=1.0)
    mgr.resample_command([0, 1, 2, 3])
    assert torch.all(mgr.command == 0.0)


def test_velocity_command_keeps_the_sampled_value_when_never_standing(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=0.0)
    mgr.resample_command([0, 1, 2, 3])
    assert torch.all(mgr.command == 3.0)


def test_standing_envs_reflects_which_envs_were_marked_standing(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=1.0)
    mgr.resample_command([0, 1, 2, 3])
    assert mgr.standing_envs.tolist() == [True, True, True, True]


def test_standing_envs_is_false_for_every_env_when_never_standing(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=0.0)
    mgr.resample_command([0, 1, 2, 3])
    assert mgr.standing_envs.tolist() == [False, False, False, False]


def test_standing_envs_defaults_to_false_before_any_resample(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=1.0)
    assert mgr.standing_envs.tolist() == [False, False, False, False]


def test_standing_envs_only_updates_the_resampled_envs(env):
    """A partial resample must not touch other envs' standing state."""
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=1.0)
    mgr.resample_command([0, 2])
    assert mgr.standing_envs.tolist() == [True, False, True, False]

    # Now resample the remaining envs with standing_probability=0.0 -- envs 0 and 2
    # must keep their earlier "standing" state since they aren't in this call.
    mgr.standing_probability = 0.0
    mgr.resample_command([1, 3])
    assert mgr.standing_envs.tolist() == [True, False, True, False]


def test_build_without_debug_visualizer_is_a_noop(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE)
    mgr.build()  # must not raise -- no scene is available in this fake env


def test_use_gamepad_maps_the_three_velocity_axes_inverted(env):
    env.step_count = 0
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE)
    gamepad = FakeGamepad({0: 0.0, 1: 0.0, 2: 0.0})
    mgr.use_gamepad(gamepad)

    cfg = mgr._gamepad_cfg
    # range_axis maps lin_vel_x->1, lin_vel_y->0, ang_vel_z->2 (the defaults); the base
    # use_gamepad then re-orders them to match self._range's own key order.
    assert cfg["axis_map"] == [1, 0, 2]
    assert cfg["axis_invert_map"] == [True, True, True]


"""
VelocityCommandManager -- range property setter delegates to CommandManager's
validation
"""


def test_velocity_range_setter_accepts_a_valid_replacement(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE)
    new_range = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-1.0, 1.0),
    }
    mgr.range = new_range
    assert mgr.range == new_range


def test_velocity_range_setter_rejects_a_size_change(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE)
    with pytest.raises(ValueError, match="shape"):
        mgr.range = {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0)}


def test_velocity_range_setter_rejects_different_dict_keys(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE)
    with pytest.raises(ValueError, match="dict keys"):
        mgr.range = {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "wrong_key": (-1.0, 1.0),
        }
