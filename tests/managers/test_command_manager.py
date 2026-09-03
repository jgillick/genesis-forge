"""Behavior of CommandManager and VelocityCommandManager: range validation,
resampling, the external-controller/gamepad override paths, and (for the velocity
variant) forcing standing environments to a zero command.

Uses a FakeGamepad -- no Genesis scene is built. Ranges are chosen with equal
min/max (e.g. (2.0, 2.0)) wherever a sampled value needs to be deterministic,
since resample_command draws from `torch.uniform_(min, max)`.
"""

import math

import numpy as np
import pytest
import torch

from genesis_forge.managers import CommandManager, VelocityCommandManager


class FakeGamepad:
    def __init__(self, axis_values: dict[int, float]):
        self._axis_values = axis_values

    def axis(self, index: int) -> float:
        return self._axis_values[index]


class FakeScene:
    """Records the debug draw calls made by the velocity debug visualizer."""

    def __init__(self, num_envs: int):
        self.envs_offset = np.zeros((num_envs, 3))
        self.vis_options = None
        self.mesh_calls: list[tuple] = []
        self.cleared: list = []

    def draw_debug_arrow(self, pos, vec, radius, color):
        return ("arrow", tuple(pos), tuple(vec))

    def draw_debug_sphere(self, pos, radius, color):
        return ("sphere", tuple(pos))

    def draw_debug_mesh(self, mesh, T):
        self.mesh_calls.append((mesh, T))
        return ("mesh", len(self.mesh_calls))

    def clear_debug_object(self, node):
        self.cleared.append(node)


class FakeRobot:
    """A stationary robot at the origin with an identity orientation."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    def get_quat(self) -> torch.Tensor:
        quat = torch.zeros(self.num_envs, 4)
        quat[:, 0] = 1.0  # (w, x, y, z)
        return quat

    def get_pos(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3)

    def get_vel(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3)

    def get_ang(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3)


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
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=1.0)
    mgr.resample_command([0, 1, 2, 3])
    assert torch.all(mgr.command == 0.0)


def test_velocity_command_keeps_the_sampled_value_when_never_standing(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=0.0)
    mgr.resample_command([0, 1, 2, 3])
    assert torch.all(mgr.command == 3.0)


def test_stopped_envs_reflects_which_envs_were_marked_standing(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=1.0)
    mgr.resample_command([0, 1, 2, 3])
    assert mgr.stopped_envs().tolist() == [True, True, True, True]


def test_stopped_envs_is_false_for_every_env_when_never_standing(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=0.0)
    mgr.resample_command([0, 1, 2, 3])
    assert mgr.stopped_envs().tolist() == [False, False, False, False]


def test_stopped_envs_is_true_before_any_resample(env):
    """Before any resample the command buffer is all zeros: no movement is commanded."""
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=1.0)
    assert mgr.stopped_envs().tolist() == [True, True, True, True]


def test_stopped_envs_only_updates_the_resampled_envs(env):
    """A partial resample must not touch other envs' commands or stopped state."""
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=0.0)
    mgr.resample_command([1, 3])
    assert mgr.stopped_envs().tolist() == [True, False, True, False]

    # Now resample envs 0 and 2 with stopped_probability=1.0 -- envs 1 and 3
    # must keep their earlier commands since they aren't in this call.
    mgr.stopped_probability = 1.0
    mgr.resample_command([0, 2])
    assert mgr.stopped_envs().tolist() == [True, False, True, False]


def test_stopped_envs_is_true_when_the_range_samples_to_zero(env):
    """An env whose command samples to zero is stopped, even without stopped_probability."""
    zero_range = {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    }
    mgr = VelocityCommandManager(env, range=zero_range, stopped_probability=0.0)
    mgr.resample_command([0, 1, 2, 3])
    assert mgr.stopped_envs().tolist() == [True, True, True, True]


def test_stopped_envs_with_zero_threshold_matches_only_exactly_zero_commands(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=0.0)
    mgr.resample_command([0, 1])  # envs 2 and 3 keep their zero command
    assert mgr.stopped_envs(threshold=0.0).tolist() == [False, False, True, True]


def test_stopped_envs_is_false_while_only_angular_velocity_is_commanded(env):
    """Turning in place is not stopped: the angular command counts as movement."""
    turn_range = {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (3.0, 3.0),
    }
    mgr = VelocityCommandManager(env, range=turn_range, stopped_probability=0.0)
    mgr.resample_command([0, 1, 2, 3])
    assert mgr.stopped_envs().tolist() == [False, False, False, False]


"""
VelocityCommandManager -- deprecated standing_probability/standing_envs aliases
"""


def test_deprecated_standing_probability_param_warns_and_sets_stopped_probability(env):
    with pytest.warns(DeprecationWarning, match="stopped_probability"):
        mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, standing_probability=0.7)
    assert mgr.stopped_probability == 0.7


def test_deprecated_standing_probability_property_stays_in_sync_with_stopped_probability(env):
    """Guards against the alias going stale after a later mutation."""
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=0.5)
    mgr.stopped_probability = 0.0
    with pytest.warns(DeprecationWarning, match="stopped_probability"):
        assert mgr.standing_probability == 0.0

    with pytest.warns(DeprecationWarning, match="stopped_probability"):
        mgr.standing_probability = 0.9
    assert mgr.stopped_probability == 0.9


def test_deprecated_standing_envs_property_warns_and_matches_stopped_envs(env):
    mgr = VelocityCommandManager(env, range=VELOCITY_RANGE, stopped_probability=1.0)
    mgr.resample_command([0, 1, 2, 3])
    with pytest.warns(DeprecationWarning, match="stopped_envs"):
        assert mgr.standing_envs.tolist() == mgr.stopped_envs().tolist()


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


"""
VelocityCommandManager -- deprecated standing_* debug visualizer config keys
"""


def test_deprecated_standing_color_key_warns_and_maps_to_stopped_color(env):
    with pytest.warns(DeprecationWarning, match="stopped_color"):
        mgr = VelocityCommandManager(
            env,
            range=VELOCITY_RANGE,
            debug_visualizer_cfg={"standing_color": (1.0, 1.0, 0.0, 1.0)},
        )
    assert mgr._debug_cfg("stopped_color") == (1.0, 1.0, 0.0, 1.0)
    assert "standing_color" not in mgr.visualizer_cfg


def test_deprecated_standing_ball_radius_key_warns_and_maps_to_stopped_ball_radius(env):
    with pytest.warns(DeprecationWarning, match="stopped_ball_radius"):
        mgr = VelocityCommandManager(
            env,
            range=VELOCITY_RANGE,
            debug_visualizer_cfg={"standing_ball_radius": 0.1},
        )
    assert mgr._debug_cfg("stopped_ball_radius") == 0.1


def test_stopped_key_wins_when_both_old_and_new_keys_are_given(env):
    with pytest.warns(DeprecationWarning, match="stopped_ball_radius"):
        mgr = VelocityCommandManager(
            env,
            range=VELOCITY_RANGE,
            debug_visualizer_cfg={"standing_ball_radius": 0.1, "stopped_ball_radius": 0.2},
        )
    assert mgr._debug_cfg("stopped_ball_radius") == 0.2


def test_deprecated_key_conversion_does_not_mutate_the_callers_config(env):
    cfg = {"standing_ball_radius": 0.1}
    with pytest.warns(DeprecationWarning):
        VelocityCommandManager(env, range=VELOCITY_RANGE, debug_visualizer_cfg=cfg)
    assert cfg == {"standing_ball_radius": 0.1}


"""
VelocityCommandManager -- debug visualizer angular velocity arcs

The arc and arrowhead meshes are built with numpy/trimesh, so they can be verified
without a Genesis scene by recording the `draw_debug_mesh` calls and measuring the
angular position of the transformed mesh's vertices around the arc's center.
"""


def make_debug_arc_manager(env, ang_vel_z: float, lin_vel: tuple[float, float] = (0.0, 0.0)):
    env.scene = FakeScene(env.num_envs)
    env.robot = FakeRobot(env.num_envs)
    mgr = VelocityCommandManager(
        env,
        range={
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-2.0, 2.0),
        },
        debug_visualizer=True,
        debug_visualizer_cfg={"envs_idx": [0]},
    )
    mgr.build()
    mgr._command[0, 0] = lin_vel[0]
    mgr._command[0, 1] = lin_vel[1]
    mgr._command[0, 2] = ang_vel_z
    return mgr


def rendered_arc_vertex_angles(env) -> np.ndarray:
    """
    The angles (radians, atan2 convention) of the single rendered arc mesh's vertices
    around the vertical axis through the arc's center.

    The fake robot sits at the world origin with no environment offset, so the arc
    center's XY is (0, 0) after the recorded transform is applied.
    """
    assert len(env.scene.mesh_calls) == 1
    mesh, transform = env.scene.mesh_calls[0]
    mesh.apply_transform(transform)
    return np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0])


def test_positive_yaw_rate_sweeps_the_arc_ccw_from_the_anchor(env):
    """
    With no linear command and an identity robot orientation the anchor angle is 0
    (the +X axis). The robot's actual yaw rate is zero, so only the commanded arc is
    drawn: it must sweep counter-clockwise (positive angles), with the arrowhead
    extending past the arc's far end.
    """
    mgr = make_debug_arc_manager(env, ang_vel_z=2.0)  # max of the range: a full sweep
    mgr._render_debug(force=True)

    angles = rendered_arc_vertex_angles(env)
    full_sweep = math.radians(45)
    assert angles.min() > -0.01  # nothing clockwise of the anchor
    assert angles.max() > full_sweep  # the head extends CCW past the arc's end


def test_negative_yaw_rate_sweeps_the_arc_cw_from_the_anchor(env):
    mgr = make_debug_arc_manager(env, ang_vel_z=-2.0)
    mgr._render_debug(force=True)

    angles = rendered_arc_vertex_angles(env)
    full_sweep = math.radians(45)
    assert angles.max() < 0.01  # nothing counter-clockwise of the anchor
    assert angles.min() < -full_sweep  # the head extends CW past the arc's end


def test_arc_is_anchored_at_the_commanded_linear_direction(env):
    """A +Y linear command rotates the whole arc to start at the +Y axis (90 degrees)."""
    mgr = make_debug_arc_manager(env, ang_vel_z=2.0, lin_vel=(0.0, 1.0))
    mgr._render_debug(force=True)

    angles = rendered_arc_vertex_angles(env)
    anchor = math.pi / 2
    assert angles.min() > anchor - 0.01
    assert angles.max() > anchor + math.radians(45)


def test_a_negligible_yaw_rate_draws_no_arc(env):
    mgr = make_debug_arc_manager(env, ang_vel_z=0.01)
    mgr._render_debug(force=True)
    assert env.scene.mesh_calls == []
