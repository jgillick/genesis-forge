"""Behavior of the action managers: BaseActionManager (via a concrete subclass),
PositionActionManager, and PositionWithinLimitsActionManager.

Uses a FakeActuatorManager -- no Genesis scene is built. DOF indices are chosen to
differ from their column position in the fake actuator's buffers (e.g. FL_hip is
dof index 100, not 0), so tests can tell apart `dofs_idx` (the real actuator DOF
indices) from `actuator_dof_filter` (the positional index into the actuator's own
per-DOF buffers) rather than have the two coincide by accident.
"""

import pytest
import torch

from genesis_forge.managers import (
    PositionActionManager,
    PositionWithinLimitsActionManager,
)
from genesis_forge.managers.action.base import BaseActionManager


class FakeActuatorManager:
    def __init__(self, dofs, default_pos, lower, upper, position=None, velocity=None, force=None):
        self.dofs = dofs
        self._idx_to_col = {idx: col for col, idx in enumerate(dofs.values())}
        self.default_dofs_pos = default_pos
        self._lower = lower
        self._upper = upper
        self._position = position
        self._velocity = velocity
        self._force = force
        self.control_calls = []

    def _cols(self, dofs_idx):
        return [self._idx_to_col[i] for i in dofs_idx]

    def get_dofs_limits(self, dofs_idx):
        cols = self._cols(dofs_idx)
        return self._lower[cols], self._upper[cols]

    def get_dofs_position(self, dofs_idx):
        return self._position[:, self._cols(dofs_idx)]

    def get_dofs_velocity(self, clip=None, dofs_idx=None):
        vel = self._velocity[:, self._cols(dofs_idx)]
        if clip is not None:
            vel = vel.clamp(*clip)
        return vel

    def get_dofs_force(self, clip_to_max_force=False, dofs_idx=None):
        return self._force[:, self._cols(dofs_idx)]

    def control_dofs_position(self, position, dofs_idx):
        self.control_calls.append((position.clone(), list(dofs_idx)))


def make_actuator_manager(num_envs=4, position=None, velocity=None, force=None):
    dofs = {"FL_hip": 100, "FL_knee": 101, "FR_hip": 102}
    default_pos = torch.tensor([[0.1, 0.2, 0.3]] * num_envs)
    lower = torch.tensor([-1.0, -1.5, -2.0])
    upper = torch.tensor([1.0, 1.5, 2.0])
    return FakeActuatorManager(dofs, default_pos, lower, upper, position, velocity, force)


"""
BaseActionManager -- construction and DOF filtering
"""


def test_requires_an_actuator_manager(env):
    with pytest.raises(ValueError, match="No ActuatorManager provided"):
        BaseActionManager(env)


def test_build_filters_dofs_by_pattern_preserving_actuator_order(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()

    assert mgr.dofs == {"FL_hip": 100, "FL_knee": 101}
    assert mgr.dofs_idx == [100, 101]
    assert torch.equal(mgr.actuator_dof_filter, torch.tensor([0, 1], dtype=torch.int32))
    assert mgr.num_actions == 2


def test_build_default_joints_selects_every_dof(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(env, actuator_manager=actuator)
    mgr.build()

    assert mgr.num_actions == 3


def test_action_space_shape_matches_num_actions(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()

    assert mgr.action_space.shape == (2,)


"""
BaseActionManager -- actions buffers
"""


def test_actions_default_to_zero_before_any_step(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()

    assert torch.equal(mgr.actions, torch.zeros((env.num_envs, 2)))
    assert torch.equal(mgr.raw_actions, torch.zeros((env.num_envs, 2)))
    assert torch.equal(mgr.last_actions, torch.zeros((env.num_envs, 2)))


def test_step_records_actions_and_shifts_last_actions(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_.*"],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
    )
    mgr.build()

    first = torch.tensor([[0.1, 0.2]] * env.num_envs)
    mgr.step(first)
    assert torch.equal(mgr.raw_actions, first)
    assert torch.equal(mgr.actions, first)
    assert torch.equal(mgr.last_actions, torch.zeros_like(first))

    second = torch.tensor([[0.3, 0.4]] * env.num_envs)
    mgr.step(second)
    assert torch.equal(mgr.last_actions, first)
    assert torch.equal(mgr.actions, second)


def test_step_delays_actions_by_delay_step(env):
    """A step's action doesn't reach `.actions` until `delay_step` steps later."""
    actuator = make_actuator_manager()
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_.*"],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
        delay_step=2,
    )
    mgr.build()
    mgr.reset(None)  # seeds the delay buffer with 2 zero placeholders

    a = torch.full((env.num_envs, 2), 0.1)
    b = torch.full((env.num_envs, 2), 0.2)

    mgr.step(a)
    assert torch.equal(mgr.actions, torch.zeros((env.num_envs, 2)))

    mgr.step(b)
    assert torch.equal(mgr.actions, torch.zeros((env.num_envs, 2)))

    mgr.step(torch.full((env.num_envs, 2), 0.3))
    assert torch.equal(mgr.actions, a)

    mgr.step(torch.full((env.num_envs, 2), 0.4))
    assert torch.equal(mgr.actions, b)


def test_reset_without_delay_step_is_a_noop(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()
    mgr.reset(None)  # must not raise


def test_get_actions_dict_maps_dof_names_to_python_floats(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_.*"],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
    )
    mgr.build()
    mgr.step(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))

    assert mgr.get_actions_dict(0) == {"FL_hip": pytest.approx(0.1), "FL_knee": pytest.approx(0.2)}
    assert mgr.get_actions_dict(1) == {"FL_hip": pytest.approx(0.3), "FL_knee": pytest.approx(0.4)}


"""
BaseActionManager -- DOF convenience wrappers use the filtered dofs_idx
"""


def test_get_dofs_wrappers_use_the_filtered_dofs_idx(env):
    actuator = make_actuator_manager(
        position=torch.tensor([[1.0, 2.0, 3.0]] * env.num_envs),
        velocity=torch.tensor([[4.0, 5.0, 6.0]] * env.num_envs),
        force=torch.tensor([[7.0, 8.0, 9.0]] * env.num_envs),
    )
    mgr = PositionActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()

    assert torch.equal(mgr.get_dofs_position(), torch.tensor([[1.0, 2.0]] * env.num_envs))
    assert torch.equal(mgr.get_dofs_velocity(), torch.tensor([[4.0, 5.0]] * env.num_envs))
    assert torch.equal(mgr.get_dofs_force(), torch.tensor([[7.0, 8.0]] * env.num_envs))


def test_base_send_actions_to_simulation_is_not_implemented(env):
    actuator = make_actuator_manager()
    mgr = BaseActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()
    with pytest.raises(NotImplementedError):
        mgr.send_actions_to_simulation()


"""
PositionActionManager -- default_dofs_pos and construction validation
"""


def test_default_dofs_pos_uses_the_actuator_dof_filter(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(env, actuator_manager=actuator, actuator_joints=["FR_.*"])
    mgr.build()

    assert torch.allclose(mgr.default_dofs_pos, torch.tensor([[0.3]] * env.num_envs))


def test_use_default_offset_with_nonzero_offset_raises(env):
    actuator = make_actuator_manager()
    with pytest.raises(ValueError, match="Cannot set both use_default_offset and offset"):
        PositionActionManager(env, actuator_manager=actuator, use_default_offset=True, offset=1.0)


"""
PositionActionManager -- process_actions: scale + offset, then clamp to limits
"""


def test_process_actions_scales_offsets_and_clamps_to_limits(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_.*"],
        scale={"FL_hip": 2.0, "FL_knee": 0.5},
        use_default_offset=True,
        quiet_action_errors=True,
    )
    mgr.build()

    # FL_hip: 10*2.0 + default_pos(0.1) = 20.1 -> clamped to upper limit 1.0
    # FL_knee: 10*0.5 + default_pos(0.2) = 5.2 -> clamped to upper limit 1.5
    processed = mgr.process_actions(torch.tensor([[10.0, 10.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[1.0, 1.5]] * env.num_envs))


def test_soft_limit_scale_factor_shrinks_the_clip_range_around_the_midpoint(env):
    actuator = make_actuator_manager()  # FL_hip limits [-1, 1] -> midpoint 0
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_hip"],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
        soft_limit_scale_factor=0.5,
        quiet_action_errors=True,
    )
    mgr.build()

    processed = mgr.process_actions(torch.tensor([[10.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[0.5]] * env.num_envs))


def test_custom_clip_overrides_the_default_limit_based_clip(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_hip"],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
        clip={"FL_hip": (-0.2, 0.2)},
        quiet_action_errors=True,
    )
    mgr.build()

    processed = mgr.process_actions(torch.tensor([[10.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[0.2]] * env.num_envs))


def test_send_actions_to_simulation_controls_the_actuator(env):
    actuator = make_actuator_manager()
    mgr = PositionActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_.*"],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
    )
    mgr.build()
    mgr.step(torch.tensor([[0.1, 0.2]] * env.num_envs))

    # The `actions` arg is accepted but ignored -- it re-reads self.actions internally.
    mgr.send_actions_to_simulation(None)

    position, dofs_idx = actuator.control_calls[0]
    assert dofs_idx == [100, 101]
    assert torch.allclose(position, mgr.actions)


"""
PositionWithinLimitsActionManager -- maps [-1, 1] to the DOF's position limits
"""


def test_within_limits_maps_full_range_to_dof_limits(env):
    actuator = make_actuator_manager()  # FL_hip [-1, 1], FL_knee [-1.5, 1.5]
    mgr = PositionWithinLimitsActionManager(env, actuator_manager=actuator, actuator_joints=["FL_.*"])
    mgr.build()

    processed = mgr.process_actions(torch.tensor([[1.0, -1.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[1.0, -1.5]] * env.num_envs))

    processed_zero = mgr.process_actions(torch.tensor([[0.0, 0.0]] * env.num_envs))
    assert torch.allclose(processed_zero, torch.tensor([[0.0, 0.0]] * env.num_envs))


def test_within_limits_clamps_actions_outside_the_unit_range(env):
    actuator = make_actuator_manager()
    mgr = PositionWithinLimitsActionManager(env, actuator_manager=actuator, actuator_joints=["FL_hip"])
    mgr.build()

    processed = mgr.process_actions(torch.tensor([[5.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[1.0]] * env.num_envs))


def test_within_limits_custom_limit_overrides_a_dof(env):
    actuator = make_actuator_manager()
    mgr = PositionWithinLimitsActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_hip"],
        limit={"FL_hip": (-0.5, 0.5)},
    )
    mgr.build()

    processed = mgr.process_actions(torch.tensor([[1.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[0.5]] * env.num_envs))


def test_within_limits_soft_limit_scale_factor_shrinks_the_range(env):
    actuator = make_actuator_manager()  # FL_hip [-1, 1] -> midpoint 0
    mgr = PositionWithinLimitsActionManager(
        env,
        actuator_manager=actuator,
        actuator_joints=["FL_hip"],
        soft_limit_scale_factor=0.5,
    )
    mgr.build()

    processed = mgr.process_actions(torch.tensor([[1.0]] * env.num_envs))
    assert torch.allclose(processed, torch.tensor([[0.5]] * env.num_envs))
