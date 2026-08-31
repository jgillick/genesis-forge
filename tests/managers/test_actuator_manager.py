"""Behavior of ActuatorManager: joint discovery, per-DOF value buffers (with NoisyValue
support), and the build()/reset() lifecycle that pushes those buffers to the robot.

Uses a FakeRobot -- no Genesis scene is built.
"""

import genesis as gs
import pytest
import torch

from genesis_forge.managers import ActuatorManager
from genesis_forge.managers.actuator import NoisyValue


class FakeJoint:
    def __init__(self, name, dof_start, joint_type=gs.JOINT_TYPE.REVOLUTE):
        self.name = name
        self.dof_start = dof_start
        self.type = joint_type


class FakeRigidOptions:
    def __init__(self, batch_dofs_info=False, batch_links_info=False):
        self.batch_dofs_info = batch_dofs_info
        self.batch_links_info = batch_links_info


class FakeScene:
    def __init__(self, batch_dofs_info=False, batch_links_info=False):
        self.rigid_options = FakeRigidOptions(batch_dofs_info, batch_links_info)


class FakeRobot:
    def __init__(self, joints, position=None, velocity=None, force=None, force_range=None, limits=None):
        self.joints = joints
        self.calls = []
        self._position = position
        self._velocity = velocity
        self._force = force
        self._force_range = force_range
        self._limits = limits
        # dof_start values are deliberately non-contiguous (see make_joints); translate
        # them back to a column position in this fake's own per-DOF buffers.
        self._idx_to_col = {
            j.dof_start: col
            for col, j in enumerate(j for j in joints if j.type == gs.JOINT_TYPE.REVOLUTE)
        }

    def _cols(self, dofs_idx):
        return [self._idx_to_col[i] for i in dofs_idx]

    def calls_named(self, name):
        return [c for c in self.calls if c[0] == name]

    def get_dofs_position(self, dofs_idx):
        return self._position[:, self._cols(dofs_idx)]

    def get_dofs_velocity(self, dofs_idx):
        return self._velocity[:, self._cols(dofs_idx)]

    def get_dofs_force(self, dofs_idx):
        return self._force[:, self._cols(dofs_idx)]

    def get_dofs_control_force(self, dofs_idx):
        # Deliberately different values from get_dofs_force, so a test can tell
        # whether a caller asked for the measured force or the control force.
        return self._force[:, self._cols(dofs_idx)] * -1

    def get_dofs_force_range(self, dofs_idx):
        cols = self._cols(dofs_idx)
        lower, upper = self._force_range
        return lower[cols], upper[cols]

    def get_dofs_limit(self, dofs_idx):
        cols = self._cols(dofs_idx)
        lower, upper = self._limits
        return lower[cols], upper[cols]

    def set_dofs_position(self, position, dofs_idx=None, dofs_idx_local=None, envs_idx=None):
        # ActuatorManager calls this two different ways: positionally with `dofs_idx`
        # (its own set_dofs_position wrapper) and by keyword with `dofs_idx_local` and
        # `envs_idx` (its reset()) -- accept both.
        idx = dofs_idx if dofs_idx is not None else dofs_idx_local
        self.calls.append(("set_dofs_position", position.clone(), list(idx), envs_idx))

    def control_dofs_position(self, position, dofs_idx):
        self.calls.append(("control_dofs_position", position.clone(), list(dofs_idx)))

    def control_dofs_velocity(self, velocity, dofs_idx):
        self.calls.append(("control_dofs_velocity", velocity.clone(), list(dofs_idx)))

    def set_dofs_kp(self, kp, dofs_idx, envs_idx):
        self.calls.append(("set_dofs_kp", kp.clone(), list(dofs_idx), envs_idx))

    def set_dofs_kv(self, kv, dofs_idx, envs_idx):
        self.calls.append(("set_dofs_kv", kv.clone(), list(dofs_idx), envs_idx))

    def set_dofs_damping(self, damping, dofs_idx, envs_idx):
        self.calls.append(("set_dofs_damping", damping.clone(), list(dofs_idx), envs_idx))

    def set_dofs_stiffness(self, stiffness, dofs_idx, envs_idx):
        self.calls.append(("set_dofs_stiffness", stiffness.clone(), list(dofs_idx), envs_idx))

    def set_dofs_frictionloss(self, frictionloss, dofs_idx, envs_idx):
        self.calls.append(("set_dofs_frictionloss", frictionloss.clone(), list(dofs_idx), envs_idx))

    def set_dofs_armature(self, armature, dofs_idx, envs_idx=None):
        self.calls.append(("set_dofs_armature", armature.clone(), list(dofs_idx), envs_idx))

    def set_dofs_force_range(self, force_min, force_max, dofs_idx, envs_idx):
        self.calls.append(
            ("set_dofs_force_range", force_min.clone(), force_max.clone(), list(dofs_idx), envs_idx)
        )


def make_joints():
    """Three revolute DOFs at non-contiguous indices, plus one non-revolute joint
    that would otherwise match every pattern -- build() must still exclude it."""
    return [
        FakeJoint("hip", 5),
        FakeJoint("knee", 8),
        FakeJoint("ankle", 12),
        FakeJoint("base", 0, joint_type=gs.JOINT_TYPE.FIXED),
    ]


"""
build() -- joint discovery
"""


def test_build_selects_only_revolute_joints_matching_the_pattern(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    assert mgr.dofs == {"hip": 5, "knee": 8, "ankle": 12}
    assert mgr.dofs_idx == [5, 8, 12]
    assert mgr.dofs_names == ["hip", "knee", "ankle"]
    assert mgr.num_dofs == 3


def test_build_filters_by_a_specific_joint_name_list(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=["hip", "ankle"])
    mgr.build()

    assert mgr.dofs == {"hip": 5, "ankle": 12}


"""
build() -- value buffers
"""


def test_default_pos_fills_per_dof_and_expands_to_all_envs(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(
        env, joint_names=".*", default_pos={"hip": 0.1, "knee": 0.2, "ankle": 0.3}
    )
    mgr.build()

    assert mgr.default_dofs_pos.shape == (env.num_envs, 3)
    assert torch.allclose(mgr.default_dofs_pos[0], torch.tensor([0.1, 0.2, 0.3]))


def test_default_pos_falls_back_to_zero_when_explicitly_none(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", default_pos=None)
    mgr.build()

    assert torch.allclose(mgr.default_dofs_pos, torch.zeros((env.num_envs, 3)))


def test_unmatched_value_pattern_raises(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", kp={"nonexistent_joint": 50})
    with pytest.raises(RuntimeError, match="not found"):
        mgr.build()


"""
reset() -- setting values once, vs. every time when there's noise
"""


def test_reset_sets_a_non_noisy_value_to_the_exact_configured_number(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", kp={"hip": 50, "knee": 30, "ankle": 30})
    mgr.build()
    mgr.reset()

    _, kp, dofs_idx, _ = env.robot.calls_named("set_dofs_kp")[0]
    assert dofs_idx == [5, 8, 12]
    assert torch.equal(kp, torch.tensor([50.0, 30.0, 30.0]))


def test_reset_applies_noisy_value_within_the_configured_range(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(
        env, joint_names=".*", kp={"hip": NoisyValue(50, 5.0), "knee": 30, "ankle": 30}
    )
    mgr.build()
    mgr.reset()

    _, kp, _, _ = env.robot.calls_named("set_dofs_kp")[0]
    assert 45.0 <= kp[0].item() <= 55.0
    assert kp[1].item() == 30.0


def test_reset_only_sets_a_non_noisy_value_once(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", kp={"hip": 50, "knee": 30, "ankle": 30})
    mgr.build()

    mgr.reset()
    mgr.reset()
    mgr.reset()

    assert len(env.robot.calls_named("set_dofs_kp")) == 1


def test_reset_reapplies_a_noisy_value_on_every_call(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(
        env, joint_names=".*", kp={"hip": NoisyValue(50, 5.0), "knee": 30, "ankle": 30}
    )
    mgr.build()

    mgr.reset()
    mgr.reset()

    assert len(env.robot.calls_named("set_dofs_kp")) == 2


def test_reset_always_reapplies_the_default_position(env):
    """Unlike kp/kv/etc, the DOF position must be reset every episode."""
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(
        env, joint_names=".*", default_pos={"hip": 0.1, "knee": 0.2, "ankle": 0.3}
    )
    mgr.build()

    mgr.reset()
    mgr.reset()

    assert len(env.robot.calls_named("set_dofs_position")) == 2


def test_reset_is_a_noop_when_disabled(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", kp=50)
    mgr.build()
    mgr.enabled = False

    mgr.reset()

    assert env.robot.calls == []


def test_reset_slices_the_default_pos_buffer_by_envs_idx(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(
        env, joint_names=".*", default_pos={"hip": 0.1, "knee": 0.2, "ankle": 0.3}
    )
    mgr.build()

    mgr.reset(envs_idx=[0, 2])

    _, position, _, envs_idx = env.robot.calls_named("set_dofs_position")[0]
    assert position.shape == (2, 3)
    assert envs_idx == [0, 2]


def test_reset_skips_values_that_were_never_configured(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")  # no kp/kv/damping/etc given
    mgr.build()

    mgr.reset()

    assert env.robot.calls_named("set_dofs_kp") == []
    assert env.robot.calls_named("set_dofs_kv") == []


"""
DOF batching -- only kp/kv/etc expand to a per-env buffer when batching is enabled
"""


def test_value_buffer_is_per_env_only_when_batching_is_enabled(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene(batch_dofs_info=True, batch_links_info=True)
    mgr = ActuatorManager(env, joint_names=".*", kp=50)
    mgr.build()
    mgr.reset()

    _, kp, _, _ = env.robot.calls_named("set_dofs_kp")[0]
    assert kp.shape == (env.num_envs, 3)


def test_value_buffer_stays_dof_only_when_batching_is_disabled(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene(batch_dofs_info=False, batch_links_info=False)
    mgr = ActuatorManager(env, joint_names=".*", kp=50)
    mgr.build()
    mgr.reset()

    _, kp, _, _ = env.robot.calls_named("set_dofs_kp")[0]
    assert kp.shape == (3,)


"""
Armature -- without batching, armature can't vary per env, so build() sets it once
immediately. With batching, that build-time call is skipped and it's deferred to
reset() instead (which re-applies it every reset, like the other actuator values).
"""


def test_armature_is_set_at_build_when_batching_is_disabled(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene(batch_dofs_info=False, batch_links_info=False)
    mgr = ActuatorManager(env, joint_names=".*", armature=0.01)
    mgr.build()

    assert len(env.robot.calls_named("set_dofs_armature")) == 1


def test_armature_is_not_set_at_build_when_batching_is_enabled(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene(batch_dofs_info=True, batch_links_info=True)
    mgr = ActuatorManager(env, joint_names=".*", armature=0.01)
    mgr.build()

    assert env.robot.calls_named("set_dofs_armature") == []


def test_armature_is_not_reapplied_at_reset_without_batching(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene(batch_dofs_info=False, batch_links_info=False)
    mgr = ActuatorManager(env, joint_names=".*", armature=0.01)
    mgr.build()

    mgr.reset()

    assert len(env.robot.calls_named("set_dofs_armature")) == 1  # only the build-time call


def test_armature_is_applied_at_reset_only_with_batching(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene(batch_dofs_info=True, batch_links_info=True)
    mgr = ActuatorManager(env, joint_names=".*", armature=0.01)
    mgr.build()

    mgr.reset()

    assert len(env.robot.calls_named("set_dofs_armature")) == 1  # reset(), not build()


"""
max_force -- normalized into a symmetric or explicit min/max range
"""


def test_max_force_single_value_becomes_a_symmetric_range(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", max_force=8.0)
    mgr.build()
    mgr.reset()

    _, force_min, force_max, _, _ = env.robot.calls_named("set_dofs_force_range")[0]
    assert torch.equal(force_min, torch.tensor([-8.0, -8.0, -8.0]))
    assert torch.equal(force_max, torch.tensor([8.0, 8.0, 8.0]))


def test_max_force_as_a_list_range_is_used_directly(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(
        env, joint_names=".*", max_force={"hip": [-2.0, 5.0], "knee": 8.0, "ankle": 8.0}
    )
    mgr.build()
    mgr.reset()

    _, force_min, force_max, _, _ = env.robot.calls_named("set_dofs_force_range")[0]
    assert force_min[0].item() == -2.0
    assert force_max[0].item() == 5.0


def test_get_dofs_max_force_returns_the_absolute_upper_limit(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*", max_force=8.0)
    mgr.build()

    limits = mgr.get_dofs_max_force()
    assert limits.shape == (env.num_envs, 3)
    assert torch.allclose(limits, torch.full((env.num_envs, 3), 8.0))


def test_get_dofs_max_force_raises_when_not_configured(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    with pytest.raises(ValueError, match="max_force is not configured"):
        mgr.get_dofs_max_force()


"""
DOF convenience wrappers
"""


def test_get_dofs_position_defaults_to_every_configured_dof(env):
    env.robot = FakeRobot(make_joints(), position=torch.tensor([[1.0, 2.0, 3.0]] * env.num_envs))
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    assert torch.equal(mgr.get_dofs_position(), torch.tensor([[1.0, 2.0, 3.0]] * env.num_envs))


def test_get_dofs_velocity_clips_when_requested(env):
    env.robot = FakeRobot(
        make_joints(), velocity=torch.tensor([[10.0, -10.0, 0.0]] * env.num_envs)
    )
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    clipped = mgr.get_dofs_velocity(clip={"min": -1.0, "max": 1.0})
    assert torch.equal(clipped, torch.tensor([[1.0, -1.0, 0.0]] * env.num_envs))


def test_get_dofs_force_clips_to_the_max_force_range(env):
    env.robot = FakeRobot(
        make_joints(),
        force=torch.tensor([[100.0, -100.0, 0.0]] * env.num_envs),
        force_range=(torch.tensor([-5.0, -5.0, -5.0]), torch.tensor([5.0, 5.0, 5.0])),
    )
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    clipped = mgr.get_dofs_force(clip_to_max_force=True)
    assert torch.equal(clipped, torch.tensor([[5.0, -5.0, 0.0]] * env.num_envs))


def test_get_dofs_control_force_reads_the_control_force_not_the_measured_force(env):
    env.robot = FakeRobot(make_joints(), force=torch.tensor([[1.0, 2.0, 3.0]] * env.num_envs))
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    assert torch.equal(mgr.get_dofs_control_force(), torch.tensor([[-1.0, -2.0, -3.0]] * env.num_envs))


def test_get_dofs_limits_wraps_the_robot(env):
    env.robot = FakeRobot(
        make_joints(), limits=(torch.tensor([-1.0, -1.5, -2.0]), torch.tensor([1.0, 1.5, 2.0]))
    )
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    lower, upper = mgr.get_dofs_limits()
    assert torch.equal(lower, torch.tensor([-1.0, -1.5, -2.0]))
    assert torch.equal(upper, torch.tensor([1.0, 1.5, 2.0]))


def test_control_dofs_position_forwards_to_the_robot(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    actions = torch.zeros((env.num_envs, 3))
    mgr.control_dofs_position(actions)

    _, position, dofs_idx = env.robot.calls_named("control_dofs_position")[0]
    assert dofs_idx == [5, 8, 12]
    assert torch.equal(position, actions)


def test_control_dofs_velocity_forwards_to_the_robot(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    actions = torch.ones((env.num_envs, 3))
    mgr.control_dofs_velocity(actions)

    _, velocity, dofs_idx = env.robot.calls_named("control_dofs_velocity")[0]
    assert dofs_idx == [5, 8, 12]
    assert torch.equal(velocity, actions)


def test_control_dofs_velocity_forwards_explicit_dofs_idx(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    actions = torch.ones((env.num_envs, 2))
    mgr.control_dofs_velocity(actions, dofs_idx=[5, 8])

    _, velocity, dofs_idx = env.robot.calls_named("control_dofs_velocity")[0]
    assert dofs_idx == [5, 8]
    assert torch.equal(velocity, actions)


def test_set_dofs_position_forwards_to_the_robot(env):
    env.robot = FakeRobot(make_joints())
    env.scene = FakeScene()
    mgr = ActuatorManager(env, joint_names=".*")
    mgr.build()

    position = torch.zeros((env.num_envs, 3))
    mgr.set_dofs_position(position)

    _, sent_position, dofs_idx, _ = env.robot.calls_named("set_dofs_position")[0]
    assert dofs_idx == [5, 8, 12]
    assert torch.equal(sent_position, position)
