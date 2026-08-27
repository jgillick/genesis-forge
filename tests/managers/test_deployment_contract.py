"""The deployment export contract on managers.

Each action manager describes its own decode as plain data, so the exporter never
inspects a subclass's internals. These tests check what each built-in manager
publishes, that custom managers can opt in without library changes, and that
per-environment divergence (domain randomization) is refused rather than guessed at.

Uses the same FakeActuatorManager shape as test_action_manager.py -- DOF indices
deliberately differ from their column positions.
"""

import pytest
import torch

from genesis_forge.managers import (
    ObservationManager,
    PositionActionManager,
    PositionWithinLimitsActionManager,
    VelocityActionManager,
)
from genesis_forge.managers.action.base import (
    BaseActionManager,
    DeploymentActionConfig,
    to_nominal_array,
)


class FakeActuatorManager:
    def __init__(self, dofs, default_pos, lower, upper):
        self.dofs = dofs
        self._idx_to_col = {idx: col for col, idx in enumerate(dofs.values())}
        self.default_dofs_pos = default_pos
        self._lower = lower
        self._upper = upper

    def _cols(self, dofs_idx):
        return [self._idx_to_col[i] for i in dofs_idx]

    def get_dofs_limits(self, dofs_idx):
        cols = self._cols(dofs_idx)
        return self._lower[cols], self._upper[cols]


def make_actuator_manager(num_envs=4, default_pos=None):
    dofs = {"FL_hip": 100, "FL_knee": 101, "FR_hip": 102}
    if default_pos is None:
        default_pos = torch.tensor([[0.1, 0.2, 0.3]] * num_envs)
    lower = torch.tensor([-1.0, -1.5, -2.0])
    upper = torch.tensor([1.0, 1.5, 2.0])
    return FakeActuatorManager(dofs, default_pos, lower, upper)


"""
PositionActionManager export
"""


def test_position_manager_exports_its_affine_decode(env):
    actuator = make_actuator_manager()
    manager = PositionActionManager(env, actuator_manager=actuator, scale=0.5)
    manager.build()

    exported = manager.get_deployment_config()

    assert isinstance(exported, DeploymentActionConfig)
    assert exported.deploy_type == "position"
    assert exported.config["scale"] == [0.5, 0.5, 0.5]
    # use_default_offset is on by default, so the offset is the per-joint default pose.
    assert exported.config["offset"] == pytest.approx([0.1, 0.2, 0.3])
    # Clip defaults to the actuator's joint limits.
    assert exported.config["post_clip_low"] == pytest.approx([-1.0, -1.5, -2.0])
    assert exported.config["post_clip_high"] == pytest.approx([1.0, 1.5, 2.0])


def test_position_manager_export_is_plain_json_ready_data(env):
    manager = PositionActionManager(env, actuator_manager=make_actuator_manager())
    manager.build()

    config = manager.get_deployment_config().config

    for name, values in config.items():
        assert isinstance(values, list), name
        assert all(isinstance(item, float) for item in values), name


def test_position_manager_export_reflects_explicit_scale_and_offset(env):
    manager = PositionActionManager(
        env,
        actuator_manager=make_actuator_manager(),
        scale={".*_hip": 0.25, ".*_knee": 0.75},
        offset=0.0,
        use_default_offset=False,
    )
    manager.build()

    config = manager.get_deployment_config().config

    assert config["scale"] == pytest.approx([0.25, 0.75, 0.25])
    assert config["offset"] == pytest.approx([0.0, 0.0, 0.0])


def test_position_manager_export_reflects_a_soft_limit_factor(env):
    manager = PositionActionManager(
        env, actuator_manager=make_actuator_manager(), soft_limit_scale_factor=0.5
    )
    manager.build()

    config = manager.get_deployment_config().config

    # Half the range around each joint's midpoint.
    assert config["post_clip_low"] == pytest.approx([-0.5, -0.75, -1.0])
    assert config["post_clip_high"] == pytest.approx([0.5, 0.75, 1.0])


def test_exported_joint_order_matches_the_managers_dof_order(env):
    manager = PositionActionManager(env, actuator_manager=make_actuator_manager())
    manager.build()

    assert list(manager.dofs.keys()) == ["FL_hip", "FL_knee", "FR_hip"]
    assert len(manager.get_deployment_config().config["scale"]) == 3


"""
PositionWithinLimitsActionManager export

It subclasses PositionActionManager but overrides process_actions with different
math, so it must override the export too -- otherwise it would publish the
parent's parameters, which its own decode never uses.
"""


def test_within_limits_manager_exports_its_own_decode(env):
    manager = PositionWithinLimitsActionManager(
        env, actuator_manager=make_actuator_manager()
    )
    manager.build()

    exported = manager.get_deployment_config()

    assert exported.deploy_type == "position_within_limits"
    assert exported.config["pre_clip"] == [-1.0, 1.0]
    # Midpoint and half-range of each joint's limits.
    assert exported.config["offset"] == pytest.approx([0.0, 0.0, 0.0])
    assert exported.config["scale"] == pytest.approx([1.0, 1.5, 2.0])


def test_within_limits_export_does_not_claim_a_post_clip(env):
    """Its process_actions applies none, so the bundle must not describe one."""
    manager = PositionWithinLimitsActionManager(
        env, actuator_manager=make_actuator_manager()
    )
    manager.build()

    config = manager.get_deployment_config().config

    assert "post_clip_low" not in config
    assert "post_clip_high" not in config


def test_within_limits_export_reflects_custom_limits(env):
    manager = PositionWithinLimitsActionManager(
        env,
        actuator_manager=make_actuator_manager(),
        limit={".*_hip": (0.0, 2.0)},
    )
    manager.build()

    config = manager.get_deployment_config().config

    # hip joints: midpoint 1.0, half-range 1.0; knee keeps the model limits.
    assert config["offset"] == pytest.approx([1.0, 0.0, 1.0])
    assert config["scale"] == pytest.approx([1.0, 1.5, 1.0])


def test_the_two_managers_publish_different_deploy_types(env):
    position = PositionActionManager(env, actuator_manager=make_actuator_manager())
    position.build()
    within = PositionWithinLimitsActionManager(
        env, actuator_manager=make_actuator_manager()
    )
    within.build()

    assert position.get_deployment_config().deploy_type != (
        within.get_deployment_config().deploy_type
    )


"""
VelocityActionManager export

Shares AffineDofActionManager's decode with the position managers, so it becomes
deployable through the shared contract rather than through anything velocity-specific.
"""


def test_velocity_manager_exports_its_affine_decode(env):
    manager = VelocityActionManager(
        env, actuator_manager=make_actuator_manager(), scale=8.0
    )
    manager.build()

    exported = manager.get_deployment_config()

    assert exported.deploy_type == "velocity"
    assert exported.config["scale"] == pytest.approx([8.0, 8.0, 8.0])
    assert exported.config["offset"] == pytest.approx([0.0, 0.0, 0.0])


def test_an_unbounded_velocity_manager_exports_no_clip(env):
    """Its clip defaults to +/-inf, which means "no clip" -- and has no JSON form."""
    manager = VelocityActionManager(env, actuator_manager=make_actuator_manager())
    manager.build()

    config = manager.get_deployment_config().config

    assert "post_clip_low" not in config
    assert "post_clip_high" not in config


def test_a_clipped_velocity_manager_exports_its_bounds(env):
    manager = VelocityActionManager(
        env, actuator_manager=make_actuator_manager(), clip=(-16.0, 16.0)
    )
    manager.build()

    config = manager.get_deployment_config().config

    assert config["post_clip_low"] == pytest.approx([-16.0, -16.0, -16.0])
    assert config["post_clip_high"] == pytest.approx([16.0, 16.0, 16.0])


def test_a_partially_clipped_velocity_manager_keeps_both_sides(env):
    """Some joints bounded, others not: the bounded values must still be recorded."""
    manager = VelocityActionManager(
        env, actuator_manager=make_actuator_manager(), clip={".*_hip": (-5.0, 5.0)}
    )
    manager.build()

    config = manager.get_deployment_config().config

    assert config["post_clip_low"][0] == pytest.approx(-5.0)
    assert config["post_clip_high"][0] == pytest.approx(5.0)
    # The unmatched joint stays unbounded.
    assert config["post_clip_low"][1] == float("-inf")


def test_every_builtin_action_manager_publishes_a_distinct_type(env):
    managers = [
        PositionActionManager(env, actuator_manager=make_actuator_manager()),
        PositionWithinLimitsActionManager(env, actuator_manager=make_actuator_manager()),
        VelocityActionManager(env, actuator_manager=make_actuator_manager()),
    ]
    for manager in managers:
        manager.build()

    types = [manager.get_deployment_config().deploy_type for manager in managers]

    assert types == ["position", "position_within_limits", "velocity"]
    assert len(set(types)) == len(types)


def test_the_shared_affine_contract_is_inherited_not_reimplemented(env):
    """Position and velocity both get the contract from AffineDofActionManager."""
    from genesis_forge.managers.action.affine_dof_action_manager import (
        AffineDofActionManager,
    )

    assert (
        PositionActionManager.get_deployment_config
        is AffineDofActionManager.get_deployment_config
    )
    assert (
        VelocityActionManager.get_deployment_config
        is AffineDofActionManager.get_deployment_config
    )
    # Within-limits decodes differently, so it must NOT share the inherited version.
    assert (
        PositionWithinLimitsActionManager.get_deployment_config
        is not AffineDofActionManager.get_deployment_config
    )


def test_exporting_an_unbuilt_affine_manager_is_refused(env):
    manager = VelocityActionManager(env, actuator_manager=make_actuator_manager())

    with pytest.raises(RuntimeError) as error:
        manager.get_deployment_config()

    assert "built" in str(error.value)


"""
Opting in (F3) and opting out
"""


def test_a_manager_without_deployment_support_raises_a_helpful_error(env):
    manager = BaseActionManager(env, actuator_manager=make_actuator_manager())
    manager.build()

    with pytest.raises(NotImplementedError) as error:
        manager.get_deployment_config()

    message = str(error.value)
    assert "BaseActionManager" in message
    assert "get_deployment_config" in message


def test_a_custom_manager_can_opt_in_without_library_changes(env):
    """Covers AE3: a third-party manager participates by implementing the contract."""

    class VelocityActionManager(BaseActionManager):
        def process_actions(self, actions):
            return actions.clamp(-1.0, 1.0)

        def get_deployment_config(self):
            return DeploymentActionConfig(
                deploy_type="velocity",
                config={"max_velocity": [1.0] * self.num_actions},
                decoder_import_path="my_robot.decoders:VelocityDecoder",
            )

    manager = VelocityActionManager(env, actuator_manager=make_actuator_manager())
    manager.build()

    exported = manager.get_deployment_config()

    assert exported.deploy_type == "velocity"
    assert exported.decoder_import_path == "my_robot.decoders:VelocityDecoder"
    assert exported.config["max_velocity"] == [1.0, 1.0, 1.0]


def test_builtin_managers_need_no_import_path(env):
    manager = PositionActionManager(env, actuator_manager=make_actuator_manager())
    manager.build()

    assert manager.get_deployment_config().decoder_import_path is None


"""
Nominal value reduction -- refusing to guess under domain randomization
"""


def test_per_environment_values_reduce_to_one_row_when_identical():
    tensor = torch.tensor([[1.0, 2.0]] * 4)

    result = to_nominal_array(
        tensor, name="offset", num_joints=2, num_envs=4, manager_name="Test"
    )

    assert result == [1.0, 2.0]


def test_a_flat_per_joint_tensor_passes_through():
    result = to_nominal_array(
        torch.tensor([1.0, 2.0]),
        name="scale",
        num_joints=2,
        num_envs=4,
        manager_name="Test",
    )

    assert result == [1.0, 2.0]


def test_divergent_environments_are_refused_with_actionable_guidance():
    randomized = torch.tensor([[1.0, 2.0], [1.0, 2.5], [1.0, 2.0], [1.0, 2.0]])

    with pytest.raises(ValueError) as error:
        to_nominal_array(
            randomized, name="offset", num_joints=2, num_envs=4, manager_name="Position"
        )

    message = str(error.value)
    assert "offset" in message
    assert "Position" in message
    assert "randomization" in message


def test_an_unexpected_shape_is_refused():
    with pytest.raises(ValueError) as error:
        to_nominal_array(
            torch.zeros(2, 3, 4), name="scale", num_joints=2, num_envs=4, manager_name="X"
        )

    assert "scale" in str(error.value)


def test_a_randomized_default_pose_blocks_export(env):
    """The realistic path to divergence: randomized default positions feed the offset."""
    randomized_default = torch.tensor(
        [[0.1, 0.2, 0.3], [0.1, 0.9, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    )
    manager = PositionActionManager(
        env, actuator_manager=make_actuator_manager(default_pos=randomized_default)
    )
    manager.build()

    with pytest.raises(ValueError) as error:
        manager.get_deployment_config()

    assert "offset" in str(error.value)


"""
ObservationManager export
"""


def observation_manager(env, **kwargs):
    return ObservationManager(
        env,
        cfg={
            "gyro": {
                "fn": lambda env: torch.ones((env.num_envs, 3)),
                "scale": 0.25,
                "description": "Body-frame angular velocity",
                "units": "rad/s",
            },
            "dof_pos": {"fn": lambda env: torch.ones((env.num_envs, 2))},
        },
        **kwargs,
    )


def test_observation_export_captures_order_sizes_and_scales(env):
    manager = observation_manager(env)
    manager.build()

    layout = manager.get_deployment_layout()

    assert [entry["name"] for entry in layout["entries"]] == ["gyro", "dof_pos"]
    assert [entry["size"] for entry in layout["entries"]] == [3, 2]
    assert layout["entries"][0]["scale"] == pytest.approx(0.25)
    assert layout["entries"][1]["scale"] == pytest.approx(1.0)


def test_observation_export_carries_metadata_when_supplied(env):
    manager = observation_manager(env)
    manager.build()

    gyro = manager.get_deployment_layout()["entries"][0]

    assert gyro["description"] == "Body-frame angular velocity"
    assert gyro["units"] == "rad/s"


def test_entries_without_metadata_simply_omit_it(env):
    manager = observation_manager(env)
    manager.build()

    dof_pos = manager.get_deployment_layout()["entries"][1]

    assert "description" not in dof_pos
    assert "units" not in dof_pos


def test_observation_export_records_history_configuration(env):
    manager = observation_manager(env, history_len=3)
    manager.build()

    layout = manager.get_deployment_layout()

    assert layout["history_length"] == 3
    assert layout["history_order"] == "newest_first"


def test_zero_width_entries_are_excluded(env):
    """Training skips them, so the bundle must not demand a value for one."""
    manager = ObservationManager(
        env,
        cfg={
            "empty": {"fn": lambda env: torch.ones((env.num_envs, 0))},
            "real": {"fn": lambda env: torch.ones((env.num_envs, 2))},
        },
    )
    manager.build()

    entries = manager.get_deployment_layout()["entries"]

    assert [entry["name"] for entry in entries] == ["real"]


def test_exporting_before_build_is_refused(env):
    manager = observation_manager(env)

    with pytest.raises(RuntimeError) as error:
        manager.get_deployment_layout()

    assert "built" in str(error.value)


def test_noise_is_never_exported(env):
    manager = ObservationManager(
        env,
        cfg={"gyro": {"fn": lambda env: torch.ones((env.num_envs, 3)), "noise": 0.5}},
    )
    manager.build()

    entry = manager.get_deployment_layout()["entries"][0]

    assert "noise" not in entry


def actuator_with_values(values, dofs=None):
    """An ActuatorManager with buffers set directly.

    Built via __new__ because the real constructor needs a Genesis entity, and the
    only thing under test here is how the buffers are read back out.
    """
    from genesis_forge.managers import ActuatorManager

    manager = ActuatorManager.__new__(ActuatorManager)
    manager._dofs = dofs if dofs is not None else {"FL_hip": 100, "FL_knee": 101}
    manager._values = values
    return manager


def test_unconfigured_actuator_values_are_skipped():
    """Every actuator parameter is pre-seeded as None; only configured ones export."""
    manager = actuator_with_values(
        {
            "kp": {"buffer": torch.tensor([50.0, 50.0]), "has_noise": False},
            "kv": None,
            "default_pos": None,
        }
    )

    exported = manager.get_deployment_values()

    assert exported["values"] == {"kp": [50.0, 50.0]}
    assert exported["joint_names"] == ["FL_hip", "FL_knee"]
    assert exported["randomized"] == []


def test_randomized_actuator_values_are_flagged():
    manager = actuator_with_values(
        {
            "kp": {"buffer": torch.tensor([50.0]), "has_noise": True},
            "kv": {"buffer": torch.tensor([0.5]), "has_noise": False},
        },
        dofs={"FL_hip": 100},
    )

    exported = manager.get_deployment_values()

    assert exported["randomized"] == ["kp"]


def test_per_environment_actuator_buffers_reduce_to_one_row():
    manager = actuator_with_values(
        {"default_pos": {"buffer": torch.tensor([[0.1, 0.2]] * 4), "has_noise": False}}
    )

    exported = manager.get_deployment_values()

    assert exported["values"]["default_pos"] == pytest.approx([0.1, 0.2])
