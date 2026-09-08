"""Exporting a manager whose joints are grouped behind fewer actions.

`action_groups` lets one policy output drive several joints -- a robot's wheels on
one side, say -- so the manager's slice is narrower than its joint list while every
decode parameter stays per joint. The bundle has to carry the mapping between them,
and the parity gate has to compare the whole path, fan-out included.
"""

import pytest
from conftest import FakeActuatorManager, FakeManagedEnv, observation_cfg

from genesis_forge.deployment import capture_environment, check_parity, export
from genesis_forge.managers import (
    ObservationManager,
    VelocityActionManager,
)
from genesis_forge_runtime import load_bundle


@pytest.fixture
def grouped_env():
    """Three joints driven by two actions: the hips paired, the knee alone."""
    env = FakeManagedEnv()
    env.actuator_manager = FakeActuatorManager(num_envs=env.num_envs)
    env.managers["actuator"].append(env.actuator_manager)
    env.action_manager = VelocityActionManager(
        env,
        actuator_manager=env.actuator_manager,
        action_groups=[["FL_hip", "FR_hip"], ["FL_knee"]],
        scale={"FL_hip": 2.0, "FR_hip": -2.0, "FL_knee": 1.0},
    )
    env.observation_manager = ObservationManager(env, cfg=observation_cfg())
    return env.build()


def test_the_manager_reports_fewer_actions_than_joints(grouped_env):
    manager = grouped_env.action_manager

    assert manager.num_actions == 2
    assert len(manager.dofs) == 3
    assert manager.joint_action_index is not None


def test_the_bundle_records_the_mapping(grouped_env, tmp_path):
    bundle = export(grouped_env, tmp_path / "bundle", verbose=False)
    spec = bundle.manifest.actions[0]

    assert spec.num_actions == 2
    assert spec.num_joints == 3
    assert spec.joint_action_index is not None
    assert len(spec.joint_action_index) == 3
    # Decode parameters are sized per joint, not per action.
    assert len(spec.config["scale"]) == 3


def test_parity_covers_the_fan_out(grouped_env):
    """The gate compares the whole path, so a wrong mapping cannot slip through."""
    report = check_parity(capture_environment(grouped_env))

    assert report.max_action_error["action_manager"] < 1e-5


def test_the_runtime_reproduces_the_grouping(grouped_env, tmp_path):
    path = export(grouped_env, tmp_path / "bundle", verbose=False).path
    decoder = load_bundle(path).create_action_decoder()

    decoded = decoder.decode([1.0, -1.0])

    assert decoder.num_actions == 2
    assert decoder.num_joints == 3

    # Both hips are driven by action 0, but with mirrored scale -- grouping shares
    # the action, not the decode, which is why the parameters stay per joint.
    targets = decoded.by_joint
    assert targets["FL_hip"] == pytest.approx(2.0)
    assert targets["FR_hip"] == pytest.approx(-2.0)
    # The knee is driven by action 1, which was -1.0.
    assert targets["FL_knee"] == pytest.approx(-1.0)


def test_an_ungrouped_manager_records_no_mapping(deployable_env, tmp_path):
    bundle = export(deployable_env, tmp_path / "bundle", verbose=False)
    spec = bundle.manifest.actions[0]

    assert spec.joint_action_index is None
    assert spec.num_actions == spec.num_joints
