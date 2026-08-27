"""Exporting a built environment into a deployment bundle.

Covers what lands in the manifest, that a written bundle actually drives the
runtime, and that a failed export leaves nothing behind.
"""

import json

import numpy as np
import pytest
import torch
from genesis_forge_deploy import load_bundle

from genesis_forge.deployment import ExportError, ParityError, export
from genesis_forge.managers import (
    ObservationManager,
    PositionActionManager,
    PositionWithinLimitsActionManager,
)
from genesis_forge.mdp.observations import current_actions

"""
A successful export
"""


def test_export_writes_a_loadable_bundle(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False)

    bundle = load_bundle(path)

    assert bundle.manifest.dt == pytest.approx(0.02)
    assert bundle.manifest.control_hz == pytest.approx(50.0)
    assert bundle.manifest.joint_names == ("FL_hip", "FL_knee", "FR_hip")


def test_export_records_the_decode_parameters(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False)

    spec = load_bundle(path).manifest.actions[0]

    assert spec.deploy_type == "position"
    assert spec.name == "action_manager"
    np.testing.assert_allclose(spec.config["scale"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(spec.config["offset"], [0.1, 0.2, 0.3], rtol=1e-6)


def test_export_records_the_observation_layout(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False)

    layout = load_bundle(path).manifest.observations

    assert [entry.name for entry in layout.entries] == ["gyro", "dof_pos"]
    assert layout.entry("gyro").units == "rad/s"
    assert layout.entry("gyro").scale == pytest.approx(0.25)


def test_export_records_actuator_values(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False)

    actuators = load_bundle(path).manifest.actuators

    assert len(actuators) == 1
    np.testing.assert_allclose(actuators[0].values["kp"], [50.0, 50.0, 50.0])


def test_export_stamps_provenance(deployable_env, tmp_path):
    path = export(
        deployable_env, tmp_path / "bundle", checkpoint="logs/model_100.pt", verbose=False
    )

    provenance = load_bundle(path).manifest.provenance

    assert provenance.exported_at is not None
    assert provenance.genesis_forge_version is not None
    assert provenance.torch_version is not None
    assert provenance.checkpoint == "logs/model_100.pt"


def test_export_ships_golden_samples(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", parity_ticks=5, verbose=False)

    bundle = load_bundle(path)

    assert bundle.golden["observations"].shape[0] == 5
    assert bundle.golden["joint_targets"].shape[0] == 5


def test_the_manifest_is_readable_json(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False)

    data = json.loads((path / "manifest.json").read_text())

    assert data["control"]["dt"] == 0.02
    assert data["actions"]["managers"][0]["joint_names"] == [
        "FL_hip",
        "FL_knee",
        "FR_hip",
    ]


def test_export_prints_a_summary(deployable_env, tmp_path, capsys):
    export(deployable_env, tmp_path / "bundle")

    output = capsys.readouterr().out

    assert "Deployment bundle written" in output
    assert "parity over" in output
    assert "50.0 Hz" in output


"""
The exported bundle drives the runtime
"""


def test_the_written_bundle_reproduces_the_training_pipeline(deployable_env, tmp_path):
    """End to end: export, load on the 'robot', and match the torch pipeline."""
    manager = deployable_env.observation_manager
    action_manager = deployable_env.action_manager

    path = export(deployable_env, tmp_path / "bundle", verbose=False)
    bundle = load_bundle(path)
    assembler = bundle.create_observation_assembler()
    decoder = bundle.create_action_decoder()

    sensors = {
        "gyro": np.array([0.3, -0.4, 0.5], dtype=np.float32),
        "dof_pos": np.array([0.11, 0.22, 0.33], dtype=np.float32),
    }
    numpy_obs = assembler.assemble(sensors)

    for tensor in manager._history:
        tensor.zero_()
    torch_obs = manager.get_observations(
        values={
            name: torch.as_tensor(
                np.tile(value, (deployable_env.num_envs, 1)), dtype=torch.float32
            )
            for name, value in sensors.items()
        }
    )[0]
    np.testing.assert_allclose(numpy_obs, torch_obs.numpy(), rtol=1.3e-6, atol=1e-5)

    raw = np.array([0.6, -0.6, 0.2], dtype=np.float32)
    numpy_targets = decoder.decode(raw).targets
    torch_targets = action_manager.process_actions(
        torch.as_tensor(np.tile(raw, (deployable_env.num_envs, 1)), dtype=torch.float32)
    )[0]
    np.testing.assert_allclose(numpy_targets, torch_targets.numpy(), rtol=1.3e-6, atol=1e-5)


def test_golden_samples_replay_through_the_loaded_runtime(deployable_env, tmp_path):
    """The on-robot smoke test: recorded actions must decode to recorded targets."""
    path = export(deployable_env, tmp_path / "bundle", verbose=False)
    bundle = load_bundle(path)
    decoder = bundle.create_action_decoder()

    for raw, expected in zip(
        bundle.golden["raw_actions"], bundle.golden["joint_targets"], strict=True
    ):
        np.testing.assert_allclose(decoder.decode(raw).targets, expected, rtol=1e-6)


"""
Failed exports write nothing (AE1)
"""


def test_a_parity_failure_aborts_before_writing_anything(
    deployable_env, tmp_path, monkeypatch
):
    destination = tmp_path / "bundle"

    # Make the exported decode disagree with process_actions.
    original = deployable_env.action_manager.get_deployment_config

    def drifted():
        contract = original()
        contract.config["scale"] = [9.9, 9.9, 9.9]
        return contract

    monkeypatch.setattr(deployable_env.action_manager, "get_deployment_config", drifted)

    with pytest.raises(ParityError) as error:
        export(deployable_env, destination, verbose=False)

    assert "action_manager" in str(error.value)
    assert not destination.exists()


def test_a_failed_export_leaves_an_existing_bundle_intact(
    deployable_env, tmp_path, monkeypatch
):
    destination = export(deployable_env, tmp_path / "bundle", verbose=False)
    original_manifest = (destination / "manifest.json").read_text()

    original = deployable_env.action_manager.get_deployment_config

    def drifted():
        contract = original()
        contract.config["offset"] = [9.9, 9.9, 9.9]
        return contract

    monkeypatch.setattr(deployable_env.action_manager, "get_deployment_config", drifted)

    with pytest.raises(ParityError):
        export(deployable_env, destination, overwrite=True, verbose=False)

    assert (destination / "manifest.json").read_text() == original_manifest


"""
Refusing unsupported configurations
"""


def test_a_plain_object_is_refused(tmp_path):
    with pytest.raises(ExportError) as error:
        export(object(), tmp_path / "bundle", verbose=False)

    assert "ManagedEnvironment" in str(error.value)


def test_an_unbuilt_environment_is_refused(tmp_path):
    from tests.deployment.conftest import FakeActuatorManager, FakeManagedEnv

    env = FakeManagedEnv()
    env.actuator_manager = FakeActuatorManager()
    env.action_manager = PositionActionManager(
        env, actuator_manager=env.actuator_manager
    )
    env.observation_manager = ObservationManager(
        env, cfg={"gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))}}
    )
    # Deliberately not built.

    with pytest.raises(ExportError) as error:
        export(env, tmp_path / "bundle", verbose=False)

    assert "build" in str(error.value)


def test_ambiguous_observation_managers_are_refused(make_env, tmp_path):
    """Two managers and neither is named 'policy' -- which one feeds the robot?"""
    env = make_env()
    env.observation_manager._name = "actor"  # give up the default "policy" name
    second = ObservationManager(
        env,
        cfg={"privileged": {"fn": lambda env: torch.ones((env.num_envs, 4))}},
        name="critic",
    )
    second.build()

    with pytest.raises(ExportError) as error:
        export(env, tmp_path / "bundle", verbose=False)

    message = str(error.value)
    assert "policy" in message
    assert "critic" in message
    assert "actor" in message


def test_the_policy_manager_is_chosen_over_a_critic(make_env, tmp_path):
    """The default manager is already named "policy", so a critic never competes."""
    env = make_env()
    critic = ObservationManager(
        env,
        cfg={"privileged": {"fn": lambda env: torch.ones((env.num_envs, 4))}},
        name="critic",
    )
    critic.build()

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))

    # Only the policy pipeline is deployable; the critic reads privileged state.
    assert [entry.name for entry in bundle.manifest.observations.entries] == [
        "gyro",
        "dof_pos",
    ]


def test_an_existing_destination_is_refused_without_overwrite(deployable_env, tmp_path):
    destination = export(deployable_env, tmp_path / "bundle", verbose=False)

    with pytest.raises(ExportError) as error:
        export(deployable_env, destination, verbose=False)

    assert "overwrite" in str(error.value)


def test_overwrite_replaces_the_bundle(deployable_env, tmp_path):
    destination = export(deployable_env, tmp_path / "bundle", verbose=False)

    again = export(deployable_env, destination, overwrite=True, verbose=False)

    assert again == destination
    assert load_bundle(again).manifest.num_actions == 3


def test_a_missing_policy_file_is_refused(deployable_env, tmp_path):
    with pytest.raises(ExportError) as error:
        export(
            deployable_env,
            tmp_path / "bundle",
            policy_path=tmp_path / "nope.onnx",
            verbose=False,
        )

    assert "nope.onnx" in str(error.value)


def test_a_supplied_policy_file_is_copied_into_the_bundle(deployable_env, tmp_path):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"stand-in for an exported graph")

    path = export(deployable_env, tmp_path / "bundle", policy_path=policy, verbose=False)

    bundle = load_bundle(path)
    assert bundle.policy_path is not None
    assert bundle.policy_path.read_bytes() == b"stand-in for an exported graph"


"""
Multiple action managers
"""


def test_two_action_managers_are_captured_with_their_slices(make_env, tmp_path):
    from tests.deployment.conftest import FakeActuatorManager, FakeManagedEnv

    env = FakeManagedEnv()
    env.actuator_manager = FakeActuatorManager(num_envs=env.num_envs)
    env.managers["actuator"].append(env.actuator_manager)
    env.hips = PositionActionManager(
        env, actuator_manager=env.actuator_manager, actuator_joints=[".*_hip"], scale=0.5
    )
    env.knees = PositionWithinLimitsActionManager(
        env, actuator_manager=env.actuator_manager, actuator_joints=[".*_knee"]
    )
    env.observation_manager = ObservationManager(
        env, cfg={"gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))}}
    )
    env.build()

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))
    specs = {spec.name: spec for spec in bundle.manifest.actions}

    assert (specs["hips"].slice_start, specs["hips"].slice_end) == (0, 2)
    assert (specs["knees"].slice_start, specs["knees"].slice_end) == (2, 3)
    assert specs["hips"].deploy_type == "position"
    assert specs["knees"].deploy_type == "position_within_limits"
    assert bundle.manifest.joint_names == ("FL_hip", "FR_hip", "FL_knee")


"""
Auto-detected pipeline-state observations (R15)
"""


def test_current_actions_without_a_manager_is_marked_raw(make_env, tmp_path):
    env = make_env()
    manager = ObservationManager(
        env,
        cfg={
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))},
            "actions": {"fn": current_actions()},
        },
    )
    env.managers["observation"] = [manager]
    env.observation_manager = manager
    env.actions = torch.zeros((env.num_envs, 3))
    manager.build()

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))
    entry = bundle.manifest.observations.entry("actions")

    assert entry.is_pipeline_state
    assert entry.pipeline_stage == "raw_actions"
    # Still caller-supplied, but listed as fed-back rather than as a sensor.
    assert "actions" not in [
        item.name for item in bundle.manifest.observations.sensor_inputs
    ]


def test_current_actions_with_a_manager_reads_that_managers_raw_slice(
    make_env, tmp_path
):
    """`current_actions(action_manager=...)` returns that manager's *raw* slice.

    Distinct from a lambda calling `get_actions()`, which returns the decoded
    targets. Feeding back the wrong one is silent on hardware, so the bundle
    records which and the listing says exactly where to read it.
    """
    env = make_env()
    manager = ObservationManager(
        env,
        cfg={
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))},
            "actions": {"fn": current_actions(action_manager=env.action_manager)},
        },
    )
    env.managers["observation"] = [manager]
    env.observation_manager = manager
    manager.build()

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))
    entry = bundle.manifest.observations.entry("actions")

    assert entry.pipeline_stage == "raw_actions"
    assert entry.action_manager == "action_manager"
    assert (
        entry.decoder_source
        == 'action_decoder.last_raw_actions_by_manager["action_manager"]'
    )


def test_only_the_inspectable_form_is_recognised(make_env, tmp_path):
    """Side by side: the MDP function is understood, the lambda is not.

    Both read the pipeline's own output, but only one says so in a way export can
    see -- which is the argument for reaching for current_actions().
    """
    env = make_env()
    manager = ObservationManager(
        env,
        cfg={
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))},
            "recognised": {"fn": current_actions(action_manager=env.action_manager)},
            "opaque": {"fn": lambda env: env.action_manager.get_actions()},
        },
    )
    env.managers["observation"] = [manager]
    env.observation_manager = manager
    manager.build()

    layout = load_bundle(export(env, tmp_path / "bundle", verbose=False)).manifest.observations

    assert layout.entry("recognised").pipeline_stage == "raw_actions"
    assert layout.entry("recognised").action_manager == "action_manager"
    # The lambda is indistinguishable from a sensor reading, so it is left as one.
    assert not layout.entry("opaque").is_pipeline_state


"""
Detecting action feedback (R15)

``current_actions`` is an MdpFn instance, so export can see what it reads straight
off the object. A lambda's body cannot be inspected, so it is treated as an
ordinary sensor input -- the safe default, and a nudge toward the MDP function.
"""


def observation_env(make_env, cfg, *, actions=None):
    """Attach a custom observation config to an otherwise standard environment.

    ``actions`` seeds ``env.actions``; the real GenesisEnv has it allocated by the
    time observations are built, while the test double starts it as None.
    """
    env = make_env()
    if actions is not None:
        env.actions = actions
    manager = ObservationManager(env, cfg=cfg)
    env.managers["observation"] = [manager]
    env.observation_manager = manager
    manager.build()
    return env


def test_a_lambda_is_treated_as_a_sensor_input(make_env, tmp_path):
    """Export will not guess at a body it cannot read, so it claims nothing."""
    env = observation_env(
        make_env,
        {
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))},
            "actions": {"fn": lambda env: env.action_manager.get_actions()},
        },
    )

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))
    entry = bundle.manifest.observations.entry("actions")

    assert not entry.is_pipeline_state
    assert entry.decoder_source == ""


def test_ordinary_sensor_entries_are_left_alone(make_env, tmp_path):
    env = observation_env(
        make_env, {"gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))}}
    )

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))

    assert not bundle.manifest.observations.entry("gyro").is_pipeline_state


def test_detection_identifies_which_manager_with_several_registered(tmp_path):
    from tests.deployment.conftest import FakeActuatorManager, FakeManagedEnv

    env = FakeManagedEnv()
    env.actuator_manager = FakeActuatorManager(num_envs=env.num_envs)
    env.managers["actuator"].append(env.actuator_manager)
    env.hips = PositionActionManager(
        env, actuator_manager=env.actuator_manager, actuator_joints=[".*_hip"]
    )
    env.knees = PositionActionManager(
        env, actuator_manager=env.actuator_manager, actuator_joints=[".*_knee"]
    )
    env.observation_manager = ObservationManager(
        env,
        cfg={
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))},
            "knee_actions": {"fn": current_actions(action_manager=env.knees)},
        },
    )
    env.build()

    bundle = load_bundle(export(env, tmp_path / "bundle", verbose=False))
    entry = bundle.manifest.observations.entry("knee_actions")

    assert entry.pipeline_stage == "raw_actions"
    assert entry.action_manager == "knees"
    assert entry.decoder_source == 'action_decoder.last_raw_actions_by_manager["knees"]'


def test_reading_from_an_unregistered_action_manager_is_refused(make_env, tmp_path):
    """A manager belonging to some other environment cannot be reproduced."""
    stranger = make_env().action_manager

    env = observation_env(
        make_env,
        {
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))},
            "actions": {"fn": current_actions(action_manager=stranger)},
        },
    )

    with pytest.raises(ExportError) as error:
        export(env, tmp_path / "bundle", verbose=False)

    assert "actions" in str(error.value)
    assert "not registered" in str(error.value)


"""
Packaging a policy of any format
"""


def a_torchscript_policy(tmp_path):
    module = torch.nn.Linear(4, 2).eval()
    path = tmp_path / "trained.pt"
    torch.jit.save(torch.jit.trace(module, torch.zeros(1, 4)), str(path))
    return path


def test_a_torchscript_policy_keeps_its_extension(deployable_env, tmp_path):
    """The bundle must not rename a torch archive to policy.onnx."""
    policy = a_torchscript_policy(tmp_path)

    path = export(
        deployable_env, tmp_path / "bundle", policy_path=policy, verbose=False
    )

    bundle = load_bundle(path)
    assert bundle.policy_path.name == "policy.pt"
    assert bundle.manifest.policy.format == "torchscript"


def test_an_onnx_policy_is_recorded_as_onnx(deployable_env, tmp_path):
    policy = tmp_path / "trained.onnx"
    policy.write_bytes(b"\x08\x07not-really-but-not-a-zip-either")

    path = export(
        deployable_env, tmp_path / "bundle", policy_path=policy, verbose=False
    )

    bundle = load_bundle(path)
    assert bundle.policy_path.name == "policy.onnx"
    assert bundle.manifest.policy.format == "onnx"


def test_a_mislabelled_policy_is_refused(deployable_env, tmp_path):
    """An .onnx file that is really a torch archive never reaches the bundle."""
    mislabelled = tmp_path / "trained.onnx"
    torch.jit.save(
        torch.jit.trace(torch.nn.Linear(4, 2).eval(), torch.zeros(1, 4)),
        str(mislabelled),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(ParityError) as error:
        export(deployable_env, destination, policy_path=mislabelled, verbose=False)

    assert "torch archive" in str(error.value)
    assert not destination.exists()


def test_an_unverified_policy_is_flagged_in_the_summary(deployable_env, tmp_path, capsys):
    policy = a_torchscript_policy(tmp_path)

    export(deployable_env, tmp_path / "bundle", policy_path=policy)

    output = capsys.readouterr().out
    assert "not verified" in output
    assert "reference_policy" in output


def test_describe_reports_the_policy_format(deployable_env, tmp_path):
    policy = a_torchscript_policy(tmp_path)
    path = export(
        deployable_env, tmp_path / "bundle", policy_path=policy, verbose=False
    )

    summary = load_bundle(path).describe()

    assert "policy.pt (torchscript)" in summary


"""
Provenance

Which export produced the bundle a robot is running is the first question asked
when it misbehaves, so the training framework is identified from the reference
policy rather than being declared separately.
"""


def test_the_training_framework_is_taken_from_the_reference_policy(
    deployable_env, tmp_path
):
    # The framework is read off the policy's defining module, which is how a real
    # rsl_rl export is identified (its wrapper lives in rsl_rl.models.mlp_model).
    class FrameworkPolicy(torch.nn.Module):
        def forward(self, observations):
            return observations[:, :2]

    FrameworkPolicy.__module__ = "rsl_rl.models.mlp_model"

    path = export(
        deployable_env,
        tmp_path / "bundle",
        reference_policy=FrameworkPolicy(),
        verbose=False,
    )

    provenance = load_bundle(path).manifest.provenance
    assert provenance.policy_framework == "rsl_rl"


def test_provenance_still_records_versions_without_a_reference_policy(
    deployable_env, tmp_path
):
    """Exporting the contract alone stays useful -- only the framework is unknown."""
    path = export(deployable_env, tmp_path / "bundle", verbose=False)

    provenance = load_bundle(path).manifest.provenance
    assert provenance.policy_framework is None
    assert provenance.torch_version
    assert provenance.exported_at


def test_the_framework_version_survives_a_renamed_distribution(
    deployable_env, tmp_path, monkeypatch
):
    """rsl_rl installs as rsl-rl-lib, so the import name alone finds no version."""
    import importlib.metadata

    class FrameworkPolicy(torch.nn.Module):
        def forward(self, observations):
            return observations[:, :2]

    FrameworkPolicy.__module__ = "rsl_rl.models.mlp_model"

    monkeypatch.setattr(
        importlib.metadata, "packages_distributions",
        lambda: {"rsl_rl": ["rsl-rl-lib"]},
    )
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: "5.4.2" if name == "rsl-rl-lib" else _no_such_package(name),
    )

    path = export(
        deployable_env,
        tmp_path / "bundle",
        reference_policy=FrameworkPolicy(),
        verbose=False,
    )

    provenance = load_bundle(path).manifest.provenance
    assert provenance.policy_framework == "rsl_rl"
    assert provenance.policy_framework_version == "5.4.2"


def _no_such_package(name):
    from importlib.metadata import PackageNotFoundError

    raise PackageNotFoundError(name)
