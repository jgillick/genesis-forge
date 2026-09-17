"""Exporting a built environment into a deployment bundle.

Covers what lands in the manifest, that a written bundle actually drives the
runtime, and that a failed export leaves nothing behind.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from genesis_forge.deployment import ExportError, ParityError, export
from genesis_forge.managers import (
    ObservationManager,
    PositionActionManager,
    PositionWithinLimitsActionManager,
)
from genesis_forge_runtime import load_bundle

"""
A successful export
"""


def test_export_writes_a_loadable_bundle(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False).path

    bundle = load_bundle(path)

    assert bundle.manifest.dt == pytest.approx(0.02)
    assert bundle.manifest.control_hz == pytest.approx(50.0)
    assert bundle.manifest.joint_names == ("FL_hip", "FL_knee", "FR_hip")


def test_export_records_the_process_parameters(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False).path

    spec = load_bundle(path).manifest.actions[0]

    assert spec.deploy_type == "position"
    assert spec.name == "action_manager"
    np.testing.assert_allclose(spec.config["scale"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(spec.config["offset"], [0.1, 0.2, 0.3], rtol=1e-6)


def test_export_records_the_observation_layout(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False).path

    layout = load_bundle(path).manifest.observations

    assert [entry.name for entry in layout.entries] == ["gyro", "dof_pos"]
    assert layout.entry("gyro").units == "rad/s"
    assert layout.entry("gyro").scale == pytest.approx(0.25)


def test_export_records_actuator_values(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False).path

    actuators = load_bundle(path).manifest.actuators

    assert len(actuators) == 1
    np.testing.assert_allclose(actuators[0].values["kp"], [50.0, 50.0, 50.0])


def test_export_stamps_what_it_can_measure_itself(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "bundle", verbose=False).path

    provenance = load_bundle(path).manifest.provenance

    assert provenance.exported_at is not None
    assert provenance.genesis_forge_version is not None
    assert provenance.torch_version is not None
    assert provenance.additional == {}


def test_export_ships_golden_samples(deployable_env, tmp_path):
    path = export(
        deployable_env, tmp_path / "bundle", parity_ticks=5, verbose=False
    ).path

    bundle = load_bundle(path)

    assert bundle.golden["observations"].shape[0] == 5
    assert bundle.golden["joint_targets"].shape[0] == 5


def test_the_manifest_is_readable_json(deployable_env, tmp_path):
    path = export(
        deployable_env, tmp_path / "bundle", archive=False, verbose=False
    ).path

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

    path = export(deployable_env, tmp_path / "bundle", verbose=False).path
    bundle = load_bundle(path)
    assembler = bundle.create_observation_assembler()
    processor = bundle.create_action_processor()

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
    numpy_targets = processor.process(raw).targets
    torch_targets = action_manager.process_actions(
        torch.as_tensor(np.tile(raw, (deployable_env.num_envs, 1)), dtype=torch.float32)
    )[0]
    np.testing.assert_allclose(
        numpy_targets, torch_targets.numpy(), rtol=1.3e-6, atol=1e-5
    )


def test_golden_samples_replay_through_the_loaded_runtime(deployable_env, tmp_path):
    """The on-robot smoke test: recorded actions must process to recorded targets."""
    path = export(deployable_env, tmp_path / "bundle", verbose=False).path
    bundle = load_bundle(path)
    processor = bundle.create_action_processor()

    for raw, expected in zip(
        bundle.golden["raw_actions"], bundle.golden["joint_targets"], strict=True
    ):
        np.testing.assert_allclose(processor.process(raw).targets, expected, rtol=1e-6)


"""
Failed exports write nothing (AE1)
"""


def test_a_parity_failure_aborts_before_writing_anything(
    deployable_env, tmp_path, monkeypatch
):
    destination = tmp_path / "bundle"

    # Make the exported process disagree with process_actions.
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
    destination = export(
        deployable_env, tmp_path / "bundle", archive=False, verbose=False
    ).path
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
    from conftest import FakeActuatorManager, FakeManagedEnv

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

    bundle = export(env, tmp_path / "bundle", verbose=False)

    # Only the policy pipeline is deployable; the critic reads privileged state.
    assert [entry.name for entry in bundle.manifest.observations.entries] == [
        "gyro",
        "dof_pos",
    ]


def test_an_existing_destination_is_refused_when_overwrite_is_off(
    deployable_env, tmp_path
):
    destination = export(deployable_env, tmp_path / "bundle", verbose=False).path

    with pytest.raises(ExportError) as error:
        export(deployable_env, destination, overwrite=False, verbose=False)

    assert "overwrite" in str(error.value)


def test_overwrite_replaces_the_bundle(deployable_env, tmp_path):
    destination = export(deployable_env, tmp_path / "bundle", verbose=False).path

    again = export(deployable_env, destination, overwrite=True, verbose=False).path

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

    path = export(
        deployable_env, tmp_path / "bundle", policy_path=policy, verbose=False
    ).path

    bundle = load_bundle(path)
    assert bundle.policy_path is not None
    assert (bundle.path / bundle.policy_path).read_bytes() == (
        b"stand-in for an exported graph"
    )


"""
Multiple action managers
"""


def test_two_action_managers_are_captured_with_their_slices(make_env, tmp_path):
    from conftest import FakeActuatorManager, FakeManagedEnv

    env = FakeManagedEnv()
    env.actuator_manager = FakeActuatorManager(num_envs=env.num_envs)
    env.managers["actuator"].append(env.actuator_manager)
    env.hips = PositionActionManager(
        env,
        actuator_manager=env.actuator_manager,
        actuator_joints=[".*_hip"],
        scale=0.5,
    )
    env.knees = PositionWithinLimitsActionManager(
        env, actuator_manager=env.actuator_manager, actuator_joints=[".*_knee"]
    )
    env.observation_manager = ObservationManager(
        env, cfg={"gyro": {"fn": lambda env: torch.ones((env.num_envs, 3))}}
    )
    env.build()

    bundle = export(env, tmp_path / "bundle", verbose=False)
    specs = {spec.name: spec for spec in bundle.manifest.actions}

    assert (specs["hips"].slice_start, specs["hips"].slice_end) == (0, 2)
    assert (specs["knees"].slice_start, specs["knees"].slice_end) == (2, 3)
    assert specs["hips"].deploy_type == "position"
    assert specs["knees"].deploy_type == "position_within_limits"
    assert bundle.manifest.joint_names == ("FL_hip", "FR_hip", "FL_knee")


"""
Packaging a policy

The bundle carries the policy files under the names it was given, in a `policy/`
directory. It never opens them, so it records nothing about what they are.
"""


def a_torchscript_policy(tmp_path):
    module = torch.nn.Linear(4, 2).eval()
    path = tmp_path / "trained.pt"
    torch.jit.save(torch.jit.trace(module, torch.zeros(1, 4)), str(path))
    return path


def test_the_policy_keeps_the_name_it_was_given(deployable_env, tmp_path):
    policy = a_torchscript_policy(tmp_path)

    bundle = export(
        deployable_env,
        tmp_path / "bundle",
        policy_path=policy,
        archive=False,
        verbose=False,
    )

    assert bundle.manifest.policy == ("trained.pt",)
    assert bundle.policy_path.name == "trained.pt"
    assert bundle.policy_path.parent.name == "policy"
    assert (bundle.path / bundle.policy_path).read_bytes() == policy.read_bytes()


def test_describe_points_at_the_policy(deployable_env, tmp_path):
    bundle = export(
        deployable_env,
        tmp_path / "bundle",
        policy_path=a_torchscript_policy(tmp_path),
        verbose=False,
    )

    assert "policy/trained.pt" in bundle.describe()


def test_every_listed_file_is_copied(deployable_env, tmp_path):
    """A graph and its weights, or OpenVINO's .xml and .bin."""
    graph = tmp_path / "trained.onnx"
    graph.write_bytes(b"graph")
    weights = tmp_path / "trained.onnx.data"
    weights.write_bytes(b"the weights")

    path = export(
        deployable_env,
        tmp_path / "bundle",
        policy_path=[graph, weights],
        archive=False,
        verbose=False,
    ).path

    assert sorted(item.name for item in (path / "policy").iterdir()) == [
        "trained.onnx",
        "trained.onnx.data",
    ]
    assert load_bundle(path).manifest.policy == (
        "trained.onnx",
        "trained.onnx.data",
    )


def test_a_file_beside_the_policy_is_not_swept_in(deployable_env, tmp_path):
    """Only what the caller listed is packaged -- no guessing from filenames."""
    graph = tmp_path / "trained.onnx"
    graph.write_bytes(b"graph")
    (tmp_path / "trained.onnx.data").write_bytes(b"not listed")

    path = export(
        deployable_env,
        tmp_path / "bundle",
        policy_path=graph,
        archive=False,
        verbose=False,
    ).path

    assert [item.name for item in (path / "policy").iterdir()] == ["trained.onnx"]


def test_a_listed_file_that_is_missing_aborts_the_export(deployable_env, tmp_path):
    graph = tmp_path / "trained.onnx"
    graph.write_bytes(b"graph")
    destination = tmp_path / "bundle"

    with pytest.raises(ExportError) as error:
        export(
            deployable_env,
            destination,
            policy_path=[graph, tmp_path / "gone.data"],
            verbose=False,
        )

    assert "gone.data" in str(error.value)
    assert not destination.exists()


def test_an_empty_policy_list_means_no_policy(deployable_env, tmp_path):
    bundle = export(deployable_env, tmp_path / "bundle", policy_path=[], verbose=False)

    assert bundle.manifest.policy == ()
    assert bundle.policy_path is None


def test_files_that_would_share_a_name_are_refused(deployable_env, tmp_path):
    """They keep their own names, so two of the same name would overwrite."""
    first = tmp_path / "a"
    first.mkdir()
    (first / "model.onnx").write_bytes(b"one")
    second = tmp_path / "b"
    second.mkdir()
    (second / "model.onnx").write_bytes(b"two")

    with pytest.raises(ExportError) as error:
        export(
            deployable_env,
            tmp_path / "bundle",
            policy_path=[first / "model.onnx", second / "model.onnx"],
            verbose=False,
        )

    assert "model.onnx" in str(error.value)


"""
Bundles written as a single file

An archive is the same bundle in one artifact -- what you copy to a robot. The
runtime reads either form, so nothing downstream needs to know which it was given.
"""


def test_an_archive_round_trips_through_the_runtime(deployable_env, tmp_path):
    source = tmp_path / "trained.pt"
    torch.jit.save(
        torch.jit.trace(torch.nn.Linear(4, 2).eval(), torch.zeros(1, 4)), str(source)
    )

    path = export(
        deployable_env,
        tmp_path / "my_policy",
        policy_path=source,
        archive=True,
        verbose=False,
    ).path

    assert path == tmp_path / "my_policy.gfb"
    assert path.is_file()

    bundle = load_bundle(path)
    assert bundle.manifest.num_actions == 3
    assert (bundle.path / bundle.policy_path).read_bytes() == source.read_bytes()
    assert bundle.golden is not None


def test_an_archive_holds_exactly_what_the_directory_would(deployable_env, tmp_path):
    import zipfile

    directory = export(
        deployable_env, tmp_path / "as_dir", archive=False, verbose=False
    ).path
    packed = export(
        deployable_env, tmp_path / "as_zip", archive=True, verbose=False
    ).path

    with zipfile.ZipFile(packed) as archive:
        assert sorted(archive.namelist()) == sorted(
            item.name for item in directory.iterdir()
        )


def test_an_existing_directory_is_not_replaced_by_an_archive(deployable_env, tmp_path):
    """Two different things sharing a name is a mistake worth stopping."""
    clash = tmp_path / "my_policy.gfb"
    clash.mkdir()

    with pytest.raises(ExportError) as error:
        export(deployable_env, clash, archive=True, overwrite=True, verbose=False)

    assert "is a directory" in str(error.value)


def test_a_path_that_already_has_a_suffix_is_left_alone(deployable_env, tmp_path):
    path = export(
        deployable_env, tmp_path / "policy.zip", archive=True, verbose=False
    ).path

    assert path == tmp_path / "policy.zip"
    assert load_bundle(path).manifest.num_actions == 3


"""
Defaults

An archive, replacing whatever bundle was there before: re-exporting after a
training run is the normal thing to do, and the result is the thing you ship.
"""


def test_the_default_is_a_single_archive(deployable_env, tmp_path):
    path = export(deployable_env, tmp_path / "my_policy", verbose=False).path

    assert path == tmp_path / "my_policy.gfb"
    assert path.is_file()
    assert load_bundle(path).manifest.num_actions == 3


def test_re_exporting_replaces_the_previous_bundle_without_asking(
    deployable_env, tmp_path
):
    first = export(deployable_env, tmp_path / "my_policy", verbose=False).path
    before = first.stat().st_mtime_ns

    again = export(deployable_env, tmp_path / "my_policy", verbose=False).path

    assert again == first
    assert again.stat().st_mtime_ns != before
    assert load_bundle(again).manifest.num_actions == 3


def test_a_file_that_is_not_a_bundle_is_never_overwritten(deployable_env, tmp_path):
    """Overwriting is the default, so a mistyped destination must not cost work."""
    precious = tmp_path / "results.gfb"
    precious.write_text("a year of experiments")

    with pytest.raises(ExportError) as error:
        export(deployable_env, precious, verbose=False)

    assert "not a deployment bundle" in str(error.value)
    assert precious.read_text() == "a year of experiments"


def test_a_directory_that_is_not_a_bundle_is_never_overwritten(
    deployable_env, tmp_path
):
    precious = tmp_path / "results"
    precious.mkdir()
    (precious / "data.csv").write_text("a year of experiments")

    with pytest.raises(ExportError) as error:
        export(deployable_env, precious, archive=False, verbose=False)

    assert "not a deployment bundle" in str(error.value)
    assert (precious / "data.csv").read_text() == "a year of experiments"


def test_an_existing_bundle_is_recognised_and_replaced(deployable_env, tmp_path):
    """The guard must not get in the way of the case it exists to allow."""
    directory = export(
        deployable_env, tmp_path / "as_dir", archive=False, verbose=False
    ).path
    archive = export(deployable_env, tmp_path / "as_zip", verbose=False).path

    export(deployable_env, directory, archive=False, verbose=False)
    export(deployable_env, archive, verbose=False)

    assert load_bundle(directory).manifest.num_actions == 3
    assert load_bundle(archive).manifest.num_actions == 3


"""
What export hands back

A Bundle, carrying the manifest and golden samples it just built. Describing or
checking what you exported should not read the bundle back off disk, and on the
common path -- an archive -- should not unpack it either.
"""


def test_export_returns_a_usable_bundle_without_reading_anything_back(
    deployable_env, tmp_path
):
    bundle = export(deployable_env, tmp_path / "my_policy", verbose=False)

    assert bundle.path == tmp_path / "my_policy.gfb"
    assert bundle.manifest.num_actions == 3
    assert bundle.golden["observations"].shape[0] == 6


def test_describing_an_archive_does_not_unpack_it(deployable_env, tmp_path):
    """The complaint this fixes: a stray directory appearing next to the artifact."""
    source = tmp_path / "trained.onnx"
    source.write_bytes(b"\x08\x07 graph")
    bundle = export(
        deployable_env, tmp_path / "my_policy", policy_path=source, verbose=False
    )

    summary = bundle.describe()

    assert "policy/trained.onnx" in summary
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "my_policy.gfb",
        "trained.onnx",
    ]


def test_the_policy_of_an_archive_is_a_path_relative_to_the_bundle(
    deployable_env, tmp_path
):
    source = tmp_path / "trained.onnx"
    source.write_bytes(b"graph")
    bundle = export(
        deployable_env, tmp_path / "my_policy", policy_path=source, verbose=False
    )

    # Relative, so it needs no unpacking to be useful, and the same expression
    # works once the archive is opened.
    assert bundle.policy_path == Path("policy") / "trained.onnx"
    with bundle.unpacked() as directory:
        assert (directory / bundle.policy_path).read_bytes() == b"graph"


def test_unpacked_yields_the_contents_and_cleans_up_after_itself(
    deployable_env, tmp_path
):
    source = tmp_path / "trained.onnx"
    source.write_bytes(b"\x08\x07 graph")
    bundle = export(
        deployable_env, tmp_path / "my_policy", policy_path=source, verbose=False
    )

    with bundle.unpacked() as directory:
        assert (directory / "policy" / "trained.onnx").read_bytes() == b"\x08\x07 graph"
        assert (directory / "manifest.json").is_file()
        scratch = directory

    assert not scratch.exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "my_policy.gfb",
        "trained.onnx",
    ]


def test_unpacked_on_a_directory_bundle_hands_back_the_directory_itself(
    deployable_env, tmp_path
):
    """Nothing is copied when the files are already sitting there."""
    bundle = export(
        deployable_env, tmp_path / "my_policy", archive=False, verbose=False
    )

    with bundle.unpacked() as directory:
        assert directory == bundle.path

    assert directory.is_dir()  # still there afterwards


def test_a_disabled_observation_manager_is_refused(deployable_env, tmp_path):
    """Training feeds zeros through a disabled manager; the bundle would not."""
    deployable_env.observation_manager.enabled = False

    with pytest.raises(ExportError) as error:
        export(deployable_env, tmp_path / "bundle", verbose=False)

    assert "disabled" in str(error.value)


def test_a_per_element_scale_is_refused_with_a_usable_message(deployable_env, tmp_path):
    """The manifest holds one scale per entry; a list would silently not fit."""
    deployable_env.observation_manager.cfg["gyro"].scale = [1.0, 2.0, 3.0]

    with pytest.raises(ExportError) as error:
        export(deployable_env, tmp_path / "bundle", verbose=False)

    message = str(error.value)
    assert "per-element scale" in message
    assert "gyro" in message
