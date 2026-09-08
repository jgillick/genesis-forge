"""Bundle schema, loading, and validation.

These tests are numpy-only on purpose: this package is what gets installed on a
robot, so nothing here may import torch or the Genesis simulator. The import
guard at the bottom enforces that in a clean subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from genesis_forge_runtime import (
    SCHEMA_VERSION,
    ActionManagerSpec,
    ActuatorSpec,
    MalformedBundleError,
    Manifest,
    ObservationEntry,
    ObservationLayout,
    Provenance,
    SchemaVersionError,
    load_bundle,
    save_bundle,
)


def make_manifest(**overrides) -> Manifest:
    """A small but complete manifest: two observation entries, one action manager."""
    defaults = {
        "dt": 0.02,
        "observations": ObservationLayout(
            entries=(
                ObservationEntry(
                    name="robot_ang_vel",
                    size=3,
                    scale=0.25,
                    description="Body-frame angular velocity",
                    units="rad/s",
                ),
                ObservationEntry(name="dof_pos", size=2, scale=1.0),
            ),
            history_length=2,
        ),
        "actions": (
            ActionManagerSpec(
                name="action_manager",
                deploy_type="position",
                joint_names=("hip", "knee"),
                slice_start=0,
                slice_end=2,
                # Plain JSON data, as the exporter writes it.
                config={
                    "scale": [0.5, 0.5],
                    "offset": [0.1, -0.1],
                    "clip_low": [-1.0, -1.0],
                    "clip_high": [1.0, 1.0],
                    "mode": "affine",
                },
            ),
        ),
        "actuators": (
            ActuatorSpec(
                name="actuator_manager",
                joint_names=("hip", "knee"),
                values={"kp": [50.0, 50.0], "kv": [0.5, 0.5]},
                randomized=("default_pos",),
            ),
        ),
        "provenance": Provenance(
            genesis_forge_version="1.0.0", exported_at="2026-08-24"
        ),
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def write_bundle(tmp_path, manifest=None, **save_kwargs):
    return save_bundle(tmp_path / "bundle", manifest or make_manifest(), **save_kwargs)


"""Round-tripping"""


def test_round_trip_preserves_scalar_fields(tmp_path):
    path = write_bundle(tmp_path)
    loaded = load_bundle(path).manifest

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.dt == pytest.approx(0.02)
    assert loaded.control_hz == pytest.approx(50.0)
    assert loaded.num_actions == 2
    assert loaded.joint_names == ("hip", "knee")
    assert loaded.provenance.genesis_forge_version == "1.0.0"


def test_round_trip_preserves_observation_layout(tmp_path):
    loaded = load_bundle(write_bundle(tmp_path)).manifest.observations

    assert [entry.name for entry in loaded.entries] == ["robot_ang_vel", "dof_pos"]
    assert loaded.single_size == 5
    assert loaded.history_length == 2
    assert loaded.total_size == 10

    ang_vel = loaded.entry("robot_ang_vel")
    assert ang_vel.scale == pytest.approx(0.25)
    assert ang_vel.units == "rad/s"
    assert ang_vel.description == "Body-frame angular velocity"


def test_config_and_actuator_values_load_as_plain_json_data(tmp_path):
    """Nothing is reshaped on load -- a decoder converts what it needs itself."""
    manifest = load_bundle(write_bundle(tmp_path)).manifest

    assert manifest.actions[0].config["scale"] == [0.5, 0.5]
    assert manifest.actions[0].config["mode"] == "affine"
    assert manifest.actuators[0].values["kp"] == [50.0, 50.0]
    assert manifest.actuators[0].randomized == ("default_pos",)


def test_manifest_json_is_human_readable(tmp_path):
    path = write_bundle(tmp_path)
    data = json.loads((path / "manifest.json").read_text())

    # Arrays are inline lists, not opaque blobs -- the point of a readable manifest.
    assert data["actions"]["managers"][0]["config"]["scale"] == [0.5, 0.5]
    assert data["control"]["dt"] == 0.02


"""Schema versioning"""


def test_newer_schema_version_names_both_versions(tmp_path):
    path = write_bundle(tmp_path)
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text())
    data["schema_version"] = SCHEMA_VERSION + 5
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(SchemaVersionError) as error:
        load_bundle(path)

    message = str(error.value)
    assert str(SCHEMA_VERSION + 5) in message
    assert str(SCHEMA_VERSION) in message


def test_older_schema_version_is_rejected(tmp_path):
    path = write_bundle(tmp_path)
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text())
    data["schema_version"] = 0
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(SchemaVersionError):
        load_bundle(path)


def test_non_integer_schema_version_is_rejected(tmp_path):
    path = write_bundle(tmp_path)
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text())
    data["schema_version"] = "1"
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(MalformedBundleError):
        load_bundle(path)


"""Malformed bundles"""


@pytest.mark.parametrize("section", ["control", "observations", "actions"])
def test_missing_section_names_the_section(tmp_path, section):
    path = write_bundle(tmp_path)
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text())
    del data[section]
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert section in str(error.value)


def test_missing_dt_names_the_field(tmp_path):
    path = write_bundle(tmp_path)
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text())
    del data["control"]["dt"]
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "dt" in str(error.value)


def test_invalid_json_is_reported_as_malformed(tmp_path):
    path = write_bundle(tmp_path)
    (path / "manifest.json").write_text("{not json")

    with pytest.raises(MalformedBundleError):
        load_bundle(path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(MalformedBundleError):
        load_bundle(tmp_path / "does_not_exist")


def test_duplicate_observation_names_are_rejected(tmp_path):
    manifest = make_manifest(
        observations=ObservationLayout(
            entries=(
                ObservationEntry(name="dup", size=1),
                ObservationEntry(name="dup", size=2),
            )
        )
    )
    path = write_bundle(tmp_path, manifest)

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "dup" in str(error.value)


def test_action_slices_must_tile_without_gaps(tmp_path):
    manifest = make_manifest(
        actions=(
            ActionManagerSpec(
                name="first",
                deploy_type="position",
                joint_names=("a",),
                slice_start=0,
                slice_end=1,
                config={},
            ),
            # Starts at 3, leaving a hole at index 1-2.
            ActionManagerSpec(
                name="second",
                deploy_type="position",
                joint_names=("b",),
                slice_start=3,
                slice_end=4,
                config={},
            ),
        )
    )
    path = write_bundle(tmp_path, manifest)

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "second" in str(error.value)


def test_joint_count_must_match_slice_width(tmp_path):
    manifest = make_manifest(
        actions=(
            ActionManagerSpec(
                name="mismatched",
                deploy_type="position",
                joint_names=("only_one",),
                slice_start=0,
                slice_end=3,
                config={},
            ),
        )
    )
    path = write_bundle(tmp_path, manifest)

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "mismatched" in str(error.value)


def test_bundle_without_policy_loads(tmp_path):
    bundle = load_bundle(write_bundle(tmp_path))

    assert bundle.manifest.policy == ()
    assert bundle.policy_path is None
    assert bundle.golden is None


def test_bundle_with_policy_resolves_its_path(tmp_path):
    manifest = make_manifest(policy=("policy.onnx",))
    path = write_bundle(tmp_path, manifest)
    (path / "policy").mkdir()
    (path / "policy" / "policy.onnx").write_bytes(b"not-a-real-onnx-file")

    bundle = load_bundle(path)

    assert bundle.policy_path == path / "policy" / "policy.onnx"
    assert bundle.policy_files == ("policy.onnx",)


def test_manifest_referencing_a_missing_policy_file_raises(tmp_path):
    manifest = make_manifest(policy=("policy.onnx",))
    path = write_bundle(tmp_path, manifest)
    # Deliberately do not write policy.onnx.

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "policy.onnx" in str(error.value)


def test_golden_samples_round_trip(tmp_path):
    golden = {
        "observations": np.arange(10, dtype=np.float32).reshape(2, 5),
        "actions": np.ones((2, 2), dtype=np.float32),
    }
    path = write_bundle(tmp_path, golden=golden)

    bundle = load_bundle(path)

    assert set(bundle.golden) == {"observations", "actions"}
    np.testing.assert_allclose(bundle.golden["actions"], np.ones((2, 2)))


def test_golden_samples_can_be_skipped(tmp_path):
    path = write_bundle(
        tmp_path, golden={"observations": np.zeros(3, dtype=np.float32)}
    )

    assert load_bundle(path, load_golden=False).golden is None


"""Listings"""


def test_the_layout_lists_its_entries_in_vector_order():
    layout = ObservationLayout(
        entries=(
            ObservationEntry(name="gyro", size=3),
            ObservationEntry(name="actions", size=2),
        )
    )

    assert [entry.name for entry in layout.entries] == ["gyro", "actions"]
    assert layout.single_size == 5


def test_describe_lists_what_to_wire(tmp_path):
    summary = load_bundle(write_bundle(tmp_path)).describe()

    assert "50.0 Hz" in summary
    assert "robot_ang_vel" in summary
    assert "rad/s" in summary
    assert "hip, knee" in summary


"""Dependency isolation (AE5)"""


def test_importing_the_runtime_does_not_pull_in_torch_or_genesis():
    """The whole point of this package: it must install and run without the sim stack.

    Runs in a clean subprocess because the repo's own test session imports torch and
    genesis via tests/conftest.py, which would make an in-process check meaningless.
    """
    probe = (
        "import sys; import genesis_forge_runtime; "
        "leaked = sorted(m for m in ('torch', 'genesis', 'genesis_forge') "
        "if m in sys.modules); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"heavy modules leaked in: {result.stdout.strip()}"
    )


"""Action groups in the manifest"""


def grouped_manager(**overrides):
    defaults = dict(
        name="wheels",
        deploy_type="velocity",
        joint_names=("fr", "br", "fl", "bl"),
        slice_start=0,
        slice_end=2,
        config={"scale": [1.0, 1.0, -1.0, -1.0]},
        joint_action_index=(0, 0, 1, 1),
    )
    defaults.update(overrides)
    return ActionManagerSpec(**defaults)


def test_the_mapping_round_trips_as_integers(tmp_path):
    """Indices stay integers -- they address joints, they are not decode values."""
    manifest = make_manifest(actions=(grouped_manager(),))
    path = write_bundle(tmp_path, manifest)

    raw = json.loads((path / "manifest.json").read_text())
    stored = raw["actions"]["managers"][0]["joint_action_index"]
    assert stored == [0, 0, 1, 1]
    assert all(isinstance(index, int) for index in stored)

    loaded = load_bundle(path).manifest.actions[0]
    assert loaded.joint_action_index == (0, 0, 1, 1)
    assert loaded.num_actions == 2
    assert loaded.num_joints == 4


def test_a_mapping_that_does_not_cover_every_joint_is_refused(tmp_path):
    with pytest.raises(MalformedBundleError) as error:
        write_bundle(
            tmp_path,
            make_manifest(actions=(grouped_manager(joint_action_index=(0, 0, 1)),)),
        )
        load_bundle(tmp_path / "bundle")

    assert "maps" in str(error.value)


def test_a_mapping_pointing_outside_the_slice_is_refused(tmp_path):
    path = write_bundle(
        tmp_path,
        make_manifest(actions=(grouped_manager(joint_action_index=(0, 0, 1, 5)),)),
    )

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "outside its slice" in str(error.value)


def test_a_non_integer_mapping_is_refused(tmp_path):
    path = write_bundle(tmp_path, make_manifest(actions=(grouped_manager(),)))
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text())
    data["actions"]["managers"][0]["joint_action_index"] = ["a", "b", "c", "d"]
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(path)

    assert "non-integer" in str(error.value)


def test_provenance_refuses_what_the_manifest_cannot_hold():
    """The invariant belongs to the field's owner, so every caller gets it."""
    with pytest.raises(TypeError) as error:
        Provenance(additional={"checkpoint": Path("logs/model.pt")})

    assert "not JSON serializable" in str(error.value)
