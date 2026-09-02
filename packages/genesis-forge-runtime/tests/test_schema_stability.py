"""A tripwire for changes to the bundle format.

A bundle outlives the code that wrote it. It sits on a robot for months, and the
runtime that reads it there is a separate install on its own release cadence -- so
the manifest's shape is a contract with every bundle ever exported, not an internal
detail of these dataclasses.

Nothing else notices when that shape changes. The round-trip tests move with the
code: rename a field and they are simply updated alongside it, which is exactly what
happened repeatedly while this package was being built. This test does not move --
it fails, and asks whether the change needs a schema version.
"""

import pytest

from genesis_forge_runtime import (
    SCHEMA_VERSION,
    ActionManagerSpec,
    ActuatorSpec,
    Manifest,
    ObservationEntry,
    ObservationLayout,
    PolicySpec,
    Provenance,
)

#: Every key a bundle can carry, as dotted paths. `[]` marks a list's element.
#:
#: Adding a line is usually safe -- readers use .get() and ignore keys they do not
#: know, so an older runtime tolerates a new optional field. Removing or renaming a
#: line is what an older reader gets wrong, and needs SCHEMA_VERSION bumped in
#: constants.py and WRITES_SCHEMA_VERSION bumped in genesis_forge.deployment.capture.
MANIFEST_SHAPE = [
    "actions.managers[].config.scale[]",
    "actions.managers[].delay_step",
    "actions.managers[].deploy_type",
    "actions.managers[].joint_names[]",
    "actions.managers[].name",
    "actions.managers[].slice[]",
    "actions.total_size",
    "actuators[].joint_names[]",
    "actuators[].name",
    "actuators[].randomized[]",
    "actuators[].values.kp[]",
    "control.control_hz",
    "control.dt",
    "observations.entries[].description",
    "observations.entries[].name",
    "observations.entries[].scale",
    "observations.entries[].size",
    "observations.entries[].units",
    "observations.history_length",
    "observations.history_order",
    "observations.single_size",
    "observations.total_size",
    "policy.file",
    "policy.format",
    "policy.input_name",
    "policy.normalizer.mean[]",
    "policy.output_name",
    "policy.output_semantics",
    "provenance.additional.checkpoint",
    "provenance.exported_at",
    "provenance.genesis_forge_version",
    "provenance.torch_version",
    "schema_version",
]


def every_field_populated() -> Manifest:
    """A manifest with every optional field set, so nothing hides from the snapshot."""
    return Manifest(
        dt=0.02,
        observations=ObservationLayout(
            entries=(
                ObservationEntry(
                    name="gyro",
                    size=3,
                    scale=0.25,
                    description="Body-frame angular velocity",
                    units="rad/s",
                ),
            ),
            history_length=2,
        ),
        actions=(
            ActionManagerSpec(
                name="action_manager",
                deploy_type="position",
                joint_names=("hip",),
                slice_start=0,
                slice_end=1,
                config={"scale": [0.25]},
                delay_step=1,
            ),
        ),
        actuators=(
            ActuatorSpec(
                name="actuator_manager",
                joint_names=("hip",),
                values={"kp": [50.0]},
                randomized=("kp",),
            ),
        ),
        policy=PolicySpec(
            file="policy.onnx",
            format="onnx",
            input_name="obs",
            output_name="actions",
            output_semantics="raw",
            normalizer={"mean": [0.0]},
        ),
        provenance=Provenance(
            exported_at="2026-01-01T00:00:00+00:00",
            genesis_forge_version="1.0.0",
            torch_version="2.13.0",
            additional={"checkpoint": "logs/run/model_1.pt"},
        ),
    )


def dotted_paths(value, prefix="") -> list[str]:
    if isinstance(value, dict):
        return [
            path
            for key in sorted(value)
            for path in dotted_paths(value[key], f"{prefix}.{key}" if prefix else key)
        ]
    if isinstance(value, list):
        return dotted_paths(value[0], f"{prefix}[]") if value else [f"{prefix}[]"]
    return [prefix]


def test_the_manifest_shape_has_not_changed_unnoticed():
    actual = dotted_paths(every_field_populated().to_dict())

    added = sorted(set(actual) - set(MANIFEST_SHAPE))
    removed = sorted(set(MANIFEST_SHAPE) - set(actual))

    assert not (added or removed), (
        f"The bundle manifest's shape changed.\n"
        f"  added:   {added or 'none'}\n"
        f"  removed: {removed or 'none'}\n"
        f"Every bundle ever exported carries this shape, and the runtime reading it "
        f"on a robot upgrades separately. Adding an optional key is usually safe -- "
        f"readers ignore what they do not recognise. Removing or renaming one is not: "
        f"bump SCHEMA_VERSION here and WRITES_SCHEMA_VERSION in "
        f"genesis_forge.deployment.capture, and teach Manifest.from_dict to read the "
        f"old shape. Then update MANIFEST_SHAPE."
    )


def test_the_two_packages_agree_on_the_schema_they_are_writing():
    """The exporter states its own version; it must be one this runtime accepts."""
    pytest.importorskip("genesis_forge")
    from genesis_forge.deployment.capture import WRITES_SCHEMA_VERSION

    assert WRITES_SCHEMA_VERSION == SCHEMA_VERSION, (
        f"genesis-forge writes bundle schema {WRITES_SCHEMA_VERSION} while "
        f"genesis-forge-runtime is at {SCHEMA_VERSION}. If that is deliberate -- an "
        f"exporter deliberately writing an older format -- relax this test. If not, "
        f"one of the two was bumped without the other."
    )
