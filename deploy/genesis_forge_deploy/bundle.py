"""Deployment bundle schema, loading, and validation.

A bundle is a directory written by ``genesis_forge.deployment.export`` on the
training machine and read here, on the robot::

    my_policy/
      manifest.json   # everything the runtime needs, human readable
      golden.npz      # recorded input/output pairs for the on-robot smoke test
      policy.onnx     # optional: the exported policy

Nothing in this module imports torch or genesis -- that is the whole point of
this package, and ``test_bundle.py`` asserts it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Schema version this runtime writes and understands.
SCHEMA_VERSION = 1

#: Oldest bundle schema this runtime can still read.
MIN_SUPPORTED_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
GOLDEN_FILENAME = "golden.npz"
POLICY_FILENAME = "policy.onnx"

#: An observation entry the user reads off real hardware each tick.
SOURCE_SENSOR = "sensor"
#: An observation entry the runtime fills from its own previous output.
SOURCE_PIPELINE_STATE = "pipeline_state"

#: Pipeline-state entry echoing the raw policy output.
STAGE_RAW_ACTIONS = "raw_actions"
#: Pipeline-state entry echoing an action manager's decoded (processed) output.
STAGE_PROCESSED_ACTIONS = "processed_actions"

#: History is concatenated newest-first, matching ObservationManager.
HISTORY_NEWEST_FIRST = "newest_first"


class BundleError(Exception):
    """Base class for every error raised while reading a bundle."""


class SchemaVersionError(BundleError):
    """The bundle was written by an incompatible version of Genesis Forge."""


class MalformedBundleError(BundleError):
    """The bundle is missing a required section or holds an unusable value."""


def _require(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    """Fetch ``key`` or raise an error that names both the key and its section."""
    if not isinstance(mapping, dict):
        raise MalformedBundleError(
            f"Expected '{where}' to be a JSON object, got {type(mapping).__name__}."
        )
    if key not in mapping:
        available = ", ".join(sorted(mapping)) or "nothing"
        raise MalformedBundleError(
            f"Missing required field '{key}' in '{where}' (found: {available})."
        )
    return mapping[key]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _decode_value(value: Any) -> Any:
    """Turn nested numeric lists into float32 arrays, leaving everything else alone.

    Action manager configs are opaque by design -- a custom manager defines its own
    schema -- so the rule is structural rather than key-based: anything that looks
    like a numeric array becomes one, so custom decoders get arrays for free.
    """
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(_is_number(item) for item in value):
            return np.asarray(value, dtype=np.float32)
        if value and all(
            isinstance(row, list) and all(_is_number(item) for item in row)
            for row in value
        ):
            return np.asarray(value, dtype=np.float32)
        return [_decode_value(item) for item in value]
    return value


def _encode_value(value: Any) -> Any:
    """Inverse of :func:`_decode_value`, for writing JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservationEntry:
    """One named slot in the policy's observation vector."""

    name: str
    size: int
    scale: float = 1.0
    description: str | None = None
    units: str | None = None
    source: str = SOURCE_SENSOR
    pipeline_stage: str | None = None
    action_manager: str | None = None

    @property
    def is_pipeline_state(self) -> bool:
        """True when the runtime fills this entry itself instead of the user."""
        return self.source == SOURCE_PIPELINE_STATE

    def describe(self) -> str:
        """One-line human summary, used by the listings and the wiring stub."""
        parts = [f"{self.name} ({self.size} value{'s' if self.size != 1 else ''})"]
        if self.units:
            parts.append(f"in {self.units}")
        if self.scale != 1.0:
            parts.append(f"scaled by {self.scale}")
        if self.is_pipeline_state:
            parts.append(f"auto-filled from {self.pipeline_stage}")
        summary = ", ".join(parts)
        if self.description:
            summary = f"{summary} -- {self.description}"
        return summary

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, where: str) -> ObservationEntry:
        name = _require(data, "name", where=where)
        entry = cls(
            name=name,
            size=int(_require(data, "size", where=f"{where}.{name}")),
            scale=float(data.get("scale", 1.0)),
            description=data.get("description"),
            units=data.get("units"),
            source=data.get("source", SOURCE_SENSOR),
            pipeline_stage=data.get("pipeline_stage"),
            action_manager=data.get("action_manager"),
        )
        if entry.source not in (SOURCE_SENSOR, SOURCE_PIPELINE_STATE):
            raise MalformedBundleError(
                f"Observation entry '{name}' has unknown source '{entry.source}'. "
                f"Expected '{SOURCE_SENSOR}' or '{SOURCE_PIPELINE_STATE}'."
            )
        if entry.is_pipeline_state and entry.pipeline_stage not in (
            STAGE_RAW_ACTIONS,
            STAGE_PROCESSED_ACTIONS,
        ):
            raise MalformedBundleError(
                f"Observation entry '{name}' is marked pipeline state but its stage is "
                f"'{entry.pipeline_stage}'. Expected '{STAGE_RAW_ACTIONS}' or "
                f"'{STAGE_PROCESSED_ACTIONS}'."
            )
        if (
            entry.pipeline_stage == STAGE_PROCESSED_ACTIONS
            and not entry.action_manager
        ):
            raise MalformedBundleError(
                f"Observation entry '{name}' echoes processed actions but does not say "
                f"which action manager they come from, so the runtime cannot fill it. "
                f"Re-export with a current version of Genesis Forge."
            )
        return entry

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "size": self.size, "scale": self.scale}
        if self.description is not None:
            data["description"] = self.description
        if self.units is not None:
            data["units"] = self.units
        if self.source != SOURCE_SENSOR:
            data["source"] = self.source
        if self.pipeline_stage is not None:
            data["pipeline_stage"] = self.pipeline_stage
        if self.action_manager is not None:
            data["action_manager"] = self.action_manager
        return data


@dataclass(frozen=True)
class ObservationLayout:
    """Ordering, scaling, and history configuration of the policy's input vector."""

    entries: tuple[ObservationEntry, ...]
    history_length: int = 1
    history_order: str = HISTORY_NEWEST_FIRST

    @property
    def single_size(self) -> int:
        """Width of one observation tick, before history stacking."""
        return sum(entry.size for entry in self.entries)

    @property
    def total_size(self) -> int:
        """Width of the full vector handed to the policy."""
        return self.single_size * self.history_length

    @property
    def required_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries the user must supply each tick (pipeline state is auto-filled)."""
        return tuple(entry for entry in self.entries if not entry.is_pipeline_state)

    @property
    def pipeline_state_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries the runtime fills from its own previous output."""
        return tuple(entry for entry in self.entries if entry.is_pipeline_state)

    def entry(self, name: str) -> ObservationEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        known = ", ".join(item.name for item in self.entries)
        raise KeyError(f"No observation entry named '{name}'. Known entries: {known}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationLayout:
        where = "observations"
        raw_entries = _require(data, "entries", where=where)
        if not isinstance(raw_entries, list) or not raw_entries:
            raise MalformedBundleError(
                "'observations.entries' must be a non-empty list of observation entries."
            )
        entries = tuple(
            ObservationEntry.from_dict(item, where=f"{where}.entries")
            for item in raw_entries
        )
        names = [entry.name for entry in entries]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise MalformedBundleError(
                f"Duplicate observation entry names: {', '.join(sorted(duplicates))}."
            )
        history_length = int(data.get("history_length", 1))
        if history_length < 1:
            raise MalformedBundleError(
                f"'observations.history_length' must be at least 1, got {history_length}."
            )
        history_order = data.get("history_order", HISTORY_NEWEST_FIRST)
        if history_order != HISTORY_NEWEST_FIRST:
            raise MalformedBundleError(
                f"Unsupported observation history order '{history_order}'. This runtime "
                f"only implements '{HISTORY_NEWEST_FIRST}'."
            )
        return cls(
            entries=entries,
            history_length=history_length,
            history_order=history_order,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_length": self.history_length,
            "history_order": self.history_order,
            "single_size": self.single_size,
            "total_size": self.total_size,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class ActionManagerSpec:
    """How one action manager's slice of the policy output is decoded."""

    name: str
    deploy_type: str
    joint_names: tuple[str, ...]
    slice_start: int
    slice_end: int
    config: dict[str, Any]
    decoder_import_path: str | None = None
    delay_step: int = 0

    @property
    def num_actions(self) -> int:
        return self.slice_end - self.slice_start

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, where: str) -> ActionManagerSpec:
        name = _require(data, "name", where=where)
        scope = f"{where}.{name}"
        bounds = _require(data, "slice", where=scope)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise MalformedBundleError(
                f"Action manager '{name}' has a malformed 'slice': expected [start, end], "
                f"got {bounds!r}."
            )
        joint_names = tuple(_require(data, "joint_names", where=scope))
        spec = cls(
            name=name,
            deploy_type=_require(data, "deploy_type", where=scope),
            joint_names=joint_names,
            slice_start=int(bounds[0]),
            slice_end=int(bounds[1]),
            config=_decode_value(data.get("config", {})),
            decoder_import_path=data.get("decoder_import_path"),
            delay_step=int(data.get("delay_step", 0)),
        )
        if spec.num_actions != len(joint_names):
            raise MalformedBundleError(
                f"Action manager '{name}' covers {spec.num_actions} actions but names "
                f"{len(joint_names)} joints; the bundle is inconsistent."
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "deploy_type": self.deploy_type,
            "slice": [self.slice_start, self.slice_end],
            "joint_names": list(self.joint_names),
            "config": _encode_value(self.config),
        }
        if self.decoder_import_path is not None:
            data["decoder_import_path"] = self.decoder_import_path
        if self.delay_step:
            data["delay_step"] = self.delay_step
        return data


@dataclass(frozen=True)
class ActuatorSpec:
    """Nominal actuator gains and defaults, recorded so the robot can match training."""

    name: str
    joint_names: tuple[str, ...]
    values: dict[str, np.ndarray]
    randomized: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, where: str) -> ActuatorSpec:
        name = _require(data, "name", where=where)
        scope = f"{where}.{name}"
        return cls(
            name=name,
            joint_names=tuple(_require(data, "joint_names", where=scope)),
            values=_decode_value(data.get("values", {})),
            randomized=tuple(data.get("randomized", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "joint_names": list(self.joint_names),
            "values": _encode_value(self.values),
        }
        if self.randomized:
            data["randomized"] = list(self.randomized)
        return data


@dataclass(frozen=True)
class PolicySpec:
    """Where the exported policy lives and what its output means."""

    file: str | None = None
    input_name: str = "obs"
    output_name: str = "actions"
    output_semantics: str = "raw"
    normalizer: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySpec:
        return cls(
            file=data.get("file"),
            input_name=data.get("input_name", "obs"),
            output_name=data.get("output_name", "actions"),
            output_semantics=data.get("output_semantics", "raw"),
            normalizer=_decode_value(data["normalizer"])
            if data.get("normalizer") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "input_name": self.input_name,
            "output_name": self.output_name,
            "output_semantics": self.output_semantics,
        }
        if self.file is not None:
            data["file"] = self.file
        if self.normalizer is not None:
            data["normalizer"] = _encode_value(self.normalizer)
        return data


@dataclass(frozen=True)
class Provenance:
    """Where this bundle came from -- the first thing to check when debugging."""

    exported_at: str | None = None
    genesis_forge_version: str | None = None
    torch_version: str | None = None
    policy_framework: str | None = None
    policy_framework_version: str | None = None
    checkpoint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            exported_at=data.get("exported_at"),
            genesis_forge_version=data.get("genesis_forge_version"),
            torch_version=data.get("torch_version"),
            policy_framework=data.get("policy_framework"),
            policy_framework_version=data.get("policy_framework_version"),
            checkpoint=data.get("checkpoint"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "exported_at": self.exported_at,
                "genesis_forge_version": self.genesis_forge_version,
                "torch_version": self.torch_version,
                "policy_framework": self.policy_framework,
                "policy_framework_version": self.policy_framework_version,
                "checkpoint": self.checkpoint,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class Manifest:
    """The full deployment contract, as read from ``manifest.json``."""

    dt: float
    observations: ObservationLayout
    actions: tuple[ActionManagerSpec, ...]
    actuators: tuple[ActuatorSpec, ...] = ()
    policy: PolicySpec | None = None
    provenance: Provenance = Provenance()
    schema_version: int = SCHEMA_VERSION

    @property
    def control_hz(self) -> float:
        """Rate the policy was trained at; run the robot loop at this rate."""
        return 1.0 / self.dt

    @property
    def num_actions(self) -> int:
        return max((spec.slice_end for spec in self.actions), default=0)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Every controlled joint, in policy-output order."""
        names: list[str] = []
        for spec in sorted(self.actions, key=lambda item: item.slice_start):
            names.extend(spec.joint_names)
        return tuple(names)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        version = _require(data, "schema_version", where="manifest")
        _check_schema_version(version)

        control = _require(data, "control", where="manifest")
        dt = float(_require(control, "dt", where="control"))
        if dt <= 0:
            raise MalformedBundleError(f"'control.dt' must be positive, got {dt}.")

        raw_actions = _require(data, "actions", where="manifest")
        actions = tuple(
            ActionManagerSpec.from_dict(item, where="actions")
            for item in _require(raw_actions, "managers", where="actions")
        )
        _check_action_slices(actions)

        return cls(
            schema_version=int(version),
            dt=dt,
            observations=ObservationLayout.from_dict(
                _require(data, "observations", where="manifest")
            ),
            actions=actions,
            actuators=tuple(
                ActuatorSpec.from_dict(item, where="actuators")
                for item in data.get("actuators", [])
            ),
            policy=PolicySpec.from_dict(data["policy"])
            if data.get("policy") is not None
            else None,
            provenance=Provenance.from_dict(data.get("provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "provenance": self.provenance.to_dict(),
            "control": {"dt": self.dt, "control_hz": self.control_hz},
            "observations": self.observations.to_dict(),
            "actions": {
                "total_size": self.num_actions,
                "managers": [spec.to_dict() for spec in self.actions],
            },
        }
        if self.actuators:
            data["actuators"] = [spec.to_dict() for spec in self.actuators]
        if self.policy is not None:
            data["policy"] = self.policy.to_dict()
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise MalformedBundleError(f"manifest.json is not valid JSON: {error}") from error
        return cls.from_dict(data)


def _check_schema_version(version: Any) -> None:
    """Fail loudly, and specifically, on a bundle this runtime cannot read."""
    if not isinstance(version, int) or isinstance(version, bool):
        raise MalformedBundleError(
            f"'schema_version' must be an integer, got {version!r}."
        )
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"This bundle uses schema version {version}, but this runtime only "
            f"understands up to version {SCHEMA_VERSION}. Upgrade genesis-forge-deploy."
        )
    if version < MIN_SUPPORTED_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"This bundle uses schema version {version}, which is older than the "
            f"oldest version this runtime supports ({MIN_SUPPORTED_SCHEMA_VERSION}). "
            f"Re-export it with a current version of Genesis Forge."
        )


def _check_action_slices(actions: tuple[ActionManagerSpec, ...]) -> None:
    """Action manager slices must tile the policy output with no gaps or overlaps."""
    if not actions:
        return
    ordered = sorted(actions, key=lambda spec: spec.slice_start)
    cursor = 0
    for spec in ordered:
        if spec.slice_start != cursor:
            raise MalformedBundleError(
                f"Action manager '{spec.name}' starts at index {spec.slice_start} but the "
                f"previous manager ended at {cursor}; slices must tile the policy output "
                f"without gaps or overlaps."
            )
        if spec.slice_end <= spec.slice_start:
            raise MalformedBundleError(
                f"Action manager '{spec.name}' has an empty slice "
                f"[{spec.slice_start}, {spec.slice_end}]."
            )
        cursor = spec.slice_end


@dataclass(frozen=True)
class Bundle:
    """A loaded deployment bundle: the manifest plus the files beside it."""

    manifest: Manifest
    path: Path
    golden: dict[str, np.ndarray] | None = None

    @property
    def policy_path(self) -> Path | None:
        """Absolute path to the exported policy, when the bundle carries one."""
        if self.manifest.policy is None or self.manifest.policy.file is None:
            return None
        return self.path / self.manifest.policy.file

    def observation_assembler(self, **kwargs: Any) -> Any:
        """Build the :class:`ObservationAssembler` for this bundle."""
        from .observations import ObservationAssembler

        return ObservationAssembler(self.manifest.observations, **kwargs)

    def action_decoder(self, **kwargs: Any) -> Any:
        """Build the :class:`ActionDecoder` for this bundle."""
        from .actions import ActionDecoder

        return ActionDecoder(self.manifest.actions, **kwargs)

    def describe(self) -> str:
        """Human-readable summary of what to wire up. Print this first."""
        layout = self.manifest.observations
        lines = [
            f"Bundle: {self.path}",
            f"  control rate: {self.manifest.control_hz:.1f} Hz (dt={self.manifest.dt})",
            (
                f"  observation vector: {layout.total_size} values "
                f"({layout.single_size} per tick x {layout.history_length} history)"
            ),
            "  inputs you supply each tick:",
        ]
        lines.extend(f"    - {entry.describe()}" for entry in layout.required_inputs)
        auto_filled = layout.pipeline_state_inputs
        if auto_filled:
            lines.append("  inputs filled automatically by the runtime:")
            lines.extend(f"    - {entry.describe()}" for entry in auto_filled)
        lines.append(f"  joint targets produced ({self.manifest.num_actions}):")
        for spec in sorted(self.manifest.actions, key=lambda item: item.slice_start):
            joints = ", ".join(spec.joint_names)
            lines.append(f"    - [{spec.deploy_type}] {joints}")
        if self.policy_path is not None:
            lines.append(f"  policy: {self.policy_path.name}")
        return "\n".join(lines)


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate just the manifest, without touching the rest of the bundle."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise MalformedBundleError(f"No manifest found at '{manifest_path}'.")
    return Manifest.from_json(manifest_path.read_text())


def load_bundle(path: str | Path, *, load_golden: bool = True) -> Bundle:
    """Load a deployment bundle directory.

    Args:
        path: The bundle directory written by ``genesis_forge.deployment.export``.
        load_golden: Read ``golden.npz`` when present. Set False to skip the
            recorded smoke-test samples.

    Returns:
        The loaded :class:`Bundle`.

    Raises:
        SchemaVersionError: The bundle was written by an incompatible version.
        MalformedBundleError: The bundle is missing something required.
    """
    bundle_path = Path(path)
    if not bundle_path.is_dir():
        raise MalformedBundleError(
            f"'{bundle_path}' is not a bundle directory. Expected a folder containing "
            f"{MANIFEST_FILENAME}."
        )

    manifest = load_manifest(bundle_path)

    golden: dict[str, np.ndarray] | None = None
    golden_path = bundle_path / GOLDEN_FILENAME
    if load_golden and golden_path.is_file():
        # allow_pickle stays False (the numpy default): a bundle travels between
        # machines, and object arrays would make loading one arbitrary-code execution.
        with np.load(golden_path, allow_pickle=False) as archive:
            golden = {key: archive[key] for key in archive.files}

    policy_file = manifest.policy.file if manifest.policy else None
    if policy_file is not None and not (bundle_path / policy_file).is_file():
        raise MalformedBundleError(
            f"Manifest references policy file '{policy_file}', but it is missing from "
            f"'{bundle_path}'."
        )

    return Bundle(manifest=manifest, path=bundle_path, golden=golden)


def save_bundle(
    bundle_path: str | Path,
    manifest: Manifest,
    *,
    golden: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a manifest (and optional golden samples) into a bundle directory.

    Used by the exporter on the training machine; kept here so the read and write
    sides of the schema cannot drift apart.
    """
    path = Path(bundle_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / MANIFEST_FILENAME).write_text(manifest.to_json() + "\n")
    if golden:
        np.savez(path / GOLDEN_FILENAME, **golden)
    return path


__all__ = [
    "GOLDEN_FILENAME",
    "HISTORY_NEWEST_FIRST",
    "MANIFEST_FILENAME",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "POLICY_FILENAME",
    "SCHEMA_VERSION",
    "SOURCE_PIPELINE_STATE",
    "SOURCE_SENSOR",
    "STAGE_PROCESSED_ACTIONS",
    "STAGE_RAW_ACTIONS",
    "ActionManagerSpec",
    "ActuatorSpec",
    "Bundle",
    "BundleError",
    "MalformedBundleError",
    "Manifest",
    "ObservationEntry",
    "ObservationLayout",
    "PolicySpec",
    "Provenance",
    "SchemaVersionError",
    "load_bundle",
    "load_manifest",
    "save_bundle",
]
