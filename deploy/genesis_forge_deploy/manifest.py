"""The deployment contract as a whole.

Ties the observation and action halves together with the control rate, the policy
description, and provenance, and validates that the pieces are mutually consistent
before any of it reaches a robot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .action_schema import ActionManagerSpec, ActuatorSpec
from .constants import MIN_SUPPORTED_SCHEMA_VERSION, SCHEMA_VERSION
from .errors import MalformedBundleError, SchemaVersionError
from .observation_schema import ObservationLayout
from .serialization import decode_value, encode_value, require


@dataclass(frozen=True)
class PolicySpec:
    """Where the exported policy lives, what format it is, and what its output means.

    The bundle records the format rather than requiring one: ONNX is the documented
    path, but a TorchScript file (or anything else you load yourself) is equally
    welcome -- the runtime never loads the policy for you.
    """

    file: str | None = None
    format: str | None = None
    input_name: str = "obs"
    output_name: str = "actions"
    output_semantics: str = "raw"
    normalizer: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySpec:
        return cls(
            file=data.get("file"),
            format=data.get("format"),
            input_name=data.get("input_name", "obs"),
            output_name=data.get("output_name", "actions"),
            output_semantics=data.get("output_semantics", "raw"),
            normalizer=decode_value(data["normalizer"])
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
        if self.format is not None:
            data["format"] = self.format
        if self.normalizer is not None:
            data["normalizer"] = encode_value(self.normalizer)
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
        version = require(data, "schema_version", where="manifest")
        _check_schema_version(version)

        control = require(data, "control", where="manifest")
        dt = float(require(control, "dt", where="control"))
        if dt <= 0:
            raise MalformedBundleError(f"'control.dt' must be positive, got {dt}.")

        raw_actions = require(data, "actions", where="manifest")
        actions = tuple(
            ActionManagerSpec.from_dict(item, where="actions")
            for item in require(raw_actions, "managers", where="actions")
        )
        _check_action_slices(actions)

        return cls(
            schema_version=int(version),
            dt=dt,
            observations=ObservationLayout.from_dict(
                require(data, "observations", where="manifest")
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
