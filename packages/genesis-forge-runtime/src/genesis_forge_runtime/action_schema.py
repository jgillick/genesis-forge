"""How the policy's output maps onto real joints.

The half of the manifest that :mod:`genesis_forge_runtime.decoders` consumes: which
slice of the policy vector belongs to each action manager, how to decode it, and
the actuator settings the robot should match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import MalformedBundleError
from .serialization import decode_value, encode_value, require


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
        name = require(data, "name", where=where)
        scope = f"{where}.{name}"
        bounds = require(data, "slice", where=scope)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise MalformedBundleError(
                f"Action manager '{name}' has a malformed 'slice': expected [start, end], "
                f"got {bounds!r}."
            )
        joint_names = tuple(require(data, "joint_names", where=scope))
        spec = cls(
            name=name,
            deploy_type=require(data, "deploy_type", where=scope),
            joint_names=joint_names,
            slice_start=int(bounds[0]),
            slice_end=int(bounds[1]),
            config=decode_value(data.get("config", {})),
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
            "config": encode_value(self.config),
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
        name = require(data, "name", where=where)
        scope = f"{where}.{name}"
        return cls(
            name=name,
            joint_names=tuple(require(data, "joint_names", where=scope)),
            values=decode_value(data.get("values", {})),
            randomized=tuple(data.get("randomized", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "joint_names": list(self.joint_names),
            "values": encode_value(self.values),
        }
        if self.randomized:
            data["randomized"] = list(self.randomized)
        return data
