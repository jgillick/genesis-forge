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
from .serialization import require


def _read_action_index(value: Any, name: str) -> tuple[int, ...] | None:
    """Read the joint-to-action mapping as integers."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise MalformedBundleError(
            f"Action manager '{name}' has a 'joint_action_index' that is not a list."
        )
    try:
        return tuple(int(index) for index in value)
    except (TypeError, ValueError) as error:
        raise MalformedBundleError(
            f"Action manager '{name}' has a non-integer entry in 'joint_action_index'."
        ) from error


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
    #: One action index per joint, positionally matched to :attr:`joint_names`.
    #: None when every joint has its own action.
    joint_action_index: tuple[int, ...] | None = None

    @property
    def num_actions(self) -> int:
        """How many policy outputs this manager consumes."""
        return self.slice_end - self.slice_start

    @property
    def num_joints(self) -> int:
        """How many joint targets it produces.

        Larger than :attr:`num_actions` when joints share an action.
        """
        return len(self.joint_names)

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
            config=data.get("config", {}),
            decoder_import_path=data.get("decoder_import_path"),
            joint_action_index=_read_action_index(data.get("joint_action_index"), name),
        )
        grouped = spec.joint_action_index is not None
        if grouped and len(spec.joint_action_index) != len(joint_names):
            raise MalformedBundleError(
                f"Action manager '{name}' maps "
                f"{len(spec.joint_action_index)} joint(s) to actions but names "
                f"{len(joint_names)}."
            )
        if grouped and any(
            index < 0 or index >= spec.num_actions for index in spec.joint_action_index
        ):
            raise MalformedBundleError(
                f"Action manager '{name}' maps a joint to an action outside its "
                f"slice of {spec.num_actions} action(s)."
            )
        if not grouped and spec.num_actions != len(joint_names):
            raise MalformedBundleError(
                f"Action manager '{name}' covers {spec.num_actions} actions but names "
                f"{len(joint_names)} joints, and records no mapping between them; "
                f"the bundle is inconsistent."
            )
        if grouped and len(joint_names) < spec.num_actions:
            raise MalformedBundleError(
                f"Action manager '{name}' groups {len(joint_names)} joints across "
                f"{spec.num_actions} actions, so at least one action drives nothing; "
                f"the bundle is inconsistent."
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "deploy_type": self.deploy_type,
            "slice": [self.slice_start, self.slice_end],
            "joint_names": list(self.joint_names),
            "config": self.config,
        }
        if self.decoder_import_path is not None:
            data["decoder_import_path"] = self.decoder_import_path
        if self.joint_action_index is not None:
            data["joint_action_index"] = [
                int(index) for index in self.joint_action_index
            ]
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
            values=data.get("values", {}),
            randomized=tuple(data.get("randomized", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "joint_names": list(self.joint_names),
            "values": self.values,
        }
        if self.randomized:
            data["randomized"] = list(self.randomized)
        return data
