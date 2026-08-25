"""Turn raw policy output into named joint targets.

Each action manager on the training side describes its own decode as plain data
(see ``BaseActionManager.get_deployment_config``); this module replays that data
without importing torch. Both built-in position managers reduce to the same
shape -- optional pre-clip, affine transform, optional post-clip -- so the
built-in decoder is driven entirely by config rather than by subclass switches.

Custom action managers ship their own decoder by subclassing :class:`ManagerDecoder`
and naming it in the manifest; see :func:`resolve_decoder_class`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .bundle import ActionManagerSpec


class DecoderError(Exception):
    """A decoder could not be resolved, or was handed unusable policy output."""


class ManagerDecoder:
    """Base class for one action manager's deployment-side decode.

    A custom action manager supports deployment by shipping a subclass of this
    alongside it -- in a module that imports cleanly without torch or Genesis --
    and naming it in the export contract.

    Subclasses override :meth:`decode`, and may keep per-step state as long as
    :meth:`reset` clears it.
    """

    def __init__(self, spec: ActionManagerSpec, *, dtype: Any = np.float32) -> None:
        self.spec = spec
        self.dtype = np.dtype(dtype)
        self.reset()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.spec.joint_names

    def reset(self) -> None:
        """Clear any per-step state. Called on construction and by the composer."""

    def decode(self, actions: np.ndarray) -> np.ndarray:
        """Convert this manager's slice of the policy output into joint targets."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode()."
        )


class AffineDecoder(ManagerDecoder):
    """Optional pre-clip, then ``actions * scale + offset``, then optional post-clip.

    This single shape covers both built-in managers:

    * ``position`` -- scale/offset from the manager's config, post-clipped to the
      joint limits.
    * ``position_within_limits`` -- pre-clipped to [-1, 1], then mapped into the
      joint's limit range, with no post-clip.

    Which of those applies is decided by the config the exporter recorded, not by
    the type name, so a future affine-ish manager needs no new runtime class.
    """

    def reset(self) -> None:
        config = self.spec.config
        self._scale = self._vector(config.get("scale"), default=1.0)
        self._offset = self._vector(config.get("offset"), default=0.0)

        pre_clip = config.get("pre_clip")
        self._pre_clip = (
            (float(pre_clip[0]), float(pre_clip[1])) if pre_clip is not None else None
        )

        low = config.get("post_clip_low")
        high = config.get("post_clip_high")
        self._post_clip_low = self._vector(low, default=None) if low is not None else None
        self._post_clip_high = (
            self._vector(high, default=None) if high is not None else None
        )

    def decode(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=self.dtype).ravel()
        if values.size != self.spec.num_actions:
            raise DecoderError(
                f"Action manager '{self.name}' expects {self.spec.num_actions} "
                f"action(s), got {values.size}."
            )

        if self._pre_clip is not None:
            values = np.clip(values, *self._pre_clip)

        values = values * self._scale + self._offset

        if self._post_clip_low is not None or self._post_clip_high is not None:
            values = np.clip(values, self._post_clip_low, self._post_clip_high)

        return values.astype(self.dtype, copy=False)

    def _vector(self, value: Any, *, default: float | None) -> np.ndarray | None:
        if value is None:
            if default is None:
                return None
            return np.full(self.spec.num_actions, default, dtype=self.dtype)
        array = np.asarray(value, dtype=self.dtype).ravel()
        if array.size == 1:
            return np.full(self.spec.num_actions, array.item(), dtype=self.dtype)
        if array.size != self.spec.num_actions:
            raise DecoderError(
                f"Action manager '{self.name}' has a decode parameter of length "
                f"{array.size}, but controls {self.spec.num_actions} joint(s)."
            )
        return array


#: Type names the runtime ships decoders for. The *name* is the manifest's
#: contract -- module layout can be refactored without invalidating bundles.
#:
#: Every built-in manager decodes with the same affine shape and differs only in
#: what its numbers mean, which is why one decoder class serves them all. The names
#: are kept distinct anyway: a robot operator needs to know whether a target is a
#: joint position or a wheel velocity, since those go to different motor commands.
BUILTIN_DECODERS: dict[str, type[ManagerDecoder]] = {
    "affine_dof": AffineDecoder,
    "position": AffineDecoder,
    "position_within_limits": AffineDecoder,
    "velocity": AffineDecoder,
}


def resolve_decoder_class(spec: ActionManagerSpec) -> type[ManagerDecoder]:
    """Find the decoder class for one action manager.

    Built-in type names resolve against :data:`BUILTIN_DECODERS`. Anything else
    resolves through the ``decoder_import_path`` the exporter recorded, written
    as ``"module.path:ClassName"``.

    Raises:
        DecoderError: The type is unknown and no import path was supplied, or the
            import path could not be loaded.
    """
    builtin = BUILTIN_DECODERS.get(spec.deploy_type)
    if builtin is not None:
        return builtin

    path = spec.decoder_import_path
    if not path:
        raise DecoderError(
            f"No decoder available for action type '{spec.deploy_type}' (action "
            f"manager '{spec.name}'). Built-in types are: "
            f"{', '.join(sorted(BUILTIN_DECODERS))}. A custom action manager must "
            f"record its decoder's import path when it exports, as "
            f"'my_package.decoders:MyDecoder'."
        )

    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise DecoderError(
            f"Decoder import path '{path}' for action manager '{spec.name}' is "
            f"malformed. Expected 'module.path:ClassName'."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise DecoderError(
            f"Could not import '{module_name}' to load the decoder for action "
            f"manager '{spec.name}'. Install the package that provides it on this "
            f"machine. Original error: {error}"
        ) from error

    try:
        decoder_class = getattr(module, class_name)
    except AttributeError as error:
        raise DecoderError(
            f"Module '{module_name}' has no attribute '{class_name}' (decoder for "
            f"action manager '{spec.name}')."
        ) from error

    if not (isinstance(decoder_class, type) and issubclass(decoder_class, ManagerDecoder)):
        raise DecoderError(
            f"Decoder '{path}' for action manager '{spec.name}' must be a subclass "
            f"of ManagerDecoder."
        )
    return decoder_class


class DecodedActions:
    """The result of one decode, viewable per joint, per manager, or as a vector."""

    __slots__ = ("_joint_names", "by_manager", "targets")

    def __init__(
        self,
        targets: np.ndarray,
        by_manager: dict[str, np.ndarray],
        joint_names: tuple[str, ...],
    ) -> None:
        self.targets = targets
        self.by_manager = by_manager
        self._joint_names = joint_names

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def by_joint(self) -> dict[str, float]:
        """Joint name to target value -- what you hand to the motor controllers."""
        return {
            name: float(value)
            for name, value in zip(self._joint_names, self.targets, strict=True)
        }

    def __len__(self) -> int:
        return int(self.targets.size)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DecodedActions({self.by_joint})"


class ActionDecoder:
    """Decodes a full policy output vector across every action manager.

    Args:
        specs: The action manager specs from a loaded bundle's manifest.
        apply_delay: Reproduce each manager's trained ``delay_step`` on-robot.
            Off by default -- the delay models physical actuation latency that
            real hardware already has, so replaying it would stack a second
            delay on top. Turn it on for debugging, or when the real control
            loop is faster than the latency the policy was trained against.
        check_finite: Raise when the policy emits NaN or infinity. On by default:
            motors must never be handed those values.
        dtype: Output dtype, float32 by default.
    """

    def __init__(
        self,
        specs: tuple[ActionManagerSpec, ...],
        *,
        apply_delay: bool = False,
        check_finite: bool = True,
        dtype: Any = np.float32,
    ) -> None:
        self._specs = tuple(sorted(specs, key=lambda spec: spec.slice_start))
        self._apply_delay = apply_delay
        self._check_finite = check_finite
        self._dtype = np.dtype(dtype)

        self._decoders: list[ManagerDecoder] = [
            resolve_decoder_class(spec)(spec, dtype=self._dtype) for spec in self._specs
        ]
        self._joint_names = tuple(
            name for spec in self._specs for name in spec.joint_names
        )
        self._delay_buffers: dict[str, list[np.ndarray]] = {}
        self.reset()

    """
    Properties
    """

    @property
    def num_actions(self) -> int:
        """Width of the policy output vector this decoder consumes."""
        return self._specs[-1].slice_end if self._specs else 0

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Every joint this decoder produces a target for, in output order."""
        return self._joint_names

    @property
    def decoders(self) -> tuple[ManagerDecoder, ...]:
        return tuple(self._decoders)

    @property
    def trained_delay_steps(self) -> dict[str, int]:
        """Each manager's trained ``delay_step``, whether or not it is applied."""
        return {spec.name: spec.delay_step for spec in self._specs}

    """
    Public methods
    """

    def reset(self) -> None:
        """Clear decoder state and any delay buffers."""
        for decoder in self._decoders:
            decoder.reset()
        self._delay_buffers = {
            spec.name: [
                np.zeros(spec.num_actions, dtype=self._dtype)
                for _ in range(spec.delay_step)
            ]
            for spec in self._specs
            if self._apply_delay and spec.delay_step > 0
        }

    def decode(self, actions: Any) -> DecodedActions:
        """Decode one policy output vector into joint targets.

        Args:
            actions: The policy's raw output, length :attr:`num_actions`. A
                leading batch dimension of 1 is accepted and squeezed.

        Returns:
            A :class:`DecodedActions` view of the result.

        Raises:
            DecoderError: Wrong length, or non-finite values when
                ``check_finite`` is on.
        """
        values = np.asarray(actions, dtype=self._dtype).ravel()
        if values.size != self.num_actions:
            raise DecoderError(
                f"Expected {self.num_actions} action(s) from the policy, got "
                f"{values.size}."
            )
        if self._check_finite and not np.all(np.isfinite(values)):
            bad = np.flatnonzero(~np.isfinite(values))
            raise DecoderError(
                f"Policy produced non-finite action(s) at index/indices "
                f"{bad.tolist()}: {values[bad].tolist()}. Refusing to send these to "
                f"the actuators. Pass check_finite=False to override for debugging."
            )

        by_manager: dict[str, np.ndarray] = {}
        pieces: list[np.ndarray] = []
        for spec, decoder in zip(self._specs, self._decoders, strict=True):
            chunk = values[spec.slice_start : spec.slice_end]
            chunk = self._apply_delay_to(spec, chunk)
            decoded = decoder.decode(chunk)
            by_manager[spec.name] = decoded
            pieces.append(decoded)

        targets = (
            pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        ).astype(self._dtype, copy=False)
        return DecodedActions(targets, by_manager, self._joint_names)

    def describe_outputs(self) -> str:
        """Human-readable listing of the joint targets this decoder produces."""
        lines = [f"Joint targets produced ({self.num_actions}):"]
        for spec in self._specs:
            joints = ", ".join(spec.joint_names)
            note = ""
            if spec.delay_step:
                applied = "applied" if self._apply_delay else "recorded, not applied"
                note = f" [delay_step={spec.delay_step}, {applied}]"
            lines.append(f"  - [{spec.deploy_type}] {joints}{note}")
        return "\n".join(lines)

    """
    Internal methods
    """

    def _apply_delay_to(self, spec: ActionManagerSpec, chunk: np.ndarray) -> np.ndarray:
        buffer = self._delay_buffers.get(spec.name)
        if not buffer:
            return chunk
        # Same FIFO as BaseActionManager.step: push the newest, pop the oldest.
        buffer.insert(0, chunk.copy())
        return buffer.pop()


__all__ = [
    "BUILTIN_DECODERS",
    "ActionDecoder",
    "AffineDecoder",
    "DecodedActions",
    "DecoderError",
    "ManagerDecoder",
    "resolve_decoder_class",
]
