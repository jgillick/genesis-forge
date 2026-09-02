"""Turning a full policy output vector into named joint targets.

Slices the vector across every action manager, hands each slice to its decoder,
and remembers the result so the caller can feed it back into the next observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .action_schema import ActionManagerSpec
from .decoders import ManagerDecoder, resolve_decoder_class
from .errors import DecoderError


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
    def last_raw_actions(self) -> np.ndarray:
        """The policy's raw output from the previous decode.

        Zeros before the first decode, matching how training starts an episode.
        Pass this to :meth:`ObservationAssembler.assemble` for an observation entry
        that echoes raw policy output.
        """
        return self._last_raw_actions.copy()

    @property
    def last_target_actions(self) -> np.ndarray:
        """The decoded joint targets from the previous decode.

        Zeros before the first decode.

        Note this is *not* what the built-in ``current_actions`` observation feeds
        back -- that one is raw policy output at both of its call shapes, so use
        :attr:`last_raw_actions` or :attr:`last_raw_actions_by_manager`. This
        property is for an observation that genuinely echoes decoded targets,
        which you would have written yourself.
        """
        return self._last_target_actions.copy()

    @property
    def last_target_actions_by_manager(self) -> dict[str, np.ndarray]:
        """Per-manager view of :attr:`last_target_actions`, for multi-manager robots.

        As with :attr:`last_target_actions`, this is decoded output -- not what
        ``current_actions`` feeds back.
        """
        return {name: values.copy() for name, values in self._last_by_manager.items()}

    @property
    def last_raw_actions_by_manager(self) -> dict[str, np.ndarray]:
        """What each manager last consumed, before its own decode.

        This is what ``current_actions(action_manager=...)`` feeds back during
        training, and it is the whole policy vector only when a single manager is
        registered.

        With ``apply_delay`` on and a manager trained with a ``delay_step``, this
        is the delayed value -- which is what training saw, because a manager
        records its raw actions after taking them off its delay buffer. With the
        delay left off, it is the value you just passed in, and a policy trained
        on delayed feedback will see something training never showed it. See
        :attr:`trained_delay_steps`.
        """
        return {name: value.copy() for name, value in self._last_raw_by_manager.items()}

    @property
    def trained_delay_steps(self) -> dict[str, int]:
        """Each manager's trained ``delay_step``, whether or not it is applied."""
        return {spec.name: spec.delay_step for spec in self._specs}

    """
    Public methods
    """

    def reset(self) -> None:
        """Clear decoder state, remembered outputs, and any delay buffers."""
        for decoder in self._decoders:
            decoder.reset()
        self._last_raw_actions = np.zeros(self.num_actions, dtype=self._dtype)
        self._last_target_actions = np.zeros(self.num_actions, dtype=self._dtype)
        self._last_by_manager = {
            spec.name: np.zeros(spec.num_actions, dtype=self._dtype)
            for spec in self._specs
        }
        self._last_raw_by_manager = {
            spec.name: np.zeros(spec.num_actions, dtype=self._dtype)
            for spec in self._specs
        }
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
        raw_by_manager: dict[str, np.ndarray] = {}
        pieces: list[np.ndarray] = []
        for spec, decoder in zip(self._specs, self._decoders, strict=True):
            chunk = values[spec.slice_start : spec.slice_end]
            # After the delay, not before: this is what the manager consumed, and
            # what training's own raw_actions held for it. Identical to the slice
            # when no delay is applied.
            chunk = self._apply_delay_to(spec, chunk)
            raw_by_manager[spec.name] = chunk.copy()
            decoded = decoder.decode(chunk)
            by_manager[spec.name] = decoded
            pieces.append(decoded)

        # copy() on the single-manager path: concatenate already returns a fresh
        # array, and without it `targets` and `by_manager[name]` would be the same
        # buffer, so writing through one would silently change the other.
        targets = (
            pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)
        ).astype(self._dtype, copy=False)

        self._last_raw_actions = values.copy()
        self._last_target_actions = targets.copy()
        self._last_raw_by_manager = raw_by_manager
        self._last_by_manager = {
            name: chunk.copy() for name, chunk in by_manager.items()
        }

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
