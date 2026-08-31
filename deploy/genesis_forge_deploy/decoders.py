"""Decoding one action manager's slice of the policy output.

A custom action manager ships a :class:`ManagerDecoder` subclass beside it and
names it in the manifest; the built-ins all share :class:`AffineDecoder`, which is
driven entirely by the parameters export recorded rather than by subclass
switching.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .action_schema import ActionManagerSpec
from .errors import DecoderError


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
