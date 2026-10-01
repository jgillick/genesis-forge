"""Processing one action manager's slice of the policy output.

A custom action manager ships a :class:`ActionManagerProcessor` subclass beside it and
names it in the manifest; the built-ins all share :class:`AffineProcessor`, which is
driven entirely by the parameters export recorded rather than by subclass
switching.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .action_schema import ActionManagerSpec
from .errors import ActionError


class ActionManagerProcessor:
    """Base class for one action manager's deployment-side process.

    A custom action manager supports deployment by shipping a subclass of this
    alongside it -- in a module that imports cleanly without torch or Genesis --
    and naming it in the export contract.

    Subclasses override :meth:`process`, and may keep per-step state as long as
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

    def process(self, actions: np.ndarray) -> np.ndarray:
        """Convert this manager's slice of the policy output into joint targets."""
        raise NotImplementedError(f"{type(self).__name__} must implement process().")

    @property
    def clip_range_by_joint(self) -> dict[str, tuple[float, float]]:
        """The clip this processor applies to each joint's target, where it clips.

        A joint appears only if its target is bounded on at least one side, and an
        unbounded side reads as an infinity. Processors that do not clip return
        nothing, which is why this is empty by default.
        """
        return {}


class AffineProcessor(ActionManagerProcessor):
    """Optional raw-action clip, ``actions * scale + offset``, then optional clip.

    This single shape covers both built-in managers:

    * ``position`` -- scale/offset from the manager's config, clipped to the
      joint limits.
    * ``position_within_limits`` -- the raw action clipped to [-1, 1], then
      mapped into the joint's limit range, with no clip on the result.

    Which of those applies is decided by the config the exporter recorded, not by
    the type name, so a future affine-ish manager needs no new runtime class.
    """

    def reset(self) -> None:
        config = self.spec.config
        self._joint_action_index = self._group_mapping(self.spec.joint_action_index)
        self._scale = self._vector(config.get("scale"), default=1.0)
        self._offset = self._vector(config.get("offset"), default=0.0)

        raw = config.get("raw_action_clip")
        self._raw_action_clip = (
            (float(raw[0]), float(raw[1])) if raw is not None else None
        )

        low = config.get("clip_low")
        high = config.get("clip_high")
        self._clip_low = self._clip_vector(low, unbounded=-np.inf)
        self._clip_high = self._clip_vector(high, unbounded=np.inf)

    def process(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=self.dtype).ravel()
        if values.size != self.spec.num_actions:
            raise ActionError(
                f"Action manager '{self.name}' expects {self.spec.num_actions} "
                f"action(s), got {values.size}."
            )

        if self._joint_action_index is not None:
            # Fan out before processing: every parameter below is per joint, and
            # grouped joints differ -- mirrored wheels share an action but take
            # opposite scale.
            values = values[self._joint_action_index]

        if self._raw_action_clip is not None:
            values = np.clip(values, *self._raw_action_clip)

        values = values * self._scale + self._offset

        if self._clip_low is not None or self._clip_high is not None:
            values = np.clip(values, self._clip_low, self._clip_high)

        return values.astype(self.dtype, copy=False)

    @property
    def clip_range_by_joint(self) -> dict[str, tuple[float, float]]:
        low, high = self._clip_low, self._clip_high
        if low is None and high is None:
            return {}
        return {
            name: (
                float(low[index]) if low is not None else -np.inf,
                float(high[index]) if high is not None else np.inf,
            )
            for index, name in enumerate(self.joint_names)
        }

    def _group_mapping(self, value: Any) -> np.ndarray | None:
        """Which action drives each joint, when the manager groups them."""
        if value is None:
            return None
        # Validated when the manifest is read; this only makes it indexable.
        return np.asarray(value, dtype=np.intp).ravel()

    def _clip_vector(self, value: Any, *, unbounded: float) -> np.ndarray | None:
        """One clip bound, or None when this side clips nothing.

        A null entry marks a single joint as unbounded on this side, which is how
        the exporter writes an infinity JSON cannot hold.
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            value = [unbounded if item is None else item for item in value]
        return self._vector(value, default=None)

    def _vector(self, value: Any, *, default: float | None) -> np.ndarray | None:
        """One process parameter, sized per joint rather than per action."""
        size = self.spec.num_joints
        if value is None:
            if default is None:
                return None
            return np.full(size, default, dtype=self.dtype)
        array = np.asarray(value, dtype=self.dtype).ravel()
        if array.size == 1:
            return np.full(size, array.item(), dtype=self.dtype)
        if array.size != size:
            raise ActionError(
                f"Action manager '{self.name}' has a processing parameter of length "
                f"{array.size}, but controls {size} joint(s)."
            )
        return array


#: Type names the runtime ships processors for. The *name* is the manifest's
#: contract, so it stays stable across refactors. Several map to one class: they
#: share an arithmetic shape and differ only in what their numbers mean.
BUILTIN_PROCESSORS: dict[str, type[ActionManagerProcessor]] = {
    "affine_dof": AffineProcessor,
    "position": AffineProcessor,
    "position_within_limits": AffineProcessor,
    "velocity": AffineProcessor,
}


def resolve_processor_class(spec: ActionManagerSpec) -> type[ActionManagerProcessor]:
    """Find the processor class for one action manager.

    Built-in type names resolve against :data:`BUILTIN_PROCESSORS`. Anything else
    resolves through the ``processor_import_path`` the exporter recorded, written
    as ``"module.path:ClassName"``.

    Raises:
        ActionError: The type is unknown and no import path was supplied, or the
            import path could not be loaded.
    """
    builtin = BUILTIN_PROCESSORS.get(spec.deploy_type)
    if builtin is not None:
        return builtin

    path = spec.processor_import_path
    if not path:
        raise ActionError(
            f"No processor available for action type '{spec.deploy_type}' (action "
            f"manager '{spec.name}'). Built-in types are: "
            f"{', '.join(sorted(BUILTIN_PROCESSORS))}. A custom action manager must "
            f"record its processor's import path when it exports, as "
            f"'my_package.processors:MyProcessor'."
        )

    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ActionError(
            f"Processor import path '{path}' for action manager '{spec.name}' is "
            f"malformed. Expected 'module.path:ClassName'."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ActionError(
            f"Could not import '{module_name}' to load the processor for action "
            f"manager '{spec.name}'. Install the package that provides it on this "
            f"machine. Original error: {error}"
        ) from error

    try:
        processor_class = getattr(module, class_name)
    except AttributeError as error:
        raise ActionError(
            f"Module '{module_name}' has no attribute '{class_name}' (processor for "
            f"action manager '{spec.name}')."
        ) from error

    if not (
        isinstance(processor_class, type)
        and issubclass(processor_class, ActionManagerProcessor)
    ):
        raise ActionError(
            f"Processor '{path}' for action manager '{spec.name}' must be a subclass "
            f"of ActionManagerProcessor."
        )
    return processor_class
