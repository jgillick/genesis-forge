"""Reading and writing a bundle directory.

A bundle is what export writes and the robot reads::

    my_policy/
      manifest.json   # the deployment contract, human readable
      golden.npz      # recorded input/output pairs for the on-robot smoke test
      policy.onnx     # optional: the exported policy

Nothing in this package imports torch or genesis -- that is the whole point of it,
and ``test_bundle.py`` asserts it in a clean subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import GOLDEN_FILENAME, MANIFEST_FILENAME
from .errors import MalformedBundleError
from .manifest import Manifest


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
            "  sensor values you supply each tick:",
        ]
        lines.extend(f"    - {entry.describe()}" for entry in layout.sensor_inputs)
        fed_back = layout.pipeline_state_inputs
        if fed_back:
            lines.append("  values you feed back from the decoder:")
            lines.extend(f"    - {entry.describe()}" for entry in fed_back)
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
