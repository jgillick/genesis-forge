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

import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionDecoder
from .archive import ensure_extracted, is_archive
from .constants import GOLDEN_FILENAME, MANIFEST_FILENAME
from .errors import MalformedBundleError
from .manifest import Manifest
from .observations import ObservationAssembler


@dataclass(frozen=True)
class Bundle:
    """A deployment bundle: the manifest, and wherever its files are.

    ``path`` is a directory when the bundle is unpacked, and an archive file when
    it is not -- which is what :func:`~genesis_forge.deployment.export` hands back
    after writing one. The manifest and the golden samples are held in memory
    either way, so describing a bundle never unpacks it.
    """

    manifest: Manifest
    path: Path
    golden: dict[str, np.ndarray] | None = None

    @property
    def is_archive(self) -> bool:
        """True when the files are still packed into a single file."""
        return not self.path.is_dir()

    @property
    def policy_file(self) -> str | None:
        """Name of the exported policy inside the bundle, if it carries one."""
        if self.manifest.policy is None:
            return None
        return self.manifest.policy.file

    @property
    def policy_path(self) -> Path | None:
        """Absolute path to the exported policy, when the bundle carries one.

        Raises:
            MalformedBundleError: The bundle is still an archive, so its files
                are not on disk. Use :meth:`unpacked` to get at them, or load the
                archive with ``load_bundle`` for a directory that persists.
        """
        if self.policy_file is None:
            return None
        if self.is_archive:
            raise MalformedBundleError(
                f"'{self.path.name}' is an archive, so its policy is not a file on "
                f"disk yet. Use `with bundle.unpacked() as directory:` to work with "
                f"the contents, or load_bundle() to unpack it beside itself."
            )
        return self.path / self.policy_file

    @contextmanager
    def unpacked(self) -> Iterator[Path]:
        """Yield a directory holding this bundle's files.

        Already-unpacked bundles yield their own directory and nothing is copied.
        An archive is unpacked to a temporary directory that is removed on the way
        out, which is what you want on a training machine -- checking what you just
        exported should not leave anything behind.
        """
        if not self.is_archive:
            yield self.path
            return
        with tempfile.TemporaryDirectory(prefix="genesis-forge-bundle-") as scratch:
            with zipfile.ZipFile(self.path) as archive:
                archive.extractall(scratch)
            yield Path(scratch)

    def create_observation_assembler(self, **kwargs: Any) -> ObservationAssembler:
        """Build a new :class:`ObservationAssembler` for this bundle.

        Each call returns a fresh assembler with its own zero-filled history, so
        create one and keep it for the life of the control loop -- a second one
        starts with no history at all.
        """
        return ObservationAssembler(self.manifest.observations, **kwargs)

    def create_action_decoder(self, **kwargs: Any) -> ActionDecoder:
        """Build a new :class:`ActionDecoder` for this bundle.

        Each call returns a fresh decoder with its own remembered actions and
        delay buffers, so create one and keep it for the life of the control loop.
        """
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
            "  values you supply each tick:",
        ]
        lines.extend(f"    - {entry.describe()}" for entry in layout.entries)
        lines.append(f"  joint targets produced ({self.manifest.num_actions}):")
        for spec in sorted(self.manifest.actions, key=lambda item: item.slice_start):
            joints = ", ".join(spec.joint_names)
            lines.append(f"    - [{spec.deploy_type}] {joints}")
        if self.policy_file is not None:
            policy_format = self.manifest.policy.format or "unknown format"
            lines.append(f"  policy: {self.policy_file} ({policy_format})")
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
    """Load a deployment bundle.

    Args:
        path: A bundle directory, or an archive holding one. An archive is
            unpacked beside itself into ``.<name>/`` and reused on later loads,
            so the files stay in one predictable place for as long as anything
            needs them. Replacing the archive unpacks it again.
        load_golden: Read ``golden.npz`` when present. Set False to skip the
            recorded smoke-test samples.

    Returns:
        The loaded :class:`Bundle`. Its :attr:`~Bundle.path` is the directory the
        files were read from, which for an archive is where they were unpacked.

    Raises:
        SchemaVersionError: The bundle was written by an incompatible version.
        MalformedBundleError: The bundle is missing something required, or an
            archive could not be unpacked.
    """
    bundle_path = Path(path)
    if is_archive(bundle_path):
        bundle_path = ensure_extracted(bundle_path)
    if not bundle_path.is_dir():
        raise MalformedBundleError(
            f"'{bundle_path}' is not a bundle directory or archive. Expected a "
            f"folder containing {MANIFEST_FILENAME}, or a single-file bundle."
        )

    manifest = load_manifest(bundle_path)

    golden: dict[str, np.ndarray] | None = None
    golden_path = bundle_path / GOLDEN_FILENAME
    if load_golden and golden_path.is_file():
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
