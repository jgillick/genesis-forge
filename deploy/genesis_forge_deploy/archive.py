"""Bundles stored as a single file.

A bundle is a directory. An *archive* is that same directory written as one zip,
which is what you copy to a robot -- one artifact instead of a folder, and a
transfer that either arrives whole or does not arrive.

Reading one extracts it beside the archive, into a dot-directory, and reuses that
on later loads. The extracted copy records a fingerprint of the archive it came
from, so replacing the archive re-extracts rather than leaving a robot quietly
running the bundle it had before.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from .constants import EXTRACT_MARKER, MANIFEST_FILENAME
from .errors import MalformedBundleError

#: Every zip begins with this. Read by content rather than by extension, so a
#: bundle someone renamed still loads.
_ZIP_MAGIC = b"PK\x03\x04"


def is_archive(path: Path) -> bool:
    """True when this file is a zip, whatever it happens to be called."""
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(4) == _ZIP_MAGIC


def fingerprint(archive: Path) -> str:
    """Identify an archive's contents without reading them.

    A zip's central directory already holds a CRC and a size for every entry, so
    this costs the same whether the bundle is a kilobyte or a gigabyte -- no
    payload is read. It is enough to notice that an archive has been replaced,
    which is what it is for; it is not a tamper check, and a bundle is trusted
    input either way.
    """
    with zipfile.ZipFile(archive) as zipped:
        entries = sorted(
            (item.filename, item.CRC, item.file_size) for item in zipped.infolist()
        )
    digest = hashlib.sha256()
    for name, crc, size in entries:
        digest.update(f"{name}\0{crc}\0{size}\0".encode())
    return digest.hexdigest()


def extract_dir_for(archive: Path) -> Path:
    """Where an archive's contents live once unpacked: ``.<name>`` beside it."""
    return archive.parent / f".{archive.stem}"


def ensure_extracted(archive: Path) -> Path:
    """Unpack an archive beside itself, reusing a previous extraction when valid.

    Returns:
        The directory holding the bundle's files.

    Raises:
        MalformedBundleError: The archive holds no manifest, the directory it
            would extract into belongs to something else, or there is nowhere
            writable to put it.
    """
    with zipfile.ZipFile(archive) as zipped:
        if MANIFEST_FILENAME not in zipped.namelist():
            raise MalformedBundleError(
                f"'{archive.name}' is a zip archive but holds no "
                f"{MANIFEST_FILENAME}, so it is not a deployment bundle."
            )

    destination = extract_dir_for(archive)
    marker = destination / EXTRACT_MARKER
    expected = fingerprint(archive)

    if destination.exists():
        if not destination.is_dir():
            raise MalformedBundleError(
                f"'{destination}' already exists and is not a directory, so "
                f"'{archive.name}' cannot be unpacked beside itself."
            )
        if not marker.is_file():
            raise MalformedBundleError(
                f"'{destination}' already exists but was not unpacked from an "
                f"archive, so it will not be overwritten. Move it aside, or unpack "
                f"'{archive.name}' yourself and load that directory instead."
            )
        if marker.read_text().strip() == expected:
            return destination
        _empty(destination)

    try:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            # extractall keeps every member inside the target directory, so a
            # crafted path cannot escape it.
            zipped.extractall(destination)
        marker.write_text(expected)
    except OSError as error:
        raise MalformedBundleError(
            f"Could not unpack '{archive.name}' into '{destination}' ({error}). "
            f"Unpack it yourself somewhere writable and load that directory."
        ) from error

    return destination


def _empty(directory: Path) -> None:
    """Clear a stale extraction, leaving the directory itself in place."""
    import shutil

    for item in directory.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
