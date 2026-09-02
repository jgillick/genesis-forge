"""Recording where a bundle came from.

The first question anyone asks when a robot misbehaves is which export produced
the bundle it is running. The exporter measures what it can see for itself -- the
versions it ran under, and when. Everything else depends on how you train, so you
record it: see ``additional_provenance`` on :func:`~genesis_forge.deployment.export`.
"""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path
from typing import Any

from genesis_forge_runtime import Provenance

from .errors import ExportError


def build_provenance(*, additional: dict[str, Any] | None = None) -> Any:
    return Provenance(
        exported_at=_datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        genesis_forge_version=_package_version("genesis-forge"),
        torch_version=_torch_version(),
        additional=clean_additional(additional),
    )


def clean_additional(additional: dict[str, Any] | None) -> dict[str, Any]:
    """Check the developer's provenance entries survive the trip to JSON.

    Done before anything else runs, so a value that cannot be written fails
    immediately rather than after the parity gate has done its work.

    Paths are converted for you, since a checkpoint path is the common case and
    the conversion is lossless. Anything else has to be JSON-friendly already --
    silently stringifying a tensor would record something worse than nothing.
    """
    if not additional:
        return {}
    if not isinstance(additional, dict):
        raise ExportError(
            f"additional_provenance must be a dict, got {type(additional).__name__}."
        )

    cleaned: dict[str, Any] = {}
    for key, value in additional.items():
        if not isinstance(key, str):
            raise ExportError(
                f"additional_provenance keys must be strings, got "
                f"{type(key).__name__} ({key!r})."
            )
        if isinstance(value, Path):
            value = str(value)
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ExportError(
                f"additional_provenance['{key}'] is a {type(value).__name__}, which "
                f"cannot be written to the manifest ({error}). Convert it to a "
                f"string, number, bool, or a list/dict of those."
            ) from error
        cleaned[key] = value
    return cleaned


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except (ImportError, PackageNotFoundError):  # pragma: no cover
        return None


def _torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:  # pragma: no cover
        return None
