"""Versions the exporter can measure for itself.

The first question anyone asks when a robot misbehaves is which export produced
the bundle it is running. Everything the exporter cannot see -- which checkpoint,
which framework -- depends on how you train, so you record it yourself: see
``additional_provenance`` on :func:`~genesis_forge.deployment.export`.
"""

from __future__ import annotations


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except (ImportError, PackageNotFoundError):  # pragma: no cover
        return None


def torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:  # pragma: no cover
        return None
