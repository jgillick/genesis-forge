"""Recording where a bundle came from.

The first question anyone asks when a robot misbehaves is which export produced
the bundle it is running, so every bundle carries the versions and the checkpoint
it was built from.
"""

from __future__ import annotations

import datetime as _datetime
from typing import Any


def build_provenance(*, checkpoint: str | None, policy: Any) -> Any:
    from genesis_forge_deploy import Provenance

    return Provenance(
        exported_at=_datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        genesis_forge_version=_package_version("genesis-forge"),
        torch_version=_torch_version(),
        policy_framework=_policy_framework(policy),
        policy_framework_version=_policy_framework_version(policy),
        checkpoint=str(checkpoint) if checkpoint else None,
    )


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


def _policy_framework(policy: Any) -> str | None:
    if policy is None:
        return None
    module = type(policy).__module__ or ""
    return module.split(".")[0] or None


def _policy_framework_version(policy: Any) -> str | None:
    framework = _policy_framework(policy)
    if not framework:
        return None
    return _package_version(framework.replace("_", "-")) or _package_version(framework)
