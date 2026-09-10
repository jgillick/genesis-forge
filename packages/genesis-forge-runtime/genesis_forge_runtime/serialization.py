"""Reading required fields out of manifest JSON.

Numeric data stays as plain JSON lists, both on disk and once loaded -- a processor
converts what it needs with ``np.asarray``. Nothing here reshapes values behind
your back; it only reports a missing field in terms a reader can act on.
"""

from __future__ import annotations

from typing import Any

from .errors import MalformedBundleError


def require(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    """Fetch ``key`` or raise an error that names both the key and its section."""
    if not isinstance(mapping, dict):
        raise MalformedBundleError(
            f"Expected '{where}' to be a JSON object, got {type(mapping).__name__}."
        )
    if key not in mapping:
        available = ", ".join(sorted(mapping)) or "nothing"
        raise MalformedBundleError(
            f"Missing required field '{key}' in '{where}' (found: {available})."
        )
    return mapping[key]
