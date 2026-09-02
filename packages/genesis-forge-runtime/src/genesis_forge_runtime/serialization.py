"""Turning manifest JSON into usable values, and back.

Numeric data is stored as plain JSON lists so a manifest stays readable and
diffable, but the runtime wants numpy arrays. These helpers convert between the
two, and report a missing field in terms a reader can act on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def decode_value(value: Any) -> Any:
    """Turn nested numeric lists into float32 arrays, leaving everything else alone.

    Action manager configs are opaque by design -- a custom manager defines its own
    schema -- so the rule is structural rather than key-based: anything that looks
    like a numeric array becomes one, so custom decoders get arrays for free.
    """
    if isinstance(value, dict):
        return {key: decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(is_number(item) for item in value):
            return np.asarray(value, dtype=np.float32)
        if value and all(
            isinstance(row, list) and all(is_number(item) for item in row)
            for row in value
        ):
            return np.asarray(value, dtype=np.float32)
        return [decode_value(item) for item in value]
    return value


def encode_value(value: Any) -> Any:
    """Inverse of :func:`_decode_value`, for writing JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    return value
