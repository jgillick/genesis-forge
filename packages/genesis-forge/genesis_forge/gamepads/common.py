from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Key:
    """Gamepad input abstraction shared across backends."""

    keytype: str
    index: int
    name: str | None = None
    value: float | None = None

    AXIS = "Axis"
    BUTTON = "Button"
    HAT = "Hat"


__all__ = ["Key"]
