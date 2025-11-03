from __future__ import annotations
from typing import Any


def ensure_dof_pattern(value: dict[str, Any] | Any) -> dict[str, Any] | None:
    """
    Ensures the value is a dictionary in the form: {<joint name or regex>: <value>}.

    Example:
        >>> ensure_dof_pattern(50)
        {".*": 50}
        >>> ensure_dof_pattern({".*": 50})
        {".*": 50}
        >>> ensure_dof_pattern({"knee_joint": 50})
        {"knee_joint": 50}

    Args:
        value: The value to convert.

    Returns:
        A dictionary of DOF name pattern to value.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {".*": value}
