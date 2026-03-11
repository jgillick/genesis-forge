"""
genesis_forge.utils.transforms
================================

Genesis-free, torch-only utilities shared between training-time managers and
deployment-time classes.  Nothing in this module may import Genesis.

Contents
--------
- :func:`position_decode` -- canonical transform for ``PositionActionManager``
- :func:`position_within_limits_decode` -- canonical transform for ``PositionWithinLimitsActionManager``
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Action transforms
# ---------------------------------------------------------------------------


def position_decode(
    actions: torch.Tensor,
    scale,
    offset,
    clip_low,
    clip_high,
) -> torch.Tensor:
    """
    Canonical affine + clamp transform for ``PositionActionManager``.

    Computes ``clamp(actions * scale + offset, clip_low, clip_high)``.

    Called by both :meth:`~genesis_forge.managers.PositionActionManager.process_actions`
    (with live pre-computed tensors) and the ``"position"`` decoder registered in
    :mod:`genesis_forge.deploy.action_decoder` (with tensors built from a JSON params
    dict).

    ``torch.as_tensor`` is used for each parameter so the function accepts either an
    existing :class:`torch.Tensor` (zero-copy when dtype already matches) or a plain
    Python list loaded from JSON.

    Args:
        actions:  Input action tensor, shape ``(..., n_joints)``.
        scale:    Per-joint scale. Tensor or list of floats.
        offset:   Per-joint offset. Tensor or list of floats.
        clip_low: Per-joint lower position limit. Tensor or list of floats.
        clip_high: Per-joint upper position limit. Tensor or list of floats.

    Returns:
        Decoded position tensor, same shape as ``actions``.
    """
    scale = torch.as_tensor(scale, dtype=torch.float32)
    offset = torch.as_tensor(offset, dtype=torch.float32)
    clip_low = torch.as_tensor(clip_low, dtype=torch.float32)
    clip_high = torch.as_tensor(clip_high, dtype=torch.float32)
    return torch.clamp(actions * scale + offset, clip_low, clip_high)


def position_within_limits_decode(
    actions: torch.Tensor,
    scale,
    offset,
) -> torch.Tensor:
    """
    Canonical clamp-then-affine transform for ``PositionWithinLimitsActionManager``.

    Computes ``clamp(actions, -1, 1) * scale + offset``.

    Called by both :meth:`~genesis_forge.managers.PositionWithinLimitsActionManager.process_actions`
    and the ``"position_within_limits"`` decoder registered in
    :mod:`genesis_forge.deploy.action_decoder`.

    Args:
        actions: Input action tensor, shape ``(..., n_joints)``.  Values outside
                 ``[-1, 1]`` are clamped before mapping.
        scale:   Per-joint scale ``(upper - lower) / 2 * soft_limit_scale_factor``.
                 Tensor or list of floats.
        offset:  Per-joint midpoint ``(upper + lower) / 2``.
                 Tensor or list of floats.

    Returns:
        Decoded position tensor, same shape as ``actions``.
    """
    scale = torch.as_tensor(scale, dtype=torch.float32)
    offset = torch.as_tensor(offset, dtype=torch.float32)
    return torch.clamp(actions, -1.0, 1.0) * scale + offset
