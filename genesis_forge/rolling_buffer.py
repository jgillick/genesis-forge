"""
genesis_forge.utils.rolling_buffer
====================================

Provides :class:`RollingBuffer`, a fixed-capacity rolling window of
pre-allocated tensor slots with no Genesis dependency.
"""

from __future__ import annotations

import torch


class RollingBuffer:
    """
    Fixed-capacity rolling window of pre-allocated tensor slots.

    Used by :class:`~genesis_forge.managers.ObservationManager` (batched,
    ``(num_envs, frame_size)`` frames) and
    :class:`~genesis_forge.deploy.ObservationBuilder` (unbatched, ``(frame_size,)``
    frames) to maintain an observation history without per-step allocation.

    The design intentionally exposes the underlying slot for **in-place** writing via
    :meth:`rotate` + :meth:`push`, which avoids creating a temporary tensor on
    every step.  :meth:`output` concatenates all frames along the **last** axis using
    ``out[..., offset:offset+n]``, making it shape-agnostic -- it works identically
    for 1-D and 2-D frames.

    Example (single-env deployment)::

        buf = RollingBuffer(history_len, torch.zeros(obs_size))

        # Each step:
        slot = buf.rotate()           # get empty slot (oldest frame, zeroed)
        slot[...] = new_obs           # fill in-place
        buf.push(slot)                # push to front
        buf.output(output_tensor)     # write history into flat output

    Example (multi-env training, 2-D frames)::

        buf = RollingBuffer(history_len,
                            torch.zeros(num_envs, obs_size, device=gs.device))

        slot = buf.rotate()
        perform_observation(slot)     # fills (num_envs, obs_size) in-place
        buf.push(slot)
        buf.output(output_tensor)     # output shape: (num_envs, obs_size * history_len)

    Args:
        capacity: Number of frames to keep (``history_len``).
        frame:    A representative tensor whose shape and device define each slot.
                  The content of ``frame`` is ignored; all slots are initialised to zero.
    """

    def __init__(self, capacity: int, frame: torch.Tensor):
        self._capacity = capacity
        self._frames: list[torch.Tensor] = [
            torch.zeros_like(frame) for _ in range(capacity)
        ]

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def rotate(self) -> torch.Tensor:
        """
        Evict the oldest slot, zero it, and return it ready for the caller to fill.

        Must be followed by :meth:`push` with the same tensor.

        Returns:
            The evicted (zeroed) tensor slot.
        """
        slot = self._frames.pop()
        slot.zero_()
        return slot

    def push(self, frame: torch.Tensor) -> None:
        """
        Push a filled frame to the front (most-recent position) of the buffer.

        Args:
            frame: The tensor returned by the preceding :meth:`rotate` call,
                   now filled with new data.
        """
        self._frames.insert(0, frame)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def output(self, out: torch.Tensor) -> None:
        """
        Copy all frames into ``out`` along the last axis (in-place, no allocation).

        Frames are written newest-first so the most recent observation is at the
        start of the concatenated vector, matching the ordering used during training.

        Works for any number of leading batch dimensions because indexing uses
        ``out[..., offset:offset+n]``.

        Args:
            out: Pre-allocated tensor whose last dimension equals
                 ``frame_size * capacity``.
        """
        offset = 0
        for frame in self._frames:
            n = frame.shape[-1]
            out[..., offset : offset + n] = frame
            offset += n

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all slots. Call at the start of each episode."""
        for frame in self._frames:
            frame.zero_()

    def __len__(self) -> int:
        return self._capacity
