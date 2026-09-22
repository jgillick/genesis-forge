"""
Gamepad wrapper for backward compatibility with the old API.

This provides a polling-based interface on top of the event-based SDL2 implementation.
"""

from __future__ import annotations

import threading

from genesis_forge.gamepads.common import Key
from genesis_forge.gamepads.sdl2 import ControllerEventLoop

__all__ = ["Gamepad"]


class Gamepad:
    """
    Wrapper around SDL2 ControllerEventLoop that provides a polling-based interface.

    This maintains axis and button state that can be queried at any time, similar to
    the old HID-based Gamepad API.

    Example::

        >>> gamepad = Gamepad()
        >>> gamepad.axis(0)  # Get left stick X axis
        0.0
        >>> gamepad.buttons()  # Get list of pressed buttons
        ['a', 'b']
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the SDL2 gamepad wrapper.

        Args:
            debug: If True, print debug information
        """
        self._debug = debug
        self.is_running = True
        self._lock = threading.Lock()

        # State storage
        self._axis_values: list[float] = [0.0] * 6
        self._button_set: set[str] = set()
        self._dpad: str | None = None

        # Start the SDL2 event loop in a background thread
        self._event_loop = ControllerEventLoop(
            handle_key=self._handle_key,
            alive=threading.Event(),
        )
        self._event_loop.alive.set()
        self._read_thread = threading.Thread(target=self._event_loop.run, daemon=True)
        self._read_thread.start()

        if self._debug:
            print("SDL2 gamepad wrapper initialized")

    def axis(self, index: int) -> float:
        """
        Get the value of an axis.

        Args:
            index: The axis index (0-5)

        Returns:
            The axis value in range [-1.0, 1.0] for sticks, [0.0, 1.0] for triggers
        """
        with self._lock:
            if index >= len(self._axis_values):
                return 0.0
            return self._axis_values[index]

    def buttons(self) -> list[str]:
        """
        Get the list of currently pressed buttons.

        Returns:
            List of button names (lowercase, e.g., 'a', 'b', 'x', 'y')
        """
        with self._lock:
            return list(self._button_set)

    @property
    def dpad(self) -> str | None:
        """Get the current D-pad direction (e.g., 'up', 'down', 'left', 'right')."""
        with self._lock:
            return self._dpad

    def _handle_key(self, key: Key) -> None:
        """Handle a key event from the SDL2 controller."""
        with self._lock:
            if key.keytype == Key.AXIS:
                # Map SDL2 axis to our axis array
                axis_idx = key.index
                if axis_idx < len(self._axis_values):
                    self._axis_values[axis_idx] = (
                        key.value if key.value is not None else 0.0
                    )

            elif key.keytype == Key.BUTTON:
                button_name = key.name or f"button_{key.index}"
                if key.value == 1:  # Button pressed
                    self._button_set.add(button_name)
                else:  # Button released
                    self._button_set.discard(button_name)

                # Handle D-pad buttons specially
                if button_name in ["dpup", "dpdown", "dpleft", "dpright"]:
                    if key.value == 1:
                        # Map to simple direction names
                        self._dpad = button_name.replace("dp", "")
                    else:
                        self._dpad = None

        if self._debug:
            print(f"Key event: {key}")

    def stop(self) -> None:
        """Stop reading gamepad input."""
        self.is_running = False
        self._event_loop.stop()
