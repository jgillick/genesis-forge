"""Reading the driving command off a gamepad, on the robot.

Uses SDL2 through `pysdl2`, the same backend `genesis_forge.gamepads` uses in the
simulator, so the axes mean the same thing on both sides.
"""

from __future__ import annotations

import time

import numpy as np
import sdl2

# The maximum control value for a fully deflected stick, matching the ranges used in training.
LIN_CTRL_LIMIT = 0.5
ANG_CTRL_LIMIT = 2.0

# The button that arms the car, before which the sticks are ignored.
ARM_BUTTON = sdl2.SDL_CONTROLLER_BUTTON_START


class Gamepad:
    """
    Turns stick positions into the policy's `velocity_cmd` observation.
    The car is entirely controlled with the left stick."""

    def __init__(self, controller) -> None:
        self._set_controller(controller)

    @classmethod
    def wait_for_connection(cls) -> Gamepad:
        """Block until a controller is plugged in and the driver presses start."""
        return cls(cls._wait_for_connect_ready())

    @property
    def connected(self) -> bool:
        """False once the pad disconnects, which has to stop the car."""
        return bool(sdl2.SDL_GameControllerGetAttached(self.controller))

    def command(self) -> np.ndarray:
        """The `velocity_cmd` observation: [forward, sideways, turn]."""
        sdl2.SDL_PumpEvents()
        forward = -self._normed_axis(sdl2.SDL_CONTROLLER_AXIS_LEFTY) * LIN_CTRL_LIMIT
        turn = -self._normed_axis(sdl2.SDL_CONTROLLER_AXIS_LEFTX) * ANG_CTRL_LIMIT
        return np.array([forward, 0.0, turn], dtype=np.float32)

    def reconnect(self) -> None:
        """Drop the detached pad and adopt a freshly connected one."""
        self.close()
        print("Gamepad disconnected.")
        controller = self._wait_for_connect_ready()
        self._set_controller(controller)

    def close(self) -> None:
        """Release the pad. SDL itself stays up, ready for the next one."""
        sdl2.SDL_GameControllerClose(self.controller)

    @staticmethod
    def _wait_for_connect_ready():
        """Wait for a controller, then for A, and return the armed handle."""
        sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER)
        print("Waiting for a gamepad...")

        while True:
            controller = Gamepad._wait_for_connection()
            if Gamepad._wait_for_arm(controller):
                return controller
            # It left before it was ever armed -- a pad SDL still lists but has
            # dropped will land here every pass, so sleep before looking again.
            sdl2.SDL_GameControllerClose(controller)
            time.sleep(0.5)

    @staticmethod
    def _wait_for_connection():
        """Block until SDL reports a game controller, and open it."""
        while True:
            sdl2.SDL_PumpEvents()
            for index in range(sdl2.SDL_NumJoysticks()):
                if sdl2.SDL_IsGameController(index):
                    return sdl2.SDL_GameControllerOpen(index)
            time.sleep(0.5)

    @staticmethod
    def _wait_for_arm(controller) -> bool:
        """Block until ARM_BUTTON is pressed; False if the pad left before it was."""
        label = sdl2.SDL_GameControllerGetStringForButton(ARM_BUTTON)
        label = label.decode().upper() if label else "the arm button"
        print(f"Press the {label} button on the gamepad...")
        while True:
            sdl2.SDL_PumpEvents()
            if not sdl2.SDL_GameControllerGetAttached(controller):
                return False
            pressed = bool(sdl2.SDL_GameControllerGetButton(controller, ARM_BUTTON))
            if pressed:
                return True
            time.sleep(0.02)

    def _set_controller(self, controller) -> None:
        """Adopt a freshly opened controller handle."""
        self.controller = controller
        name = sdl2.SDL_GameControllerName(controller)
        self.name = name.decode() if name else "gamepad"

    def _normed_axis(self, axis: int) -> float:
        """One stick axis normalized to -1..1."""
        value = sdl2.SDL_GameControllerGetAxis(self.controller, axis) / 32767.0
        return max(-1.0, min(1.0, value))
