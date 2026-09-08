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


class Gamepad:
    """Turns stick positions into the policy's `velocity_cmd` observation.

    `VelocityCommandManager.use_gamepad` maps SDL axis 1 (left stick, vertical)
    to `lin_vel_x` and axis 2 (right stick, horizontal) to `ang_vel_z`, inverting
    both because SDL reports up and left as negative. Reproducing that here is
    what makes driving the car feel like driving it in the viewer. `lin_vel_y`
    stays zero: the car cannot move sideways, and training fixed its range at 0.
    """

    def __init__(self, controller) -> None:
        self.controller = controller
        name = sdl2.SDL_GameControllerName(controller)
        self.name = name.decode() if name else "gamepad"

    @classmethod
    def wait_for_connection(cls) -> Gamepad:
        """Block until a controller is plugged in or paired."""
        sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER)
        print("Waiting for a gamepad...")
        while True:
            sdl2.SDL_PumpEvents()
            for index in range(sdl2.SDL_NumJoysticks()):
                if sdl2.SDL_IsGameController(index):
                    return cls(sdl2.SDL_GameControllerOpen(index))
            time.sleep(0.5)

    @property
    def attached(self) -> bool:
        """False once the pad disconnects, which has to stop the car."""
        return bool(sdl2.SDL_GameControllerGetAttached(self.controller))

    def command(self) -> np.ndarray:
        """The `velocity_cmd` observation: [forward, sideways, turn]."""
        sdl2.SDL_PumpEvents()
        forward = -self._axis(sdl2.SDL_CONTROLLER_AXIS_LEFTY) * LIN_CTRL_LIMIT
        turn = -self._axis(sdl2.SDL_CONTROLLER_AXIS_RIGHTX) * ANG_CTRL_LIMIT
        return np.array([forward, 0.0, turn], dtype=np.float32)

    def close(self) -> None:
        """Release the pad and shut SDL down. `wait_for_connection` starts it again."""
        sdl2.SDL_GameControllerClose(self.controller)
        sdl2.SDL_Quit()

    def _axis(self, axis: int) -> float:
        """One stick axis normalized to -1..1."""
        value = sdl2.SDL_GameControllerGetAxis(self.controller, axis) / 32767.0
        return max(-1.0, min(1.0, value))
