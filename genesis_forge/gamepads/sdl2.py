from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import sdl2

from genesis_forge.gamepads.common import Key

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ControllerEventLoop", "controller_key_from_event"]


AXIS_NAMES: dict[int, str] = {
    sdl2.SDL_CONTROLLER_AXIS_LEFTX: "leftx",
    sdl2.SDL_CONTROLLER_AXIS_LEFTY: "lefty",
    sdl2.SDL_CONTROLLER_AXIS_RIGHTX: "rightx",
    sdl2.SDL_CONTROLLER_AXIS_RIGHTY: "righty",
    sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT: "lefttrigger",
    sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT: "righttrigger",
}

BUTTON_NAMES: dict[int, str] = {
    sdl2.SDL_CONTROLLER_BUTTON_A: "a",
    sdl2.SDL_CONTROLLER_BUTTON_B: "b",
    sdl2.SDL_CONTROLLER_BUTTON_X: "x",
    sdl2.SDL_CONTROLLER_BUTTON_Y: "y",
    sdl2.SDL_CONTROLLER_BUTTON_BACK: "back",
    sdl2.SDL_CONTROLLER_BUTTON_GUIDE: "guide",
    sdl2.SDL_CONTROLLER_BUTTON_START: "start",
    sdl2.SDL_CONTROLLER_BUTTON_LEFTSTICK: "leftstick",
    sdl2.SDL_CONTROLLER_BUTTON_RIGHTSTICK: "rightstick",
    sdl2.SDL_CONTROLLER_BUTTON_LEFTSHOULDER: "leftshoulder",
    sdl2.SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: "rightshoulder",
    sdl2.SDL_CONTROLLER_BUTTON_DPAD_UP: "dpup",
    sdl2.SDL_CONTROLLER_BUTTON_DPAD_DOWN: "dpdown",
    sdl2.SDL_CONTROLLER_BUTTON_DPAD_LEFT: "dpleft",
    sdl2.SDL_CONTROLLER_BUTTON_DPAD_RIGHT: "dpright",
}


def _rescale(
    value: int, in_min: int, in_max: int, out_min: float, out_max: float
) -> float:
    span_in = in_max - in_min
    if span_in == 0:
        return out_min
    return (float(value - in_min) / span_in) * (out_max - out_min) + out_min


def _is_trigger_axis(axis: int) -> bool:
    return axis in (
        sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT,
        sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT,
    )


def controller_key_from_event(
    event: sdl2.SDL_Event, deadzone: float = 0.05
) -> Key | None:
    etype = event.type

    if etype == sdl2.SDL_CONTROLLERAXISMOTION:
        axis = event.caxis.axis
        raw = event.caxis.value
        limits = (0.0, 1.0) if _is_trigger_axis(axis) else (-1.0, 1.0)
        value = _rescale(raw, -32768, 32767, limits[0], limits[1])
        if not _is_trigger_axis(axis) and abs(value) < deadzone:
            value = 0.0
        return Key(Key.AXIS, axis, AXIS_NAMES.get(axis), value)

    if etype in (sdl2.SDL_CONTROLLERBUTTONDOWN, sdl2.SDL_CONTROLLERBUTTONUP):
        button = event.cbutton.button
        val = 1 if etype == sdl2.SDL_CONTROLLERBUTTONDOWN else 0
        return Key(Key.BUTTON, button, BUTTON_NAMES.get(button), val)

    return None


class ControllerEventLoop:
    """
    Minimal SDL2 controller event loop for genesis-forge.

    This class runs in a background thread and processes SDL2 controller events,
    converting them to Key objects and passing them to a callback function.

    Args:
        handle_key: Callback function that receives Key objects for each controller event
        alive: Optional threading.Event to control the event loop. When cleared, the loop exits.
        timeout: Milliseconds to wait for events before checking the alive flag (default: 2000ms)

    Example::

        def on_key(key: Key):
            print(f"Key event: {key}")

        loop = ControllerEventLoop(handle_key=on_key)
        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        # Later, to stop:
        loop.stop()
    """

    def __init__(
        self,
        handle_key: Callable[[Key], None],
        alive: threading.Event | None = None,
        timeout: int = 2000,
    ) -> None:
        self.handle_key = handle_key
        self.alive = alive or threading.Event()
        if not self.alive.is_set():
            self.alive.set()
        self.timeout = timeout
        self._controllers: dict[int, sdl2.SDL_GameController] = {}

    def stop(self) -> None:
        self.alive.clear()

    def _open_controller(self, device_index: int) -> None:
        controller = sdl2.SDL_GameControllerOpen(device_index)
        if controller:
            joy = sdl2.SDL_GameControllerGetJoystick(controller)
            instance_id = sdl2.SDL_JoystickInstanceID(joy)
            self._controllers[int(instance_id)] = controller

    def _close_controller(self, instance_id: int) -> None:
        controller = self._controllers.pop(instance_id, None)
        if controller:
            sdl2.SDL_GameControllerClose(controller)

    def _handle_event(self, event: sdl2.SDL_Event) -> None:
        etype = event.type
        if etype == sdl2.SDL_CONTROLLERDEVICEADDED:
            self._open_controller(event.cdevice.which)
            return
        if etype == sdl2.SDL_CONTROLLERDEVICEREMOVED:
            self._close_controller(event.cdevice.which)
            return
        key = controller_key_from_event(event)
        if key and self.handle_key:
            self.handle_key(key)

    def run(self) -> None:
        if sdl2.SDL_WasInit(sdl2.SDL_INIT_GAMECONTROLLER) == 0:
            sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_GAMECONTROLLER)

        for idx in range(sdl2.SDL_NumJoysticks()):
            if sdl2.SDL_IsGameController(idx):
                self._open_controller(idx)

        event = sdl2.SDL_Event()
        try:
            while self.alive.is_set():
                if sdl2.SDL_WaitEventTimeout(event, self.timeout):
                    self._handle_event(event)
        finally:
            for instance_id in list(self._controllers.keys()):
                self._close_controller(instance_id)
            sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_GAMECONTROLLER)
