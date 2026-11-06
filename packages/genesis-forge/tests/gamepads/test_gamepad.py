"""Tests for Gamepad connection tracking and its reset on disconnect.

The SDL2 calls that open and close controllers are faked and the event loop is
never run, so no controller or SDL subsystem is needed.
"""

import pytest

from genesis_forge.gamepads import Gamepad, Key
from genesis_forge.gamepads import sdl2 as gamepad_sdl2
from genesis_forge.gamepads.sdl2 import ControllerEventLoop


@pytest.fixture
def gamepad(monkeypatch):
    # Device index N opens as a controller whose instance id is N
    monkeypatch.setattr(
        gamepad_sdl2.sdl2, "SDL_GameControllerOpen", lambda idx: idx + 1
    )
    monkeypatch.setattr(gamepad_sdl2.sdl2, "SDL_GameControllerGetJoystick", lambda c: c)
    monkeypatch.setattr(gamepad_sdl2.sdl2, "SDL_JoystickInstanceID", lambda j: j - 1)
    monkeypatch.setattr(gamepad_sdl2.sdl2, "SDL_GameControllerClose", lambda c: None)
    monkeypatch.setattr(ControllerEventLoop, "run", lambda self: None)

    gp = Gamepad()
    yield gp
    gp.stop()


def press(gamepad: Gamepad) -> None:
    gamepad._handle_key(Key(Key.AXIS, 1, "lefty", -0.8))
    gamepad._handle_key(Key(Key.BUTTON, 0, "a", 1))
    gamepad._handle_key(Key(Key.BUTTON, 11, "dpup", 1))


def test_not_connected_until_a_controller_is_opened(gamepad):
    assert not gamepad.connected
    assert gamepad.wait_for_connection(timeout=0.01) is False

    gamepad._event_loop._open_controller(0)

    assert gamepad.connected
    assert gamepad.wait_for_connection(timeout=0.01) is True


def test_stays_connected_until_the_last_controller_is_removed(gamepad):
    gamepad._event_loop._open_controller(0)
    gamepad._event_loop._open_controller(1)

    gamepad._event_loop._close_controller(0)
    assert gamepad.connected

    gamepad._event_loop._close_controller(1)
    assert not gamepad.connected


def test_disconnect_resets_axes_buttons_and_dpad(gamepad):
    gamepad._event_loop._open_controller(0)
    press(gamepad)
    assert gamepad.axis(1) == pytest.approx(-0.8)

    gamepad._event_loop._close_controller(0)

    assert gamepad.axis(1) == 0.0
    assert gamepad.buttons() == []
    assert gamepad.dpad is None


def test_removing_any_controller_resets_state(gamepad):
    gamepad._event_loop._open_controller(0)
    gamepad._event_loop._open_controller(1)
    press(gamepad)

    gamepad._event_loop._close_controller(1)

    assert gamepad.connected
    assert gamepad.axis(1) == 0.0
    assert gamepad.buttons() == []


def test_closing_an_unknown_controller_keeps_state(gamepad):
    gamepad._event_loop._open_controller(0)
    press(gamepad)

    gamepad._event_loop._close_controller(42)

    assert gamepad.axis(1) == pytest.approx(-0.8)
    assert gamepad.buttons()
