from typing import TypedDict
from enum import Enum


class Button(Enum):
    A = 1
    B = 2
    X = 3
    Y = 4
    LB = 5
    RB = 6
    LT = 7
    RT = 8
    BACK = 9
    START = 10
    MODE = 11
    UP = 12
    DOWN = 13
    LEFT = 14
    RIGHT = 15
    LEFT_JOYSTICK = 16
    RIGHT_JOYSTICK = 17


class DPad(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class GamepadMapping(TypedDict):
    """
    Defines the how to extract a value from a gamepad data array.
    """

    data: int
    """The index of the data in the gamepad data array."""

    button: Button
    """If this is a button, what is it's name"""

    axis: int
    """If this is a joystick, what is it's axis number"""

    dpad: DPad
    """D-pad direction."""

    bitmask: int
    """The bitmask to extract the value from the data."""

    value: int
    """Match this value to the value at the data index."""


class GamepadState:
    """Data from a gamepad."""

    axis_values: list[float] = []
    """The value (-1 to 1) for each axes."""

    buttons: list[str] = []
    """A list of the buttons that are pressed."""

    dpad: DPad
    """D-pad direction."""

    def axis(self, index: int):
        """
        Get the axis value at an index.
        This is the preferred way to get the axis value, because the axis array will not be filled until the gamepad
        receives input.

        Args:
            index: The index of the axis to get the value of.

        Returns:
            The value of the axis at the index.
        """
        if index >= len(self.axis_values):
            return 0.0
        return self.axis_values[index]

    def __repr__(self):
        return f"GamepadState(axis={self.axis_values}, buttons={self.buttons}, dpad={self.dpad})"


class GamepadConfig(TypedDict):
    """
    Defines a gamepad, how to connect to it, and how to map the buttons and axes.
    """

    name: str
    """The name of the gamepad."""

    vendor_id: int
    """The vendor id of the gamepad."""

    product_id: int
    """The product id of the gamepad."""

    mapping: list[GamepadMapping]
    """The mapping of the gamepad."""
