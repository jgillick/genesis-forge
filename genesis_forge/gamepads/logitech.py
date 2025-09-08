"""Logitech F310/F710 Gamepad configuration."""

from .config import Button, DPad, GamepadConfig

VENDOR_ID = 0x046D

LOGITECH_F710_CONFIG: GamepadConfig = {
    "name": "F710",
    "vendor_id": VENDOR_ID,
    "product_id": 0xC219,
    "mapping": [
        {"axis": 0, "data": 1},
        {"axis": 1, "data": 2},
        {"axis": 2, "data": 3},
        {"axis": 3, "data": 4},
        {"dpad": DPad.UP, "data": 5, "bitmask": 15, "value": 0},
        {"dpad": DPad.DOWN, "data": 5, "bitmask": 15, "value": 4},
        {"dpad": DPad.RIGHT, "data": 5, "bitmask": 15, "value": 2},
        {"dpad": DPad.LEFT, "data": 5, "bitmask": 15, "value": 6},
        {"button": Button.A, "data": 5, "bitmask": 32},
        {"button": Button.B, "data": 5, "bitmask": 64},
        {"button": Button.X, "data": 5, "bitmask": 16},
        {"button": Button.Y, "data": 5, "bitmask": 128},
        {"button": Button.LB, "data": 6, "bitmask": 1},
        {"button": Button.RB, "data": 6, "bitmask": 2},
        {"button": Button.LT, "data": 6, "bitmask": 4},
        {"button": Button.RT, "data": 6, "bitmask": 8},
        {"button": Button.BACK, "data": 6, "bitmask": 16},
        {"button": Button.START, "data": 6, "bitmask": 32},
        {"button": Button.MODE, "data": 7, "bitmask": 8},
        {"button": Button.LEFT_JOYSTICK, "data": 6, "bitmask": 64},
        {"button": Button.RIGHT_JOYSTICK, "data": 6, "bitmask": 128},
    ],
}

LOGITECH_F310_CONFIG: GamepadConfig = {
    "name": "F310",
    "vendor_id": VENDOR_ID,
    "product_id": 0xC216,
    "mapping": [
        {"axis": 0, "data": 0},
        {"axis": 1, "data": 1},
        {"axis": 2, "data": 2},
        {"axis": 3, "data": 3},
        {"dpad": DPad.UP, "data": 4, "bitmask": 15, "value": 0},
        {"dpad": DPad.DOWN, "data": 4, "bitmask": 15, "value": 4},
        {"dpad": DPad.RIGHT, "data": 4, "bitmask": 15, "value": 2},
        {"dpad": DPad.LEFT, "data": 4, "bitmask": 15, "value": 6},
        {"button": Button.A, "data": 4, "bitmask": 32},
        {"button": Button.B, "data": 4, "bitmask": 64},
        {"button": Button.X, "data": 4, "bitmask": 16},
        {"button": Button.Y, "data": 4, "bitmask": 128},
        {"button": Button.LB, "data": 5, "bitmask": 1},
        {"button": Button.RB, "data": 5, "bitmask": 2},
        {"button": Button.LT, "data": 5, "bitmask": 4},
        {"button": Button.RT, "data": 5, "bitmask": 8},
        {"button": Button.BACK, "data": 5, "bitmask": 16},
        {"button": Button.START, "data": 5, "bitmask": 32},
        {"button": Button.MODE, "data": 6, "bitmask": 8},
        {"button": Button.LEFT_JOYSTICK, "data": 5, "bitmask": 64},
        {"button": Button.RIGHT_JOYSTICK, "data": 5, "bitmask": 128},
    ],
}
