"""DragonRise Inc. Generic USB Joystick configuration.

Common cheap USB gamepad found on Amazon/AliExpress.
Vendor ID: 0x0079, Product ID: 0x0006
"""

from .config import GamepadConfig

VENDOR_ID = 0x0079
PRODUCT_ID = 0x0006

# HID data mapping (determined via testing):
#   Byte 0: Left stick X (0=left, 128=center, 255=right)
#   Byte 1: Left stick Y (0=up/forward, ~109=center, 255=down/back)
#   Byte 2: Right stick X (0=left, ~125=center, 255=right)
#   Byte 3: Right stick Y (128=center)
#   Byte 4: Triggers/Z-axis
#   Byte 5: D-pad and face buttons (lower nibble = d-pad, upper nibble = buttons)
#   Byte 6: Shoulder buttons
#   Byte 7: Mode/special buttons

DRAGONRISE_CONFIG: GamepadConfig = {
    "name": "DragonRise Generic USB Joystick",
    "vendor_id": VENDOR_ID,
    "product_id": PRODUCT_ID,
    "mapping": [
        # Analog sticks
        {"axis": 0, "data": 0},  # Left stick X
        {"axis": 1, "data": 1},  # Left stick Y
        {"axis": 2, "data": 2},  # Right stick X
        {"axis": 3, "data": 3},  # Right stick Y
    ],
}
