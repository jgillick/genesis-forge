import time
import argparse
import hid
import threading

from .config import GamepadConfig, GamepadState
from .logitech import LOGITECH_F710_CONFIG, LOGITECH_F310_CONFIG

GAMEPAD_CONFIGS = [
    LOGITECH_F710_CONFIG,
    LOGITECH_F310_CONFIG,
]


class Gamepad:
    """
    General gamepad controller, which automatically attempts to connect to known gamepads (currentlyLogitech F710 and F310).

    Example::

        >>> gamepad = Gamepad()
        >>> gamepad.state
        GamepadState(axis=[0.0, 0.0, 0.0, 0.0], buttons=[A], dpad=UP)
        >>> gamepad.state.axis
        [0.0, 0.0, 0.0, 0.0]
        >>> gamepad.state.buttons
        ["A"]
        >>> gamepad.state.dpad
        "UP"
        >>> gamepad.state.buttons = [Button.A]

    Example connecting to a specific gamepad:

        >>> gamepad = Gamepad(config=LOGITECH_F710_CONFIG)


    Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/a873d53765a4c83572cf44fa74768ab62ceb7be1/mujoco_playground/experimental/sim2sim/gamepad_reader.py.
    """

    def __init__(
        self,
        config: GamepadConfig = None,
        vendor_id=None,
        product_id=None,
        debug=False,
    ):
        self._config = config
        self._vendor_id = vendor_id
        self._product_id = product_id

        if vendor_id is None and config is not None:
            self._vendor_id = config["vendor_id"]
        if product_id is None and config is not None:
            self._product_id = config["product_id"]

        self._state = GamepadState()
        self._debug = debug

        self.is_running = True
        self._device = None

        self.connect()
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

    @property
    def state(self) -> GamepadState:
        """
        The current state of the gamepad.
        """
        return self._state

    def parse_data(self, data: list[int]) -> GamepadState:
        """Parse gamepad data into a GamepadState object."""
        axis = []
        buttons = []
        dpad = None

        # No gamepad config, so we cann't parse the data
        if self._config is None:
            return

        for cfg in self._config["mapping"]:
            if "data" not in cfg:
                print(f"Warning: {cfg} has no data value")
                continue
            if cfg["data"] >= len(data):
                print(f"Error: {cfg} data is out of range")
                continue
            value = data[cfg["data"]]
            value_truthy = False

            # Apply the bitmask to the value
            if "bitmask" in cfg:
                value = value & cfg["bitmask"]
                if value != 0:
                    value_truthy = True
            elif "button" in cfg or "dpad" in cfg:
                print(f"Warning: {cfg} has no bitmask value")
                continue

            # Check if value is matches
            if "value" in cfg:
                value_truthy = value == cfg["value"]

            if "button" in cfg and value_truthy:
                buttons.append(cfg["button"].name)
            elif "dpad" in cfg and value_truthy:
                dpad = cfg["dpad"].name
            elif "axis" in cfg:
                value = -(value - 128) / 128.0
                axis.insert(cfg["axis"], value)

        self._state.axis_values = axis
        self._state.buttons = buttons
        self._state.dpad = dpad
        return self._state

    def auto_connect(self):
        """
        Loop through the known gamepad configs until one connects.
        """
        for config in GAMEPAD_CONFIGS:
            self._vendor_id = config["vendor_id"]
            self._product_id = config["product_id"]
            self._config = config
            try:
                if self.connect():
                    return
            except:
                pass
        raise IOError(f"Could not find a gamepad to connect to")

    def connect(self, vendor_id=None, product_id=None):
        """
        Attempt to connect to a gamepad.

        Args:
            vendor_id: The vendor id of the gamepad to connect to.
            product_id: The product id of the gamepad to connect to.

        Returns:
            True if the gamepad connected successfully, False otherwise.
        """
        if vendor_id is None:
            vendor_id = self._vendor_id
        if product_id is None:
            product_id = self._product_id

        # If the vendor/product IDs aren't set, loop through the available gamepad configs
        if product_id is None and vendor_id is None:
            self.auto_connect()
            return

        try:
            self._device = hid.device()
            self._device.open(vendor_id, product_id)
            self._device.set_nonblocking(True)
            print(
                f"Connected to gamepad {self._device.get_manufacturer_string()} {self._device.get_product_string()}"
            )
            return True
        except IOError as e:
            raise IOError(
                f"Error connecting to gamepad 0x{vendor_id:04x}:0x{product_id:04x}: {e}"
            )

    def _read_loop(self):
        """
        Wait for gamepad input, and then update the gamepad state.
        """
        while self.is_running:
            try:
                data = self._device.read(64)
                if data:
                    try:
                        self._state = self.parse_data(data)
                        if self._debug:
                            print(self._state)
                    except Exception as e:
                        print(f"Error parsing data: {e}")
            except Exception as e:
                print(f"Error reading from device: {e}")

        self._device.close()

    def stop(self):
        """
        Stop reading gamepad input.
        """
        self.is_running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Gamepad connection", add_help=True
    )
    args = parser.parse_args()

    gamepad = Gamepad(debug=True)
    while True:
        time.sleep(1.0)
