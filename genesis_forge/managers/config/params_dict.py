from typing import Callable


class ParamsDict(dict):
    """
    A dictionary with an on-change callback handler.
    We use this to rebuild a reset class when a parameter is changed.
    """

    def __init__(self, params: dict, on_change: Callable[[], None]):
        super().__init__(params)
        self._on_change = on_change

    def __setitem__(self, key, value):
        """Call the on change function when a value is changed."""
        super().__setitem__(key, value)
        self._on_change()

    def __delitem__(self, key):
        """Call the on change function when a value is deleted."""
        super().__delitem__(key)
        self._on_change()
