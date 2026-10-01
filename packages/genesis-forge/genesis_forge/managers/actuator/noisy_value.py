class NoisyValue:
    """
    Defines a base value and the noise which will be added to it.
    This class is merely a configuration container, and does not apply the noise directly.

    Args:
        value: The value to configure the manager with.
        noise: The noise (+/-) to apply to the value as noise.

    Example::

        value = NoisyValue(10.0, noise=2.0)
        # The base value is 10.0, and the noise will be +/- 2.0
        # So the final value will be between 8.0 and 12.0
    """

    def __init__(
        self,
        value: float,
        noise: float = 0.0,
    ):
        self.value = value
        self.noise = noise
