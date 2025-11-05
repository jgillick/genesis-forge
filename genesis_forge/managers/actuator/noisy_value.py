class NoisyValue:
    """
    Defines a value with some noise settings.
    """

    def __init__(
        self,
        value: float,
        noise: float = 0.0,
    ):
        """
        Args:
            value: The value to configure the manager with.
            noise: The noise (+/-) to apply to the value as noise.

        Example:
            >>> value = NoisyValue(10.0, noise=0.01)
        """
        self.value = value
        self.noise = noise
