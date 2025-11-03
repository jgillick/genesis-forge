class NoisyValue:
    """
    Defines a value with some noise settings.
    """

    def __init__(
        self,
        value: float | None = None,
        noise: float | None = None,
    ):
        """
        Args:
            value: The value to configure the manager with.
            noise: The noise scale (+/-) to apply to the value as noise.

        Example:
            >>> value = NoisyValue(10.0, noise=0.01)
        """
        self.value = value
        self.noise = 0
