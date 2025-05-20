class FINNError(Exception):
    """Base-class for FINN exceptions. Useful to differentiate exceptions while catching."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UserError(FINNError):
    """Custom exception class which should be used to
    print errors without stacktraces if debug is disabled."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNConfigurationError(FINNError):
    """Error emitted when FINN is configured incorrectly"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
