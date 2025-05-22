"""
FINNError is the base class for all errors.
FINNUserError is a purely user-facing error that has nothing to do with FINNs internals
FINNInternalError is a compiler internal error

Every error should subclass FINNUserError or FINNInternalError
"""


class FINNError(Exception):
    """Base-class for FINN exceptions. Useful to differentiate exceptions while catching."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNInternalError(FINNError):
    """Custom exception class for internal compiler errors"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNUserError(FINNError):
    """Custom exception class which should be used to
    print errors without stacktraces if debug is disabled."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNConfigurationError(FINNUserError):
    """(User) Error emitted when FINN is configured incorrectly"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNDataflowError(FINNInternalError):
    """(Internal) Errors regarding the dataflow, dataflow config, step resolution, etc."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNMultiFPGAError(FINNInternalError):
    """(Internal) Multi-FPGA error during one of the transformations"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNMultiFPGAConfigError(FINNUserError):
    """(User) Multi-FPGA Error in the configuration or the model"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNVitisLinkConfigError(FINNInternalError):
    """(Internal) An error appearing in a vitis link configuration when trying to
    generate a script or config. May happen in both single- and multifpga cases"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
