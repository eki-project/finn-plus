"""Here we organize FINN+`s exceptions and error handling."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

"""
FINNError is the base class for all errors.
FINNUserError is a purely user-facing error that has nothing to do with FINNs internals
FINNInternalError is a compiler internal error

Every error should subclass FINNUserError or FINNInternalError
"""


class FINNError(Exception):
    """Base-class for all FINN exceptions. Useful to differentiate exceptions while catching."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNError."""
        super().__init__(*args)


class FINNInternalError(FINNError):
    """Custom exception class for internal compiler errors."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNInternalError."""
        super().__init__(*args)


class FINNUserError(FINNError):
    """Custom exception class which should be used to
    print errors without stacktraces if debug is disabled.
    """

    def __init__(self, *args: object) -> None:
        """Create a new FINNUserError."""
        super().__init__(*args)


class FINNDependencyInstallationError(FINNUserError):
    """Error emitted by the DependencyManager if something fails."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNValidationError."""
        super().__init__(*args)


class FINNValidationError(FINNUserError):
    """Error emitted if the settings could not be properly parsed by Pydantic."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNValidationError."""
        super().__init__(*args)


class FINNSynthesisError(FINNUserError):
    """Error emitted if synthesis fails. Contains the path to the Vivado logfile."""

    def __init__(self, msg: str, vivado_logfile: Path) -> None:
        """Create a new FINNSynthesisError."""
        super().__init__(msg)
        self.msg = msg
        self.vivado_logfile = vivado_logfile


class FINNConfigurationError(FINNUserError):
    """Error emitted if FINN is configured incorrectly."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNConfigurationError."""
        super().__init__(*args)


class FINNNodeError(FINNUserError):
    """User-facing error that can be attributed to a single node of the model graph.

    The message is available as ``msg`` (and via ``str()``), the name and the op type of the
    node as ``node_name`` and ``op_type``. Both are ``None`` if the error could not be
    attributed to a single node.
    """

    def __init__(self, msg: str, node_name: str | None = None, op_type: str | None = None) -> None:
        """Create a new FINNNodeError."""
        super().__init__(msg)
        self.msg = msg
        self.node_name = node_name
        self.op_type = op_type


class FINNUnsupportedOpError(FINNNodeError):
    """Error emitted if a node has no hardware mapping or no hardware implementation.

    ``unmapped_nodes`` is filled by the dataflow partitioning step with one entry per node
    that is not mapped to a hardware layer, in the format of the ``unmapped_layer_report``
    analysis (keys ``name``, ``op_type``, ``domain`` and ``position``). It is empty otherwise.
    """

    def __init__(
        self,
        msg: str,
        node_name: str | None = None,
        op_type: str | None = None,
        unmapped_nodes: list[dict[str, str]] | None = None,
    ) -> None:
        """Create a new FINNUnsupportedOpError."""
        super().__init__(msg, node_name, op_type)
        self.unmapped_nodes = [] if unmapped_nodes is None else list(unmapped_nodes)


class FINNFoldingError(FINNNodeError):
    """Error emitted if a folding (parallelization) configuration is illegal or cannot be
    achieved.

    ``violations`` is filled by ``step_apply_folding_config``, which checks the whole model:
    it maps the name of every node with an illegal folding to its message, and ``node_name``
    and ``op_type`` describe the first of them. It is empty otherwise.
    """

    def __init__(
        self,
        msg: str,
        node_name: str | None = None,
        op_type: str | None = None,
        violations: dict[str, str] | None = None,
    ) -> None:
        """Create a new FINNFoldingError."""
        super().__init__(msg, node_name, op_type)
        self.violations = {} if violations is None else dict(violations)


class FINNDataflowError(FINNInternalError):
    """(Internal) Errors regarding the dataflow, dataflow config, step resolution, etc."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNDataflowError."""
        super().__init__(*args)


class FINNMultiFPGAError(FINNInternalError):
    """(Internal) Multi-FPGA error during one of the transformations."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNMultiFPGAError."""
        super().__init__(*args)


class FINNMultiFPGAUserError(FINNUserError):
    """(User) Multi-FPGA error during one of the transformations."""

    # TODO: Reorganize error-types

    def __init__(self, *args: object) -> None:
        """Create a new FINNMultiFPGAUserError."""
        super().__init__(*args)


class FINNMultiFPGAConfigError(FINNUserError):
    """(User) Multi-FPGA Error in the configuration or the model."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNMultiFPGAConfigError."""
        super().__init__(*args)


class FINNMultiFPGAPartitionerError(FINNUserError):
    """(User) Multi-FPGA Error in the partitioning of the model."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNMultiFPGAPartitionerError."""
        super().__init__(*args)


class FINNVitisLinkConfigError(FINNInternalError):
    """(Internal) An error appearing in a vitis link configuration when trying to
    generate a script or config. May happen in both single- and multifpga cases.
    """

    def __init__(self, *args: object) -> None:
        """Create a new FINNVitisLinkConfigError."""
        super().__init__(*args)
