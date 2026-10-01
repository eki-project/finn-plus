"""Tests for the registry of Multi-FPGA communication backends."""

import pytest

from finn.builder.build_dataflow_config import (
    MFCommunicationKernel,
    MFVerbosity,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga import backend as backend_module
from finn.transformation.fpgadataflow.multifpga.backend import get_backend, register_backend
from finn.transformation.fpgadataflow.multifpga.create_network_metadata import CreateNetworkMetadata
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError


@pytest.mark.multifpga
@pytest.mark.parametrize("communication_kernel", list(MFCommunicationKernel))
def test_every_communication_kernel_has_backend(
    communication_kernel: MFCommunicationKernel,
) -> None:
    """Test that every selectable communication kernel has a registered backend."""
    assert get_backend(communication_kernel).kernel == communication_kernel


@pytest.mark.multifpga
def test_missing_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that looking up a kernel without a backend raises a configuration error, also when
    done indirectly by a transformation of the flow.
    """
    monkeypatch.delitem(backend_module._BACKENDS, MFCommunicationKernel.AURORA)  # noqa: SLF001
    with pytest.raises(FINNMultiFPGAConfigError, match="AURORA has no Multi-FPGA backend"):
        get_backend(MFCommunicationKernel.AURORA)
    mfcfg = MultiFPGAConfiguration(
        communication_kernel=MFCommunicationKernel.AURORA, verbosity=MFVerbosity.NONE
    )
    with pytest.raises(FINNMultiFPGAConfigError, match="AURORA has no Multi-FPGA backend"):
        CreateNetworkMetadata(mfcfg)


@pytest.mark.multifpga
def test_duplicate_registration_raises() -> None:
    """Test that a communication kernel cannot be registered twice."""
    existing = get_backend(MFCommunicationKernel.AURORA)
    with pytest.raises(FINNInternalError, match="AURORA is already registered"):
        register_backend(
            MFCommunicationKernel.AURORA,
            partitioner=existing.partitioner,
            metadata_type=existing.metadata_type,
            prepare_kernels=existing.prepare_kernels,
            modify_link_config=existing.modify_link_config,
        )
    assert get_backend(MFCommunicationKernel.AURORA) is existing
