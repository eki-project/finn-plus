"""Tests for the registry of Multi-FPGA communication backends."""

import pytest

from finn.builder.build_dataflow_config import (
    MFNetworkBackendType,
    MFVerbosity,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga import backend as backend_module
from finn.transformation.fpgadataflow.multifpga.backend import get_backend, register_backend
from finn.transformation.fpgadataflow.multifpga.create_network_metadata import CreateNetworkMetadata
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError


@pytest.mark.multifpga
@pytest.mark.parametrize("network_backend", list(MFNetworkBackendType))
def test_every_network_backend_is_registered(
    network_backend: MFNetworkBackendType,
) -> None:
    """Test that every selectable communication kernel has a registered backend."""
    assert get_backend(network_backend).type == network_backend


@pytest.mark.multifpga
def test_missing_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that looking up a kernel without a backend raises a configuration error, also when
    done indirectly by a transformation of the flow.
    """
    monkeypatch.delitem(backend_module._BACKENDS, MFNetworkBackendType.AURORAFLOW)  # noqa: SLF001
    with pytest.raises(FINNMultiFPGAConfigError, match="AURORAFLOW is not registered"):
        get_backend(MFNetworkBackendType.AURORAFLOW)
    mfcfg = MultiFPGAConfiguration(
        network_backend=MFNetworkBackendType.AURORAFLOW, verbosity=MFVerbosity.NONE
    )
    with pytest.raises(FINNMultiFPGAConfigError, match="AURORAFLOW is not registered"):
        CreateNetworkMetadata(mfcfg)


@pytest.mark.multifpga
def test_duplicate_registration_raises() -> None:
    """Test that a communication kernel cannot be registered twice."""
    existing = get_backend(MFNetworkBackendType.AURORAFLOW)
    with pytest.raises(FINNInternalError, match="AURORAFLOW is already registered"):
        register_backend(
            MFNetworkBackendType.AURORAFLOW,
            partitioner=existing.partitioner,
            metadata_type=existing.metadata_type,
            prepare_kernels=existing.prepare_kernels,
            modify_link_config=existing.modify_link_config,
        )
    assert get_backend(MFNetworkBackendType.AURORAFLOW) is existing
