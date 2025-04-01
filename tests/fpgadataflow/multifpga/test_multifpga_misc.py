import pytest

from finn.transformation.fpgadataflow.multifpga_kernel_preparation import PrepareAuroraFlow


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("from_xo", ["aurora_flow_0.xo", "aurora_flow_1.xo"])
@pytest.mark.parametrize("to_xo", ["tested.xo"])
@pytest.mark.parametrize("args", ["", "FIFO_WIDTH=32 TX_FIFO_SIZE=8192 RX_FIFO_SIZE=65536"])
def test_aurora_package_single(from_xo: str, to_xo: str, args: str) -> None:
    """Not only checks that packaging works, but also that the names output by aurora flow
    stay the same"""
    prep = PrepareAuroraFlow()
    assert prep.aurora_storage.exists()
    moved = prep.package(args, from_xo, to_xo)
    assert moved.exists()
