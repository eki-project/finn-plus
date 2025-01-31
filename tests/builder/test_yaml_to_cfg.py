import pytest

from finn.builder.build_dataflow_config import DataflowOutputType
from finn.builder.yaml_to_cfg import variant_from_str


@pytest.mark.yaml_builder
@pytest.mark.parametrize(
    "variant_and_expect_exist",
    [("STITCHED_IP", True), ("ESTIMATE_REPORTS", True), ("ABCD", False), ("", False)],
)
def test_yaml_to_enum(variant_and_expect_exist: tuple[str, bool]):
    assert (
        variant_from_str(DataflowOutputType, variant_and_expect_exist[0]) is not None
        or variant_and_expect_exist[1] is False
    )
