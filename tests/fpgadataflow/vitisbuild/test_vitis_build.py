import pytest

import onnx.helper as oh
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.vitis_build import VitisLinkConfiguration
from finn.util.basic import make_build_dir
from finn.util.exception import FINNVitisLinkConfigError
from finn.util.logging import log


def get_empty_modelwrapper() -> ModelWrapper:
    """Return an empty modelwrapper, with a graph with a single node
    without any graph or node inputs/outputs.
    This can be useful for testing metadata_prop related functionality.
    """
    identity_node = oh.make_node("Identity", [], [])
    graph = oh.make_graph([identity_node], "test_graph", [], [])
    onnx_model = oh.make_model(graph)
    return ModelWrapper(onnx_model)


def test_link_config() -> None:
    """Test that the link config is created and generated properly."""
    target_config = Path(__file__).parent / "example_config.txt"
    link_config = VitisLinkConfiguration(
        Path(make_build_dir("link_test_")) / "config.txt", 100, "", ""
    )
    link_config.add_cu("comm_kernel", "comm1")
    link_config.add_cu("comm_kernel", "comm2")
    link_config.add_cu("compute", "compute1")
    link_config.add_cu("compute", "compute2")
    link_config.add_cu("compute", "compute3")
    link_config.add_sp("compute1.data_port", "HBM[0]")
    link_config.add_sc("comm1.m_axis", "comm2.s_axis")
    link_config.add_sc("compute1.out", "compute2.in")
    link_config.add_sc("compute2.out", "compute3.in")
    link_config.add_sc("compute3.out", "comm1.s_axis")
    link_config.generate_config()
    with (link_config.config_path).open() as f, (target_config).open() as g:
        assert f.read() == g.read()


def test_stops_invalid_config_generation() -> None:
    """Test that you cannot generate an invalid linking config."""
    # Disable logger to prevent the CI logs from being full of errors due to this test
    log.propagate = False

    # Non-existing CUs
    with pytest.raises(FINNVitisLinkConfigError):
        lc = VitisLinkConfiguration(Path(make_build_dir("link_test_")) / "config.txt", 100, "", "")
        lc.add_cu("A", "a1")
        lc.add_sc("a1:out", "b2:in")
        lc.generate_config()

    # Wrong formatted sender / receiver ports
    with pytest.raises(FINNVitisLinkConfigError):
        lc = VitisLinkConfiguration(Path(make_build_dir("link_test_")) / "config.txt", 100, "", "")
        lc.add_cu("A", "a1")
        lc.add_cu("B", "b1")
        lc.add_sc("a1", "b1")
        lc.add_sc("a1:a", "b1:b")
        lc.generate_config()

    # Two same named CUs
    with pytest.raises(FINNVitisLinkConfigError):
        lc = VitisLinkConfiguration(Path(make_build_dir("link_test_")) / "config.txt", 100, "", "")
        lc.add_cu("A", "x")
        lc.add_cu("B", "x")
        lc.generate_config()

    # Two same named CUs, manually changed
    with pytest.raises(FINNVitisLinkConfigError):
        lc = VitisLinkConfiguration(Path(make_build_dir("link_test_")) / "config.txt", 100, "", "")
        lc.nk.append(("A", "a"))
        lc.nk.append(("B", "a"))
        lc.generate_config()

    # Re-enable logger
    log.propagate = True


def test_script_generation() -> None:
    """Test that the script generation considers every parameter necessary in the
    v++ call.
    """
    log.propagate = False
    lc = VitisLinkConfiguration(
        Path(make_build_dir("link_test_")) / "config.txt", 100, "O2", "testplatform"
    )
    xo = lc.config_path.parent / "A.xo"
    xo.write_text("")
    lc.add_xo(xo)
    lc.generate_run_script()
    assert lc.run_script_path.exists()
    with lc.run_script_path.open() as f:
        text = f.read().split("\n")
        assert len(text) == 2
        command = text[1]
        assert command.startswith("v++")
        assert "--target hw" in command
        assert f"--optimize {lc.optimization_level}" in command
        assert "--report_level estimate" in command
        assert f"--config {lc.config_path}" in command
        assert "--link " + " ".join(str(xo_path) for xo_path in lc.xo) in command
        assert f"--kernel_frequency {lc.f_mhz}" in command
        assert f"--platform {lc.platform}" in command
    log.propagate = True
