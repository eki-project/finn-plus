import pytest

from pathlib import Path

from finn.transformation.fpgadataflow.vitis_build import (
    InvalidVitisLinkConfigError,
    VitisLinkConfiguration,
)
from finn.util.basic import make_build_dir


def test_link_config() -> None:
    """Test that the link config is created and generated properly"""
    link_test_dir = Path(make_build_dir("link_test_"))
    target_config = Path(__file__).parent / "example_config.txt"
    generate_config = link_test_dir / "config.txt"
    link_config = VitisLinkConfiguration("testplatform", "", 250)
    link_config.add_cu("comm_kernel", "comm1")
    link_config.add_cu("comm_kernel", "comm2")
    link_config.add_cu("compute", "compute1")
    link_config.add_cu("compute", "compute2")
    link_config.add_cu("compute", "compute3")
    link_config.add_sp("compute1.data_port", "HBM")
    link_config.add_sc("comm1.m_axis", "comm2.s_axis")
    link_config.add_sc("compute1.out", "compute2.in")
    link_config.add_sc("compute2.out", "compute3.in")
    link_config.add_sc("compute3.out", "comm1.s_axis")
    link_config.generate_config(generate_config)
    with (generate_config).open() as f, (target_config).open() as g:
        assert f.read() == g.read()


def test_stops_invalid_config_generation() -> None:
    """Test that you cannot generate an invalid linking config"""
    config_path = Path(make_build_dir("test_invalid_config_")) / "config.txt"

    # Non-existing CUs
    with pytest.raises(InvalidVitisLinkConfigError):
        lc = VitisLinkConfiguration("", "", 100)
        lc.add_cu("A", "a1")
        lc.add_sc("a1:out", "b2:in")
        lc.generate_config(config_path)

    # Wrong formatted sender / receiver ports
    with pytest.raises(InvalidVitisLinkConfigError):
        lc = VitisLinkConfiguration("", "", 100)
        lc.add_cu("A", "a1")
        lc.add_cu("B", "b1")
        lc.add_sc("a1", "b1")
        lc.add_sc("a1:a", "b1:b")
        lc.generate_config(config_path)

    # Two same named CUs
    with pytest.raises(InvalidVitisLinkConfigError):
        lc = VitisLinkConfiguration("", "", 100)
        lc.add_cu("A", "x")
        lc.add_cu("B", "x")
        lc.generate_config(config_path)

    # Two same named CUs, manually changed
    with pytest.raises(InvalidVitisLinkConfigError):
        lc = VitisLinkConfiguration("", "", 100)
        lc.nk.append(("A", "a"))
        lc.nk.append(("B", "a"))
        lc.generate_config(config_path)


def test_script_generation() -> None:
    """Test that the script generation considers every parameter necessary in the
    v++ call"""
    testdir = Path(make_build_dir("test_script_generation_"))
    config_path = testdir / "config.txt"
    script_path = testdir / "run.sh"
    lc = VitisLinkConfiguration("testplatform", "O2", 200)
    lc.add_xo(Path("A.xo"))
    lc.generate_run_script(config_path, script_path)
    assert script_path.exists()
    with script_path.open() as f:
        text = f.read().split("\n")
        assert len(text) == 2
        command = text[1]
        assert command.startswith("v++")
        assert "--target hw" in command
        assert f"--optimize {lc.optimization_level}" in command
        assert "--report_level estimate" in command
        assert f"--config {config_path}" in command
        assert "--link " + " ".join(lc.xo) in command
        assert f"--kernel_frequency {lc.f_mhz}" in command
        assert f"--platform {lc.platform}" in command
