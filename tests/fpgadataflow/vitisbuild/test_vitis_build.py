import pytest

from pathlib import Path

from finn.transformation.fpgadataflow.vitis_build import VitisLinkConfiguration
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
    link_config.add_sp("compute1:data_port", "HBM")
    link_config.add_sc("comm1.m_axis", "comm2.s_axis")
    link_config.add_sc("compute1.out", "compute2.in")
    link_config.add_sc("compute2.out", "compute3.in")
    link_config.add_sc("compute3.out", "comm1.s_axis")
    link_config.generate_config(generate_config)
    with (generate_config).open() as f, (target_config).open() as g:
        assert f.read() == g.read()
