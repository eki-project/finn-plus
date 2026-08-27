"""Dummy class to mock the HBM interface for simulation purposes."""

from jinja2 import Environment
from pathlib import Path

from finn.util.settings import get_settings


class HBMDummy:
    """Dummy class to mock the HBM interface for simulation purposes."""

    def __init__(self, name: str, addr_width: int, data_width: int, codegen_dir: Path) -> None:
        """Initialize the dummy HBM interface.

        Parameters
        ----------
        name : str
            Name of the HBM interface
        addr_width : int
            Width of the address bus in bits
        data_width : int
            Width of the data bus in bits
        codegen_dir : Path
            Directory where the generated HDL code should be saved
        """
        self.name = name
        self.addr_width = addr_width
        self.data_width = data_width
        self.data_bytes = data_width // 8
        self.codegen_dir = codegen_dir

    def generate_hdl(self) -> None:
        """Render the mock HBM HDL template into the code generation directory."""
        rtlsrc = Path(get_settings().finn_rtllib) / "mock_hbm" / "hdl"
        template_path = rtlsrc / "mock_template.v"

        self.codegen_dir.mkdir(parents=True, exist_ok=True)

        with template_path.open() as f:
            template = f.read()

        template_dict = {
            "TOP_MODULE_NAME": self.name,
            "ADDR_WIDTH": self.addr_width,
            "DATA_WIDTH": self.data_width,
            "DATA_BYTES": self.data_bytes,
        }

        env = Environment()
        rendered_hdl = env.from_string(template).render(**template_dict)
        output_path = self.codegen_dir / f"{self.name}.v"
        output_path.write_text(rendered_hdl)

    def code_generation_ipi(self) -> list[str]:
        """Code generation for IP integration."""
        f = self.codegen_dir / f"{self.name}.v"
        return [
            f"add_files -norecurse {f}",
            f"create_bd_cell -type module -reference {self.name} {self.name}",
        ]

    def code_clk_rst(self) -> list[str]:
        """Code generation for clock and reset signals."""
        return [
            # f"make_bd_pins_external [get_bd_pins {self.name}/ap_clk]",
            # "set_property name ap_clk [get_bd_ports ap_clk_0]",
            # f"make_bd_pins_external [get_bd_pins {self.name}/ap_rst_n]",
            # "set_property name ap_rst_n [get_bd_ports ap_rst_n_0]",
            f"connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins {self.name}/ap_rst_n]",
            f"connect_bd_net [get_bd_ports ap_clk] [get_bd_pins {self.name}/ap_clk]",
        ]
