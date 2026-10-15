# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL backend implementation of the uniform-affine requantization operator."""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, Any, Literal, cast

from finn.custom_op.fpgadataflow.requant import NodeAttrTypes, Requant
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import fifo_rtl_files, get_dsp_block, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

# finn-rtllib/requant/hdl sources of the two parameter memory modes
_RTL_SOURCES_EMBEDDED = ["requant.sv", "requant_axi.sv"]
_RTL_SOURCES_DECOUPLED = ["requant_decoupled.sv", "requant_axi_decoupled.sv"]


def _twos(value: int, width: int) -> int:
    """Return the ``width``-bit two's-complement representation of ``value``.

    Args:
        value: Signed integer value.
        width: Bit width of the representation.

    Returns:
        The unsigned integer holding the two's-complement bit pattern.
    """
    return int(value) & ((1 << width) - 1)


def _clog2(n: int) -> int:
    """Ceil(log2(n)) matching Verilog ``$clog2`` for n >= 1.

    Args:
        n: Positive integer.

    Returns:
        The number of bits needed to address ``n`` entries.
    """
    return (n - 1).bit_length()


@register_custom_op
class Requant_rtl(Requant, RTLBackend):
    """RTL backend for Requant operation using finn-rtllib/requant."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Requant.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(
            {
                # memory mode for the scale/bias parameters
                # internal_embedded: constant-folded into the datapath (default)
                # internal_decoupled: streamed from a unified on-chip memstream
                "mem_mode": (
                    "s",
                    False,
                    "internal_embedded",
                    {"internal_embedded", "internal_decoupled"},
                ),
                "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            }
        )
        return my_attrs

    @property
    def mem_mode(self) -> str:
        """Get the memory mode of the scale/bias parameters."""
        return cast("str", self.get_nodeattr("mem_mode"))

    @property
    def mlo_max_iter(self) -> int:
        """Get the number of loop iterations (0 if not inside an MLO loop body)."""
        return cast("int", self.get_nodeattr("mlo_max_iter"))

    def adapt_for_loop_body(self, input_types: "list[LoopBodyInputType]") -> None:  # noqa: ARG002
        """Adapt Requant_rtl for loop body (MLO) execution.

        Requant's default ``mem_mode`` is ``internal_embedded``, which
        constant-folds scale/bias into the SV datapath and therefore cannot vary
        per loop iteration. When LoopRolling flags this node for MLO (it sets
        ``mlo_max_iter`` on the consumer of each per-iteration PARAMETER input),
        switch to ``internal_decoupled`` so scale+bias are streamed from the
        unified param memstream instead (mirrors MVAU_rtl's
        embedded->external_mem switch, but with Requant's own decoupled target
        mode). Gate on the per-node ``mlo_max_iter`` signal rather than the
        positional ``input_types``.

        Args:
            input_types: Loop body input types of this node (unused, see above).
        """
        if self.mlo_max_iter > 0:
            self.set_nodeattr("mem_mode", "internal_decoupled")

    def _mlo_set_bits_padded(self) -> int:
        """Byte-padded width of the per-iteration set-select stream.

        Must match the FINNLoop stream_tap ``$DATA_WIDTH$`` so that the tap's
        ``m_axis`` connects cleanly to the ``in1_V`` set-select slave pin.

        Returns:
            The set-select stream width in bits, rounded up to a multiple of 8.
        """
        data_width = DataType.get_smallest_possible(self.mlo_max_iter).bitwidth()
        return roundup_to_integer_multiple(data_width, 8)

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Interface names for the Requant top module.

        Base case exposes only the activation stream ``in0_V`` and output
        ``out0_V`` (scale/bias are internal to the decoupled hierarchy). In MLO
        mode one extra set-select slave pin ``in1_V`` is appended; it is driven
        by a FINNLoop stream_tap and selects the active parameter set in the
        unified param memstream.

        Returns:
            Dict mapping interface protocols to their interface names (and widths).
        """
        s_axis = [("in0_V", self.get_instream_width_padded(0))]
        if self.mlo_max_iter > 0 and self.mem_mode == "internal_decoupled":
            s_axis.append(("in1_V", self._mlo_set_bits_padded()))
        return {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "s_axis": s_axis,
            "m_axis": [("out0_V", self.get_outstream_width_padded(0))],
            "aximm": [],
            "axilite": [],
            "ap_none": [],
        }

    def _resolve_dsp_version(self, fpgapart: str) -> Literal[3, 2, 1]:
        """Determine DSP version based on FPGA part."""
        match get_dsp_block(fpgapart):
            case "DSP58":
                return 3
            case "DSP48E2":
                return 2
            case _:
                return 1

    def _derive_decoupled_widths(self, model: "ModelWrapper", fpgapart: str) -> dict[str, Any]:
        """Derive the fixed-point stream layout for decoupled mode.

        All widths are consistent with the localparam derivation in
        ``requant_axi_decoupled.sv``.

        Args:
            model: The model holding the scale/bias initializers of this node.
            fpgapart: Target FPGA part, used to resolve the DSP version.

        Returns:
            A dict with the decomposed params plus the per-lane and byte-aligned
            stream widths for the parameter stream and the input/output data
            streams.
        """
        version = self._resolve_dsp_version(fpgapart)
        params = self.decompose_params(model, version)
        s_width = params["s_width"]
        x_width = params["x_width"]
        shift_min = params["shift_min"]
        shift_max = params["shift_max"]

        pe = self.pe
        k = self.get_input_datatype(0).bitwidth()
        n = self.get_output_datatype().bitwidth()

        # In MLO mode the scale stream WIDTH and the core SHIFT_MIN/SHIFT_MAX are
        # baked once at generate_hdl time but must cover *every* loop iteration's
        # params. Since a valid shift is structurally bounded to
        # 0 <= shift <= s_width + x_width + 1 - n (see decompose_params, purely
        # datatype/version-derived), size the window to that worst case instead
        # of the per-layer [min, max]. This makes the layout iteration-invariant
        # and leaf-local (no cross-iteration knowledge needed).
        if self.mlo_max_iter > 0:
            shift_min = 0
            shift_max = s_width + x_width + 1 - n

        bias_width = s_width + x_width
        shift_range = shift_max - shift_min + 1
        shift_width = max(1, _clog2(shift_range)) if shift_range > 1 else 1
        params_lane_width = s_width + shift_width + bias_width

        info = dict(params)
        info.update(
            {
                "version": version,
                "shift_min": shift_min,
                "shift_max": shift_max,
                "bias_width": bias_width,
                "shift_width": shift_width,
                "params_lane_width": params_lane_width,
                "in_stream_width": roundup_to_integer_multiple(pe * k, 8),
                "out_stream_width": roundup_to_integer_multiple(pe * n, 8),
                "params_stream_width": roundup_to_integer_multiple(pe * params_lane_width, 8),
            }
        )
        return info

    def _pack_param_words(self, info: dict[str, Any]) -> list[int]:
        """Pack the decomposed params into per-fold stream words.

        All three fields (scale, shift-offset, bias) are packed into a single
        struct per PE lane. Lane 0 (pe=0) occupies the least significant bits.

        Per-lane layout (LSB to MSB):
            [S_WIDTH-1 : 0]                                        = SCALE (signed mantissa)
            [S_WIDTH+SHIFT_WIDTH-1 : S_WIDTH]                      = T (shift-SHIFT_MIN, unsigned)
            [S_WIDTH+SHIFT_WIDTH+BIAS_WIDTH-1 : S_WIDTH+SHIFT_WIDTH] = BIAS (signed)

        Args:
            info: Stream layout as returned by ``_derive_decoupled_widths``.

        Returns:
            A list of ``CF`` Python integers, one stream word per channel fold.
        """
        pe = self.pe
        cf = self.num_channels // pe

        scale = info["scale"]
        bias = info["bias"]
        shift = info["shift"]
        shift_min = info["shift_min"]
        s_width = info["s_width"]
        shift_width = info["shift_width"]
        bias_width = info["bias_width"]
        params_lane_width = info["params_lane_width"]

        params_words = []
        for c in range(cf):
            p_word = 0
            for p in range(pe):
                t_off = int(shift[p][c]) - shift_min
                # Pack: { BIAS[BIAS_WIDTH], T[SHIFT_WIDTH], SCALE[S_WIDTH] }
                lane = (
                    (_twos(bias[p][c], bias_width) << (s_width + shift_width))
                    | (_twos(t_off, shift_width) << s_width)
                    | _twos(scale[p][c], s_width)
                )
                p_word |= lane << (p * params_lane_width)
            params_words.append(p_word)
        return params_words

    @staticmethod
    def _write_memblock(path: Path, words: list[int], stream_width: int) -> None:
        """Write memstream init .dat: one hex word per line, zero-padded.

        Args:
            path: Output file path.
            words: Stream words to write.
            stream_width: Stream width in bits (determines the zero-padding).
        """
        hex_digits = stream_width // 4
        path.write_text("".join(f"{int(w):0{hex_digits}x}\n" for w in words))

    def _write_params(self, model: "ModelWrapper", path: str | Path, fpgapart: str) -> list[int]:
        """Emit ``params_memblock.dat`` into ``path`` and return the packed words.

        Args:
            model: The model holding the scale/bias initializers of this node.
            path: Directory in which the memstream init file is written.
            fpgapart: Target FPGA part, used to resolve the DSP version.

        Returns:
            The packed ``params_words`` list, one word per channel fold.
        """
        info = self._derive_decoupled_widths(model, fpgapart)
        params_words = self._pack_param_words(info)
        self._write_memblock(
            Path(path) / "params_memblock.dat",
            params_words,
            info["params_stream_width"],
        )
        return params_words

    def generate_params(
        self, model: "ModelWrapper", path: str | Path, fpgapart: str | None = None
    ) -> None:
        """Emit the unified decoupled memstream init file for the current params.

        Writes ``params_memblock.dat`` into ``path`` for the scale/bias
        currently attached to this node in ``model``. Used both by standalone
        decoupled generation and, per loop iteration, by
        ``FINNLoop.generate_params`` (which sets the per-iteration initializers
        before calling this).

        Args:
            model: The model holding the scale/bias initializers of this node.
            path: Directory in which the memstream init file is written.
            fpgapart: Target FPGA part. Required to resolve the DSP version so the
                fixed-point layout matches the elaborated core.

        Raises:
            FINNInternalError: If ``fpgapart`` is not given.
        """
        if fpgapart is None:
            raise FINNInternalError("Requant_rtl.generate_params requires fpgapart")
        self._write_params(model, path, fpgapart)

    def generate_hdl(
        self, model: "ModelWrapper", fpgapart: str, clk: float  # noqa: ARG002
    ) -> None:
        """Generate RTL code for the requant operation."""
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        if code_gen_dir == "":
            code_gen_dir = str(make_build_dir("requant_rtl_ipgen_"))
            self.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)

        mem_mode = self.mem_mode
        if mem_mode == "internal_embedded":
            self._generate_hdl_embedded(model, fpgapart, Path(code_gen_dir))
        elif mem_mode == "internal_decoupled":
            self._generate_hdl_decoupled(model, fpgapart, Path(code_gen_dir))
        else:
            raise FINNInternalError(f"{self.onnx_node.name}: unsupported mem_mode {mem_mode}.")

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stitch ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _generate_hdl_embedded(
        self, model: "ModelWrapper", fpgapart: str, code_gen_path: Path
    ) -> None:
        """Embedded mode: constant-fold scale/bias into the SV datapath.

        Args:
            model: The model holding the scale/bias initializers of this node.
            fpgapart: Target FPGA part, used to resolve the DSP version.
            code_gen_path: Directory in which the generated HDL is written.
        """
        pe = self.pe
        num_channels = self.num_channels
        cf = num_channels // pe  # Channel fold

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()
        k = idt.bitwidth()  # Input precision
        n = odt.bitwidth()  # Output precision

        version = self._resolve_dsp_version(fpgapart)

        # Get scale and bias from model
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        # Broadcast scalar scale/bias to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: the RTL expects [PE][CF] layout
        scale_reshaped = scale.reshape(cf, pe).T  # [PE][CF]
        bias_reshaped = bias.reshape(cf, pe).T  # [PE][CF]

        def format_sv_array(arr: np.ndarray) -> str:
            """Format a 2D numpy array as a SystemVerilog array literal."""
            lines = []
            for pe_idx in range(arr.shape[0]):
                # Fixed-point notation with 6 decimal places (shortreal is 32-bit float)
                row = ", ".join(f"{float(v):.6f}" for v in arr[pe_idx])
                lines.append("'{" + row + "}")
            return "'{" + ", ".join(lines) + "}"

        scales_sv = format_sv_array(scale_reshaped)
        biases_sv = format_sv_array(bias_reshaped)

        # Calculate stream widths (byte-aligned)
        in_stream_width = ((pe * k + 7) // 8) * 8
        out_stream_width = ((pe * n + 7) // 8) * 8

        top_module_name = self.get_verilog_top_module_name()
        rtllib_dir = Path(get_settings().finn_rtllib) / "requant" / "hdl"

        # Generate SystemVerilog implementation module (with _impl suffix)
        sv_code = (rtllib_dir / "requant_wrapper_template.sv").read_text()
        for placeholder, value in {
            "$TOP_MODULE_NAME$": top_module_name,
            "$VERSION$": str(version),
            "$K$": str(k),
            "$N$": str(n),
            "$C$": str(num_channels),
            "$PE$": str(pe),
            "$SCALES$": scales_sv,
            "$BIASES$": biases_sv,
            "$SIGNED_OUT$": str(int(odt.signed())),
            "$IN_STREAM_WIDTH$": str(in_stream_width),
            "$OUT_STREAM_WIDTH$": str(out_stream_width),
        }.items():
            sv_code = sv_code.replace(placeholder, value)
        (code_gen_path / f"{top_module_name}_impl.sv").write_text(sv_code)

        # Generate Verilog stub wrapper (for IP packaging - must be .v)
        v_code = (rtllib_dir / "requant_wrapper_template.v").read_text()
        for placeholder, value in {
            "$TOP_MODULE_NAME$": top_module_name,
            "$IN_STREAM_WIDTH$": str(in_stream_width),
            "$OUT_STREAM_WIDTH$": str(out_stream_width),
        }.items():
            v_code = v_code.replace(placeholder, value)
        (code_gen_path / f"{top_module_name}.v").write_text(v_code)

        self.set_nodeattr("gen_top_module", top_module_name)

    def _generate_hdl_decoupled(
        self, model: "ModelWrapper", fpgapart: str, code_gen_path: Path
    ) -> None:
        """Decoupled mode: stream scale/shift/bias from a unified param memstream.

        The float->fixed-point decomposition is done in Python
        (``decompose_params``); the resulting per-channel words are packed into
        a single struct-based stream and emitted as one memstream init file, and
        the compute core is elaborated with the worst-case SHIFT_MIN/SHIFT_MAX
        window instead of embedded params.

        Args:
            model: The model holding the scale/bias initializers of this node.
            fpgapart: Target FPGA part, used to resolve the DSP version.
            code_gen_path: Directory in which the generated HDL is written.
        """
        info = self._derive_decoupled_widths(model, fpgapart)

        pe = self.pe
        num_channels = self.num_channels
        cf = num_channels // pe
        k = self.get_input_datatype(0).bitwidth()
        n = self.get_output_datatype().bitwidth()

        top_module_name = self.get_verilog_top_module_name()
        rtllib_dir = Path(get_settings().finn_rtllib) / "requant" / "hdl"

        odt = self.get_output_datatype()
        subst = {
            "$TOP_MODULE_NAME$": top_module_name,
            "$VERSION$": str(info["version"]),
            "$K$": str(k),
            "$N$": str(n),
            "$C$": str(num_channels),
            "$PE$": str(pe),
            "$SHIFT_MIN$": str(info["shift_min"]),
            "$SHIFT_MAX$": str(info["shift_max"]),
            "$SIGNED_OUT$": str(int(odt.signed())),
        }

        # Verilog wrapper (.v) for IP packaging
        v_code = (rtllib_dir / "requant_wrapper_decoupled_template.v").read_text()
        for key, val in subst.items():
            v_code = v_code.replace(key, val)
        (code_gen_path / f"{top_module_name}.v").write_text(v_code)

        self.set_nodeattr("gen_top_module", top_module_name)

        # Emit memstream init file (.dat) via the shared generate_params path,
        # plus the packed words as .npy for standalone rtlsim.
        params_words = self._write_params(model, code_gen_path, fpgapart)
        np.save(
            str(code_gen_path / "params_words.npy"),
            np.array(params_words, dtype=object),
        )

        # Emit the unified memstream wrapper (params = scale + shift + bias)
        node_name = self.onnx_node.name
        self.generate_hdl_memstream(
            fpgapart,
            name=node_name + "_params",
            depth=cf,
            width=info["params_stream_width"],
            init_file=str(code_gen_path / "params_memblock.dat"),
            ram_style="auto",
        )

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return list of RTL files needed for this node."""
        rtllib_dir = str(Path(get_settings().finn_rtllib) / "requant" / "hdl") + "/"
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))

        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        if top_module == "":
            top_module = self.get_verilog_top_module_name()

        if self.mem_mode == "internal_decoupled":
            rtl_files = [rtllib_dir + f for f in _RTL_SOURCES_DECOUPLED] + [
                # generated Verilog wrapper (directly instantiates the SV core)
                str(code_gen_dir / f"{top_module}.v"),
            ]
        else:
            rtl_files = [rtllib_dir + f for f in _RTL_SOURCES_EMBEDDED] + [
                # generated SystemVerilog impl + Verilog stub wrapper
                str(code_gen_dir / f"{top_module}_impl.sv"),
                str(code_gen_dir / f"{top_module}.v"),
            ]
        rtl_files += fifo_rtl_files()

        if abspath:
            return rtl_files
        return [Path(f).name for f in rtl_files]

    def code_generation_ipi(self) -> list[str]:
        """Return the Vivado IPI Tcl commands that instantiate this node."""
        if self.mem_mode == "internal_decoupled":
            return self._code_generation_ipi_decoupled()

        sourcefiles = self.get_rtl_file_list(abspath=True)
        top_module = cast("str", self.get_nodeattr("gen_top_module"))

        cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
        cmd += [f"create_bd_cell -type module -reference {top_module} {self.onnx_node.name}"]
        return cmd

    def _code_generation_ipi_decoupled(self) -> list[str]:
        """Instantiate the decoupled core plus the unified param memstream.

        Returns:
            List of Vivado IPI Tcl commands.
        """
        node_name = self.onnx_node.name
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

        cmd = []

        # Hierarchy with external data in/out pins (params are internal)
        cmd.append(f"create_bd_cell -type hier {node_name}")
        cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/ap_clk")
        cmd.append(f"create_bd_pin -dir I -type rst /{node_name}/ap_rst_n")
        cmd.append(
            "create_bd_intf_pin -mode Master "
            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/out0_V"
        )
        cmd.append(
            "create_bd_intf_pin -mode Slave "
            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/in0_V"
        )

        # MLO: expose per-iteration set-select slave pin for the unified param
        # memstream (in1_V). This carries the memstream set index driven by the
        # FINNLoop stream-tap graph; standalone (SETS=1) leaves it absent.
        mlo = self.mlo_max_iter > 0
        if mlo:
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/in1_V"
            )

        # Compute core
        cmd.extend(f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True))
        cmd.append(f"create_bd_cell -type module -reference {top_module} /{node_name}/{node_name}")
        cmd.append(
            f"connect_bd_net [get_bd_pins {node_name}/ap_clk] "
            f"[get_bd_pins {node_name}/{node_name}/ap_clk]"
        )
        cmd.append(
            f"connect_bd_net [get_bd_pins {node_name}/ap_rst_n] "
            f"[get_bd_pins {node_name}/{node_name}/ap_rst_n]"
        )
        cmd.append(
            f"connect_bd_intf_net [get_bd_intf_pins {node_name}/in0_V] "
            f"[get_bd_intf_pins {node_name}/{node_name}/in0_V]"
        )
        cmd.append(
            f"connect_bd_intf_net [get_bd_intf_pins {node_name}/out0_V] "
            f"[get_bd_intf_pins {node_name}/{node_name}/out0_V]"
        )

        # Shared memstream sources
        axi_dir = str(Path(get_settings().finn_rtllib) / "axi" / "hdl") + "/"
        ms_dir = str(Path(get_settings().finn_rtllib) / "memstream" / "hdl") + "/"
        for f in [axi_dir + "axilite.sv", ms_dir + "memstream_axi.sv", ms_dir + "memstream.sv"]:
            cmd.append(f"add_files -norecurse {f}")

        # Unified param memstreamer feeding the core's s_params_V port.
        # In MLO mode the external set-select pin (in1_V) drives the
        # memstreamer's s_axis_0; standalone leaves s_axis_0 unwired (SETS=1).
        suffix = "_params"
        core_port = "s_params_V"
        ext_pin = "in1_V"

        # Discover the generated wrapper by suffix rather than reconstructing
        # from node_name: inside a FINNLoop body the node is renamed between
        # generate_hdl (files carry the loop-prefixed name) and ipi time, so
        # the module name lives in gen_top_module / the filename, not in
        # self.onnx_node.name (mirrors MVAU's code_generation_ipi).
        file_suffix = suffix + "_memstream_wrapper.v"
        matches = [f.name for f in Path(code_gen_dir).iterdir() if f.name.endswith(file_suffix)]
        if not matches:
            raise FINNInternalError(
                f"Requant decoupled: could not find {file_suffix} in {code_gen_dir}"
            )
        wrapper_fname = matches[-1]
        wrapper_file = str(Path(code_gen_dir) / wrapper_fname)
        cmd.append(f"add_files -norecurse {wrapper_file}")
        strm_mod = wrapper_fname[:-2]
        strm_inst = node_name + suffix + "_wstrm"
        cmd.append(f"create_bd_cell -type hier -reference {strm_mod} /{node_name}/{strm_inst}")
        cmd.append(
            f"connect_bd_net [get_bd_pins {node_name}/ap_clk] "
            f"[get_bd_pins {node_name}/{strm_inst}/ap_clk]"
        )
        cmd.append(
            f"connect_bd_net [get_bd_pins {node_name}/ap_clk] "
            f"[get_bd_pins {node_name}/{strm_inst}/ap_clk2x]"
        )
        cmd.append(
            f"connect_bd_net [get_bd_pins {node_name}/ap_rst_n] "
            f"[get_bd_pins {node_name}/{strm_inst}/ap_rst_n]"
        )
        cmd.append(
            f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{strm_inst}/m_axis_0] "
            f"[get_bd_intf_pins {node_name}/{node_name}/{core_port}]"
        )
        if mlo:
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{ext_pin}] "
                f"[get_bd_intf_pins {node_name}/{strm_inst}/s_axis_0]"
            )

        cmd.append("save_bd_design")
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node, using RTL simulation if exec_mode is rtlsim."""
        mode = self.get_nodeattr("exec_mode")
        if mode != "rtlsim":
            # Use base class Python execution
            Requant.execute_node(self, context, graph)
            return

        node = self.onnx_node
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        decoupled = self.mem_mode == "internal_decoupled"

        # Process input 0 (data tensor)
        inp = node.input[0]
        exp_ishape = tuple(self.get_normal_input_shape(0))
        folded_ishape = self.get_folded_input_shape(0)
        inp_val = context[inp]
        if str(inp_val.dtype) != "float32":
            raise FINNInternalError(f"{node.name}: input datatype is not float32")
        if inp_val.shape != exp_ishape:
            raise FINNInternalError(f"{node.name}: input shape {inp_val.shape} != {exp_ishape}")
        export_idt = self.get_input_datatype(0)

        reshaped_input = inp_val.reshape(folded_ishape)
        np.save(str(Path(code_gen_dir) / "input_0.npy"), reshaped_input)
        nbits = self.get_instream_width(0)
        rtlsim_inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)

        io_dict = {
            "inputs": {"in0": rtlsim_inp},
            "outputs": {"out0": []},
        }

        if decoupled:
            # A SETS>1 (MLO) memstream needs a per-iteration set-select
            # index on s_axis_0, which the standalone .npy streaming path
            # cannot supply. Such nodes must be executed via the stitched
            # FINNLoop rtlsim, which drives the set index in hardware.
            if self.mlo_max_iter != 0:
                raise FINNUserError(
                    f"{node.name}: standalone rtlsim cannot drive the set-select index; "
                    "a SETS>1 (MLO) Requant must be executed via FINNLoop."
                )
            # Feed the unified parameter stream alongside the data.
            # The memstream cycles CF words; replicate per input vector.
            params_words = list(
                np.load(str(Path(code_gen_dir) / "params_words.npy"), allow_pickle=True)
            )
            num_vectors = int(np.prod(self.num_input_vectors))
            io_dict["inputs"]["s_params"] = [int(w) for w in params_words] * num_vectors

        sim = self.get_rtlsim()
        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict)
        self.close_rtlsim(sim)

        # Process output
        rtlsim_output = io_dict["outputs"]["out0"]
        odt = self.get_output_datatype(0)
        target_bits = odt.bitwidth()
        packed_bits = self.get_outstream_width(0)
        out_npy_path = f"{code_gen_dir}/output.npy"
        out_shape = self.get_folded_output_shape(0)
        rtlsim_output_to_npy(rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits)

        # Load and reshape output
        exp_oshape = tuple(self.get_normal_output_shape(0))
        output = np.load(out_npy_path)
        output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
        context[node.output[0]] = output

        if context[node.output[0]].shape != exp_oshape:
            raise FINNInternalError(
                f"{node.name}: output shape {context[node.output[0]].shape} != {exp_oshape}"
            )
