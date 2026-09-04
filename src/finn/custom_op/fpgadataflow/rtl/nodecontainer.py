"""Custom RTL op for wrapping multiple DNN bodies in a single NodeContainer."""

import json
import math
import numpy as np
import numpy.typing as npt
import os
import shutil
from collections.abc import Sequence
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name, qonnx_make_model, roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

# Value types accepted/returned by the base ``get_nodeattr``/``set_nodeattr``.
BaseNodeAttrValue = int | float | str | bool | npt.NDArray | list[str | int | float] | None
# ``set_nodeattr`` on this op additionally accepts a graph value for the ``body_*``
# attributes (as a ``ModelWrapper`` or a raw ``GraphProto``).
SetNodeAttrValue = ModelWrapper | GraphProto | BaseNodeAttrValue
# Shape of the dict returned by ``get_nodeattr_types``: attribute name ->
# (dtype, required, default[, allowed_values]). Must match the base classes.
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | npt.NDArray | list]
    | tuple[str, bool, int | float | str | bool | npt.NDArray | list, set | None],
]


@register_custom_op
class NodeContainer(RTLBackend, HWCustomOp):
    """Container node holding several DNN bodies that share one hardware instance.

    Some functions are (partially) copied from FINNLoop.
    Currently unsupported features:
        - Multiple inputs/outputs
        - FIFO sizing
        - Minimizing bitwitdh
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize NodeContainer and read the number of body graphs."""
        super().__init__(onnx_node, **kwargs)
        bodies_attr = get_by_name(self.onnx_node.attribute, "bodies")
        self.bodies = bodies_attr.i if bodies_attr is not None else 0

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return attribute type definitions including per-body graph attributes."""
        b: NodeAttrTypes = {f"body_{i}": ("g", True, "") for i in range(self.bodies)}
        my_attrs: NodeAttrTypes = {
            "bodies": ("i", True, 0),
            "multi_dnn_type": ("s", True, ""),
            "pblock": ("s", False, ""),
            # Width of the tUSER field on the external AXI-Stream interfaces of the
            # DFX Wrapper. 0 means "auto" (computed from ceil(log2(bodies)) at BD
            # generation time). Only meaningful for partial_reconfiguration type.
            "tuser_width": ("i", False, 0),
            **b,
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_nodeattr(self, name: str) -> BaseNodeAttrValue:
        """Get a node attribute by name, handling graph-type attributes.

        Note: the graph-typed ``body_*`` attributes are returned as a
        ``ModelWrapper``. The return type is kept identical to the base method;
        callers that need the ``ModelWrapper`` cast it themselves (or go through
        :meth:`_get_body`).
        """
        try:
            (dtype, req, def_val, _allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    return cast("BaseNodeAttrValue", ModelWrapper(qonnx_make_model(ret)))
                return super().get_nodeattr(name)
            if req:
                raise FINNUserError(
                    f"Required attribute {name} unspecified in a {self.onnx_node.op_type} node"
                )
            # not set, return default value
            return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name) from None

    def set_nodeattr(self, name: str, value: SetNodeAttrValue) -> None:
        """Set a node attribute by name, handling graph-type attributes.

        In addition to the base value types, the graph-typed ``body_*`` attributes
        may be set from a ``ModelWrapper`` or a raw ``GraphProto``.
        """
        try:
            (dtype, _req, _def_val, _allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None and dtype == "g":
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if isinstance(value, ModelWrapper):
                    value = value.model.graph
                if not isinstance(value, GraphProto):
                    raise FINNInternalError(
                        "Value for graph attribute must be a GraphProto or ModelWrapper"
                    )
                attr.g.CopyFrom(value)
            else:
                super().set_nodeattr(name, cast("BaseNodeAttrValue", value))
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name) from None

    @property
    def multi_dnn_type(self) -> str:
        """Return the multi-DNN container flavour.

        Either ``selectable_weights`` or ``partial_reconfiguration``.
        """
        return cast("str", self.get_nodeattr("multi_dnn_type"))

    @property
    def num_bodies(self) -> int:
        """Return the number of body graphs held by this container."""
        return cast("int", self.get_nodeattr("bodies"))

    @property
    def code_gen_dir_ipgen(self) -> str:
        """Return the IP-generation code directory."""
        return cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

    def _get_body(self, index: int) -> ModelWrapper:
        """Return the body model with the given index as a ``ModelWrapper``."""
        return cast("ModelWrapper", self.get_nodeattr(f"body_{index}"))

    def _get_reference_body(self) -> ModelWrapper:
        """Return the first body model (body_0) as the reference."""
        # Return the first body
        # For the selectable_weights case we can assume that all bodies have the same structure
        return self._get_body(0)

    def _get_reference_node(self) -> NodeProto:
        """Return the first node of the first body as the reference node."""
        # Return the first node of the first body
        # For the selectable_weights case we can assume that all bodies have one node
        # And that they have the same structure (folding, datatype, etc)
        body = self._get_reference_body()
        return body.graph.node[0]

    def _check_types(self, node: NodeProto, types: list[str]) -> bool:
        """Return True if node.op_type matches any entry in the types list."""
        node_type = node.op_type
        for t in types:
            if t.endswith(("_hls", "_rtl")):
                if node_type == t:
                    return True
            elif node_type.startswith(t):
                return True
        return False

    def _require_single_io(self, ind: int) -> None:
        """Raise if an input/output index other than 0 is requested."""
        if ind != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: NodeContainer currently supports only a single "
                f"input/output, but index {ind} was requested"
            )

    def get_normal_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the unfolded input shape."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[0]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_normal_input_shape(ind)

    def get_normal_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the unfolded output shape."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_normal_output_shape(ind)

    def get_folded_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the folded input shape."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[0]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_folded_input_shape(ind)

    def get_folded_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the folded output shape."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_folded_output_shape(ind)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer output datatype (not applicable for NodeContainer)."""

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return the input datatype."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        first_inst = cast("HWCustomOp", getCustomOp(body.graph.node[0]))
        return first_inst.get_input_datatype(ind)

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        """Return the output datatype."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        last_inst = cast("HWCustomOp", getCustomOp(body.graph.node[-1]))
        return last_inst.get_output_datatype(ind)

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the input stream width."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[0]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_instream_width(ind)

    def get_exp_cycles(self) -> int:
        """Return expected cycle count based on the multi_dnn_type attribute."""
        multi_dnn_type = self.multi_dnn_type
        if multi_dnn_type == "selectable_weights":
            body = self._get_reference_body()
            node = body.graph.node[-1]
            inst = cast("HWCustomOp", getCustomOp(node))
            return inst.get_exp_cycles()
        if multi_dnn_type == "partial_reconfiguration":
            exp_cycles = 0
            for i in range(self.bodies):
                temp_exp_cycles = 0
                body = self._get_body(i)
                for node in body.graph.node:
                    inst = cast("HWCustomOp", getCustomOp(node))
                    temp_exp_cycles += inst.get_exp_cycles()
                exp_cycles = max(exp_cycles, temp_exp_cycles)
            return exp_cycles
        raise FINNUserError(
            f"{self.onnx_node.name}: unsupported multi_dnn_type {multi_dnn_type!r}, expected "
            f"'selectable_weights' or 'partial_reconfiguration'"
        )

    def get_outstream_width(self, ind: int = 0) -> int:
        """Return the output stream width."""
        self._require_single_io(ind)  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_outstream_width(ind=ind)

    def generate_hdl_memstream(self, fpgapart: str, pumped_memory: int = 0) -> None:
        """Delegate memstream HDL generation to the reference node's implementation."""
        inst = cast("HWCustomOp", getCustomOp(self._get_reference_node()))
        inst.set_nodeattr("bodies", self.num_bodies)
        inst.set_nodeattr("code_gen_dir_ipgen", self.code_gen_dir_ipgen)
        if pumped_memory in inst.get_nodeattr_types():
            pumped_memory = cast("int", inst.get_nodeattr("pumpedMemory"))
        inst.generate_hdl_memstream(fpgapart, pumped_memory)

    def _rename_body_params(
        self, node: NodeProto, inst: HWCustomOp, path: Path, index: int
    ) -> None:
        """Rename the parameter files a single body just generated so they survive."""
        if self._check_types(node, ["MVAU", "Elementwise", "Thresholding_hls", "VVAU"]):
            # rename so it doesn't get overwritten
            shutil.move(path / "memblock.dat", path / f"{node.op_type}_memblock_{index}.dat")
        elif self._check_types(node, ["Thresholding_rtl"]):
            # get all generated Thresholding dat files
            pe = cast("int", inst.get_nodeattr("PE"))
            output_data_type = cast("str", inst.get_nodeattr("outputDataType"))
            o_bitwidth = DataType[output_data_type].bitwidth()
            for stage in range(o_bitwidth):
                for pe_value in range(pe):
                    param_path = path / f"{node.name}_threshs_{pe_value}_{stage}.dat"
                    new_param_file = param_path.with_name(
                        param_path.stem + "_i" + str(index) + param_path.suffix
                    )
                    shutil.move(param_path, new_param_file)
        else:
            raise FINNUserError(
                f"{self.onnx_node.name}: parameter generation is not supported for body node "
                f"{node.name} of type {node.op_type}"
            )

    def _concat_threshold_params(
        self, reference_node: NodeProto, reference_inst: HWCustomOp, path: Path, num_bodies: int
    ) -> None:
        """Concatenate the per-body RTL threshold files back into one file per PE/stage."""
        pe = cast("int", reference_inst.get_nodeattr("PE"))
        output_data_type = cast("str", reference_inst.get_nodeattr("outputDataType"))
        o_bitwidth = DataType[output_data_type].bitwidth()
        for stage in range(o_bitwidth):
            for pe_value in range(pe):
                param_file = path / f"{reference_node.name}_threshs_{pe_value}_{stage}.dat"
                with param_file.open("w") as outfile:
                    for i in range(num_bodies):
                        body_file = path / (
                            f"{reference_node.name}_threshs_{pe_value}_{stage}_i{i}.dat"
                        )
                        with body_file.open() as infile:
                            cnt = 0
                            hex_len = 0
                            for line in infile:
                                if cnt == 0:
                                    hex_len = len(line.strip())
                                cnt += 1
                                outfile.write(line)
                            # is power of 2?
                            if (cnt & (cnt - 1)) != 0:
                                # pad with max value
                                next_pow2 = 2 ** math.ceil(math.log2(cnt))
                                pad_val = 2**o_bitwidth - 1
                                for _ in range(next_pow2 - cnt):
                                    # write out as hex of len hex_len
                                    outfile.write(hex(pad_val)[2:].zfill(hex_len) + "\n")
                        body_file.unlink()

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:  # noqa: ARG002
        """Write weight parameter files for all bodies into the given path."""
        num_bodies = self.num_bodies
        param_dir = Path(path)
        reference_node = self._get_reference_node()
        reference_inst = cast("HWCustomOp", getCustomOp(reference_node))

        for i in range(num_bodies):
            body = self._get_body(i)
            node = body.graph.node[-1]
            inst = cast("HWCustomOp", getCustomOp(node))
            inst.set_nodeattr("bodies", num_bodies)
            inst.generate_params(body, path)
            self._rename_body_params(node, inst, param_dir, i)

        if self._check_types(reference_node, ["MVAU", "Elementwise", "Thresholding_hls", "VVAU"]):
            # concatinate all .dat files together
            with (param_dir / "memblock.dat").open("w") as outfile:
                for i in range(num_bodies):
                    memblock_file = param_dir / f"{reference_node.op_type}_memblock_{i}.dat"
                    with memblock_file.open() as infile:
                        for line in infile:
                            outfile.write(line)
                    memblock_file.unlink()
        elif self._check_types(reference_node, ["Thresholding_rtl"]):
            # concatinate all .dat files together
            self._concat_threshold_params(reference_node, reference_inst, param_dir, num_bodies)

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate HDL for the NodeContainer based on multi_dnn_type."""
        multi_dnn_type = self.multi_dnn_type
        if multi_dnn_type != "selectable_weights":
            raise FINNUserError(
                f"{self.onnx_node.name}: HDL generation is only supported for multi_dnn_type "
                f"'selectable_weights', got {multi_dnn_type!r}"
            )
        self.generate_hdl_memstream(fpgapart)
        self.generate_params(model, self.code_gen_dir_ipgen)
        self.generate_hdl_stream_tap()

        code_gen_dir_ipgen = Path(self.code_gen_dir_ipgen)
        items = list(code_gen_dir_ipgen.iterdir())
        tmpdir = code_gen_dir_ipgen / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        for item_path in items:
            shutil.move(item_path, tmpdir / item_path.name)

        # Generate reference node hw and copy needed files to correct location
        reference_node = self._get_reference_node()
        reference_inst = cast("HWCustomOp", getCustomOp(reference_node))

        has_mem_mode = "mem_mode" in reference_inst.get_nodeattr_types()
        memode = reference_inst.get_nodeattr("mem_mode") if has_mem_mode else "internal_decoupled"
        if memode is None:
            log.warning(
                f"Node {reference_node.name} of type "
                f"{reference_node.op_type} does not have a set mem_mode, "
                f"which is required for selectable weights extraction. "
                f"Assuming 'internal_decoupled'."
            )
            reference_inst.set_nodeattr("mem_mode", "internal_decoupled")
        elif memode != "internal_decoupled":
            raise FINNUserError(
                f"Node {reference_node.name} has mem_mode {memode}, "
                f"which is not supported for selectable weights extraction. "
                f"Only 'internal_decoupled' is supported."
            )

        reference_inst.set_nodeattr("code_gen_dir_ipgen", str(code_gen_dir_ipgen))
        reference_inst.set_nodeattr("bodies", self.num_bodies)
        if self._check_types(
            reference_node, ["Elementwise", "MVAU_hls", "Thresholding_hls", "VVAU_hls"]
        ):
            hls_inst = cast("HLSBackend", reference_inst)
            hls_inst.code_generation_ipgen(self._get_reference_body(), fpgapart, clk)
            hls_inst.ipgen_singlenode_code()
        else:
            cast("RTLBackend", reference_inst).generate_hdl(
                self._get_reference_body(), fpgapart, clk
            )
        set_attr_container = ["ip_path", "ipgen_path"]
        if is_hls_node(reference_node):
            set_attr_container += ["ip_vlnv"]
        if is_rtl_node(reference_node):
            set_attr_container += ["gen_top_module"]
        for attr in set_attr_container:
            attr_val = reference_inst.get_nodeattr(attr)
            self.set_nodeattr(attr, attr_val)

        # Replace files in code_gen_dir_ipgen with files from tmpdir
        for item_path in list(tmpdir.iterdir()):
            shutil.move(item_path, code_gen_dir_ipgen / item_path.name)
        tmpdir.rmdir()

    def collect_ip_dirs(self, model: ModelWrapper, ipstitch_path: str) -> list[str]:
        """Collect IP directories needed for stitching from all nodes in the model."""
        # collect list of all IP dirs
        ip_dirs = []
        need_memstreamer = False
        for node in model.graph.node:
            node_inst = cast("HWCustomOp", getCustomOp(node))
            ip_dir_value = cast("str", node_inst.get_nodeattr("ip_path"))
            if not Path(ip_dir_value).is_dir():
                raise FINNInternalError(
                    f"{node.name}: the directory that should contain the generated ip blocks "
                    f"doesn't exist: {ip_dir_value}"
                )
            ip_dirs += [ip_dir_value]
            if (
                node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls"
            ) and node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                need_memstreamer = True
        ip_dirs += [ipstitch_path + "/ip"]
        if need_memstreamer:
            # add RTL streamer IP
            ip_dirs.append(str(Path(get_settings().finn_rtllib) / "memstream"))
        return ip_dirs

    def _code_generation_ipi_stream_tap(self, node_name: str) -> list[str]:
        """Return the IPI commands adding the stream-tap sources for an HLS-style body."""
        stname = f"{node_name}_stream_tap_wrapper"
        stream_tap = Path(self.code_gen_dir_ipgen) / (stname + ".v")
        source_target = f"./ip/verilog/rtl_ops/{node_name}"
        rtllib = os.environ["FINN_RTLLIB"]
        return [
            f"add_files -copy_to {source_target} -norecurse {stream_tap}",
            f"add_files -copy_to {source_target} -norecurse "
            f"{rtllib + '/stream_tap/hdl/stream_tap.sv'}",
            f"add_files -copy_to {source_target} -norecurse "
            f"{rtllib + '/stream_tap/hdl/skid.sv'}",
        ]

    def _code_generation_ipi_tap_hls(self, hier: str, stname: str) -> list[str]:
        """Return the IPI commands wiring the stream tap for MVAU/VVAU/HLS-threshold bodies."""
        name = self.onnx_node.name
        s_axis = cast("list[tuple[str, int]]", self.get_verilog_top_module_intf_names()["s_axis"])
        tap_pin = s_axis[-1][0]
        return [
            f"create_bd_cell -type module -reference {stname} {hier}/{stname}",
            f"connect_bd_net [get_bd_pins {name}/ap_clk] [get_bd_pins {hier}/{stname}/ap_clk]",
            f"connect_bd_net [get_bd_pins {name}/ap_rst_n] "
            f"[get_bd_pins {hier}/{stname}/ap_rst_n]",
            f"connect_bd_intf_net [get_bd_intf_pins {hier}/{stname}/m_axis_1]"
            f" [get_bd_intf_pins {hier}/{name + '_wstrm'}/s_axis_0]",
            f"create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 "
            f"{hier}/{tap_pin}",
            f"connect_bd_intf_net [get_bd_intf_pins {hier}/{tap_pin}] "
            f"[get_bd_intf_pins {hier}/{stname}/s_axis_0]",
        ]

    def _code_generation_ipi_tap_rtl(self, hier: str, stname: str) -> list[str]:
        """Return the IPI commands wiring the stream tap for an RTL Thresholding body."""
        name = self.onnx_node.name
        cmd = [
            f"set_property name ip_{name} [get_bd_cells {name}]",
            f"group_bd_cells {hier} [get_bd_cells ip_{name}]",
            f"set_property name {name} [get_bd_cells {hier}/ip_{name}]",
            f"create_bd_cell -type module -reference {stname} {hier}/{stname}",
            "save_bd_design",
            # Internal connection: stream tap output -> inner IP data input
            f"connect_bd_intf_net "
            f"[get_bd_intf_pins {hier}/{stname}/m_axis_1] [get_bd_intf_pins {hier}/{name}/in1_V]",
        ]
        # Expose all hierarchy pins and connect them to internal cells
        intf_names = self.get_verilog_top_module_intf_names()
        for intf_name, _width in cast("list[tuple[str, int]]", intf_names.get("s_axis", [])):
            cmd += [
                f"create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 {hier}/{intf_name}"
            ]
            inner_cell = stname if intf_name == "s_axis_tap" else name
            inner_port = "s_axis_0" if intf_name == "s_axis_tap" else intf_name
            cmd += [
                f"connect_bd_intf_net [get_bd_intf_pins {hier}/{intf_name}] "
                f"[get_bd_intf_pins {hier}/{inner_cell}/{inner_port}]"
            ]
        for intf_name, _width in cast("list[tuple[str, int]]", intf_names.get("m_axis", [])):
            cmd += [
                f"create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 {hier}/{intf_name}"
            ]
            cmd += [
                f"connect_bd_intf_net [get_bd_intf_pins {hier}/{intf_name}] "
                f"[get_bd_intf_pins {hier}/{name}/{intf_name}]"
            ]
        for clk_name in cast("list[str]", intf_names.get("clk", [])):
            cmd += [f"create_bd_pin -dir I -type clk {hier}/{clk_name}"]
            cmd += [
                f"connect_bd_net "
                f"[get_bd_pins {hier}/{clk_name}] [get_bd_pins {hier}/{name}/{clk_name}] "
                f"[get_bd_pins {hier}/{stname}/{clk_name}]"
            ]
        for rst_name in cast("list[str]", intf_names.get("rst", [])):
            cmd += [f"create_bd_pin -dir I -type rst {hier}/{rst_name}"]
            cmd += [
                f"connect_bd_net "
                f"[get_bd_pins {hier}/{rst_name}] [get_bd_pins {hier}/{name}/{rst_name}] "
                f"[get_bd_pins {hier}/{stname}/{rst_name}]"
            ]
        return cmd

    def code_generation_ipi(self) -> list[str]:
        """Return Vivado IPI tcl commands to instantiate the NodeContainer IP."""
        ip_vlnv = cast("str", self.get_nodeattr("ip_vlnv"))
        stitched_top = self.onnx_node.name + "_wrapper"
        if ip_vlnv and self.get_nodeattr("gen_top_module") == stitched_top:
            cmd = []

            code_gen_dir_ipgen = self.code_gen_dir_ipgen
            if code_gen_dir_ipgen and Path(code_gen_dir_ipgen).is_dir():
                cmd.append(
                    "set_property ip_repo_paths "
                    f"[concat [get_property ip_repo_paths [current_project]] "
                    f"{code_gen_dir_ipgen}] [current_project]"
                )

            cmd.append("update_ip_catalog -rebuild -scan_changes")
            cmd.append(f"create_bd_cell -type ip -vlnv {ip_vlnv} {self.onnx_node.name}")
            stname = f"IN_{self.onnx_node.name}"
            cmd.append(
                f"make_bd_intf_pins_external -name {stname} "
                f"[get_bd_intf_pins {self.onnx_node.name}/{stname}]"
            )
            return cmd

        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = cast("RTLBackend", getCustomOp(node))
        set_attr_inst = ["code_gen_dir_ipgen", "ipgen_path"]
        if is_hls_node(node):
            set_attr_inst += ["ip_vlnv"]
        if is_rtl_node(node):
            set_attr_inst += ["gen_top_module"]

        for attr in set_attr_inst:
            attr_val = self.get_nodeattr(attr)
            inst.set_nodeattr(attr, attr_val)
        inst.set_nodeattr("bodies", self.num_bodies)

        orginal_name, inst.onnx_node.name = inst.onnx_node.name, self.onnx_node.name
        cmd = inst.code_generation_ipi()
        inst.onnx_node.name = orginal_name

        # Here we unify the representation of the IPs with Streamtap
        # The IO is always the same as the reference IP, but we add a stream tapper
        # The stream tapper is connect as the last s_axis

        if self._check_types(inst.onnx_node, ["MVAU", "Thresholding", "Elementwise", "VVAU"]):
            stname = f"{self.onnx_node.name}_stream_tap_wrapper"
            hier = self.onnx_node.name  # We sometimes have to make sure the hier exists

            cmd += self._code_generation_ipi_stream_tap(self.onnx_node.name)
            if self._check_types(
                inst.onnx_node, ["MVAU", "Thresholding_hls", "VVAU", "Elementwise"]
            ):
                cmd += self._code_generation_ipi_tap_hls(hier, stname)
            else:
                # Thresholding_rtl
                cmd += self._code_generation_ipi_tap_rtl(hier, stname)
        return cmd

    def execute_node(self, context: dict[str, npt.NDArray], graph: GraphProto) -> None:
        """Execute the NodeContainer by delegating to the reference node's executor."""
        node = self._get_reference_node()
        inst = cast("RTLBackend", getCustomOp(node))
        set_attr_inst = ["code_gen_dir_ipgen", "gen_top_module"]
        for attr in set_attr_inst:
            attr_val = self.get_nodeattr(attr)
            inst.set_nodeattr(attr, attr_val)

        inst.execute_node(context, graph)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str] | list[Path]:
        """Return the list of RTL source files."""
        node = self._get_reference_node()
        inst = cast("RTLBackend", getCustomOp(node))
        inst.set_nodeattr("code_gen_dir_ipgen", self.code_gen_dir_ipgen)
        inst.set_nodeattr("gen_top_module", self.get_nodeattr("gen_top_module"))
        return inst.get_rtl_file_list(abspath)

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return Verilog interface names for the NodeContainer top module."""
        multi_dnn_type = self.multi_dnn_type
        if multi_dnn_type == "selectable_weights":
            inst = cast("HWCustomOp", getCustomOp(self._get_reference_node()))
            intf_names = inst.get_verilog_top_module_intf_names()
            s_axis = cast("list[tuple[str, int]]", intf_names["s_axis"])
            if self._check_types(inst.onnx_node, ["Thresholding", "Elementwise"]):
                s_axis = [x for x in s_axis if x[0] != "in1_V"]
            s_axis.append(("s_axis_tap", 32))
            intf_names["s_axis"] = s_axis
            return intf_names
        if multi_dnn_type == "partial_reconfiguration":
            body = self._get_reference_body()
            ifnames_raw = body.get_metadata_prop("vivado_stitch_ifnames")
            if ifnames_raw is None:
                raise FINNInternalError(
                    f"{self.onnx_node.name}: reference body is missing the "
                    f"vivado_stitch_ifnames metadata property"
                )
            return cast("dict[str, list[tuple[str, int]] | list[str]]", json.loads(ifnames_raw))
        raise FINNUserError(
            f"{self.onnx_node.name}: unsupported multi_dnn_type {multi_dnn_type!r}, expected "
            f"'selectable_weights' or 'partial_reconfiguration'"
        )

    def _get_stream_tap_rep(self, node: NodeProto, reference_inst: HWCustomOp) -> int:
        """Return the TAP_REP value for the stream tap of the given reference node."""
        tap_rep: np.integer | int = 1
        if self._check_types(node, ["Thresholding_hls", "MVAU"]):
            tap_rep = np.prod(cast("list[int]", reference_inst.get_nodeattr("numInputVectors")))
        elif self._check_types(node, ["Thresholding_rtl"]):
            # for RTL Thresholds this value is fm size / pe
            tap_rep = np.prod(reference_inst.get_folded_input_shape(0)[:-1])
        elif self._check_types(node, ["VVAU"]):
            tap_rep = np.prod(cast("list[int]", reference_inst.get_nodeattr("Dim")))
        elif self._check_types(node, ["Elementwise"]):
            tap_rep = np.prod(reference_inst.get_normal_output_shape()[:-1])
        return int(tap_rep)

    def generate_hdl_stream_tap(self) -> None:
        """Generate the verilog code for the stream tap components."""
        template_path = Path(
            os.environ["FINN_RTLLIB"] + "/stream_tap/hdl/stream_tap_wrapper_template.v"
        )

        node = self._get_reference_node()
        reference_inst = cast("HWCustomOp", getCustomOp(node))

        num_bodies = self.num_bodies
        if num_bodies:
            data_width = DataType.get_smallest_possible(num_bodies).bitwidth()
            data_width = roundup_to_integer_multiple(data_width, 8)
            code_gen_dir = Path(self.code_gen_dir_ipgen)
            # calculate TAP_REP
            tap_rep = self._get_stream_tap_rep(node, reference_inst)

            stname = self.onnx_node.name
            code_gen_dict = {
                "$MODULE_NAME$": [stname],
                "$DATA_WIDTH$": [str(data_width)],
                "$TAP_REP$": [str(tap_rep)],
            }
            # apply code generation to template
            template_wrapper = template_path.read_text()
            for key in code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(code_gen_dict[key])
                template_wrapper = template_wrapper.replace(key, code_gen_line)
            (code_gen_dir / (stname + "_stream_tap_wrapper.v")).write_text(template_wrapper)
