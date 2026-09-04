"""Custom RTL op for wrapping multiple DNN bodies in a single NodeContainer."""
import json
import math
import numpy as np
import os
import shutil
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name, qonnx_make_model, roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log
from finn.util.settings import get_settings


class NodeContainer(RTLBackend, HWCustomOp):
    """Some functions are (partially) copied from FINNLoop
    Currently unsupported features:
        - Multiple inputs/outputs
        - FIFO sizing
        - Minimizing bitwitdh
    """

    def __init__(self, onnx_node, **kwargs):
        """Initialize NodeContainer and read the number of body graphs."""
        super().__init__(onnx_node, **kwargs)
        bodies_attr = get_by_name(self.onnx_node.attribute, "bodies")
        self.bodies = bodies_attr.i if bodies_attr is not None else 0

    def get_nodeattr_types(self):
        """Return attribute type definitions including per-body graph attributes."""
        b = {f"body_{i}": ("g", True, "") for i in range(self.bodies)}
        my_attrs = {
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

    def get_nodeattr(self, name):
        """Get a node attribute by name, handling graph-type attributes."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    ret = ModelWrapper(qonnx_make_model(ret))
                    return ret
                else:
                    return super().get_nodeattr(name)
            else:
                if req:
                    raise Exception(
                        """Required attribute %s unspecified in
                    a %s node"""
                        % (name, self.onnx_node.op_type)
                    )
                else:
                    # not set, return default value
                    return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        """Set a node attribute by name, handling graph-type attributes."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    attr.g.CopyFrom(value.graph)
                else:
                    super().set_nodeattr(name, value)
            else:
                super().set_nodeattr(name, value)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def _get_reference_body(self):
        """Return the first body model (body_0) as the reference."""
        # Return the first body
        # For the selectable_weights case we can assume that all bodies have the same structure
        return self.get_nodeattr("body_0")

    def _get_reference_node(self):
        """Return the first node of the first body as the reference node."""
        # Return the first node of the first body
        # For the selectable_weights case we can assume that all bodies have one node
        # And that they have the same structure (folding, datatype, etc)
        body = self._get_reference_body()
        return body.graph.node[0]

    def _check_types(self, node, types: list):
        """Return True if node.op_type matches any entry in the types list."""
        node_type = node.op_type
        for t in types:
            if t.endswith("_hls") or t.endswith("_rtl"):
                if node_type == t:
                    return True
            else:
                if node_type.startswith(t):
                    return True
        return False

    def get_normal_input_shape(self, ind=0):
        """Return the unfolded input shape."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[0]
        inst = getCustomOp(node)
        ishape = inst.get_normal_input_shape(ind)
        return ishape

    def get_normal_output_shape(self, ind=0):
        """Return the unfolded output shape."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        oshape = inst.get_normal_output_shape(ind)
        return oshape

    def get_folded_input_shape(self, ind=0):
        """Return the folded input shape."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[0]
        inst = getCustomOp(node)
        ishape = inst.get_folded_input_shape(ind)
        return ishape

    def get_folded_output_shape(self, ind=0):
        """Return the folded output shape."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        s = inst.get_folded_output_shape(ind)
        return s

    def infer_node_datatype(self, model):
        """Infer output datatype (not applicable for NodeContainer)."""
        pass

    def get_input_datatype(self, ind=0):
        """Return the input datatype."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        first_inst = getCustomOp(body.graph.node[0])
        idt = first_inst.get_input_datatype(ind)
        return idt

    def get_output_datatype(self, ind=0):
        """Return the output datatype."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        last_inst = getCustomOp(body.graph.node[-1])
        odt = last_inst.get_output_datatype(ind)
        return odt

    def get_instream_width(self, ind=0):
        """Return the input stream width."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[0]
        inst = getCustomOp(node)
        iwidth = inst.get_instream_width(ind)
        return iwidth

    def get_exp_cycles(self):
        """Return expected cycle count based on the multi_dnn_type attribute."""
        if self.get_nodeattr("multi_dnn_type") == "selectable_weights":
            body = self._get_reference_body()
            node = body.graph.node[-1]
            inst = getCustomOp(node)
            exp_cycles = inst.get_exp_cycles()
        elif self.get_nodeattr("multi_dnn_type") == "partial_reconfiguration":
            exp_cycles = 0
            for i in range(self.bodies):
                temp_exp_cycles = 0
                body = self.get_nodeattr(f"body_{i}")
                for node in body.graph.node:
                    inst = getCustomOp(node)
                    temp_exp_cycles += inst.get_exp_cycles()
                exp_cycles = max(exp_cycles, temp_exp_cycles)
        return exp_cycles

    def get_outstream_width(self, ind=0):
        """Return the output stream width."""
        assert ind == 0  # We currently only support one input
        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_outstream_width(ind=ind)

    def generate_hdl_memstream(self, fpgapart, pumped_memory=0):
        """Delegate memstream HDL generation to the reference node's implementation."""
        inst = getCustomOp(self._get_reference_node())
        bodies = self.get_nodeattr("bodies")
        inst.set_nodeattr("bodies", bodies)
        code_gen_dir_ipgen = self.get_nodeattr("code_gen_dir_ipgen")
        inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir_ipgen)
        if pumped_memory in inst.get_nodeattr_types():
            pumped_memory = inst.get_nodeattr("pumpedMemory")
        inst.generate_hdl_memstream(fpgapart, pumped_memory)

    def generate_params(self, model, path):
        """Write weight parameter files for all bodies into the given path."""
        num_bodies = self.get_nodeattr("bodies")
        reference_node = self._get_reference_node()
        reference_inst = getCustomOp(reference_node)

        for i in range(num_bodies):
            body = self.get_nodeattr(f"body_{i}")
            node = body.graph.node[-1]
            inst = getCustomOp(node)
            inst.set_nodeattr("bodies", num_bodies)
            inst.generate_params(body, path)
            param_file = "{}/memblock.dat".format(path)
            new_param_file = "{}/{}_memblock_{}.dat".format(path, node.op_type, i)
            if self._check_types(node, ["MVAU", "Elementwise", "Thresholding_hls", "VVAU"]):
                # rename so it doesn't get overwritten
                shutil.move(param_file, new_param_file)
            elif self._check_types(node, ["Thresholding_rtl"]):
                # get all generated Thresholding dat files
                pe = inst.get_nodeattr("PE")
                output_data_type = inst.get_nodeattr("outputDataType")
                o_bitwidth = DataType[output_data_type].bitwidth()
                param_files = []
                for stage in range(o_bitwidth):
                    for pe_value in range(pe):
                        param_files.append(
                            path
                            + "/%s_threshs_%s_%s.dat"
                            % (
                                node.name,
                                pe_value,
                                stage,
                            )
                        )
                for param_file in param_files:
                    param_path = Path(param_file)
                    new_param_file = param_path.with_name(
                        param_path.stem + "_i" + str(i) + param_path.suffix
                    )
                    shutil.move(param_path, new_param_file)
            else:
                raise Exception

        if self._check_types(node, ["MVAU", "Elementwise", "Thresholding_hls", "VVAU"]):
            # concatinate all .dat files together
            param_file = "{}/memblock.dat".format(path)
            with open(param_file, "w") as outfile:
                for i in range(num_bodies):
                    memblock_file = "{}/{}_memblock_{}.dat".format(path, reference_node.op_type, i)
                    with open(memblock_file, "r") as infile:
                        for line in infile:
                            outfile.write(line)
                    os.remove(memblock_file)
        elif self._check_types(reference_node, ["Thresholding_rtl"]):
            # concatinate all .dat files together
            pe = reference_inst.get_nodeattr("PE")
            output_data_type = reference_inst.get_nodeattr("outputDataType")
            o_bitwidth = DataType[output_data_type].bitwidth()
            for stage in range(o_bitwidth):
                for pe_value in range(pe):
                    param_file = path + "/%s_threshs_%s_%s.dat" % (
                        reference_node.name,
                        pe_value,
                        stage,
                    )
                    with open(param_file, "w") as outfile:
                        for i in range(num_bodies):
                            body_file = "{}/{}_threshs_{}_{}_i{}.dat".format(
                                path, reference_node.name, pe_value, stage, i
                            )
                            with open(body_file, "r") as infile:
                                cnt = 0
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
                            os.remove(body_file)

    def generate_hdl(self, model, fpgapart, clk):
        """Generate HDL for the NodeContainer based on multi_dnn_type."""
        multi_dnn_type = self.get_nodeattr("multi_dnn_type")
        if multi_dnn_type == "selectable_weights":
            self.generate_hdl_memstream(fpgapart)
            self.generate_params(model, self.get_nodeattr("code_gen_dir_ipgen"))
            self.generate_hdl_stream_tap()

            code_gen_dir_ipgen = self.get_nodeattr("code_gen_dir_ipgen")
            items = os.listdir(code_gen_dir_ipgen)
            tmpdir = os.path.join(code_gen_dir_ipgen, "tmp")
            os.makedirs(tmpdir, exist_ok=True)
            for item in items:
                item_path = os.path.join(code_gen_dir_ipgen, item)
                shutil.move(item_path, os.path.join(tmpdir, item))

            # Generate reference node hw and copy needed files to correct location
            reference_node = self._get_reference_node()
            reference_inst = getCustomOp(reference_node)

            has_mem_mode = "mem_mode" in reference_inst.get_nodeattr_types()
            memode = (
                reference_inst.get_nodeattr("mem_mode") if has_mem_mode else "internal_decoupled"
            )
            if memode is None:
                log.warning(
                    f"Node {reference_node.name} of type "
                    f"{reference_node.op_type} does not have a set mem_mode, "
                    f"which is required for selectable weights extraction. "
                    f"Assuming 'internal_decoupled'."
                )
                reference_inst.set_nodeattr("mem_mode", "internal_decoupled")
            elif memode != "internal_decoupled":
                raise Exception(
                    f"Node {reference_node.name} has mem_mode {memode}, "
                    f"which is not supported for selectable weights extraction. "
                    f"Only 'internal_decoupled' is supported."
                )

            reference_inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir_ipgen)
            bodies = self.get_nodeattr("bodies")
            reference_inst.set_nodeattr("bodies", bodies)
            if self._check_types(
                reference_node, ["Elementwise", "MVAU_hls", "Thresholding_hls", "VVAU_hls"]
            ):
                reference_inst.code_generation_ipgen(self._get_reference_body(), fpgapart, clk)
                reference_inst.ipgen_singlenode_code()
            else:
                reference_inst.generate_hdl(self._get_reference_body(), fpgapart, clk)
            set_attr_container = ["ip_path", "ipgen_path"]
            if is_hls_node(reference_node):
                set_attr_container += ["ip_vlnv"]
            if is_rtl_node(reference_node):
                set_attr_container += ["gen_top_module"]
            for attr in set_attr_container:
                attr_val = reference_inst.get_nodeattr(attr)
                self.set_nodeattr(attr, attr_val)

            # Replace files in code_gen_dir_ipgen with files from tmpdir
            for item in os.listdir(tmpdir):
                shutil.move(os.path.join(tmpdir, item), os.path.join(code_gen_dir_ipgen, item))
            os.rmdir(tmpdir)
        else:
            raise ValueError  # Make more verbose?
        return

    def collect_ip_dirs(self, model, ipstitch_path):
        """Collect IP directories needed for stitching from all nodes in the model."""
        # collect list of all IP dirs
        ip_dirs = []
        need_memstreamer = False
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            assert os.path.isdir(
                ip_dir_value
            ), """The directory that should
            contain the generated ip blocks doesn't exist."""
            ip_dirs += [ip_dir_value]
            if node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls":
                if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                    need_memstreamer = True
        ip_dirs += [ipstitch_path + "/ip"]
        if need_memstreamer:
            # add RTL streamer IP
            ip_dirs.append(os.path.join(get_settings().finn_rtllib, "memstream"))
        return ip_dirs

    def code_generation_ipi(self):
        """Return Vivado IPI tcl commands to instantiate the NodeContainer IP."""
        ip_vlnv = self.get_nodeattr("ip_vlnv")
        stitched_top = self.onnx_node.name + "_wrapper"
        if ip_vlnv and self.get_nodeattr("gen_top_module") == stitched_top:
            cmd = []

            code_gen_dir_ipgen = self.get_nodeattr("code_gen_dir_ipgen")
            if code_gen_dir_ipgen and os.path.isdir(code_gen_dir_ipgen):
                cmd.append(
                    "set_property ip_repo_paths "
                    "[concat [get_property ip_repo_paths [current_project]] %s] "
                    "[current_project]" % code_gen_dir_ipgen
                )

            cmd.append("update_ip_catalog -rebuild -scan_changes")
            cmd.append("create_bd_cell -type ip -vlnv %s %s" % (ip_vlnv, self.onnx_node.name))
            stname = "IN_%s" % self.onnx_node.name
            cmd.append(
                "make_bd_intf_pins_external -name %s [get_bd_intf_pins %s/%s]"
                % (stname, self.onnx_node.name, stname)
            )
            return cmd

        body = self._get_reference_body()
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        set_attr_inst = ["code_gen_dir_ipgen", "ipgen_path"]
        if is_hls_node(node):
            set_attr_inst += ["ip_vlnv"]
        if is_rtl_node(node):
            set_attr_inst += ["gen_top_module"]

        for attr in set_attr_inst:
            attr_val = self.get_nodeattr(attr)
            inst.set_nodeattr(attr, attr_val)
        inst.set_nodeattr("bodies", self.get_nodeattr("bodies"))

        orginal_name, inst.onnx_node.name = inst.onnx_node.name, self.onnx_node.name
        cmd = inst.code_generation_ipi()
        inst.onnx_node.name = orginal_name

        # Here we unify the representation of the IPs with Streamtap
        # The IO is always the same as the reference IP, but we add a stream tapper
        # The stream tapper is connect as the last s_axis

        if self._check_types(inst.onnx_node, ["MVAU", "Thresholding", "Elementwise", "VVAU"]):
            stname = f"{self.onnx_node.name}_stream_tap_wrapper"
            hier = self.onnx_node.name  # We sometimes have to make sure the hier exists

            stream_tap = os.path.join(self.get_nodeattr("code_gen_dir_ipgen"), stname + ".v")
            source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
            cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, stream_tap)]
            cmd += [
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.environ["FINN_RTLLIB"] + "/stream_tap/hdl/stream_tap.sv")
            ]
            cmd += [
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.environ["FINN_RTLLIB"] + "/stream_tap/hdl/skid.sv")
            ]
            if self._check_types(
                inst.onnx_node, ["MVAU", "Thresholding_hls", "VVAU", "Elementwise"]
            ):
                cmd += ["create_bd_cell -type module -reference %s %s/%s" % (stname, hier, stname)]
                cmd += [
                    "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/%s/ap_clk]"
                    % (self.onnx_node.name, hier, stname)
                ]
                cmd += [
                    "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins %s/%s/ap_rst_n]"
                    % (self.onnx_node.name, hier, stname)
                ]
                cmd += [
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_1]"
                    " [get_bd_intf_pins %s/%s/s_axis_0]"
                    % (hier, stname, hier, self.onnx_node.name + "_wstrm")
                ]
                cmd += [
                    "create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 %s/%s"
                    % (hier, self.get_verilog_top_module_intf_names()["s_axis"][-1][0])
                ]
                cmd += [
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s/s_axis_0]"
                    % (
                        hier,
                        self.get_verilog_top_module_intf_names()["s_axis"][-1][0],
                        hier,
                        stname,
                    )
                ]
            else:
                # Thresholding_rtl
                cmd += [
                    "set_property name ip_%s [get_bd_cells %s]"
                    % (self.onnx_node.name, self.onnx_node.name)
                ]
                cmd += ["group_bd_cells %s [get_bd_cells ip_%s]" % (hier, self.onnx_node.name)]
                cmd += [
                    "set_property name %s [get_bd_cells %s/ip_%s]"
                    % (self.onnx_node.name, hier, self.onnx_node.name)
                ]
                cmd += ["create_bd_cell -type module -reference %s %s/%s" % (stname, hier, stname)]
                cmd += ["save_bd_design"]
                # Internal connection: stream tap output -> inner IP data input
                cmd += [
                    "connect_bd_intf_net "
                    "[get_bd_intf_pins %s/%s/m_axis_1] [get_bd_intf_pins %s/%s/in1_V]"
                    % (hier, stname, hier, self.onnx_node.name)
                ]
                # Expose all hierarchy pins and connect them to internal cells
                intf_names = self.get_verilog_top_module_intf_names()
                for intf_name, _width in intf_names.get("s_axis", []):
                    cmd += [
                        "create_bd_intf_pin -mode Slave "
                        "-vlnv xilinx.com:interface:axis_rtl:1.0 %s/%s" % (hier, intf_name)
                    ]
                    inner_cell = stname if intf_name == "s_axis_tap" else self.onnx_node.name
                    inner_port = "s_axis_0" if intf_name == "s_axis_tap" else intf_name
                    cmd += [
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s/%s]"
                        % (hier, intf_name, hier, inner_cell, inner_port)
                    ]
                for intf_name, _width in intf_names.get("m_axis", []):
                    cmd += [
                        "create_bd_intf_pin -mode Master "
                        "-vlnv xilinx.com:interface:axis_rtl:1.0 %s/%s" % (hier, intf_name)
                    ]
                    cmd += [
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s/%s]"
                        % (hier, intf_name, hier, self.onnx_node.name, intf_name)
                    ]
                for clk_name in intf_names.get("clk", []):
                    cmd += ["create_bd_pin -dir I -type clk %s/%s" % (hier, clk_name)]
                    cmd += [
                        "connect_bd_net "
                        "[get_bd_pins %s/%s] [get_bd_pins %s/%s/%s] [get_bd_pins %s/%s/%s]"
                        % (
                            hier,
                            clk_name,
                            hier,
                            self.onnx_node.name,
                            clk_name,
                            hier,
                            stname,
                            clk_name,
                        )
                    ]
                for rst_name in intf_names.get("rst", []):
                    cmd += ["create_bd_pin -dir I -type rst %s/%s" % (hier, rst_name)]
                    cmd += [
                        "connect_bd_net "
                        "[get_bd_pins %s/%s] [get_bd_pins %s/%s/%s] [get_bd_pins %s/%s/%s]"
                        % (
                            hier,
                            rst_name,
                            hier,
                            self.onnx_node.name,
                            rst_name,
                            hier,
                            stname,
                            rst_name,
                        )
                    ]
        return cmd

    def execute_node(self, context, graph):
        """Execute the NodeContainer by delegating to the reference node's executor."""
        node = self._get_reference_node()
        inst = getCustomOp(node)
        set_attr_inst = ["code_gen_dir_ipgen", "gen_top_module"]
        for attr in set_attr_inst:
            attr_val = self.get_nodeattr(attr)
            inst.set_nodeattr(attr, attr_val)

        ret = inst.execute_node(context, graph)
        return ret

    def get_rtl_file_list(self, abspath=False):
        """Return the list of RTL source files."""
        node = self._get_reference_node()
        inst = getCustomOp(node)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        gen_top_module = self.get_nodeattr("gen_top_module")
        inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        inst.set_nodeattr("gen_top_module", gen_top_module)
        return inst.get_rtl_file_list(abspath)

    def get_verilog_top_module_intf_names(self):
        """Return Verilog interface names for the NodeContainer top module."""
        if self.get_nodeattr("multi_dnn_type") == "selectable_weights":
            inst = getCustomOp(self._get_reference_node())
            intf_names = inst.get_verilog_top_module_intf_names()
            if self._check_types(inst.onnx_node, ["Thresholding", "Elementwise"]):
                intf_names["s_axis"] = [x for x in intf_names["s_axis"] if x[0] != "in1_V"]
            intf_names["s_axis"].append(("s_axis_tap", 32))
            return intf_names
        elif self.get_nodeattr("multi_dnn_type") == "partial_reconfiguration":
            body = self._get_reference_body()
            ifnames_raw = body.get_metadata_prop("vivado_stitch_ifnames")
            return json.loads(ifnames_raw)

    def generate_hdl_stream_tap(self):
        """Helper function to generate verilog code for stream tap components."""
        template_path = os.environ["FINN_RTLLIB"] + "/stream_tap/hdl/stream_tap_wrapper_template.v"

        node = self._get_reference_node()
        reference_inst = getCustomOp(node)

        if self.get_nodeattr("bodies"):
            data_width = DataType.get_smallest_possible(self.get_nodeattr("bodies")).bitwidth()
            data_width = roundup_to_integer_multiple(data_width, 8)
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # calculate TAP_REP
            tap_rep = 1
            if self._check_types(node, ["Thresholding_hls"]):
                tap_rep = np.prod(reference_inst.get_nodeattr("numInputVectors"))
            elif self._check_types(node, ["Thresholding_rtl"]):
                # for RTL Thresholds this value is fm size / pe
                tap_rep = np.prod(reference_inst.get_folded_input_shape(0)[:-1])
            elif self._check_types(node, ["MVAU"]):
                tap_rep = np.prod(reference_inst.get_nodeattr("numInputVectors"))
            elif self._check_types(node, ["VVAU"]):
                tap_rep = np.prod(reference_inst.get_nodeattr("Dim"))
            elif self._check_types(node, ["Elementwise"]):
                tap_rep = np.prod(reference_inst.get_normal_output_shape()[:-1])
            tap_rep = int(tap_rep)

            stname = self.onnx_node.name
            code_gen_dict = {
                "$MODULE_NAME$": [stname],
                "$DATA_WIDTH$": [str(data_width)],
                "$TAP_REP$": [str(tap_rep)],
            }
            # apply code generation to template
            with open(template_path, "r") as f:
                template_wrapper = f.read()
            for key in code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(code_gen_dict[key])
                template_wrapper = template_wrapper.replace(key, code_gen_line)
            with open(
                os.path.join(code_gen_dir, stname + "_stream_tap_wrapper.v"),
                "w",
            ) as f:
                f.write(template_wrapper)
