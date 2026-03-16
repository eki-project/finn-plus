import math
import numpy as np
import os
import shutil
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name, is_finn_op, qonnx_make_model, roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.fpgadataflow import is_hls_node, is_rtl_node


class NodeContainer(HWCustomOp, RTLBackend):
    # Some functions are (partially) copied from FINNLoop
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        bodies_attr = get_by_name(self.onnx_node.attribute, "bodies")
        self.bodies = bodies_attr.i if bodies_attr is not None else 0

    def get_nodeattr_types(self):
        b = {f"body_{i}": ("g", True, "") for i in range(self.bodies)}
        my_attrs = {
            "bodies": ("i", True, 0),
            "multi_dnn_type": ("s", True, ""),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            **b,
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_nodeattr(self, name):
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

    def get_normal_input_shape(self, ind=0):
        body = self.get_nodeattr("body_0")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = body.graph.node[0]
            if is_finn_op(node.domain):
                inst = getCustomOp(node)
                ishape = inst.get_normal_input_shape(0)
            else:
                ishape = body.get_tensor_shape(node.input[0])
        else:
            body = self.get_nodeattr("body_0")
            tensor = body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = body.find_consumer(tensor)
            if is_finn_op(param_node.domain):
                inst = getCustomOp(param_node)
                ishape = inst.get_normal_input_shape(1)
            else:
                ishape = body.get_tensor_shape(tensor)
        return ishape

    def get_normal_output_shape(self, ind=0):
        body = self.get_nodeattr("body_0")
        node = body.graph.node[-1]
        if is_finn_op(node.domain):
            inst = getCustomOp(node)
            oshape = inst.get_normal_output_shape(0)
        else:
            oshape = body.get_tensor_shape(node.output[0])
        return oshape

    def get_folded_input_shape(self, ind=0):
        body = self.get_nodeattr("body_0")
        if ind == 0:
            node = body.graph.node[0]
            inst = getCustomOp(node)
            ishape = inst.get_folded_input_shape(0)
        else:
            tensor = body.graph.input[ind].name
            param_node = body.find_consumer(tensor)
            inst = getCustomOp(param_node)
            ishape = inst.get_folded_input_shape(1)
        return ishape

    def get_folded_output_shape(self, ind=0):
        body = self.get_nodeattr("body_0")
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        s = inst.get_folded_output_shape(0)
        return s

    def infer_node_datatype(self, model):
        for i in range(self.get_nodeattr("bodies")):
            body = self.get_nodeattr(f"body_{i}")
            node = body.graph.node[-1]
            inst = getCustomOp(node)
            inst.infer_node_datatype(body)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            idt = DataType[self.get_nodeattr("inputDataType")]
        else:
            loop_body = self.get_nodeattr("body")
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if is_finn_op(param_node.domain):
                inst = getCustomOp(param_node)
                idt = inst.get_input_datatype(1)
            else:
                idt = loop_body.get_tensor_datatype(tensor)
        return idt

    def get_output_datatype(self, ind=0):
        odt = DataType[self.get_nodeattr("outputDataType")]
        return odt

    def get_instream_width(self, ind=0):
        body = self.get_nodeattr("body_0")
        if ind == 0:
            node = body.graph.node[0]
            inst = getCustomOp(node)
            iwidth = inst.get_instream_width(0)
        else:
            tensor = body.graph.input[ind].name
            param_node = body.find_consumer(tensor)
            inst = getCustomOp(param_node)
            iwidth = inst.get_instream_width(1)
        return iwidth

    def make_shape_compatible_op(self, model):
        assert False

    def _get_reference_node(self):
        # Return the first body and first node in that body
        # This can be used because we made sure all bodies have the same structure
        body = self.get_nodeattr("body_0")
        node = body.graph.node[0]
        return node

    def _get_reference_body(self):
        body = self.get_nodeattr("body_0")
        return body

    def generate_hdl_memstream(self, fpgapart, pumped_memory=0):
        inst = getCustomOp(self._get_reference_node())
        bodies = self.get_nodeattr("bodies")
        inst.set_nodeattr("bodies", bodies)
        code_gen_dir_ipgen = self.get_nodeattr("code_gen_dir_ipgen")
        inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir_ipgen)
        if pumped_memory in inst.get_nodeattr_types():
            pumped_memory = inst.get_nodeattr("pumpedMemory")
        inst.generate_hdl_memstream(fpgapart, pumped_memory)

    def generate_params(self, model, path):
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
            if node.op_type.startswith("MVAU") or node.op_type.startswith("Elementwise"):
                # rename so it doesn't get overwritten
                shutil.move(param_file, new_param_file)
            elif node.op_type.startswith("Thresholding"):
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

        if reference_node.op_type.startswith("MVAU") or reference_node.op_type.startswith(
            "Elementwise"
        ):
            # concatinate all .dat files together
            param_file = "{}/memblock.dat".format(path)
            with open(param_file, "w") as outfile:
                for i in range(num_bodies):
                    memblock_file = "{}/{}_memblock_{}.dat".format(path, reference_node.op_type, i)
                    with open(memblock_file, "r") as infile:
                        for line in infile:
                            outfile.write(line)
                    os.remove(memblock_file)
        elif reference_node.op_type.startswith("Thresholding"):
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
        multi_dnn_type = self.get_nodeattr("multi_dnn_type")
        if multi_dnn_type:
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
            reference_inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir_ipgen)
            bodies = self.get_nodeattr("bodies")
            reference_inst.set_nodeattr("bodies", bodies)
            if reference_node.op_type.startswith("Elementwise"):
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

    def code_generation_ipi(self):
        body = self.get_nodeattr("body_0")
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

        orginal_name, inst.onnx_node.name = inst.onnx_node.name, self.onnx_node.name
        cmd = inst.code_generation_ipi()
        inst.onnx_node.name = orginal_name

        # TODO Merge this code
        if inst.onnx_node.op_type == "MVAU_rtl":
            stname = "IN_%s" % self.onnx_node.name
            stream_tap = os.path.join(
                self.get_nodeattr("code_gen_dir_ipgen"), stname + "_stream_tap_wrapper.v"
            )
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
            cmd += [
                "create_bd_cell -type module -reference %s %s/%s"
                % (
                    stname + "_stream_tap_wrapper",
                    self.onnx_node.name,
                    stname + "_stream_tap_wrapper",
                )
            ]
            cmd += [
                "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/%s/ap_clk]"
                % (self.onnx_node.name, self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins %s/%s/ap_rst_n]"
                % (self.onnx_node.name, self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_1]"
                " [get_bd_intf_pins %s/%s/s_axis_0]"
                % (
                    self.onnx_node.name,
                    stname + "_stream_tap_wrapper",
                    self.onnx_node.name,
                    self.onnx_node.name + "_wstrm",
                )
            ]
            cmd += [
                "make_bd_intf_pins_external -name %s [get_bd_intf_pins %s/%s/s_axis_0]"
                % (stname, self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]

        elif inst.onnx_node.op_type == "Thresholding_rtl":
            stname = "IN_%s" % self.onnx_node.name
            stream_tap = os.path.join(
                self.get_nodeattr("code_gen_dir_ipgen"), stname + "_stream_tap_wrapper.v"
            )
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
            cmd += [
                "create_bd_cell -type module -reference %s %s"
                % (stname + "_stream_tap_wrapper", stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/ap_clk]"
                % (self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins %s/ap_rst_n]"
                % (self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_1] [get_bd_intf_pins %s/in1_V]"
                % (stname + "_stream_tap_wrapper", self.onnx_node.name)
            ]
            cmd += [
                "make_bd_intf_pins_external -name %s [get_bd_intf_pins %s/s_axis_0]"
                % (stname, stname + "_stream_tap_wrapper")
            ]

        elif inst.onnx_node.op_type.startswith("Elementwise"):
            source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name

            stname = "IN_%s" % self.onnx_node.name
            stream_tap = os.path.join(
                self.get_nodeattr("code_gen_dir_ipgen"), stname + "_stream_tap_wrapper.v"
            )
            cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, stream_tap)]
            cmd += [
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.environ["FINN_RTLLIB"] + "/stream_tap/hdl/stream_tap.sv")
            ]
            cmd += [
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.environ["FINN_RTLLIB"] + "/stream_tap/hdl/skid.sv")
            ]
            cmd += [
                "create_bd_cell -type module -reference %s %s"
                % (stname + "_stream_tap_wrapper", stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/ap_clk]"
                % (self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins %s/ap_rst_n]"
                % (self.onnx_node.name, stname + "_stream_tap_wrapper")
            ]
            cmd += [
                "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_1] [get_bd_intf_pins %s/%s/s_axis_0]"
                % (stname + "_stream_tap_wrapper", self.onnx_node.name, self.onnx_node.name + "_wstrm")
            ]
            cmd += [
                "make_bd_intf_pins_external -name %s [get_bd_intf_pins %s/s_axis_0]"
                % (stname, stname + "_stream_tap_wrapper")
            ]

        return cmd

    def execute_node(self, context, graph):
        body = self.get_nodeattr("body_0")
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        set_attr_inst = ["code_gen_dir_ipgen", "gen_top_module"]
        for attr in set_attr_inst:
            attr_val = self.get_nodeattr(attr)
            inst.set_nodeattr(attr, attr_val)

        ret = inst.execute_node(context, graph)
        return ret

    # def get_input_datatype(self, ind=0):
    #     """Returns FINN DataType of input."""
    #     if ind == 0:
    #         idt = DataType[self.get_nodeattr("inputDataType")]
    #     else:
    #         body = self.get_nodeattr("body_0")
    #         tensor = body.graph.input[ind].name
    #         param_node = body.find_consumer(tensor)
    #         if is_finn_op(param_node.domain):
    #             inst = getCustomOp(param_node)
    #             idt = inst.get_input_datatype(1)
    #         else:
    #             idt = body.get_tensor_datatype(tensor)
    #     return idt

    def get_outstream_width(self, ind=0):
        body = self.get_nodeattr("body_0")
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_outstream_width(ind=ind)

    def get_rtl_file_list(self, abspath=False):
        body = self.get_nodeattr("body_0")
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        gen_top_module = self.get_nodeattr("gen_top_module")
        inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        inst.set_nodeattr("gen_top_module", gen_top_module)
        return inst.get_rtl_file_list(abspath)

    def get_exp_cycles(self):
        body = self.get_nodeattr("body_0")
        node = body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_exp_cycles()

    def get_verilog_top_module_intf_names(self):
        inst = getCustomOp(self._get_reference_node())
        intf_names = inst.get_verilog_top_module_intf_names()
        name = f"s_axis_0_{self.onnx_node.name.split('_')[-1]}"
        intf_names["s_axis"].append((name, 32))
        return intf_names

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
            # for Thresholds this value is fm size / pe
            tap_rep = 1
            if node.op_type == "Thresholding_rtl":
                tap_rep = np.prod(reference_inst.get_folded_input_shape(0)[:-1])
            elif node.op_type == "MVAU_rtl":
                num_inp_vec = reference_inst.get_nodeattr("numInputVectors")
                tap_rep = np.prod(num_inp_vec)
            elif node.op_type == "ElementwiseAdd_hls":
                tap_rep = 1

            tap_rep = int(tap_rep)

            stname = "IN_%s" % self.onnx_node.name
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

    def _get_larget_dt(self, dt_list):
        def _category(dt):
            if dt.get_canonical_name().startswith("FLOAT"):
                return "float"
            elif dt.is_integer():
                return "integer"
            elif dt.is_fixed_point():
                return "fixed_point"
            else:
                raise ValueError(f"Unknown datatype category for {dt}")

        categories = {_category(dt) for dt in dt_list}
        if len(categories) > 1:
            raise ValueError(
                "All datatypes must be either all FLOAT, all integer, or all fixed-point."
            )

        largest_dt = dt_list[0]
        for dt in dt_list[1:]:
            if dt.bitwidth() > largest_dt.bitwidth():
                largest_dt = dt
        return largest_dt

    def minimize_weight_bit_width(self, model):
        if self._get_reference_node().op_type.startswith(
            "VVAU"
        ) or self._get_reference_node().op_type.startswith("MVAU"):
            # VVAU, MVAU just return the bit width
            wdts = []
            for body_ind in range(self.get_nodeattr("bodies")):
                body = self.get_nodeattr(f"body_{body_ind}")
                node = body.graph.node[0]
                inst = getCustomOp(node)
                wdts += [inst.minimize_weight_bit_width(body)]
            largest_dt = self._get_larget_dt(wdts)
            for body_ind in range(self.get_nodeattr("bodies")):
                body = self.get_nodeattr(f"body_{body_ind}")
                node = body.graph.node[0]
                inst = getCustomOp(node)
                inst.set_nodeattr("weightDataType", largest_dt.name)
                self.set_nodeattr(f"body_{body_ind}", body)
        elif self._get_reference_node().op_type.startswith("Elementwise"):
            # Elementwise applies directly
            lhs_wdts = []
            rhs_wdts = []
            for body_ind in range(self.get_nodeattr("bodies")):
                body = self.get_nodeattr(f"body_{body_ind}")
                node = body.graph.node[0]
                inst = getCustomOp(node)
                inst.minimize_weight_bit_width(body)

                lhs_dt = None
                if body.get_initializer(node.input[0]) is not None:
                    lhs_dt = body.get_tensor_datatype(node.input[0])
                lhs_wdts += [lhs_dt]

                rhs_dt = None
                if body.get_initializer(node.input[1]) is not None:
                    rhs_dt = body.get_tensor_datatype(node.input[1])
                rhs_wdts += [rhs_dt]

            # Check if all are none or all are not none
            if None in lhs_wdts and not all(wdt is None for wdt in lhs_wdts):
                raise ValueError("Either all or none of the inputs should have a weight datatype")
            if None in rhs_wdts and not all(wdt is None for wdt in rhs_wdts):
                raise ValueError("Either all or none of the inputs should have a weight datatype")

            largest_lhs_dt = None
            if all(wdt is not None for wdt in lhs_wdts):
                largest_lhs_dt = self._get_larget_dt(lhs_wdts)

            largest_rhs_dt = None
            if all(wdt is not None for wdt in rhs_wdts):
                largest_rhs_dt = self._get_larget_dt(rhs_wdts)

            for body_ind in range(self.get_nodeattr("bodies")):
                body = self.get_nodeattr(f"body_{body_ind}")
                node = body.graph.node[0]
                inst = getCustomOp(node)
                if largest_lhs_dt is not None:
                    inst.set_nodeattr("lhs_dtype", largest_lhs_dt.name)
                    body.set_tensor_datatype(node.input[0], largest_lhs_dt)

                if largest_rhs_dt is not None:
                    inst.set_nodeattr("rhs_dtype", largest_rhs_dt.name)
                    body.set_tensor_datatype(node.input[1], largest_rhs_dt)

                self.set_nodeattr(f"body_{body_ind}", body)

    def minimize_accumulator_width(self, model):
        # Ignore attention at first
        # VVAU, Thres, MVAU, Elementwise
        if (
            self._get_reference_node().op_type.startswith("VVAU")
            or self._get_reference_node().op_type.startswith("MVAU")
            or self._get_reference_node().op_type.startswith("Elementwise")
            or self._get_reference_node().op_type.startswith("Thresholding")
        ):
            adts = []
            for body_ind in range(self.get_nodeattr("bodies")):
                body = self.get_nodeattr(f"body_{body_ind}")
                node = body.graph.node[0]
                inst = getCustomOp(node)
                adts += [inst.minimize_accumulator_width(body)]
            largest_adt = self._get_larget_dt(adts)
            for body_ind in range(self.get_nodeattr("bodies")):
                body = self.get_nodeattr(f"body_{body_ind}")
                node = body.graph.node[0]
                inst = getCustomOp(node)
                if node.op_type.startswith("MVAU") or node.op_type.startswith("VVAU"):
                    if inst.get_nodeattr("noActivation"):
                        inst.set_nodeattr("outputDataType", largest_adt.name)
                    inst.set_nodeattr("accDataType", largest_adt.name)
                elif node.op_type.startswith("Thresholding"):
                    inst.set_nodeattr("weightDataType", largest_adt.name)
                elif node.op_type.startswith("Elementwise"):
                    inst.set_nodeattr("out_dtype", largest_adt.name)
                self.set_nodeattr(f"body_{body_ind}", body)
