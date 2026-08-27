"""Transformations for generating and naming NodeContainer stitched IP blocks."""
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP


class GenerateNodeContainerStitched(Transformation):
    """Generate stitched HLS/RTL IP for each NodeContainer in the model."""

    def __init__(self, cfg):
        """Initialize with the DataflowBuildConfig used for IP generation."""
        self.cfg = cfg
        super().__init__()

    def apply(self, model):
        """Generate stitched IP for selectable-weights and PR NodeContainer nodes."""
        for node in model.graph.node:
            if node.op_type == "NodeContainer":
                node_inst = getCustomOp(node)
                if node_inst.get_nodeattr("multi_dnn_type") == "selectable_weights":
                    inshape = list(node_inst.get_folded_input_shape())
                    oshape = list(node_inst.get_folded_output_shape())
                    inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, inshape)
                    out = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
                    graph = helper.make_graph([node], node.name, [inp], [out])
                    node_model = ModelWrapper(qonnx_make_model(graph))
                    node_model = node_model.transform(
                        PrepareIP(self.cfg._resolve_fpga_part(), self.cfg._resolve_hls_clk_period())
                    )
                    node_model = node_model.transform(HLSSynthIP(self.cfg._resolve_fpga_part()))
                    node_model = node_model.transform(
                        CreateStitchedIP(
                            self.cfg._resolve_fpga_part(),
                            self.cfg.synth_clk_period_ns,
                            ip_name=node.name,
                            vitis=False,
                            nodecontainer=True,
                        )
                    )
                    inner_node_inst = getCustomOp(node_model.graph.node[0])
                    vivado_stitch_proj_dir = node_model.get_metadata_prop("vivado_stitch_proj")
                    wrapper_filename = node_model.get_metadata_prop("wrapper_filename")
                    block_vlnv = node_model.get_metadata_prop("vivado_stitch_vlnv")
                    node_inst.set_nodeattr("ipgen_path", wrapper_filename)
                    node_inst.set_nodeattr("ip_path", vivado_stitch_proj_dir + "/ip")
                    node_inst.set_nodeattr("gen_top_module", "%s_wrapper" % node.name)
                    node_inst.set_nodeattr("ip_vlnv", block_vlnv)
                    node_inst.set_nodeattr(
                        "code_gen_dir_ipgen", inner_node_inst.get_nodeattr("code_gen_dir_ipgen")
                    )
                elif node_inst.get_nodeattr("multi_dnn_type") == "partial_reconfiguration":
                    # Local import to avoid a circular import: build_dataflow_steps
                    # imports GenerateNodeContainerStitched from this module.
                    from finn.builder.build_dataflow_steps import step_set_fifo_depths

                    node_inst = getCustomOp(node)
                    bodies = node_inst.get_nodeattr("bodies")
                    for id in range(bodies):
                        body_attr = f"body_{id}"
                        node_model = node_inst.get_nodeattr(body_attr)
                        # Give each PR body the same FIFO treatment as the top-level flow
                        node_model = step_set_fifo_depths(
                            node_model,
                            self.cfg,
                            parent_node=node.name + "_" + body_attr + "_",
                        )
                        node_model = node_model.transform(
                            PrepareIP(
                                self.cfg._resolve_fpga_part(), self.cfg._resolve_hls_clk_period()
                            )
                        )
                        node_model = node_model.transform(HLSSynthIP(self.cfg._resolve_fpga_part()))
                        node_model = node_model.transform(
                            CreateStitchedIP(
                                self.cfg._resolve_fpga_part(),
                                self.cfg.synth_clk_period_ns,
                                ip_name=f"{node.name}_{id}",
                                vitis=False,
                                nodecontainer=True,
                            )
                        )
                        node_inst.set_nodeattr(body_attr, node_model)
                        # Set Nodecontainer attributes for stitiched IP generation
                        if id == 0:
                            vivado_stitch_proj_dir = node_model.get_metadata_prop(
                                "vivado_stitch_proj"
                            )
                            wrapper_filename = node_model.get_metadata_prop("wrapper_filename")
                            block_vlnv = node_model.get_metadata_prop("vivado_stitch_vlnv")
                            node_inst.set_nodeattr("ipgen_path", wrapper_filename)
                            node_inst.set_nodeattr("ip_path", vivado_stitch_proj_dir + "/ip")
                            node_inst.set_nodeattr("gen_top_module", "%s_wrapper" % node.name)
                            node_inst.set_nodeattr("ip_vlnv", block_vlnv)
                            node_inst.set_nodeattr("code_gen_dir_ipgen", vivado_stitch_proj_dir)
        return (model, False)


class NameNodeContainerNodes(Transformation):
    """Assign unique names to nodes inside partial-reconfiguration NodeContainer bodies."""

    def apply(self, model):
        """Rename all nodes inside PR NodeContainer bodies with unique prefixed names."""
        for node in model.graph.node:
            if node.op_type != "NodeContainer":
                continue
            node_inst = getCustomOp(node)
            if node_inst.get_nodeattr("multi_dnn_type") == "partial_reconfiguration":
                bodies = node_inst.get_nodeattr("bodies")
                for id in range(bodies):
                    body_attr = f"body_{id}"
                    body_model = node_inst.get_nodeattr(body_attr)
                    prefix = f"{node.name}_body_{id}_"

                    optype_count = {}
                    for n in body_model.graph.node:
                        if n.op_type not in optype_count.keys():
                            optype_count[n.op_type] = 0
                        if not n.name.startswith(prefix):
                            n.name = "%s%s_%d" % (prefix, n.op_type, optype_count[n.op_type])
                        optype_count[n.op_type] += 1

                    node_inst.set_nodeattr(body_attr, body_model)
        return (model, False)
