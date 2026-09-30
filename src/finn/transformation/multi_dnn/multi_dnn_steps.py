"""Build-flow steps for multi-DNN model construction and collapsing."""
import json
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.attention_heads import InferSplitIntoSplitMultiHeads
from finn.transformation.fpgadataflow.convert_to_hw.concat import InferConcatLayer
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.multi_dnn.multi_dnn_pr import ApplyPartialReconfiguration
from finn.transformation.multi_dnn.multi_dnn_selectable import ExtractSelectableWeights
from finn.transformation.multi_dnn.multi_dnn_wrapper_transformations import (
    CollapseModels,
    CombineInputsChannelwise,
    CombineOutputsChannelwise,
    MultiDNNWrapperExposeIO,
)
from finn.transformation.multi_dnn.nodecontainer_transformations import NameNodeContainerNodes


def _resolve_multi_dnn_mode(cfg: DataflowBuildConfig):
    """Read the generation mode and kwargs from the multi-DNN config JSON."""
    with open(cfg.multi_dnn_config_path, "r") as fp_json:
        multi_dnn_config = json.load(fp_json)
    gen = multi_dnn_config.get("Generation")
    return gen["mode"], gen.get("kwargs", None)


def step_apply_multi_dnn(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Apply the appropriate multi-DNN transformation (Parallel, SelectableWeights, or PR)."""
    mode, kwargs = _resolve_multi_dnn_mode(cfg)
    if mode == "Parallel":
        if kwargs is not None:
            combine_inputs_channelwise = kwargs.get("combine_inputs_channelwise", None)
            combine_outputs_channelwise = kwargs.get("combine_outputs_channelwise", None)
        model = model.transform(MultiDNNWrapperExposeIO())
        model = model.transform(CombineInputsChannelwise()) if combine_inputs_channelwise else model
        model = (
            model.transform(CombineOutputsChannelwise()) if combine_outputs_channelwise else model
        )
    elif mode == "SelectableWeights":
        model = model.transform(ExtractSelectableWeights(**kwargs))
        model = model.transform(MultiDNNWrapperExposeIO())
    elif mode == "PartialReconfiguration":
        model = model.transform(ApplyPartialReconfiguration(**kwargs))
        model = model.transform(MultiDNNWrapperExposeIO())
    else:
        raise Exception("This Mode is not implemented")

    return model


def step_collapse_multi_dnn(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Collapse all DNNContainer subgraphs and specialize the resulting concat/split nodes."""
    model = model.transform(CollapseModels())
    model = model.transform(InferSplitIntoSplitMultiHeads())
    model = model.transform(InferConcatLayer())
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))  # For Concat and Split
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(NameNodeContainerNodes())
    return model


def step_maximize_concat_split_simd(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Set SIMD on StreamingConcat_hls nodes to match surrounding MVAU parallelism.

    The Parallel multi-DNN flow inserts SplitMultiHeads_hls (already fully
    parallel, no SIMD attribute) and StreamingConcat_hls (initialized with
    SIMD=1). This step sets SIMD on StreamingConcat_hls nodes to PE*2 of the
    upstream MVAU (x2 because the concat handles two input streams), using the
    largest valid common divisor of ChannelsPerStream that does not exceed that
    target. Falls back to the maximum common divisor if no upstream MVAU is found.
    """
    from qonnx.custom_op.registry import getCustomOp

    from finn.transformation.fpgadataflow.set_folding import common_divisors

    for node in model.graph.node:
        if node.op_type == "StreamingConcat_hls":
            node_inst = getCustomOp(node)
            channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
            valid_divisors = sorted(common_divisors(channels_per_stream))

            # Find PE of the first upstream MVAU node
            upstream_pe = None
            for inp_tensor in node.input:
                producer = model.find_producer(inp_tensor)
                if producer is not None and "MVAU" in producer.op_type:
                    upstream_pe = getCustomOp(producer).get_nodeattr("PE")
                    break

            if upstream_pe is not None:
                # x2: the concat must handle two input streams simultaneously
                target_simd = upstream_pe * 2
                valid = [d for d in valid_divisors if d <= target_simd]
                simd = int(max(valid)) if valid else int(min(valid_divisors))
            else:
                # Fallback: use the maximum common divisor
                simd = int(max(valid_divisors))

            node_inst.set_nodeattr("SIMD", simd)
    return model
