"""Ad-hoc build steps for transformer models, covering hardware conversion and
folding configuration for operators not yet handled by the standard FINN
pipeline, in particular scaled dot-product attention.
"""

# QONNX arbitrary precision datatype annotations
# Use "greatest common divisor" in folding calculations
import contextlib
import math

# Custom folding is specified and saved as YAML instead of JSON
import yaml
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Get QONNX CustomOp instance for the given ONNX node, if it exists.
from qonnx.custom_op.registry import getCustomOp

# Cleanup transformation giving unique names to all nodes reflecting the op-type
# and graph ordering
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes

# QONNX type and shape inference transformations
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING, cast

# Configuration for FINN dataflow builds passed through the build steps by the
# FINN frontend
from finn.builder.build_dataflow_config import DataflowBuildConfig

# ONNX operator to FINN HWCustomOp conversion steps, also inferring custom-ops
# from patterns of operators
from finn.transformation.fpgadataflow import convert_to_hw_layers as hardware
from finn.transformation.fpgadataflow.attention import InferScaledDotProductAttention
from finn.transformation.fpgadataflow.replicate_stream import InferReplicateStream

# Reuse FINN auto-folding functionality to build folding of attention operators
from finn.transformation.fpgadataflow.set_folding import (
    AnnotateCycles,
    SetFolding,
    common_divisors,
    dataflow_performance,
)
from finn.transformation.general import ApplyConfig
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

# FINN Streamlining transformations still required during hardware conversion
from finn.transformation.streamline import RoundAndClipThresholds
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hls.attention_hls import ScaledDotProductAttention_hls


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Convert a streamlined transformer ONNX model to FINN hardware operators.

    Applies the full sequence of hardware-conversion transformations, including
    split/concat infrastructure, pooling, convolution input generation, fused
    scaled dot-product attention, MVAUs/VVAUs, thresholding, lookup layers, and
    elementwise binary operators.  A cleanup pass (type/shape inference, unique
    node naming) is executed at the end.

    Args:
        model: QONNX ``ModelWrapper`` of the streamlined model to convert.
        cfg: FINN ``DataflowBuildConfig`` controlling optional steps such as
            ``standalone_thresholds``.

    Returns:
        The transformed ``ModelWrapper`` with all supported operators replaced
        by FINN HW custom-ops.
    """
    # Start with inferring splitting and concatenating infrastructure operators
    # as these also address QONNX type inference defects.
    model = model.transform(hardware.InferSplitLayer())
    model = model.transform(hardware.InferConcatLayer())

    # Convert pooling and convolution-related operators next as these might
    # affect type and shape inference for subsequent transformations.
    model = model.transform(hardware.InferPool())
    model = model.transform(hardware.InferPoolFromReduce())
    model = model.transform(hardware.InferConvInpGen())
    model = model.transform(hardware.InferFMPadding())

    # Infer fused scaled dot-product attention before inferring standalone
    # thresholds and MVUs
    model = model.transform(InferScaledDotProductAttention())

    # Usual FINN hardware conversion transformations: Start with standalone
    # thresholds to avoid fusing these into MVAUs, from there on the order does
    # not really matter.
    if cfg.standalone_thresholds:
        model = model.transform(hardware.InferThresholdingLayer())

    model = model.transform(hardware.InferBinaryMatrixVectorActivation())
    model = model.transform(hardware.InferQuantizedMatrixVectorActivation())
    model = model.transform(hardware.InferVectorVectorActivation())
    model = model.transform(hardware.InferThresholdingLayer())
    model = model.transform(hardware.InferLabelSelectLayer())
    model = model.transform(hardware.InferLabelSelectLayer())

    # Inferring Gather as Lookup layers needs some special treatment: The ONNX
    # standard requires signed-integer index inputs whereas FINN assumes float
    # container types annotated as unsigned integer.
    for _index, node in enumerate(model.graph.node):
        # If this is a Gather node, force the input (index) type annotation
        if node.op_type == "Gather":
            # Force to unsigned 64-bit integer for now
            model.set_tensor_datatype(node.input[1], DataType["UINT64"])
            # Get the value info for the input tensor to have access to the ONNX
            # datatype of the tensor
            value_info = model.get_tensor_valueinfo(node.input[1])
            if value_info is None:
                raise FINNUserError(
                    f"Value info for Gather index input {node.input[1]}"
                    " not found, cannot set datatype."
                )
            # Force the container datatype of the input to be a float
            value_info.type.tensor_type.elem_type = 1

    # Now Gather operators should be inferred as Lookup layers
    model = model.transform(hardware.InferLookupLayer())

    # TODO: Theses two should actually be considered as streamlining of already
    #  lowered operator representations, but pooling conversion above might
    #  introduce new transposes to the graph
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(RemoveCNVtoFCFlatten())

    # Implement remaining elementwise binary operators as hardware operators,
    # including floating-point operators, excluding the final dequantizer scale.
    model = model.transform(
        hardware.InferElementwiseBinaryOperation(
            hardware.InferElementwiseBinaryOperation.reject_output_dequant
        )
    )
    # Any remaining reshape operator must be implemented to keep the graph valid
    # while also not breaking the chain of FINN operators.
    model = model.transform(hardware.InferReshape())
    # Explicitly replicate stream connections between layers as hardware does
    # not allow multiple consumer of a single AXI stream.
    model = model.transform(InferReplicateStream())

    # Cleanup the graph after hardware conversion by redoing type and shape
    # inference. There is also more potential for rounding thresholds after
    # addressing QONNX type inference defects regarding Split and Concat.
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(RoundAndClipThresholds())

    # TODO: Giving unique node names apparently is not part of the default
    #  cleanup transformations...?
    return model.transform(GiveUniqueNodeNames())


def _set_folding_attention(model: ModelWrapper, target_cycles_per_frame: int) -> ModelWrapper:
    """Set folding parameters for all ScaledDotProductAttention_hls nodes.

    Iterates the graph and, for every attention operator, greedily increases
    parallelism first along the embedding dimension and then along the sequence
    dimension until ``target_cycles_per_frame`` is met or the maximum
    parallelism is reached.  Annotates cycle estimates for all operators
    afterwards.

    Args:
        model: ``ModelWrapper`` containing the dataflow graph.
        target_cycles_per_frame: Cycle budget per input frame.

    Returns:
        The ``ModelWrapper`` with updated folding attributes and cycle
        annotations.
    """
    # Run over all nodes in the model graph to look for attention operators,
    # which are currently not handled by the SetFolding transformation
    for _index, node in enumerate(model.graph.node):
        if node.op_type == "ScaledDotProductAttention_hls":
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = cast("ScaledDotProductAttention_hls", getCustomOp(node))

            # Extract sequence length and embedding dimension from the
            # operator attributes
            qkdim, _qlen, vdim, kvlen = inst.shapes

            # Initialize folding to sequential processing. As the same
            # folding is applied to query, key and value embeddings, but
            # these might be different, use the greatest common divisor.
            inst.set_nodeattr("EmbFold", math.gcd(cast("int", qkdim), cast("int", vdim)))
            inst.set_nodeattr("SeqFold", cast("int", kvlen))

            # Try to unfold along the embedding dimension first, increasing
            # parallelism in steps following the common divisors the inputs.
            for fold in reversed(common_divisors([cast("int", qkdim), cast("int", vdim)])):
                # Configure the folding attribute
                inst.set_nodeattr("EmbFold", int(fold))
                # Check if this is sufficient to meet the cycles target
                if cast("int", inst.get_exp_cycles()) <= target_cycles_per_frame:
                    break

            # Try to unfold along the sequence dimension next, increasing
            # parallelism in steps divisors of the key and value sequence.
            for fold in reversed(common_divisors([cast("int", kvlen)])):
                # Configure the folding attribute
                inst.set_nodeattr("SeqFold", int(fold))
                # Check if this is sufficient to meet the cycles target
                if cast("int", inst.get_exp_cycles()) <= target_cycles_per_frame:
                    break

    # Annotate cycles estimates for all operators n the model
    model = model.transform(GiveUniqueNodeNames())
    return model.transform(AnnotateCycles())


def step_set_folding(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Configure the folding (parallelism) of all dataflow operators in *model*.

    When ``cfg`` specifies a throughput target the function first applies
    ``_set_folding_attention`` for attention operators and then FINN's
    ``SetFolding`` for the remaining operators.  An optional two-pass relaxation
    step re-runs attention folding at the cycle count of the identified
    bottleneck.  When only a ``folding_config_file`` is provided (no FPS
    target), the config is applied directly without any auto-folding.

    After auto-folding, the resolved configuration is written to
    ``<output_dir>/auto_folding_config.yaml``.  If ``cfg.folding_config_file``
    is also set its entries are applied on top to override resource-style
    defaults (``mem_mode``, ``ram_style``, etc.).

    Args:
        model: ``ModelWrapper`` containing the dataflow graph.
        cfg: FINN ``DataflowBuildConfig`` with throughput targets and optional
            folding overrides.

    Returns:
        The ``ModelWrapper`` with folding attributes fully configured.

    Raises:
        FINNUserError: If neither a throughput target nor a folding config file
            is provided.
    """
    # Resolve the target cycles per from the build configuration, considering
    # clock and target throughput
    target_cycles_per_frame = cfg._resolve_cycles_per_frame()  # noqa: SLF001

    # If no target_cycles_per_frame is given, assume we are provided with a full
    # folding config JSON (including parallelism settings): Apply it and skip remaining step
    if target_cycles_per_frame is None:
        if cfg.folding_config_file is None:
            raise FINNUserError("Neither FPS target nor folding config provided.")
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.folding_config_file))
        return model

    # Set folding to target cycles for all attention operators in the model
    model = _set_folding_attention(model, target_cycles_per_frame)

    # Use FINN auto-folding to configure all other operators to reach the
    # same target cycles
    model = model.transform(
        SetFolding(target_cycles_per_frame, cfg.mvau_wwidth_max, cfg.folding_two_pass_relaxation)
    )

    # Two-pass relaxation for attention operators: Redo folding settings
    # with lower target based on cycles of the slowest operator
    if cfg.folding_two_pass_relaxation:
        # Extract dataflow performance from the graph (has been annotated by
        # SetFolding transformation)
        perf_dict = model.analysis(dataflow_performance)
        # Estimated cycles are above the target, so this is bottlenecked by
        # some operator
        if cast("int", perf_dict["max_cycles"]) > target_cycles_per_frame:
            # Set folding to lower target cycles for all attention operators
            # in the model
            model = _set_folding_attention(model, cast("int", perf_dict["max_cycles"]))

    # TODO: The following export of auto_folding_config.yaml is largely redundant
    # because later steps generate a final_hw_config.json, so it may be removed

    # Hardware attributes to be extracted from each node
    hw_attrs = {
        "PE",
        "SIMD",
        "EmbFold",
        "SeqFold",
        "parallel_window",
        "ram_style",
        "ram_style_thresholds",
        "ram_style_mask",
        "depth",
        "impl_style",
        "resType",
        "mac_resource",
        "mem_mode",
        "runtime_writeable_weights",
        "inFIFODepths",
        "outFIFODepths",
        "depth_trigger_uram",
        "depth_trigger_bram",
    }

    # Start collecting the configuration from the model graph as a dictionary
    config = {"defaults": {}}
    # Iterate all nodes in the graph keeping track of the index
    for _index, node in enumerate(model.graph.node):
        # Convert this to the custom-op instance for easy access to node
        # attributes
        inst = getCustomOp(node)
        # Prepare the node-specific configuration entry for this node
        config[node.name] = {}
        # Collect attribute values for all specified hardware attributes
        for key in hw_attrs:
            # Some hardware attributes may not be present for all nodes or
            # op-types, this will be signaled via exception
            with contextlib.suppress(AttributeError):
                # Try extracting the configuration value from the node
                # custom-op instance
                config[node.name][key] = inst.get_nodeattr(key)
        # Cleanup: If no attribute is present for this node, there is no
        # need to keep this in the configuration dictionary as there is
        # nothing to be restored later
        if not config[node.name]:
            # Remove the entry form the configuration dictionary
            del config[node.name]

    # Create/Open a YAML file to store the configuration for later reuse
    with Path(cast("str", cfg.output_dir) + "/auto_folding_config.yaml").open("w") as file:
        # Store the configuration dictionary as YAML code
        yaml.safe_dump(config, file)

    # If a folding configuration file is given, load and parse YAML and apply to
    # all nodes in the model
    # NOTE: This is intended to apply a folding YAML file that defines sensible
    # defaults for resource selection (mem_mode, ram_style, etc.) in addition to the
    # auto folding above, NOT for setting parallelism attributes (SIMD, PE, etc.)
    if cfg.folding_config_file is not None:
        with Path(cfg.folding_config_file).open() as file:
            config = yaml.safe_load(file)

        for _index, node in enumerate(model.graph.node):
            # A node should not be named "defaults"...
            assert node.name != "defaults", "Node has reserved name 'defaults'"
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Apply the per operator type default configurations to the node
            if node.op_type in config["defaults"]:
                # Run over all default options to be applied to this node
                for key, value in config["defaults"][node.op_type].items():
                    # Set the nodes attribute to the default option value
                    inst.set_nodeattr(key, value)
            # If there is an individual, node-specific configuration apply
            # this next, potentially overriding the defaults set above
            if node.name in config:
                # Run over all node-specific options to be applied to this
                # node
                for key, value in config[node.name].items():
                    # Set the nodes attribute to the option value
                    inst.set_nodeattr(key, value)

    # Model with applied auto and manual folding configuration
    return model
