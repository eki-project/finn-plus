"""Integrates ONNX Passes and ONNX Script passes into the FINN build steps."""

# Transformation bases from ONNX Passes to simplify setup and configuration of
# transformation passes
# Constant value and shapes are always expressed in numpy compatible format, so
# we use numpy to operate on those
import numpy as np

# ONNX Passes and ONNX Script infrastructure is based on ONNX IR to interact
# with the model, graph, nodes and values
import onnx_ir as ir

# YAML for loading layout assumption/conversion configuration from file
import yaml

# ONNX Passes provides onnxruntime-executable reference implementations of
# custom operators which we need to transplant back into the QONNX domain
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
from onnx_passes.ops import inject_custom_ops

# Make custom Im2Col operator available for convolution lowering
from onnx_passes.ops.im2col import Im2Col  # noqa: Used indirectly via registry
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN

# Collects named passes from the ONNX Passes registry
from onnx_passes.passes import collect
from onnx_passes.passes.base import RewriteRulePass, Transformation

# Utility testing IR values for being constant (or initializers) tensors
from onnx_passes.passes.util import is_constant

# QONNX datatype annotations for quantized tensors
from qonnx.core.datatype import DataType

# QONNX representation wrapper of ONNX models is used on the interface side to
# bridge between the FINN and the new ONNX IR representation
from qonnx.core.modelwrapper import ModelWrapper

# FINN steps are configured via a global configuration object passed into each
# step
from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType


def _make_pass_config(cfg: DataflowBuildConfig):
    """Creates ONNX Passes configuration from FINN build configuration."""
    # If specified, load data layout annotations from file
    if cfg.layouts_config_file is not None:
        with open(cfg.layouts_config_file, "r") as file:
            layouts = yaml.safe_load(file)
    # Otherwise assume emtpy layout annotations
    else:
        layouts = {}

    # Construct configuration dictionary with subset of options from the
    # DataflowBuildConfig and some other ONNX Passes specific options
    return {
        # Reference data for verification and analysis: Inputs, expected
        # outputs, ...
        "reference": {"inp": [cfg.verify_input_npy], "out": [cfg.verify_expected_output_npy]},
        # Configuration ONNX Runtime used for model evaluation during
        # verification and analysis passes - see the ONNX Runtime API
        # documentation for details
        "onnxruntime": {
            # Execution providers for accelerated inference
            "providers": [["CPUExecutionProvider", {}]],
            # Produce a full execution context dump
            "full_context_dump": cfg.verify_save_full_context,
        },
        # Configuration of model verification methods
        "verify": {
            # Tolerance-based verification, parameters passed to
            # np.allclose(...)
            "tolerance": {"atol": cfg.verification_atol, "rtol": cfg.verification_rtol}
        }
        if VerificationStepType.PASSES_FRONTEND in cfg._resolve_verification_steps()
        else {},  # noqa: protected
        # Configuration of the model checker pass: Options according to the ONNX
        # IR reference: https://onnx.ai/ir-py/api/ir_passes_common.html
        "model_checker": {"full_check": True},
        # Configuration of logging and verbosity
        "logging": {
            # Enable all passes to print a message when entering/leaving
            # TODO: control from build config or logging level
            "verbose": False
        },
        # Forward layout configuration loaded from file
        "layouts": layouts,
    }


def _apply_passes(model: ir.Model, passes: list[str], cfg: dict, state: dict):
    """Resolves and applies the list of passes to the ONNX model."""

    # Collect and instantiate all ONNX IR passes from the sequence by name and
    # connect each pass to the shared configuration and state dictionary
    passes = [cls(cfg, state) for cls in collect(passes)]
    # Pass manager instance which repeatedly runs the sequence of passes on
    # the model and evaluates pre- and post-conditions of each pass, e.g.,
    # for automatic verification.
    passes = ir.passes.PassManager(passes=passes, steps=1)
    # Inject custom operator ONNX functions into the model before applying the
    # configured pass sequence
    return passes(inject_custom_ops(model)).model


def prepare(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Prepares a model to be processed by ONNX Passes."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}
    # Imports the QONNX operators (if present) into the custom domain and
    # convert data layouts at the input/output if configured
    passes = ["import-qonnx", "convert-layouts", "shape-inference", "checker"]

    # Apply passes and serialize the resulting ONNX IR format back to ONNX proto
    # wrapped by QONNX
    return ModelWrapper(ir.to_proto(_apply_passes(model, passes, cfg, state)))


def inline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Applies ONNX Passes inlining transformations."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}
    # Operator inlining passes and shape annotations
    passes = [
        # Expresses QONNX Quant nodes as rounding, clipping and scaling
        "inline-qonnx",
        # Expresses batchnorm as affine scale and bias
        "inline-batchnorm",
        # Expresses Gemm as MatMul (+ bias and transposes)
        "inline-gemm",
        # Expresses Conv as Im2Col + MatMul (+ bias and transposes)
        "lower-conv",
        # Expresses pooling as Im2Col + Reshape + Reduce* (+ transposes)
        "lower-pooling",
        # Adds shape annotations
        "shape-inference",
        # Make sure the model is still valid
        "checker",
    ]

    # Apply passes and serialize the resulting ONNX IR format back to ONNX proto
    # wrapped by QONNX
    return ModelWrapper(ir.to_proto(_apply_passes(model, passes, cfg, state)))


def streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Applies ONNX Passes streamlining transformations."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}
    # Streamlining and threshold conversion passes
    passes = ["streamline-thresholds", "streamline", "checker"]

    # Apply passes and serialize the resulting ONNX IR format back to ONNX proto
    # wrapped by QONNX
    return ModelWrapper(ir.to_proto(_apply_passes(model, passes, cfg, state)))


class _ExportThresholdsToFINN(Transformation, RewriteRulePass):
    """Exports MultiThreshold representation from ONNX Passes to FINN format."""

    def pattern(self, op, x, thresholds, weights):
        """Target pattern to match."""

        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def check(self, op, x, thresholds, weights):
        """Match condition."""

        # Threshold parameter tensors must be constant, otherwise compatibility
        # with FINN cannot be checked...
        # TODO: Extend this to support non-constant thresholds to support
        #  runtime-writable parameters?
        if not is_constant(thresholds) or not is_constant(weights):
            return False

        # FINN does not support weighted, i.e., non-monotonic or non-unit step
        # thresholds, at the moment
        if np.any(ir.convenience.get_const_tensor(weights).numpy() != 1):
            return False

        # FINN only supports at most per-channel (last axis) granularity for
        # thresholds, all leading dimensions must have size 1
        if np.any(np.asarray(thresholds.shape[:-2]) != 1):
            return False

        # Matched format is supported by QONNX and FINN
        return True

    def rewrite(self, op, x, thresholds, weights):
        """Replacement pattern."""

        # Remove leading dimensions from the thresholds parameter tensor as
        # expected by QONNX
        thresholds = ir.convenience.get_const_tensor(thresholds).numpy()
        thresholds = thresholds.reshape(thresholds.shape[-2:])

        # QONNX requires per-tensor thresholds explicitly marked as 1xN shape
        # Needs to be checked and corrected here due to effects of the un-
        # broadcasting transformation
        if len(thresholds.shape) < 2:
            thresholds = thresholds.reshape((1, -1))

        # Infer the output bitwidth based on the number of thresholds
        out_dtype = f"UINT{int(np.ceil(np.log2(thresholds.shape[-1] + 1)))}"

        # Create a new constant operator for the squeezed thresholds input
        thresholds = op.Constant(value=ir.tensor(thresholds))

        # Generate daty layouts with unknows up to the final axis, which is the
        # known channel axis
        layout = (len(x.shape) - 1) * "." + "C"

        # Custom operator attributes according to QONNX: currently QONNX
        # defaults to NCHW layout and converts later, while the new flow
        # already exports NHWC layout (not entirely true, appropriate layout
        # conversion needs to be inserted)
        attributes = {"out_dtype": out_dtype, "data_layout": layout}

        # Replacement pattern: MultiThreshold operator in QONNX domain without
        # weights and with explicit datatype attribute
        return op.MultiThreshold(x, thresholds, **attributes, _domain=QONNX_DOMAIN)


def _export_thresholds_to_finn(model: ir.Model):
    """Exports MultiThreshold representation from ONNX Passes to FINN format."""
    return _ExportThresholdsToFINN(config={}, state={})(model).model


class _ExportIm2ColToFINN(Transformation, RewriteRulePass):
    """Exports Im2Col representation from ONNX Passes to FINN format."""

    def pattern(self, op, x, indices, dilations, kernel_shape, strides):
        """Target pattern to match."""

        return op.Im2Col(
            # Proper input and auxiliary index input holding the access pattern
            x,
            indices,
            # Attributes from which the access pattern ca be re-derived
            dilations=dilations,
            kernel_shape=kernel_shape,
            strides=strides,
            # Part of the ONNX Passes custom domain
            _domain=CUSTOM_DOMAIN,
        )

    def check(self, op, x, indices, dilations, kernel_shape, strides):
        """Match condition."""

        # QONNX needs statically annotated input shape as this will be turned
        # into an attribute of the node
        return x.shape is not None and x.shape.is_static()

    def rewrite(self, op, x, indices, dilations, kernel_shape, strides):
        """Replacement pattern."""

        # Convert attributes to format required by QONNX
        attributes = {
            # TODO: Apparently QONNX needs the shape as a string...
            "input_shape": "({})".format(",".join(map(str, x.shape.numpy()))),
            # Remaining attributes are named differently but accepted as lists
            "dilations": dilations.as_ints(),
            "kernel_size": kernel_shape.as_ints(),
            "stride": strides.as_ints(),
            # Padding attributes left as defaults, i.e., no padding, as ONNX
            # Passes makes padding explicit and standalone
            # "pad_amount":..., "pad_value":...,
            # ONNX Passes never generates a depthwise inout generator, grouped
            # convolutions are split explicitly
            # "depthwise": 0,
        }

        # Omit precomputed access pattern and transplant into the QONNX domain
        return op.Im2Col(x, **attributes, _domain=QONNX_DOMAIN)


def _export_im2col_to_finn(model: ir.Model):
    """Exports Im2Col representation from ONNX Passes to FINN format."""
    return _ExportIm2ColToFINN(config={}, state={})(model).model


def _infer_qonnx_datatypes(model: ModelWrapper):
    """Adds QONNX datatypes to a model by inferring types from values."""

    # Try inferring new datatype annotations for all tensors in the model
    for name in model.get_all_tensor_names():
        # Only apply datatype inference on initializer tensors, for all other
        # tensors there is no mechanism to tests whether all values are integer
        if (init := model.get_initializer(name)) is not None:
            # Do not change annotation if already annotated as some integer
            if not model.get_tensor_datatype(name).is_integer():
                # If all values are integers, i.e., do not change when rounding
                # and casting to integer, infer this as an integer tensor
                if np.all(np.asarray(np.round(init), dtype=np.int64) == init):
                    # Set to some large integer type, should be minimized later
                    model.set_tensor_datatype(name, DataType["INT64"])

    # Potentially modified model, still as QONNX ModelWrapper, this step
    # operates in-place modifying the original
    return model


def export(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Converts the model back to the FINN compatible format."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}

    # Cleanup passes ensuring threshold compatibility with the FINN format
    passes = [
        # Before exporting back to the FINN format, try to make all thresholds
        # per-channel at the expense of extra per-element additions
        "decompose-thresholds",
        # One more time cleanup the model and fill in missing shape annotations,
        # also make sure the model is still valid ONNX at this point
        "shape-inference",
        "fold-constants",
        "eliminate",
        "cleanup",
        "checker",
        "verify",
    ]

    # Apply passes sequence with configuration and global state, stay within
    # ONNX IR format here
    model = _apply_passes(model, passes, cfg, state)

    # Export custom operators to the FINN representation
    model = _export_thresholds_to_finn(model)
    model = _export_im2col_to_finn(model)

    # Finalize the data layout annotations and get rid of custom functions:
    # more of a workaround as qonnx execution does not understand these...
    model = _apply_passes(model, ["absorb-layouts", "inline-functions"], {}, {})

    # Serialize the resulting ONNX IR format back to ONNX proto wrapped by QONNX
    # and add quantization datatype annotations
    return _infer_qonnx_datatypes(ModelWrapper(ir.to_proto(model)))


def step_passes_frontend(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Meta build step calling the ONNX Passes steps in the expected order."""

    model = prepare(model, cfg)
    model = inline(model, cfg)
    model = streamline(model, cfg)
    model = export(model, cfg)

    return model
