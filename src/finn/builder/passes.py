"""Integrates ONNX Passes and ONNX Script passes into the FINN build steps."""

# Transformation bases from ONNX Passes to simplify setup and configuration of
# transformation passes
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Collects named passes from the ONNX Passes registry
from onnx_passes.passes import collect

# Utility testing IR values for being constant (or initializers) tensors
from onnx_passes.passes.util import is_constant

# Makes custom QONNX import and inlining passes available
import onnx_passes.passes.imports.qonnx  # noqa: Used indirectly via registry
import onnx_passes.passes.inline.qonnx  # noqa: Used indirectly via registry

# ONNX Passes provides onnxruntime-executable reference implementations of
# custom operators which we need to transplant back into the QONNX domain
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN, inject_custom_ops
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN

# QONNX representation wrapper of ONNX models is used on the interface side to
# bridge between the FINN and the new ONNX IR representation
from qonnx.core.modelwrapper import ModelWrapper
# QONNX datatype annotations for quantized tensors
from qonnx.core.datatype import DataType

# FINN steps are configured via a global configuration object passed into each
# step
from finn.builder.build_dataflow_config import DataflowBuildConfig

# ONNX Passes and ONNX Script infrastructure is based on ONNX IR to interact
# with the model, graph, nodes and values
import onnx_ir as ir

# Constant value and shapes are always expressed in numpy compatible format, so
# we use numpy to operate on those
import numpy as np


def _make_pass_config(cfg: DataflowBuildConfig):
    """Creates ONNX Passes configuration from FINN build configuration."""
    return {
        # Reference data for verification and analysis: Inputs, expected
        # outputs, ...
        "reference": {
            "inp": [cfg.verify_input_npy],
            "out": [cfg.verify_expected_output_npy]
        },
        # Configuration ONNX Runtime used for model evaluation during
        # verification and analysis passes - see the ONNX Runtime API
        # documentation for details
        "onnxruntime": {
            # Execution providers for accelerated inference
            "providers": [
                ["CPUExecutionProvider", {}]
            ],
            # Produce a full execution context dump
            "full_context_dump": cfg.verify_save_full_context
        },
        # Configuration of model verification methods
        "verify": {
            # Tolerance-based verification, parameters passed to
            # np.allclose(...)
            "tolerance": {
                "atol": 1.0e-6
            }
        },
        # Configuration of the model checker pass: Options according to the ONNX
        # IR reference: https://onnx.ai/ir-py/api/ir_passes_common.html
        "model_checker": {
            "full_check": True
        },
        # Configuration of logging and verbosity
        "logging": {
            # Enable all passes to print a message when entering/leaving
            "verbose": cfg.verbose
        }
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
    # Load ONNX IR to modify/analyze from the model file - format should
    # be inferred and apply the sequence of passes
    return passes(inject_custom_ops(model)).model


def prepare(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Prepares a model to be processed by ONNX Passes."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}
    # Imports the QONNX operators (if present) into the custom domain
    passes = ["import-qonnx", "shape-inference", "checker"]

    # Apply passes and serialize the resulting ONNX IR format back to ONNX proto
    # wrapped by QONNX
    return ModelWrapper(ir.to_proto(_apply_passes(model, passes, cfg, state)))


def inline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Applies ONNX Passes inlining transformations."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}
    # Imports the QONNX operators (if present) into the custom domain
    passes = ["inline-qonnx", "inline-batchnorm", "shape-inference", "checker"]

    # Apply passes and serialize the resulting ONNX IR format back to ONNX proto
    # wrapped by QONNX
    return ModelWrapper(ir.to_proto(_apply_passes(model, passes, cfg, state)))


def streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Applies ONNX Passes streamlining transformations."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Create configuration for all passes and assume initially empty state
    cfg, state = _make_pass_config(cfg), {}
    # Imports the QONNX operators (if present) into the custom domain
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

        # Infer the output bitwidth based on the number of thresholds
        out_dtype = f"UINT{int(np.ceil(np.log2(thresholds.shape[-1] + 1)))}"

        # Create a new constant operator for the squeezed thresholds input
        thresholds = op.Constant(value=ir.tensor(thresholds))

        # Replacement pattern: MultiThreshold operator in QONNX domain without
        # weights and with explicit datatype attribute
        return op.MultiThreshold(
            x, thresholds, out_dtype=out_dtype, _domain=QONNX_DOMAIN
        )


def _export_thresholds_to_finn(model: ir.Model):
    """Exports MultiThreshold representation from ONNX Passes to FINN format."""
    return _ExportThresholdsToFINN(config={}, state={})(model).model


def _export_im2col_to_finn(model: ir.Model):
    """Exports Im2Col representation from ONNX Passes to FINN format."""
    return model  # TODO: Implement the export transformation...


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


def export(model: ModelWrapper, _: DataflowBuildConfig) -> ModelWrapper:
    """Converts the model back to the FINN compatible format."""

    # Deserialize ONNX proto representation wrapped by QONNX to ONNX IR format
    model = ir.from_proto(model.model)

    # Export custom operators to the FINN representation
    model = _export_thresholds_to_finn(model)
    model = _export_im2col_to_finn(model)

    # Serialize the resulting ONNX IR format back to ONNX proto wrapped by QONNX
    # and add quantization datatype annotations
    return _infer_qonnx_datatypes(ModelWrapper(ir.to_proto(model)))
