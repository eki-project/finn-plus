"""Integrates ONNX Passes and ONNX Script passes into the FINN build steps."""

# Transformation bases from ONNX Passes to simplify setup and configuration of
# transformation passes
# Constant value and shapes are always expressed in numpy compatible format, so
# we use numpy to operate on those
import numpy as np

# ONNX Passes and ONNX Script infrastructure is based on ONNX IR to interact
# with the model, graph, nodes and values
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Makes custom QONNX import and inlining passes available
import onnx_passes.passes.imports.qonnx

# YAML for loading layout assumption/conversion configuration from file
import yaml

# ONNX Passes provides onnxruntime-executable reference implementations of
# custom operators which we need to transplant back into the QONNX domain
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
from onnx_passes.ops import inject_custom_ops

# Make custom Im2Col operator available for convolution lowering
from onnx_passes.ops.im2col import Im2Col  # noqa # noqa: Used indirectly via registry
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN

# Collects named passes from the ONNX Passes registry
from onnx_passes.passes import collect
from onnx_passes.passes.base import RewriteRulePass, RewriteRuleSetPass, Transformation

# Collecting node attributes with optional defaults
# Utility testing IR values for being constant (or initializers) tensors
from onnx_passes.passes.util import collect_attrs, is_constant
from pathlib import Path

# QONNX datatype annotations for quantized tensors
from qonnx.core.datatype import DataType

# QONNX representation wrapper of ONNX models is used on the interface side to
# bridge between the FINN and the new ONNX IR representation
from qonnx.core.modelwrapper import ModelWrapper
from typing import Callable, cast

# FINN steps are configured via a global configuration object passed into each
# step
from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType

import onnx_passes.passes.inline.qonnx  # noqa


def _make_pass_config(
    cfg: DataflowBuildConfig,
) -> dict[str, dict[str, list[Path] | bool | float | list[list[str | dict]] | dict]]:
    """Create ONNX Passes configuration from FINN build configuration."""
    # If specified, load data layout annotations from file
    if cfg.layouts_config_file is not None:
        with cfg.layouts_config_file.open("r") as file:
            layouts: dict = yaml.safe_load(file)
    # Otherwise assume emtpy layout annotations
    else:
        layouts: dict = {}

    # Construct configuration dictionary with subset of options from the
    # DataflowBuildConfig and some other ONNX Passes specific options
    return {
        # Reference data for verification and analysis: Inputs, expected
        # outputs, ...
        "reference": {
            "inp": [Path(cfg.verify_input_npy)],
            "out": [Path(cfg.verify_expected_output_npy)],
        },
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
        if VerificationStepType.PASSES_FRONTEND in cfg._resolve_verification_steps()  # noqa: SLF001
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


def _apply_passes(model: ir.Model, passes: list[str | type], cfg: dict, state: dict) -> ir.Model:
    """Resolve and applies the list of passes to the ONNX model."""
    # Collect and instantiate all ONNX IR passes from the sequence by name and
    # connect each pass to the shared configuration and state dictionary
    passes_list = [cls(cfg, state) for cls in collect(passes)]
    # Pass manager instance which repeatedly runs the sequence of passes on
    # the model and evaluates pre- and post-conditions of each pass, e.g.,
    # for automatic verification.
    passes_manager = ir.passes.PassManager(passes=passes_list, steps=1)
    # Inject custom operator ONNX functions into the model before applying the
    # configured pass sequence
    return passes_manager(inject_custom_ops(model)).model


def prepare(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Prepare a model to be processed by ONNX Passes."""
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
        # Move reduction axes to the back of reduction operators to allow hardware implementation
        "inline-move-reduce-axis",
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


class _MoveReduceAxisToBack(Transformation, RewriteRuleSetPass):
    """Moves the reduce axis to the back of the tensor."""

    __OP__: Callable
    __OPAX__: Callable

    def _normalize_axes(self, axes, ndim: int):
        """Convert axis spec to sorted tuple of unique positive axes."""
        if axes is None:
            return tuple(range(ndim))
        if isinstance(axes, int):
            axes = (axes,)

        out: list[int] = []
        for a in axes:
            if a < 0:
                a += ndim
            if not (0 <= a < ndim):
                raise ValueError(f"Axis {a} out of range for ndim={ndim}")
            out.append(int(a))

        if len(set(out)) != len(out):
            raise ValueError(f"Duplicate axes in {axes}")

        return tuple(sorted(out))

    def pattern(self):
        # ReduceSum optionally receives axes as a second input.
        return [
            lambda op, x, axes: self.__OPAX__(op, x, axes),
            lambda op, x: self.__OP__(op, x),
        ]

    def check(self):
        """Match condition."""

        def _check(
            op,
            x: ir.Value,
            axes: ir.Value | None = None,
        ):
            # QONNX needs statically annotated input shape as this will be turned
            # into an attribute of the node
            if not (x.shape is not None and x.shape.is_static()):
                return False
            if axes is not None:
                if not (axes.shape is not None and axes.shape.is_static()):
                    return False
            else:
                return False
            # Check if the axes are already at the back of the tensor
            # Last dimension (C) is handled differently and can be ignored
            shape = cast("ir.Shape", x.shape)
            ax = ir.convenience.get_const_tensor(axes)
            if ax is None:
                return False
            ax = self._normalize_axes(ax.numpy(), shape.rank())
            if ax[-1] == shape.rank() - 1:
                ax = ax[:-1]
            return any(a != shape.rank() - 2 - i for i, a in enumerate(reversed(ax)))

        return [
            lambda op, x, axes, *args, **kwargs: _check(op, x, axes),
            lambda op, x, *args, **kwargs: _check(op, x),
        ]

    def rewrite(self):
        """Replacement pattern."""

        def _replace(
            op,
            x: ir.Value,
            y: ir.Value,
            axes: ir.Value | None = None,
        ):
            # Defaults according to ONNX operators reference documentation:
            #   https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html
            attributes = collect_attrs(
                y.producer(),
                {
                    "keepdims": (ir.AttributeType.INT, 1),
                    "noop_with_empty_axes": (ir.AttributeType.INT, 0),
                },
            )

            # Implementation for replacement logic
            if axes is None:
                return op

            ax = ir.convenience.get_const_tensor(axes)
            if ax is None:
                raise ValueError("Axes must be a constant tensor for replacement.")
            ax = ax.numpy()
            shape = cast("ir.Shape", x.shape)
            in_shape = shape.numpy()
            ndim = shape.rank()
            red_axes = self._normalize_axes(ax, ndim)
            keep_axes = tuple(i for i in range(ndim) if i not in red_axes)
            last_axis_reduction = red_axes[-1] == ndim - 1
            if last_axis_reduction:
                perm = keep_axes + red_axes
            else:
                perm = keep_axes[:-1] + red_axes + (keep_axes[-1],)
            print(f"AxesPermutation: {perm}")
            shuf_shape = tuple(in_shape[p] for p in perm)

            reshuffled_axes = []
            for a in red_axes:
                # Find the new position of the axis after permutation
                new_a = perm.index(a)
                reshuffled_axes.append(new_a)

            ret = self.__OPAX__(
                op,
                op.Reshape(
                    op.Transpose(x, perm=ir.Attr("perm", ir.AttributeType.INTS, perm)),
                    op.Constant(value_ints=shuf_shape),
                ),
                axes=op.Constant(value=ir.tensor(reshuffled_axes, name="axes")),
                keepdims=attributes["keepdims"],
                noop_with_empty_axes=attributes["noop_with_empty_axes"],
            )
            if not bool(attributes["keepdims"].value):
                return ret
            # If keepdims is True, we need to reshuffle + reshape the
            # output back to the original shape
            keep_shape = tuple(in_shape[a] for a in keep_axes)
            y_pre_shape = keep_shape + tuple(1 for _ in red_axes)

            inv_perm = tuple(np.argsort(perm))
            target_shape = tuple(1 if i in red_axes else in_shape[i] for i in range(ndim))

            return op.Reshape(
                op.Transpose(
                    op.Reshape(ret, op.Constant(value_ints=y_pre_shape)),
                    perm=ir.Attr("perm", ir.AttributeType.INTS, inv_perm),
                ),
                op.Constant(value_ints=target_shape),
            )

        # Omit precomputed access pattern and transplant into the QONNX domain
        return [
            lambda op, x, axes, reduced: _replace(op, x, reduced, axes),
            lambda op, x, reduced: _replace(op, x, reduced),
        ]


@passes.verify.tolerance
@passes.register("inline-move-reduce-axis")
class MoveReduceSumAxis(_MoveReduceAxisToBack):
    def __OP__(_, op, x, **kwargs):
        return op.ReduceSum(x, _outputs=["reduced"], **kwargs)

    def __OPAX__(_, op, x, axes, **kwargs):
        return op.ReduceSum(x, axes, _outputs=["reduced"], **kwargs)


@passes.verify.tolerance
@passes.register("inline-move-reduce-axis")
class MoveReduceMinAxis(_MoveReduceAxisToBack):
    def __OP__(_, op, x, **kwargs):
        return op.ReduceMin(x, _outputs=["reduced"], **kwargs)

    def __OPAX__(_, op, x, axes, **kwargs):
        return op.ReduceMin(x, axes, _outputs=["reduced"], **kwargs)


@passes.verify.tolerance
@passes.register("inline-move-reduce-axis")
class MoveReduceMaxAxis(_MoveReduceAxisToBack):
    def __OP__(_, op, x, **kwargs):
        return op.ReduceMax(x, _outputs=["reduced"], **kwargs)

    def __OPAX__(_, op, x, axes, **kwargs):
        return op.ReduceMax(x, axes, _outputs=["reduced"], **kwargs)


@passes.verify.tolerance
@passes.register("inline-move-reduce-axis")
class MoveReduceProdAxis(_MoveReduceAxisToBack):
    def __OP__(_, op, x, **kwargs):
        return op.ReduceProd(x, _outputs=["reduced"], **kwargs)

    def __OPAX__(_, op, x, axes, **kwargs):
        return op.ReduceProd(x, axes, _outputs=["reduced"], **kwargs)


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
    model.save("test_move_axis_inline.onnx")
    print("Saved model after inline passes to test_move_axis_inline.onnx")
    model = streamline(model, cfg)
    model = export(model, cfg)

    return model
