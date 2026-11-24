"""Integrates ONNX Passes and ONNX Script passes into the FINN build steps."""

import numpy as np
import onnx_ir as ir
from onnx_passes.ops import DOMAIN as PASSES_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN
from onnx_passes.passes import (
    canonicalize,
    fold_constants,
    inline,
    lower,
    multithreshold,
    reorder,
    unbroadcast,
)
from onnx_passes.passes.base import RewriteRule, Sequential
from onnx_passes.passes.config import Config as PassesConfig
from onnx_passes.passes.config import LoggingConfig, VerifyConfig
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.util.exception import FINNFrontendExportError


class ExportIm2ColToFINN_v1(RewriteRule):
    """Export ONNX Passes Im2Col operator to QONNX custom operator."""

    @staticmethod
    def pattern(op, x, dilations, kernel_shape, strides):
        """Match Im2Col from the ONNX Passes custom domain."""
        return op.Im2Col(
            x,
            dilations=dilations,
            kernel_shape=kernel_shape,
            strides=strides,
            _domain=PASSES_DOMAIN,
        )

    @staticmethod
    def check(context, x, dilations, kernel_shape, strides):
        """Require a static input shape, QONNX needs it as an attribute."""
        return x.shape is not None and x.shape.is_static()

    @staticmethod
    def rewrite(op, x, dilations, kernel_shape, strides):
        """Transplant the node into the QONNX domain with QONNX attributes."""
        # The replacement must live in the QONNX domain: emitting it into the
        # ONNX Passes domain matches the pattern of this rule again and the
        # rewriter never terminates. QONNX also names the attributes
        # differently and needs the input shape as a string attribute. Padding
        # is explicit and standalone in ONNX Passes and grouped convolutions
        # are split explicitly, so the QONNX padding and depthwise attributes
        # keep their defaults.
        return op.Im2Col(
            x,
            input_shape="({})".format(",".join(map(str, x.shape.numpy()))),
            dilations=dilations.as_ints(),
            kernel_size=kernel_shape.as_ints(),
            stride=strides.as_ints(),
            _domain=QONNX_DOMAIN,
        )


def multithreshold_export_blocker(thresholds: ir.Value, weights: ir.Value) -> str | None:
    """Explain why a MultiThreshold cannot be exported to QONNX, None if it can.

    The QONNX MultiThreshold has per-channel thresholds of shape (C, N) and
    implicit unit step weights, whereas the ONNX Passes variant allows arbitrary
    broadcastable threshold shapes and explicit, arbitrary step weights. The
    reason is used both as the match condition of the export rule and for the
    error message if such an operator remains in the model.
    """
    if thresholds.shape is None or thresholds.shape.is_dynamic():
        return f"thresholds {thresholds.name!r} have no static shape"

    if np.any(np.asarray(thresholds.shape[:-2]) != 1):
        return (
            f"thresholds {thresholds.name!r} of shape {tuple(thresholds.shape)} are not"
            " per-channel, i.e., have leading dimensions other than 1"
        )

    if (tensor := ir.convenience.get_const_tensor(weights)) is None:
        return f"step weights {weights.name!r} are not constant"

    if not np.all(tensor.numpy() == 1):
        values = ", ".join(map(str, np.unique(tensor.numpy())[:8]))
        return (
            f"step weights {weights.name!r} of shape {tuple(tensor.shape)} are not all 1"
            f" (values: {values})"
        )

    return None


class ExportMultiThresholdToFINN_v1(RewriteRule):
    """Export ONNX Passes MultiThreshold operator to QONNX custom operator."""

    @staticmethod
    def pattern(op, x, thresholds, weights):
        """Match MultiThreshold from the ONNX Passes custom domain."""
        return op.MultiThreshold(x, thresholds, weights, _domain=PASSES_DOMAIN)

    @staticmethod
    def check(context, x, thresholds, weights):
        """Accept only static, per-channel thresholds with unit step weights."""
        return multithreshold_export_blocker(thresholds, weights) is None

    @staticmethod
    def rewrite(op, x, thresholds, weights):
        """Transplant the node into the QONNX domain with QONNX attributes."""
        # Extract constant parameter tensors as NumPy arrays: according to the
        # match conditions these are never None and safe to access.
        thresholds = ir.convenience.get_const_tensor(thresholds).numpy()  # noqa
        weights = ir.convenience.get_const_tensor(weights).numpy()  # noqa

        # QONNX requires per-tensor thresholds explicitly marked as 1xN shape,
        # whereas the ONNX passes variant supports N via unbroadcasting.
        if len(thresholds.shape) < 2:
            thresholds = thresholds.reshape((1, -1))

        # Infer the output bitwidth based on the number of thresholds and insert
        # the domain-transplanted replacement operator into the graph
        out_dtype = f"UINT{int(np.ceil(np.log2(thresholds.shape[-1] + 1)))}"

        return op.MultiThreshold(
            x, op.Constant(value=ir.tensor(thresholds)), out_dtype=out_dtype, _domain=QONNX_DOMAIN
        )


class ExportLayoutConverterToFINN_v1(RewriteRule):
    """Export ONNX Passes LayoutConverter operator to Transpose."""

    @staticmethod
    def pattern(op, x, assumes, perm):
        """Match LayoutConverter from the ONNX Passes custom domain."""
        return op.LayoutConverter(x, assumes=assumes, perm=perm, _domain=PASSES_DOMAIN)

    @staticmethod
    def rewrite(op, x, assumes, perm):
        """Replace by a plain Transpose with the same permutation."""
        return op.Transpose(x, perm=perm)


def _export_blocker(node: ir.Node) -> str:
    """Explain why a node from the ONNX Passes domain was not exported to QONNX."""
    inputs = node.inputs

    if node.op_type == "MultiThreshold" and len(inputs) == 3 and all(inputs):
        return multithreshold_export_blocker(inputs[1], inputs[2]) or "unknown reason"

    if node.op_type == "Im2Col" and len(inputs) == 1 and inputs[0] is not None:
        if inputs[0].shape is None or inputs[0].shape.is_dynamic():
            return f"input {inputs[0].name!r} has no static shape"
        return "unknown reason"

    return "no export rule for this operator"


def assert_exported(model: ir.Model, save_to: Path | None = None):
    """Ensure no operators from the ONNX Passes domain remain in the model.

    QONNX and FINN do not know this domain, the first transformation touching
    such a node, usually the QONNX datatype inference, fails with an obscure
    "Op not found in domain" error. Instead, raise a FINNFrontendExportError
    naming every remaining node and the reason why its export rule did not
    fire. If save_to is given, the model is saved there before raising.
    """
    remaining = [node for node in model.graph.all_nodes() if node.domain == PASSES_DOMAIN]

    if not remaining:
        return

    lines = []

    for node in remaining:
        outputs = ", ".join(repr(out.name) for out in node.outputs if out is not None)
        lines.append(f"  {node.op_type} {node.name!r} (outputs {outputs}): {_export_blocker(node)}")

    message = (
        f"{len(remaining)} operator(s) from the ONNX Passes domain {PASSES_DOMAIN!r} remain in"
        " the model after the frontend passes and cannot be handed over to QONNX/FINN:\n"
        + "\n".join(lines)
    )

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        ir.save(model, save_to)
        message += f"\nThe model as left by the passes is saved at {save_to}"

    raise FINNFrontendExportError(message)


class FINNPasses_v1(Sequential):
    """Default ONNX Passes pass sequence used by FINN."""

    passes = [
        # Cleanup and prepare the graph: Link custom operator functions and
        # normalize operators
        canonicalize,
        # Operator lowering: the most important ones are Conv to Im2Col-MatMul
        # and Pooling to Im2Col-Reduce combinations
        lower.gemm,
        lower.batchnorm,
        lower.conv,
        lower.pooling,
        # Inline quantizers into the graph and optimize the model by reordering
        # and constant-folding: ~ the first phase of streamlining
        inline.quantizer,
        reorder,
        fold_constants.release,
        # MultiThreshold function compilation and decomposition (matches what is
        # supported by the FINN RTL backend): ~ the second phase of streamlining
        multithreshold,
        multithreshold.decompose,
        # Remove redundant parameters and Expand operations according to
        # broadcasting rules
        unbroadcast,
        # ONNX Passes to QONNX/FINN export and adapter passes to ensure
        # compatibility
        ExportIm2ColToFINN_v1,
        ExportMultiThresholdToFINN_v1,
        ExportLayoutConverterToFINN_v1,
    ]


def _infer_qonnx_datatypes(model: ModelWrapper):
    """Adds QONNX datatypes to a model by inferring types from values."""

    # Try inferring new datatype annotations for all tensors with initializer in
    # the model that not already have a QONNX integer type annotation
    for name in model.get_all_tensor_names():
        if (init := model.get_initializer(name)) is not None:
            if not model.get_tensor_datatype(name).is_integer():
                # If all values are integers, i.e., do not change when rounding
                # and casting to integer, infer this as an integer tensor
                if np.all(np.asarray(np.round(init), dtype=np.int64) == init):
                    # Set to some large integer type, should be minimized later
                    model.set_tensor_datatype(name, DataType["INT64"])

    # Potentially modified model, still as QONNX ModelWrapper, this step
    # operates in-place modifying the original
    return model.transform(InferDataTypes())


def step_passes_frontend(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Meta build step calling the ONNX Passes steps in the expected order."""

    # Convert the QONNX ModelWrapper instance to ONNX IR for pass application
    # and setup ONNX Passes configuration from the DataflowBuildConfig
    model = ir.from_proto(model.model)

    verify = False

    if cfg.verify_steps is not None:
        if VerificationStepType.PASSES_FRONTEND in cfg.verify_steps:
            verify = VerifyConfig(
                tolerance=VerifyConfig.Tolerance(
                    atol=cfg.verification_atol,
                    rtol=cfg.verification_rtol,
                ),
                full_context_dump=cfg.verify_save_full_context,
                inputs=[cfg.verify_input_npy],
                expected=[cfg.verify_expected_output_npy],
            )

    config = PassesConfig(logging=LoggingConfig(verbose=cfg.verbose), verify=verify)

    # Apply the FINNPasses sequence and convert the resulting model back to the
    # proto representation wrapped as QONNX ModelWrapper
    result = FINNPasses_v1(config).call(model)

    if state := result.model.meta["passes"]:
        total, modified_by, verified_by = (
            state.counter,
            len(state.modified_by),
            len(state.verified_by),
        )

        print(
            f"Applied {total} passes (model modified by {modified_by}," f" verified {verified_by})"
        )

    # Fail with a clear error if any operator from the ONNX Passes domain
    # survived the export rules: QONNX cannot process these and FINN would fail
    # on the first transformation touching them
    save_to = None

    if cfg.save_intermediate_models:
        save_to = Path(cfg.output_dir) / "intermediate_models" / "step_passes_frontend_failed.onnx"

    assert_exported(result.model, save_to)

    model = ModelWrapper(ir.to_proto(result.model))

    # Annotate integer-valued constant tensors with QONNX type annotations for
    # further processing by FINN
    return _infer_qonnx_datatypes(model)
