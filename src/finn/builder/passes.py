"""Integrates ONNX Passes and ONNX Script passes into the FINN build steps."""

from onnx_passes.passes.base import RewriteRule, Sequential
from onnx_passes.passes.config import Config as PassesConfig
from onnx_passes.passes.config import VerifyConfig, LoggingConfig

from onnx_passes.ops import DOMAIN as PASSES_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_config import VerificationStepType

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

import onnx_ir as ir
import numpy as np


class ExportIm2ColToFINN_v1(RewriteRule):
    """Export ONNX Passes Im2Col operator to QONNX custom operator."""

    @staticmethod
    def pattern(op, x, dilations, kernel_shape, strides):
        return op.Im2Col(
            x,
            dilations=dilations,
            kernel_shape=kernel_shape,
            strides=strides,
            _domain=PASSES_DOMAIN
        )

    @staticmethod
    def check(context, x, dilations, kernel_shape, strides):
        return x.shape is not None and x.shape.is_static()

    @staticmethod
    def rewrite(op, x, dilations, kernel_shape, strides):
        return op.Im2Col(
            x,
            # QONNX needs the input shape as a node attribute
            input_shape="({})".format(",".join(map(str, x.shape.numpy()))),
            dilations=dilations,
            kernel_shape=kernel_shape,
            strides=strides,
            _domain=PASSES_DOMAIN
        )


class ExportMultiThresholdToFINN_v1(RewriteRule):
    """Export ONNX Passes MultiThreshold operator to QONNX custom operator."""

    @staticmethod
    def pattern(op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=PASSES_DOMAIN)

    @staticmethod
    def check(context, x, thresholds, weights):
        if thresholds.shape is None or thresholds.shape.is_dynamic():
            return False

        if np.any(np.asarray(thresholds.shape[:-2]) != 1):
            return False

        if (weights := ir.convenience.get_const_tensor(weights)) is not None:
            return np.all(weights.numpy() == 1)

        return False

    @staticmethod
    def rewrite(op, x, thresholds, weights):
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
            x,
            op.Constant(value=ir.tensor(thresholds)),
            out_dtype=out_dtype,
            _domain=QONNX_DOMAIN
        )


class ExportLayoutConverterToFINN_v1(RewriteRule):
    """Export ONNX Passes LayoutConverter operator to Transpose."""

    @staticmethod
    def pattern(op, x, assumes, perm):
        return op.LayoutConverter(x, assumes=assumes, perm=perm)

    @staticmethod
    def rewrite(op, x, assumes, perm):
        return op.Transpose(x, perm=perm)


from onnx_passes.passes import canonicalize
from onnx_passes.passes import lower
from onnx_passes.passes import inline
from onnx_passes.passes import reorder
from onnx_passes.passes import fold_constants
from onnx_passes.passes import multithreshold
from onnx_passes.passes import unbroadcast


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
        ExportLayoutConverterToFINN_v1
    ]


from qonnx.transformation.infer_datatypes import InferDataTypes


def _infer_qonnx_datatypes(model: ModelWrapper):
    """Adds QONNX datatypes to a model by inferring types from values."""

    # Try inferring new datatype annotations for all tensors with initializer in
    # the model that not already have a QONNX integer type annotation
    for name in model.get_all_tensor_names():
        if (init := model.get_initializer(name)) is not None:
            if not model.get_tensor_datatype(name).is_integer():
                # If all values are integers, i.e., do not change when rounding
                # and casting to integer, infer this as an integer tensor
                if np.all(
                        np.asarray(np.round(init), dtype=np.int64) == init):
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
                expected=[cfg.verify_expected_output_npy]
            )

    config = PassesConfig(
        logging=LoggingConfig(
            verbose=cfg.verbose
        ),
        verify=verify
    )

    # Apply the FINNPasses sequence and convert the resulting model back to the
    # proto representation wrapped as QONNX ModelWrapper
    result = FINNPasses_v1(config).call(model)

    model = ModelWrapper(ir.to_proto(result.model))

    if state := result.model.meta["passes"]:
        total, modified_by, verified_by = (
            state.counter, len(state.modified_by), len(state.verified_by)
        )

        print(
            f"Applied {total} passes (model modified by {modified_by},"
            f" verified {verified_by})"
        )

    # Annotate integer-valued constant tensors with QONNX type annotations for
    # further processing by FINN
    return _infer_qonnx_datatypes(model)
