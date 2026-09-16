# Disable formatter as black messes with the nesting below, spreading ([ and ])
# to multiple lines...
# fmt: off

"""Exhaustive composition of ONNX graph streamlining transformations."""

from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.composed import ComposedTransformation

# Some extra QONNX conversion, streamlining transformations
from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd

from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbTransposeIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
    CollapseRepeatedTranspose,
)
from finn.transformation.streamline.remove import RemoveIdentityReshape, RemoveIdentityTranspose
from finn.transformation.streamline.reorder import (
    MoveAddPastConv,
    MoveAddPastJoinAdd,
    MoveAddPastJoinConcat,
    MoveAddPastMul,
    MoveAffinePastJoinConcat,
    MoveChannelwiseLinearPastFork,
    MoveConstMulPastJoinMul,
    MoveLinearPastEltwiseAdd,
    MoveMulPastFork,
    MoveMulPastJoinAdd,
    MoveMulPastJoinConcat,
    MoveMulPastMaxPool,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarLinearPastSplit,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
    MoveScalesPastIm2Col,
    MoveSqueezePastMultiThreshold,
    MoveTransposePastEltwise,
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
    MoveTransposePastJoinConcat,
    MoveTransposePastJoinMul,
    MoveTransposePastSplit,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres


# Define a set of custom streamlining transformations: These are applied once
# during the actual streamlining step and once after converting attention to
# hardware (the associated cleanup afterward might enable some Streamlining
# transformations once again)
def StreamlinePlus() -> ComposedTransformation:  # noqa: N802  (factory, class-like name)
    # Return a set of exhaustively applied transformations
    """Compose the exhaustive FINN+ streamlining transformation.

    Returns a :class:`ComposedTransformation` that applies the standard and
    residual-topology streamlining passes until the graph stops changing.
    """
    return ComposedTransformation([
        # On skip-connections: prefer pushing scalar multiplication forward
        # before MoveAddPastMul
        MoveMulPastFork(),
        # The "standard" set of FINN streamlining transformations or at least
        # inspired by them but applied exhaustively until none of them changes
        # the graph anymore.
        # Note: Covers most parts of non-branching linear topologies
        ComposedTransformation([
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveMulPastMaxPool(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveScalarLinearPastInvariants(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
        ]),
        # Streamlining scales and biases forward through residual topologies
        # Note: This mostly covers forking and joining operations
        ComposedTransformation([
            # Note: This is probably the most common way of joining skip
            # connections, i.e., this corresponds to the original residual
            # addition, i.e., y = f(x) + x
            MoveLinearPastEltwiseAdd(),
            MoveChannelwiseLinearPastFork(),
            MoveScalarLinearPastInvariants(),
            MoveMulPastFork(),
            MoveMulPastJoinAdd(),
            MoveAddPastJoinAdd(),
            # Note: This brings constant Muls (i.e., quantizer scales to be
            # removed) forward through joining Muls (i.e., those ending up
            # as actual hardware operators).
            MoveConstMulPastJoinMul(),
        ]),
        # Streamlining scales and biases forward through shape/layout changing
        # operations, i.e., mostly transposes
        ComposedTransformation([
            # Convolution inputs and padding
            MoveScalesPastIm2Col(),
            # Streamlining for Split and Concat operations
            MoveScalarLinearPastSplit(),
            MoveAffinePastJoinConcat(),
            MoveMulPastJoinConcat(),
            MoveAddPastJoinConcat(),
            # Move transposes around to some place where they could be removed
            # later, i.e., where they collapse into identities
            MoveTransposePastFork(),
            MoveTransposePastSplit(),
            MoveTransposePastJoinConcat(),
            MoveTransposePastEltwise(),
            MoveTransposePastJoinMul(),
            MoveTransposePastJoinAdd(),
            CollapseRepeatedTranspose(),
            # Remove identity shape/layout transformations
            RemoveIdentityTranspose(),
            RemoveIdentityReshape(),
            # Squeeze operators can be moved past the thresholding
            MoveSqueezePastMultiThreshold(),
            # A certain type of 4d-layout transpose can be absorbed (actually
            # moved past) MultiThreshold operations
            AbsorbTransposeIntoMultiThreshold(),
        ]),
        # Only round and clip after all streamlining transformations have
        # been applied exhaustively.
        # Note: Might still enable another round of streamlining.
        RoundAndClipThresholds(),
    ])
