"""Utils for Multi-FPGA testing."""
import pytest

import brevitas.nn as qnn
import configparser
import hashlib
import logging
import onnx.helper as oh
import random
import torch
from brevitas.export import export_qonnx
from brevitas_examples.bnn_pynq.models.resnet import quant_resnet18
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from networkx.classes.digraph import DiGraph
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.general import GiveUniqueNodeNames, RemoveUnusedTensors
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from testing_util.test import get_test_model
from typing import Any, cast

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow import resolve_build_steps
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.custom_step_library.resnet import (
    step_resnet_convert_to_hw,
    step_resnet_streamline,
    step_resnet_tidy,
)
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import make_build_dir
from tests.end2end.test_end2end_bnn_pynq import get_folding_function

# ---- MOCK MODEL CLASSES ----


@dataclass
class MockNode:  # noqa
    name: str


@dataclass
class MockGraph:  # noqa
    node: list[MockNode]


@dataclass
class MockModelWrapper:  # noqa
    graph: MockGraph


def mock_model(nodes: int) -> MockModelWrapper:
    """Return a mock model with `nodes` nodes, with their indices as names."""
    return MockModelWrapper(MockGraph([MockNode(str(i)) for i in range(nodes)]))


# ---- RESNET 18 ----


class RN18(torch.nn.Module):
    """A simple ResNet-18 with an input quantizer."""

    def __init__(self, cfg: configparser.ConfigParser) -> None:
        """Create a simple ResNet-18 from scratch with the given config."""
        super().__init__()
        self.inpQuantizer = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.resnet = quant_resnet18(cfg)

    def forward(self, x):  # noqa
        x = self.inpQuantizer(x)
        return self.resnet(x)


def _create_rn18_model(w: int, a: int, classes: int = 100) -> RN18:
    """Create a ResNet-18 (Brevitas model) with the given weight and activation bitwidths."""
    cfg = configparser.ConfigParser()
    cfg["MODEL"] = {"NUM_CLASSES": str(classes)}
    cfg["QUANT"] = {"WEIGHT_BIT_WIDTH": str(w), "ACT_BIT_WIDTH": str(a)}
    return RN18(cfg)


def _create_rn18_onnx(path: Path, w: int, a: int, classes: int = 100) -> None:
    """Create a ResNet-18 and export as a QONNX model to the given path."""
    model = _create_rn18_model(w, a, classes)
    model.eval()
    inp = torch.zeros((1, 3, 32, 32))
    _ = model(inp)
    export_qonnx(model, (inp,), str(path.absolute()))


def generate_rn18(
    model_onnx_path: Path, w: int, a: int, classes: int = 100
) -> tuple[ModelWrapper, Path]:
    """Generate a new ResNet-18 in a fresh build directory, usable for testing."""
    _create_rn18_onnx(model_onnx_path, w, a, classes)
    return ModelWrapper(str(model_onnx_path)), model_onnx_path


# Steps for a resnet to execute before reaching the point at which Multi-FPGA can be started
rn18_pre_multifpga_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    step_resnet_tidy,
    step_resnet_streamline,
    "step_convert_to_hw",
    step_resnet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]
"""Steps required to prepare the ResNet18 model for Multi-FPGA usage."""

# ---- MOBILENET ----

mn_pre_multifpga_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_streamline",  # Custom step
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_lower_convs",  # Custom step
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_convert_to_hw_layers",
    # "finn.builder.custom_step_library.mobilenet.step_mobilenet_convert_to_hw_layers_separate_th",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    # "step_transpose_decomposition",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]


def generate_mobilenet(
    model_onnx_path: Path, wbits: int, abits: int, pretrained: bool
) -> ModelWrapper:
    """Provide a mobilenet modelwrapper."""
    fc = get_test_model("mobilenet", wbits, abits, pretrained)
    export_qonnx(fc, torch.randn((1, 3, 224, 224)), str(model_onnx_path))
    model = ModelWrapper(str(model_onnx_path))
    model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
    model.save(str(model_onnx_path))
    qonnx_cleanup(str(model_onnx_path), out_file=str(model_onnx_path))
    return ModelWrapper(str(model_onnx_path))


# ---- BASIC MODELS ----


def bnn_make_step_streamline_bnn_pynq(
    model_type: str,
) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]:
    """Produce a streamline step function, depending on the incoming model type."""

    def streamline(model: ModelWrapper, _: DataflowBuildConfig) -> ModelWrapper:
        model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
        # move past any reshapes to be able to streamline input scaling
        model = model.transform(MoveScalarLinearPastInvariants())
        model = model.transform(Streamline())
        if "fc" not in model_type.lower():
            model = model.transform(LowerConvsToMatMul())
            model = model.transform(MakeMaxPoolNHWC())
            model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(ConvertBipolarMatMulToXnorPopcount())
        model = model.transform(Streamline())
        # absorb final add-mul nodes into TopK
        model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
        model = model.transform(InferDataLayouts())
        model = model.transform(RemoveUnusedTensors())
        return model

    return streamline


def bnn_make_step_convert_to_hw(
    topology: str, wbits: int, abits: int
) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]:
    """Make a custom convert_to_hw step for BNN models, based on model type."""
    topology = topology.lower()

    def convert(model: ModelWrapper, _: DataflowBuildConfig) -> ModelWrapper:
        if topology == "tfc" and wbits == 1 and abits == 1:
            # use standalone thresholds for tfc-w1a1 to also exercise that option
            model = model.transform(to_hw.InferThresholdingLayer())
        # needed for bipolar MatMul layers
        model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
        # needed for non-bipolar MatMul layers
        model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
        # TopK to LabelSelect
        model = model.transform(to_hw.InferLabelSelectLayer())
        # input quantization (if any) to standalone thresholding
        model = model.transform(to_hw.InferThresholdingLayer())
        # needed for convolutions
        if "fc" not in topology:
            model = model.transform(to_hw.InferPool())
            model = model.transform(to_hw.InferConvInpGen())
            model = model.transform(RemoveCNVtoFCFlatten())
        # get rid of Tranpose -> Tranpose identity seq
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataLayouts())
        return model

    return convert


def bnn_make_step_fold(
    topology: str, wbits: int, abits: int
) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper] | str:
    """Make the folding function for the given BNN model."""
    if topology.lower() == "sfc":
        return "step_target_fps_parallelization"
    folding_fxn = get_folding_function(topology.lower(), wbits, abits)

    def fold(model: ModelWrapper, _: DataflowBuildConfig) -> ModelWrapper:
        return folding_fxn(model)

    return fold


def bnn_step_minimize_bitwidth(model: ModelWrapper, _: DataflowBuildConfig) -> ModelWrapper:  # noqa
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(MinimizeWeightBitWidth())
    return model


def bnn_steps(topology: str, wbits: int, abits: int) -> list[str | Callable]:
    """Return the step list for the givenn bnn pynq model topology."""
    return [
        "finn.builder.custom_step_library.general.add_preproc_divide_by_255",  # Custom step
        "finn.builder.custom_step_library.general.add_postproc_top1",  # Custom step
        "step_qonnx_to_finn",
        "step_tidy_up",
        bnn_make_step_streamline_bnn_pynq(model_type=topology),
        bnn_make_step_convert_to_hw(topology, wbits, abits),
        "step_create_dataflow_partition",
        "step_specialize_layers",
        bnn_make_step_fold(topology, wbits, abits),
        "step_apply_folding_config",
        bnn_step_minimize_bitwidth,
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
    ]


def generate_basic_model(
    model_onnx_path: Path, typename: str, wbits: int, abits: int, pretrained: bool
) -> ModelWrapper:
    """Provide a basic modelwrapper. This can be one of the TFC, CNV, etc. models."""
    fc = get_test_model(typename, wbits, abits, pretrained)
    ishape = (1, 1, 28, 28)
    if typename == "CNV":
        ishape = (1, 3, 32, 32)
    export_qonnx(fc, torch.randn(ishape), str(model_onnx_path))
    qonnx_cleanup(str(model_onnx_path), out_file=str(model_onnx_path))
    return ModelWrapper(str(model_onnx_path))


# ---- GRAPH UTILITY ----


def list_contains_all_elements(this: list, other: list) -> bool:
    """Return whether a list contain all elements of another list."""
    return all(n in this for n in other)


def networkx_to_onnx(g: DiGraph) -> ModelWrapper:
    """Convert a networkx graph into a QONNX ModelWrapper.
    All nodes will be StreamingDataflowPartitions.
    """
    nodes = [oh.make_node("StreamingDataflowPartition", [], [], n) for n in g.nodes]
    get_node_by_name = lambda name: [n for n in nodes if n.name == name][0]  # noqa
    for i, edge in enumerate(g.edges):
        source_node = get_node_by_name(edge[0])
        target_node = get_node_by_name(edge[1])
        source_node.output.append(f"edge_{i}")
        target_node.input.append(f"edge_{i}")
    # TODO: Make graph inputs and outputs
    graph = oh.make_graph(nodes, "graph", [], [])
    model = oh.make_model(graph)
    return ModelWrapper(model)


# ---- CACHE ----


def get_model(  # noqa
    model: str,
    wbits: int,
    abits: int,
    pretrained: bool,
    until_step: str | None,
    skip_fifo_sizing: bool,
    cfg: DataflowBuildConfig,
    pytestconfig: pytest.Config | None,
    identifier: str | None = None,
    write_cache: bool = False,
) -> tuple[ModelWrapper, DataflowBuildConfig]:
    """Get a prepared model. The test identifier does not consider
    partitioning configuration parameters. This means that configs with
    for example 2 different device counts are considered the same model.

    By default, this does NOT fill the cache.

    Arguments:
    ---------
        `model`: The model to generate. Will be created.
        `wbits`: Weight-bitwidth of the model.
        `abits`: Act-bitwidth of the model.
        `pretrained`: Whether the model should be pretrained.
        `until_step`: The last step until which the FINN flow is executed (inclusive).
            If it is None instead, the model will be returned without any modification.
        `cfg`: The config to use for the transformations.
        `pytestconfig`: If given, this is used to build an identifying string and retrieve
            cached models matching the requested specification.
        `identifier`: (Optional) unique identifier for cache requests, which is appended to the
            internally generated identifier.
        `skip_fifo_sizing`: Whether to skip step_set_fifo_depths.
        `write_cache`: Can be set to False to skip writing the cache.
            This can be done to avoid issues
            during toolchain calls.

    Returns:
    -------
        `ModelWrapper, DataflowBuildConfig`: The prepared model and the modified dataflow config.

    Raises:
    ------
        `FINNError`: If the last step is unknown, no valid model(type) is passed, etc.
    """
    log = logging.getLogger("Model Request")

    def get_test_identifier(
        steps: list[str | Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]]
    ) -> str:
        return (
            f"{identifier}_{model}_{wbits}_{abits}_{pretrained}_"
            f"{skip_fifo_sizing}_{cfg.target_fps}_{cfg.mvau_wwidth_max}_"
            f"{cfg.folding_two_pass_relaxation}_"
            f"{cfg.shell_flow_type.name if cfg.shell_flow_type is not None else 'SHELL'}_"
            f"{cfg.board}_{cfg._resolve_fpga_part()}_{cfg.hls_clk_period_ns}_"  # noqa
            f"{'_'.join([step if type(step) is str else step.__name__ for step in steps])}"
        )

    def get_cache_key(identifier: str) -> str:
        """Get the sha256 hash of the identifier, otherwise the cache key is too long (Errno 36)."""
        h = hashlib.new("sha256")
        h.update(identifier.encode("utf-8"))
        return h.hexdigest()

    filename = f"{model}_w{wbits}_a{abits}_pretrained-{pretrained}.onnx"

    # Create the model and fetch the steps
    modelwrapper = None
    steps = []
    match model:
        case "resnet18":
            modelwrapper = generate_rn18(Path(cfg.output_dir) / filename, wbits, abits)[0]
            steps = deepcopy(rn18_pre_multifpga_steps)
        case "mobilenetv1":
            modelwrapper = generate_mobilenet(
                Path(cfg.output_dir) / filename, wbits, abits, pretrained
            )
            steps = deepcopy(mn_pre_multifpga_steps)
        case "LFC" | "TFC" | "SFC" | "CNV":
            modelwrapper = generate_basic_model(
                Path(cfg.output_dir) / filename, model, wbits, abits, pretrained
            )
            steps = bnn_steps(model.lower(), wbits, abits)
        case _:
            raise NotImplementedError(f"Unknown test model: {model}.")

    # Check that we have the steps
    if len(steps) == 0:
        raise NotImplementedError(f"Could not find steps to prepare model of type {model}..")

    # If the last step is not given, we return the unmodified model
    if until_step is None:
        return modelwrapper, cfg

    # Check that the step can be found
    if until_step not in steps:
        raise NotImplementedError(
            f"Cannot prepare until given step {until_step} which "
            f"was not found in the default preparation steps: {steps}"
        )
    # Resolve steps
    cfg.steps = steps
    if skip_fifo_sizing:
        # If the set fifo step is the selected last one and we skip, set the last step to be
        # the one before
        if until_step == "step_set_fifo_depths":
            until_step = str(cfg.steps[cfg.steps.index(until_step) - 1])
        cfg.steps.remove("step_set_fifo_depths")

    # Had to manually cast here because of the type checker
    cached_steps = cast(
        "list[str | Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]]",
        cfg.steps[: cfg.steps.index(until_step) + 1],
    )
    leftover_steps = []

    # Test the cache
    # Remove steps one by one to find the latest model
    if pytestconfig is not None:
        while True:
            log.debug(
                f"TESTING MODEL CACHE: Trying cached {len(cached_steps)} "
                f"/ leftover {len(leftover_steps)} "
                f"({get_cache_key(get_test_identifier(cached_steps))[:5]}...)"
            )
            value = pytestconfig.cache.get(
                get_cache_key(get_test_identifier(cached_steps)), default=None
            )
            if value is not None:
                modelpath = Path(value)
                if modelpath.exists():
                    log.debug(
                        "FOUND CACHED MODEL. Most recent "
                        + f"step in this model was: {cached_steps[-1]} ("
                        + get_cache_key(get_test_identifier(cached_steps))[:5]
                        + "...)"
                    )
                    modelwrapper = ModelWrapper(str(modelpath.absolute()))
                    break

            # If we have checked all steps, break
            if len(cached_steps) == 0:
                log.debug("NO CACHED MODEL FOUND")
                break

            # Add the latest step to the list yet to do
            leftover_steps.insert(0, cached_steps.pop())
    else:
        leftover_steps = cached_steps
        cached_steps = []

    # Run all steps
    cache_dir = Path(make_build_dir("model_cache_"))
    all_steps = deepcopy(cfg.steps)
    cfg.steps = leftover_steps
    steps = resolve_build_steps(cfg)
    done = []

    # Build cache.
    # If A and B were already done, and C, D and E are left over,
    # we start with cached_steps = [A,B], leftover_steps = [C,D,E] and done = []
    # We have loaded the model with A and B, and now execute C
    # Afterwards we store the cache key "identifier_A_B_C"
    # and have cached_steps = [A,B], leftover_steps = [D,E] and done = [C]
    # ...
    # cached_steps = [A,B], leftover_steps = [], done = [C,D,E], last model saved
    # is then "identifier_A_B_C_D_E"
    for i, step in enumerate(steps):
        # Execute the new step
        log.debug("RUNNING: " + str(step.__name__))
        modelwrapper = step(modelwrapper, cfg)

        # From this generate the newest identifier
        done.append(leftover_steps[i])
        test_identifier = get_test_identifier(cached_steps + done)

        # Store the model with its identifier in the cache, using a pseudo-random filename
        if write_cache:
            fn = cache_dir / (
                "".join([random.choice("ABCDEF0123456789") for _ in range(30)]) + ".onnx"
            )
            modelwrapper.save(str(fn))
            if pytestconfig is not None:
                log.debug(
                    f"STORING model after step: {done[-1]} ("
                    + get_cache_key(test_identifier)[:5]
                    + "...)"
                )
                pytestconfig.cache.set(get_cache_key(test_identifier), str(fn))

    # Restore steps
    cfg.steps = all_steps
    return modelwrapper, cfg


class TestingNode(CustomOp):
    """A CustomOp purely for testing graph related functionality."""

    def get_nodeattr_types(self) -> dict[str, tuple[str, bool, Any]]:
        """Node attribute definitions."""
        return {
            # To store the original index, if given, from an nx DiGraph
            "original_index": ("i", False, 0),
            "partition_id": ("i", False, 0),
            "device_id": ("i", False, 0),
            # SDP attributes
            "res_estimate": ("s", False, ""),
            "res_hls": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "slr": ("i", False, -1),
            "mem_port": ("s", False, ""),
            "instance_name": ("s", False, ""),
            "return_full_exec_context": ("i", False, 0),
            "network_connections": ("strings", False, []),
        }

    def get_folded_input_shape(self, i: int) -> list:
        return [32]

    def get_folded_output_shape(self, i: int) -> list:
        return [32]

    def get_normal_input_shape(self, i: int) -> list:
        return [32]

    def get_normal_output_shape(self, i: int) -> list:
        return [32]

    def make_shape_compatible_op(self, model):  # noqa
        pass

    def infer_node_datatype(self, model):  # noqa
        pass

    def execute_node(self, context, graph):  # noqa
        pass

    def verify_node(self):  # noqa
        pass


# Register test node when importing this testing module
from finn.custom_op.fpgadataflow import custom_op  # noqa

custom_op["TestingNode"] = TestingNode
