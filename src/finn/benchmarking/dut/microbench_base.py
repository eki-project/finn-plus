"""Common base class for single-operator microbenchmark DUTs.

A :class:`MicrobenchDUT` generates a one-node FINN dataflow graph from a parameter set,
builds it with the full flow (up to bitfile and deployment package) and records auxiliary
facts about the generated node in ``report/dut_info.json``. Subclasses provide:

* :attr:`NAME` (== ``params["dut"]`` == database subfolder == ``OperatorFeatureSpec.name``),
* :attr:`OP_TYPES` (specialized op types the DUT may produce),
* :meth:`validate` – a pure function rejecting invalid parameter sets *without* building
  anything (also used by the random sampler),
* :meth:`param_space` – the declarative parameter space for random sampling,
* :meth:`make_model` – model construction, returning the model and the dut_info dict.

Parameters that do not apply to the selected backend/variant must be ``None`` (strings) or
``0`` (numbers), which :meth:`validate` enforces; otherwise the parameter-set based
deduplication of the result database would keep spurious duplicates.
"""

import json
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import Any, Optional

import finn.builder.build_dataflow_config as build_cfg
from finn.benchmarking.bench_base import bench
from finn.benchmarking.param_space import ParamSpace
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import MAX_ALLOWED_AP_INT_W, part_map
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

#: Widest stream the instrumentation shell (and HLS ap_uint) can handle
MAX_STREAM_WIDTH = min(1024, MAX_ALLOWED_AP_INT_W)

#: Build steps of a microbenchmark: everything up to bitfile + deployment package
MICROBENCH_BUILD_STEPS = [
    "step_create_dataflow_partition",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_vivado_power_estimation",
    "step_synthesize_bitfile",
    "step_make_driver",
    "step_deployment_package",
]


def resolve_part(params: dict) -> str:
    """FPGA part for a parameter set (``part`` or ``board`` key, default RFSoC2x2 like
    :class:`bench`)."""
    if params.get("part"):
        return str(params["part"])
    board = params.get("board", "RFSoC2x2")
    if board not in part_map:
        raise ValueError(f"No part known for board {board}")
    return part_map[board]


def backend_of(node) -> str:
    """``"hls"`` or ``"rtl"`` from a specialized node's op_type."""
    return "hls" if node.op_type.endswith("_hls") else "rtl"


def stream_width_ok(bits: int) -> bool:
    """Whether a stream of ``bits`` fits the instrumentation shell."""
    return 0 < bits <= MAX_STREAM_WIDTH


def check_foreign(params: dict, keys: list[str], reason: str) -> Optional[str]:
    """Reject parameter sets where inapplicable keys are set (not None/0)."""
    for key in keys:
        value = params.get(key)
        if value is not None and value != 0:
            return f"{key} must be unset ({reason})"
    return None


def specialize_single_node(model: ModelWrapper, backend: str, fpga_part: str) -> ModelWrapper:
    """Set ``preferred_impl_style`` on all HW nodes and run SpecializeLayers."""
    for node in model.graph.node:
        if node.domain == "finn.custom_op.fpgadataflow":
            getCustomOp(node).set_nodeattr("preferred_impl_style", backend)
    return model.transform(SpecializeLayers(fpga_part))


class MicrobenchDUT(bench):
    """Base class for single-operator microbenchmarks (see module docstring)."""

    NAME: str = ""
    OP_TYPES: tuple[str, ...] = ()
    #: parameter name -> short description (documentation for configs and the sampler)
    PARAMS: dict[str, str] = {}
    SEED: int = 123456

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        """Return a rejection reason for invalid parameter sets, None if valid."""
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        """Declarative parameter space for random sampling (see param_space.py)."""
        raise NotImplementedError

    @classmethod
    def make_model(cls, params: dict, fpga_part: str) -> tuple[ModelWrapper, dict[str, Any]]:
        """Build the single-node model and return it with the dut_info dict."""
        raise NotImplementedError

    @classmethod
    def build_steps(cls, params: dict) -> list[str]:
        """Build steps for this parameter set (default: the full microbenchmark flow)."""
        return list(MICROBENCH_BUILD_STEPS)

    @classmethod
    def sample_defaults(cls) -> dict[str, Any]:
        """Fixed parameters every sampled run of this DUT needs (overridable in configs)."""
        return {
            "dut": cls.NAME,
            "instrumentation_no_dma": True,
            "store_results_in_dvc_data": True,
            "generate_outputs": [
                "estimate_reports",
                "stitched_ip",
                "rtlsim_performance",
                "out_of_context_synth",
                "bitfile",
                "pynq_driver",
                "deployment_package",
            ],
        }

    def _step_export_onnx(self, onnx_export_path):
        reason = self.validate(self._params)
        if reason is not None:
            print(f"Invalid configuration, skipping: {reason}")
            return "skipped"

        np.random.seed(self.SEED)
        model, info = self.make_model(self._params, self._part)
        model = model.transform(GiveUniqueNodeNames())

        hw_nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]
        if len(hw_nodes) != 1:
            print(f"Expected exactly one hardware node, got {len(hw_nodes)}, skipping")
            return "skipped"
        node = hw_nodes[0]
        if node.op_type not in self.OP_TYPES:
            # SpecializeLayers falls back silently (e.g. RTL -> HLS) if the requested
            # backend is not possible for this configuration
            print(f"Node specialized to {node.op_type} instead of {self.OP_TYPES}, skipping")
            return "skipped"

        info["dut_node_name"] = node.name
        info["dut_op_type"] = node.op_type
        info["dut_backend"] = backend_of(node)
        with open(self._build_inputs["build_dir"] / "report/dut_info.json", "w") as f:
            json.dump(info, f, indent=2, default=_jsonable)

        model.save(onnx_export_path)
        return None

    def _step_build_setup(self):
        return build_cfg.DataflowBuildConfig(target_fps=None, steps=self.build_steps(self._params))


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
