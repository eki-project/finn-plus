"""Empirical (learned) resource and power estimation for dataflow models.

The estimators are regression models fitted on the CI microbenchmark database (see
:mod:`finn.qor`). They are loaded from the directory given by the ``FINN_QOR_MODEL_DIR``
environment variable and applied per node; nodes or resource types without a fitted model
fall back to the analytical FINN estimate (resources) or are reported as 0 (power).

Every supported operator registers a node feature extractor with
:func:`register_node_features`. The extractor must produce exactly the columns of the
operator's :class:`~finn.qor.features.OperatorFeatureSpec`, using the same pure helpers as
the database side (:mod:`finn.qor.features`) for derived values.
"""

import logging
import numpy as np
import os
import pandas as pd
from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import Callable, Optional

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.qor.database import POWER_TARGET_COL
from finn.qor.estimator import MODEL_DIR_ENV_VAR, QoREstimator, resource_target
from finn.qor.features import SPECS, broadcast_kind, datatype_features, spec_for_op_type
from finn.util.basic import getHWCustomOp
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

logger = logging.getLogger(__name__)

#: Resource types reported by the analytical estimate, in the order used in the reports.
RESOURCE_TYPES = ("LUT", "DSP", "BRAM_18K", "URAM")

NodeFeatureExtractor = Callable[[ModelWrapper, NodeProto, HWCustomOp], dict]

#: Operator (spec) name -> function mapping a node to its feature dict
NODE_FEATURE_EXTRACTORS: dict[str, NodeFeatureExtractor] = {}


def register_node_features(spec_name: str):
    """Decorator registering the node feature extractor of an operator spec."""
    assert spec_name in SPECS, f"No feature spec named {spec_name}"

    def decorator(fn: NodeFeatureExtractor) -> NodeFeatureExtractor:
        NODE_FEATURE_EXTRACTORS[spec_name] = fn
        return fn

    return decorator


def backend_of(node: NodeProto) -> str:
    """``"hls"`` or ``"rtl"`` from a specialized node's op_type."""
    return "hls" if node.op_type.endswith("_hls") else "rtl"


def _dt(inst: HWCustomOp, attr: str) -> DataType:
    return DataType[inst.get_nodeattr(attr)]


def _zero_fraction(weights: np.ndarray) -> float:
    return round(float((weights == 0).sum()) / weights.size, 2)


def node_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> pd.DataFrame:
    """Map a hardware node to the feature row expected by the QoR estimators of its operator.

    The columns must match ``OperatorFeatureSpec.feature_cols`` of the operator exactly, which
    is asserted, so that a model fitted on the microbenchmark database applies to the node.
    Raises ValueError for unsupported node types.
    """
    spec = spec_for_op_type(node.op_type)
    if spec is None:
        raise ValueError(f"No QoR feature spec for op_type {node.op_type}")
    extractor = NODE_FEATURE_EXTRACTORS.get(spec.name)
    if extractor is None:
        raise ValueError(f"No node feature extractor registered for operator {spec.name}")
    features = extractor(model, node, inst)
    assert sorted(features) == sorted(
        spec.feature_cols
    ), f"feature mapping of {spec.name} out of sync with spec"
    return pd.DataFrame([features])[spec.feature_cols]


@register_node_features("mvau")
def _mvau_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    weights = model.get_initializer(node.input[1])
    return {
        "params.backend": backend_of(node),
        "params.mem_mode": inst.get_nodeattr("mem_mode"),
        "params.ram_style": inst.get_nodeattr("ram_style"),
        "params.ram_style_thr": inst.get_nodeattr("ram_style_thresholds"),
        **datatype_features("idt", inst.get_nodeattr("inputDataType")),
        **datatype_features("wdt", inst.get_nodeattr("weightDataType")),
        "params.act": None
        if inst.get_nodeattr("noActivation") == 1
        else inst.get_nodeattr("outputDataType"),
        "params.mw": inst.get_nodeattr("MW"),
        "params.mh": inst.get_nodeattr("MH"),
        "dut_info.simd": inst.get_nodeattr("SIMD"),
        "dut_info.pe": inst.get_nodeattr("PE"),
        "dut_info.zero_weights": _zero_fraction(weights),
    }


@register_node_features("thresholding")
def _thresholding_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    hls = backend_of(node) == "hls"
    return {
        "params.backend": backend_of(node),
        **datatype_features("idt", inst.get_nodeattr("inputDataType")),
        **datatype_features("odt", inst.get_nodeattr("outputDataType")),
        "params.ch": inst.get_nodeattr("NumChannels"),
        "params.pe": inst.get_nodeattr("PE"),
        "tdt_bitwidth": _dt(inst, "weightDataType").bitwidth(),
        "params.mem_mode": inst.get_nodeattr("mem_mode") if hls else None,
        "params.ram_style": inst.get_nodeattr("ram_style") if hls else None,
        "params.depth_trigger_bram": 0 if hls else inst.get_nodeattr("depth_trigger_bram"),
        "params.depth_trigger_uram": 0 if hls else inst.get_nodeattr("depth_trigger_uram"),
    }


@register_node_features("swg")
def _swg_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    k_h, k_w = inst.get_nodeattr("ConvKernelDim")
    ifm_h, ifm_w = inst.get_nodeattr("IFMDim")
    stride_h, stride_w = inst.get_nodeattr("Stride")
    dilation_h, dilation_w = inst.get_nodeattr("Dilation")
    return {
        "idt_bitwidth": _dt(inst, "inputDataType").bitwidth(),
        "params.ifm_ch": inst.get_nodeattr("IFMChannels"),
        "ifm_h": ifm_h,
        "ifm_w": ifm_w,
        "k_h": k_h,
        "k_w": k_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "params.simd": inst.get_nodeattr("SIMD"),
        "params.depthwise": inst.get_nodeattr("depthwise"),
        "params.parallel_window": inst.get_nodeattr("parallel_window"),
        "params.m": inst.get_nodeattr("M"),
        "params.ram_style": inst.get_nodeattr("ram_style"),
        "dut_info.impl_style": inst.select_impl_style(),
        "dut_info.buffer_depth": int(inst.get_buffer_depth()),
        "dut_info.is1D": inst.get_nodeattr("is1D"),
    }


@register_node_features("vvau")
def _vvau_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    k_h, k_w = inst.get_nodeattr("Kernel")
    weights = model.get_initializer(node.input[1])
    hls = backend_of(node) == "hls"
    return {
        "params.backend": backend_of(node),
        "params.mem_mode": inst.get_nodeattr("mem_mode"),
        "params.ram_style": inst.get_nodeattr("ram_style"),
        "params.resType": inst.get_nodeattr("resType") if hls else None,
        **datatype_features("idt", inst.get_nodeattr("inputDataType")),
        **datatype_features("wdt", inst.get_nodeattr("weightDataType")),
        "params.act": None
        if inst.get_nodeattr("noActivation") == 1
        else inst.get_nodeattr("outputDataType"),
        "params.ch": inst.get_nodeattr("Channels"),
        "k_h": k_h,
        "k_w": k_w,
        "params.pe": inst.get_nodeattr("PE"),
        "params.simd": inst.get_nodeattr("SIMD"),
        "dut_info.zero_weights": _zero_fraction(weights),
    }


@register_node_features("fifo")
def _fifo_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    impl_style = inst.get_nodeattr("impl_style")
    width = inst.get_instream_width()
    depth_adjusted = int(inst.get_adjusted_depth())
    return {
        "params.impl_style": impl_style,
        "params.ram_style": inst.get_nodeattr("ram_style") if impl_style == "vivado" else None,
        "dut_info.width_bits": width,
        "params.depth": inst.get_nodeattr("depth"),
        "dut_info.depth_adjusted": depth_adjusted,
        "dut_info.capacity_bits": width * depth_adjusted,
    }


@register_node_features("dwc")
def _dwc_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    in_width = inst.get_nodeattr("inWidth")
    out_width = inst.get_nodeattr("outWidth")
    big, small = max(in_width, out_width), min(in_width, out_width)
    return {
        "params.backend": backend_of(node),
        "dtype_bitwidth": _dt(inst, "dataType").bitwidth(),
        "params.ch": int(inst.get_nodeattr("inShape")[-1]),
        "dut_info.in_width": in_width,
        "dut_info.out_width": out_width,
        "dut_info.ratio": big / small,
        "dut_info.integer_ratio": bool(big % small == 0),
    }


@register_node_features("pool")
def _pool_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    k_h, k_w = inst.get_nodeattr("KernelSize")
    return {
        "params.function": inst.get_nodeattr("Function"),
        **datatype_features("idt", inst.get_nodeattr("InputDataType")),
        "odt_bitwidth": _dt(inst, "OutputDataType").bitwidth(),
        "params.ch": inst.get_nodeattr("Channels"),
        "params.pe": inst.get_nodeattr("PE"),
        "k_h": k_h,
        "k_w": k_w,
        "dut_info.accum_bits": inst.get_nodeattr("AccumBits"),
    }


@register_node_features("fmpadding")
def _fmpadding_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    idim_h, idim_w = inst.get_nodeattr("ImgDim")
    pad_t, pad_l, pad_b, pad_r = inst.get_nodeattr("Padding")
    return {
        "idt_bitwidth": _dt(inst, "inputDataType").bitwidth(),
        "params.ch": inst.get_nodeattr("NumChannels"),
        "params.simd": inst.get_nodeattr("SIMD"),
        "idim_h": idim_h,
        "idim_w": idim_w,
        "pad_t": pad_t,
        "pad_l": pad_l,
        "pad_b": pad_b,
        "pad_r": pad_r,
    }


@register_node_features("eltwise")
def _eltwise_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> dict:
    op = node.op_type.replace("Elementwise", "").rsplit("_", 1)[0]
    lhs_style = inst.get_nodeattr("lhs_style")
    rhs_style = inst.get_nodeattr("rhs_style")
    # the microbenchmarks stream the lhs and store the rhs; Add/Mul are commutative, so a
    # constant lhs takes the role of the rhs
    if lhs_style == "const" and rhs_style == "input":
        stream_dtype, const_dtype = inst.get_nodeattr("rhs_dtype"), inst.get_nodeattr("lhs_dtype")
        const_shape = inst.get_nodeattr("lhs_shape")
    else:
        stream_dtype, const_dtype = inst.get_nodeattr("lhs_dtype"), inst.get_nodeattr("rhs_dtype")
        const_shape = inst.get_nodeattr("rhs_shape")
    out_shape = inst.get_nodeattr("out_shape")
    both_streams = lhs_style == "input" and rhs_style == "input"
    return {
        "params.backend": backend_of(node),
        "params.op": op,
        **datatype_features("lhs", stream_dtype, with_kind=True),
        **datatype_features("rhs", const_dtype, with_kind=True),
        "out_bitwidth": _dt(inst, "out_dtype").bitwidth(),
        "params.pe": inst.get_nodeattr("PE"),
        "params.rhs_bcast": "input" if both_streams else broadcast_kind(const_shape, out_shape),
        "dut_info.rhs_num_elems": int(np.prod(const_shape)),
        "c": int(out_shape[-1]),
        "params.mem_mode": inst.get_nodeattr("mem_mode"),
        "params.ram_style": inst.get_nodeattr("ram_style"),
    }


class QoRModelSet:
    """All fitted QoR estimators found in a model directory, addressable by node op_type."""

    def __init__(self, estimators: Optional[list[QoREstimator]] = None, model_dir: str = ""):
        """Index the given estimators by operator name and target column."""
        self.model_dir = model_dir
        self._by_operator: dict[str, dict[str, QoREstimator]] = {}
        for est in estimators or []:
            self._by_operator.setdefault(est.operator, {})[est.target] = est

    @classmethod
    def load(cls, model_dir: str) -> "QoRModelSet":
        """Load every model with a JSON sidecar in ``model_dir`` (models whose runtime
        dependencies are missing are skipped with a warning)."""
        estimators = []
        for op, target in QoREstimator.available_models(model_dir):
            try:
                estimators.append(QoREstimator.load(model_dir, op, target))
            except Exception as e:  # noqa: BLE001 - one broken model must not disable all
                logger.warning("Skipping QoR model %s/%s: %s", op, target, e)
        logger.info("Loaded %d QoR models from %s", len(estimators), model_dir)
        return cls(estimators, model_dir)

    @classmethod
    def load_from_env(cls) -> Optional["QoRModelSet"]:
        """Load models from ``$FINN_QOR_MODEL_DIR``; None if unset, missing or empty."""
        model_dir = os.environ.get(MODEL_DIR_ENV_VAR)
        if not model_dir:
            return None
        if not os.path.isdir(model_dir):
            logger.warning("%s=%s is not a directory", MODEL_DIR_ENV_VAR, model_dir)
            return None
        models = cls.load(model_dir)
        return models if len(models) > 0 else None

    def __len__(self) -> int:
        """Number of loaded estimators (over all operators and targets)."""
        return sum(len(targets) for targets in self._by_operator.values())

    def get(self, op_type: str, target: str) -> Optional[QoREstimator]:
        """Estimator for a node op_type and target column, or None if not available."""
        spec = spec_for_op_type(op_type)
        if spec is None:
            return None
        return self._by_operator.get(spec.name, {}).get(target)


def empirical_res_estimation(
    model: ModelWrapper, fpgapart: str, models: QoRModelSet
) -> dict[str, dict[str, int | float]]:
    """Estimate the resources of every hardware node using the fitted QoR models.

    Resource types without a model for the node's operator (and nodes of unsupported
    operators, or whose categorical features were not covered by the training data) use the
    analytical ``node_res_estimation`` instead, so the result has the same shape as
    ``finn.analysis.fpgadataflow.res_estimation.res_estimation``. Ensure that all nodes have
    unique names (GiveUniqueNodeNames) prior to calling this analysis pass.
    """
    res_dict: dict[str, dict[str, int | float]] = {}
    for node in model.graph.node:
        if not (is_hls_node(node) or is_rtl_node(node)):
            continue
        inst = getHWCustomOp(node)
        analytical = inst.node_res_estimation(fpgapart)
        estimates = {res: models.get(node.op_type, resource_target(res)) for res in RESOURCE_TYPES}
        if not any(estimates.values()):
            res_dict[node.name] = analytical
            continue
        features = node_features(model, node, inst)
        result = dict(analytical)
        for res, est in estimates.items():
            if est is None:
                continue
            if not est.covers(features):
                logger.info(
                    "%s: %s features outside training categories, using analytical estimate",
                    node.name,
                    res,
                )
                continue
            result[res] = est.predict(features)
        res_dict[node.name] = result
    return res_dict


def empirical_power_estimation(
    model: ModelWrapper, models: QoRModelSet
) -> dict[str, dict[str, float]]:
    """Estimate the (dynamic) power of every hardware node using the fitted QoR models.

    The unit is that of the measurements in the microbenchmark database (mW for the CI boards).
    Nodes without a power model are reported as 0.0 (there is no analytical fallback); the
    number of such nodes is logged so that the coverage of the total is visible.
    """
    res_dict: dict[str, dict[str, float]] = {}
    unsupported = []
    for node in model.graph.node:
        if not (is_hls_node(node) or is_rtl_node(node)):
            continue
        est = models.get(node.op_type, POWER_TARGET_COL)
        if est is None:
            unsupported.append(node.name)
            res_dict[node.name] = {POWER_TARGET_COL: 0.0}
            continue
        inst = getHWCustomOp(node)
        features = node_features(model, node, inst)
        if not est.covers(features):
            unsupported.append(node.name)
            res_dict[node.name] = {POWER_TARGET_COL: 0.0}
            continue
        res_dict[node.name] = {POWER_TARGET_COL: float(est.predict(features))}
    if unsupported:
        logger.info(
            "No power model for %d of %d nodes (reported as 0)", len(unsupported), len(res_dict)
        )
    return res_dict
