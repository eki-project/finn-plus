"""Empirical (learned) resource and power estimation for dataflow models.

The estimators are regression models fitted on the CI microbenchmark database (see
:mod:`finn.qor`). They are loaded from the directory given by the ``FINN_QOR_MODEL_DIR``
environment variable and applied per node; nodes or resource types without a fitted model
fall back to the analytical FINN estimate (resources) or are reported as 0 (power).
"""

import logging
import os
import pandas as pd
from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import Optional

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.qor.database import POWER_TARGET_COL
from finn.qor.estimator import MODEL_DIR_ENV_VAR, RESOURCE_TARGET_PREFIX, QoREstimator
from finn.qor.features import spec_for_op_type
from finn.util.basic import getHWCustomOp
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

logger = logging.getLogger(__name__)

#: Resource types reported by the analytical estimate, in the order used in the reports.
RESOURCE_TYPES = ("LUT", "DSP", "BRAM_18K", "URAM")


def node_features(model: ModelWrapper, node: NodeProto, inst: HWCustomOp) -> pd.DataFrame:
    """Map a hardware node to the feature row expected by the QoR estimators of its operator.

    The columns must match ``OperatorFeatureSpec.feature_cols`` of the operator exactly, which
    is asserted, so that a model fitted on the microbenchmark database applies to the node.
    Raises ValueError for unsupported node types.
    """
    spec = spec_for_op_type(node.op_type)
    if spec is None:
        raise ValueError(f"No QoR feature spec for op_type {node.op_type}")

    features: dict[str, object] = {}
    if spec.name == "mvau":
        idt = DataType[inst.get_nodeattr("inputDataType")]
        wdt = DataType[inst.get_nodeattr("weightDataType")]
        weights = model.get_initializer(node.input[1])
        features = {
            "params.backend": "hls" if node.op_type.endswith("_hls") else "rtl",
            "params.mem_mode": inst.get_nodeattr("mem_mode"),
            "params.ram_style": inst.get_nodeattr("ram_style"),
            "params.ram_style_thr": inst.get_nodeattr("ram_style_thresholds"),
            "idt_bitwidth": idt.bitwidth(),
            "idt_sign": idt.signed(),
            "wdt_bitwidth": wdt.bitwidth(),
            "wdt_sign": wdt.signed(),
            "params.act": None
            if inst.get_nodeattr("noActivation") == 1
            else inst.get_nodeattr("outputDataType"),
            "params.mw": inst.get_nodeattr("MW"),
            "params.mh": inst.get_nodeattr("MH"),
            "dut_info.simd": inst.get_nodeattr("SIMD"),
            "dut_info.pe": inst.get_nodeattr("PE"),
            "dut_info.zero_weights": round(float((weights == 0).sum()) / weights.size, 2),
        }

    assert sorted(features) == sorted(spec.feature_cols), "feature mapping out of sync with spec"
    return pd.DataFrame([features])[spec.feature_cols]


class QoRModelSet:
    """All fitted QoR estimators found in a model directory, addressable by node op_type."""

    def __init__(self, estimators: Optional[list[QoREstimator]] = None, model_dir: str = ""):
        self.model_dir = model_dir
        self._by_operator: dict[str, dict[str, QoREstimator]] = {}
        for est in estimators or []:
            self._by_operator.setdefault(est.operator, {})[est.target] = est

    @classmethod
    def load(cls, model_dir: str) -> "QoRModelSet":
        """Load every model with a JSON sidecar in ``model_dir``."""
        estimators = [
            QoREstimator.load(model_dir, op, target)
            for op, target in QoREstimator.available_models(model_dir)
        ]
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
    operators) use the analytical ``node_res_estimation`` instead, so the result has the
    same shape as ``finn.analysis.fpgadataflow.res_estimation.res_estimation``. Ensure that
    all nodes have unique names (GiveUniqueNodeNames) prior to calling this analysis pass.
    """
    res_dict: dict[str, dict[str, int | float]] = {}
    for node in model.graph.node:
        if not (is_hls_node(node) or is_rtl_node(node)):
            continue
        inst = getHWCustomOp(node)
        analytical = inst.node_res_estimation(fpgapart)
        estimates = {
            res: models.get(node.op_type, RESOURCE_TARGET_PREFIX + res) for res in RESOURCE_TYPES
        }
        if not any(estimates.values()):
            res_dict[node.name] = analytical
            continue
        features = node_features(model, node, inst)
        res_dict[node.name] = {
            res: est.predict(features) if est is not None else analytical[res]
            for res, est in estimates.items()
        }
    return res_dict


def empirical_power_estimation(
    model: ModelWrapper, models: QoRModelSet
) -> dict[str, dict[str, float]]:
    """Estimate the (dynamic) power in W of every hardware node using the fitted QoR models.

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
        res_dict[node.name] = {
            POWER_TARGET_COL: float(est.predict(node_features(model, node, inst)))
        }
    if unsupported:
        logger.info(
            "No power model for %d of %d nodes (reported as 0 W)", len(unsupported), len(res_dict)
        )
    return res_dict
