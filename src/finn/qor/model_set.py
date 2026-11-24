"""Sets of fitted QoR models (one directory of stored estimators), addressable by node op_type.

Lives in ``finn.qor`` (no FINN/QONNX imports) so that the CI evaluation scripts can load
model sets without a FINN installation; the empirical estimation analysis re-exports it.
"""

import logging
import os
from pathlib import Path

from finn.qor.estimator import MODEL_DIR_ENV_VAR, QoREstimator
from finn.qor.features import spec_for_op_type

logger = logging.getLogger(__name__)

#: Basename of the per-node feature report a build writes next to its estimate reports
NODE_FEATURES_REPORT = "qor_node_features"


class QoRModelSet:
    """All fitted QoR estimators found in a model directory, addressable by node op_type."""

    def __init__(self, estimators: list[QoREstimator] | None = None, model_dir: str = "") -> None:
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
            except Exception as e:  # noqa: PERF203 (one broken model must not disable all)
                logger.warning("Skipping QoR model %s/%s: %s", op, target, e)
        logger.info("Loaded %d QoR models from %s", len(estimators), model_dir)
        return cls(estimators, model_dir)

    @classmethod
    def load_from_env(cls) -> "QoRModelSet | None":
        """Load models from ``$FINN_QOR_MODEL_DIR``; None if unset, missing or empty."""
        model_dir = os.environ.get(MODEL_DIR_ENV_VAR)
        if not model_dir:
            return None
        if not Path(model_dir).is_dir():
            logger.warning("%s=%s is not a directory", MODEL_DIR_ENV_VAR, model_dir)
            return None
        models = cls.load(model_dir)
        return models if len(models) > 0 else None

    def __len__(self) -> int:
        """Return the number of loaded estimators (over all operators and targets)."""
        return sum(len(targets) for targets in self._by_operator.values())

    def get(self, op_type: str, target: str) -> QoREstimator | None:
        """Estimator for a node op_type and target column, or None if not available."""
        spec = spec_for_op_type(op_type)
        if spec is None:
            return None
        return self._by_operator.get(spec.name, {}).get(target)


#: Basename of the per-node feature report a build writes next to its estimate reports
NODE_FEATURES_REPORT = "qor_node_features"
