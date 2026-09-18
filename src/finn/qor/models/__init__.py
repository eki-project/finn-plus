"""Additional regression models for the QoR estimators (beyond the scikit-learn tree zoo).

All models are scikit-learn compatible estimators, so they drop into the existing
preprocessing pipeline, GridSearchCV selection and pickle persistence unchanged:

* :class:`LogTargetHGBRegressor`, :class:`LogTargetMLPRegressor` – always available.
* :class:`TabTransformerRegressor` – FT-Transformer style tabular transformer (needs torch).
* :class:`SymbolicRegressor` – PySR symbolic regression (needs pysr + Julia to fit, only
  sympy to predict).

:func:`optional_regressor_grid` lists the grid entries whose runtime dependencies are
importable (checked via ``importlib.util.find_spec`` so that merely listing the grid never
triggers PySR's Julia installation).
"""

import importlib.util
import logging
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from typing import Any, Optional

from finn.qor.models.sklearn_extras import (
    RANDOM_STATE,
    LogTargetHGBRegressor,
    LogTargetMLPRegressor,
    LogTargetRegressor,
    log1p_float,
    signed_expm1,
    signed_log1p,
)
from finn.qor.models.symbolic import SymbolicRegressor
from finn.qor.models.transformer import TabTransformerRegressor

__all__ = [
    "LogTargetHGBRegressor",
    "LogTargetMLPRegressor",
    "LogTargetRegressor",
    "SymbolicRegressor",
    "TabTransformerRegressor",
    "OPTIONAL_DEPENDENCIES",
    "dependency_status",
    "optional_regressor_grid",
    "log1p_float",
    "signed_log1p",
    "signed_expm1",
]

logger = logging.getLogger(__name__)

#: Regressor name -> python modules it needs at fit time
OPTIONAL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "TabTransformerRegressor": ("torch",),
    "SymbolicRegressor": ("pysr", "sympy"),
}


def dependency_status() -> dict[str, Optional[str]]:
    """Per optional regressor: None if all dependencies are importable, else the reason."""
    status: dict[str, Optional[str]] = {}
    for name, modules in OPTIONAL_DEPENDENCIES.items():
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        status[name] = None if not missing else "missing: " + ", ".join(missing)
    return status


def _log_input_scaler() -> Pipeline:
    """Numeric preprocessing alternative: log1p of the raw features, then standardized."""
    return Pipeline([("log", FunctionTransformer(log1p_float)), ("scale", StandardScaler())])


def optional_regressor_grid(quick: bool = False) -> list[tuple[type, dict[str, list[Any]]]]:
    """Grid entries of the additional models whose dependencies are available."""
    grid: list[tuple[type, dict[str, list[Any]]]] = [
        (
            LogTargetHGBRegressor,
            {
                "regressor__regressor": [HistGradientBoostingRegressor(random_state=RANDOM_STATE)],
                "regressor__regressor__max_iter": [200] if quick else [200, 500],
                "regressor__regressor__learning_rate": [0.1] if quick else [0.05, 0.1],
                "regressor__regressor__max_depth": [None] if quick else [None, 7],
                "regressor__regressor__min_samples_leaf": [5] if quick else [5, 20],
            },
        ),
        (
            LogTargetMLPRegressor,
            {
                "regressor__regressor": [
                    MLPRegressor(early_stopping=True, max_iter=2000, random_state=RANDOM_STATE)
                ],
                "regressor__regressor__hidden_layer_sizes": [(64, 64)]
                if quick
                else [(64, 64), (128, 128), (256, 128, 64)],
                "regressor__regressor__alpha": [1e-3] if quick else [1e-4, 1e-2],
                "regressor__regressor__learning_rate_init": [1e-3] if quick else [1e-3, 3e-3],
            },
        ),
    ]
    status = dependency_status()
    if status["TabTransformerRegressor"] is None:
        grid.append(
            (
                TabTransformerRegressor,
                {
                    "regressor__d_model": [16] if quick else [32, 64],
                    "regressor__n_layers": [1] if quick else [2, 3],
                    "regressor__loss": ["mse"] if quick else ["mse", "huber"],
                    "regressor__max_epochs": [20] if quick else [300],
                    "regressor__target_transform": ["log1p"],
                    "regressor__random_state": [RANDOM_STATE],
                    "preprocessor__num": [StandardScaler()]
                    if quick
                    else [StandardScaler(), _log_input_scaler()],
                },
            )
        )
    else:
        logger.info("TabTransformerRegressor not available (%s)", status["TabTransformerRegressor"])
    if status["SymbolicRegressor"] is None:
        grid.append(
            (
                SymbolicRegressor,
                {
                    # raw feature units: no scaling of the numeric inputs
                    "preprocessor__num": [
                        FunctionTransformer(validate=True, feature_names_out="one-to-one")
                    ],
                    "regressor__unary_operators": [("square", "ceil")]
                    if quick
                    else [("square", "ceil"), ("square", "ceil", "log2_abs")],
                    "regressor__maxsize": [15] if quick else [25, 35],
                    "regressor__niterations": [5] if quick else [60],
                    "regressor__timeout_in_seconds": [60] if quick else [None],
                    "regressor__random_state": [RANDOM_STATE],
                },
            )
        )
    else:
        logger.info("SymbolicRegressor not available (%s)", status["SymbolicRegressor"])
    return grid
