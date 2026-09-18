"""Fitting, persisting and applying regression models for empirical QoR estimation.

A :class:`QoREstimator` wraps a scikit-learn pipeline (feature preprocessing + regressor)
that predicts one target metric (e.g. post-synthesis LUTs or measured power) of one
operator from the features defined in :mod:`finn.qor.features`. Models are persisted as a
pickle of the fitted sklearn pipeline plus a JSON sidecar with metadata, so that loading
does not depend on the FINN class layout.

This module must stay importable without FINN/QONNX (pandas, numpy, scikit-learn only),
because the CI job that fits the models runs outside of the FINN environment.
"""

import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import sklearn
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import get_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Optional

from finn.qor.database import BRAM_TARGET_COL, POWER_TARGET_COL
from finn.qor.features import SPECS, OperatorFeatureSpec

logger = logging.getLogger(__name__)

#: Environment variable pointing to the directory with fitted models.
MODEL_DIR_ENV_VAR = "FINN_QOR_MODEL_DIR"

#: Target column for post-synthesis resource counts, per resource type.
RESOURCE_TARGET_PREFIX = "metrics.synth.resources."

#: Resource type (as reported by the analytical estimate) -> target column of the model.
#: BRAMs are modelled in 18K-block equivalents (BRAM_36K counts twice) so that the
#: prediction maps onto the analytical ``BRAM_18K`` key.
RESOURCE_TARGETS: dict[str, str] = {
    "LUT": RESOURCE_TARGET_PREFIX + "LUT",
    "DSP": RESOURCE_TARGET_PREFIX + "DSP",
    "BRAM_18K": BRAM_TARGET_COL,
    "URAM": RESOURCE_TARGET_PREFIX + "URAM",
}
#: Default targets to fit, resources first
DEFAULT_TARGETS: list[str] = [*RESOURCE_TARGETS.values(), POWER_TARGET_COL]


def resource_target(res_type: str) -> str:
    """Target column of a resource type (``LUT``, ``DSP``, ``BRAM_18K``, ``URAM``)."""
    return RESOURCE_TARGETS.get(res_type, RESOURCE_TARGET_PREFIX + res_type)


RANDOM_STATE = 42

#: Base regressor classes and their hyperparameter grids for model selection (tree
#: ensembles and k-NN). Linear models and SVR perform poorly for this use case. Additional
#: models (log-target boosting/MLP, tabular transformer, symbolic regression) live in
#: :mod:`finn.qor.models` and are added by :func:`available_regressor_grid`.
REGRESSOR_GRID: list[tuple[type[BaseEstimator], dict[str, list[Any]]]] = [
    (
        KNeighborsRegressor,
        {
            "regressor__n_neighbors": [3, 5, 7, 10, 15, 20],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
            "regressor__algorithm": ["auto", "ball_tree", "kd_tree"],
        },
    ),
    (
        DecisionTreeRegressor,
        {
            "regressor__max_depth": [None, 3, 5, 7, 10, 15],
            "regressor__min_samples_split": [2, 5, 10, 20],
            "regressor__min_samples_leaf": [1, 2, 5, 10],
            "regressor__max_features": ["sqrt", "log2", None],
            "regressor__random_state": [RANDOM_STATE],
        },
    ),
    (
        RandomForestRegressor,
        {
            "regressor__n_estimators": [50, 100, 200, 300],
            "regressor__max_depth": [None, 5, 10, 15, 20],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__max_features": ["sqrt", "log2", None],
            "regressor__bootstrap": [True, False],
            "regressor__random_state": [RANDOM_STATE],
        },
    ),
    (
        GradientBoostingRegressor,
        {
            "regressor__n_estimators": [50, 100, 200, 300],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "regressor__max_depth": [3, 4, 5, 6],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__subsample": [0.8, 0.9, 1.0],
            "regressor__random_state": [RANDOM_STATE],
        },
    ),
    (
        HistGradientBoostingRegressor,
        {
            "regressor__max_iter": [100, 200, 300, 500],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "regressor__max_depth": [None, 3, 5, 7, 10],
            "regressor__min_samples_leaf": [1, 5, 10, 20],
            "regressor__l2_regularization": [0.0, 0.1, 1.0],
            "regressor__random_state": [RANDOM_STATE],
        },
    ),
]

#: Strongly reduced grid for smoke tests and quick experiments.
REGRESSOR_GRID_QUICK: list[tuple[type[BaseEstimator], dict[str, list[Any]]]] = [
    (KNeighborsRegressor, {"regressor__n_neighbors": [3, 5]}),
    (
        RandomForestRegressor,
        {"regressor__n_estimators": [50], "regressor__random_state": [RANDOM_STATE]},
    ),
]


GridEntry = tuple[type[BaseEstimator], dict[str, list[Any]]]


def regressor_name(reg_cls: type) -> str:
    """Name under which a regressor class appears in selection results and CLI options."""
    return reg_cls.__name__


def available_regressor_grid(
    quick: bool = False,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> list[GridEntry]:
    """The base grid plus the additional models of :mod:`finn.qor.models` whose runtime
    dependencies are importable, optionally filtered by regressor name (unknown names in
    ``include``/``exclude`` raise ValueError listing the known ones)."""
    from finn.qor.models import optional_regressor_grid  # lazy: models import this module

    grid = list(REGRESSOR_GRID_QUICK if quick else REGRESSOR_GRID) + optional_regressor_grid(quick)
    known = [regressor_name(cls) for cls, _ in grid]
    for names in (include, exclude):
        unknown = [n for n in (names or []) if n not in known]
        if unknown:
            raise ValueError(f"Unknown regressor(s) {unknown}, known: {known}")
    if include:
        grid = [entry for entry in grid if regressor_name(entry[0]) in include]
    if exclude:
        grid = [entry for entry in grid if regressor_name(entry[0]) not in exclude]
    return grid


def default_scoring(target: str) -> str:
    """Scoring used for model selection: R^2 for power, relative error (MAPE) for LUTs and
    absolute error (MAE) for the zero-inflated DSP/BRAM/URAM counts, where a relative error
    is undefined for most samples."""
    if target == POWER_TARGET_COL:
        return "r2"
    if target == RESOURCE_TARGETS["LUT"]:
        return "neg_mean_absolute_percentage_error"
    if target.startswith(RESOURCE_TARGET_PREFIX):
        return "neg_mean_absolute_error"
    return "neg_mean_absolute_percentage_error"


def target_signal(y: np.ndarray) -> dict[str, float]:
    """Summary of a target vector: sample count, nonzero count, distinct values and the
    fraction of zeros (used to decide whether a model is worth fitting)."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    return {
        "n": int(len(y)),
        "n_nonzero": int(np.count_nonzero(y)),
        "n_distinct": int(len(np.unique(y))),
        "zero_fraction": float(np.mean(y == 0)) if len(y) else float("nan"),
    }


def has_enough_signal(y: np.ndarray, min_nonzero: int = 10, min_distinct: int = 3) -> bool:
    """Whether a target has enough nonzero and distinct values to fit a regression model on
    (otherwise the analytical estimate remains the better choice)."""
    signal = target_signal(y)
    return signal["n_nonzero"] >= min_nonzero and signal["n_distinct"] >= min_distinct


def model_basename(operator: str, target: str) -> str:
    """File basename (without extension) under which a model is stored."""
    return f"{operator}__{target.replace('.', '_')}"


class QoREstimator:
    """Regression model predicting one QoR target metric for one operator type."""

    def __init__(
        self,
        operator: str,
        target: str,
        spec: Optional[OperatorFeatureSpec] = None,
        pipeline: Optional[Pipeline] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Create an estimator for operator/target, unfitted unless ``pipeline`` is given."""
        self.operator = operator
        self.target = target
        self.spec = spec or SPECS[operator]
        self.pipeline = pipeline
        self.metadata: dict[str, Any] = metadata or {}

    @property
    def feature_cols(self) -> list[str]:
        """Feature columns in the order fed to the pipeline: those recorded at fit time
        (loaded models), else the operator spec's."""
        recorded = self.metadata.get("feature_cols")
        return list(recorded) if recorded else self.spec.feature_cols

    @property
    def is_integer_target(self) -> bool:
        """Resource counts are reported as integers, everything else as float."""
        return self.target.startswith(RESOURCE_TARGET_PREFIX)

    def categorical_cols(self, df: pd.DataFrame) -> list[str]:
        """Feature columns that hold strings (or None) and need one-hot encoding."""
        return [
            c for c in self.feature_cols if df[c].dtype == object or df[c].dtype.name == "category"
        ]

    def build_pipeline(self, regressor: BaseEstimator, df: pd.DataFrame) -> Pipeline:
        """Create the (unfitted) preprocessing + regressor pipeline for the given data."""
        categorical = self.categorical_cols(df)
        numeric = [c for c in self.feature_cols if c not in categorical]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])

    def rows_with_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows of ``df`` that have a value for the target (e.g. runs without a power
        measurement have NaN there and cannot be used for fitting)."""
        if self.target not in df.columns:
            raise KeyError(f"Target column '{self.target}' not in data")
        valid = df.dropna(subset=[self.target])
        if len(valid) < len(df):
            logger.info(
                "%s/%s: ignoring %d of %d samples without a target value",
                self.operator,
                self.target,
                len(df) - len(valid),
                len(df),
            )
        if valid.empty:
            raise ValueError(f"No samples with a value for target '{self.target}'")
        return valid

    def fit(self, df: pd.DataFrame, regressor: Optional[BaseEstimator] = None) -> "QoREstimator":
        """Fit ``regressor`` (default: :class:`KNeighborsRegressor`) on all rows of ``df``
        that have a target value."""
        df = self.rows_with_target(df)
        pipeline = self.build_pipeline(regressor or KNeighborsRegressor(), df)
        pipeline.fit(df[self.feature_cols], df[self.target].values)
        self.pipeline = pipeline
        self.metadata.update(self._fit_metadata(df))
        return self

    def predict(self, features: pd.DataFrame) -> int | float:
        """Predict the target for a single sample given as a one-row DataFrame."""
        if self.pipeline is None:
            raise ValueError("Estimator is not fitted")
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            raise ValueError(f"Feature columns missing from input: {missing}")
        if len(features) != 1:
            raise ValueError("predict() expects exactly one sample")
        prediction = float(self.pipeline.predict(features[self.feature_cols])[0])
        if self.is_integer_target:
            # resource counts are non-negative integers
            return int(round(max(prediction, 0.0)))
        return prediction

    def check_runtime(self) -> None:
        """Raise ImportError if a step of the fitted pipeline needs a missing dependency
        (e.g. torch for the tabular transformer)."""
        if self.pipeline is None:
            return
        for _, step in self.pipeline.steps:
            check = getattr(step, "check_runtime", None)
            if check is not None:
                check()

    def covers(self, features: pd.DataFrame) -> bool:
        """Whether every categorical feature value of ``features`` occurred in the training
        data (recorded in the metadata at fit time). Values outside the training categories
        would be one-hot encoded as all-zeros, i.e. silently extrapolated; callers should
        fall back to the analytical estimate instead. Always True for models without the
        recorded categories (fitted with an older version)."""
        categories = self.metadata.get("categories")
        if not categories:
            return True
        for col, allowed in categories.items():
            if col not in features.columns:
                return False
            value = features[col].iloc[0]
            key = (
                "None"
                if value is None or (isinstance(value, float) and np.isnan(value))
                else str(value)
            )
            if key not in allowed:
                return False
        return True

    def _fit_metadata(self, df: pd.DataFrame) -> dict[str, Any]:
        """Describe the fitted pipeline and its training data for the JSON sidecar."""
        regressor = self.pipeline.named_steps["regressor"]
        categorical = self.categorical_cols(df)
        extra: dict[str, Any] = {}
        fit_info = getattr(regressor, "fit_info_", None)
        if fit_info:
            extra["regressor_info"] = _jsonable(fit_info)
            for key in ("torch_version", "pysr_version"):
                if key in fit_info:
                    extra[key] = fit_info[key]
        if hasattr(regressor, "expr_srepr_"):
            extra["equation"] = self._equation_metadata(regressor)
        return {
            **extra,
            "operator": self.operator,
            "target": self.target,
            "feature_cols": self.feature_cols,
            "categorical_cols": categorical,
            # observed values per categorical column (None as "None"), see covers()
            "categories": {
                c: sorted({"None" if pd.isna(v) else str(v) for v in df[c].tolist()})
                for c in categorical
            },
            "regressor": type(regressor).__name__,
            "regressor_params": _jsonable(regressor.get_params()),
            "n_samples": int(len(df)),
            "target_signal": target_signal(df[self.target].values),
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "database_commits": sorted(df["commit"].dropna().unique().tolist())
            if "commit" in df.columns
            else [],
            "database_pipeline_ids": sorted(int(x) for x in df["pipeline_id"].dropna().unique())
            if "pipeline_id" in df.columns
            else [],
        }

    def _equation_metadata(self, regressor) -> dict[str, Any]:
        """Symbolic regression result with the pipeline's feature names substituted (and
        standardized numeric inputs unscaled), as sympy string, LaTeX and PySR form."""
        import sympy

        from finn.qor.models.symbolic import (
            rename_expression,
            sanitize_feature_names,
            unscale_expression,
        )

        preprocessor = self.pipeline.named_steps["preprocessor"]
        names = sanitize_feature_names(preprocessor.get_feature_names_out())
        expr = rename_expression(regressor.sympy(), names)
        num = preprocessor.named_transformers_.get("num")
        scaler = num if isinstance(num, StandardScaler) else None
        if scaler is not None and getattr(scaler, "mean_", None) is not None:
            n_num = len(scaler.mean_)
            expr = unscale_expression(expr, names[:n_num], scaler.mean_, scaler.scale_)
        return {
            "variables": names,
            "sympy": str(expr),
            "latex": sympy.latex(expr),
            "pysr": regressor.equation_,
            "complexity": regressor.complexity_,
            "loss": regressor.loss_,
            "pareto_front": regressor.pareto_front_,
        }

    def save(self, model_dir: str) -> tuple[str, str]:
        """Store the fitted pipeline (pickle) and its metadata (JSON) in ``model_dir``."""
        if self.pipeline is None:
            raise ValueError("Estimator is not fitted")
        os.makedirs(model_dir, exist_ok=True)
        base = os.path.join(model_dir, model_basename(self.operator, self.target))
        with open(base + ".pkl", "wb") as f:
            pickle.dump(self.pipeline, f)
        with open(base + ".json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info("Saved %s/%s model to %s.pkl", self.operator, self.target, base)
        return base + ".pkl", base + ".json"

    @classmethod
    def load(cls, model_dir: str, operator: str, target: str) -> "QoREstimator":
        """Load a model stored by :meth:`save`."""
        base = os.path.join(model_dir, model_basename(operator, target))
        with open(base + ".json", "r") as f:
            metadata = json.load(f)
        with open(base + ".pkl", "rb") as f:
            pipeline = pickle.load(f)
        if metadata.get("sklearn_version") != sklearn.__version__:
            logger.warning(
                "Model %s was fitted with scikit-learn %s, running %s",
                base,
                metadata.get("sklearn_version"),
                sklearn.__version__,
            )
        estimator = cls(operator, target, pipeline=pipeline, metadata=metadata)
        estimator.check_runtime()
        return estimator

    @staticmethod
    def available_models(model_dir: str) -> list[tuple[str, str]]:
        """List ``(operator, target)`` pairs stored in ``model_dir`` (via their JSON sidecars)."""
        if not os.path.isdir(model_dir):
            return []
        found = []
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(model_dir, filename), "r") as f:
                meta = json.load(f)
            if "operator" in meta and "target" in meta:
                pkl = os.path.join(model_dir, model_basename(meta["operator"], meta["target"]))
                if os.path.isfile(pkl + ".pkl"):
                    found.append((meta["operator"], meta["target"]))
        return found


@dataclass
class SelectionResult:
    """Outcome of :func:`select_regressor`."""

    #: Name of the best regressor class.
    best_name: str
    #: Fitted best pipeline (fitted on all data by GridSearchCV's refit).
    best_pipeline: Pipeline
    #: Mean cross-validated score of the best regressor/parameter set.
    best_score: float
    #: Scoring name the selection was based on.
    scoring: str
    #: One row per regressor: mean/std/min/max score of the best parameter set, mean/min/max of
    #: the worst parameter set, fold-wise MAPE/MSE/RMSE, best params and grid size.
    scores: pd.DataFrame
    #: Per regressor: out-of-fold predictions and per-sample errors (columns y_true, y_pred,
    #: abs_error, abs_pct_error [NaN for y_true == 0]).
    sample_errors: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Per regressor: fold scores of the best parameter set.
    fold_scores: dict[str, np.ndarray] = field(default_factory=dict)
    #: Per regressor: the complete GridSearchCV ``cv_results_`` (one row per parameter set).
    cv_results: dict[str, pd.DataFrame] = field(default_factory=dict)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error over the samples with a nonzero target (NaN if none),
    consistent with :func:`finn.qor.evaluation.error_metrics`."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def _zero_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of zero targets that are predicted as zero after rounding (NaN if the target
    has no zeros); relevant for zero-inflated resource counts (DSP, BRAM, URAM)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    zeros = y_true == 0
    if not zeros.any():
        return float("nan")
    return float(np.mean(np.round(np.maximum(y_pred[zeros], 0.0)) == 0))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _sample_errors(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-sample absolute and percentage errors (NaN percentage where the target is 0)."""
    abs_error = np.abs(y_pred - y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(y_true != 0, abs_error / np.abs(y_true) * 100, np.nan)
    return pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "abs_error": abs_error, "abs_pct_error": ape}
    )


def _oof_evaluate(
    model: Pipeline, X: pd.DataFrame, y: np.ndarray, cv_obj, n_jobs: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit ``model`` on every CV split and evaluate on the held-out part: returns the
    fold-wise MAPE/MAE/MSE/RMSE and the out-of-fold prediction of every sample (averaged
    over repeats for RepeatedKFold)."""

    def one(train_idx, test_idx):
        fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
        pred = np.asarray(fitted.predict(X.iloc[test_idx]), dtype=float)
        y_test = y[test_idx]
        return (
            test_idx,
            pred,
            {
                "mape": _mape(y_test, pred),
                "mae": _mae(y_test, pred),
                "mse": float(mean_squared_error(y_test, pred)),
                "rmse": _rmse(y_test, pred),
            },
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(one)(train_idx, test_idx) for train_idx, test_idx in cv_obj.split(X)
    )
    y_pred = np.zeros(len(y))
    counts = np.zeros(len(y))
    folds: dict[str, list[float]] = {"mape": [], "mae": [], "mse": [], "rmse": []}
    for test_idx, pred, metrics in results:
        y_pred[test_idx] += pred
        counts[test_idx] += 1
        for key, value in metrics.items():
            folds[key].append(value)
    y_pred = y_pred / np.maximum(counts, 1)
    return {k: np.array(v) for k, v in folds.items()}, y_pred


def _single_predict_ms(model: Pipeline, X: pd.DataFrame, repeats: int = 20) -> float:
    """Median wall time of predicting one sample (the per-node cost in a build), in ms."""
    row = X.iloc[[0]]
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times))


def select_regressor(
    estimator: QoREstimator,
    df: pd.DataFrame,
    scoring: Optional[str] = None,
    cv: int = 5,
    n_repeats: int = 1,
    n_jobs: int = -1,
    grid: Optional[list[tuple[type[BaseEstimator], dict[str, list[Any]]]]] = None,
) -> SelectionResult:
    """Grid-search all regressors in ``grid`` and pick the best one by cross-validated score.

    For each regressor, GridSearchCV tunes the hyperparameters with a RepeatedKFold split. The
    best parameter set is additionally evaluated with fold-wise MAPE/MAE/MSE/RMSE and
    out-of-fold per-sample errors, so that regressors can be compared beyond the primary
    score, and its fit/predict wall times are recorded (cost vs. accuracy). Regressors that
    fail are recorded with NaN scores instead of aborting the selection. Regressors may set
    ``preferred_n_jobs`` (e.g. 1 for PySR, whose Julia runtime must not be forked). The
    winning pipeline is stored in ``estimator`` (with updated metadata) and returned. Samples
    without a target value are ignored.
    """
    scoring = scoring or default_scoring(estimator.target)
    grid = grid if grid is not None else REGRESSOR_GRID
    df = estimator.rows_with_target(df)
    X = df[estimator.feature_cols]
    y = df[estimator.target].values
    cv_obj = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=RANDOM_STATE)
    n_splits = cv_obj.get_n_splits()

    rows = []
    sample_errors: dict[str, pd.DataFrame] = {}
    fold_scores: dict[str, np.ndarray] = {}
    cv_results: dict[str, pd.DataFrame] = {}
    best: Optional[tuple[float, str, Pipeline]] = None
    for reg_cls, param_grid in grid:
        name = regressor_name(reg_cls)
        n_jobs_eff = getattr(reg_cls, "preferred_n_jobs", None) or n_jobs
        try:
            pipeline = estimator.build_pipeline(reg_cls(), df)
            search = GridSearchCV(
                pipeline, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs_eff
            )
            t_search = time.perf_counter()
            search.fit(X, y)
            search_time = time.perf_counter() - t_search
            res = search.cv_results_
            cv_results[name] = pd.DataFrame(res).assign(regressor=name)
            best_idx, worst_idx = search.best_index_, int(np.argmin(res["mean_test_score"]))
            best_folds = np.array([res[f"split{i}_test_score"][best_idx] for i in range(n_splits)])
            worst_folds = np.array(
                [res[f"split{i}_test_score"][worst_idx] for i in range(n_splits)]
            )
            model = search.best_estimator_

            t_oof = time.perf_counter()
            folds, y_pred = _oof_evaluate(model, X, y, cv_obj, n_jobs_eff)
            oof_time = time.perf_counter() - t_oof
            sample_errors[name] = _sample_errors(y, y_pred)
            fold_scores[name] = best_folds
            rows.append(
                {
                    "regressor": name,
                    "mean_score": search.best_score_,
                    "std_score": res["std_test_score"][best_idx],
                    "min_fold_score": best_folds.min(),
                    "max_fold_score": best_folds.max(),
                    "worst_mean_score": res["mean_test_score"][worst_idx],
                    "worst_min_fold_score": worst_folds.min(),
                    "worst_max_fold_score": worst_folds.max(),
                    "fold_mape": float(np.nanmean(folds["mape"])),
                    "fold_mae": float(np.mean(folds["mae"])),
                    "fold_mse": float(np.mean(folds["mse"])),
                    "fold_rmse": float(np.mean(folds["rmse"])),
                    "sample_mape": float(np.nanmean(sample_errors[name]["abs_pct_error"])),
                    "sample_mae": float(np.mean(sample_errors[name]["abs_error"])),
                    "zero_hit_rate": _zero_hit_rate(y, y_pred),
                    "n_param_combinations": len(res["params"]),
                    "best_params": json.dumps(_jsonable(search.best_params_)),
                    # cost: mean fit/score time of the best parameter set (one fold), the
                    # whole grid search, the out-of-fold pass and a single prediction
                    "fit_time_s": float(res["mean_fit_time"][best_idx]),
                    "predict_time_s": float(res["mean_score_time"][best_idx]),
                    "search_time_s": search_time,
                    "oof_time_s": oof_time,
                    "single_predict_ms": _single_predict_ms(model, X),
                }
            )
            logger.info(
                "%s: mean %s = %.4f (std %.4f), fold MAPE %.2f%%",
                name,
                scoring,
                search.best_score_,
                res["std_test_score"][best_idx],
                rows[-1]["fold_mape"],
            )
            if best is None or search.best_score_ > best[0]:
                best = (search.best_score_, name, model)
        except Exception as e:  # noqa: BLE001 - one failing regressor must not abort selection
            logger.warning("%s failed: %s", name, e)
            rows.append({"regressor": name, "mean_score": np.nan, "error": str(e)})

    if best is None:
        raise RuntimeError("All regressors failed during model selection")
    best_score, best_name, best_pipeline = best
    logger.info("Best regressor: %s (mean %s = %.4f)", best_name, scoring, best_score)

    estimator.pipeline = best_pipeline
    estimator.metadata.update(estimator._fit_metadata(df))
    estimator.metadata.update({"cv_scoring": scoring, "cv_score": float(best_score)})
    return SelectionResult(
        best_name=best_name,
        best_pipeline=best_pipeline,
        best_score=float(best_score),
        scoring=scoring,
        scores=pd.DataFrame(rows).set_index("regressor"),
        sample_errors=sample_errors,
        fold_scores=fold_scores,
        cv_results=cv_results,
    )


def learning_curve(
    estimator: QoREstimator,
    df: pd.DataFrame,
    fractions: list[float],
    scoring: Optional[str] = None,
    cv: int = 5,
    n_repeats: int = 1,
    n_jobs: int = -1,
    test_size: float = 0.2,
    evaluation: str = "holdout",
    grid: Optional[list[tuple[type[BaseEstimator], dict[str, list[Any]]]]] = None,
) -> pd.DataFrame:
    """Evaluate regressor performance as a function of the amount of training data.

    The data is split once into train/test. For every fraction, a subset of the training data
    of that size is drawn, every regressor in ``grid`` is tuned on it (GridSearchCV) and the
    tuned model is scored either on the held-out test set (``evaluation="holdout"``) or on the
    complete dataset (``evaluation="full_dataset"``, optimistic, but uses all data).

    Returns a DataFrame indexed by fraction with one column per regressor (NaN where a
    fraction had too few samples for the CV or the regressor failed).
    """
    if evaluation not in ("holdout", "full_dataset"):
        raise ValueError(f"Unknown evaluation mode '{evaluation}'")
    scoring = scoring or default_scoring(estimator.target)
    grid = grid if grid is not None else REGRESSOR_GRID
    scorer = get_scorer(scoring)
    df = estimator.rows_with_target(df)

    if evaluation == "holdout":
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
    else:
        train_df, test_df = df, df
    X_test, y_test = test_df[estimator.feature_cols], test_df[estimator.target].values
    cv_obj = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=RANDOM_STATE)

    results: dict[float, dict[str, float]] = {}
    for frac in fractions:
        n_train = int(len(train_df) * frac)
        results[frac] = {}
        if n_train < cv:
            logger.info("Fraction %.2f: only %d samples, skipping", frac, n_train)
            continue
        subset = train_df.sample(n=n_train, random_state=RANDOM_STATE)
        X_train, y_train = subset[estimator.feature_cols], subset[estimator.target].values
        for reg_cls, param_grid in grid:
            name = regressor_name(reg_cls)
            n_jobs_eff = getattr(reg_cls, "preferred_n_jobs", None) or n_jobs
            try:
                pipeline = estimator.build_pipeline(reg_cls(), df)
                search = GridSearchCV(
                    pipeline, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs_eff
                )
                search.fit(X_train, y_train)
                results[frac][name] = float(scorer(search.best_estimator_, X_test, y_test))
            except Exception as e:  # noqa: BLE001
                logger.warning("%s failed at fraction %.2f: %s", name, frac, e)
                results[frac][name] = np.nan
        logger.info("Fraction %.2f (%d samples): %s", frac, n_train, results[frac])
    curve = pd.DataFrame.from_dict(results, orient="index")
    curve.index.name = "fraction"
    curve.attrs.update({"scoring": scoring, "evaluation": evaluation, "n_train": len(train_df)})
    return curve


def _jsonable(obj: Any) -> Any:
    """Make sklearn parameter dicts JSON serializable (numpy scalars, classes, ...)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
