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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Optional

from finn.qor.database import POWER_TARGET_COL
from finn.qor.features import SPECS, OperatorFeatureSpec

logger = logging.getLogger(__name__)

#: Environment variable pointing to the directory with fitted models.
MODEL_DIR_ENV_VAR = "FINN_QOR_MODEL_DIR"

#: Target column for post-synthesis resource counts, per resource type.
RESOURCE_TARGET_PREFIX = "metrics.synth.resources."

RANDOM_STATE = 42

#: Regressor classes and their hyperparameter grids for model selection.
#: Linear models, SVR and MLPs were evaluated during development and perform poorly for
#: this use case, so they are not included.
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


def default_scoring(target: str) -> str:
    """Scoring used for model selection: relative error for resources, R^2 for power."""
    return "r2" if target == POWER_TARGET_COL else "neg_mean_absolute_percentage_error"


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
        self.operator = operator
        self.target = target
        self.spec = spec or SPECS[operator]
        self.pipeline = pipeline
        self.metadata: dict[str, Any] = metadata or {}

    @property
    def feature_cols(self) -> list[str]:
        return self.spec.feature_cols

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

    def fit(self, df: pd.DataFrame, regressor: Optional[BaseEstimator] = None) -> "QoREstimator":
        """Fit ``regressor`` (default: :class:`KNeighborsRegressor`) on all rows of ``df``."""
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
        return int(round(prediction)) if self.is_integer_target else prediction

    def _fit_metadata(self, df: pd.DataFrame) -> dict[str, Any]:
        regressor = self.pipeline.named_steps["regressor"]
        return {
            "operator": self.operator,
            "target": self.target,
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols(df),
            "regressor": type(regressor).__name__,
            "regressor_params": _jsonable(regressor.get_params()),
            "n_samples": int(len(df)),
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "database_commits": sorted(df["commit"].dropna().unique().tolist())
            if "commit" in df.columns
            else [],
            "database_pipeline_ids": sorted(int(x) for x in df["pipeline_id"].dropna().unique())
            if "pipeline_id" in df.columns
            else [],
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
        return cls(operator, target, pipeline=pipeline, metadata=metadata)

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


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_safe)) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _sample_errors(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    abs_error = np.abs(y_pred - y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(y_true != 0, abs_error / np.abs(y_true) * 100, np.nan)
    return pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "abs_error": abs_error, "abs_pct_error": ape}
    )


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
    best parameter set is additionally evaluated with fold-wise MAPE/MSE/RMSE and out-of-fold
    per-sample errors, so that regressors can be compared beyond the primary score. Regressors
    that fail are recorded with NaN scores instead of aborting the selection. The winning
    pipeline is stored in ``estimator`` (with updated metadata) and returned.
    """
    scoring = scoring or default_scoring(estimator.target)
    grid = grid if grid is not None else REGRESSOR_GRID
    X = df[estimator.feature_cols]
    y = df[estimator.target].values
    cv_obj = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=RANDOM_STATE)
    n_splits = cv_obj.get_n_splits()
    extra_scorers = {
        "mape": make_scorer(_mape, greater_is_better=False),
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "rmse": make_scorer(_rmse, greater_is_better=False),
    }

    rows = []
    sample_errors: dict[str, pd.DataFrame] = {}
    fold_scores: dict[str, np.ndarray] = {}
    best: Optional[tuple[float, str, Pipeline]] = None
    for reg_cls, param_grid in grid:
        name = reg_cls.__name__
        try:
            pipeline = estimator.build_pipeline(reg_cls(), df)
            search = GridSearchCV(pipeline, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs)
            search.fit(X, y)
            res = search.cv_results_
            best_idx, worst_idx = search.best_index_, int(np.argmin(res["mean_test_score"]))
            best_folds = np.array([res[f"split{i}_test_score"][best_idx] for i in range(n_splits)])
            worst_folds = np.array(
                [res[f"split{i}_test_score"][worst_idx] for i in range(n_splits)]
            )
            model = search.best_estimator_

            extra = cross_validate(model, X, y, cv=cv_obj, scoring=extra_scorers, n_jobs=n_jobs)
            y_pred = cross_val_predict(model, X, y, cv=cv_obj, n_jobs=n_jobs)
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
                    "fold_mape": float(np.mean(-extra["test_mape"])),
                    "fold_mse": float(np.mean(-extra["test_mse"])),
                    "fold_rmse": float(np.mean(-extra["test_rmse"])),
                    "sample_mape": float(np.nanmean(sample_errors[name]["abs_pct_error"])),
                    "n_param_combinations": len(res["params"]),
                    "best_params": json.dumps(_jsonable(search.best_params_)),
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
            rows.append({"regressor": name, "mean_score": np.nan})

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
            name = reg_cls.__name__
            try:
                pipeline = estimator.build_pipeline(reg_cls(), df)
                search = GridSearchCV(
                    pipeline, param_grid, cv=cv_obj, scoring=scoring, n_jobs=n_jobs
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
