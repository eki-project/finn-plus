"""Small scikit-learn building blocks for the QoR model zoo.

* :func:`log1p_float` / :func:`signed_log1p` – picklable transforms for log-scaled inputs and
  targets (resource counts span several orders of magnitude, and minimizing the squared error
  in log space approximates minimizing the relative error the LUT models are selected by).
* :class:`LogTargetHGBRegressor` / :class:`LogTargetMLPRegressor` – gradient boosting and a
  tuned multi-layer perceptron fitted on the log-transformed target.

Only depends on numpy and scikit-learn.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

RANDOM_STATE = 42


def log1p_float(X):
    """``log1p`` of the (non-negative) input as float64; module-level so that pipelines using
    it pickle, and explicit about the dtype (bool/int mixes would otherwise degrade)."""
    return np.log1p(np.asarray(X, dtype=float))


def signed_log1p(y):
    """Sign-preserving ``log1p`` (``sign(y) * log1p(|y|)``), defined for any real target."""
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(y):
    """Inverse of :func:`signed_log1p`."""
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.expm1(np.abs(y))


class LogTargetRegressor(TransformedTargetRegressor):
    """Regressor fitted on the sign-preserving log of the target (predictions are mapped
    back). ``regressor`` defaults to :class:`HistGradientBoostingRegressor`."""

    def __init__(self, regressor=None, func=signed_log1p, inverse_func=signed_expm1):
        if regressor is None:
            regressor = self._default_regressor()
        super().__init__(
            regressor=regressor, func=func, inverse_func=inverse_func, check_inverse=False
        )

    @staticmethod
    def _default_regressor():
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE)


class LogTargetHGBRegressor(LogTargetRegressor):
    """Histogram gradient boosting on the log target."""


class LogTargetMLPRegressor(LogTargetRegressor):
    """Multi-layer perceptron (early stopping) on the log target; the inputs are expected to
    be standardized by the pipeline's preprocessor."""

    @staticmethod
    def _default_regressor():
        return MLPRegressor(early_stopping=True, max_iter=2000, random_state=RANDOM_STATE)
