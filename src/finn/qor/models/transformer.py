"""Tabular transformer regressor (FT-Transformer style) as a scikit-learn estimator.

The estimator consumes the preprocessed feature matrix of the QoR pipeline (standardized
numeric columns and one-hot encoded categorical columns) and embeds every column as one
token. torch is imported lazily in ``fit``/``predict`` only; the pickled state holds the
hyperparameters and the weights as numpy arrays, so models load on CPU-only machines and the
pickle does not depend on the torch version.
"""

import importlib.util
import numpy as np
import time
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from typing import Any

from finn.qor.models.sklearn_extras import RANDOM_STATE, signed_expm1, signed_log1p

_HP_KEYS = (
    "d_model",
    "n_layers",
    "n_heads",
    "ffn_mult",
    "dropout",
    "pooling",
    "loss",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "lr_patience",
    "lr_factor",
)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


#: rebuilt torch modules per estimator instance (kept outside __dict__ so that predict() does
#: not modify the estimator and pickles never contain torch objects)
_CACHE: dict[int, Any] = {}


class TabTransformerRegressor(BaseEstimator, RegressorMixin):
    """Feature-tokenizer transformer for tabular regression.

    Parameters mirror the usual training knobs; ``target_transform`` (``"none"``, ``"log1p"``
    or ``"standardize"``) is applied to the target before training. ``torch_threads`` limits
    the intra-op threads so that GridSearchCV's process parallelism does not oversubscribe
    the CPU.
    """

    #: hint for the model selection: use the caller's n_jobs
    preferred_n_jobs = None

    def __init__(
        self,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_mult=2,
        dropout=0.1,
        pooling="cls",
        target_transform="log1p",
        loss="mse",
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        max_epochs=300,
        patience=30,
        validation_fraction=0.1,
        lr_patience=10,
        lr_factor=0.5,
        torch_threads=1,
        random_state=RANDOM_STATE,
        verbose=0,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.dropout = dropout
        self.pooling = pooling
        self.target_transform = target_transform
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.torch_threads = torch_threads
        self.random_state = random_state
        self.verbose = verbose

    # -- helpers ---------------------------------------------------------------------------
    def check_runtime(self) -> None:
        """Raise ImportError if torch is not installed."""
        if not torch_available():
            raise ImportError("TabTransformerRegressor requires torch")

    def _hp(self) -> dict:
        return {k: getattr(self, k) for k in _HP_KEYS}

    @staticmethod
    def _dense(X) -> np.ndarray:
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return signed_log1p(y)
        if self.target_transform == "standardize":
            return (y - self.y_shift_) / self.y_scale_
        if self.target_transform == "none":
            return np.asarray(y, dtype=float)
        raise ValueError(f"unknown target_transform {self.target_transform!r}")

    def _inverse_target(self, y: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return signed_expm1(y)
        if self.target_transform == "standardize":
            return y * self.y_scale_ + self.y_shift_
        return y

    # -- sklearn API -----------------------------------------------------------------------
    def fit(self, X, y):
        self.check_runtime()
        from finn.qor.models import _torch_impl

        X, y = check_X_y(X, y, accept_sparse=True, y_numeric=True)
        X = self._dense(X)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.y_shift_ = float(np.mean(y))
        self.y_scale_ = float(np.std(y)) or 1.0
        yt = self._transform_target(y)
        if 0 < self.validation_fraction < 1 and len(X) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, yt, test_size=self.validation_fraction, random_state=self.random_state
            )
        else:
            X_train, X_val, y_train, y_val = X, X, yt, yt
        start = time.perf_counter()
        state, info = _torch_impl.train(
            X_train,
            y_train,
            X_val,
            y_val,
            self._hp(),
            int(self.random_state),
            int(self.torch_threads),
            int(self.verbose),
        )
        self.state_dict_ = state
        self.fit_info_ = {**info, "train_time_s": time.perf_counter() - start}
        _CACHE.pop(id(self), None)
        return self

    def predict(self, X):
        check_is_fitted(self, "state_dict_")
        self.check_runtime()
        from finn.qor.models import _torch_impl

        X = self._dense(check_array(X, accept_sparse=True))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        module = _CACHE.get(id(self))
        if module is None:
            module = _torch_impl.load_module(self.n_features_in_, self._hp(), self.state_dict_)
            _CACHE[id(self)] = module
        return self._inverse_target(_torch_impl.predict(module, X, int(self.torch_threads)))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.non_deterministic = False
        return tags

    # -- persistence: no torch objects in the pickle -----------------------------------------
    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)
        _CACHE.pop(id(self), None)
