"""Pre-trained TabPFN tabular foundation model as a scikit-learn regressor for the QoR zoo.

TabPFN (Prior Labs) is a transformer pre-trained on synthetic tabular tasks: there is no
training, ``fit`` only stores the training rows as the in-context examples and ``predict``
runs one forward pass over context plus query rows. :class:`TabPFNRegressor` wraps
``tabpfn.TabPFNRegressor`` so that it drops into the QoR pipeline, GridSearchCV selection
and pickle persistence:

* ``tabpfn`` and ``torch`` are imported lazily; :func:`finn.qor.models.dependency_status`
  reports the class as unavailable without them, so the default fitting job never needs them.
* The weights are never downloaded (that needs a Prior Labs account and license acceptance):
  ``model_path`` must name a staged checkpoint, either as a path or as a bare file name that
  is resolved against ``$TABPFN_MODEL_CACHE_DIR`` (the package's own cache directory
  setting). The default is the TabPFN-3.5 checkpoint file name.
* Categorical columns should reach TabPFN ordinal-encoded with their positions in
  ``categorical_features_indices`` (Prior Labs discourages one-hot encoding and scaling);
  the grid entry in :mod:`finn.qor.models` replaces the pipeline's preprocessors
  accordingly and :meth:`finn.qor.estimator.QoREstimator.build_pipeline` fills in the
  positions.
* The pickle holds the hyperparameters and the training context as numpy arrays, not the
  876 MB of weights: after unpickling, the first ``predict`` re-fits the TabPFN estimator
  from the stored context with ``fit_mode="fit_with_cache"`` (the context is encoded once,
  single-row predictions then cost tens of milliseconds instead of a full forward pass).
"""

import importlib.util
import logging
import numpy as np
import os
import time
import weakref
from pathlib import Path
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from typing import Any, Optional

from finn.qor.models.sklearn_extras import RANDOM_STATE, signed_expm1, signed_log1p

logger = logging.getLogger(__name__)

#: File name of the TabPFN-3.5 checkpoint (as distributed by Prior Labs).
DEFAULT_MODEL_FILE = "tabpfn-v3.5-20260909.safetensors"

#: Environment variable of the tabpfn package that points to the directory with the weights.
MODEL_CACHE_DIR_ENV_VAR = "TABPFN_MODEL_CACHE_DIR"


def tabpfn_available() -> bool:
    """Whether tabpfn and torch are importable (checked via find_spec)."""
    return all(importlib.util.find_spec(m) is not None for m in ("tabpfn", "torch"))


def model_cache_dir() -> Optional[Path]:
    """The weights directory configured via ``$TABPFN_MODEL_CACHE_DIR``, if any."""
    value = os.environ.get(MODEL_CACHE_DIR_ENV_VAR)
    return Path(value).expanduser() if value else None


def resolve_model_path(model_path: str = DEFAULT_MODEL_FILE) -> Optional[Path]:
    """Absolute path of a staged checkpoint, or None if it does not exist.

    Mirrors the package's rule for bare file names (current directory first, then the cache
    directory); explicit paths are taken as they are. Never triggers a download.
    """
    path = Path(model_path).expanduser()
    if path.is_file():
        return path.resolve()
    if path.parent == Path() and (cache := model_cache_dir()) is not None:
        candidate = cache / path
        if candidate.is_file():
            return candidate.resolve()
    return None


def weights_available(model_path: str = DEFAULT_MODEL_FILE) -> bool:
    """Whether the checkpoint can be found without downloading it."""
    return resolve_model_path(model_path) is not None


def resolve_device(device: str = "auto") -> str:
    """``"auto"`` picks CUDA if torch sees a GPU, otherwise the CPU."""
    if device != "auto":
        return str(device)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


#: fitted tabpfn estimators per wrapper instance, held outside ``__dict__`` so that pickles
#: never contain torch objects and the weights are freed together with the wrapper (every
#: fitted instance holds its own copy of the weights on the device)
_MODELS: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


class TabPFNRegressor(BaseEstimator, RegressorMixin):
    """Pre-trained TabPFN regressor; see the module docstring.

    ``n_estimators`` is the size of TabPFN's internal ensemble (forward passes over
    differently preprocessed views of the data; cost grows linearly). ``target_transform``
    (``"none"`` or ``"log1p"``) is applied before the data become the context (TabPFN
    standardizes the target itself, but its output distribution cannot leave the context's
    target range, so the transform changes what it can represent). ``device`` ``"auto"``
    uses CUDA when available; ``torch_threads`` limits the CPU intra-op threads (None keeps
    torch's default). ``categorical_features_indices`` are the column positions TabPFN
    treats as categorical (filled in by the QoR pipeline builder). ``fit_mode`` is passed to
    TabPFN at fit time, ``reload_fit_mode`` when a pickled estimator re-fits itself from
    the stored context.
    """

    #: hint for the model selection: serial (every worker would load the weights again)
    preferred_n_jobs = 1

    #: hint for the latency measurement of the selection: a single prediction re-encodes the
    #: whole context in the default fit mode (seconds on a CPU), so fewer repetitions
    single_predict_repeats = 3

    def __init__(
        self,
        n_estimators=8,
        target_transform="log1p",
        model_path=DEFAULT_MODEL_FILE,
        device="auto",
        torch_threads=None,
        categorical_features_indices=None,
        inference_precision="auto",
        fit_mode="fit_preprocessors",
        reload_fit_mode="fit_with_cache",
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
    ):
        """Store the hyperparameters (sklearn convention: no validation here)."""
        self.n_estimators = n_estimators
        self.target_transform = target_transform
        self.model_path = model_path
        self.device = device
        self.torch_threads = torch_threads
        self.categorical_features_indices = categorical_features_indices
        self.inference_precision = inference_precision
        self.fit_mode = fit_mode
        self.reload_fit_mode = reload_fit_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------------------------
    def check_runtime(self) -> None:
        """Raise ImportError if tabpfn/torch are not installed or the weights are not staged
        (both are needed to predict, as the weights are not part of the pickle)."""
        if not tabpfn_available():
            raise ImportError("TabPFNRegressor requires tabpfn and torch")
        if resolve_model_path(self.model_path) is None:
            raise ImportError(
                f"TabPFN checkpoint {self.model_path!r} not found (stage the weights and point "
                f"${MODEL_CACHE_DIR_ENV_VAR} to their directory; they are not downloaded)"
            )

    @staticmethod
    def _dense(X) -> np.ndarray:
        """Dense float array from a dense or sparse input."""
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        """Apply the configured target transform."""
        if self.target_transform == "log1p":
            return signed_log1p(y)
        if self.target_transform == "none":
            return np.asarray(y, dtype=float)
        raise ValueError(f"unknown target_transform {self.target_transform!r}")

    def _inverse_target(self, y: np.ndarray) -> np.ndarray:
        """Invert the configured target transform."""
        return signed_expm1(y) if self.target_transform == "log1p" else y

    def _set_threads(self) -> None:
        """Apply ``torch_threads`` (CPU intra-op parallelism) if configured."""
        if self.torch_threads:
            import torch

            torch.set_num_threads(max(1, int(self.torch_threads)))

    def _build(self, fit_mode: str):
        """Unfitted ``tabpfn.TabPFNRegressor`` with the configured settings."""
        self.check_runtime()
        from tabpfn import TabPFNRegressor as _TabPFNRegressor

        indices = list(self.categorical_features_indices or [])
        return _TabPFNRegressor(
            n_estimators=int(self.n_estimators),
            categorical_features_indices=indices or None,
            model_path=str(resolve_model_path(self.model_path)),
            device=resolve_device(self.device),
            ignore_pretraining_limits=bool(self.ignore_pretraining_limits),
            inference_precision=self.inference_precision,
            fit_mode=fit_mode,
            random_state=self.random_state,
        )

    def _fit_context(self, fit_mode: str):
        """Fit a TabPFN estimator on the stored context and cache it for this instance."""
        self._set_threads()
        model = self._build(fit_mode)
        model.fit(self.X_context_, self.y_context_)
        _MODELS[self] = model
        return model

    # -- sklearn API -----------------------------------------------------------------------
    def fit(self, X, y):
        """Store ``X``/``y`` as the in-context examples and encode them with TabPFN."""
        # NaN is allowed: TabPFN handles missing values natively (an ordinal encoder may
        # map a missing category to NaN)
        X, y = check_X_y(X, y, accept_sparse=True, y_numeric=True, ensure_all_finite=False)
        self.X_context_ = self._dense(X)
        self.n_features_in_ = self.X_context_.shape[1]
        self.y_context_ = self._transform_target(np.asarray(y, dtype=float))
        start = time.perf_counter()
        model = self._fit_context(self.fit_mode)
        import tabpfn
        import torch

        self.fit_info_ = {
            "tabpfn_version": getattr(tabpfn, "__version__", "?"),
            "torch_version": torch.__version__,
            "device": resolve_device(self.device),
            "model_file": Path(model.model_path).name,
            "n_estimators": int(self.n_estimators),
            "n_context": int(len(self.X_context_)),
            "fit_mode": self.fit_mode,
            "fit_time_s": time.perf_counter() - start,
        }
        return self

    def predict(self, X):
        """Predict ``X`` (mean of TabPFN's predictive distribution, target transform
        inverted); a pickled estimator re-fits itself from the stored context first."""
        check_is_fitted(self, "X_context_")
        X = self._dense(check_array(X, accept_sparse=True, ensure_all_finite=False))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        model = _MODELS.get(self)
        if model is None:
            model = self._fit_context(self.reload_fit_mode)
        else:
            self._set_threads()
        return self._inverse_target(np.asarray(model.predict(X), dtype=float))

    def __sklearn_tags__(self):
        """sklearn estimator tags (regressor, sparse input accepted)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = True
        tags.non_deterministic = False
        return tags

    # -- persistence: no torch objects in the pickle -----------------------------------------
    def __getstate__(self):
        """Pickle the hyperparameters and the context (the TabPFN estimator is not in
        ``__dict__``)."""
        return dict(self.__dict__)

    def __setstate__(self, state):
        """Restore the state; the TabPFN estimator is rebuilt on the first prediction."""
        self.__dict__.update(state)
