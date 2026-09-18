"""Symbolic regression (PySR) as a scikit-learn estimator with Julia-free inference.

Fitting runs PySR (Julia backend) and keeps only the selected expression as a sympy
``srepr`` string plus the Pareto front of the run. Prediction evaluates the expression with
``sympy.lambdify`` (numpy), so a fitted model needs neither PySR nor Julia, and the pickle
contains no Julia state. One-hot encoded categorical features enter as 0/1 variables; feed
raw (unscaled) numeric features so that the formula is in feature units.

Only sympy is required at inference; ``pysr`` (2.x API) at fit time.
"""

import importlib.util
import numpy as np
import re
import shutil
import tempfile
import time
import uuid
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from typing import Any, Optional

from finn.qor.models.sklearn_extras import RANDOM_STATE


def pysr_available() -> bool:
    return importlib.util.find_spec("pysr") is not None


def sympy_available() -> bool:
    return importlib.util.find_spec("sympy") is not None


#: compiled (lambdified) expressions per estimator instance, kept outside __dict__
_CACHE: dict[int, Any] = {}


def sanitize_feature_names(names) -> list[str]:
    """Turn pipeline feature names (``num__params.mw``, ``cat__params.backend_hls``) into
    unique python identifiers (``mw``, ``backend_hls``)."""
    out: list[str] = []
    for name in names:
        name = re.sub(r"^(num|cat|remainder)__", "", str(name))
        name = re.sub(r"^(params|dut_info)\.", "", name)
        name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        if not name or name[0].isdigit():
            name = "f_" + name
        base, i = name, 1
        while name in out:
            i += 1
            name = f"{base}_{i}"
        out.append(name)
    return out


def rename_expression(expr, names: list[str]):
    """Replace the positional symbols ``x0..xN`` of a PySR expression by ``names``."""
    import sympy

    mapping = {sympy.Symbol(f"x{i}"): sympy.Symbol(n) for i, n in enumerate(names)}
    return expr.xreplace(mapping)


def unscale_expression(expr, names: list[str], means, scales):
    """Substitute standardized variables ``(name - mean) / scale`` so that an expression fitted
    on scaled inputs is stated in raw feature units."""
    import sympy

    mapping = {}
    for name, mean, scale in zip(names, means, scales):
        sym = sympy.Symbol(name)
        mapping[sym] = (sym - float(mean)) / float(scale)
    return expr.xreplace(mapping)


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """PySR symbolic regression; see the module docstring.

    ``relative_weights`` weights samples by ``1 / max(|y|, eps)^2`` so that the fit
    approximates minimizing the relative error (the LUT models' selection criterion) while the
    formula stays in target units. ``deterministic=True`` forces serial execution.
    """

    #: never fork the Julia runtime into loky workers
    preferred_n_jobs = 1

    def __init__(
        self,
        niterations=60,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=("square", "ceil"),
        maxsize=30,
        populations=15,
        population_size=33,
        constraints=None,
        nested_constraints=None,
        complexity_of_operators=None,
        model_selection="best",
        elementwise_loss=None,
        relative_weights=True,
        timeout_in_seconds=None,
        deterministic=True,
        parallelism="serial",
        procs=None,
        precision=32,
        batching="auto",
        output_directory=None,
        random_state=RANDOM_STATE,
        verbosity=0,
    ):
        self.niterations = niterations
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.maxsize = maxsize
        self.populations = populations
        self.population_size = population_size
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.complexity_of_operators = complexity_of_operators
        self.model_selection = model_selection
        self.elementwise_loss = elementwise_loss
        self.relative_weights = relative_weights
        self.timeout_in_seconds = timeout_in_seconds
        self.deterministic = deterministic
        self.parallelism = parallelism
        self.procs = procs
        self.precision = precision
        self.batching = batching
        self.output_directory = output_directory
        self.random_state = random_state
        self.verbosity = verbosity

    # -- helpers ---------------------------------------------------------------------------
    def check_runtime(self) -> None:
        """Raise ImportError if sympy (needed for prediction) is not installed."""
        if not sympy_available():
            raise ImportError("SymbolicRegressor requires sympy for prediction")

    @staticmethod
    def _dense(X) -> np.ndarray:
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def sympy(self):
        """The selected expression as a sympy object (symbols ``x0..xN``)."""
        import sympy

        check_is_fitted(self, "expr_srepr_")
        return sympy.sympify(self.expr_srepr_)

    def latex(self, names: Optional[list[str]] = None) -> str:
        """LaTeX form of the expression (optionally with renamed variables)."""
        import sympy

        expr = self.sympy()
        if names is not None:
            expr = rename_expression(expr, names)
        return sympy.latex(expr)

    def _function(self):
        import sympy

        fn = _CACHE.get(id(self))
        if fn is None:
            symbols = [sympy.Symbol(f"x{i}") for i in range(self.n_features_in_)]
            fn = sympy.lambdify(symbols, self.sympy(), modules=["numpy"])
            _CACHE[id(self)] = fn
        return fn

    # -- sklearn API -----------------------------------------------------------------------
    def fit(self, X, y):
        if not pysr_available():
            raise ImportError("SymbolicRegressor.fit requires pysr (and Julia)")
        import sympy
        from pysr import PySRRegressor

        X, y = check_X_y(X, y, accept_sparse=True, y_numeric=True)
        X = self._dense(X)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        weights = None
        if self.relative_weights:
            weights = 1.0 / np.maximum(np.abs(y), 1e-3) ** 2
        out_dir = self.output_directory or tempfile.mkdtemp(prefix="pysr_")
        kwargs = dict(
            niterations=self.niterations,
            binary_operators=list(self.binary_operators),
            unary_operators=list(self.unary_operators),
            maxsize=self.maxsize,
            populations=self.populations,
            population_size=self.population_size,
            constraints=self.constraints,
            nested_constraints=self.nested_constraints,
            complexity_of_operators=self.complexity_of_operators,
            model_selection=self.model_selection,
            elementwise_loss=self.elementwise_loss,
            timeout_in_seconds=self.timeout_in_seconds,
            deterministic=self.deterministic,
            parallelism=self.parallelism if self.deterministic else self.parallelism,
            procs=self.procs,
            precision=self.precision,
            batching=self.batching,
            random_state=self.random_state,
            verbosity=self.verbosity,
            progress=False,
            temp_equation_file=True,
            delete_tempfiles=True,
            output_directory=out_dir,
            run_id=f"qor_{uuid.uuid4().hex[:8]}",
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        start = time.perf_counter()
        model = PySRRegressor(**kwargs)
        model.fit(X, y, weights=weights)
        best = model.get_best()
        expr = model.sympy()
        self.expr_srepr_ = sympy.srepr(expr)
        self.equation_ = str(best["equation"])
        self.complexity_ = int(best["complexity"])
        self.loss_ = float(best["loss"])
        front = model.equations_[["complexity", "loss", "score", "equation"]]
        self.pareto_front_ = [
            {
                "complexity": int(r.complexity),
                "loss": float(r.loss),
                "score": float(r.score),
                "equation": str(r.equation),
            }
            for r in front.itertuples()
        ]
        _CACHE.pop(id(self), None)
        own = self.predict(X)
        ref = np.asarray(model.predict(X), dtype=float)
        self.fit_info_ = {
            "pysr_version": getattr(__import__("pysr"), "__version__", "?"),
            "equation": self.equation_,
            "sympy": str(expr),
            "complexity": self.complexity_,
            "loss": self.loss_,
            "n_equations": len(self.pareto_front_),
            "fit_time_s": time.perf_counter() - start,
            "lambdify_max_abs_diff": float(np.nanmax(np.abs(own - ref))) if len(own) else 0.0,
        }
        if self.output_directory is None:
            shutil.rmtree(out_dir, ignore_errors=True)
        return self

    def predict(self, X):
        check_is_fitted(self, "expr_srepr_")
        self.check_runtime()
        X = self._dense(check_array(X, accept_sparse=True))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        fn = self._function()
        columns = [X[:, i] for i in range(X.shape[1])]
        out = np.asarray(fn(*columns), dtype=float)
        out = np.broadcast_to(out, (X.shape[0],)).astype(float)
        return np.nan_to_num(out, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)
        _CACHE.pop(id(self), None)
