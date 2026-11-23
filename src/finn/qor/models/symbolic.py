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
from collections.abc import Callable, Iterable
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from types import ModuleType
from typing import Any, Self

from finn.qor.models.sklearn_extras import RANDOM_STATE


def pysr_available() -> bool:
    """Whether pysr is importable (checked via find_spec, never importing it)."""
    return importlib.util.find_spec("pysr") is not None


def sympy_available() -> bool:
    """Whether sympy is importable."""
    return importlib.util.find_spec("sympy") is not None


#: compiled (lambdified) expressions per estimator instance, kept outside __dict__
_CACHE: dict[int, Any] = {}


#: Unary operators that SymbolicRegression.jl does not provide itself: name -> (Julia
#: definition handed to PySR, sympy construction for exporting the fitted expression).
CUSTOM_UNARY_OPERATORS: dict[str, tuple[str, Any]] = {
    "log2_abs": ("log2_abs(x) = log2(abs(x))", lambda x: _sympy().log(_sympy().Abs(x), 2)),
}


def _sympy() -> ModuleType:
    """Lazy import of sympy."""
    import sympy

    return sympy


def translate_operators(names: Iterable[str]) -> tuple[list[str], dict[str, Any]]:
    """Map operator names to what PySR expects: built-in names unchanged, custom ones as a
    Julia definition plus their ``extra_sympy_mappings`` entry."""
    operators, mappings = [], {}
    for name in names:
        if name in CUSTOM_UNARY_OPERATORS:
            definition, to_sympy = CUSTOM_UNARY_OPERATORS[name]
            operators.append(definition)
            mappings[name] = to_sympy
        else:
            operators.append(name)
    return operators, mappings


_NUMPY_STR = re.compile(r"(?:np|numpy)\.str_\((['\"][^'\"]*['\"])\)")


def plain_srepr(text: str) -> str:
    """``srepr`` text with numpy string literals turned into plain ones. PySR names its
    features with a numpy string array, so the symbols' ``srepr`` contains
    ``Symbol(np.str_('x7'))``, which ``sympify`` cannot parse back (and sympy's symbol cache
    keeps returning the numpy-named instance for ``Symbol('x7')``)."""
    return _NUMPY_STR.sub(r"\1", text)


def sanitize_feature_names(names: Iterable[Any]) -> list[str]:
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


def rename_expression(expr: Any, names: list[str]) -> Any:
    """Replace the positional symbols ``x0..xN`` of a PySR expression by ``names``."""
    import sympy

    mapping = {sympy.Symbol(f"x{i}"): sympy.Symbol(n) for i, n in enumerate(names)}
    return expr.xreplace(mapping)


def unscale_expression(
    expr: Any, names: list[str], means: Iterable[float], scales: Iterable[float]
) -> Any:
    """Substitute standardized variables ``(name - mean) / scale`` so that an expression fitted
    on scaled inputs is stated in raw feature units."""
    import sympy

    mapping = {}
    for name, mean, scale in zip(names, means, scales, strict=True):
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
        niterations: int = 60,
        binary_operators: tuple[str, ...] = ("+", "-", "*", "/"),
        unary_operators: tuple[str, ...] = ("square", "ceil"),
        maxsize: int = 30,
        populations: int = 15,
        population_size: int = 33,
        constraints: dict[str, Any] | None = None,
        nested_constraints: dict[str, Any] | None = None,
        complexity_of_operators: dict[str, int] | None = None,
        model_selection: str = "best",
        elementwise_loss: str | None = None,
        relative_weights: bool = True,
        timeout_in_seconds: float | None = None,
        deterministic: bool = True,
        parallelism: str = "serial",
        procs: int | None = None,
        precision: int = 32,
        batching: bool | str = "auto",
        output_directory: str | None = None,
        random_state: int = RANDOM_STATE,
        verbosity: int = 0,
    ) -> None:
        """Store the hyperparameters (sklearn convention: no validation here)."""
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
    def _dense(x: Any) -> np.ndarray:
        """Dense float array from a dense or sparse input."""
        if sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x, dtype=np.float64)

    def sympy(self) -> Any:
        """Return the selected expression as a sympy object (symbols ``x0..xN``)."""
        import sympy

        check_is_fitted(self, "expr_srepr_")
        return sympy.sympify(self.expr_srepr_)

    def latex(self, names: list[str] | None = None) -> str:
        """LaTeX form of the expression (optionally with renamed variables)."""
        import sympy

        expr = self.sympy()
        if names is not None:
            expr = rename_expression(expr, names)
        return sympy.latex(expr)

    def _function(self) -> Callable[..., Any]:
        """Return the cached numpy function of the fitted expression (sympy lambdify)."""
        import sympy

        fn = _CACHE.get(id(self))
        if fn is None:
            symbols = [sympy.Symbol(f"x{i}") for i in range(self.n_features_in_)]
            fn = sympy.lambdify(symbols, self.sympy(), modules=["numpy"])
            _CACHE[id(self)] = fn
        return fn

    # -- sklearn API -----------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> Self:  # noqa: N803
        """Fit the model on ``X``/``y`` (see the class docstring)."""
        if not pysr_available():
            raise ImportError("SymbolicRegressor.fit requires pysr (and Julia)")
        import sympy
        from pysr import PySRRegressor

        x, y = check_X_y(X, y, accept_sparse=True, y_numeric=True)
        x = self._dense(x)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = x.shape[1]
        weights = None
        if self.relative_weights:
            weights = 1.0 / np.maximum(np.abs(y), 1e-3) ** 2
        out_dir = self.output_directory or tempfile.mkdtemp(prefix="pysr_")
        binary_operators, binary_mappings = translate_operators(self.binary_operators)
        unary_operators, unary_mappings = translate_operators(self.unary_operators)
        kwargs = {
            "niterations": self.niterations,
            "binary_operators": binary_operators,
            "unary_operators": unary_operators,
            "extra_sympy_mappings": {**binary_mappings, **unary_mappings} or None,
            "maxsize": self.maxsize,
            "populations": self.populations,
            "population_size": self.population_size,
            "constraints": self.constraints,
            "nested_constraints": self.nested_constraints,
            "complexity_of_operators": self.complexity_of_operators,
            "model_selection": self.model_selection,
            "elementwise_loss": self.elementwise_loss,
            "timeout_in_seconds": self.timeout_in_seconds,
            "deterministic": self.deterministic,
            "parallelism": self.parallelism,
            "procs": self.procs,
            "precision": self.precision,
            "batching": self.batching,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "progress": False,
            # PySR creates its run directory below tempdir and asserts (2.6) that no
            # output_directory is given together with temp_equation_file
            "temp_equation_file": True,
            "delete_tempfiles": True,
            "tempdir": out_dir,
            "run_id": f"qor_{uuid.uuid4().hex[:8]}",
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        start = time.perf_counter()
        model = PySRRegressor(**kwargs)
        model.fit(x, y, weights=weights)
        best = model.get_best()
        expr = model.sympy()
        self.expr_srepr_ = plain_srepr(sympy.srepr(expr))
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
        own = self.predict(x)
        ref = np.asarray(model.predict(x), dtype=float)
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

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """Predict ``X`` with the fitted model."""
        check_is_fitted(self, "expr_srepr_")
        self.check_runtime()
        x = self._dense(check_array(X, accept_sparse=True))
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x.shape[1]}")
        fn = self._function()
        columns = [x[:, i] for i in range(x.shape[1])]
        out = np.asarray(fn(*columns), dtype=float)
        out = np.broadcast_to(out, (x.shape[0],)).astype(float)
        return np.nan_to_num(out, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)

    def __sklearn_tags__(self) -> Tags:
        """Sklearn estimator tags (regressor, sparse input accepted)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def __getstate__(self) -> dict[str, Any]:
        """Pickle without the cached runtime objects."""
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state; runtime objects are rebuilt lazily."""
        self.__dict__.update(state)
        _CACHE.pop(id(self), None)
