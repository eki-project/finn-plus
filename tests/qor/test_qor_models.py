# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Additional QoR regression models: grid discovery, log-target wrappers, the tabular
transformer (needs torch) and symbolic regression (sympy-only inference; PySR fit path is
only exercised if pysr and Julia are usable)."""

import pytest

import numpy as np
import pandas as pd
import pickle
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.estimator_checks import (
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_estimators_pickle,
    check_get_params_invariance,
    check_set_params,
)

from finn.qor.database import POWER_TARGET_COL
from finn.qor.estimator import (
    RESOURCE_TARGETS,
    QoREstimator,
    available_regressor_grid,
    regressor_name,
    select_regressor,
)
from finn.qor.evaluation import plot_pareto_front, symbolic_equation_markdown
from finn.qor.models import (
    LogTargetHGBRegressor,
    LogTargetMLPRegressor,
    SymbolicRegressor,
    TabTransformerRegressor,
    dependency_status,
    signed_expm1,
    signed_log1p,
)
from finn.qor.models.symbolic import rename_expression, sanitize_feature_names, unscale_expression

pytestmark = pytest.mark.qor


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    mw = rng.choice([16, 32, 64, 128], n)
    simd = np.array([rng.choice([d for d in (1, 2, 4, 8) if w % d == 0]) for w in mw])
    pe = rng.choice([1, 2, 4, 8], n)
    wbits = rng.choice([2, 4, 8], n)
    X = pd.DataFrame(
        {"mw": mw, "simd": simd, "pe": pe, "wbits": wbits, "backend": rng.choice(["hls", "rtl"], n)}
    )
    y = 3.0 * np.ceil(mw / simd) * pe + 50 + rng.normal(0, 2, n)
    return X, y


def _numeric(n=200, seed=0):
    X, y = _synthetic(n, seed)
    return X.drop(columns=["backend"]).to_numpy(dtype=float), y


def test_grid_discovery_and_filtering():
    grid = available_regressor_grid()
    names = [regressor_name(cls) for cls, _ in grid]
    assert {"KNeighborsRegressor", "LogTargetHGBRegressor", "LogTargetMLPRegressor"} <= set(names)
    status = dependency_status()
    assert ("TabTransformerRegressor" in names) == (status["TabTransformerRegressor"] is None)
    assert ("SymbolicRegressor" in names) == (status["SymbolicRegressor"] is None)
    only = available_regressor_grid(include=["LogTargetHGBRegressor"])
    assert [regressor_name(cls) for cls, _ in only] == ["LogTargetHGBRegressor"]
    without = available_regressor_grid(quick=True, exclude=["KNeighborsRegressor"])
    assert "KNeighborsRegressor" not in [regressor_name(cls) for cls, _ in without]
    with pytest.raises(ValueError):
        available_regressor_grid(include=["Nope"])


def test_signed_log_round_trip():
    y = np.array([-10.0, 0.0, 3.5, 1e6])
    assert np.allclose(signed_expm1(signed_log1p(y)), y)


@pytest.mark.parametrize("cls", [LogTargetHGBRegressor, LogTargetMLPRegressor])
def test_log_target_regressors(cls):
    X, y = _numeric()
    X = StandardScaler().fit_transform(X)
    model = cls().fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape and np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y)[0, 1] > (0.8 if cls is LogTargetMLPRegressor else 0.9)
    cloned = clone(model)
    assert type(cloned.regressor) is type(model.regressor)
    assert pickle.loads(pickle.dumps(model)).predict(X[:3]).shape == (3,)


def test_log_target_inside_qor_estimator(tmp_path):
    X, y = _synthetic()
    df = X.rename(columns={"mw": "params.mw", "backend": "params.backend"})
    from finn.qor.features import SPECS

    # a dedicated tiny spec keeps the test independent of the MVAU feature list
    spec = SPECS["mvau"].__class__(
        name="mvau",
        op_types=("MVAU_hls",),
        feature_cols=["params.mw", "simd", "pe", "wbits", "params.backend"],
    )
    df[RESOURCE_TARGETS["LUT"]] = y
    est = QoREstimator("mvau", RESOURCE_TARGETS["LUT"], spec=spec).fit(df, LogTargetHGBRegressor())
    assert isinstance(est.predict(df.iloc[[0]]), int)
    est.save(str(tmp_path))
    loaded = QoREstimator.load(str(tmp_path), "mvau", RESOURCE_TARGETS["LUT"])
    assert loaded.predict(df.iloc[[0]]) == est.predict(df.iloc[[0]])
    assert loaded.metadata["regressor"] == "LogTargetHGBRegressor"


# ---------------------------------------------------------------- tabular transformer


@pytest.fixture
def torch_available():
    pytest.importorskip("torch")


def test_transformer_fit_predict_determinism_pickle(torch_available):
    X, y = _numeric()
    X = StandardScaler().fit_transform(X)
    kwargs = dict(d_model=8, n_layers=1, n_heads=2, max_epochs=8, patience=3, batch_size=64)
    a = TabTransformerRegressor(**kwargs).fit(X, y)
    b = TabTransformerRegressor(**kwargs).fit(X, y)
    pa, pb = a.predict(X), b.predict(X)
    assert pa.shape == y.shape and np.all(np.isfinite(pa))
    assert np.allclose(pa, pb), "same seed must give the same model"
    assert a.fit_info_["epochs_trained"] <= 8 and a.fit_info_["n_params"] > 0
    # the pickle holds numpy weights only and rebuilds the module lazily
    restored = pickle.loads(pickle.dumps(a))
    assert all(isinstance(v, np.ndarray) for v in restored.state_dict_.values())
    assert np.allclose(restored.predict(X[:5]), pa[:5])
    # sparse input and target transforms
    from scipy import sparse

    assert np.allclose(a.predict(sparse.csr_matrix(X[:3])), pa[:3])
    for transform in ("none", "standardize"):
        m = TabTransformerRegressor(target_transform=transform, **kwargs).fit(X, y)
        assert np.all(np.isfinite(m.predict(X[:3])))
    with pytest.raises(ValueError):
        a.predict(X[:, :2])


def test_transformer_sklearn_conventions(torch_available):
    est = TabTransformerRegressor(d_model=8, n_layers=1, n_heads=2, max_epochs=3, patience=2)
    name = "TabTransformerRegressor"
    check_get_params_invariance(name, est)
    check_set_params(name, est)
    check_dict_unchanged(name, est)
    check_dont_overwrite_parameters(name, est)
    check_estimators_pickle(name, est)


def test_transformer_in_pipeline_and_selection(torch_available):
    X, y = _synthetic(n=120)
    df = X.rename(columns={"mw": "params.mw", "backend": "params.backend"})
    from finn.qor.features import OperatorFeatureSpec

    spec = OperatorFeatureSpec(
        "mvau", ("MVAU_hls",), ["params.mw", "simd", "pe", "wbits", "params.backend"]
    )
    df[POWER_TARGET_COL] = y
    est = QoREstimator("mvau", POWER_TARGET_COL, spec=spec)
    grid = [
        (
            TabTransformerRegressor,
            {
                "regressor__d_model": [8],
                "regressor__n_heads": [2],
                "regressor__n_layers": [1],
                "regressor__max_epochs": [5],
                "regressor__patience": [2],
            },
        )
    ]
    result = select_regressor(est, df, cv=2, n_jobs=1, grid=grid)
    assert result.best_name == "TabTransformerRegressor"
    assert {"fit_time_s", "single_predict_ms", "search_time_s"} <= set(result.scores.columns)
    assert "TabTransformerRegressor" in result.cv_results
    assert est.metadata["regressor_info"]["n_params"] > 0 and "torch_version" in est.metadata
    assert isinstance(est.predict(df.iloc[[0]]), float)


# ---------------------------------------------------------------- symbolic regression


def _fitted_symbolic(expr_srepr, n_features):
    import sympy

    model = SymbolicRegressor()
    model.n_features_in_ = n_features
    model.expr_srepr_ = expr_srepr
    model.equation_ = "manual"
    model.complexity_ = 5
    model.loss_ = 0.1
    model.pareto_front_ = [
        {"complexity": 1, "loss": 10.0, "score": 0.0, "equation": "x0"},
        {"complexity": 5, "loss": 0.1, "score": 1.0, "equation": "manual"},
    ]
    return model


def test_symbolic_predict_without_julia(tmp_path):
    sympy = pytest.importorskip("sympy")
    x0, x1, x2 = sympy.symbols("x0 x1 x2")
    expr = 3 * sympy.ceiling(x0 / x1) * x2 + 50
    model = _fitted_symbolic(sympy.srepr(expr), 3)
    X = np.array([[64, 8, 2], [64, 3, 1], [16, 16, 4]], dtype=float)
    expected = 3 * np.ceil(X[:, 0] / X[:, 1]) * X[:, 2] + 50
    assert np.allclose(model.predict(X), expected)
    assert model.sympy() == expr
    assert "ceil" in model.latex().lower() or "lceil" in model.latex()
    # constant expression broadcasts, pickle round trip drops the compiled function
    const = _fitted_symbolic(sympy.srepr(sympy.Integer(7)), 3)
    assert np.allclose(const.predict(X), 7)
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(restored.predict(X), expected)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])

    # feature naming helpers
    names = sanitize_feature_names(
        ["num__params.mw", "num__dut_info.simd", "cat__params.backend_hls", "num__params.mw"]
    )
    assert names == ["mw", "simd", "backend_hls", "mw_2"]
    renamed = rename_expression(expr, ["mw", "simd", "pe"])
    assert str(renamed) == "3*pe*ceiling(mw/simd) + 50"
    unscaled = unscale_expression(sympy.Symbol("a") * 2, ["a"], [1.0], [0.5])
    assert sympy.simplify(unscaled - (4 * sympy.Symbol("a") - 4)) == 0

    equation = {
        "variables": ["mw", "simd", "pe"],
        "sympy": str(renamed),
        "latex": sympy.latex(renamed),
        "pysr": "manual",
        "complexity": 5,
        "loss": 0.1,
        "pareto_front": model.pareto_front_,
    }
    md = symbolic_equation_markdown(equation)
    assert "3*pe*ceiling(mw/simd) + 50" in md and "Pareto front" in md
    plot_pareto_front(equation, str(tmp_path / "pareto.png"))
    assert (tmp_path / "pareto.png").is_file()


def test_symbolic_equation_metadata_via_estimator(tmp_path):
    sympy = pytest.importorskip("sympy")
    from finn.qor.features import OperatorFeatureSpec

    X, y = _synthetic(n=50)
    df = X.rename(columns={"mw": "params.mw", "backend": "params.backend"})
    df[RESOURCE_TARGETS["LUT"]] = y
    spec = OperatorFeatureSpec(
        "mvau", ("MVAU_hls",), ["params.mw", "simd", "pe", "wbits", "params.backend"]
    )
    est = QoREstimator("mvau", RESOURCE_TARGETS["LUT"], spec=spec)
    # numeric features raw (FunctionTransformer), one-hot backend as x4/x5
    pipeline = est.build_pipeline(SymbolicRegressor(), df)
    pipeline.set_params(
        preprocessor__num=FunctionTransformer(validate=True, feature_names_out="one-to-one")
    )
    x = sympy.symbols("x0:6")
    model = _fitted_symbolic(sympy.srepr(3 * sympy.ceiling(x[0] / x[1]) * x[2] + 50), 6)
    pipeline.steps[-1] = ("regressor", model)
    pipeline.named_steps["preprocessor"].fit(df[spec.feature_cols])
    est.pipeline = pipeline
    est.metadata.update(est._fit_metadata(df))
    eq = est.metadata["equation"]
    assert eq["variables"][:4] == ["mw", "simd", "pe", "wbits"]
    assert eq["sympy"] == "3*pe*ceiling(mw/simd) + 50"
    assert eq["complexity"] == 5 and len(eq["pareto_front"]) == 2
    pred = est.predict(df.iloc[[0]])
    row = df.iloc[0]
    assert pred == int(round(3 * np.ceil(row["params.mw"] / row["simd"]) * row["pe"] + 50))
    est.save(str(tmp_path))
    loaded = QoREstimator.load(str(tmp_path), "mvau", RESOURCE_TARGETS["LUT"])
    assert loaded.predict(df.iloc[[0]]) == pred


def test_symbolic_fit_with_pysr():
    try:
        import pysr
    except Exception as e:  # noqa: BLE001 - juliacall raises non-ImportError on a broken Julia
        pytest.skip(f"pysr/Julia not usable: {e}")
    X, y = _numeric(n=80)
    model = SymbolicRegressor(niterations=3, maxsize=10, timeout_in_seconds=120, populations=4)
    model.fit(X, y)
    assert model.fit_info_["lambdify_max_abs_diff"] < 1e-3
    assert np.all(np.isfinite(model.predict(X)))


def test_estimator_load_skips_missing_runtime(tmp_path, monkeypatch):
    from finn.qor.features import OperatorFeatureSpec

    X, y = _synthetic(n=50)
    df = X.rename(columns={"mw": "params.mw", "backend": "params.backend"})
    df[RESOURCE_TARGETS["LUT"]] = y
    spec = OperatorFeatureSpec(
        "mvau", ("MVAU_hls",), ["params.mw", "simd", "pe", "wbits", "params.backend"]
    )
    est = QoREstimator("mvau", RESOURCE_TARGETS["LUT"], spec=spec).fit(df, LogTargetHGBRegressor())
    est.save(str(tmp_path))

    class Broken:
        def check_runtime(self):
            raise ImportError("no torch")

    est.pipeline = Pipeline([("regressor", Broken())])
    with pytest.raises(ImportError):
        est.check_runtime()
