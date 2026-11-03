# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""TabPFN wrapper of the QoR model zoo: dependency discovery and grid entry (always), and a
tiny fit/predict/pickle round trip that only runs with tabpfn installed and staged weights."""

import pytest

import numpy as np
import pandas as pd
import pickle
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.utils.estimator_checks import (
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_get_params_invariance,
    check_set_params,
)

from finn.qor.database import POWER_TARGET_COL
from finn.qor.estimator import (
    RESOURCE_TARGETS,
    QoREstimator,
    _single_predict_ms,
    available_regressor_grid,
    regressor_name,
    select_regressor,
)
from finn.qor.features import OperatorFeatureSpec
from finn.qor.models import OPTIONAL_DEPENDENCIES, TabPFNRegressor, dependency_status
from finn.qor.models.tabpfn import (
    DEFAULT_MODEL_FILE,
    MODEL_CACHE_DIR_ENV_VAR,
    resolve_device,
    resolve_model_path,
    tabpfn_available,
    weights_available,
)

pytestmark = pytest.mark.qor

#: Checkpoints the smoke test can run with: (file name, kwargs). The v2 regressor checkpoint
#: (44 MB, Prior Labs License) is tried after the 3.5 default, so the mechanics can also be
#: checked without the research-only 3.5 weights.
SMOKE_CHECKPOINTS = (DEFAULT_MODEL_FILE, "tabpfn-v2-regressor.ckpt")


def _synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    mw = rng.choice([16, 32, 64, 128], n)
    simd = np.array([rng.choice([d for d in (1, 2, 4, 8) if w % d == 0]) for w in mw])
    pe = rng.choice([1, 2, 4, 8], n)
    X = pd.DataFrame(
        {
            "params.mw": mw,
            "simd": simd,
            "pe": pe,
            "params.backend": rng.choice(["hls", "rtl"], n),
            "params.ram_style": rng.choice(["auto", "block", None], n),
        }
    )
    # object dtype like the database loader's strings (pandas 3 would infer "str")
    X = X.astype({"params.backend": object, "params.ram_style": object})
    y = 3.0 * np.ceil(mw / simd) * pe + 50 + rng.normal(0, 2, n)
    return X, y


def _spec():
    return OperatorFeatureSpec(
        "mvau",
        ("MVAU_hls",),
        ["params.mw", "simd", "pe", "params.backend", "params.ram_style"],
    )


# ---------------------------------------------------------------- no tabpfn needed


def test_dependency_status_and_grid_entry(monkeypatch):
    assert OPTIONAL_DEPENDENCIES["TabPFNRegressor"] == ("tabpfn", "torch")
    names = [regressor_name(cls) for cls, _ in available_regressor_grid()]
    assert ("TabPFNRegressor" in names) == (dependency_status()["TabPFNRegressor"] is None)
    # without tabpfn the class is reported as missing and cannot be selected explicitly
    monkeypatch.setitem(OPTIONAL_DEPENDENCIES, "TabPFNRegressor", ("no_such_module_tabpfn",))
    assert dependency_status()["TabPFNRegressor"] == "missing: no_such_module_tabpfn"
    assert "TabPFNRegressor" not in [regressor_name(cls) for cls, _ in available_regressor_grid()]
    with pytest.raises(ValueError, match="not available"):
        available_regressor_grid(include=["TabPFNRegressor"])
    # the grid entry itself (independent of the installed packages)
    monkeypatch.setitem(OPTIONAL_DEPENDENCIES, "TabPFNRegressor", ("sklearn",))
    entries = [e for e in available_regressor_grid() if e[0] is TabPFNRegressor]
    assert len(entries) == 1
    _, grid = entries[0]
    assert grid["regressor__n_estimators"] == [1, 8]
    assert grid["regressor__target_transform"] == ["none", "log1p"]
    assert isinstance(grid["preprocessor__cat"][0], OrdinalEncoder)
    assert isinstance(grid["preprocessor__num"][0], FunctionTransformer)
    quick = [e for e in available_regressor_grid(quick=True) if e[0] is TabPFNRegressor][0][1]
    assert quick["regressor__n_estimators"] == [1]
    # every grid key is a valid pipeline parameter
    pipeline = QoREstimator("mvau", RESOURCE_TARGETS["LUT"], spec=_spec()).build_pipeline(
        TabPFNRegressor(), _synthetic()[0]
    )
    params = pipeline.get_params()
    assert all(key in params for key in grid)


def test_pipeline_builder_passes_categorical_positions():
    X, _ = _synthetic()
    est = QoREstimator("mvau", RESOURCE_TARGETS["LUT"], spec=_spec())
    pipeline = est.build_pipeline(TabPFNRegressor(), X)
    # three numeric columns first, then the two categorical ones
    assert pipeline.named_steps["regressor"].categorical_features_indices == [3, 4]
    # the positions survive sklearn's clone (they are constructor parameters)
    assert clone(pipeline).named_steps["regressor"].categorical_features_indices == [3, 4]
    assert TabPFNRegressor.preferred_n_jobs == 1 and TabPFNRegressor.single_predict_repeats == 3


def test_sklearn_parameter_conventions_without_fit():
    est = TabPFNRegressor(n_estimators=1, device="cpu")
    check_get_params_invariance("TabPFNRegressor", est)
    check_set_params("TabPFNRegressor", est)
    unfitted = pickle.loads(pickle.dumps(est))
    assert unfitted.get_params() == est.get_params()
    with pytest.raises(Exception):  # noqa: B017 - sklearn's NotFittedError
        est.predict(np.zeros((1, 3)))


def test_model_path_resolution(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / DEFAULT_MODEL_FILE).write_bytes(b"not a checkpoint")
    monkeypatch.delenv(MODEL_CACHE_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_model_path() is None and not weights_available()
    monkeypatch.setenv(MODEL_CACHE_DIR_ENV_VAR, str(weights))
    assert resolve_model_path() == (weights / DEFAULT_MODEL_FILE).resolve()
    assert resolve_model_path(str(weights / DEFAULT_MODEL_FILE)) is not None
    assert resolve_model_path("other.safetensors") is None
    # the bare file name is checked in the working directory first (package rule)
    (tmp_path / "other.safetensors").write_bytes(b"x")
    assert resolve_model_path("other.safetensors") == (tmp_path / "other.safetensors").resolve()
    if not tabpfn_available():
        with pytest.raises(ImportError, match="requires tabpfn"):
            TabPFNRegressor().check_runtime()
    else:
        with pytest.raises(ImportError, match="not found"):
            TabPFNRegressor(model_path="missing.ckpt").check_runtime()


def test_single_predict_repeats_cap():
    class Counting:
        single_predict_repeats = 2
        calls = 0

        def predict(self, X):
            Counting.calls += 1
            return np.zeros(len(X))

    class FakePipeline(list):
        def predict(self, X):
            return self[-1].predict(X)

    X = pd.DataFrame({"a": [1.0, 2.0]})
    _single_predict_ms(FakePipeline([Counting()]), X)
    assert Counting.calls == 2


# ---------------------------------------------------------------- real tabpfn + weights


def _staged_checkpoint():
    """First checkpoint of SMOKE_CHECKPOINTS that is importable and staged, else skip."""
    if not tabpfn_available():
        pytest.skip("tabpfn/torch not installed")
    for name in SMOKE_CHECKPOINTS:
        if weights_available(name):
            return name
    pytest.skip(f"no TabPFN checkpoint staged (set ${MODEL_CACHE_DIR_ENV_VAR})")


def test_tabpfn_fit_predict_pickle_cpu():
    model_path = _staged_checkpoint()
    X, y = _synthetic(n=60)
    est = QoREstimator("mvau", POWER_TARGET_COL, spec=_spec())
    df = X.copy()
    df[POWER_TARGET_COL] = y
    pipeline = est.build_pipeline(
        TabPFNRegressor(n_estimators=1, device="cpu", model_path=model_path, torch_threads=2), df
    )
    pipeline.set_params(
        preprocessor__num=FunctionTransformer(validate=True, feature_names_out="one-to-one"),
        preprocessor__cat=OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
        ),
    )
    pipeline.fit(X, y)
    reg = pipeline.named_steps["regressor"]
    assert reg.fit_info_["device"] == "cpu" and reg.fit_info_["n_context"] == 60
    assert reg.fit_info_["model_file"] == model_path.rsplit("/", 1)[-1]
    pred = pipeline.predict(X)
    assert pred.shape == y.shape and np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y)[0, 1] > 0.8
    # sklearn conventions on the fitted estimator (predict leaves __dict__ alone)
    check_dict_unchanged("TabPFNRegressor", clone(reg).set_params(torch_threads=2))
    check_dont_overwrite_parameters("TabPFNRegressor", clone(reg).set_params(torch_threads=2))
    # the pickle holds the context, not the weights, and re-fits with the cache on demand
    blob = pickle.dumps(pipeline)
    assert len(blob) < 1_000_000
    restored = pickle.loads(blob)
    restored_reg = restored.named_steps["regressor"]
    assert np.allclose(restored_reg.X_context_, reg.X_context_)
    again = restored.predict(X.iloc[:5])
    assert np.allclose(again, pred[:5], rtol=0.05, atol=1.0)
    with pytest.raises(ValueError):
        reg.predict(reg.X_context_[:, :2])
    # unseen categories are encoded as -1 and still predicted
    unseen = X.iloc[[0]].assign(**{"params.backend": "versal"})
    assert np.isfinite(restored.predict(unseen)[0])


def test_tabpfn_in_model_selection():
    model_path = _staged_checkpoint()
    X, y = _synthetic(n=40)
    df = X.copy()
    df[RESOURCE_TARGETS["LUT"]] = y
    est = QoREstimator("mvau", RESOURCE_TARGETS["LUT"], spec=_spec())
    grid = [
        (
            TabPFNRegressor,
            {
                "preprocessor__num": [
                    FunctionTransformer(validate=True, feature_names_out="one-to-one")
                ],
                "preprocessor__cat": [
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-2,
                    )
                ],
                "regressor__n_estimators": [1],
                "regressor__device": ["cpu"],
                "regressor__model_path": [model_path],
                "regressor__target_transform": ["none", "log1p"],
            },
        )
    ]
    result = select_regressor(est, df, cv=2, n_jobs=1, grid=grid)
    assert result.best_name == "TabPFNRegressor"
    assert result.scores.loc["TabPFNRegressor", "n_param_combinations"] == 2
    assert np.isfinite(result.scores.loc["TabPFNRegressor", "single_predict_ms"])
    info = est.metadata["regressor_info"]
    assert info["device"] == "cpu" and "tabpfn_version" in info
    assert isinstance(est.predict(df.iloc[[0]]), int)
    assert resolve_device("cpu") == "cpu"
