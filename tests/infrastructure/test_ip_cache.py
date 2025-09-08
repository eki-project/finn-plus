"""Test that the IP cache is working correctly. (No false positives, no collisions, speed, etc.)."""
import pytest

import numpy as np
import os
import time
from copy import deepcopy
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
from typing import cast

from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.ip_cache import CachedIPGen, IPCache
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import alveo_part_map
from finn.util.deps import get_cache_path
from tests.fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper


def mvau_hls_create_model(fpgapart: str) -> tuple[ModelWrapper, np.ndarray, np.ndarray]:
    """Create and sanity check a model for testing MVAU_hls caching.

    Returns:
        ModelWrapper, NDArray, NDArray: Model, weights, thresholds.
    """
    # TODO: Fix gen_finn_dt_tensor issue in our QONNX (same values
    # for subsequent calls of the function)
    W = gen_finn_dt_tensor(DataType["UINT4"], (10, 10), seed=1)
    T = gen_finn_dt_tensor(DataType["UINT4"], (10, 10), seed=1)

    # Creating the model
    model = make_single_fclayer_modelwrapper(
        W, 1, 1, DataType["UINT4"], DataType["UINT4"], DataType["UINT4"], T, DataType["UINT4"]
    )
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Some sanity checks
    # Not explicitly set. If the default behaviour changes, we need to fix this to be HLS
    assert model.graph.node[0].op_type == "MVAU_hls"
    assert getCustomOp(model.graph.node[0]).get_nodeattr("mem_mode") in [
        "internal_decoupled",
        "internal_embedded",
    ]
    return model, W, T


def mvau_hls_specific_asserts(
    model: ModelWrapper,
    original_op: HWCustomOp,
    original_cache: IPCache,
    original_key: str,
    W: np.ndarray,
    T: np.ndarray,
) -> None:  # noqa
    """Run MVAU_hls specific asserts to validate caching."""
    for attribute in [
        "resType",
        "MW",
        "MH",
        "SIMD",
        "PE",
        "inputDataType",
        "weightDataType",
        "outputDataType",
    ]:
        original_value = original_op.get_nodeattr(attribute)
        if attribute in ["MW", "MH", "SIMD", "PE"]:
            original_op.set_nodeattr(attribute, original_value + 1)  # type: ignore
        elif attribute == "resType":
            assert original_value == "auto"
            original_op.set_nodeattr(attribute, "dsp")
        else:
            original_op.set_nodeattr(attribute, "UINT6")
        assert original_cache.get_key(original_op, model) != original_key
        original_op.set_nodeattr(attribute, original_value)

    # Check that the hash changes with the parameters
    # Weights
    new_W = gen_finn_dt_tensor(DataType["UINT4"], (10, 10), seed=2)
    assert not np.array_equal(W, new_W)
    weight_init = model.graph.node[0].input[1]
    model.set_initializer(weight_init, new_W)
    new_key = original_cache.get_key(original_op, model)
    assert original_key != new_key
    model.set_initializer(weight_init, W)

    # Thresholds
    new_T = gen_finn_dt_tensor(DataType["UINT4"], (10, 10), seed=2)
    assert not np.array_equal(T, new_T)
    thresh_init = model.graph.node[0].input[2]
    model.set_initializer(thresh_init, new_T)
    new_key = original_cache.get_key(original_op, model)
    assert original_key != new_key
    model.set_initializer(thresh_init, T)


def get_first_op(model: ModelWrapper) -> HWCustomOp:
    """Return the op of the first node in the model."""
    return getCustomOp(model.graph.node[0])


@pytest.mark.parametrize("op_type", [MVAU_hls])
@pytest.mark.parametrize("hashfunc", ["sha256"])
@pytest.mark.parametrize("fpgapart", [alveo_part_map["U280"]])
@pytest.mark.parametrize("hls_clk", [2.5])
def test_ip_hash_key(op_type: type, hashfunc: str, fpgapart: str, hls_clk: float) -> None:
    """Test IP Caching.

    To do so, we create models that we then run the cache on. We check, that for
    changes in any attribute, external parameter and clock the hash generated changes as well.
    We also check, that the generated IP is at the correct path, with all meta-information,
    and that subsequent synthesis actually use the cached IP by measuring the time needed
    to re-run synthesis on a fresh copy of the original model.
    """
    os.environ["FINN_IP_CACHE"] = os.environ["FINN_BUILD_DIR"]

    # Create the model
    model: ModelWrapper
    if op_type is MVAU_hls:
        model, W, T = mvau_hls_create_model(fpgapart)
    else:
        raise AssertionError(f"Cache test for op {op_type.__name__} not yet implemented!")

    # Save a copy of the unsynthesized model for later
    unsynth_model = deepcopy(model)

    # Run the cache transformation
    model = model.transform(
        CachedIPGen(
            hash_function=hashfunc,
            include_prepare_ip=True,
            cache_clock=True,
            clk=hls_clk,
            cache_fpgapart=True,
            fpgapart=fpgapart,
        )
    )
    cache = IPCache(
        cache_dir=get_cache_path(),
        hashfunc=hashfunc,
        hls_clk_period=hls_clk,
        cache_hls_clk=True,
        fpgapart=fpgapart,
        cache_fpgapart=True,
    )
    original_op = get_first_op(model)
    original_key = cache.get_key(original_op, model)

    # Check that the hash changes with the attributes
    if op_type is MVAU_hls:
        mvau_hls_specific_asserts(model, original_op, cache, original_key, W, T)
    else:
        raise AssertionError(f"{op_type.__name__} specific cache test asserts not yet implemented!")

    # Check that the IP was cached at the correct path
    path = cache._cache_dir_path(cache.get_hash_hex(original_key))
    assert path.exists()
    assert (path / "nodeattrs.json").exists()
    assert (path / "key.txt").exists()
    with (path / "key.txt").open("r") as f:
        data = f.read()
        assert f"type:{op_type.__name__}" in data
        assert f"Hashed using {hashfunc}" in data
        assert original_key in data

    # Check that a different HLS clk generates a different key
    other_clk_cache = IPCache(get_cache_path(), hashfunc, hls_clk + 1.0, True, fpgapart, True)
    assert cache.get_key(original_op, model) != other_clk_cache.get_key(original_op, model)

    # Check speed of the second call (should be much faster)
    start: float = time.time()
    unsynth_model = unsynth_model.transform(
        CachedIPGen(hashfunc, True, True, hls_clk, True, fpgapart)
    )
    ms_elapsed = time.time() - start

    # Time in seconds that the cached transform may take.
    # 10s should be enough, even on slow systems, but if it is clear that
    # there isn't a bug, this can be adjusted if it leads to failing
    # CI runs.
    CACHE_TIME_ALLOWED = 10
    assert ms_elapsed <= 1000 * CACHE_TIME_ALLOWED

    # Check that the cached and re-used IP does exist
    first_op = get_first_op(unsynth_model)
    expected_ip_path = (
        Path(cast(str, first_op.get_nodeattr("code_gen_dir_ipgen")))
        / f"project_{first_op.onnx_node.name}"
        / "sol1"
        / "impl"
        / "ip"
    )
    assert expected_ip_path.exists()
