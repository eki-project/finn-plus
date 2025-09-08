"""Test that the IP cache is working correctly. (No false positives, no collisions, speed, etc.)."""
import pytest

import numpy as np
import os
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.transformation.fpgadataflow.ip_cache import CachedIPGen, IPCache
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import alveo_part_map
from finn.util.deps import get_cache_path
from tests.fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper


@pytest.mark.parametrize("op_type", [MVAU_hls])
@pytest.mark.parametrize("hashfunc", ["sha256"])
@pytest.mark.parametrize("fpgapart", [alveo_part_map["U280"]])
def test_ip_hash_key(op_type: type, hashfunc: str, fpgapart: str) -> None:
    """Test that key generation doesnt create false positives or collisions."""
    os.environ["FINN_IP_CACHE"] = os.environ["FINN_BUILD_DIR"]
    if op_type is MVAU_hls:
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

        # Run the cache transformation
        model = model.transform(CachedIPGen(hashfunc, True, True, 2.5, True, fpgapart))
        cache = IPCache(get_cache_path(), hashfunc, 2.5, True, fpgapart, True)
        original_key = cache.get_key(getCustomOp(model.graph.node[0]), model)

        # Check that the hash changes with the attributes
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
            op = getCustomOp(model.graph.node[0])
            original_value = op.get_nodeattr(attribute)
            if attribute in ["MW", "MH", "SIMD", "PE"]:
                op.set_nodeattr(attribute, original_value + 1)
            elif attribute == "resType":
                assert original_value == "auto"
                op.set_nodeattr(attribute, "dsp")
            else:
                op.set_nodeattr(attribute, "UINT6")
            assert cache.get_key(op, model) != original_key
            op.set_nodeattr(attribute, original_value)

        # Check that the hash changes with the parameters
        # Weights
        new_W = gen_finn_dt_tensor(DataType["UINT4"], (10, 10), seed=2)
        assert not np.array_equal(W, new_W)
        weight_init = model.graph.node[0].input[1]
        model.set_initializer(weight_init, new_W)
        new_key = cache.get_key(getCustomOp(model.graph.node[0]), model)
        assert original_key != new_key
        model.set_initializer(weight_init, W)

        # Thresholds
        new_T = gen_finn_dt_tensor(DataType["UINT4"], (10, 10), seed=2)
        assert not np.array_equal(T, new_T)
        thresh_init = model.graph.node[0].input[2]
        model.set_initializer(thresh_init, new_T)
        new_key = cache.get_key(getCustomOp(model.graph.node[0]), model)
        assert original_key != new_key
        model.set_initializer(thresh_init, T)

        # Check that the IP was cached at the correct path
        path = cache._cache_dir_path(cache.get_hash_hex(original_key))
        assert path.exists()
        assert (path / "nodeattrs.json").exists()
        assert (path / "key.txt").exists()
        with (path / "key.txt").open("r") as f:
            data = f.read()
            assert "type:MVAU_hls" in data
            assert f"Hashed using {hashfunc}" in data
            assert original_key in data

    else:
        raise RuntimeError(f"Test for op type {op_type.__name__} not implemented!")
