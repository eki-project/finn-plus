"""Tests for the IP cache (finn.transformation.fpgadataflow.ip_cache)."""

import pytest

import json
import numpy as np
import os
import threading
from copy import deepcopy
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import step_hw_codegen, step_hw_ipgen
from finn.interface.settings import FINN_ROOT, FINNSettings
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.ip_cache import (
    CACHE_KEY_FILE,
    CACHE_META_FILE,
    PENDING_KEYS_METADATA_PROP,
    IPCache,
    RestoreCachedIPs,
    StoreGeneratedIPs,
    get_environment_identity,
    is_cacheable_node,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import getHWCustomOp, pynq_part_map
from finn.util.settings import get_settings
from tests.fpgadataflow.test_fpgadataflow_fmpadding import make_single_fmpadding_modelwrapper
from tests.fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper

FPGAPART = pynq_part_map["Pynq-Z1"]
CLK_NS = 5.0


def make_fmpadding_rtl_model():
    """Return a model with a single FMPadding_rtl node (no initializers, no HLS needed)."""
    model = make_single_fmpadding_modelwrapper([8, 8], [1, 1, 1, 1], 4, 2, DataType["INT4"])
    getHWCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", "rtl")
    model = model.transform(SpecializeLayers(FPGAPART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    assert model.graph.node[0].op_type == "FMPadding_rtl"
    return model


def make_mvau_hls_model(seed: int = 0):
    """Return a model with a single MVAU_hls node with weights and thresholds as initializers."""
    rng = np.random.default_rng(seed)
    mw, mh = 8, 8
    idt = wdt = odt = DataType["INT4"]
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    n_steps = odt.get_num_possible_values() - 1
    T = np.sort(rng.integers(-32, 32, (mh, n_steps)), axis=1).astype(np.float32)
    model = make_single_fclayer_modelwrapper(W, 2, 2, wdt, idt, odt, T, DataType["INT32"])
    getHWCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", "hls")
    model = model.transform(SpecializeLayers(FPGAPART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    assert model.graph.node[0].op_type == "MVAU_hls"
    return model


def first_node(model):
    """Return the first node of the model and its custom op instance."""
    node = model.graph.node[0]
    return node, getHWCustomOp(node)


def relative_files(directory: Path, ignore: tuple[str, ...] = ()) -> set[str]:
    """Return the set of all file paths below directory, relative to it."""
    return {
        str(p.relative_to(directory))
        for p in Path(directory).rglob("*")
        if p.is_file() and p.name not in ignore
    }


def files_containing(directory: Path, needle: str) -> list[Path]:
    """Return all files below directory containing the given string."""
    return [
        p for p in Path(directory).rglob("*") if p.is_file() and needle.encode() in p.read_bytes()
    ]


@pytest.fixture
def cache_dir(tmp_path):
    """Return an empty cache directory for one test."""
    directory = tmp_path / "ip_cache"
    directory.mkdir()
    return directory


@pytest.mark.infrastructure
def test_ip_cache_key_sensitivity(cache_dir):
    cache = IPCache(cache_dir, FPGAPART, CLK_NS)

    # RTL node without initializers
    model = make_fmpadding_rtl_model()
    node, op = first_node(model)
    assert is_cacheable_node(node)
    key = cache.build_key(node, model)
    assert f"node_name: {node.name}" in key
    assert "attr SIMD: 2" in key
    assert "initializer" not in key
    copied = deepcopy(model)
    assert cache.build_key(copied.graph.node[0], copied) == key
    assert cache.key_hash(key) == cache.key_hash(cache.build_key(node, model))

    # Attributes influencing the IP change the key, ignored attributes don't
    op.set_nodeattr("SIMD", 4)
    assert cache.build_key(node, model) != key
    op.set_nodeattr("SIMD", 2)
    assert cache.build_key(node, model) == key
    op.set_nodeattr("inFIFODepths", [42])
    op.set_nodeattr("slr", 3)
    op.set_nodeattr("code_gen_dir_ipgen", "/some/where")
    assert cache.build_key(node, model) == key

    # The node name is baked into the generated IP and thus part of the key
    node.name = "renamed_node"
    assert cache.build_key(node, model) != key
    node.name = "FMPadding_rtl_0"
    assert cache.build_key(node, model) == key

    # Target changes the key
    assert IPCache(cache_dir, FPGAPART, CLK_NS + 1.0).build_key(node, model) != key
    assert IPCache(cache_dir, "xcu250-figd2104-2L-e", CLK_NS).build_key(node, model) != key

    # Environment identity is part of the key
    for name in (
        "finn_plus_version",
        "finn_plus_commit",
        "xilinx_vivado",
        "dependency finn-hlslib",
    ):
        assert f"{name}: {get_environment_identity()[name]}" in key

    # HLS node with initializers: weights and thresholds are part of the key
    model = make_mvau_hls_model()
    node, op = first_node(model)
    key = cache.build_key(node, model)
    assert "input[1] initializer: sha256=" in key
    assert "input[2] initializer: sha256=" in key
    W = model.get_initializer(node.input[1])
    model.set_initializer(node.input[1], W * -1)
    assert cache.build_key(node, model) != key
    model.set_initializer(node.input[1], W)
    assert cache.build_key(node, model) == key
    T = model.get_initializer(node.input[2])
    model.set_initializer(node.input[2], T + 1)
    assert cache.build_key(node, model) != key
    model.set_initializer(node.input[2], T)
    assert cache.build_key(node, model) == key
    # Same content, different shape
    model.set_initializer(node.input[2], T.reshape(T.shape[1], T.shape[0]))
    assert cache.build_key(node, model) != key

    # Nodes that are not specialized to a backend are not cacheable
    unspecialized = make_single_fmpadding_modelwrapper([8, 8], [1, 1, 1, 1], 4, 2, DataType["INT4"])
    assert not is_cacheable_node(unspecialized.graph.node[0])


@pytest.mark.infrastructure
def test_ip_cache_rtl_roundtrip(cache_dir):
    model = make_fmpadding_rtl_model()
    original = deepcopy(model)

    # First build: cache miss, key is remembered until the IP has been generated and stored
    model = model.transform(RestoreCachedIPs(FPGAPART, CLK_NS, cache_dir))
    node, op = first_node(model)
    assert op.get_nodeattr("code_gen_dir_ipgen") == ""
    pending = json.loads(model.get_metadata_prop(PENDING_KEYS_METADATA_PROP))
    assert set(pending.keys()) == {node.name}
    cache = IPCache(cache_dir, FPGAPART, CLK_NS)
    assert pending[node.name]["hash"] == cache.key_hash(cache.build_key(node, model))
    assert cache.num_entries() == 0

    model = model.transform(PrepareIP(FPGAPART, CLK_NS))
    model = model.transform(StoreGeneratedIPs(FPGAPART, CLK_NS, cache_dir))
    node, op = first_node(model)
    assert model.get_metadata_prop(PENDING_KEYS_METADATA_PROP) is None
    assert cache.num_entries() == 1
    entry = cache.entry_dir(pending[node.name]["hash"])
    assert (entry / CACHE_META_FILE).is_file()
    assert (entry / CACHE_KEY_FILE).is_file()
    assert f"op_type: {node.op_type}" in (entry / CACHE_KEY_FILE).read_text()
    meta = json.loads((entry / CACHE_META_FILE).read_text())
    generated_dir = op.get_nodeattr("code_gen_dir_ipgen")
    assert meta["code_gen_dir_ipgen"] == generated_dir
    assert meta["node_name"] == node.name
    assert meta["nodeattrs"]["gen_top_module"] == op.get_nodeattr("gen_top_module")
    generated_files = relative_files(generated_dir)
    assert len(generated_files) > 0
    assert relative_files(entry, ignore=(CACHE_META_FILE, CACHE_KEY_FILE)) == generated_files
    assert not list(cache_dir.glob(".staging/*"))

    # Second build of the same model: cache hit
    restored = original.transform(RestoreCachedIPs(FPGAPART, CLK_NS, cache_dir))
    assert restored.get_metadata_prop(PENDING_KEYS_METADATA_PROP) is None
    node, op = first_node(restored)
    restored_dir = op.get_nodeattr("code_gen_dir_ipgen")
    assert restored_dir != "" and restored_dir != generated_dir
    assert Path(restored_dir).is_relative_to(get_settings().finn_build_dir)
    assert relative_files(restored_dir) == generated_files
    assert op.get_nodeattr("ipgen_path") == restored_dir
    assert op.get_nodeattr("ip_path") == restored_dir
    assert op.get_nodeattr("gen_top_module") == meta["nodeattrs"]["gen_top_module"]
    # Nothing refers to the build directory of the first build anymore
    assert files_containing(restored_dir, generated_dir) == []

    # Code generation and IP generation are skipped for the restored node
    mtimes = {p: p.stat().st_mtime_ns for p in Path(restored_dir).rglob("*")}
    restored = restored.transform(PrepareIP(FPGAPART, CLK_NS))
    restored = restored.transform(HLSSynthIP(FPGAPART))
    assert {p: p.stat().st_mtime_ns for p in Path(restored_dir).rglob("*")} == mtimes
    contents = {p: p.read_bytes() for p in Path(restored_dir).rglob("*") if p.is_file()}
    restored = restored.transform(ReplaceVerilogRelPaths())
    assert {p: p.read_bytes() for p in Path(restored_dir).rglob("*") if p.is_file()} == contents
    # Nothing new to store
    restored = restored.transform(StoreGeneratedIPs(FPGAPART, CLK_NS, cache_dir))
    assert cache.num_entries() == 1

    # A model with a different configuration does not hit the entry
    other = make_fmpadding_rtl_model()
    first_node(other)[1].set_nodeattr("SIMD", 4)
    other = other.transform(RestoreCachedIPs(FPGAPART, CLK_NS, cache_dir))
    assert first_node(other)[1].get_nodeattr("code_gen_dir_ipgen") == ""
    assert other.get_metadata_prop(PENDING_KEYS_METADATA_PROP) is not None


@pytest.mark.infrastructure
def test_ip_cache_build_steps(cache_dir, tmp_path):
    """The build steps use the cache directory from the global settings."""
    settings = get_settings()
    previous = settings.finn_ip_cache
    previous_env = os.environ.get("FINN_IP_CACHE")
    settings.finn_ip_cache = cache_dir
    try:
        cfg = DataflowBuildConfig(
            output_dir=str(tmp_path / "out"),
            synth_clk_period_ns=CLK_NS,
            fpga_part=FPGAPART,
            generate_outputs=[],
        )
        cache = IPCache(cache_dir, FPGAPART, CLK_NS)

        model = step_hw_ipgen(step_hw_codegen(make_fmpadding_rtl_model(), cfg), cfg)
        assert cache.num_entries() == 1
        generated_dir = first_node(model)[1].get_nodeattr("code_gen_dir_ipgen")
        assert model.get_metadata_prop(PENDING_KEYS_METADATA_PROP) is None

        model = step_hw_ipgen(step_hw_codegen(make_fmpadding_rtl_model(), cfg), cfg)
        assert cache.num_entries() == 1
        restored_dir = first_node(model)[1].get_nodeattr("code_gen_dir_ipgen")
        assert restored_dir != generated_dir
        assert relative_files(restored_dir) == relative_files(generated_dir)

        # Caching can be switched off per build
        cfg.use_ip_cache = False
        other = make_fmpadding_rtl_model()
        first_node(other)[1].set_nodeattr("SIMD", 4)
        other = step_hw_ipgen(step_hw_codegen(other, cfg), cfg)
        assert first_node(other)[1].get_nodeattr("code_gen_dir_ipgen") != ""
        assert cache.num_entries() == 1
    finally:
        settings.finn_ip_cache = previous
        if previous_env is None:
            os.environ.pop("FINN_IP_CACHE", None)
        else:
            os.environ["FINN_IP_CACHE"] = previous_env


@pytest.mark.infrastructure
def test_ip_cache_concurrent_store(cache_dir, tmp_path):
    """Several builds storing the same entry at once yield exactly one complete entry."""
    model = make_fmpadding_rtl_model()
    node, _ = first_node(model)
    cache = IPCache(cache_dir, FPGAPART, CLK_NS)
    key = cache.build_key(node, model)
    key_hash = cache.key_hash(key)
    code_gen_dir = tmp_path / "code_gen_dir"
    (code_gen_dir / "sub").mkdir(parents=True)
    (code_gen_dir / "top.v").write_text("module top; endmodule\n")
    (code_gen_dir / "sub" / "data.dat").write_text("00\n")
    (code_gen_dir / "input_1.npy").write_bytes(b"cached, generated by codegen")
    (code_gen_dir / "sim.wdb").write_bytes(b"not cached")
    nodeattrs = {"ipgen_path": str(code_gen_dir), "ip_path": str(code_gen_dir)}

    results = []
    barrier = threading.Barrier(4)

    def store():
        barrier.wait()
        results.append(cache.store(node, key, key_hash, str(code_gen_dir), nodeattrs))

    threads = [threading.Thread(target=store) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False, False, False, True]
    assert cache.num_entries() == 1
    entry = cache.entry_dir(key_hash)
    assert relative_files(entry, ignore=(CACHE_META_FILE, CACHE_KEY_FILE)) == {
        "top.v",
        "sub/data.dat",
        "input_1.npy",
    }
    assert (entry / CACHE_KEY_FILE).read_text() == key
    assert not list(cache_dir.glob(".staging/*"))
    # Storing again is a no-op
    assert cache.store(node, key, key_hash, str(code_gen_dir), nodeattrs) is False


@pytest.mark.infrastructure
def test_ip_cache_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("FINN_IP_CACHE", raising=False)

    def init(**kwargs):
        return FINNSettings.init(
            override_settings_path=tmp_path / "missing_settings.yaml",
            flow_config=Path(),
            auto_set_environment_vars=False,
            **kwargs,
        )

    assert init().finn_ip_cache == FINN_ROOT / "FINN_IP_CACHE"
    assert init(finn_ip_cache="relative/cache").finn_ip_cache == FINN_ROOT / "relative/cache"
    assert init(finn_ip_cache=str(tmp_path)).finn_ip_cache == tmp_path
    for disabled in ["", "none", "NONE", "off", "disabled"]:
        assert init(finn_ip_cache=disabled).finn_ip_cache is None

    # Environment variable and settings file
    monkeypatch.setenv("FINN_IP_CACHE", str(tmp_path / "from_env"))
    assert init().finn_ip_cache == tmp_path / "from_env"
    monkeypatch.setenv("FINN_IP_CACHE", "")
    assert init().finn_ip_cache is None
    monkeypatch.delenv("FINN_IP_CACHE")

    settings = init(finn_ip_cache="none")
    settings_path = tmp_path / "settings.yaml"
    settings.save(installation_independent=True, path=settings_path)
    assert "finn_ip_cache: ''" in settings_path.read_text()
    reloaded = FINNSettings.init(
        override_settings_path=settings_path, flow_config=Path(), auto_set_environment_vars=False
    )
    assert reloaded.finn_ip_cache is None

    # Settings with automatic environment export (restored by monkeypatch afterwards)
    for key in (
        "FINN_BUILD_DIR",
        "FINN_DEPS",
        "FINN_DEPS_DEFINITIONS",
        "FINN_IP_CACHE",
        "NUM_DEFAULT_WORKERS",
    ):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)
    settings = init(finn_ip_cache=str(tmp_path / "from_file"))
    settings.save(installation_independent=True, path=settings_path)
    reloaded = FINNSettings.init(
        override_settings_path=settings_path, flow_config=Path(), auto_set_environment_vars=True
    )
    assert reloaded.finn_ip_cache == tmp_path / "from_file"
    assert os.environ["FINN_IP_CACHE"] == str(tmp_path / "from_file")
    reloaded.finn_ip_cache = None
    assert os.environ["FINN_IP_CACHE"] == ""


@pytest.mark.infrastructure
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_ip_cache_hls_roundtrip(cache_dir):
    """A restored HLS IP is complete, skips HLS synthesis and behaves like the generated one."""
    model = make_mvau_hls_model()
    original = deepcopy(model)
    inp, outp = model.graph.input[0].name, model.graph.output[0].name
    x = gen_finn_dt_tensor(DataType["INT4"], (1, 8))

    model = model.transform(RestoreCachedIPs(FPGAPART, CLK_NS, cache_dir))
    model = model.transform(PrepareIP(FPGAPART, CLK_NS))
    model = model.transform(HLSSynthIP(FPGAPART))
    model = model.transform(StoreGeneratedIPs(FPGAPART, CLK_NS, cache_dir))
    node, op = first_node(model)
    cache = IPCache(cache_dir, FPGAPART, CLK_NS)
    assert cache.num_entries() == 1
    entry = next(p for p in cache_dir.iterdir() if (p / CACHE_META_FILE).is_file())
    project = entry / f"project_{node.name}"
    assert (project / "sol1" / "impl" / "ip").is_dir()
    assert (project / "sol1" / "impl" / "verilog").is_dir()
    assert (project / "sol1" / "syn" / "report").is_dir()
    assert not (project / "sol1" / ".autopilot").exists()
    generated_dir = op.get_nodeattr("code_gen_dir_ipgen")

    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    y_generated = oxe.execute_onnx(model, {inp: x})[outp]

    restored = original.transform(RestoreCachedIPs(FPGAPART, CLK_NS, cache_dir))
    node, op = first_node(restored)
    restored_dir = op.get_nodeattr("code_gen_dir_ipgen")
    assert restored_dir != generated_dir
    assert op.get_nodeattr("ipgen_path") == f"{restored_dir}/project_{node.name}"
    assert op.get_nodeattr("ip_path") == f"{restored_dir}/project_{node.name}/sol1/impl/ip"
    assert Path(op.get_nodeattr("ip_path")).is_dir()
    assert op.get_nodeattr("ip_vlnv") == f"xilinx.com:hls:{node.name}:1.0"
    assert files_containing(restored_dir, generated_dir) == []

    # PrepareIP and HLSSynthIP must not run again (HLS would recreate the project database)
    restored = restored.transform(PrepareIP(FPGAPART, CLK_NS))
    restored = restored.transform(HLSSynthIP(FPGAPART))
    assert not (Path(restored_dir) / f"project_{node.name}" / "sol1" / ".autopilot").exists()

    restored = restored.transform(ReplaceVerilogRelPaths())
    restored = restored.transform(SetExecMode("rtlsim"))
    restored = restored.transform(PrepareRTLSim())
    y_restored = oxe.execute_onnx(restored, {inp: x})[outp]
    assert np.array_equal(y_restored, y_generated)
