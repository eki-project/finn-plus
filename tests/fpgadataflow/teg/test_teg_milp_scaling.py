# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""MILP size and solve-time scaling (paper table and figure, no Vivado).

``test_milp_scaling_table`` is the quick prototype experiment on the synthetic chain.
``test_milp_scaling_plan`` runs an experiment plan (the ``milp_scaling`` CI test suite):
instance sizes of the regression models, solves of synthetic chains, complete small models and
layer prefixes of larger ones, and the power-law fit with its extrapolation. Configured
through environment variables (the pipeline variables of the CI):

* ``MILP_SCALING_PLAN``: preset name (``smoke``, ``cluster``, ``cluster_full``) or the path of
  a JSON/YAML plan ``{"instances": [{"kind": "chain", "n": 64}, {"kind": "model", "name":
  "tfc-w1a1"}, {"kind": "prefix", "name": "cnv-w1a1", "nodes": 5}, {"kind": "size", "name":
  "resnet18"}, ...]}``; an instance may carry its own ``time_limit``;
* ``MILP_SCALING_SOLVER`` (``highs``/``gurobi``), ``MILP_SCALING_TIME_LIMIT`` (seconds per
  instance), ``MILP_SCALING_THREADS``, ``MILP_SCALING_MEM_LIMIT_GB``,
  ``MILP_SCALING_MAX_CONSTRAINTS`` (instances above it are only sized, 0: no limit);
* results go to ``$CI_PROJECT_DIR/reports/milp_scaling`` (archived by the CI) or the FINN
  build directory.

The regression models are built without Vivado up to their folding/bit-width step (the
``dut`` recipes of the bench flow), which gives the exact instance structure; HLS latencies
of the templates then take their defaults.
"""

import pytest

import json
import os
import time
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.analysis.fpgadataflow.teg import scaling
from finn.analysis.fpgadataflow.teg.milp import fit_power_law, solve_milp
from finn.analysis.fpgadataflow.teg.simulate import simulate
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from finn.util.logging import log
from tests.fpgadataflow.teg.test_teg_core import chain_graph, unit_candidates
from tests.fpgadataflow.teg.test_teg_end_to_end import MODELS, describe_missing, ensure_model

CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

#: regression models: (dut recipe, parameters beyond the recipe), as in the CI bench configs
REGRESSION_MODELS: dict[str, tuple[str, dict]] = {
    "vgg10": ("vgg10", {}),
    "cybsec": ("cybsec", {}),
    "cnv-w1a1": (
        "bnn-pynq",
        {
            "model_path": "models/bnn-pynq/cnv-w1a1_qonnx.onnx",
            "folding_config_file": "models/bnn-pynq/cnv-w1a1_folding_config.json",
            "specialize_layers_config_file": "models/bnn-pynq/cnv-w1a1_specialize_layers.json",
        },
    ),
    "cnv-w1a2": (
        "bnn-pynq",
        {
            "model_path": "models/bnn-pynq/cnv-w1a2_qonnx.onnx",
            "folding_config_file": "models/bnn-pynq/cnv-w1a2_folding_config.json",
            "specialize_layers_config_file": "models/bnn-pynq/cnv-w1a2_specialize_layers.json",
        },
    ),
    "cnv-w2a2": (
        "bnn-pynq",
        {
            "model_path": "models/bnn-pynq/cnv-w2a2_qonnx.onnx",
            "folding_config_file": "models/bnn-pynq/cnv-w2a2_folding_config.json",
            "specialize_layers_config_file": "models/bnn-pynq/cnv-w2a2_specialize_layers.json",
        },
    ),
    "tfc-w1a1": (
        "bnn-pynq",
        {
            "model_path": "models/bnn-pynq/tfc-w1a1_qonnx.onnx",
            "folding_config_file": "models/bnn-pynq/tfc-w1a1_folding_config.json",
            "specialize_layers_config_file": "models/bnn-pynq/tfc-w1a1_specialize_layers.json",
        },
    ),
    "tfc-w1a2": (
        "bnn-pynq",
        {
            "model_path": "models/bnn-pynq/tfc-w1a2_qonnx.onnx",
            "folding_config_file": "models/bnn-pynq/tfc-w1a2_folding_config.json",
            "specialize_layers_config_file": "models/bnn-pynq/tfc-w1a2_specialize_layers.json",
        },
    ),
    "tfc-w2a2": (
        "bnn-pynq",
        {
            "model_path": "models/bnn-pynq/tfc-w2a2_qonnx.onnx",
            "folding_config_file": "models/bnn-pynq/tfc-w2a2_folding_config.json",
            "specialize_layers_config_file": "models/bnn-pynq/tfc-w2a2_specialize_layers.json",
        },
    ),
    "gtsrb": ("gtsrb", {}),
    "kws": ("kws", {}),
    "transformer-benchmark": (
        "transformer",
        {
            "model_path": "models/transformer/finn-transformers/benchmark/streamlined.onnx",
            "target_fps": 1500,
        },
    ),
    "transformer-radioml": (
        "transformer",
        {
            "model_path": "models/transformer/finn-transformers/radioml/streamlined.onnx",
            "target_fps": 95000,
        },
    ),
    "transformer-vision": (
        "transformer",
        {
            "model_path": "models/transformer/finn-transformers/vision/streamlined.onnx",
            "target_fps": 1525,
        },
    ),
    "transformer-language": (
        "transformer",
        {
            "model_path": "models/transformer/finn-transformers/language/streamlined.onnx",
            "target_fps": 1000,
        },
    ),
    "mobilenetv1": ("mobilenetv1", {}),
    "resnet18": (
        "resnet18",
        {
            "model_path": "models/resnet18/resnet18_w3a3_cifar100.onnx",
            "folding_config_file": "models/resnet18/resnet18_folding_config.json",
            "specialize_layers_config_file": "models/resnet18/resnet18_specialize_layers.json",
            "synth_clk_period_ns": 10,
        },
    ),
}
#: the build stops after the last of these steps (folding and bit widths are final then)
STOP_STEPS = {
    "step_minimize_bit_width",
    "step_apply_folding_config",
    "finn.builder.custom_step_library.transformer_adhoc.step_set_folding",
}

PRESETS: dict[str, dict] = {
    "smoke": {
        "instances": [
            {"kind": "chain", "n": 32},
            {"kind": "chain", "n": 64},
            {"kind": "model", "name": "tfc-w1a1"},
            {"kind": "prefix", "name": "tfc-w1a1", "nodes": 4},
            {"kind": "size", "name": "cnv-w1a1"},
        ]
    },
    "cluster": {
        "instances": [
            *({"kind": "size", "name": n} for n in REGRESSION_MODELS),
            *(
                {"kind": "chain", "n": n}
                for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
            ),
            *(
                {"kind": "model", "name": n}
                for n in ("tfc-w1a1", "tfc-w1a2", "tfc-w2a2", "cybsec", "kws")
            ),
            *({"kind": "prefix", "name": "vgg10", "nodes": k} for k in (6, 12, 20, 32, 48)),
            *(
                {"kind": "prefix", "name": "cnv-w1a1", "nodes": k}
                for k in (3, 5, 7, 9, 12, 16, 20, 26)
            ),
            *(
                {"kind": "prefix", "name": "transformer-benchmark", "nodes": k}
                for k in (4, 8, 12, 16, 20)
            ),
            *(
                {"kind": "prefix", "name": "transformer-radioml", "nodes": k}
                for k in (8, 16, 24, 32, 48)
            ),
            *(
                {"kind": "prefix", "name": "transformer-vision", "nodes": k}
                for k in (8, 16, 24, 32)
            ),
            {"kind": "model", "name": "vgg10"},
            {"kind": "model", "name": "transformer-radioml"},
            {"kind": "model", "name": "cnv-w1a1"},
            {"kind": "model", "name": "gtsrb"},
            {"kind": "model", "name": "transformer-benchmark"},
        ]
    },
}
# the demonstration solves of the large models: days each, best effort, incumbents on disk
PRESETS["cluster_full"] = {
    "instances": [
        *PRESETS["cluster"]["instances"],
        *(
            {"kind": "model", "name": n}
            for n in (
                "cnv-w1a2",
                "cnv-w2a2",
                "transformer-vision",
                "resnet18",
                "transformer-language",
                "mobilenetv1",
            )
        ),
    ]
}

pytestmark = [pytest.mark.fpgadataflow, pytest.mark.slow]


@pytest.mark.fifo_model
def test_milp_scaling_table() -> None:
    """Solve the chain instance for growing token counts and fit ``t = a * n^b``."""
    rows = []
    points = []
    for n, burst in [(32, 8), (64, 8), (128, 8), (256, 16)]:
        m = chain_graph(n=n, burst=burst)
        target = m.bottleneck_interval()
        t0 = time.time()
        res = solve_milp(m, target, unit_candidates(m, CANDIDATES), time_limit=300)
        dt = time.time() - t0
        assert res.depths is not None, res.message
        assert simulate(m, res.depths).interval == target
        rows.append(
            (n, res.size.variables, res.size.constraints, res.size.binaries, dt, res.depths)
        )
        points.append((res.size.constraints, dt))
    a, b = fit_power_law(points)
    print("\n  N   vars   cons  binaries  time[s]  depths")
    for n, nv, nc, nb, dt, d in rows:
        print(f"  {n:4d} {nv:6d} {nc:7d} {nb:8d} {dt:8.2f}  {d}")
    print(f"  fit: t = {a:.3g} * cons^{b:.2f}; extrapolated 1e8 rows: {a * 1e8**b:.3g} s")
    # the constraint count grows linearly with the tokens per frame, the time faster than that
    assert rows[-1][2] > 6 * rows[0][2]
    assert b > 1.0


def _recipe(dut: str) -> dict:
    """The dut recipe of the bench flow."""
    import yaml
    from importlib.resources import files

    return yaml.safe_load((files("finn.benchmarking") / "dut" / f"{dut}.yml").read_text())


_GRAPHS: dict[str, ModelWrapper] = {}


def dry_run_graph(name: str) -> ModelWrapper:
    """The dataflow graph of regression model ``name`` after folding and bit-width
    minimisation, with data width converters inserted and layers specialised: the input of
    the TEG builder, built without Vivado (cached per test process)."""
    if name in _GRAPHS:
        return _GRAPHS[name]
    dut, params = REGRESSION_MODELS[name]
    rec = _recipe(dut)
    cfg_params = {k: v for k, v in rec.items() if k != "steps"}
    cfg_params.update(params)
    steps = rec.get("steps") or list(build_cfg.default_build_dataflow_steps)
    cut = max((i for i, s in enumerate(steps) if s in STOP_STEPS), default=len(steps) - 1)
    steps = steps[: cut + 1]
    model_path = MODELS.parent / cfg_params.pop("model_path")
    pulled = ensure_model(model_path)
    if not model_path.is_file():
        pytest.skip(f"{model_path} not available: {describe_missing(model_path)}; {pulled}")
    for key in ("folding_config_file", "specialize_layers_config_file"):
        if cfg_params.get(key):
            cfg_params[key] = str(MODELS.parent / cfg_params[key])
    for key in (
        "verify_steps",
        "verify_save_rtlsim_waveforms",
        "verification_atol",
        "stitched_ip_gen_dcp",
        "validation_dataset",
    ):
        cfg_params.pop(key, None)
    cfg_params.setdefault("synth_clk_period_ns", 10.0)
    out_dir = Path(make_build_dir(f"milp_scaling_{name}_"))
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=str(out_dir),
        board="RFSoC2x2",
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        steps=steps,
        **cfg_params,
    )
    build.build_dataflow_cfg(str(model_path), cfg)
    last = steps[-1].split(".")[-1]
    model = ModelWrapper(str(out_dir / "intermediate_models" / f"{last}.onnx"))
    sdp = [n for n in model.graph.node if n.op_type == "StreamingDataflowPartition"]
    if sdp:
        model = ModelWrapper(getCustomOp(sdp[0]).get_nodeattr("model"))
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    _GRAPHS[name] = model
    return model


def load_plan() -> dict:
    """The experiment plan from ``MILP_SCALING_PLAN`` (preset name or file)."""
    spec = os.environ.get("MILP_SCALING_PLAN", "smoke")
    if spec in PRESETS:
        return PRESETS[spec]
    path = Path(spec)
    text = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def results_dir() -> Path:
    """Where the CI archives the results from (``reports/`` of the project directory)."""
    base = os.environ.get("CI_PROJECT_DIR")
    if base:
        return Path(base) / "reports" / "milp_scaling"
    return Path(make_build_dir("milp_scaling_results_"))


@pytest.mark.milp_scaling
def test_milp_scaling_plan() -> None:
    """Run the configured experiment plan and write sizes, solves and the fit."""
    plan = load_plan()
    solver = os.environ.get("MILP_SCALING_SOLVER", "highs")
    time_limit = float(os.environ.get("MILP_SCALING_TIME_LIMIT", "900"))
    threads = int(os.environ.get("MILP_SCALING_THREADS", "0"))
    mem_limit = os.environ.get("MILP_SCALING_MEM_LIMIT_GB")
    max_rows = int(os.environ.get("MILP_SCALING_MAX_CONSTRAINTS", "0"))
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(
        json.dumps(
            {
                "plan": plan,
                "solver": solver,
                "time_limit": time_limit,
                "threads": threads,
                "mem_limit_gb": mem_limit,
                "max_constraints": max_rows,
            },
            indent=2,
        )
    )
    log.info(f"milp scaling: {len(plan['instances'])} instances, results in {out}")
    failures = []
    for spec in plan["instances"]:
        kind = spec["kind"]
        t0 = time.time()
        try:
            if kind == "chain":
                inst = scaling.chain_instance(int(spec["n"]), int(spec.get("burst", 8)))
            else:
                graph = dry_run_graph(spec["name"])
                nodes = int(spec["nodes"]) if kind == "prefix" else None
                label = spec["name"] + (f" prefix {nodes} nodes" if nodes else "")
                inst = scaling.model_instance(label, graph, nodes)
                if kind == "size":
                    inst.kind = "size"
            if kind == "size":
                row = scaling.record_size(inst, out)
            else:
                size = scaling.size_row(inst)
                if max_rows and size["size_constraints"] > max_rows:
                    log.warning(
                        f"{inst.label}: {size['size_constraints']} rows exceed "
                        f"MILP_SCALING_MAX_CONSTRAINTS = {max_rows}; size only"
                    )
                    row = scaling.record_size(inst, out)
                else:
                    row = scaling.run_instance(
                        inst,
                        out,
                        solver=solver,
                        time_limit=float(spec.get("time_limit", time_limit)),
                        threads=threads,
                        mem_limit_gb=float(mem_limit) if mem_limit else None,
                    )
            log.info(
                f"{inst.label}: {row['size_constraints']} rows, "
                f"{row.get('solve_s', '-')} s, {row.get('message', 'size only')} "
                f"({time.time() - t0:.0f} s total)"
            )
        except Exception as exc:  # keep going: one failed instance must not lose the others
            log.error(f"instance {spec} failed: {exc!r}")
            failures.append((spec, repr(exc)))
            (out / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = scaling.summarize(out)
    print((out / "table.md").read_text())
    assert summary["rows"], "no instance produced a result"
    assert not failures, f"{len(failures)} instances failed: {failures}"
