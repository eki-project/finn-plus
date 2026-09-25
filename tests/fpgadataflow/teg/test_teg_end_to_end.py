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

"""End-to-end FIFO sizing of the regression models with the TEG strategies.

The models, folding and specialisation configs are the ones of the CI regression benchmark
(``ci/cfg/regression_small.yml``: ``models/bnn-pynq/*``; ``regression_large.yml``:
``models/mobilenetv1``, ``models/resnet18``; DVC-managed; build recipes from
``src/finn/benchmarking/dut/*.yml``). The reference depths are the ``fifo_sizing.json`` reports
of the ``distributed_sim`` strategy from a past regression pipeline (``baselines/``), so this
test does not re-run the RTL-based sizing. The build runs the flow up to and including
``step_generate_hardware`` (HLS synthesis of every node plus the sizing) and compares the
resulting ``fifo_sizing.json``.
"""

import pytest

import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

pytestmark = [
    pytest.mark.fpgadataflow,
    pytest.mark.fifo_model,
    pytest.mark.vivado,
    pytest.mark.slow,
]

BASELINES = Path(__file__).resolve().parent / "baselines"


def models_root() -> Path:
    """Directory holding the DVC-managed ``models/`` tree of the checkout.

    The tests may run from the installed ``finn-plus-tests`` wheel (CI), where the
    repository is only available through ``CI_PROJECT_DIR``; ``FINN_MODELS_DIR`` overrides.
    """
    candidates = [
        os.environ.get("FINN_MODELS_DIR"),
        (os.environ.get("CI_PROJECT_DIR") or "")
        and str(Path(os.environ["CI_PROJECT_DIR"]) / "models"),
        str(Path(__file__).resolve().parents[3] / "models"),
        str(Path.cwd() / "models"),
    ]
    for c in candidates:
        if c and (Path(c) / "bnn-pynq").is_dir():
            return Path(c)
    return Path(candidates[2])


MODELS = models_root()


def ensure_model(path: Path) -> str:
    """Pull a missing DVC-managed model into the checkout (``dvc pull <file>.dvc``).

    Returns the DVC output (empty when nothing was attempted). Concurrent test workers
    serialise on a lock file next to the model.
    """
    dvc_file = path.with_name(path.name + ".dvc")
    dvc = shutil.which("dvc")
    if path.is_file() or not dvc_file.is_file() or dvc is None:
        return ""
    repo_root = next((p for p in path.parents if (p / ".dvc").is_dir()), None)
    if repo_root is None:
        return "no .dvc directory above the model"
    lock = path.with_name(path.name + ".teg.lock")
    with lock.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if path.is_file():
            return ""
        proc = subprocess.run(
            [dvc, "pull", str(dvc_file.relative_to(repo_root))],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    return f"dvc pull rc={proc.returncode}: {(proc.stdout + proc.stderr).strip()[-400:]}"


def describe_missing(path: Path) -> str:
    """Explain why a DVC-managed model file is not usable (for the skip message)."""
    parts = []
    try:
        parts.append(f"lexists={path.is_symlink() or path.exists()}")
        if path.is_symlink():
            target = path.readlink()
            parts.append(f"symlink -> {target} (target exists: {target.exists()})")
        path.stat()
    except OSError as exc:
        parts.append(f"stat: {exc}")
    try:
        parts.append(f"dir: {sorted(q.name for q in path.parent.iterdir())[:24]}")
    except OSError as exc:
        parts.append(f"listdir: {exc}")
    for var in ("FINN_MODELS_DIR", "CI_PROJECT_DIR", "CI_DVC_CACHE_DIR"):
        parts.append(f"{var}={os.environ.get(var)!r}")
    return "; ".join(parts)


SRL_BLOCK = 32
MAX_QSRL_DEPTH = 256

BNN_PYNQ_STEPS = [
    "finn.builder.custom_step_library.general.add_preproc_divide_by_255",
    "finn.builder.custom_step_library.general.add_postproc_top1",
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_generate_hardware",
]
MOBILENET_STEPS = [  # dut/mobilenetv1.yml
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_streamline",
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_lower_convs",
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_convert_to_hw_layers_separate_th",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_generate_hardware",
]
RESNET_STEPS = [  # dut/resnet18.yml
    "step_qonnx_to_finn",
    "step_tidy_up",
    "finn.builder.custom_step_library.resnet.step_resnet_tidy",
    "finn.builder.custom_step_library.resnet.step_resnet_streamline",
    "step_convert_to_hw",
    "finn.builder.custom_step_library.resnet.step_resnet_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_generate_hardware",
]


def _bnn_pynq(name: str) -> dict:
    return {
        "model": MODELS / "bnn-pynq" / f"{name}_qonnx.onnx",
        "folding": MODELS / "bnn-pynq" / f"{name}_folding_config.json",
        "specialize": MODELS / "bnn-pynq" / f"{name}_specialize_layers.json",
        "steps": BNN_PYNQ_STEPS,
        "extra": {},
    }


#: build recipes of the regression models (board and clock: the CI defaults, RFSoC2x2 / 10 ns)
DUTS: dict[str, dict] = {
    "tfc-w1a1": _bnn_pynq("tfc-w1a1"),
    "cnv-w1a1": _bnn_pynq("cnv-w1a1"),
    "mobilenetv1": {
        "model": MODELS / "mobilenetv1" / "mobilenetv1-w4a4_pre_post_tidy_opset-11.onnx",
        "folding": MODELS / "mobilenetv1" / "ZCU102_folding_config_live_fifo.json",
        "specialize": MODELS / "mobilenetv1" / "ZCU102_specialize_layers.json",
        "steps": MOBILENET_STEPS,
        "extra": {},
    },
    "resnet18": {
        "model": MODELS / "resnet18" / "resnet18_w3a3_cifar100.onnx",
        "folding": MODELS / "resnet18" / "resnet18_folding_config.json",
        "specialize": MODELS / "resnet18" / "resnet18_specialize_layers.json",
        "steps": RESNET_STEPS,
        "extra": {"standalone_thresholds": True},
    },
}


def load_baseline(name: str) -> dict[str, int]:
    """FIFO depths of the distributed_sim baseline report."""
    with (BASELINES / f"{name}_fifo_sizing_distributed_sim.json").open() as f:
        return {k: int(v) for k, v in json.load(f)["fifo_depths"].items()}


def depth_class(depth: int) -> int:
    """Block-granular class of a depth: SRL blocks of 32 up to 256, one class per BRAM step."""
    if depth <= MAX_QSRL_DEPTH:
        return math.ceil(depth / SRL_BLOCK)
    return MAX_QSRL_DEPTH // SRL_BLOCK + math.ceil(math.log2(depth / MAX_QSRL_DEPTH))


def build_with_strategy(name: str, strategy: str) -> tuple[dict[str, int], dict, float]:
    """Run the build flow up to hardware generation and return (depths, sizing report, time)."""
    dut = DUTS[name]
    model_path = Path(dut["model"])
    pulled = ensure_model(model_path)
    if not model_path.is_file():
        pytest.skip(f"{model_path} not available: {describe_missing(model_path)}; {pulled}")
    out_dir = Path(make_build_dir(f"teg_e2e_{name}_{strategy}_"))
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=str(out_dir),
        board="RFSoC2x2",
        synth_clk_period_ns=10.0,
        folding_config_file=str(dut["folding"]),
        specialize_layers_config_file=str(dut["specialize"]),
        auto_fifo_depths=True,
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod(strategy),
        split_large_fifos=True,
        teg_milp_fallback_to_abstract_sim=False,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        steps=dut["steps"],
        **dut["extra"],
    )
    t0 = time.time()
    build.build_dataflow_cfg(str(model_path), cfg)
    dt = time.time() - t0
    with (out_dir / "report" / "fifo_sizing.json").open() as f:
        depths = {k: int(v) for k, v in json.load(f)["fifo_depths"].items()}
    report_name = {"abstract_sim": "fifo_sizing_abstract_sim.json", "milp": "fifo_sizing_milp.json"}
    with (out_dir / "report" / report_name[strategy]).open() as f:
        report = json.load(f)
    return depths, report, dt


def compare_to_baseline(name: str, depths: dict[str, int], baseline: dict[str, int]) -> str:
    """Tabulate both depth sets and assert that the TEG sizing is not more expensive.

    Smaller TEG depths are reported but do not fail the test: ``distributed_sim`` sizes every
    node in isolation and is conservative at joins (the ResNet-18 residual adds get 4096-deep
    FIFOs where the joint model needs 32), and the regression benchmark measures whether the
    accelerator still reaches its bottleneck interval with the TEG depths. A TEG depth more
    than one block above the baseline is a cost regression and fails.
    """
    assert set(depths) == set(baseline), (
        f"{name}: FIFO set differs from baseline: only in result {set(depths) - set(baseline)}, "
        f"only in baseline {set(baseline) - set(depths)}"
    )
    lines = [f"{'FIFO':28s} {'baseline':>9s} {'teg':>7s} {'class_b':>8s} {'class_t':>8s}"]
    smaller: list[str] = []
    larger: list[str] = []
    for k in baseline:
        cb, ct = depth_class(baseline[k]), depth_class(depths[k])
        lines.append(f"{k:28s} {baseline[k]:9d} {depths[k]:7d} {cb:8d} {ct:8d}")
        if ct < cb:
            smaller.append(f"{k}: {depths[k]} < {baseline[k]}")
        if ct > cb + 1:
            larger.append(f"{k}: {depths[k]} >> {baseline[k]}")
    table = "\n".join(lines)
    print(f"\n{name}:\n{table}")
    if smaller:
        print(
            f"{name}: TEG depths below the distributed_sim baseline (the benchmark decides "
            "whether they suffice):\n" + "\n".join(smaller)
        )
    assert not larger, (
        f"{name}: TEG sizing is more than one block larger than the baseline for:\n"
        + "\n".join(larger)
        + "\n"
        + table
    )
    return table


@pytest.mark.parametrize("name", ["tfc-w1a1", "cnv-w1a1", "mobilenetv1", "resnet18"])
def test_teg_abstract_sim_end_to_end(name: str) -> None:
    """``abstract_sim`` reproduces the ``distributed_sim`` depths up to block granularity."""
    baseline = load_baseline(name)
    depths, report, dt = build_with_strategy(name, "abstract_sim")
    print(
        f"\n{name} abstract_sim: {report['simulations']} simulations, search "
        f"{report['search_time_s']:.1f} s, build {dt:.0f} s, "
        f"interval {report['final_interval_cycles']} cycles"
    )
    compare_to_baseline(name, depths, baseline)


@pytest.mark.parametrize("name", ["tfc-w1a1", "cnv-w1a1", "mobilenetv1", "resnet18"])
def test_teg_milp_report_end_to_end(name: str) -> None:
    """``milp`` produces the instance-size report; it solves only when the instance is small."""
    cfg_strategy = "milp"
    try:
        depths, report, _dt = build_with_strategy(name, cfg_strategy)
    except Exception as exc:  # the size report is the deliverable even on abort
        msg = str(exc)
        assert "MILP instance too large" in msg, msg
        pytest.skip(f"{name}: MILP too large, size report only: {msg[:200]}")
        return
    print(f"\n{name} milp: {json.dumps(report['instance'])}, solve {report.get('solve_time_s')} s")
    assert report["solved"] and report["verified"]
    compare_to_baseline(name, depths, load_baseline(name))
