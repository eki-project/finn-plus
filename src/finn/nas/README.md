# FINN+ in the loop for NAS

`finn.nas` lets a neural architecture search (NAS) use FINN+ as a fast oracle: for each
candidate network it answers *"is this FINN-compatible?"* with a machine-readable reason and
returns the analytical resource, cycle and throughput estimates of the folded dataflow
accelerator. Only the cheap graph-transformation part of the build flow is executed (up to
`step_minimize_bit_width` plus the estimate analyses); no Vivado/HLS is involved.

```python
from finn.nas import FinnOracle, FoldingSpec, OracleConfig

oracle = FinnOracle(
    OracleConfig(
        board="Pynq-Z1",                       # or fpga_part="xc7z020clg400-1"
        synth_clk_period_ns=10.0,
        folding=FoldingSpec(target_fps=3000),  # SetFolding towards this throughput
        cache_path="oracle_cache.sqlite",      # memoize by architecture hash
        isolation="process",                   # spawn + timeout per candidate
        timeout_s=300,
    )
)
res = oracle.evaluate(brevitas_module, input_shape=(1, 3, 32, 32))
if res.feasible:
    print(res.resources_total, res.est_fps, res.device_fit)
else:
    print(res.failure_category, res.failure_stage, res.failure_op_type, res.failure_message)
```

## Command line

`finn nas eval` runs the oracle on an exported (Q)ONNX model inside the regular FINN+ CLI
environment:

```bash
finn nas eval model.onnx --board Pynq-Z1 --target-fps 3000 -o result.json
finn nas eval model.onnx --fpga-part xc7z020clg400-1 --folding-config folding.json --strict-codegen
```

`finn nas sample OUTPUT_DIR -n 1000 --board Pynq-Z1 --target-fps 100 --data-root ../datasets/cifar10`
draws random networks from the search space and evaluates them in parallel worker processes
(oracle verdict, estimates and zero-cost proxies per network, written as JSON lines; re-running
with a larger count extends the run).

`finn nas eval` prints a summary (compatibility, failure reason, resource table, throughput and
latency estimates) and optionally writes the complete `EvalResult` as JSON (`-o`). Exactly one of
`--target-fps` and `--folding-config` is required, as is one of `--board` and `--fpga-part`.
`--timeout SECONDS` evaluates in a separate, killable worker process. The exit code is 0 for a
compatible model, 1 for an incompatible one and 3 if the evaluation itself failed (timeout,
misconfiguration, internal error), so the command can gate scripts.

Experiment scripts that use the Python API should also be started through the CLI
(`finn run script.py`), so that the environment and external dependencies are set up.

## Result (`EvalResult`)

| Field | Meaning |
|---|---|
| `feasible` | all cheap build steps succeeded and the configured checks passed |
| `failure_category` | one of `FailureCategory` (see `result.py`); coarse on purpose so a search can learn from it |
| `failure_stage`, `failure_node`, `failure_op_type`, `failure_message` | where/what failed |
| `resources_total`, `resources_per_node` | `LUT`, `BRAM_18K`, `URAM`, `DSP` from `res_estimation` (no analytical `FF` model exists) |
| `cycles_per_node`, `max_cycles`, `critical_path_cycles`, `est_fps`, `est_latency_ns` | from `exp_cycles_per_layer` / `dataflow_performance` |
| `device_fit` | `total / (board capacity * max_utilization)` per resource, if `board` is known to `finn.util.platforms` |
| `folding` | PE/SIMD/ram_style/... per node as chosen by the folding policy |
| `throughput_target_missed`, `unmapped_boundary_nodes`, `warnings`, `timings_s` | diagnostics |

Results of category `TIMEOUT`, `INTERNAL_ERROR` and `CONFIG_ERROR` are never cached.

## How failures are classified

FINN+ signals incompatibility in different ways (typed `FINNUserError`, plain `Exception`
with a known message, bare `assert`, or a logged warning). `classify.py` maps them:

| Signal | Category |
|---|---|
| export raised | `EXPORT_FAILED` |
| `step_qonnx_to_finn` / `step_tidy_up` / `step_streamline` raised | `FRONTEND_FAILED` |
| unmapped node between HW layers of an op with no HW conversion (`Sin`, `LeakyRelu`, `AveragePool`, ...), or "has no hw implementation variant" | `UNSUPPORTED_OP` |
| a `Quant`/`Trunc`/`MultiThreshold` that FINN did not convert (bit width above `max_multithreshold_bit_width`, unsupported signedness, ...), unquantized weights, or non-integer data (float ReLU, missing input quantizer) feeding a layer | `DATATYPE_UNSUPPORTED` |
| a float scale/shift that streamlining could not absorb (`ElementwiseMul`/`Add` with a float constant: unabsorbed BatchNorm, residual branches with different or non-unit scales) in front of an unmapped layer or on an interior HW node, or a leftover `Transpose`/`Reshape` between HW layers | `STREAMLINE_INCOMPLETE` |
| other unmapped-layer / multi-partition errors | `UNMAPPED_LAYER` |
| `SetFolding` raised, "doesn't know how to handle op_type" (`fail_on_unhandled_op`), or a HW layer rejects its `PE`/`SIMD` after folding | `FOLDING_INFEASIBLE` |
| throughput target missed and `require_target_met` | `THROUGHPUT_TARGET_MISSED` |
| `strict_codegen` and `step_hw_codegen` raised an assertion or `FINNUserError` | `CODEGEN_ASSERT` |
| `device_fit` > 1 for any resource | `RESOURCE_BUDGET_EXCEEDED` |

When FINN lists several unmapped nodes, only those between HW layers are considered and the
most likely root cause among them is blamed: an unsupported op first, then an unconverted
quantizer, then the producer of non-integer data that starves a convertible layer (`MatMul`,
`Im2Col`, ...), then leftover streamlinable ops. The original FINN message is kept in
`failure_message` after the diagnosis.

Nodes at the graph boundary that FINN leaves to the host are listed in
`unmapped_boundary_nodes`. Pure layout ops there (the NCHW -> NHWC `Transpose` in front of
every CNN) are harmless. Anything else means the estimates do not cover the whole network:
a `QuantLinear`/`QuantConv2d` without an input quantizer runs its *entire first layer* on the
host and the oracle would still report `feasible=True` if `boundary_unmapped_is_failure` were
off. It is therefore on by default: the boundary nodes are classified like interior ones
(missing input quantizer -> `DATATYPE_UNSUPPORTED`, float op -> `UNSUPPORTED_OP`). Set it to
False only when host-side pre/post-processing is intended.

Feasibility is weakly weight-dependent: a freshly initialized BatchNorm is a uniform scale
that streamlining moves like a scalar, so `conv -> BN -> conv` passes with untrained
statistics and fails with per-channel ones. Candidates should be evaluated with BatchNorm
statistics that look trained (see `randomize_batchnorm` in `tests/nas/broken_models.py`).

### Strict code generation

Some folding constraints of the custom ops are only checked when HLS/RTL code is generated,
e.g. `MW % SIMD == 0` of the HLS matrix-vector unit or its `SIMD >= MW / 1024` requirement,
which `SetFolding` itself can violate for wide layers. `strict_codegen=True` runs
`step_hw_codegen` after the estimates to surface those as `CODEGEN_ASSERT` with the offending
node. Code generation only writes source files: it starts no Vivado/Vitis HLS process, needs
no licence and no external dependencies (the finn-hlslib path is merely written into the
generated Tcl script). It costs 10-20 ms for tiny networks and grows with the weight memory
that has to be written (about 0.4 s for a 2048x8 layer, 3 s for a 192-channel CNN). It is
recommended whenever `FoldingSpec(kind="fixed")` is used, since illegal `SIMD` values are not
caught by the estimate flow at all. It does not catch everything (e.g. `PE` of a pooling
layer not dividing the channel count passes).

## Broken candidate classes covered by the tests

`tests/nas/test_broken_candidates.py` evaluates one network per defect
(`tests/nas/broken_models.py`) and asserts the category, stage and blamed node:

| Defect | Category (stage) | Blamed node |
|---|---|---|
| `torch.sin`, `LeakyReLU` between layers | `UNSUPPORTED_OP` (partition) | the op |
| float `AvgPool2d` | `UNSUPPORTED_OP` (partition) | `AveragePool` |
| `nn.Upsample` (`Resize`) | `UNMAPPED_LAYER` (partition) | `Resize` |
| BatchNorm between two convs (trained statistics) | `STREAMLINE_INCOMPLETE` (partition) | `ElementwiseMul` (the BN) |
| residual add with different scales | `STREAMLINE_INCOMPLETE` (partition) | `ElementwiseMul` (branch scale) |
| residual add with equal, non-unit scales | `STREAMLINE_INCOMPLETE` (partition) | `ElementwiseMul` (scale at the fork) |
| float `nn.ReLU` | `DATATYPE_UNSUPPORTED` (partition) | `ElementwiseMax` (the ReLU) |
| float `nn.Linear` / `nn.Conv2d` | `DATATYPE_UNSUPPORTED` (partition) | the `MatMul` (unquantized weights) |
| 16-bit activation | `DATATYPE_UNSUPPORTED` (partition) | the `Quant` |
| 1-bit unsigned `QuantReLU` | `DATATYPE_UNSUPPORTED` (`step_convert_to_hw`) | the `MultiThreshold` |
| kernel larger than the input | `EXPORT_FAILED` (export) | - |
| grouped (non-depthwise) conv | `FRONTEND_FAILED` (`step_streamline`) | the `Conv` |
| no input quantizer, float op after the last layer | `DATATYPE_UNSUPPORTED` / `UNSUPPORTED_OP` (default); feasible with boundary nodes listed if `boundary_unmapped_is_failure=False` | the layer / the op |
| too wide for `board="Pynq-Z1"` | `RESOURCE_BUDGET_EXCEEDED` (estimate) | - |
| `SIMD` not dividing the layer width (fixed folding) | feasible without, `CODEGEN_ASSERT` with `strict_codegen` | the `MVAU_hls` |
| `PE` not dividing the layer height (fixed folding) | `FOLDING_INFEASIBLE` (`step_minimize_bit_width`) | the `MVAU_hls` |

Accepted although one might expect otherwise: two convs without an activation in between
(the second consumes the wide accumulator), overlapping `MaxPool2d(3, stride=2)`, a residual
add whose branches both have scale 1, and the untrained-BatchNorm case above.

## Package layout

- `oracle.py` – `FinnOracle.evaluate` / `evaluate_many`: export, hash, cache, dispatch, cleanup.
- `pipeline.py` – `run_pipeline`: the in-process step loop (`FRONTEND_STEPS`, i.e. the builder's
  estimate-only flow up to folding -> folding policy ->
  `step_minimize_bit_width` -> estimate analyses -> optional `step_hw_codegen`).
- `classify.py` – exception/warning -> `FailureCategory`, node and op type extraction.
- `folding.py` – `TargetFpsFolding` (FINN `SetFolding`) and `FixedFolding` (folding JSON).
- `worker.py` – spawn-isolated execution with timeout and stage heartbeat.
- `hashing.py` / `cache.py` – structural hash of the exported QONNX graph; SQLite or
  JSON-directory cache.
- `export.py` – Brevitas `export_qonnx`; the only module importing torch (lazily).
- `cli.py` – summary printing and exit codes behind `finn nas eval` (the command itself is
  defined in `finn.interface.run_finn`).
- `logcapture.py` – collects FINN log warnings during an evaluation.
- `space/` – the grammar-based search space (genotypes, shape inference, sampling, mutation,
  crossover, seeds, Brevitas module construction); see its README.
- `zc/` – zero-cost proxies vendored from NASLib and adapted to Brevitas networks (see its
  README for provenance, per-measure notes and the default measure list).
- `data.py` / `train.py` – CIFAR-10 as device tensors and a short QAT training loop with a
  robust default recipe, for proxy ground truth and for training selected candidates.

Tests: `tests/nas/`, run from the repository root with
`finn test --batch --variant custom --args "$PWD/tests/nas"`. They do not run synthesis and
need no Xilinx tools beyond what the `finn` CLI itself requires to start.
