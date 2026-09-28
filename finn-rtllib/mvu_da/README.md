# Distributed-arithmetic MVU (`MVAU_rtl` with `mem_mode=internal_embedded`)

A pure-RTL implementation of the matrix-vector unit for layers whose weight
matrix is **constant and fully unrolled** (`SIMD = MW`, `PE = MH`). The weight
matrix is compiled by [alkaid](https://pypi.org/project/alkaid/) (formerly
`da4ml`, Sun et al., *da4ml: Distributed Arithmetic for Real-time Neural
Networks on FPGAs*, TRETS 2026) into a pipelined, multiplier-free adder graph:
canonical-signed-digit recoding, common-subexpression sharing across all
outputs, exact per-node bit widths and a depth-constrained tree construction.
FINN wraps that graph in its AXI-stream protocol; no HLS is involved, so the
build time of a layer drops from a Vitis HLS run to a few seconds of solver
time plus the Vivado synthesis of plain adders.

## When to use it

The core replaces the compute of one `MVAU` when all of the following hold:

| Requirement | Attribute / condition |
|---|---|
| RTL backend, embedded weights | `preferred_impl_style: rtl`, `mem_mode: internal_embedded` |
| Fully unrolled folding | `SIMD = MW`, `PE = MH` (checked at code generation, `SetFolding` does this automatically) |
| Standalone thresholds | `noActivation = 1` (use `standalone_thresholds: true` in the build config) |
| Static integer weights | initializer present, `runtime_writeable_weights = 0`, integer datatypes of at most 24 bits, no `binaryXnorMode` |
| LUT resources | `resType: lut` (or `auto`); `dsp` is rejected, the graph never contains multipliers |
| No tiling, pumping, MLO | `TH = 1`, `pumpedCompute = pumpedMemory = 0`, not a loop body with per-iteration weights |

Typical candidates are small latency-critical MLPs and the first/last layers of
small CNNs, i.e. every layer that reaches `WMEM = 1`. Reuse across input
vectors (`numInputVectors > 1`, convolutions after im2col) is fine: the core
accepts one input vector per clock cycle.

Layer selection is **opt-in**: `SpecializeLayers` picks the core only when the
node has `preferred_impl_style="rtl"` and `mem_mode="internal_embedded"`. With
an empty preference such a node stays an HLS MVAU, and an info message points
at the RTL alternative when the layer qualifies. Per-layer configuration in the
specialize/folding config files:

```json
"MVAU_0": {
    "preferred_impl_style": "rtl",
    "mem_mode": "internal_embedded",
    "resType": "lut",
    "SIMD": 64,
    "PE": 32
}
```

## Node attributes

| Attribute | Default | Meaning |
|---|---|---|
| `da_hard_dc` | 2 | depth constraint of the adder trees passed to the solver (`-1`: unconstrained, fewer adders but deeper trees) |
| `da_latency_cutoff` | 0 | per-stage latency budget in alkaid's surrogate units (roughly one unit per adder level); `0` derives it from the target clock period (`DA_LATENCY_CUTOFF_PER_NS` in `da_codegen.py`) |
| `da_ternary_fuse` | 1 | fuse adder pairs into three-input adders (alkaid's default flow) |
| `da_cost`, `da_adders`, `da_depth`, `da_latency_cycles` | – | written by the code generator: solver cost (active adder result bits), adder count, combinational depth and pipeline latency in cycles |

`lut_estimation` returns `da_cost * DA_LUT_PER_COST_BIT` plus one LUT per output
bit for the SRL output queue once the solver has run (`PrepareIP`, or `MVAU_rtl.prepare_da_solution(model, clk)`), before that a
coarse estimate of one LUT per weight and operand bit. `dsp_estimation` is 0.

## Files

| File | Role |
|---|---|
| `mvu_da_axi.sv` | shared flow-control bracket: `tready`, a valid shift register alongside the free-running core and an SRL output queue (same scheme as `mvu_vvu_axi.sv`, output queue depth `CORE_LATENCY + 2`, bound verified in `tb/`) |
| `mvu_da_axi_wrapper.v` | per-node wrapper template (`$...$` placeholders) instantiating the bracket, the generated core and the output-slot widening glue |
| `tb/mvu_da_axi_tb.sv` | self-checking testbench of the bracket with a fake fixed-latency core, random valid/ready patterns and a mid-stream reset (`xvlog -sv ../mvu_da_axi.sv mvu_da_axi_tb.sv && xelab mvu_da_axi_tb -s sim && xsim sim -R`) |
| `src/finn/custom_op/fpgadataflow/rtl/da_codegen.py` | alkaid driver: datatype mapping, solver, pipelining, Verilog generation, module-name prefixing, interface facts |

Generated per node into `code_gen_dir_ipgen`: `<node>_wrapper.v` (rendered
template), `<node>_da_core.v` (pipeline registers), `<node>_da_core_logicN.v`
(one combinational stage each), `<node>_da_core_wrapper.v` (alkaid io wrapper
with uniform per-element slots), `<node>_da_core_tables.v` (partial-sum LUTs,
1-bit inputs only) and prefixed copies of the alkaid primitives
(`<node>_da_core_shift_adder.v`, `..._ternary_adder.v`, `..._negative.v`, ...).
Every module name carries the node name, so several cores coexist in one
Vivado project. The primitive sources are library code of alkaid (LGPL-3.0),
copied at build time from the installed package.

## Datatype mapping

* Activations: FINN `INTn`/`UINTn` map to alkaid `(k, i, f) = (1, n-1, 0)` /
  `(0, n, 0)`. `BIPOLAR` activations travel as single bits `b` on FINN streams;
  the core computes `b @ (2W) - sum(W)`, so the bias is folded into the graph.
* Weights are used as stored in the initializer (`BIPOLAR` as ±1, `TERNARY`,
  `INTn`), the solver handles zeros, ±1 and powers of two for free.
* Outputs: alkaid derives the exact range of every output; the wrapper widens
  each slot to FINN's accumulator width (`accDataType`, sign- or zero-extended
  and shifted left when every output is a multiple of a power of two). The
  code generator asserts that the exact range fits the accumulator datatype.

## Verification

* `tests/fpgadataflow/test_da_codegen.py` (no Vivado): golden equivalence of
  the traced graph against `x @ W` for all datatype pairs incl. dead inputs,
  zero-width outputs and negative output fractions; interface facts; module
  prefixing; determinism; rejections.
* `tests/fpgadataflow/test_specialize_layers_mvau.py`: opt-in, rejections and
  the folding check at code generation time.
* `tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau_da`
  and `..._with_thresholding`: node-by-node and stitched-IP rtlsim against the
  ONNX reference, incl. two DA cores plus RTL thresholding in one design.
* `ci/cfg/microbenchmark_da_baseline.yml` / `microbenchmark_da.yml`: HLS vs. DA
  out-of-context synthesis sweeps for the calibration of the LUT estimate and
  the latency budget.
