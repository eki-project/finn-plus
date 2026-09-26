# Changelog

The latest current work-in-progress version resides in `dev`, with `main` containing the last stable release.

The changelog lists mostly user-facing changes. For more detailed information please check out the pull requests or the wiki.

Entries marked with `(Xilinx)` are features pulled from AMD's upstream dev branch of FINN.

## 1.6.0 - 28.09.2026

### Added
- **Sync with upstream `dev` branch, including FINN v1.0.0-alpha** (Xilinx#1687, up to Xilinx#1703): pulls in all upstream `dev` changes since April 2026, see the sections below for the user-facing ones (eki-project#259)
- **Empirical QoR estimation**: regression models fitted on the CI microbenchmark database predict post-synthesis LUTs and power per layer (eki-project#260)
    - *Corresponding poster @ FPT'25: ["Empirical QoR Estimation Flow for Fast Design Space Exploration of DNN Dataflow Accelerators"](https://doi.org/10.1109/ICFPT67023.2025.00044)*
    - See [QoR README](src/finn/qor/README.md); this is the initial implementation, extensions will follow soon
- **Microbenchmarks for more operators**: single-operator microbenchmark DUTs (`finn.benchmarking.dut.*`, base class `MicrobenchDUT`) and QoR feature specs for Thresholding, ConvolutionInputGenerator (SWG), VVAU, StreamingFIFO, StreamingDataWidthConverter, Pool, FMPadding and elementwise Add/Mul in addition to the MVAU
- **Random sampling of microbenchmark configurations**: `mode: sample` config entries (and `finn bench --sample DUT:N`) draw new configurations from each DUT's parameter space and skip those already in the result database (`finn.benchmarking.sampling`, `finn.qor.params_key`); `ci/cfg/microbenchmark_sample_*.yml`, pipeline variables `SAMPLE_COUNT`/`SAMPLE_SEED`
- **Empirical DSP/BRAM/URAM estimation**: models for `metrics.synth.resources.{DSP,URAM}` and BRAM in 18K-block equivalents next to LUTs and power; zero-inflated targets are selected by MAE, only fitted with enough signal, and reported with MAE/zero hit rate in the fitting artifacts
- **Advanced QoR models** (`finn.qor.models`): log-target gradient boosting/MLP, an FT-Transformer style tabular transformer (torch) and PySR symbolic regression with Julia-free inference (sympy); model selection records fit/predict cost next to accuracy, `fit_estimators.py` gains `--regressors`/`--skip-regressors`/`--list-regressors`/`--subset`/`--set` and writes CV results, out-of-fold predictions and symbolic equations (Markdown/LaTeX/Pareto plot); new manual CI job `QoR Symbolic Regression`; new dependency `sympy`, optional extra `qor-symbolic` (`pysr`)
- **Benchmark artifact exchange via the cluster fileshare**: per-run reports and bitstreams are exchanged between the build, measurement and collection runners through `FINN_BENCH_EXCHANGE_DIR` (`finn.benchmarking.exchange`) instead of GitLab artifacts; new `Exchange Cleanup` CI job with `EXCHANGE_RETENTION_DAYS`/`KEEP_EXCHANGE_DEPLOY`
- `verify_nodewise_report` build option: the `folded_hls_cppsim` and `node_by_node_rtlsim` verification steps also execute the folded graph with the Python implementation of every layer and report the first node whose simulated output deviates (eki-project#263)
- Improved diagnostics in the PYNQ driver: multi-pass dataset validation (`passes` kwarg of `validate`) with per-sample predictions saved next to the report, and the driver's unit tests run on the board at the start of every CI measurement (eki-project#263)
- Build config option `large_fifo_mem_style` is back (`auto`/`block`/`distributed`/`ultra`): it pins the memory style of the FIFOs that `fifo.sv` would back with BRAM/URAM after sizing, since the RTL's `auto` takes URAM for every FIFO deeper than 2028 entries (eki-project#259)
- (Xilinx) New hardware operators: `PWPolyF` (piecewise-polynomial GELU/SiLU/Sigmoid/Tanh, RTL, Versal only) with a `PWPolyFunction` QONNX op and PyTorch export modules in `finn.util.torch_hw_modules` (Xilinx#1573), `HWWhere` (Xilinx#1579), `Pad1D` (1D padding / CLS token insertion) (Xilinx#1620), `SelectToken` and `Crop_rtl` (Xilinx#1639), `HWSoftmax_rtl` (Xilinx#1624)
    - New conversion transformations in the `convert_to_hw` package: `InferPWPolyFLayer`, `InferWhereLayer`, `InferPad1DLayer`, `InferSelectTokenLayer`
- (Xilinx) MLO: tiled RTL MVAU (`TH` attribute, Xilinx#1566, Xilinx#1594) and DDR weight streaming for `FINNLoop` (`mem_type`, `address_offset`, `AssignMemoryOffset`, new build step `step_assign_ddr_weight_offsets`, MLO weight export in the driver) (Xilinx#1607, Xilinx#1664)
- (Xilinx) `Requant_rtl` supports `mem_mode=internal_decoupled` with memstreamed scale and bias, also inside `FINNLoop` (Xilinx#1657)
- (Xilinx) URAM support and resource/efficiency estimation for `Thresholding_hls`, `Lookup` (new `ram_style` attribute), `ElementwiseBinary_hls` and the RTL sliding window generator (Xilinx#1586)
- (Xilinx) RTL elementwise operations now support int/float and int/int operand combinations (Xilinx#1570, Xilinx#1649)
- (Xilinx) `InnerShuffle` supports a fused reshape via the `transpose_in_shape` attribute (Xilinx#1649, Xilinx#1669)
- (Xilinx) New streamlining transformations `ExtractMultiThresholdScaleBias` (Xilinx#1567) and `MoveMulPastJoinMul` (Xilinx#1275); unsigned identity Quant nodes are converted to MultiThreshold (Xilinx#1653)
- (Xilinx) `AbsorbElementwiseOpsIntoRequant` absorbs scalar Mul/Add nodes into `Requant`; `InferRequantLayer` gained a `bitwidth_threshold` (build config `requant_bitwidth_threshold`, default 9) to prefer Requant over Thresholding for high-bitwidth activations (Xilinx#1569)
- (Xilinx) Build config: `inject_steps_before`/`inject_steps_after` to run custom steps around named steps (Xilinx#1591), `verify_rtlsim_behavioral` (behavioral models + FIFO gauge for rtlsim verification) and `debug_fifo` (per-FIFO transaction logs in `<output_dir>/debug/fifo_logs`) (Xilinx#1592)
- (Xilinx) rtlsim: watchdog timeouts are derived from the cycle estimate (`LIVENESS_THRESHOLD` only raises them), stitched-IP rtlsim of models with AXI-MM weight streaming loads the weight images automatically, `FINN_XELAB_MT` bounds the xelab thread count (Xilinx#1612, Xilinx#1614, Xilinx#1674)
- (Xilinx) `FINN_TOOL_DIR_OVERRIDE` redirects Xilinx tool invocations (`vivado`, `vitis_hls`, `vitis-run`, `xelab`) to a shim directory (Xilinx#1600)
- (Xilinx) `ApplyConfig` reports configurations for non-custom-op nodes instead of silently ignoring them (Xilinx#1593); `execute_onnx` validates input tensor names (Xilinx#1576)
- (Xilinx) Board support: `AUP-ZU3_8GB` (Xilinx#1659)

### Changed
- New build step `step_minimize_bit_width_initial` runs the bit width minimization once directly after the conversion to HW layers (default steps and benchmark DUT step lists), so that layer specialization and folding see minimized datatypes instead of the placeholders left by datatype inference (32 bit MAC results, 64 bit initializers from the ONNX passes frontend). Upstream's datatype-only first pass `step_minimize_bit_width_datatype_only` (Xilinx#1700) is available as well but not used by default: with the INT64 placeholder annotation of the onnx-passes frontend a datatype-only pass leaves the inflated widths in place (eki-project#259)
- `step_prepare_synthesis` is skipped when no bitfile is requested (in line with `step_synthesize_bitfile`) (eki-project#259)
- `InferRequantLayer` derives the output datatype of converted `Quant` nodes from the node attributes instead of the (possibly missing) tensor annotation (eki-project#259); it skips signed-output `MultiThreshold` nodes with `out_bias == 0` and points to `AbsorbScalarBiasIntoMultiThreshold` (Xilinx#1699)
- Improved stability and diagnostics of the test/regression CI infrastructure (eki-project#255, eki-project#259, eki-project#261, eki-project#267, eki-project#268, eki-project#272)
- (Xilinx) FIFO consolidation (Xilinx#1658, Xilinx#1679): all RTL FIFOs are built from a single `finn-rtllib/fifo/hdl/fifo.sv` (SRL / LUTRAM / BRAM / URAM selected by `ram_style`, new value `srl`); the Vivado `axis_data_fifo` implementation (`impl_style=vivado`), `SplitLargeFIFOs` and the `split_large_fifos` build config option are removed
- (Xilinx) Out-of-context synthesis now runs place & route inside the stitched-IP Vivado project (`CreateStitchedIP(run_synth, run_pnr)`); `step_out_of_context_synthesis` keeps its name and report (`ooc_synth_and_timing.json`) but reuses the stitched-IP project, the separate `vivadocompile` flow is gone. `CreateStitchedIP` takes `run_synth` instead of `vitis`. Power estimation opens the routed checkpoint. (Xilinx#1587)
- (Xilinx) `AbsorbSignBiasIntoMultiThreshold` is renamed to `AbsorbScalarBiasIntoMultiThreshold` (absorbs any scalar bias, bounded by `max_bitwidth_increase`); the old name remains as a deprecated alias (Xilinx#1173)
- (Xilinx) MVAU: the `dynamic_input` attribute is replaced by `mem_mode="dynamic"`, new memory mode `external_mem` for MLO weight streaming (Xilinx#1566)
- (Xilinx) `InsertFIFO` takes `ram_style` instead of `max_qsrl_depth`/`vivado_ram_style`; `LoopExtraction` takes the loop body template path (default: build directory) instead of writing into the working directory (Xilinx#1680)
- (Xilinx) `npy2apintstream`/`apintstream2npy` lost the element-bits template parameter and cnpy is vendored into `src/finn/templates/npy2stream` (no separate `cnpy` dependency anymore, cppsim compiles with C++17); `finnpy_to_packed_bytearray` lost the `fast_mode` argument and is much faster (Xilinx#1588, Xilinx#1625)
- (Xilinx) `finn.util.mlo_sim` moved to `finn.util.rtlsim`, `is_mlo` to `finn.util.fpgadataflow`; `finn-rtllib/mlo/fetch_weights*` moved to `finn-rtllib/fetch_weights/`; upstream `derive_characteristic` based FIFO sizing remains removed in FINN+ (eki-project#259)
- (Xilinx) finn-hlslib dependency bumped to `8d979e2b` (Xilinx#1588)
- (Xilinx) Node-by-node rtlsim verification is skipped for models mixing HLS floating-point ops with RTL LayerNorm (known xsim DSP conflict) (Xilinx#1564)
- (Xilinx) DWC consolidation (Xilinx#1660): the RTL DWC (`finn-rtllib/dwc/hdl/vpc.sv`) handles any input/output width ratio, so the integer-ratio restriction at specialization is gone
- (Xilinx) RTL sources are referenced from `finn-rtllib` in the stitched-IP project instead of being copied per node (Xilinx#1703)
- (Xilinx) RTL SWG index computations optimized for timing (Xilinx#1509); RTL elementwise broadcast memstreams compacted (Xilinx#1616)
- (Xilinx) `finn-rtllib/swg` is identical to upstream again: the depthwise-mode deadlock fix and counter width tightening of Xilinx#1698 supersede the FINN+ fix (eki-project#264)
- (Xilinx) RTL MVAU is selected for datatypes wider than 8 bit as well (Xilinx#1568), bounded by the DSP datapath widths of the target part (`get_dsp_datapath_limits`, Xilinx#1696)
- (Xilinx) Datatype minimization for more layers: `Lookup` minimizes its embedding datatype and `GlobalAccPool` its output datatype (Xilinx#1701); `Pool` minimizes its accumulator/output datatype (`AccPool`/`AvgPool`; `AccumBits` of `QuantAvgPool`) and passes the input datatype through for `MaxPool` when datatypes change after conversion (Xilinx#1697, Xilinx#1701, eki-project#259)

### Fixed
- ImageNet validation in the Pynq driver could count a first image in place of a last one at the end of a pass, making the reported top-1 accuracy vary by single images between runs of the same bitfile (eki-project#263)
- Runtime-writable weights were never written on the CI board because the driver looked for `runtime_weights/` relative to the working directory and skipped the load silently when it was missing; the weight directory is now resolved next to `settings.json` and a missing directory is an error for accelerators with runtime-writable weights (eki-project#255)
- The MLO (DDR) weight directory of the Pynq driver was resolved relative to the working directory as well; it is now taken from next to `settings.json` like `runtime_weights/`, and a missing directory reports what to pass instead of failing on the first `.dat` file (eki-project#259)
- The distributed (FIFO-sizing) simulation rebuilt each isolated node from every declared attribute type, materializing defaults such as `address_offset=0`; since the FINNLoop/MVAU DDR base-address logic is gated on the presence of that attribute, the loop IP generation inside the sizing build failed for every non-tiled MLO design. Only the attributes a node carries are copied now, and a failed loop IP generation quotes Vivado's errors (eki-project#271, eki-project#259)
- The per-node projects of the FIFO-sizing simulation synthesize with two Vivado jobs instead of one per FINN worker, which got parallel node builds OOM-killed (`CreateStitchedIP(synth_jobs=...)`); `Reshape_rtl` referenced the DWC source that upstream renamed to `vpc.sv` (eki-project#259)
- The RTL sliding window generator deadlocked on the last element of a feature map when its input arrived no faster than its window rate (eki-project#264, superseded by Xilinx#1698)
- The Pynq driver failed to unpack FLOAT outputs on the board (`packed_bytearray_to_finnpy_float` with reversed endianness) (eki-project#265)
- The CIFAR-100 validation in the Pynq driver fed raw pixel values into the float input of the ResNet-18 accelerator; the CIFAR validator now normalizes the inputs (`normalize`, `norm_mean`, `norm_std` kwargs of `validate`) (eki-project#255)
- (Xilinx) `OuterShuffle` cycle estimation works without Vivado (Xilinx#1688)
- (Xilinx) HLS `Requant` achieves II=1 again on Vitis HLS 2024.2 (Xilinx#1563)
- (Xilinx) `MoveAddPastMul` left a stale integer datatype annotation on the folded bias (Xilinx#1571)
- (Xilinx) `Thresholding_rtl` rejects unsorted thresholds at code generation (Xilinx#1583) and lays out the runtime-writable threshold file like the RTL address space, which mismatched for `numSteps < 2^outputBits` (Xilinx#1685)
- (Xilinx) MLO: stream width mismatch in the intermediate frame buffer when the first and last loop-body layers have different PE (Xilinx#1584), tiled MVAU fails at code generation instead of stalling in hardware when `TH` does not divide the tile (Xilinx#1606), FINNLoop adjacency guard for thresholding layers (Xilinx#1622), parameter stream numbering with decoupled `Requant_rtl` (Xilinx#1666)
- (Xilinx) `ElementwiseBinary_hls` cppsim over-read a broadcast operand on the last axis (Xilinx#1610)
- (Xilinx) Quant to MultiThreshold conversion accumulates thresholds in float64, so they no longer differ from the Quant node by one step at level boundaries (Xilinx#1635)
- (Xilinx) rtlsim: the performance report separates end-to-end latency from steady-state throughput (Xilinx#1632), NaN/Inf values in rtlsim inputs or outputs raise a clear error (Xilinx#1564)

### Removed
- Not pulled from upstream: the SLASH/V80 linker (`alveo_build.py`), phase-based build steps, `build_dataflow_checks` (including the folding-config/target-fps and verification-prerequisite checks of Xilinx#1665/Xilinx#1673), the Jenkins CI package (eki-project#259)
- The outdated `tutorials/fpga_flow` tutorial (still documented the Docker-based flow) (eki-project#259)

## 1.5.0 - 05.09.2026

### Added
- **New distributed simulation infrastructure** for search-based FIFO sizing and performance simulation (eki-project#187)
    - *To be presented as a poster @ FPL'26 and full paper @ H2RC (SC'26)*
- **Multi-FPGA inference support** (eki-project#23)
    - *Corresponding paper @ HEART'25: ["AuroraFlow, an Easy-to-Use, Low-Latency FPGA Communication Solution Demonstrated on Multi-FPGA Neural Network Inference"](https://doi.org/10.1145/3728179.3728190)*
    - Initial communication backend: [AuroraFlow](https://github.com/pc2/AuroraFlow) (new dependency)
    - See [MultiFPGA README](src/finn/transformation/fpgadataflow/multifpga/README.md) for usage and development information
- **Live-FIFO sizing improvements**: parallelized SDP creation, improved resilience against latency jitter (eki-project#237)
    - *Corresponding paper @ ARC'26: ["LiveFIFO: FPGA-in-the-Loop Buffer Sizing for Dataflow Accelerators"](https://doi.org/10.1007/978-3-032-29365-7_3)*
- **Experimental Multi-DNN support**: run several DNNs on one accelerator (eki-project#213)
    - Three modes, selected via the `Generation.mode` key of the multi-DNN config JSON (`multi_dnn_config_path`):
        - `Parallel`: the models run side by side, with their inputs/outputs optionally combined channelwise
        - `SelectableWeights`: the models share one datapath and are switched by swapping weights at runtime
        - `PartialReconfiguration`: the models are swapped at runtime via DFX, using a partial bitstream per model
    - New build steps: `step_apply_multi_dnn`, `step_collapse_multi_dnn` and `step_prepare_nodecontainer`
    - New custom ops: `DNNContainer` and `NodeContainer`
    - New RTL components for the DFX flow under `finn-rtllib/dfx/` (wrapper, scheduler, tUSER passthrough, decoupling) plus a `selector` and an ICAPE3 wrapper
    - Automatic DFX floorplanning (`dfx_auto_floorplanning.tcl`); PR region resource reports and an SVG floorplan diagram are written to the report directory
    - Partial bitstreams are copied into `<output_dir>/bitfile/partial_bitstreams`
    - The Pynq driver can drive multi-DNN accelerators, including DFX reconfiguration and tUSER-based round-robin scheduling
- Support for Python 3.14 (eki-project#233)
- Added ResNet-18 model support and build flows (eki-project#182)
- Added dataset validation to the Pynq driver and CI validation workflow (eki-project#173)
- Error lines from Vivado logs are printed to console in case of failing synthesis runs (eki-project#190)
- Added `CHANGELOG.md` and `CITATION.cff` files
- (Xilinx) **Multi-Layer Offload (MLO / FINNLoop)**: looping execution across layers (loop rolling, stream tapping with skid buffer, weight fetching, and intermediate frame buffering on HBM/DRAM) (Xilinx#1489, Xilinx#1415, Xilinx#1466, Xilinx#1559)
- (Xilinx) RTL and HLS integer Requantization operators (`Requant_rtl`, `Requant_hls`) with baked-in weights (Xilinx#1557)
- (Xilinx) Float2Int custom operator and conversion transformation (`InferQuantAsFloat2Int`) (Xilinx#1512, Xilinx#1523)
- (Xilinx) Support for FLOAT32 RTL elementwise operations (`ElementwiseBinary_rtl`) (Xilinx#1530, Xilinx#1545)
- (Xilinx) Node-by-node waveform saving during RTL simulation (Xilinx#1547)

### Changed
- Modular creation of linker files (New class: `VitisLinkConfig`) (formerly eki-project#27, now eki-project#23)
    - Linker config files are now changed using `Transformation`s as well
    - Linker config and runner scripts are now provided as Jinja2 templates
- New step: `step_prepare_synthesis` (eki-project#23)
    - (Vitis Alveo) Adds IODMAs, creates StreamingDataflowPartitions, stitches IPs, generates XOs.
    - (Zynq) Nothing changed.
    - New build path: `... -> step_prepare_synthesis -> step_synthesize_bitfile -> ...`
- Vivado Stitch Projects have names specifying the nodes they contain if there are 3 or fewer nodes in the project (eki-project#190, eki-project#222)
- The dependency definition file can now be found at `src/finn/interface/external_dependencies.yaml` instead of the repository root (eki-project#23, eki-project#216)
- Split the monolithic `convert_to_hw_layers.py` into a `convert_to_hw` package with one file per operator for better maintainability (eki-project#220)
- Dependencies can now be cached even without commit hash or Last-Modified header (eki-project#242)
- CI caches dependencies using the dependency definition file as key (eki-project#242)
- The Pynq driver is now a standalone package: `finn-plus-driver` (eki-project#234)
- `onnx-passes` is now a Python package dependency instead of a FINN+ external dependency (eki-project#233)
- If the `target_fps` cannot be met during folding, FINN+ prints a warning with details (eki-project#209)
- Added warnings for floating point operations in the graph (eki-project#123, eki-project#227)
- Failed tests in the CI are now immediately printed (eki-project#229)
- Set the default start method for multiprocessing from `fork` to `spawn` (eki-project#229)
- Tests receive deterministic per-item seeds for RNG (eki-project#230)
- Zynq builds now generate the accelerator clock with a PLL (Clocking Wizard) instead of the processing system, for exact and static clock generation
- The instrumentation core now supports `tLast` generation and simple `tUSER`-based scheduling, and computes the running average in software to reduce pipeline depth (eki-project#212)
- (Xilinx) LayerNorm parallelism scaling below N/SIMD = 12 (Xilinx#1534)
- (Xilinx) Generic `step_convert_to_hw` transformation application in default build flow (Xilinx#1542)
- (Xilinx) Optional skipping of the very first transpose node (NCHW -> NHWC) in build flow (Xilinx#1535)
- (Xilinx) Accurate expected cycle estimation for Shuffle operations accounting for subsequent decompositions (Xilinx#1537)
- (Xilinx) Optimized RAM style attributes for baked-in weights thresholding (Xilinx#1515)
- (Xilinx) Support for `DuplicateStreams` across additional graph patterns (Xilinx#1529, Xilinx#1532)

#### Removed
- Removed old "largefifo_rtlsim" and "characterize" FIFO sizing methods, superseded by the new distributed simulation based sizing (eki-project#187)
- Removed deprecated ops (`AddStreams`, `ChannelwiseLinear`, `StreamingEltwise`) (eki-project#162, eki-project#225)
- Removed Jupyter notebooks and subpackage in favor of an updated Wiki (eki-project#241)
- Removed end2end tests from the Pytest test suite in favor of full regression testing builds in our CI (eki-project#223)

#### Fixes
- Git timeouts now display a timeout message instead of "Internal Exception" (eki-project#243)
- Fixed incorrect clock frequency reporting in the Pynq driver
- Fixed warnings raised during streamlining and added more descriptive details to reorder and absorb warnings (eki-project#207)
- Fixed that `wget` timeouts would crash FINN+, even if dependencies were only checked, not updated (eki-project#23, eki-project#208)
- Fixed cases in which folding would accidentally create streams wider than `mvau_wwidth_max` due to a missing PE check (eki-project#209)
- `MVAU_hls` now correctly checks that `AP_INT_MAX_W` is below 8192 (eki-project#209)
- Fixed `LD_LIBRARY_PATH` not being set correctly for PyBind-based simulation (eki-project#224)
- Fixed segmentation faults in Vivado simulation infrastructure (eki-project#228)
- Fixed C++ driver deployment package generation, Vitis synthesis log handling, and post-synthesis report collection (eki-project#249)
- (Xilinx) Fixed threshold datatype minimization for HLS MVAU/VVAU and narrowing cast handling when WT < WI (Xilinx#1561, Xilinx#1553)
- (Xilinx) Fixed integer processing in `minimize_accumulator_width` (Xilinx#1524)

## 1.4.0 - 03.03.2026
### Added
- Reworked user interface, settings and dependency management (eki-project#118)
    - Various new CLI commands. Documentation can be found in PR eki-project#118 or the Wiki or by typing `finn --help`
    - Added new method to fetch custom dependencies (`external_dependencies.yaml`)
    - Added wizards to help setup FINN+s' settings and build flows
    - Added option to specify a model in the build flow config itself
    - `XILINX_LOCAL_USER_DATA=no` will now be set automatically, unless specified otherwise
- Updated Pynq driver (eki-project#100)
- Experimental addition of [ONNX Passes](https://github.com/iksnagreb/onnx-passes) (eki-project#116)
- Enable node rtlsim for Attention CustomOp (eki-project#167)
- (Xilinx) FP16 and fixed-point support for thresholding and elementwise ops (Xilinx#1422, Xilinx#1444, Xilinx#1445)
- (Xilinx) Support for multiple weight sets for the memstreamer component (Xilinx#1441, Xilinx#1443)
- (Xilinx) Support for QONNX' new operator versioning scheme, specifically Trunc v2 (Xilinx#1468, Xilinx#1480)
- (Xilinx) New HLS Softmax operator (Xilinx#1439)
- (Xilinx) New HLS Crop operator (Xilinx#1501)
- (Xilinx) New RTL + HLS LayerNorm operators (Xilinx#1498, Xilinx#1506)
- (Xilinx) Support for Relu activation as elementwise operator (Xilinx#1479)


### Changed
- Build flow configs are not allowed to contain unknown keys anymore (eki-project#118)
- By default _all_ `DataflowOutputType` will be produced now (eki-project#118)
- Updated QONNX to version `1.0.0` and moved into project dependencies
- Moved Brevitas to project dependencies
- Improved dependency management (FINN+ should start quicker now)
- Improved Live-FIFO sizing (eki-project#158)
- Rework of Transformer example models and their build flows (eki-project#129, eki-project#160)
- Supporting different input and output shapes for DataWidthConverters (eki-project#163)
- `AddStreams`, `Channelwise_Op`, `DuplicateStream` and `StreamingEltwise` are _marked_ as deprecated. They will be deprectated in 1.5.0 (eki-project#166)
- (Xilinx) Generalized transpose and reshape support (Xilinx#1419)


### Deprecated
- Mostly deprecated use of environment variables in eki-project#118

### Removed
- Removed unused parts of `build_dataflow.py`

### Fixes
- Fix possibility to neither specify a folding config nor a target FPS (eki-project#118)
- Fixed wrong behaviour when specifying `output_dir: ~` in the build flow config
- Fixed that just-installed packages were not immediately available
- Fixed wrong transformation application which could cause large runtimes and unexpected ordering of the model graph (eki-project#147)
- Fixed `minimize_accumulator_width` failures that appeared due to floating point rounding errors (eki-project#153)
