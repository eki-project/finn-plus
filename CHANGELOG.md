# Changelog

The latest current work-in-progress version resides in `dev`, with `main` containing the last stable release.

The changelog lists mostly user-facing changes. For more detailed information please check out the pull requests or the wiki.

Entries marked with `(Xilinx)` are features pulled from AMD's upstream dev branch of FINN.

## Unreleased

### Added
- **FIFO sizing on a formal model of the accelerator**: two new `auto_fifo_strategy` options built on a timed-event-graph (TEG) model of the operator control paths instead of RTL simulation
    - `abstract_sim`: SimFIFO's per-FIFO minimisation (same search, orders and block granularity as `distributed_sim`, factored into `finn.transformation.fpgadataflow.fifo_depth_search`) with a pure-Python discrete-event simulation of the model as the oracle; no synthesis, no XSI, no board at sizing time. Sizes tfc/cnv in seconds to a few minutes and reproduces the `distributed_sim` depths of the CI regression builds
    - `milp`: exact minimum-cost sizing as a mixed-integer linear program (periodic-schedule certificate, HiGHS via SciPy); always writes an instance-size report `fifo_sizing_milp.json`, solves only below `teg_milp_max_constraints` and verifies every solution with the simulator (`teg_milp_time_limit`, `teg_milp_fallback_to_abstract_sim`)
    - New package `finn.analysis.fpgadataflow.teg`: model (`Chain`, `FIFOEdge`, `Arc`, freeze groups for HLS pipelines), simulator, MILP, search, `from_onnx` model construction and operator templates transcribed from the RTL/HLS sources (Thresholding RTL/HLS, MVAU RTL/HLS, VVAU HLS, StreamingDataWidthConverter RTL/HLS, ConvolutionInputGenerator RTL default template incl. depthwise and imperfect strides and the parallel template of 1x1 kernels, FMPadding RTL, DuplicateStreams/ReplicateStream HLS, all `Elementwise*` HLS binary operations incl. broadcasting, Pool/GlobalAccPool/LabelSelect HLS, Reshape RTL); HLS stage latencies, register slices, dataflow processes and rewind delays are read from the Vitis synthesis reports and schedule database (`teg.hls_report`)
    - The simulator has a native backend (`teg.native`, C++ compiled on first use with the system compiler like the XSI shim, ~30-50x faster than the Python event loop with identical results); selected with the `teg_sim_backend` build option or the `FINN_TEG_SIM_BACKEND` environment variable (`auto`/`python`/`native`)
    - Build options `teg_minimization_orders`, `teg_max_frames`, `teg_target_interval`, `teg_dump_model`, `teg_sim_backend`
    - Every model of the CI regression suite builds a complete model: Squeeze/Unsqueeze/Requant/Lookup HLS (1:1 loops), StreamingSplit/StreamingConcat HLS (one token per top-level interval: the generated top function is not pipelined, 5 cycles per token) and `ScaledDotProductAttention_hls` (the dataflow region as nine processes with its internal FIFOs and the per-frame restart; validated up to a few cycles per query row, see the xfail note of its test) were added for the transformers. The new pipeline variable `BENCH_PARAM_OVERRIDES` (JSON object applied to every run of a bench job, e.g. `{"auto_fifo_depths": true, "auto_fifo_strategy": "abstract_sim"}`) runs an existing bench config such as `regression_small`/`regression_large` with the new sizing method; the result collection compares such runs against the standard `ci_known_good` reference of the same DUT (only the two FIFO-sizing parameters may differ) and enforces the comparison like a standard build
    - Differential tests (`tests/fpgadataflow/teg`, marker `fifo_model`) drive every operator template and the real RTL in XSI with identical randomised stall patterns and require identical handshake traces; end-to-end tests compare against recorded `distributed_sim` reports of the CI regression models (tfc, cnv, MobileNet-v1, ResNet-18). The CI test suite variant `fifo_model_ci` (`TEST_SUITE` pipeline variable, `finn test --variant fifo_model_ci`) runs only these tests and skips the benchmark jobs
- **Empirical QoR estimation**: regression models fitted on the CI microbenchmark database predict post-synthesis LUTs and power per layer
    - *Corresponding poster @ FPT'25: ["Empirical QoR Estimation Flow for Fast Design Space Exploration of DNN Dataflow Accelerators"](https://doi.org/10.1109/ICFPT67023.2025.00044)*
    - New package `finn.qor` (database loading, model fitting/selection, evaluation) and analysis passes in `finn.analysis.fpgadataflow.empirical_qor_estimation`
    - `step_generate_estimate_reports` writes `estimate_layer_resources_empirical.json` and `estimate_power_empirical.json` if `FINN_QOR_MODEL_DIR` points to fitted models
    - CI scripts `ci/qor/fit_estimators.py` (refit models whenever microbenchmark results are added) and `ci/qor/end2end_report.py` (figures/tables comparing estimates with measured results)
    - See [QoR README](src/finn/qor/README.md); new core dependencies `scikit-learn` and `matplotlib`
- `verify_nodewise_report` build option: the `folded_hls_cppsim` and `node_by_node_rtlsim` verification steps also execute the folded graph with the Python implementation of every layer and report the first node whose simulated output deviates
- Multi-pass dataset validation in the Pynq driver (`passes` kwarg of `validate`): per-sample predictions are saved next to the report and samples with differing predictions between passes are re-run and listed
- The driver's unit tests (`driver/tests`) run on the board at the start of every CI measurement

### Fixed
- Multi-FPGA partitioning with Gurobi failed sporadically with `Gurobi environment could not be loaded, check your license` when the floating license token server was momentarily out of tokens or unreachable, which happened regularly in the highly parallel CI test suite; every partitioner kept its Gurobi environment (and thereby its token) alive until Python's cyclic garbage collector eventually got to it, which let a single test suite job hold several hundred tokens; the environment is now released as soon as the partitioner is dropped or closed, creating it is retried with an exponential backoff (number of attempts tunable through `FINN_MIP_SOLVER_MAX_ATTEMPTS`) and the final error reports Gurobi's actual reason instead of python-mip's generic license hint
- The merged CI test report classified the retried attempt of a test as passed and thereby hid the final failure of tests that failed again on their rerun
- The CMake build of the distributed RTL simulation backend (FIFO sizing, performance simulation) downloaded nlohmann/json from GitHub at configure time, so a network hiccup in the middle of a long build job failed it; the sources are now a regular external dependency (`nlohmann-json` in `external_dependencies.yaml`) fetched by `finn deps update` at the start of a job and cached in the dependency directory
- ImageNet validation in the Pynq driver could count a first image in place of a last one at the end of a pass, making the reported top-1 accuracy vary by single images between runs of the same bitfile
- Runtime-writable weights were never written on the CI board because the driver looked for `runtime_weights/` relative to the working directory and skipped the load silently when it was missing; the weight directory is now resolved next to `settings.json` and a missing directory is an error for accelerators with runtime-writable weights
- The CIFAR-100 validation in the Pynq driver fed raw pixel values into the float input of the ResNet-18 accelerator; the CIFAR validator now normalizes the inputs (`normalize`, `norm_mean`, `norm_std` kwargs of `validate`)

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
