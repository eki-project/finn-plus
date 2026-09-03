# Changelog

The latest current work-in-progress version resides in `dev`, with `main` containing the last stable release.

The changelog lists mostly user-facing changes. For more detailed information please check out the pull requests or the wiki.

Entries marked with `(Xilinx)` are features pulled from AMD's upstream dev branch of FINN.

## Unreleased
Planned release: 1.5.0.

### Added
- Support for Python 3.14 (#233)
- Added distributed simulation based FIFO sizing and a new performance simulation (#187)
- Multi-FPGA inference support (#23)
    - Initial communication backend: [AuroraFlow](https://github.com/pc2/AuroraFlow) (new dependency)
    - See [MultiFPGA README](src/finn/transformation/fpgadataflow/multifpga/README.md) for usage and development information
- Error lines from Vivado logs are printed to console in case of failing synthesis runs (#190)
- Added a `CHANGELOG.md` file
- Added `CITATION.cff` file

### Changed
- Modular creation of linker files (New class: `VitisLinkConfig`) (formerly #27, now #23)
    - Linker config files are now changed using `Transformation`s as well
    - Linker config and runner scripts are now provided as Jinja2 templates
- New step: `step_prepare_synthesis` (#23)
    - (Vitis Alveo) Adds IODMAs, creates StreamingDataflowPartitions, stitches IPs, generates XOs.
    - (Zynq) Nothing changed.
    - New build path: `... -> step_prepare_synthesis -> step_synthesize_bitfile -> ...`
- Vivado Stitch Projects have names specifying the nodes they contain if there are 3 or fewer nodes in the project (#190)
- The dependency definition file can now be found at `src/finn/interface/external_dependencies.yaml` instead of the repository root (#23, #216)
- Split the monolithic `convert_to_hw_layers.py` into a `convert_to_hw` package with one file per operator for better maintainability (#220)
- Dependencies can now be cached even without commit hash or Last-Modified header (#242)
- CI caches dependencies using the dependency definition file as key (#242)
- The Pynq driver is now a standalone package: `finn-plus-driver` (#234)
- `onnx-passes` is now a Python package dependency instead of a FINN+ external dependency (#233, #199)
- If the `target_fps` cannot be met during folding, FINN+ prints a warning with details (#209)
- Added warnings for floating point operations in the graph (#123, #227)
- Failed tests in the CI are now immediately printed (#229)
- Set the default start method for multiprocessing from `fork` to `spawn` (#229)
    - Enabled `FINNSettings` to work correctly in Python 3.14
    - Prevents RuntimeErrors in CI/test
- Tests receive deterministic per-item seeds for RNG (#230)

#### Removed
- Removed old simulation based FIFO sizing, superseded by the new distributed simulation based sizing (#187)
- Removed Jupyter notebooks and subpackage in favor of an updated Wiki (#241)
- Removed deprecated ops (`AddStreams`, `ChannelwiseLinear`, `StreamingEltwise`) (#162, #225)
- Removed end2end tests (#223)

#### Fixes
- Git timeouts now display a timeout message instead of "Internal Exception" (#243)
- Fixed warnings raised during streamlining and added more descriptive details to reorder and absorb warnings (#207)
- Fixed that `wget` timeouts would crash FINN+, even if dependencies were only checked, not updated (#23, #208)
- Fixed cases in which folding would accidentally create streams wider than `mvau_wwidth_max` due to a missing PE check (#209)
- `MVAU_hls` now correctly checks that `AP_INT_MAX_W` is below 8192 (#209)

## 1.4.0 - 03.03.2026
### Added
- Reworked user interface, settings and dependency management (#118)
    - Various new CLI commands. Documentation can be found in PR #118 or the Wiki or by typing `finn --help`
    - Added new method to fetch custom dependencies (`external_dependencies.yaml`)
    - Added wizards to help setup FINN+s' settings and build flows
    - Added option to specify a model in the build flow config itself
    - `XILINX_LOCAL_USER_DATA=no` will now be set automatically, unless specified otherwise
- Updated Pynq driver (#100)
- Experimental addition of [ONNX Passes](https://github.com/iksnagreb/onnx-passes) (#116)
- Enable node rtlsim for Attention CustomOp (#167)
- (Xilinx) FP16 and fixed-point support for thresholding and elementwise ops (Xilinx#1422, Xilinx#1444, Xilinx#1445)
- (Xilinx) Support for multiple weight sets for the memstreamer component (Xilinx#1441, Xilinx#1443)
- (Xilinx) Support for QONNX' new operator versioning scheme, specifically Trunc v2 (Xilinx#1468, Xilinx#1480)
- (Xilinx) New HLS Softmax operator (Xilinx#1439)
- (Xilinx) New HLS Crop operator (Xilinx#1501)
- (Xilinx) New RTL + HLS LayerNorm operators (Xilinx#1498, Xilinx#1506)
- (Xilinx) Support for Relu activation as elementwise operator (Xilinx#1479)


### Changed
- Build flow configs are not allowed to contain unknown keys anymore (#118)
- By default _all_ `DataflowOutputType` will be produced now (#118)
- Updated QONNX to version `1.0.0` and moved into project dependencies
- Moved Brevitas to project dependencies
- Improved dependency management (FINN+ should start quicker now)
- Improved Live-FIFO sizing (#158)
- Rework of Transformer example models and their build flows (#129, #160)
- Supporting different input and output shapes for DataWidthConverters (#163)
- `AddStreams`, `Channelwise_Op`, `DuplicateStream` and `StreamingEltwise` are _marked_ as deprecated. They will be deprectated in 1.5.0 (#166)
- (Xilinx) Generalized transpose and reshape support (Xilinx#1419)


### Deprecated
- Mostly deprecated use of environment variables in #118

### Removed
- Removed unused parts of `build_dataflow.py`

### Fixes
- Fix possibility to neither specify a folding config nor a target FPS (#118)
- Fixed wrong behaviour when specifying `output_dir: ~` in the build flow config
- Fixed that just-installed packages were not immediately available
- Fixed wrong transformation application which could cause large runtimes and unexpected ordering of the model graph (#147)
- Fixed `minimize_accumulator_width` failures that appeared due to floating point rounding errors (#153)
