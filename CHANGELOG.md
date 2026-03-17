# Changelog

The latest current work-in-progress version resides in `dev`, with `main` containing the last stable release.

The changelog lists mostly user-facing changes. For more detailed information please check out the pull requests or the wiki.

Entries marked with `(Xilinx)` are features pulled from AMD's upstream dev branch of FINN.

## Unreleased
Planned for 1.5.0.

**TODO**


## 1.4.0 - 03.03.2026
#### Added
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


#### Changed
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


#### Deprecated
- Mostly deprecated use of environment variables in #118

#### Removed
- Removed unused parts of `build_dataflow.py`

#### Fixes
- Fix possibility to neither specify a folding config nor a target FPS (#118)
- Fixed wrong behaviour when specifying `output_dir: ~` in the build flow config
- Fixed that just-installed packages were not immediately available
- Fixed wrong transformation application which could cause large runtimes and unexpected ordering of the model graph (#147)
- Fixed `minimize_accumulator_width` failures that appeared due to floating point rounding errors (#153)
