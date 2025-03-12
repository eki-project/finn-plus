<img src=https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/finn-plus_logo.png width=196/>

# Dataflow Compiler for Fast, Scalable Quantized Neural Network Inference on FPGAs

**FINN+** is a fork of **FINN**, an experimental framework from Integrated Communications and AI Lab of AMD Research & Advanced Development to explore deep neural network inference on FPGAs.
It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

**To get an overview of how FINN+ is used, take a look at the Getting Started section below!**

**For the time being, we refer to the original project [webpage](https://xilinx.github.io/finn/) for further information and documentation.**

## FINN+ Extensions
**FINN+** aims to incorporate all development from the upstream repository (dev branch) while extending **FINN** in all directions, including the following list of features that are either in progress or already completed:
- Transformer/Attention support
- Improved streamlining
- Improved automatic folding and FIFO-sizing
- Empirical quality-of-result (QoR) estimation
- Back-end extensions
    - Instrumentation for accurate performance profiling in simulation and on hardware
    - Improved Alveo build flow
    - Multi-FPGA support
    - Optimized C++ driver
- Quality-of-live improvements
    - Better logging and error handling
    - Type hinting/checking
    - Alternative YAML-based build configuration
    - Containerless setup

Please refer to our [**Feature Tracker**](https://github.com/orgs/eki-project/projects/1) for the current status of individual features.
While some items are already on-track to be merged into the upstream repository, we try to merge them into the **FINN+** dev branch as early as possible to increase development pace and drive our research forward.

## Getting Started
### Requirements
First of all, you will need [Poetry](https://python-poetry.org/), which we use for package-management and building the project. Simply follow the installation instructions and return.
Poetry will then manage everything Python related.

**If you have limited space, remember to set Poetry's [cache path](https://python-poetry.org/docs/configuration/#virtualenvspath) to somehwhere else!**

### First Time Setup
If you want to set everything up, simply run 
```
finn deps update
```
once. This will fetch all non-Python dependencies, boardfiles and a local install of `verilator`. The first time you run this command it can take a minute, since it downloads quite a lot of data. 
Afterwards, all dependencies will lay in the default FINN folder in `~/.finn/deps`. If you need them installed at another place or simply want multiple versions, you can specify the paths to the dependency folder for
almost all FINN+ commands. Simply use `--help` to display all arguments for any subcommand.

FINN requires some environment variables to be set. These can either be already in scope, defined in your environment config (default in `~/.finn/envvars.yaml`) or the defaults will be used. Priority is in this order. It is recommended though, that you do the one-time setup, and define everything
in your default environment config file. If you then need special settings, you can pass them on a per-run basis in the shell directly.

### Running existing FINN builds or other FINN-related scripts
FINN+ can be executed in two ways. First, you can let `finn` set up the needed environment and then simply execute any python script in this environment. This setup merely checks and sets environment variables and updates dependencies if necessary.
This is especially important for compatibility with FINN builds from the original repository - since the default way to run a flow there is to define a Python script that runs a `DataflowBuildConfig`. So to execute such a flow, simply run
```
finn run my_build_script.py
```

### Running new FINN builds
If you want to start a new flow or can easily convert an old one, we recommend using the new flow. Simply specify all arguments that you would otherwise specify in the build script in a YAML file. All parameters are exactly the same. Then, with your ONNX model ready, simply run
```
finn build my_config.yaml my_model.onnx
```
This will read the configuration, create a `DataflowBuildConfig` from it and run it with the given model. This has the advantage that it is much easier to manage parameter sweeps that can cover hundreds or thousands of runs. (_Support for execution of custom steps and operations is planned_).

## About Us
This repository is maintained by researchers from the [Computer Engineering Group](https://en.cs.uni-paderborn.de/ceg) (CEG) and [Paderborn Center for Parallel Computing](https://pc2.uni-paderborn.de/) (PC²) at Paderborn University, Germany as part of the [eki research project](https://www.eki-project.tech/).

<p align="left">
<a href="https://en.cs.uni-paderborn.de/ceg"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/UPB_Logo_ENG_coloured_RGB.jpg" alt="logo" style="margin-right: 20px" width="250"/></a>
<a href="https://pc2.uni-paderborn.de/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/PC2_logo.png" alt="logo" style="margin-right: 20px" width="250"/></a>
</p>

<p align="left">
<a href="https://www.eki-project.tech/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/eki-RGB-EN-s.png" alt="logo" style="margin-right: 20px" width="250"/></a>
<a href="https://www.bmuv.de/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/BMUV_Fz_2021_Office_Farbe_en.png" alt="logo" style="margin-right: 20px" width="250"/></a>
</p>
