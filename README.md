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
The main dependencies currently are:
- git
- Python >= 3.10 (<= 3.12)
- Poetry
- Vivado, Vitis, Vitis-HLS
- verilator (will be installed automatically)


#### Poetry
First of all, you will need [Poetry](https://python-poetry.org/), which we use for package-management and building the project. Simply follow the installation instructions and return.
Poetry will then manage everything Python related.

> [!WARNING]
> If you have limited space, remember to set Poetry's [cache path](https://python-poetry.org/docs/configuration/#virtualenvspath) to somehwhere else!

You can use Poetry for development, by using it's environment - simply install the environment, activate it and choose it's interpreter for your IDE's interpreter. The other option is to run the `poetry build` command to produce a wheel in the `dist/` directory. This you can then
install anywhere you like - this is more suited for usage of FINN rather than development. 

#### Synthesis Tools
FINN only runs until a certain step by itself. At latest when you want to do the IP Generation step, you will need some FPGA tooling. Specifically, you need `Vivado`, `Vitis` and `Vitis_HLS` (commonly only called HLS). We do _not_ install these for you - you need to take care of having a license, account etc. for yourself.

#### Verilator
To simulate designs and do FIFO-sizing, `verilator` is required. FINN takes care of retrieving the correct version itself. This version is stored in your FINN dependency directory and does not modify any existing `verilator` install.

#### Various Dependencies
Since FINN will only install the top-level dependencies such as `verilator` itself, you may need install other packages beforehand. Which packages are missing is depending on your machine, but we make an effort to provide a script that
can install the needed dependencies automatically. As always recommended when executing shell scripts, read the contents of `installDependencies.sh` first, and if you are alright with it, you can run it to install potentially missing packages.

### Quick Start
Clone FINN+ to a location of your choice, for example in `~`. Then:
```
cd finn-plus
poetry install
source <your-poetry-env>          # See Poetry docs for details 
finn deps update                  # Pull dependencies
finn config set <key> <value>     # Change config if necessary
finn run build.py                 # Run a standard FINN flow
# or
finn build build_config.yaml model.onnx    # Run a new YAML based FINN flow
```

### Installing FINN+
To install FINN+ for development, clone this repository to a location of your choice. Then do
```
poetry install
```
to get all Python-dependencies installed. Afterwards you can either run
```
poetry run python3 run_finn.py ...
```
or you simply source the Poetry environment (can be checked via `poetry env info`) and then just run 
```
finn ...
```
from anywhere. If you want to develop components used by FINN, you can modify `pyproject.toml` to install certain packages editable. For more information take a look at the Poetry documentation.

To install FINN+ for pure usage, clone the respository, then run
```
poetry install
poetry build
```
This will produce sdist and wheel files in `dist/`, which you can install into any Python environment.

### First Time Setup
> [!CAUTION]
> If you have limited space, before anything else run `finn config set DEFAULT_DEPS <your-path>` to set the default dependency path that FINN+ uses to a location with more space!
 
If you want to set everything up, simply run
```
finn deps update
```
once. This will fetch all non-Python dependencies, boardfiles and a local install of `verilator`. The first time you run this command it can take a minute, since it downloads quite a lot of data.
Afterwards, all dependencies will lay in the default FINN folder in `~/.finn/deps`. All your defaults can be modified. This is explained in more detail below.


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

> [!NOTE]
> If at any point you need an overview over all commands, simply run your (sub)command with the `--help` flag to get a short description of the command!

### Configuration
For most commands you can pass dependency or environment configuration paths manually. However you can also change the defaults that FINN+ assumes when no other option is specified. To get a list of all currently set configurations, simply run
```
finn config list
```
If you run this for the first time, it will error, since such a file does not yet exist. 
> [!NOTE]
> This will likely change quite a bit in the future.

As soon as you want to change any of the defaults, you can simply run
```
finn config set <key> <value>
```
This will set any value, regardless of whether the key is recognized by FINN+. This can be used if you ever want to change FINN+s behaviour. If no config file exists, you will be asked whether you want to create a new one.

To get the value of a specific key, simply run
```
finn config get <key>
```

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
