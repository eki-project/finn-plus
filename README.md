<img src=<https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/finn-plus_logo.png> width=196/>

# Dataflow Compiler for Fast, Scalable Quantized Neural Network Inference on FPGAs

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/finn-plus?period=total&units=ABBREVIATION&left_color=GREY&right_color=GREEN&left_text=Downloads)](https://pepy.tech/projects/finn-plus)
[![PyPI version](https://img.shields.io/pypi/v/finn-plus?logo=pypi&logoColor=white&color=brightgreen)](https://badge.fury.io/py/finn-plus)
[![GitHub license](https://img.shields.io/badge/License-BSD-purple.svg?logo=bsd)](https://raw.githubusercontent.com/eki-project/finn-plus/refs/heads/main/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/Documentation-Wiki-blue?logo=github)](https://github.com/eki-project/finn-plus/wiki)
![GitHub branch status](https://img.shields.io/github/checks-status/eki-project/finn-plus/main?label=CI&logo=gitlab&logoColor=white)
[![Go to Python website](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Feki-project%2Ffinn-plus%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=tool.poetry.dependencies.python&label=python&logo=python&logoColor=white)](https://python.org)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/eki-project/finn-plus?label=Pull%20Requests&color=green&logo=githubactions&logoColor=white)
![GitLab CI](https://img.shields.io/badge/GitLab%20CI-FC6D26?logo=gitlab&logoColor=fff)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
[![Go to QONNX website](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Feki-project%2Ffinn-plus%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=tool.poetry.dependencies.qonnx&label=QONNX&logo=onnx&logoColor=white&color=orange)](https://github.com/fastmachinelearning/qonnx)
[![Go to Brevitas website](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Feki-project%2Ffinn-plus%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=tool.poetry.dependencies.brevitas&logo=pytorch&logoColor=white&label=Brevitas&color=%23bd0000)](https://github.com/Xilinx/brevitas)

**FINN+** is a fork of **FINN**, an experimental framework from the Integrated Communications and AI Lab of AMD Research & Advanced Development to explore deep neural network inference on FPGAs.
It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

## Quick Links

- **[Getting Started](#getting-started)** - Start using FINN+ in minutes
- **[Wiki Documentation](https://github.com/eki-project/finn-plus/wiki)** - Complete documentation and guides
- **[Feature Tracker](https://github.com/orgs/eki-project/projects/1)** - Current development status

## What's New in FINN+

FINN+ incorporates all upstream FINN development while adding significant enhancements across multiple areas:

### Core Improvements

- **Transformer/Attention Support** - Native support for modern transformer architectures
- **Enhanced Streamlining** - Improved optimization pipeline for better performance
- **Smart FIFO Sizing (WIP)** - Automatic folding and FIFO-sizing with better algorithms
- **QoR Estimation (WIP)** - Empirical quality-of-result estimation for design space exploration

### Backend Extensions

- **Hardware Profiling** - Instrumentation for accurate performance measurement in simulation and hardware
- **Alveo Support** - Enhanced build flow for Xilinx Alveo cards
- **Multi-FPGA** - Support for distributed inference across multiple FPGAs
- **Optimized Drivers** - High-performance C++ drivers for better host-accelerator communication

### Developer Experience

- **Better Diagnostics** - Improved logging and error handling throughout the framework
- **Type Safety** - Comprehensive type hinting and checking for better code quality
- **YAML Configuration** - Alternative YAML-based build configuration system
- **Simplified Setup** - Containerless installation and setup process

**Track Development**: Check our [Feature Tracker](https://github.com/orgs/eki-project/projects/1) for real-time status updates on all features. We merge improvements early to accelerate development and enable cutting-edge research.

## Getting Started

This is a quick overview of how to get started, for additional information please refer to our [**Wiki**](https://github.com/eki-project/finn-plus/wiki)!

### Prerequisites

Before installing FINN+, ensure you have:

- **Python**: Version 3.10 or 3.11 (Python 3.12+ not yet supported)
- **Xilinx Tools**: Vivado, Vitis, and Vitis HLS (2022.2 or 2024.2)
- **System Dependencies**: See our [dependency installation script](installDependencies.sh) for required packages

### Installing via pip

After preparing the dependencies mentioned above, simply run the following to start a build flow:

```
# Make sure to create a fresh virtual environment for FINN+
pip install finn-plus                     # Install FINN+ and its Python dependencies via pip
finn deps update                          # Ensure FINN+ pulled all further dependencies (this might update packages in your venv!)
finn build build_config.yaml model.onnx   # Run a FINN+ build defined in a YAML file
```

For more detailed instructions, like installation for development use, please refer to our [**Wiki**](https://github.com/eki-project/finn-plus/wiki)!

> [!NOTE]
> Please note, that `finn deps update` (and most other commands) will automatically download and update dependencies required for FINN to work (mostly the same as the original FINN repository).
> This is done to provide a better user experience and to not require the user to manage a dozen dependencies on their own.
> If you want to know which dependencies will be installed before continuing, check out `external_dependencies.yaml`.

## About Us

FINN+ is maintained by researchers from the [Computer Engineering Group](https://en.cs.uni-paderborn.de/ceg) (CEG) and [Paderborn Center for Parallel Computing](https://pc2.uni-paderborn.de/) (PC²) at Paderborn University, Germany as part of the [eki research project](https://www.eki-project.tech/).

<p align="left">
<a href="https://en.cs.uni-paderborn.de/ceg"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/UPB_Logo_ENG_coloured_RGB.jpg" alt="logo" style="margin-right: 20px" width="250"/></a>
<a href="https://pc2.uni-paderborn.de/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/PC2_logo.png" alt="logo" style="margin-right: 20px" width="250"/></a>
</p>

<p align="left">
<a href="https://www.eki-project.tech/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/eki-RGB-EN-s.png" alt="logo" style="margin-right: 20px" width="250"/></a>
<a href="https://www.bmuv.de/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/BMUV_Fz_2021_Office_Farbe_en.png" alt="logo" style="margin-right: 20px" width="250"/></a>
</p>
