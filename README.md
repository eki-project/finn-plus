<img src=https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/finn-plus_logo.png width=196/>

# Dataflow Compiler for Fast, Scalable Quantized Neural Network Inference on FPGAs

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/finn-plus?period=total&units=ABBREVIATION&left_color=GREY&right_color=GREEN&left_text=Downloads)](https://pepy.tech/projects/finn-plus) [![PyPI version](https://badge.fury.io/py/finn-plus.svg)](https://badge.fury.io/py/finn-plus)

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
