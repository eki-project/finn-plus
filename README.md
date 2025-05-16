<img src=https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/finn-plus_logo.png width=196/>

# Dataflow Compiler for Fast, Scalable Quantized Neural Network Inference on FPGAs

**FINN+** is a fork of **FINN**, an experimental framework from the Integrated Communications and AI Lab of AMD Research & Advanced Development to explore deep neural network inference on FPGAs.
It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

**To get an overview of how FINN+ is used, take a look at the Getting Started section below!**

**While our [Wiki](https://github.com/eki-project/finn-plus/wiki) is under construction, we refer to the original [FINN homepage](https://xilinx.github.io/finn/) for further information.**

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
This is a quick overview of how to get started, for additional information please refer to our [**Wiki**](https://github.com/eki-project/finn-plus/wiki)!

### Requirements
The primary dependencies currently are:
- Python >= 3.10 (< 3.12)
- Vivado, Vitis, Vitis HLS (2022.2 or 2024.2)
- Some basic system-level packages, refer to the [**dependency installation script**](https://github.com/eki-project/finn-plus/blob/main/installDependencies.sh)

### Installing via pip
After preparing the dependencies mentioned above, simply run the following to start a build flow:
```
# Make sure to create a fresh virtual environment for FINN+
pip install finn-plus          # Install FINN+ and its Python dependencies via pip
finn deps update               # Ensure FINN+ pulled all further dependencies (this might update packages in your venv!)
finn build build_config.yaml   # Run a FINN+ build defined in a YAML file
```

### Installing from the repository
To install directly from the repository, you'll need Poetry (>= 2.0) for dependency management. After cloning the repo and setting up the system-level dependencies, run the following to start a build flow:
```
cd finn-plus
poetry install                 # Install Python packages into a Poetry-managed virtual environment
source <your-poetry-venv>      # Use "poetry env info" to find the path to your Poetry venv. For further information visit the Poetry documentation
finn config create             # Create a default configuration in ~/.finn/settings.yaml. Optional but recommended
finn deps update               # Ensure FINN+ pulled all further dependencies (this might update packages in your venv!)
finn build build_config.yaml   # Run a FINN+ build defined in a YAML file
```

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
