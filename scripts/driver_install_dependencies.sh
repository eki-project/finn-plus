#!/bin/bash
# Install the FINN runtime driver and its dependencies on a PYNQ board.
#
# The driver is installed from the repository checkout rather than from PyPI, so that the CI
# runs against the driver version belonging to the commit under test. finn-plus-driver is a
# pure-Python package, so no cross-compilation is required for the board's arm64 architecture.
#
# Run this from the repository root inside the PYNQ virtual environment.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$script_dir")"

# Workaround for https://discuss.pynq.io/t/how-to-address-axilite-interface-in-pynq-v3-0/4831
# Installed first and pinned so the driver install below does not resolve a different version.
# Only actually required on boards with PYNQ images older than 3.1.1.
pip install pynqmetadata==0.1.5

# Installs the driver together with its dependencies (numpy, grpcio, bitstring, qonnx,
# click, matplotlib, h5py, pillow, finn-dataset-loading).
pip install "$repo_root/driver"
