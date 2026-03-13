#!/bin/sh

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/vivadoprojgen.sh"

# compile and map through the steps of xilinx flow..
# my Ubuntu install doesn't like this
export LC_ALL=C
vivado -mode tcl -source $1.tcl -tclargs $1

cd ..
