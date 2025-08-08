#!/bin/zsh

source $FINN_QNN_DATA/vivado_scripts/vivadoprojgen.sh

# compile and map through the steps of xilinx flow..
# my Ubuntu install doesn't like this
export LC_ALL=C
vivado -mode tcl -source $1.tcl -tclargs $1

cd ..
