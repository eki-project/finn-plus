#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	Build FINN XSI Python bindings
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

.PHONY: clean

#############################################################################
# Reproduction Build

default: xsi.so

# Python Binding
xsi.so: xsi_bind.cpp xsi_finn.cpp
	g++ --std=c++17 -Wall -O3 -shared -fPIC \
	$$(python3.10-config --includes) -I$(XILINX_VIVADO)/data/xsim/include \
	-Wl,-rpath=$(XILINX_VIVADO)/lib/lnx64.o,-rpath=/lib/x86_64-linux-gnu,-rpath=$(XILINX_VITIS)/lnx64/tools/fpo_v7_1 \
	-o$@ $^ -ldl -lrt


#############################################################################
# Cleanup Working Directory
clean:
	rm -rf xsi.so
