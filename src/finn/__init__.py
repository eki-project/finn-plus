############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""
FINN+: A Framework for Fast, Scalable Quantized Neural Network Inference

This package provides tools for building and deploying quantized neural networks on FPGAs.
"""

import os
from pathlib import Path


def _setup_environment():
    """Configure FINN environment variables on import."""
    # 1. Configure LD_LIBRARY_PATH for Xilinx tools
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths_to_add = []

    # Vivado libraries
    if vivado_path := os.environ.get("XILINX_VIVADO"):
        if (vivado_lib := Path(vivado_path) / "lib" / "lnx64.o").exists():
            paths_to_add.append(str(vivado_lib))
        if (system_lib := Path("/lib/x86_64-linux-gnu")).exists():
            paths_to_add.append(str(system_lib))

    # Vitis FPO libraries
    if vitis_path := os.environ.get("VITIS_PATH"):
        if (vitis_fpo := Path(vitis_path) / "lnx64" / "tools" / "fpo_v7_1").exists():
            paths_to_add.append(str(vitis_fpo))

    # Update LD_LIBRARY_PATH
    if paths_to_add:
        existing_paths = ld_library_path.split(":") if ld_library_path else []
        for path in paths_to_add:
            if path not in existing_paths:
                existing_paths.append(path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_paths)


# Configure environment on import
_setup_environment()
