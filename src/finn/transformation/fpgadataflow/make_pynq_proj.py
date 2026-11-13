############################################################################
# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Upstream names for the PYNQ (Zynq and embedded Versal) project transformations.

Upstream FINN renamed ``MakeZYNQProject``/``ZynqBuild`` to ``MakePYNQProject``/``PYNQBuild``
and moved them here when the transformations learned to build Versal PYNQ overlays. FINN+
keeps the implementation, with the FINN+ shell extensions (instrumentation, live FIFO
sizing, multi-DNN), in `finn.transformation.fpgadataflow.make_zynq_proj`, which builds
Versal overlays as well. This module only provides the upstream names.
"""

from finn.transformation.fpgadataflow.make_zynq_proj import MakeZYNQProject, ZynqBuild


class MakePYNQProject(MakeZYNQProject):
    """Upstream name of `MakeZYNQProject`."""


class PYNQBuild(ZynqBuild):
    """Upstream name of `ZynqBuild`."""
