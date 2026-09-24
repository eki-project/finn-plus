# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Timed-event-graph (TEG) model of FINN dataflow accelerators for FIFO sizing.

The package models an accelerator with fixed FIFO depths as a timed marked graph at the
level of individual token transfers (see ``fifo_sizing_formal_model.md`` of the FIFO sizing
paper notes). Operators are described by *event chains* (``model.Chain``), FIFOs by
``model.FIFOEdge`` and non-FIFO intra-operator dependencies by index-mapped ``model.Arc``s.

* ``model``: the graph description shared by all analyses.
* ``patterns``: deterministic stall patterns for the environment (source / sink) chains.
* ``simulate``: event-driven self-timed execution (least fixed point) of a model.
* ``milp``: exact periodic-schedule MILP formulation, instance-size report and solver.
* ``search``: per-FIFO monotone minimisation (SimFIFO's algorithm) on the simulator.
* ``templates``: operator control-path templates instantiated from ONNX node attributes.
* ``from_onnx``: construction of a model from a FINN ``ModelWrapper``.
"""

from finn.analysis.fpgadataflow.teg.model import Arc, Chain, FIFOEdge, TEGModel
from finn.analysis.fpgadataflow.teg.patterns import StallPattern
from finn.analysis.fpgadataflow.teg.simulate import SimResult, evaluate, simulate

__all__ = [
    "Arc",
    "Chain",
    "FIFOEdge",
    "SimResult",
    "StallPattern",
    "TEGModel",
    "evaluate",
    "simulate",
]
