"""FINN+ in the loop for neural architecture search (NAS).

The central entry point is :class:`~finn.nas.oracle.FinnOracle`, which takes a candidate
network (a Brevitas/PyTorch module or an exported QONNX file), runs the cheap, analytical part
of the FINN+ build flow on it and reports whether the network is FINN-compatible together with
resource, cycle and throughput estimates. No Vivado/HLS invocation is involved.

This package must stay importable without ``torch``: only :mod:`finn.nas.export` imports
torch/brevitas, and only lazily inside the export function.
"""

from finn.nas.config import FoldingSpec, OracleConfig
from finn.nas.oracle import FinnOracle
from finn.nas.result import EvalResult, FailureCategory

__all__ = ["EvalResult", "FailureCategory", "FinnOracle", "FoldingSpec", "OracleConfig"]
