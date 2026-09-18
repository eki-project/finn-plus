"""Benchmark DUT (design under test) definitions.

:data:`MICROBENCH_DUTS` lists the single-operator microbenchmarks (subclasses of
:class:`~finn.benchmarking.dut.microbench_base.MicrobenchDUT`) keyed by their ``NAME``,
which is also the ``dut`` parameter of benchmark configs and the operator name in the
microbenchmark database / QoR feature specs.
"""

from finn.benchmarking.dut.dwc import bench_dwc
from finn.benchmarking.dut.eltwise import bench_eltwise
from finn.benchmarking.dut.fifo import bench_fifo
from finn.benchmarking.dut.fmpadding import bench_fmpadding
from finn.benchmarking.dut.microbench_base import MicrobenchDUT
from finn.benchmarking.dut.mvau import bench_mvau
from finn.benchmarking.dut.pool import bench_pool
from finn.benchmarking.dut.swg import bench_swg
from finn.benchmarking.dut.thresholding import bench_thresholding
from finn.benchmarking.dut.vvau import bench_vvau

MICROBENCH_DUTS: dict[str, type[MicrobenchDUT]] = {
    cls.NAME: cls
    for cls in (
        bench_mvau,
        bench_thresholding,
        bench_swg,
        bench_vvau,
        bench_fifo,
        bench_dwc,
        bench_pool,
        bench_fmpadding,
        bench_eltwise,
    )
}
