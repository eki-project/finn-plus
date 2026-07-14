#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	SimEngine abstraction for running FINN task in simulated hardware.
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################
"""Simulation engine utilities for FINN XSI-based hardware runs."""

import numpy as np

# provided via pybind11
import xsi
from collections.abc import Generator, Iterator
from numpy._typing._array_like import NDArray
from typing import Literal


class SimEngine:
    """SimEngine abstraction for orchestrating XSI simulation tasks."""

    # ------------------------------------------------------------------------
    # Classes
    class Watchdog:
        """Track simulation cycles and signal when a timeout is reached."""

        def __init__(self, name: str, timeout: int) -> None:
            """Create a watchdog with a label and timeout in cycles."""
            self.name = name
            self.ticks = 0
            self.timeout = timeout

        def __bool__(self) -> bool:
            """Return True while the watchdog has not timed out."""
            return self.ticks < self.timeout

        def __repr__(self) -> str:
            """Return the watchdog name for debugging."""
            return self.name

        def __call__(self) -> None:
            """Advance the watchdog by one tick."""
            self.ticks += 1

        def reset(self) -> None:
            """Reset the watchdog tick counter."""
            self.ticks = 0

    # ------------------------------------------------------------------------
    # Life Cycle
    def __init__(
        self, kernel: str, design: str, log: str | None = None, wdb: str | None = None
    ) -> None:
        """Create a simulation engine bound to the given kernel and design."""
        k = xsi.Kernel(kernel)
        top = xsi.Design(k, design, log, wdb)
        clk = top.getPort("ap_clk")
        # If clock pumping is disabled, set clk2x to None
        try:
            clk2x = top.getPort("ap_clk2x")
        except RuntimeError:
            clk2x = None
        for p in top.ports():
            if p.isInput():
                p.clear().write_back()

        def cycle(updates: dict[xsi.Port, str]) -> None:
            """Perform one clock cycle with the given port updates."""
            # Rising Edge
            top.run(1)
            clk.set(1).write_back()
            if clk2x is not None:
                clk2x.set(1).write_back()
            # Updates after Active Edge
            top.run(1)
            for port, update in updates.items():
                port.set_hexstr(update).write_back()

            # Edges inactive on interface & finish Cycle
            if clk2x is None:
                top.run(4999)
                clk.set(0).write_back()
                top.run(4999)
            else:
                top.run(2499)
                clk2x.set(0).write_back()
                top.run(2500)
                clk.set(0).write_back()
                clk2x.set(1).write_back()
                top.run(2500)
                clk2x.set(0).write_back()
                top.run(2499)

        self.top = top
        self.cycle = cycle
        self.ticks = 0
        self.tasks: list[
            SimEngine.Reset
            | SimEngine.InputStreamer
            | SimEngine.OutputCollector
            | SimEngine.StreamTracer
            | SimEngine.AxiLiteWriter
            | SimEngine.AxiLiteReader
            | SimEngine.AximmRoImage
            | SimEngine.AximmQueue
        ] = []
        self.watchdogs: list[SimEngine.Watchdog] = []

    # ------------------------------------------------------------------------
    # Utility
    def get_bus_port(self, bus: str, suffix: str) -> "xsi.Port":
        """Return a port by bus name and suffix, trying lower/upper variants."""
        try:
            port = self.top.getPort(bus + "_" + suffix.lower())
        except RuntimeError:
            port = None
        return port if port is not None else self.top.getPort(bus + "_" + suffix.upper())

    # ------------------------------------------------------------------------
    # Simulation Setup

    # Task Scheduling
    def enlist(
        self,
        task: "SimEngine.Reset | SimEngine.InputStreamer | SimEngine.OutputCollector | SimEngine.StreamTracer | SimEngine.AxiLiteWriter | SimEngine.AxiLiteReader | SimEngine.AximmRoImage | SimEngine.AximmQueue",  # noqa
    ) -> None:
        """Register a task to be driven by the simulation loop."""
        self.tasks.append(task)

    # Watchdog Generation
    def create_watchdog(self, name: str, timeout: int) -> "SimEngine.Watchdog":
        """Create and register a watchdog with the given timeout."""
        ret = SimEngine.Watchdog(name, timeout)
        self.watchdogs.append(ret)
        return ret

    def remove_watchdog(self, watchdog: "SimEngine.Watchdog") -> None:
        """Remove a previously registered watchdog."""
        self.watchdogs.remove(watchdog)

    # ------------------------------------------------------------------------
    # Execution
    def run(self, cycles: int | None = None) -> list[Watchdog]:
        """Run all tasks to completion or until a watchdog triggers."""
        timeout = None if cycles is None else self.create_watchdog("Run Timeout", cycles)

        woken = []
        while len(self.tasks) > 0 and len(woken := [w for w in self.watchdogs if not w]) == 0:
            # Process Tasks and Collect Updates to Write Back
            tasks = []
            updates = {}

            # Execute Cycle
            self.ticks += 1
            strong = False
            for task in self.tasks:
                # Tasks read signals and derive updates to schedule for after the clock cycle
                ret = task(self)
                if ret is not None:
                    updates.update(ret)
                    tasks.append(task)
                    strong |= bool(task)
            self.cycle(updates)

            # Step Watchdogs
            for watchdog in self.watchdogs:
                watchdog()

            # Update to Unfinished Tasks
            self.tasks = tasks if strong else []

        # Return List of Woken Watchdogs
        if timeout is not None:
            self.remove_watchdog(timeout)
        return woken

    # ------------------------------------------------------------------------
    # Standard Tasks
    class Reset:
        """Drive the reset signal for a fixed number of cycles."""

        def __init__(self, top: "xsi.Design") -> None:
            """Bind to the design reset port."""
            self.cnt = 0
            self.rst_n: xsi.Port = top.getPort("ap_rst_n")

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Return port updates to perform the reset sequence."""
            cnt = self.cnt
            self.cnt = cnt + 1

            if cnt == 0:
                return {self.rst_n: "0"}
            if cnt < 16:
                return {}
            if cnt == 16:
                return {self.rst_n: "1"}
            return None

    def do_reset(self) -> None:
        """Schedule a reset sequence."""
        self.enlist(SimEngine.Reset(self.top))

    class InputStreamer:
        """Drive an AXI-Stream input from an iterator of values."""

        def __init__(
            self, top: "SimEngine", istream: str, values: Generator[str], throttle: tuple
        ) -> None:
            """Bind to the stream ports and configure throttling."""
            self.vld: xsi.Port = top.get_bus_port(istream, "TVALID")
            self.rdy: xsi.Port = top.get_bus_port(istream, "TREADY")
            self.dat: xsi.Port = top.get_bus_port(istream, "TDATA")
            self.values = values

            self.throttle = throttle
            self.await_tick = 0
            self.count_txns = throttle[0]

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:
            """Advance one cycle of input streaming."""
            vld = self.vld.as_bool()
            if vld and not self.rdy.read().as_bool():
                return {}

            # Track Transaction Count
            if vld:
                self.count_txns += 1

            # Proceed according to Throttling Rate
            if self.count_txns < self.throttle[0] or not sim.ticks < self.await_tick:
                # Try Feed
                val = next(self.values, None)
                if val is None:
                    # Unset vld, then exit
                    return {self.vld: "0", self.dat: "0"} if vld else None

                # Feed next Value
                ret = {self.dat: val}
                if not vld:
                    ret[self.vld] = "1"
                if self.count_txns == self.throttle[0]:
                    self.count_txns = 0
                    self.await_tick = sim.ticks + self.throttle[1]
                return ret

            # Stall Feed
            return {self.vld: "0", self.dat: "0"} if vld else {}

    def stream_input(
        self,
        istream: str,
        values: Generator[str],
        throttle: tuple[float, float] = (float("inf"), 0),
    ) -> None:
        """Stream all values from the passed iterator into the specified stream."""
        self.enlist(SimEngine.InputStreamer(self, istream, values, throttle))

    class OutputCollector:
        """Collect a fixed number of AXI-Stream output values."""

        def __init__(
            self, top: "SimEngine", ostream: str, size: int, watchdog: "SimEngine.Watchdog | None"
        ) -> None:
            """Bind to the stream ports and prepare a buffer."""
            self.size = size
            self.vld = top.get_bus_port(ostream, "TVALID")
            self.rdy = top.get_bus_port(ostream, "TREADY")
            self.dat = top.get_bus_port(ostream, "TDATA")
            self.buf: list[str] = []
            self.watchdog = watchdog

        def __iter__(self) -> Iterator[str]:
            """Iterate over collected output values."""
            return iter(self.buf)

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Advance one cycle of output collection."""
            if self.rdy.as_bool():
                if self.vld.read().as_bool():
                    # Have a n Output Transaction
                    if self.watchdog is not None:
                        self.watchdog.reset()
                    val = self.dat.read().as_hexstr()
                    self.buf.append(val)
                    if len(self.buf) == self.size:
                        return {self.rdy: "0"}
                return {}

            if len(self.buf) < self.size:
                return {self.rdy: "1"}
            return None

    def collect_output(
        self, ostream: str, size: int, watchdog: "SimEngine.Watchdog | None" = None
    ) -> "SimEngine.OutputCollector":
        """Collect size outputs from the specified stream into the returned iterable buffer."""
        ret = SimEngine.OutputCollector(self, ostream, size, watchdog)
        self.enlist(ret)
        return ret

    class StreamTracer:
        """Trace AXI-Stream activity as a string of 0/1 tokens."""

        def __init__(self, sim: "SimEngine", stream: str) -> None:
            """Bind to the stream ports to trace handshakes."""
            self.vld = sim.get_bus_port(stream, "TVALID")
            self.rdy = sim.get_bus_port(stream, "TREADY")
            self.trace = ""

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Advance one cycle of trace collection."""
            self.trace += "1" if self.vld.read().as_bool() and self.rdy.read().as_bool() else "0"
            return {}

        def __bool__(self) -> Literal[False]:
            """Report false to keep the task alive."""
            return False

        def __str__(self) -> str:
            """Return the collected trace string."""
            return self.trace

    def trace_stream(self, stream: str) -> "SimEngine.StreamTracer":
        """Monitor an AXI-Stream and trace its transaction activity."""
        ret = SimEngine.StreamTracer(self, stream)
        self.enlist(ret)
        return ret

    class AxiLiteWriter:
        """Drive AXI-Lite writes from a list of address/value pairs."""

        INIT = 0
        FEED = 1
        COOL = 2

        def __init__(
            self, top: "SimEngine", m_axilite: str, writes: Iterator[tuple[int, str]]
        ) -> None:
            """Bind to AXI-Lite write channels and store the write iterator."""
            self.awready = top.get_bus_port(m_axilite, "awready")
            self.awvalid = top.get_bus_port(m_axilite, "awvalid")
            self.awaddr = top.get_bus_port(m_axilite, "awaddr")
            self.wready = top.get_bus_port(m_axilite, "wready")
            self.wvalid = top.get_bus_port(m_axilite, "wvalid")
            self.wdata = top.get_bus_port(m_axilite, "wdata")
            wstrb = top.get_bus_port(m_axilite, "wstrb")
            wstrb.set_binstr("1" * wstrb.width()).write_back()
            self.bready = top.get_bus_port(m_axilite, "bready")
            self.bvalid = top.get_bus_port(m_axilite, "bvalid")
            self.bresp = top.get_bus_port(m_axilite, "bresp")
            self.writes = writes
            self.state = self.INIT
            self.pending = 0

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Advance one cycle of AXI-Lite write transactions."""
            # Termination
            if self.state == self.COOL and not self.bready.as_bool():
                return None

            ret = {}

            # Always Monitor Completions
            if self.state == self.INIT:
                ret[self.bready] = "1"
                self.state = self.FEED

            if self.bvalid.read().as_bool():
                if self.pending < 1:
                    print("Received spurious completion on", self.bresp.name())
                else:
                    self.pending -= 1
                    if self.pending == 0 and self.state == self.COOL:
                        ret[self.bready] = "0"

                if self.bresp.read().as_unsigned() != 0:
                    print("Received error indication on", self.bresp.name())

            # Transaction Feed
            if self.state == self.FEED:
                step = True

                # Check for busy address feed
                avld = self.awvalid.as_bool()
                aclr = False
                if avld:
                    if self.awready.read().as_bool():
                        aclr = True
                    else:
                        step = False

                # Check for busy data feed
                wvld = self.wvalid.as_bool()
                wclr = False
                if wvld:
                    if self.wready.read().as_bool():
                        wclr = True
                    else:
                        step = False

                # Proceed with next Write
                if step:
                    item = next(self.writes, None)
                    if item is not None:
                        addr, val = item
                        ret[self.awaddr] = f"{addr:x}"
                        ret[self.wdata] = val
                        if not avld:
                            ret[self.awvalid] = "1"
                        if not wvld:
                            ret[self.wvalid] = "1"
                        self.pending += 1
                        return ret
                    if not self.pending:
                        ret[self.bready] = "0"
                    self.state = self.COOL

                # Deassert completed feed
                if aclr:
                    ret[self.awvalid] = "0"
                if wclr:
                    ret[self.wvalid] = "0"

            return ret

    class AxiLiteReader:
        """Collect AXI-Lite reads for a list of addresses."""

        def __init__(self, top: "SimEngine", m_axilite: str, reads: Iterator[int]) -> None:
            """Bind to AXI-Lite read channels and store the address iterator."""
            self.arready = top.get_bus_port(m_axilite, "arready")
            self.arvalid = top.get_bus_port(m_axilite, "arvalid")
            self.araddr = top.get_bus_port(m_axilite, "araddr")
            self.rready = top.get_bus_port(m_axilite, "rready")
            self.rvalid = top.get_bus_port(m_axilite, "rvalid")
            self.rdata = top.get_bus_port(m_axilite, "rdata")
            self.reads = reads
            self.pending = []
            self.draining = False
            self.replies: dict[xsi.Port, str] = {}

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Advance one cycle of AXI-Lite read transactions."""
            ret = {}

            # Address Stream Feed: assert self.draining when done
            if not self.draining and (self.arready.read().as_bool() or not self.arvalid.as_bool()):
                addr = next(self.reads, None)
                if addr is None:
                    ret[self.arvalid] = "0"
                    self.draining = True
                else:
                    ret[self.arvalid] = "1"
                    ret[self.araddr] = f"{addr:x}"
                    self.pending.append(addr)

            # Reply Collection
            if not self.rready.as_bool():
                # Termination
                if self.draining:
                    return None
                # Activation
                ret[self.rready] = "1"
            elif self.rvalid.read().as_bool():
                assert len(self.pending) > 0, "Spurious reply."
                self.replies[self.pending.pop(0)] = self.rdata.read().as_hexstr()
                if self.draining and len(self.pending) == 0:
                    ret[self.rready] = "0"

            return ret

        def __iter__(self) -> Iterator[xsi.Port]:
            """Iterate over completed read replies."""
            return iter(self.replies)

        def __getitem__(self, addr: xsi.Port) -> str:
            """Return the reply value for a specific address."""
            return self.replies[addr]

    def write_axilite(self, m_axilite: str, writes: Iterator[tuple[int, str]]) -> None:
        """Execute writes specified as a list of (addr, val)-tuples to AXI-lite interface."""
        self.enlist(SimEngine.AxiLiteWriter(self, m_axilite, writes))

    def read_axilite(self, m_axilite: str, reads: Iterator[int]) -> "SimEngine.AxiLiteReader":
        """Execute reads specified as a list of addresses from AXI-lite interface."""
        ret = SimEngine.AxiLiteReader(self, m_axilite, reads)
        self.enlist(ret)
        return ret

    class AximmRoImage:
        """Serve a read-only AXI memory image from a byte buffer."""

        def __init__(
            self, top: "SimEngine", mm_axi: "str", base: int, img: NDArray[np.uint8]
        ) -> None:
            """Bind to AXI memory ports and stage the image data."""
            self.mm_axi = mm_axi
            self.rd_count = 0
            # Tie off Write Channels
            for tie_off in ("awready", "wready", "bvalid"):
                port = top.get_bus_port(mm_axi, tie_off)
                if port is not None:
                    port.set(0).write_back()

            # Collect Ports of Read Channels
            self.arready = top.get_bus_port(mm_axi, "arready")
            self.arvalid = top.get_bus_port(mm_axi, "arvalid")
            self.araddr = top.get_bus_port(mm_axi, "araddr")
            self.arlen = top.get_bus_port(mm_axi, "arlen")
            self.arburst = top.get_bus_port(mm_axi, "arburst")
            self.arsize = top.get_bus_port(mm_axi, "arsize")
            self.rready = top.get_bus_port(mm_axi, "rready")
            self.rvalid = top.get_bus_port(mm_axi, "rvalid")
            self.rdata = top.get_bus_port(mm_axi, "rdata")
            self.rresp = top.get_bus_port(mm_axi, "rresp")
            self.rlast = top.get_bus_port(mm_axi, "rlast")

            self.arready.set(1).write_back()
            self.rvalid.set(0).write_back()
            self.rresp.set(0).write_back()

            # Hold on to Image
            self.base = base
            self.img = [f"{_:02x}" for _ in np.array(img).astype(np.uint8)]
            # This is a hack to account for the minimum DMA burst read size of 32 bytes.
            for _i in range(32):
                self.img.append("00")  # Pad to 32 bytes
            self.queue = []

        def __bool__(self) -> Literal[False]:
            """Report false to keep the task alive."""
            return False

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Advance one cycle of read-only memory servicing."""
            ret = {}

            # Push out Read Replies
            if self.rready.read().as_bool() or not self.rvalid.as_bool():
                if len(self.queue) > 0:
                    # Work on Head of Queue
                    addr, length, size = self.queue.pop(0)
                    data = ""
                    for _i in range(size):
                        data = self.img[addr] + data
                        addr += 1
                    ret[self.rdata] = data

                    if length > 1:
                        self.queue.insert(0, (addr, length - 1, size))
                        ret[self.rlast] = "0"
                    else:
                        ret[self.rlast] = "1"
                    ret[self.rvalid] = "1"

                elif self.rvalid.as_bool():
                    # Silent Reply Interface
                    ret[self.rvalid] = "0"

            # Queue up newly received Read Requests
            if self.arvalid.read().as_bool():
                assert self.arburst.read().as_unsigned() == 1, "Only INCR bursts supported."

                addr = int(self.araddr.read().as_hexstr(), 16)
                # addr = addr - 8*self.rd_count
                # self.rd_count = self.rd_count + 2

                assert self.base <= addr, "Read address out of range."
                addr -= self.base

                length = 1 + self.arlen.read().as_unsigned()
                size = 2 ** self.arsize.read().as_unsigned()
                if addr + (length * size) > len(
                    self.img
                ):  # account for minimum dma burst read size of 32 bytes
                    print(f"Range extends beyond range {addr=} {length=} {size=}")
                    # assert addr + length * size < len(self.img), "Read extends beyond range."

                self.queue.append((addr, length, size))

            return ret

    def aximm_ro_image(
        self, mm_axi: "str", base: int, img: NDArray[np.uint8]
    ) -> "SimEngine.AximmRoImage":
        """Register a read-only AXI memory image task."""
        ret = SimEngine.AximmRoImage(self, mm_axi, base, img)
        self.enlist(ret)
        return ret

    class AximmQueue:
        """Queue AXI-MM writes and replay them on reads."""

        def __init__(self, top: "SimEngine", mm_axi: "str") -> None:
            """Bind to AXI-MM channels and initialize queues."""
            # Collect Ports of Read Channels
            self.awready = top.get_bus_port(mm_axi, "awready")
            self.awvalid = top.get_bus_port(mm_axi, "awvalid")
            self.awaddr = top.get_bus_port(mm_axi, "awaddr")
            self.awlen = top.get_bus_port(mm_axi, "awlen")
            self.awburst = top.get_bus_port(mm_axi, "awburst")
            self.awsize = top.get_bus_port(mm_axi, "awsize")
            self.wready = top.get_bus_port(mm_axi, "wready")
            self.wvalid = top.get_bus_port(mm_axi, "wvalid")
            self.wdata = top.get_bus_port(mm_axi, "wdata")
            self.wlast = top.get_bus_port(mm_axi, "wlast")
            self.bready = top.get_bus_port(mm_axi, "bready")
            self.bvalid = top.get_bus_port(mm_axi, "bvalid")
            self.bresp = top.get_bus_port(mm_axi, "bresp")
            self.arready = top.get_bus_port(mm_axi, "arready")
            self.arvalid = top.get_bus_port(mm_axi, "arvalid")
            self.araddr = top.get_bus_port(mm_axi, "araddr")
            self.arlen = top.get_bus_port(mm_axi, "arlen")
            self.arburst = top.get_bus_port(mm_axi, "arburst")
            self.arsize = top.get_bus_port(mm_axi, "arsize")
            self.rready = top.get_bus_port(mm_axi, "rready")
            self.rvalid = top.get_bus_port(mm_axi, "rvalid")
            self.rdata = top.get_bus_port(mm_axi, "rdata")
            self.rresp = top.get_bus_port(mm_axi, "rresp")
            self.rlast = top.get_bus_port(mm_axi, "rlast")
            self.awready.set(1).write_back()
            self.wready.set(1).write_back()
            self.bvalid.set(0).write_back()
            self.bresp.set(0).write_back()
            self.arready.set(1).write_back()
            self.rvalid.set(0).write_back()
            self.rresp.set(0).write_back()

            # Hold on to Contents Map per transfer: addr -> data
            self.map = {}  # addr -> (data, size)

            # Queued transactions
            self.wa_queue = []  # Write Addresses (addr, len, size)
            self.wd_queue = []  # Write Data      (data)
            self.ra_queue = []  # Read Addresses  (addr, len, size)
            self.wr_completion_queue = []  # A queue to track the write completions

        def __bool__(self) -> Literal[False]:
            """Report false to keep the task alive."""
            return False

        def __call__(self, sim: "SimEngine") -> dict[xsi.Port, str] | None:  # noqa: ARG002
            """Advance one cycle of AXI-MM queue servicing."""
            ret = {}

            # Process Write Updates
            while len(self.wa_queue) > 0:
                addr, length, size = self.wa_queue.pop(0)
                while length > 0:
                    if len(self.wd_queue) > 0:
                        self.map[addr] = (self.wd_queue.pop(0), size)
                        addr += size
                        length -= 1
                        if length == 0:
                            self.wr_completion_queue.append((0, 1))
                    else:
                        self.wa_queue.insert(0, (addr, length, size))
                        break
                if len(self.wd_queue) == 0:
                    break

            # Push out Read Replies
            if self.rready.read().as_bool() or not self.rvalid.as_bool():
                if len(self.ra_queue) > 0:
                    # Work on Head of Queue
                    addr, length, size0 = self.ra_queue.pop(0)
                    assert addr in self.map, "Missing data entry"
                    data, size = self.map[addr]
                    assert size == size0, "Write and read size mismatch."
                    ret[self.rdata] = data
                    if length > 1:
                        self.ra_queue.insert(0, (addr + size, length - 1, size))
                        ret[self.rlast] = "0"
                    else:
                        ret[self.rlast] = "1"
                    ret[self.rvalid] = "1"
                elif self.rvalid.as_bool():
                    # Silent Reply Interface
                    ret[self.rvalid] = "0"

            # Process write completion queue items
            if len(self.wr_completion_queue) > 0:
                if self.bready.read().as_bool():
                    ret[self.bvalid] = "1"
                    _ = self.wr_completion_queue.pop(0)
            else:
                ret[self.bvalid] = "0"

            # Queue new Write Address Requests
            if self.awvalid.read().as_bool():
                assert self.awburst.read().as_unsigned() == 1, "Only INCR bursts supported."

                addr = int(self.awaddr.read().as_hexstr(), 16)
                length = 1 + self.awlen.read().as_unsigned()
                size = 2 ** self.awsize.read().as_unsigned()
                self.wa_queue.append((addr, length, size))

            # Queue received Write Data
            if self.wvalid.read().as_bool():
                self.wd_queue.append(self.wdata.read().as_hexstr())

            # Queue new Read Requests
            if self.arvalid.read().as_bool():
                assert self.arburst.read().as_unsigned() == 1, "Only INCR bursts supported."

                addr = int(self.araddr.read().as_hexstr(), 16)
                length = 1 + self.arlen.read().as_unsigned()
                size = 2 ** self.arsize.read().as_unsigned()
                self.ra_queue.append((addr, length, size))

            return ret

    def aximm_queue(self, mm_axi: "str") -> None:
        """Pick up all write requests to carry them over to complete
        a later read request with the same address and size."""
        self.enlist(SimEngine.AximmQueue(self, mm_axi))
