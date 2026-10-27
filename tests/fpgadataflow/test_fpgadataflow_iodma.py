"""C simulation of the generated IODMA_hls kernels (register and ring control, replication)."""

import pytest

import numpy as np
import os
import subprocess
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hls.iodma_hls import IODMA_hls
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import make_build_dir
from finn.util.settings import get_settings

FPGA_PART = "xcu55c-fsvh2892-2L-e"

# (memory word bits, stream word bits, frame bits): equal widths, memory wider than the stream,
# stream wider than memory, and a pair that needs two conversions through lcm(48, 32) = 96
WIDTHS = [(64, 64, 128), (512, 64, 4096), (64, 192, 192), (16, 80, 80), (32, 48, 96)]


def make_iodma(
    direction: str, intfw: int, streamw: int, frame_bits: int, replicas: int, mode: str
) -> ModelWrapper:
    """Return a single-node model with one IODMA of UINT8 frames."""
    nbytes = frame_bits // 8
    node = helper.make_node(
        "IODMA_hls",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        name="IODMA_hls_0",
        numInputVectors=[1],
        NumChannels=nbytes,
        dataType="UINT8",
        intfWidth=intfw,
        streamWidth=streamw,
        direction=direction,
        numReplicas=replicas,
        ctrlMode=mode,
        ringSlots=4,
    )
    graph = helper.make_graph(
        [node],
        "iodma",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, nbytes])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nbytes])],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="iodma-test"))


def make_testbench(
    direction: str, intfw: int, streamw: int, frame_bits: int, n: int, mode: str, frames: int
) -> str:
    """Return a C++ testbench that runs the generated top and checks every word.

    The memory holds `frames` frames; frame g travels through stream port g % n. Ring tests post
    their descriptors in reverse address order and with frames split unevenly over them.
    """
    fw = frame_bits // intfw
    sw = frame_bits // streamw
    ports = ", ".join(f"s[{p}]" for p in range(n))
    if mode == "ring":
        # descriptor d covers frames [first[d], first[d] + count[d]) of the frame sequence; the
        # frame sequence is stored back to front in memory, so the addresses go down
        counts = [1] * frames if frames <= 3 else [1, frames - 3, 2]
        ndesc = len(counts)
        firsts = list(np.cumsum([0, *counts[:-1]]))
        desc = "\n".join(
            f"    lo[{d % 4}] = (unsigned)((frames - {f0} - {c}) * FW * (IW / 8)); "
            f"hi[{d % 4}] = 0; fr[{d % 4}] = {c};"
            for d, (f0, c) in enumerate(zip(firsts, counts, strict=True))
        )
        if ndesc > 4:
            raise ValueError("the testbench posts all descriptors at once, ringSlots is 4")
        call_args = "mem_b, " + ports if direction == "in" else ports + ", mem_b"
        call = f"IODMA_hls_0({call_args}, lo, hi, fr, prod, cons);"
        pre = f"""#define IODMA_FOREVER false
#define IODMA_TEST_FRAMES {frames}
#define IODMA_TEST_DESC {ndesc}
"""
        setup = f"""    unsigned lo[4], hi[4], fr[4];
{desc}
    volatile unsigned prod = {ndesc}, cons = 0;
    hls::burst_maxi<ap_uint<IW>> mem_b(mem);
"""
        # frame g of the sequence sits at memory frame position frames - 1 - g ... within its
        # descriptor block the order is ascending, blocks are placed back to front
        pos = f"""    auto memFrame = [&](unsigned g) {{
        const unsigned first[] = {{{", ".join(str(int(x)) for x in firsts)}}};
        const unsigned count[] = {{{", ".join(str(c) for c in counts)}}};
        for (unsigned d = 0; d < {ndesc}; d++)
            if (g < first[d] + count[d]) return frames - first[d] - count[d] + (g - first[d]);
        return 0u;
    }};
"""
        post = f'    if (cons != {ndesc}) {{ std::printf("cons %u\\n", cons); return 1; }}\n'
    else:
        call_args = "mem, " + ports if direction == "in" else ports + ", mem"
        call = f"IODMA_hls_0({call_args}, frames);"
        pre, setup, post = "", "", ""
        pos = "    auto memFrame = [&](unsigned g) { return g; };\n"
    # bit b of sequence frame g, from the reference memory image
    common = f"""#include <cstdio>
#include <cstdlib>
{pre}#include "top_IODMA_hls_0.cpp"
constexpr unsigned IW = {intfw}, SW = {streamw}, FW = {fw}, SWF = {sw}, N = {n};
static ap_uint<IW> ref[{frames} * FW], mem[{frames} * FW];
int main() {{
    const unsigned frames = {frames};
    std::srand(1);
    for (unsigned i = 0; i < frames * FW; i++)
        for (unsigned b = 0; b < IW; b += 8) ref[i](b + 7, b) = std::rand() & 0xFF;
{pos}    auto bit = [&](unsigned g, unsigned b) {{
        return (unsigned)ref[memFrame(g) * FW + b / IW][b % IW];
    }};
    hls::stream<ap_uint<SW>> s[N];
"""
    if direction == "in":
        body = f"""    for (unsigned i = 0; i < frames * FW; i++) mem[i] = ref[i];
{setup}    {call}
{post}    unsigned seen[N] = {{}};
    for (unsigned g = 0; g < frames; g++) {{
        const unsigned p = g % N;
        for (unsigned k = 0; k < SWF; k++) {{
            if (s[p].empty()) {{
                std::printf("port %u empty at frame %u word %u\\n", p, g, k); return 1; }}
            ap_uint<SW> w = s[p].read();
            for (unsigned b = 0; b < SW; b++)
                if ((unsigned)w[b] != bit(g, k * SW + b)) {{
                    std::printf("frame %u word %u bit %u differs\\n", g, k, b); return 1; }}
        }}
        seen[p]++;
    }}
    for (unsigned p = 0; p < N; p++)
        if (!s[p].empty()) {{ std::printf("port %u has extra data\\n", p); return 1; }}
    std::printf("OK\\n");
    return 0;
}}
"""
    else:
        body = f"""    for (unsigned g = 0; g < frames; g++)
        for (unsigned k = 0; k < SWF; k++) {{
            ap_uint<SW> w;
            for (unsigned b = 0; b < SW; b++) w[b] = bit(g, k * SW + b);
            s[g % N].write(w);
        }}
{setup}    {call}
{post}    for (unsigned i = 0; i < frames * FW; i++)
        if (mem[i] != ref[i]) {{ std::printf("memory word %u differs\\n", i); return 1; }}
    for (unsigned p = 0; p < N; p++)
        if (!s[p].empty()) {{ std::printf("port %u not drained\\n", p); return 1; }}
    std::printf("OK\\n");
    return 0;
}}
"""
    return common + body


def run_csim(model: ModelWrapper, tb: str) -> tuple[int, str]:
    """Generate the IODMA code, compile and run it with the testbench; return exit code, output."""
    model = model.transform(PrepareIP(FPGA_PART, 5.0))
    inst = getCustomOp(model.graph.node[0])
    assert isinstance(inst, IODMA_hls)
    code_dir = Path(str(inst.get_nodeattr("code_gen_dir_ipgen")))
    (code_dir / "tb.cpp").write_text(tb)
    hls = os.environ["XILINX_HLS"]
    cmd = [
        "g++",
        "-std=c++17",
        "-O1",
        f"-I{code_dir}",
        f"-I{get_settings().finn_deps / 'finn-hlslib'}",
        f"-I{get_settings().finn_custom_hls}",
        f"-I{hls}/include",
        str(code_dir / "tb.cpp"),
        "-o",
        str(code_dir / "tb"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    res = subprocess.run([str(code_dir / "tb")], capture_output=True, text=True, timeout=600)
    return res.returncode, res.stdout + res.stderr


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("direction", ["in", "out"])
@pytest.mark.parametrize("mode", ["register", "ring"])
@pytest.mark.parametrize("replicas", [1, 2, 3])
@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("frames", [1, 2, 7])
def test_iodma_csim(
    direction: str, mode: str, replicas: int, widths: tuple[int, int, int], frames: int
) -> None:
    """Every word reaches the right port (in) or memory location (out), in frame order."""
    intfw, streamw, frame_bits = widths
    os.environ.setdefault("FINN_BUILD_DIR", str(make_build_dir("iodma_test_")))
    model = make_iodma(direction, intfw, streamw, frame_bits, replicas, mode)
    rc, out = run_csim(
        model, make_testbench(direction, intfw, streamw, frame_bits, replicas, mode, frames)
    )
    assert rc == 0 and "OK" in out.splitlines(), out
