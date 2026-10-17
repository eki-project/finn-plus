"""AXI-stream register slices at the inputs of dataflow nodes.

A FINN dataflow node answers the valid of the stream it receives with a ready
that is, for most node types, a combinational function of that valid and of the
node's state. The producer's next state depends on that ready, so every such
stream carries a combinational round trip: producer register -> consumer logic
-> producer register. That round trip crosses the distance between the two nodes
twice, so it is the stream path class distance breaks: on mobilenetv1 (U55C) a
FIFO 300 tiles from the window buffer it fed was 0.52 ns short at 200 MHz (15 Oct
2026).

An AXI-stream register slice at the consumer's input breaks the round trip: the
producer sees a registered ready, and the consumer's combinational ready reaches
only the slice. The slice is Vivado's auto-pipelined one: the placer inserts as
many register stages between producer and consumer as their distance needs (up to
24), with a credit buffer at the consumer, so throughput is unchanged and latency
grows by the stages inserted. Measured against a fully registered slice on
mobilenetv1, free placement, cores at the target clock (17 Oct 2026): 280.8 /
269.9 / 278.2 MHz against 259.9 / 260.5 / 270.9 at 300 / 350 / 400 MHz requested,
for 8 % more kernel LUTs and registers (the credit buffers are LUT RAM) and about 40
minutes more link time. The fully registered mode was removed on that measurement.

This module holds the rule of where a slice goes, used by `CreateStitchedIP`.
"""

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper

#: Vivado's AXI4-Stream register slice.
SLICE_VLNV = "xilinx.com:ip:axis_register_slice:1.1"

#: ``REG_CONFIG`` 16 is "Auto-pipelined" (choices of the IP in Vivado 2024.2): its
#: registers carry ``autopipeline_*`` attributes, limit 24 stages, and the placer
#: decides how many a stream gets.
SLICE_REG_CONFIG = 16


def slice_cell_name(node_name: str, input_index: int) -> str:
    """Name of the register slice in front of input ``input_index`` of a node."""
    return f"{node_name}_in{input_index}_slice"


def slice_reset_dont_touch_tcl() -> list[str]:
    """Tcl, for the synthesized kernel, that keeps the slices' own reset pipeline registers.

    The slice pipelines its reset in the auto-pipelined part: ``s_/m_areset_fwd_reg``
    in the ``fwd`` group, next to the payload, and ``s_/m_areset_resp_reg`` in the
    ``resp`` group. The placer inserts the same number of taps into every net of a
    group, and ``write_bitstream`` checks that (DRC NTCN-15). All slices share one
    reset, so all their reset registers at equal delay are equivalent, and the
    placer's equivalent driver rewiring (Physical Synthesis In Placer, after the taps
    go in) merges them across slices. On the vision transformer at 300 MHz (17 Oct
    2026) it grafted 49 tap chains onto another slice's ``resp`` register, so their
    groups counted a tap short: NTCN-15 on two slices, no bitstream.

    ``DONT_TOUCH`` on these four registers per slice takes them out of the rewiring.
    Replayed from that link's post-``opt_design`` netlist (18 Oct 2026): no chain
    ends on a foreign register any more (49 -> 0), NTCN-15 282 -> 0, the same 111
    groups get taps, routed WNS +0.007 against +0.016 ns. Taps of equal stage are
    still shared across slices, which the DRC accepts. ``DONT_TOUCH`` on the root
    ``areset_r_reg`` did not help (the rewiring follows the logic through it), nor
    did ``EQUIVALENT_DRIVER_OPT``; ``place_design -no_psip`` drops the taps
    altogether. The nets stay free: the taps are inserted into them. Kernels
    without slices (the IODMA partitions) match no cell: ``set_property`` on an
    empty list is an error.
    """
    return [
        (
            "set finn_slice_rst [get_cells -quiet -hier -filter {IS_SEQUENTIAL &&"
            " NAME =~ *_slice/inst/axisc_register_slice_0/gen_auto_slr.auto/*_areset_*_reg &&"
            " NAME !~ *resp2_reg}]"
        ),
        "if {[llength $finn_slice_rst]} { set_property DONT_TOUCH true $finn_slice_rst }",
    ]


def register_slice_inputs(node: NodeProto, model: ModelWrapper) -> list[int]:
    """Return the input indices of ``node`` that get a register slice.

    Every input another dataflow node drives, unless the node is a FIFO: a FIFO's
    input ready is a registered "not full" flag already (`finn-rtllib/fifo/hdl/Q_srl.v`
    drives it from a register; the XPM-based variant likewise), so a slice in front of
    it would only add latency. Inputs with an initializer or from outside the
    partition have no producer here and get none.
    """
    if "StreamingFIFO" in node.op_type:
        return []
    inputs: list[int] = []
    for i, tensor in enumerate(node.input):
        producer = model.find_producer(tensor)
        if producer is None or "fpgadataflow" not in producer.domain:
            continue
        inputs.append(i)
    return inputs
