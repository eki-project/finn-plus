"""AXI-stream register slices at the inputs of dataflow nodes.

A FINN dataflow node answers the valid of the stream it receives with a ready
that is, for most node types, a combinational function of that valid and of the
node's state. The producer's next state depends on that ready, so every such
stream carries a combinational round trip: producer register -> consumer logic
-> producer register. With the producer and the consumer in different placement
units that round trip crosses the gap between them twice, and it is the one
inter-unit path class that distance breaks: on the floorplanned mobilenetv1 a
FIFO 597 tiles from the window buffer it fed had 0.97 ns of slack at 100 MHz
(rebuild6, 14 Oct 2026), and at 200 MHz the same pair 300 tiles apart was 0.52 ns
short (per_node_5ns, 15 Oct 2026). The free placer has the same class, with the
units simply closer.

A fully registered AXI-stream register slice at the consumer's input breaks the
round trip: its ready output is a register, so the producer sees a registered
ready, and the consumer's combinational ready reaches only the slice, which sits
in the consumer's placement unit. Throughput is unchanged (the slice holds two
beats), the pipeline gets one cycle of latency per slice, and the cost is two
registers per stream bit -- 9 % more flip-flops on mobilenetv1 at every
non-FIFO input.

This module holds the rule of where a slice goes, shared by `CreateStitchedIP`
(which instantiates them) and the floorplanner's binding (which puts each slice
into the unit of the node it feeds).
"""

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper

#: Vivado's AXI4-Stream register slice.
SLICE_VLNV = "xilinx.com:ip:axis_register_slice:1.1"

#: ``REG_CONFIG`` 8 is "FWD_REV": both the forward (valid/data) and the reverse
#: (ready) direction registered, two beats of storage, no throughput loss.
SLICE_REG_CONFIG = 8


def slice_cell_name(node_name: str, input_index: int) -> str:
    """Name of the register slice in front of input ``input_index`` of a node."""
    return f"{node_name}_in{input_index}_slice"


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
