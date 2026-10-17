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
