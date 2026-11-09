"""Ops that stream their weights or parameters from an on-chip memstream.

A memstream (``finn-rtllib/memstream``) holds a node's weights, thresholds or
parameters in block, ultra or distributed RAM and streams them into the node's
compute core. :class:`MemStreamOp` gives the ops that have one the code generation
for its Verilog wrapper and the node attributes that configure it; the other
fpgadataflow ops carry neither.
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import is_versal
from finn.util.settings import get_settings


def memstream_cascade_height(fpgapart: str, ram_style: str, depth: int) -> int:
    """Return the CASCADE_HEIGHT for a memstream of ``depth`` words, 0 leaving it to synthesis.

    Fitted to out-of-context runs of memstream (U55C at 2.5, 3.3 and 5 ns, ZU28DR and
    V80 at 5 ns; block and ultra RAM, 8 to 512 bits wide, 2K to 128K words). On
    UltraScale+ the synthesis default chains up to eight RAMs and leaves the read register
    in the fabric, losing up to 1.9 ns to the best height. The heights below were best or
    within 0.1 ns of best in 54 of 60 cases:

    - Block RAM holds 32K words per RAMB36 without cascading, so height 1. Between 32K and
      64K words each bit takes exactly a RAMB36 pair, whose built-in cascade replaces a 2:1
      read mux in the fabric: height 2 saved 75 LUTs and 0.2 ns on 64000 x 64 words at
      3.3 ns. At 128K words heights 1 and 2 were even. "auto" mapped deep memories to block
      RAM with the same results.
    - URAM holds 4K words; cascading only what an 8:1 read mux in the fabric cannot take
      gives height 1 up to 32K words and 4 at 128K words.

    On Versal the default was as good as any fixed height; 7-series ignores the attribute.
    Unlike the default, bounded block RAM heights cannot pack 64- or 512-bit words into
    RAMB36 parity bits and take up to 12% more RAMB36; ``ram_cascade_height`` = 0 on a
    node trades that back for timing.
    """
    if is_versal(fpgapart) or fpgapart[2] == "7" or ram_style == "distributed":
        return 0
    if ram_style == "ultra":
        return max(1, -(-depth // (8 * 4096)))
    return 2 if 32768 < depth <= 65536 else 1


class MemStreamOp(HWCustomOp, ABC):
    """Base of the ops whose weights or parameters stream from a memstream.

    An op takes it next to its other bases and implements :meth:`memstream_depth`; the
    width defaults to the op's second input. Ops whose memstream holds something else
    (RTL Requant streams scale, shift and bias together) pass the width explicitly to
    :meth:`generate_hdl_memstream`.
    """

    def get_nodeattr_types(
        self,
    ) -> dict[
        str,
        tuple[str, bool, int | float | str | bool | np.ndarray | list]
        | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
    ]:
        """Return the memstream's node attributes on top of the next base's.

        ``ram_cascade_height`` is the longest chain of block or ultra RAMs the memstream
        may join through their cascade pins: -1 picks it from the part and memory shape
        (:func:`memstream_cascade_height`), 0 leaves it to synthesis, n > 0 forces n. Like
        ``ram_style`` it can be set per node in the folding config.
        """
        attrs = super().get_nodeattr_types()
        attrs.update({"ram_cascade_height": ("i", False, -1)})
        return attrs

    @abstractmethod
    def memstream_depth(self) -> int:
        """Return the number of words in one set of the memstream."""

    def memstream_width(self) -> int:
        """Return the bit width of a memstream word, by default that of the second input."""
        return self.get_instream_width_padded(1)

    def generate_hdl_memstream(
        self,
        fpgapart: str,
        pumped_memory: int = 0,
        name: str | None = None,
        width: int | None = None,
        init_file: str | None = None,
        ram_style: str | None = None,
    ) -> None:
        """Generate the Verilog wrapper of the node's memstream.

        By default the wrapper is named after the node, with width, init file and RAM
        style taken from the node. A caller whose memstream differs (RTL Requant) passes
        ``name``, ``width``, ``init_file`` and ``ram_style``.

        Args:
            fpgapart: Target FPGA part.
            pumped_memory: 1 to run the memory at twice the clock with half the width.
            name: Module name prefix of the wrapper, the node name if None.
            width: Word width in bits, :meth:`memstream_width` if None.
            init_file: Memory initialisation file, ``memblock.dat`` in the node's code
                generation directory if None.
            ram_style: RAM style, the node's ``ram_style`` if None.
        """
        template_path = (
            Path(get_settings().finn_rtllib) / "memstream" / "hdl" / "memstream_wrapper_template.v"
        )
        mname = name if name is not None else self.onnx_node.name
        sets = 1
        mlo_max_iter = cast("int", self.get_nodeattr("mlo_max_iter"))
        bodies = cast("int", self.get_nodeattr("bodies"))
        if mlo_max_iter:
            sets = mlo_max_iter
        elif bodies:
            sets = bodies
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        depth = self.memstream_depth()
        if width is None:
            width = self.memstream_width()
        if ram_style is None:
            ram_style = cast("str", self.get_nodeattr("ram_style"))
        if init_file is None:
            init_file = code_gen_dir + "/memblock.dat"
        if ram_style == "ultra" and not is_versal(fpgapart):
            init_file = ""
        cascade_height = cast("int", self.get_nodeattr("ram_cascade_height"))
        if cascade_height < 0:
            # a pumped memory holds twice the words at half the width
            words = sets * depth * (2 if pumped_memory else 1)
            cascade_height = memstream_cascade_height(fpgapart, ram_style, words)
        code_gen_dict = {
            "$MODULE_NAME$": mname,
            "$SETS$": str(sets),
            "$DEPTH$": str(depth),
            "$WIDTH$": str(width),
            "$INIT_FILE$": init_file,
            "$RAM_STYLE$": ram_style,
            "$CASCADE_HEIGHT$": str(cascade_height),
            "$PUMPED_MEMORY$": str(pumped_memory),
        }
        template_wrapper = template_path.read_text()
        for key, value in code_gen_dict.items():
            template_wrapper = template_wrapper.replace(key, value)
        (Path(code_gen_dir) / (mname + "_memstream_wrapper.v")).write_text(template_wrapper)
