"""Base class for both Mux and Demux."""
import jinja2
from abc import abstractmethod
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from typing import cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError


class MuxDemux(HWCustomOp):
    """General mux/demux operator. Do not instantiate directly. Does a best effort to
    already set pragmas, blackboxfunction, etc. For special cases, these methods need to
    be overwritten in the inheriting class.

    This operator has some nodeattributes describing streams. For a Mux, these are the input
    streams, for a Demux these are the output streams. The default signature for these functions
    are then something like:
    ```
    void Multiplexer_hls(
        hls::stream<..> &in0_V, hls::stream<..> &in1_V, ...,
        hls::stream<..> &out0_V
    )
    ```

    and

    ```
    void Demultiplexer_hls(
        hls::stream<..> &in0_V
        hls::stream<..> &out0_V, hls::stream<..> &out1_V, ...
    )
    ```

    with matching pragmas provided as well.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs) -> None:  # noqa
        """Create a mux node."""
        super().__init__(onnx_node, **kwargs)

    @abstractmethod
    def get_op_type(self) -> str:
        """Overwritten by inheriting classes. Should return "mux" or "demux".
        Could also be read from the type name, however using a custom
        method is more flexible with regard to future extensions.
        """

    def defines(self, var) -> None:  # noqa
        self.code_gen_dict["$DEFINES$"] = []

    def pragmas(self) -> None:  # noqa
        self.code_gen_dict["$PRAGMAS$"] = []
        op_type = self.get_op_type()
        if op_type in ["mux", "combined"]:
            for i in range(self.get_stream_count()):
                self.code_gen_dict["$PRAGMAS$"].append(f"#pragma HLS INTERFACE axis port=in{i}_V")
        else:
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in0_V")
        if op_type in ["demux", "combined"]:
            for i in range(self.get_stream_count()):
                self.code_gen_dict["$PRAGMAS$"].append(f"#pragma HLS INTERFACE axis port=out{i}_V")
        else:
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def get_input_stream_types(self) -> str:
        """Get a comma separated list of hls::stream types for the inputs."""
        op_type = self.get_op_type()
        if op_type in ["mux", "combined"]:
            return ", ".join(
                [
                    f"hls::stream<{self.get_input_datatype(i).get_hls_datatype_str()}> &in{i}_V"
                    for i in range(self.get_stream_count())
                ]
            )
        elif op_type == "demux":  # noqa
            return f"hls::stream<{self.get_connection_dtype().get_hls_datatype_str()}> &in0_V"
        return "UNDEFINED"

    def get_output_stream_types(self) -> str:
        """Get a comma separated list of hls::stream types for the outputs."""
        op_type = self.get_op_type()
        if op_type in ["demux", "combined"]:
            return ", ".join(
                [
                    f"hls::stream<{self.get_output_datatype(i).get_hls_datatype_str()}> &out{i}_V"
                    for i in range(self.get_stream_count())
                ]
            )
        elif op_type == "mux":  # noqa
            return f"hls::stream<{self.get_connection_dtype().get_hls_datatype_str()}> &out0_V"
        return "UNDEFINED"

    def blackboxfunction(self) -> None:  # noqa
        ins = self.get_input_stream_types()
        outs = self.get_output_stream_types()
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [f"void {self.onnx_node.name}({ins}, {outs})"]

    def render_compute_template(self, **kwargs) -> str:  # noqa
        """Render a template of the given variant with the passed keyword arguments.
        If no such template is found, raise an error.

        `op_type` can be "mux" or "demux".
        """
        op_type = self.get_op_type()
        variant = str(self.get_nodeattr("muxVariant"))
        subtype = str(self.get_nodeattr("muxVariantSubtype"))
        template_map = {
            "mux": {"static_schedule": {"round_robin": "static_schedule/docompute_mux.cpp.jinja2"}},
            "demux": {
                "static_schedule": {"round_robin": "static_schedule/docompute_demux.cpp.jinja2"}
            },
            "combined": {
                "static_schedule": {
                    "round_robin": "static_schedule/docompute_combined_muxdemux.cpp.jinja2"
                }
            },
        }
        try:
            template_path = template_map[op_type][variant][subtype]
        except KeyError:
            raise FINNInternalError(
                f"Cannot instantiate template for operator: No such "
                f"variant found: {op_type} - {variant} - {subtype}."
            ) from None
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                Path(__file__).parent.parent.parent.parent.parent / "custom_hls" / "mux"
            )
        )
        template = env.get_template(template_path)
        return template.render(**kwargs)

    def global_includes(self) -> None:
        """Add the global includes for all mux variants."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "mux/static_schedule/static_mux.hpp"']

    def get_nodeattr_types(self) -> dict:
        """Node attribute defs."""
        attrs = HWCustomOp.get_nodeattr_types(self)
        attrs.update(
            {
                # Describes the general way in which the mux works
                "muxVariant": ("s", True, "static_schedule"),
                # Describes the subtype. Round-robin is for example a
                # statically scheduled variant.
                "muxVariantSubtype": ("s", True, "round_robin"),
                "streamNames": ("strings", True, []),
                "streamWidths": ("ints", True, []),
                "streamDataTypes": ("strings", True, []),
                # A shape is stored as a string with "," separating the tuple elements
                "streamsFoldedShapes": ("strings", True, []),
                "streamsNormalShapes": ("strings", True, []),
                "connectionStream": ("s", True, ""),
            }
        )
        return attrs

    def check_correct_nodeattributes(self) -> None:
        """Check that all node attributes have the correct count.
        If not, raise an error.
        """
        names = len(self.get_stream_names())
        widths = len(self.get_stream_widths())
        dts = len(self.get_stream_dts())
        foldeds = len(cast("list", self.get_nodeattr("streamsFoldedShapes")))
        normals = len(cast("list", self.get_nodeattr("streamsNormalShapes")))
        if not (names == widths and widths == dts and dts == foldeds and foldeds == normals):
            raise FINNInternalError(
                f"(De)Mux operator attributes incorrect. "
                f"Non equal number of names, widths, datatypes, "
                f"folded shapes or normal shapes ({normals}, "
                f"{widths}, {dts}, {foldeds}, {normals})."
            )

    def get_stream_count(self) -> int:
        """Return the number of connected streams (inputs for mux, outputs for demux)."""
        self.check_correct_nodeattributes()
        return len(self.get_stream_names())

    def get_stream_names(self) -> list[str]:
        """Return a list of stream names."""
        return cast("list[str]", self.get_nodeattr("streamNames"))

    def get_stream_widths(self) -> list[int]:
        """Return a list of all stream widths."""
        return cast("list[int]", self.get_nodeattr("streamWidths"))

    def get_stream_dts(self) -> list[BaseDataType]:
        """Return a list of all stream datatypes."""
        return [DataType[s] for s in cast("list[str]", self.get_nodeattr("streamDataTypes"))]

    def get_stream_folded_shape(self, ind: int) -> list[int]:
        """Return the folded shape of the given stream."""
        return self.nodeattr_string_to_shape("streamsFoldedShapes", ind)

    def get_stream_normal_shape(self, ind: int) -> list[int]:
        """Return the normal shape of the given stream."""
        return self.nodeattr_string_to_shape("streamsNormalShapes", ind)

    def required_bitwidth(self, count: int) -> int:
        """Return number of bits required to count to this number."""
        return count if count < 2 else 1 + self.required_bitwidth(count // 2)

    def get_connection_dtype(self) -> BaseDataType:
        """For the given streams (can be either mux inputs or demux outputs),
        determine the "network" datatype. This will always be an unsigned AP
        integer with the bitwidth of the largest stream
        dtype, with additional bits to accommodate the header with the source stream index.
        """
        stream_widths = self.get_stream_widths()
        largest_dt_bitwidth = max(stream_widths)
        header_bitwidth = self.required_bitwidth(len(stream_widths))
        return DataType[f"UINT{largest_dt_bitwidth + header_bitwidth}"]

    def nodeattr_string_to_shape(self, key: str, ind: int) -> list[int]:
        """Try to parse the nodeattribute with the given key into a list
        of ints (shape). (From [shape0, shape1, ...].
        """
        return [int(i) for i in cast("list[str]", self.get_nodeattr(key))[ind].split(",")]
