"""Base class for both Mux and Demux."""
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from typing import cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError


class MuxDemux(HWCustomOp):
    def __init__(self, onnx_node: NodeProto, **kwargs) -> None:  # noqa
        """Create a mux node."""
        super().__init__(onnx_node, **kwargs)

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
        foldeds = len(cast("list", self.get_nodeattr("streamFoldedShapes")))
        normals = len(cast("list", self.get_nodeattr("streamNormalShapes")))
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
        return self.nodeattr_string_to_shape("streamFoldedShapes", ind)

    def get_stream_normal_shape(self, ind: int) -> list[int]:
        """Return the normal shape of the given stream."""
        return self.nodeattr_string_to_shape("streamNormalShapes", ind)

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
