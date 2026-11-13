"""Legacy HLS backend for the operators that still use the packed ap_uint stream interface
of the finn-hlslib generation before its move to tensor streams."""

import numpy as np
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.templates import load_codegen_template
from finn.util.settings import get_settings

#: Name of the dependency (inside FINN_DEPS) that holds the finn-hlslib generation the
#: legacy operators are written against. It still provides bnn-library.h, flatten.hpp
#: and interpret.hpp, which the attention-hlslib and the npy2apintstream helper include.
LEGACY_HLSLIB_DEP = "finn-hlslib-legacy"


class LegacyHLSBackend(HLSBackend):
    """HLSBackend with the stream interface FINN used before finn-hlslib moved to
    ``tensor<T, N>`` streams: packed ``ap_uint`` streams, bnn-library.h as the default
    include, the npy2apintstream helper and a cppsim that calls the top function once.

    It exists for the operators that have not been ported to the tensor interface (the
    attention operators, which depend on the attention-hlslib, Squeeze, Unsqueeze and
    the HLS StreamingFIFO) and should disappear together with them.
    """

    def get_hlslib_path(self) -> Path:
        """Return the legacy finn-hlslib checkout."""
        return get_settings().finn_deps / LEGACY_HLSLIB_DEP

    def cppsim_template(self) -> str:
        """Return the legacy cppsim template, which calls the top function once."""
        return load_codegen_template("legacy/execute_single_node.cpp")

    def ipgen_template(self) -> str:
        """Return the legacy C++ template of the HLS top function."""
        return load_codegen_template("legacy/hls_ipgen.tcl")

    def encode_input_for_npy(
        self, inp_val: np.ndarray, dtype: BaseDataType
    ) -> tuple[np.ndarray, BaseDataType]:
        """Store bipolar inputs as binary, which is what the legacy npy helper reads."""
        if dtype == DataType["BIPOLAR"]:
            return (inp_val + 1) / 2, DataType["BINARY"]
        return inp_val, dtype

    def _elem_hls_type(self, dtype: BaseDataType) -> str:
        """Return the HLS element type of a FINN DataType, bipolar being stored as binary."""
        if dtype == DataType["BIPOLAR"]:
            dtype = DataType["BINARY"]
        return dtype.get_hls_datatype_str()

    def read_npy_data(self) -> None:
        """Generate commands for reading data from .npy file in C++.
        Might need to be overwritten depending on CustomOp."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$READNPYDATA$"] = []

        for i, _inp in enumerate(self.onnx_node.input):
            iwidth = self.get_instream_width(i)
            # if the stream is not exposed, it has 0 width and no npy file will be created
            if iwidth == 0:
                continue
            elem_hls_type = self._elem_hls_type(self.get_input_datatype(i))
            npy_type = "half" if elem_hls_type == "half" else "float"
            npy_in = f"{code_gen_dir}/input_{i}.npy"
            self.code_gen_dict["$READNPYDATA$"].append(
                f"npy2apintstream<ap_uint<{iwidth}>, {elem_hls_type}, {npy_type}>"
                f'("{npy_in}", in{i}_V);'
            )

    def strm_decl(self) -> None:
        """Generate commands for stream declaration in C++.
        Might need to be overwritten depending on CustomOp."""
        node = self.onnx_node
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        for i, _inp in enumerate(node.input):
            if self.get_instream_width(i):
                self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                    f'hls::stream<ap_uint<{self.get_instream_width(i)}>> in{i}_V ("in{i}_V");'
                )
        for o, _outp in enumerate(node.output):
            if self.get_outstream_width(o):
                self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                    f'hls::stream<ap_uint<{self.get_outstream_width(o)}>> out{o}_V ("out{o}_V");'
                )

    def dataoutstrm(self) -> None:
        """Generate commands for reading out data from C++ and converting to npy format.
        Might need to be overwritten depending on CustomOp."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

        for o, _outp in enumerate(self.onnx_node.output):
            elem_hls_type = self._elem_hls_type(self.get_output_datatype(o))
            npy_type = "half" if elem_hls_type == "half" else "float"
            npy_out = f"{code_gen_dir}/output_{o}.npy"
            oshape = self.get_folded_output_shape(o)
            oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
            packed_hls_type = f"ap_uint<{self.get_outstream_width(o)}>"
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {npy_type}>"
                f'(out{o}_V, {oshape_cpp_str}, "{npy_out}");'
            )

    def pragmas(self) -> None:
        """Generate pragma commands in C++.
        Might need to be overwritten depending on CustomOp."""
        # no DATAFLOW pragma here: the legacy tops are not written as dataflow regions
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
