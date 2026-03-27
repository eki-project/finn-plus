"""RTL implementation for RemoveDataPath custom operation.

This module provides the RTL backend implementation for the RemoveDataPath
custom operation, which removes data from the datapath while maintaining
the control flow.
"""

import numpy as np
import os
from collections.abc import Sequence
from numpy import ndarray
from numpy import typing as npt
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from typing import Any, cast

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError
from finn.util.logging import log


class RemoveDataPath_rtl(RTLBackend):
    """RTL implementation for RemoveDataPath custom op."""

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        """Initialize RemoveDataPath RTL backend.

        Args:
            onnx_node: The ONNX node proto for this operation.
            **kwargs: Additional keyword arguments passed to parent class.

        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> dict:
        """Return node attribute types for this custom operation.

        Returns:
            Dictionary mapping attribute names to their type specifications.

        """
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                # folded shape of input/output
                "folded_shape": ("ints", True, []),
                # normal shape of input/output
                "normal_shape": ("ints", True, []),
                # FINN DataTypes for inputs/outputs
                "dataType": ("s", True, ""),
            }
        )
        return my_attrs

    def infer_node_datatype(self, model: Any) -> None:
        """Infer and set the output datatype based on input datatype.

        Args:
            model: The model wrapper containing this node.

        """
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: {self.get_input_datatype()} -> {idt}"
            )
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def get_rtl_file_list(self, abspath: bool = False) -> list[Path]:
        """Return list of RTL files required for this custom operation.

        Args:
            abspath: Whether to return absolute paths (default: False).

        Returns:
            List of Path objects pointing to required RTL files.

        Raises:
            FINNInternalError: If code_gen_dir_ipgen or gen_top_module attributes are invalid.

        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""

        top_name = self.get_nodeattr("gen_top_module")
        if type(code_gen_dir) is not str:
            raise FINNInternalError(
                f"code_gen_dir_ipgen attribute not set in {self.onnx_node.name}, "
                "cannot get RTL file list"
            )
        if type(top_name) is not str or top_name == "":
            raise FINNInternalError(
                f"gen_top_module attribute not set in {self.onnx_node.name}, "
                "cannot get RTL file list"
            )

        code_gen_dir_path = Path(code_gen_dir)

        verilog_files = [
            code_gen_dir_path / f"{top_name}.v",
        ]
        return verilog_files

    def generate_hdl(self, model: Any, fpgapart: str, clk: str) -> None:  # noqa: ARG002
        """Generate the RTL code for this custom op.

        Args:
            model: The model wrapper containing this node (unused).
            fpgapart: Target FPGA part string (unused).
            clk: Clock period in nanoseconds (unused).

        Raises:
            FINNInternalError: If code_gen_dir_ipgen attribute is invalid.

        """
        rtlsrc = Path(os.environ["FINN_RTLLIB"]) / "removedatapath" / "hdl"
        template_path = rtlsrc / "dummy_template.v"

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width_padded()

        code_gen_dict = {"$TOP_MODULE_NAME$": topname, "$WIDTH$": str(in_width)}

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if type(code_gen_dir) is not str or code_gen_dir == "":
            raise FINNInternalError(
                f"code_gen_dir_ipgen attribute not set in {topname}, cannot generate RTL code"
            )
        with Path.open(template_path) as f:
            template = f.read()

        for placeholder, value in code_gen_dict.items():
            template = template.replace(placeholder, value)

        output_path = Path(code_gen_dir) / f"{self.get_verilog_top_module_name()}.v"
        with Path.open(output_path, "w") as f:
            f.write(template)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        # i.e. during the HLSSynthIP() transformation
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def code_generation_ipi(self) -> list[str]:
        """Code generation for IP integration."""
        sourcefiles = self.get_rtl_file_list(abspath=True)

        cmd = []
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {f}"]
        cmd += [
            "create_bd_cell -type module -reference "
            f"{self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        ]
        return cmd

    def get_normal_input_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the normal (unfolded) input shape.

        Args:
            ind: Input index (unused, kept for interface compatibility).

        Returns:
            The normal input shape dimensions.

        Raises:
            FINNInternalError: If normal_shape attribute is invalid or empty.

        """
        normal_shape = self.get_nodeattr("normal_shape")
        if (
            type(normal_shape) is not list
            and type(normal_shape) is not tuple
            and not isinstance(normal_shape, ndarray)
        ):
            raise FINNInternalError(
                f"normal_shape attribute not set correctly in {self.onnx_node.name}, "
                "cannot get normal input shape"
            )
        if len(normal_shape) == 0:
            raise FINNInternalError(
                f"normal_shape attribute is empty in {self.onnx_node.name}, "
                "cannot get normal input shape"
            )
        if type(normal_shape[0]) is not int:
            raise FINNInternalError(
                f"normal_shape attribute not set correctly in {self.onnx_node.name}, "
                "cannot get normal input shape"
            )
        return normal_shape

    def get_normal_output_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the normal (unfolded) output shape.

        Args:
            ind: Output index (unused, kept for interface compatibility).

        Returns:
            Tuple containing the normal output shape dimensions.

        """
        return self.get_normal_input_shape()

    def get_folded_input_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the folded input shape.

        Args:
            ind: Input index (unused, kept for interface compatibility).

        Returns:
            Tuple containing the folded input shape dimensions.

        """
        folded_shape = self.get_nodeattr("folded_shape")
        if (
            type(folded_shape) is not list
            and type(folded_shape) is not tuple
            and not isinstance(folded_shape, ndarray)
        ):
            raise FINNInternalError(
                f"folded_shape attribute not set correctly in {self.onnx_node.name}, "
                "cannot get folded input shape"
            )
        if len(folded_shape) == 0:
            raise FINNInternalError(
                f"folded_shape attribute is empty in {self.onnx_node.name}, "
                "cannot get folded input shape"
            )
        if type(folded_shape[0]) is not int:
            raise FINNInternalError(
                f"folded_shape attribute not set correctly in {self.onnx_node.name}, "
                "cannot get folded input shape"
            )
        return cast("Sequence[int]", folded_shape)

    def get_folded_output_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the folded output shape.

        Args:
            ind: Output index (unused, kept for interface compatibility).

        Returns:
            Tuple containing the folded output shape dimensions.

        """
        return self.get_folded_input_shape()

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the input stream width in bits.

        Args:
            ind: Input index (unused, kept for interface compatibility).

        Returns:
            Input stream width in bits.

        """
        dtype = self.get_nodeattr("dataType")
        if type(dtype) is not str:
            raise FINNInternalError(
                f"dataType attribute not set correctly in {self.onnx_node.name}, "
                "cannot get instream width"
            )
        dtype = DataType[dtype]
        folded_shape = self.get_nodeattr("folded_shape")
        if (
            type(folded_shape) is not list
            and type(folded_shape) is not tuple
            and not isinstance(folded_shape, ndarray)
        ):
            raise FINNInternalError(
                f"folded_shape attribute not set correctly in {self.onnx_node.name}, "
                "cannot get outstream width"
            )
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width in bits.

        Args:
            ind: Output index (unused, kept for interface compatibility).

        Returns:
            Output stream width in bits.

        Raises:
            FINNInternalError: If dataType or folded_shape attributes are invalid.

        """
        dtype = self.get_nodeattr("dataType")
        if type(dtype) is not str:
            raise FINNInternalError(
                f"dataType attribute not set correctly in {self.onnx_node.name}, "
                "cannot get outstream width"
            )
        dtype = DataType[dtype]
        folded_shape = self.get_nodeattr("folded_shape")
        if (
            type(folded_shape) is not list
            and type(folded_shape) is not tuple
            and not isinstance(folded_shape, ndarray)
        ):
            raise FINNInternalError(
                f"folded_shape attribute not set correctly in {self.onnx_node.name}, "
                "cannot get outstream width"
            )
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the input data type.

        Args:
            ind: Input index (unused, kept for interface compatibility).

        Returns:
            The QONNX data type for the input.

        Raises:
            FINNInternalError: If dataType attribute is invalid.

        """
        dtype = self.get_nodeattr("dataType")
        if type(dtype) is not str:
            raise FINNInternalError(
                f"dataType attribute not set correctly in {self.onnx_node.name}, "
                "cannot get outstream width"
            )
        dtype = DataType[dtype]
        return dtype

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the output data type.

        Args:
            ind: Output index (unused, kept for interface compatibility).

        Returns:
            The QONNX data type for the output.

        Raises:
            FINNInternalError: If dataType attribute is invalid.

        """
        dtype = self.get_nodeattr("dataType")
        if type(dtype) is not str:
            raise FINNInternalError(
                f"dataType attribute not set correctly in {self.onnx_node.name}, "
                "cannot get outstream width"
            )
        dtype = DataType[dtype]
        return dtype
