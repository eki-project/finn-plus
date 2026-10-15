# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL backend implementation of the SelectToken layer."""

import numpy as np
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtl.crop_rtl import Crop_rtl
from finn.custom_op.fpgadataflow.selecttoken import NodeAttrTypes, SelectToken
from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from onnx import GraphProto


@register_custom_op
class SelectToken_rtl(SelectToken, Crop_rtl):
    """RTL SelectToken implemented by the shared Crop core."""

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, combining SelectToken and Crop_rtl attributes."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(SelectToken.get_nodeattr_types(self))
        my_attrs.update(Crop_rtl.get_nodeattr_types(self))
        return my_attrs

    def _get_template_param_dict(self) -> dict[str, int | str]:
        """Return crop-core template parameters that select the single token column."""
        num_tokens = self.num_tokens
        token_index = self._resolved_token_index()
        channels = self.num_channels
        simd = self.simd
        if channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")
        # Selecting one of NumTokens vectors is a crop of a 1 x NumTokens feature
        # map down to the single selected column, so map onto the crop core.
        return {
            "H": 1,
            "W": num_tokens,
            "CF": channels // simd,
            "FOLD_WIDTH": self.get_input_datatype().bitwidth() * simd,
            "CROP_N": 0,
            "CROP_E": num_tokens - token_index - 1,
            "CROP_S": 0,
            "CROP_W": token_index,
        }

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute the node via Python (cppsim) or rtlsim depending on exec_mode."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            SelectToken.execute_node(self, context, graph)
        elif mode == "rtlsim":
            Crop_rtl.execute_node(self, context, graph)
        else:
            raise FINNInternalError('exec_mode must be either "cppsim" or "rtlsim"')
