"""ConvolutionInputGenerator (sliding window generator, RTL) microbenchmark DUT."""

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MicrobenchDUT,
    specialize_single_node,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Conditional, Divisor, Fixed, ParamSpace


class bench_swg(MicrobenchDUT):
    NAME = "swg"
    OP_TYPES = ("ConvolutionInputGenerator_rtl",)
    PARAMS = {
        "idt": "input datatype",
        "ifm_ch": "input channels",
        "ifm_dim": "input feature map size [H, W] (H or W == 1 for 1D)",
        "k": "kernel size [kh, kw]",
        "stride": "stride [sh, sw]",
        "dilation": "dilation [dh, dw]",
        "simd": "channel parallelism",
        "depthwise": "0/1, depthwise output layout",
        "parallel_window": "0/1, output the whole window in parallel",
        "m": "multiplicity (parallel_window only), 1 otherwise",
        "ram_style": "auto, block, distributed or ultra",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        idt = DataType[params["idt"]]
        ifm_ch, simd = int(params["ifm_ch"]), int(params["simd"])
        ifm_h, ifm_w = (int(x) for x in params["ifm_dim"])
        k_h, k_w = (int(x) for x in params["k"])
        stride_h, stride_w = (int(x) for x in params["stride"])
        dil_h, dil_w = (int(x) for x in params["dilation"])
        depthwise, parallel_window = int(params["depthwise"]), int(params["parallel_window"])
        m = int(params.get("m", 1))
        if params.get("ram_style") not in ("auto", "block", "distributed", "ultra"):
            return "invalid ram_style"
        if min(ifm_ch, simd, ifm_h, ifm_w, k_h, k_w, stride_h, stride_w, dil_h, dil_w, m) < 1:
            return "dimensions must be >= 1"
        k_dil_h, k_dil_w = (k_h - 1) * dil_h + 1, (k_w - 1) * dil_w + 1
        if k_dil_h > ifm_h or k_dil_w > ifm_w or stride_h > ifm_h or stride_w > ifm_w:
            return "kernel or stride larger than the feature map"
        is1d = ifm_h == 1 or ifm_w == 1
        if is1d:
            if ifm_h == 1 and (k_h != 1 or stride_h != 1 or dil_h != 1):
                return "1D: kernel/stride/dilation must be 1 along the unit dimension"
            if ifm_w == 1 and (k_w != 1 or stride_w != 1 or dil_w != 1):
                return "1D: kernel/stride/dilation must be 1 along the unit dimension"
        if m > 1 and not parallel_window:
            return "m > 1 requires parallel_window"
        if ifm_ch % simd != 0:
            return "simd must divide ifm_ch"
        parallel = parallel_window or (k_h == 1 and k_w == 1)
        if parallel and not (depthwise or (k_h == 1 and k_w == 1)) and simd != ifm_ch:
            return "parallel_window (non-depthwise) requires simd == ifm_ch"
        in_width = simd * idt.bitwidth()
        out_width = in_width * k_h * k_w if parallel_window else in_width
        if not stream_width_ok(in_width) or not stream_width_ok(out_width):
            return "stream width exceeds the instrumentation limit"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "idt": Choice(["BINARY", "UINT2", "INT2", "UINT4", "INT4", "UINT8", "INT8"]),
            "ifm_ch": Choice([3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "ifm_dim": Choice(
                [
                    [4, 4],
                    [7, 7],
                    [8, 8],
                    [14, 14],
                    [16, 16],
                    [28, 28],
                    [32, 32],
                    [56, 56],
                    [112, 112],
                    [224, 224],
                    [1, 64],
                    [1, 256],
                    [1, 1024],
                ]
            ),
            "k": Conditional(
                "ifm_dim",
                {True: Choice([[1, 3], [1, 5], [1, 7]])},
                Choice([[1, 1], [2, 2], [3, 3], [5, 5], [7, 7]]),
                key=lambda dim: dim[0] == 1,
            ),
            "stride": Choice([[1, 1], [1, 1], [2, 2]]),
            "dilation": Fixed([1, 1]),
            "depthwise": Choice([0, 1]),
            "parallel_window": Choice([0, 0, 1]),
            "m": Fixed(1),
            "simd": Divisor("ifm_ch", pow2=False, hi=64),
            "ram_style": Choice(["auto", "block", "distributed", "ultra"]),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        idt = DataType[params["idt"]]
        ifm_ch, simd = int(params["ifm_ch"]), int(params["simd"])
        ifm_h, ifm_w = (int(x) for x in params["ifm_dim"])
        k_h, k_w = (int(x) for x in params["k"])
        stride_h, stride_w = (int(x) for x in params["stride"])
        dil_h, dil_w = (int(x) for x in params["dilation"])
        ofm_h = compute_conv_output_dim(ifm_h, k_h, stride_h, 0, dil_h)
        ofm_w = compute_conv_output_dim(ifm_w, k_w, stride_w, 0, dil_w)
        is1d = int(ifm_h == 1 or ifm_w == 1)

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_h, ifm_w, ifm_ch])
        outp = helper.make_tensor_value_info(
            "outp", TensorProto.FLOAT, [1, ofm_h, ofm_w, k_h * k_w * ifm_ch]
        )
        node = helper.make_node(
            "ConvolutionInputGenerator",
            ["inp"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            ConvKernelDim=[k_h, k_w],
            IFMChannels=ifm_ch,
            IFMDim=[ifm_h, ifm_w],
            OFMDim=[ofm_h, ofm_w],
            SIMD=simd,
            Stride=[stride_h, stride_w],
            Dilation=[dil_h, dil_w],
            inputDataType=idt.name,
            outputDataType=idt.name,
            depthwise=int(params["depthwise"]),
            ram_style=params["ram_style"],
            parallel_window=int(params["parallel_window"]),
            is1D=is1d,
            dynamic_mode=0,
        )
        graph = helper.make_graph([node], "swg_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="swg-model"))
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", idt)

        model = specialize_single_node(model, "rtl", fpga_part)
        inst = getCustomOp(model.graph.node[0])
        inst.set_nodeattr("M", int(params.get("m", 1)))
        info = {
            "impl_style": inst.select_impl_style(),
            "buffer_depth": int(inst.get_buffer_depth()),
            "ofm_dim": [ofm_h, ofm_w],
            "out_width": int(inst.get_outstream_width()),
            "is1D": is1d,
        }
        return model, info
