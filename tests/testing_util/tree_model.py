"""Test helpers that validate the FIFO-sizing tree models against rtlsim.

The analytical FIFO sizing (``auto_fifo_strategy: heuristic_analytical``) takes each
operator's token access vectors (TAVs) from its tree model. A tree model is a
hand-written, cycle-exact description of an operator's read/write schedule, so the
only way to know it still matches the hardware is to compare it against a trace of
the real RTL. Upstream FINN carries that trace as a second sizing strategy
(``heuristic_rtlsim``) and as a fallback inside the builder; FINN+ does not. The
per-node rtlsim trace therefore lives here, as a test-only reference built on the
ordinary node-level rtlsim (``prepare_rtlsim`` / ``rtlsim_multi_io``) and the
``SimEngine`` stream tracer.

Ported from ``finn.util.test`` and ``HWCustomOp.derive_token_access_vectors_using_rtlsim``
of Xilinx/finn#1185.
"""

import pytest

import copy
import numpy as np
from collections.abc import Callable
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import TYPE_CHECKING, Any

from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import DeriveTokenAccessVectors
from finn.transformation.fpgadataflow.fifo_sizing.tav import load_tav_npy, save_tav_npy
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import getHWCustomOp

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    from finn.xsi import SimEngine

IODict = dict[str, dict[str, list[int]]]


def get_tav_io_dict(inst: "HWCustomOp") -> IODict:
    """Return the rtlsim stimulus for one period of characterization of ``inst``.

    The format is the ``io_dict`` of ``rtlsim_multi_io``. By default ``in0`` gets one
    folded input frame and ``out0`` is drained. Operators that need other streams
    fed or drained to keep running get them here (upstream FINN has these as
    ``get_tav_io_dict`` overrides on the operator classes).

    Args:
        inst: Operator to characterize.
    """
    op_type = inst.onnx_node.op_type
    n_inps = int(np.prod(inst.get_folded_input_shape()[:-1]))
    io_dict: IODict = {"inputs": {"in0": list(range(n_inps))}, "outputs": {"out0": []}}

    if op_type.startswith("StreamingConcat"):
        # every input has to be fed: the round-robin stalls on the first that runs dry
        io_dict["inputs"] = {
            f"in{i}": [0] * int(np.prod(inst.get_folded_input_shape(i)[:-1]))
            for i in range(inst.get_n_inputs())
        }
    elif op_type.startswith("StreamingSplit"):
        # every output has to be drained: the round-robin stalls on the first not read
        io_dict["inputs"]["in0"] = [0] * n_inps
        io_dict["outputs"] = {f"out{i}": [] for i in range(inst.get_n_outputs())}
    elif op_type.startswith("Elementwise") and op_type.endswith("_rtl"):
        io_dict["inputs"]["in1"] = list(range(n_inps))
    elif op_type == "Thresholding_hls":
        io_dict["inputs"]["in0"] = [0] * n_inps
        if inst.get_nodeattr("mem_mode") in ["internal_decoupled", "external"]:
            num_w_reps = int(np.prod(inst.get_nodeattr("numInputVectors")))
            io_dict["inputs"]["in1"] = [0] * (num_w_reps * inst.calc_tmem())
    elif op_type.startswith("VVAU"):
        if inst.get_nodeattr("mem_mode") in ["internal_decoupled", "external"]:
            io_dict["inputs"]["in1"] = [0] * n_inps
    elif op_type.startswith("MVAU"):
        # a dynamic node's rtlsim is the bare MVU without its dynamic_load, so
        # its in1 also takes one weight tile per compute cycle
        if inst.get_nodeattr("mem_mode") in ["internal_decoupled", "external", "dynamic"]:
            num_w_reps = int(np.prod(inst.get_nodeattr("numInputVectors")))
            io_dict["inputs"]["in1"] = list(range(num_w_reps * inst.calc_wmem()))
    return io_dict


def derive_token_access_vectors_using_rtlsim(
    inst: "HWCustomOp",
    pre_hook: Callable[["SimEngine"], Any] | None = None,
    periods_to_simulate: int = 5,
) -> None:
    """Derive the token access vectors of ``inst`` by rtlsim of the node in isolation.

    The node is driven with ``get_tav_io_dict`` repeated for ``periods_to_simulate``
    periods and the vectors are stored like the tree-model ones (``io_chrc_in`` /
    ``io_chrc_out`` / ``io_chrc_period``). Only ``in0`` and ``out0`` are traced.

    Args:
        inst: Operator to characterize, with IP generated.
        pre_hook: Optional callable taking the simulation, called after reset and
            before the simulation runs.
        periods_to_simulate: How many frames to drive. Five by default, of which the
            middle two are stored: the first frame carries the node's wind-up
            (pipeline fill, weight priming) and the last its drain, and neither
            belongs in a schedule the sizer reads as periodic. With one frame, that
            frame is stored twice.
    """
    periods_to_store = 2
    if inst.get_nodeattr("rtlsim_so") == "":
        inst.prepare_rtlsim()
    assert inst.get_nodeattr("rtlsim_so") != "", f"rtlsim not ready for {inst.onnx_node.name}"

    sim = inst.get_rtlsim()
    io_dict = get_tav_io_dict(inst)
    for input_key in io_dict["inputs"]:
        io_dict["inputs"][input_key] = io_dict["inputs"][input_key] * periods_to_simulate

    sname = "_V"
    inst.reset_rtlsim(sim)
    if pre_hook is not None:
        pre_hook(sim)

    # stream tracers for the first input and output stream (not weight streams etc.)
    tracers_in = {k: sim.trace_stream(k + sname) for k in io_dict["inputs"] if "in0" in k}
    tracers_out = {k: sim.trace_stream(k + sname) for k in io_dict["outputs"] if "out0" in k}
    try:
        inst.rtlsim_multi_io(sim, io_dict, sname=sname, batch_size=periods_to_simulate)
    finally:
        # the stream tracers keep their contents after the simulation is closed
        inst.close_rtlsim(sim)

    total_cycle_count = inst.get_nodeattr("cycles_rtlsim")
    period = total_cycle_count // periods_to_simulate
    inst.set_nodeattr("io_chrc_period", period)
    # the stored window is the last two periods before the drain, i.e. the middle two
    # of the default five. With a single simulated frame it is period 0, wrapped.
    store_from = max(0, periods_to_simulate - 3) * period
    span = period * periods_to_store
    wrap = period * periods_to_simulate

    def accumulate(tracer: Any) -> np.ndarray:
        """Return the cumulative token count over the stored window of one stream."""
        txns = [int(c) for c in str(tracer)]
        # the trace must cover every simulated period, the window is indexed modulo it
        txns += [0] * max(0, wrap - len(txns))
        idx = (store_from + np.arange(span)) % wrap
        return np.cumsum(np.asarray(txns)[idx]).astype(np.int32)

    for attr_name, tracers in (("io_chrc_in", tracers_in), ("io_chrc_out", tracers_out)):
        rows = np.stack([accumulate(t) for t in tracers.values()])
        inst.set_nodeattr(attr_name, save_tav_npy(inst, attr_name, rows))


def tav_tolerance(frac: float, scale: float, const: float) -> float:
    """Return ``const + frac * scale``, the two terms a TAV error actually has.

    The proportional term belongs in a budget: a period reproduced to within four
    cycles is excellent on a 100k-cycle frame and useless on a 40-cycle one. The
    constant term is the fixed part of the error (pipeline wind-up, an adder tree's
    depth, the cycle rtlsim's ``cycles_rtlsim // periods_to_simulate`` rounds away),
    which costs the same few cycles whatever the frame length, so the two add.
    """
    return float(const) + float(frac) * float(scale)


def compare_two_chr_funcs(
    a: np.ndarray,
    b: np.ndarray,
    max_allowed_volume_frac: float,
    max_allowed_length_frac: float,
    volume_const: float = 0,
    length_const: float = 0,
) -> bool:
    """Compare an analytical TAV ``a`` against the rtlsim ground truth ``b``.

    ``max_allowed_volume_frac`` is a fraction of the tokens the reference moves in the
    window, ``max_allowed_length_frac`` a fraction of the reference's length; the
    ``*_const`` arguments are the fixed counts that add to the proportional term (see
    ``tav_tolerance``). Output delays due to long pipelines generally do not affect
    FIFO sizes, so a larger relaxation on the output side is expected.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    lower_len = min(len(a), len(b))

    len_allowed = tav_tolerance(max_allowed_length_frac, len(b), length_const)
    if len(a) != len(b):
        len_dif = abs(len(a) - len(b))
        print(
            f"TAV length delta: {len_dif} ({100.0 * len_dif / max(1, len(b)):.2f}% of {len(b)}, "
            f"allowed {len_allowed:.0f})"
        )
        if len_dif > len_allowed:
            return False

    # scale the volume budget by the tokens actually moved, not by the cycle count: a
    # peak cumulative delta is a token count and the two differ by the folding factor
    total_tokens = int(b[lower_len - 1]) if lower_len else 0
    vol_allowed = tav_tolerance(max_allowed_volume_frac, total_tokens, volume_const)
    peak_volume_delta = np.max(np.abs(a[:lower_len] - b[:lower_len])) if lower_len else 0
    print(
        f"TAV peak volume delta: {peak_volume_delta} "
        f"({100.0 * peak_volume_delta / max(1, total_tokens):.2f}% of {total_tokens} tokens, "
        f"allowed {vol_allowed:.0f})"
    )
    return bool(peak_volume_delta <= vol_allowed)


def get_characteristic_fnc(
    model: ModelWrapper, part: str, target_clk_ns: float, strategy: str
) -> "HWCustomOp":
    """Characterize the single node of ``model`` and return its custom op.

    Args:
        model: Single-node model.
        part: FPGA part to specialize and generate IP for.
        target_clk_ns: Target clock period.
        strategy: ``"tree_model"`` (analytical, as the builder does it) or ``"rtlsim"``
            (the test-only reference trace). A node without a tree model gets no
            token access vectors with ``"tree_model"``.
    """
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateCycles())
    if strategy == "tree_model":
        model = model.transform(DeriveTokenAccessVectors())
        return getHWCustomOp(model.graph.node[0])
    assert strategy == "rtlsim", f"unknown characterization strategy {strategy}"
    model = model.transform(PrepareIP(part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    inst = getHWCustomOp(model.graph.node[0])
    derive_token_access_vectors_using_rtlsim(inst)
    return inst


def _characterize_both_ways(
    model: ModelWrapper, node_details: Any, part: str, target_clk_ns: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(chr_in, chr_out, rtlsim_in, rtlsim_out)`` of the single node of ``model``.

    Skips the calling test if this node configuration has no tree model: there is
    nothing to validate then, and the builder leaves such a node uncharacterized.
    """
    print("TAV comparison for", node_details)
    model_rtl = copy.deepcopy(model)
    node_analytical = get_characteristic_fnc(model, part, target_clk_ns, "tree_model")
    if node_analytical.get_nodeattr("io_chrc_in") == "":
        pytest.skip(f"no tree model for this configuration of {node_analytical.onnx_node.op_type}")
    node_rtlsim = get_characteristic_fnc(model_rtl, part, target_clk_ns, "rtlsim")
    return (
        load_tav_npy(node_analytical.get_nodeattr("io_chrc_in"))[0],
        load_tav_npy(node_analytical.get_nodeattr("io_chrc_out"))[0],
        load_tav_npy(node_rtlsim.get_nodeattr("io_chrc_in"))[0],
        load_tav_npy(node_rtlsim.get_nodeattr("io_chrc_out"))[0],
    )


def tree_model_test(
    model: ModelWrapper,
    node_details: Any,
    part: str,
    target_clk_ns: float,
    max_allowed_volume_frac: float,
    max_allowed_length_frac: float,
    volume_const: float = 0,
    length_const: float = 0,
) -> bool:
    """Characterize ``model``'s single node both ways and compare the two TAVs.

    The tolerances are fractions, of the tokens moved and of the frame length
    respectively, with an absolute floor for the fixed part of the error. See
    ``compare_two_chr_funcs``.
    """
    chr_in, chr_out, rtlsim_in, rtlsim_out = _characterize_both_ways(
        model, node_details, part, target_clk_ns
    )
    tolerances = (max_allowed_volume_frac, max_allowed_length_frac, volume_const, length_const)
    input_check = compare_two_chr_funcs(chr_in, rtlsim_in, *tolerances)
    output_check = compare_two_chr_funcs(chr_out, rtlsim_out, *tolerances)
    return input_check and output_check


def compare_chr_funcs_up_to_rotation(
    chr_in: np.ndarray,
    chr_out: np.ndarray,
    rtlsim_in: np.ndarray,
    rtlsim_out: np.ndarray,
    max_allowed_volume_frac: float,
    max_allowed_length_frac: float,
    volume_const: float = 0,
    length_const: float = 0,
) -> tuple[bool, int, int | None, int | None]:
    """Compare a tree model against rtlsim allowing one *shared* phase offset.

    Weaker than ``compare_two_chr_funcs`` in exactly one way: the tree model may place
    the frame anywhere within the period, as long as it places the input and the
    output row at the *same* offset. That is the right assertion for a node whose
    steady-state schedule is known exactly but whose wind-up is not. A model that got
    the read-to-write relationship, the period or the token count wrong still fails,
    and those are what FIFO sizing integrates. Only the absolute placement of the
    frame inside rtlsim's two-period window is forgiven, which is an artefact of where
    the window was cut.

    The length excess is one-sided: rtlsim's period carries a share of the wind-up, so
    a model that reproduces the steady state exactly is *short* by that share and
    never long. Running long is rejected outright.

    Returns:
        ``(ok, rotation, in_delta, out_delta)``.
    """
    a_in, b_in = np.asarray(chr_in), np.asarray(rtlsim_in)
    a_out, b_out = np.asarray(chr_out), np.asarray(rtlsim_out)

    len_dif = len(b_in) - len(a_in)
    len_allowed = tav_tolerance(max_allowed_length_frac, len(b_in), length_const)
    if len_dif < 0 or len_dif > len_allowed:
        print(
            f"TAV length delta: {len_dif} ({100.0 * len_dif / max(1, len(b_in)):.2f}% of "
            f"{len(b_in)}, allowed {len_allowed:.0f})"
        )
        return False, 0, None, None

    def one_period(cum: np.ndarray) -> np.ndarray:
        """Return one period of per-cycle deltas, which is what a rotation acts on."""
        n = len(cum) // 2
        d = np.diff(np.concatenate(([0], np.asarray(cum, dtype=np.int64))))
        return d[:n]

    d_in, d_out = one_period(a_in), one_period(a_out)
    r_in, r_out = one_period(b_in), one_period(b_out)

    # Deliberately no equal-token-count assertion between the two. rtlsim's period is
    # cycles_rtlsim // periods_to_simulate and rounds *up* off the wind-up, so on a
    # short period its window holds a little more than one frame, and demanding
    # equality would fail the model for the reference's rounding.
    n = min(len(d_in), len(r_in))
    ref_in = np.cumsum(r_in[:n])
    ref_out = np.cumsum(r_out[:n])
    best: tuple[int, int] | None = None
    for rot in range(len(d_in)):
        v = int(np.max(np.abs(np.cumsum(np.roll(d_in, rot)[:n]) - ref_in)))
        if best is None or v < best[0]:
            best = (v, rot)
    assert best is not None, "empty token access vector"
    in_delta, rot = best
    out_delta = int(np.max(np.abs(np.cumsum(np.roll(d_out, rot)[:n]) - ref_out)))
    vol_allowed = tav_tolerance(
        max_allowed_volume_frac, max(int(ref_in[-1]), int(ref_out[-1])) if n else 0, volume_const
    )
    print(
        f"TAV rotation: {rot}, peak volume delta in/out: {in_delta}/{out_delta} "
        f"(allowed {vol_allowed:.0f})"
    )
    ok = in_delta <= vol_allowed and out_delta <= vol_allowed
    return ok, rot, in_delta, out_delta


def tree_model_test_up_to_rotation(
    model: ModelWrapper,
    node_details: Any,
    part: str,
    target_clk_ns: float,
    max_allowed_volume_frac: float,
    max_allowed_length_frac: float,
    volume_const: float = 0,
    length_const: float = 0,
) -> bool:
    """Run ``tree_model_test`` with ``compare_chr_funcs_up_to_rotation`` as the check."""
    chr_in, chr_out, rtlsim_in, rtlsim_out = _characterize_both_ways(
        model, node_details, part, target_clk_ns
    )
    ok, _, _, _ = compare_chr_funcs_up_to_rotation(
        chr_in,
        chr_out,
        rtlsim_in,
        rtlsim_out,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    )
    return ok
