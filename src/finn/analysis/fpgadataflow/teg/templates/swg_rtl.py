# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Class B2: RTL ConvolutionInputGenerator, default template
(``finn-rtllib/swg/swg_template_default.sv``, controller in ``swg_common.sv``; parameters from
``convolutioninputgenerator_rtl.py::prepare_codegen_default``).

Three chains per frame:

* ``R_k`` (``k = 0..h*w*cf-1``): input reads, one AXI handshake each (``read_ok``, ``:139``),
* ``F_j`` (``j = 0..out_h*out_w*kh*kw*cf-1``): buffer fetches (``fetch_cmd``, ``:129``);
  ``c(j)`` is the input element fetched at ``j`` (the ``Current_elem`` trajectory),
  ``fenw(j)`` the ``First_elem_next_window`` register after fetch ``j``,
* ``W_j``: output writes (``write_ok``, ``:124``).

Arcs and edges, each a transcription of one line of the RTL:

* ``F_j >= R_{c(j)} + 1``: fetch only if ``Current_elem <= Newest_buffered_elem`` (``:129``;
  ``Newest_buffered_elem`` is registered, ``:165``).
* ``F_j >= W_{j-1} + 0`` and ``W_j >= F_j + 1``: the single output register (``Write_cmd``,
  ``:125-126, :216``): internal edge ``F -> W`` of capacity 1, ``lf = 1``, ``lb = 0``.
* ``R_k >= F_{j*(k)} + 1`` for ``k >= BUF``: the slot of element ``k - BUF`` may be overwritten
  only when that element is dead, ``Newest - (BUF-1) < First_elem_next_window`` and
  ``< Current_elem`` (``:132-138``), evaluated on the registers after fetch ``j``
  (``Current_elem = c(j+1)``, ``First_elem_next_window = fenw(j)``); ``Fetching_done`` lifts
  the guard after the last fetch (``:134``). ``j*(k)`` is the first fetch after which both hold.
* ``R_0^{n+1} >= W_last^n + 1`` and ``F_0^{n+1} >= W_last^n + 1``: all counters reset after
  the frame's last write (``:222-233``), no cross-frame overlap: edges ``W -> R`` and
  ``W -> F`` with one initial token.

Depthwise windows with a channel fold ``cf > 1`` (``IS_DEPTHWISE``) rearrange the controller's
loop nest (``prepare_codegen_default``: the ``kh`` counter iterates the channel folds, ``kw``
the kernel rows, ``simd`` the kernel columns): every channel fold fetches its complete ``kh*kw``
window before the next fold, ``ELEM_PER_WINDOW = kh*kw`` and ``First_elem_next_window``
advances by 1 after each fold except the last (``swg_common.sv:73-75``,
``tail_incr_inner_condition``), where ``TAIL_INCR_W/H/LAST`` are reduced by ``cf - 1`` so
that the register lands on the next window's base.

Imperfect strides (``skip_rows``/``skip_columns`` in ``prepare_codegen_default``) leave
trailing input rows and columns that are read but never fetched: the window positions are
unchanged, ``LAST_WRITE_ELEM`` stops the fetches after the last window and ``Fetching_done``
lifts the slot-reuse guard for the remaining reads (``:140``).

Restrictions of this transcription (raise for anything else): dilation 1, ``M = 1``.

Parallel template (``swg_template_parallel.sv``, selected for 1x1 kernels and
``parallel_window = 1``): the buffer is a shift register, the window is read from fixed taps
and there is no output register (class C: ``in0_V_V_TREADY = read_ok`` is combinational in
``TVALID``). Two chains, ``R_k`` (``read_ok``, ``:135``) and ``W_j`` (``write_ok``, ``:126``),
where ``c(j)`` is the newest input element of window ``j`` (``Current_elem``):

* ``W_j >= R_{c(j)} + 1``: ``write_cmd`` needs ``Current_elem <= Newest_buffered_elem`` (``:124``;
  registered, ``:157``).
* ``R_k >= W_{J(k)} + d``: ``read_cmd`` needs ``Newest_buffered_elem <= Current_elem`` (``:131``),
  i.e. every window with ``c(j) < k - 1`` written in an earlier cycle (``J(k)`` the last such
  window, ``d = 1``); when ``c(J(k)) = k - 1`` the read is additionally blocked while that
  window's write is pending (``!write_blocked``, ``:134``), so the write may be in the same cycle
  (``d = 0``). After the last write ``Writing_done`` frees the reads (``:131``).
* ``R_0^{n+1} >= W_last^n + 1``: counters reset at the later of the last read and the last write
  (``:158-181``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Arc, Chain, FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def swg_default_trajectory(
    h: int, w: int, kh: int, kw: int, sh: int, sw: int, cf: int, depthwise: bool = False
) -> tuple[list[int], list[int], int, int]:
    """Return ``(c, fenw, buffer_min_size, n_in)`` of the default SWG for these parameters.

    ``c[j]`` is the input element fetched at fetch ``j``; ``fenw[j]`` the value of
    ``First_elem_next_window`` after fetch ``j`` (the first fetch of a window advances it by
    ``tail_incr`` to the base of the next window: ``swg_common.sv:74-78`` with
    ``TAIL_INCR_W/H/LAST`` from ``prepare_codegen_default``). For depthwise windows with
    ``cf > 1`` the channel fold is the outer loop of the window and the register advances by
    one per fold (see the module docstring).
    """
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    buffer_min_size = ((kh - 1) * w + (kw - 1) + 1) * cf  # prepare_codegen_default:405
    n_in = h * w * cf  # LAST_READ_ELEM + 1
    dw = depthwise and cf > 1  # IS_DEPTHWISE (prepare_codegen_default:466-478)
    c: list[int] = []
    fenw: list[int] = []
    for oy in range(out_h):
        for ox in range(out_w):
            base = (oy * sh * w + ox * sw) * cf
            if ox + 1 < out_w:
                nxt = base + sw * cf  # TAIL_INCR_W (+ cf - 1 folds of 1 when depthwise)
            elif oy + 1 < out_h:
                nxt = ((oy + 1) * sh * w) * cf  # TAIL_INCR_H
            else:
                nxt = base + buffer_min_size - 1  # TAIL_INCR_LAST
            if dw:
                for s in range(cf):
                    # first fetch of fold s: tail_incr = 1 while Counter_loop_kh >= 0
                    reg = base + s + 1 if s < cf - 1 else nxt
                    for ky in range(kh):
                        for kx in range(kw):
                            c.append(base + (ky * w + kx) * cf + s)
                            fenw.append(reg)
            else:
                for ky in range(kh):
                    for kx in range(kw):
                        for s in range(cf):
                            c.append(base + (ky * w + kx) * cf + s)
                            fenw.append(nxt)
    return c, fenw, buffer_min_size, n_in


def swg_default(
    prefix: str,
    in_edge: str,
    out_edge: str,
    h: int,
    w: int,
    k: int | tuple[int, int],
    stride: int | tuple[int, int],
    cf: int,
    buf_total: int | None = None,
    depthwise: bool = False,
) -> OpModel:
    """Build the chains of one default-template SWG instance.

    Args:
        prefix: chain name prefix.
        in_edge: external edge feeding ``R``.
        out_edge: external edge written by ``W``.
        h: input feature map height.
        w: input feature map width.
        k: kernel size (``kh, kw``).
        stride: stride (``sh, sw``).
        cf: channel fold ``IFMChannels / SIMD``.
        buf_total: ``BUF_ELEM_TOTAL``; defaults to ``buffer_min_size + 1``
            (``get_buffer_depth`` for stride 1).
        depthwise: depthwise window order (``IS_DEPTHWISE``).
    """
    kh, kw = k if isinstance(k, tuple) else (k, k)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    c, fenw, buffer_min_size, n_in = swg_default_trajectory(h, w, kh, kw, sh, sw, cf, depthwise)
    buf = buffer_min_size + 1 if buf_total is None else buf_total
    n_out = len(c)

    r = Chain(f"{prefix}.R")
    f = Chain(f"{prefix}.F")
    wch = Chain(f"{prefix}.W")
    pipe = f"{prefix}.out_reg"
    restart_r = f"{prefix}.restart_r"
    restart_f = f"{prefix}.restart_f"

    # R: input reads; event 0 also consumes the frame-restart token
    r.event(1, reads=[in_edge, restart_r])
    r.events(n_in - 1, 1, reads=[in_edge])
    # F: fetches; event 0 consumes the restart token; every fetch fills the output register
    f.event(1, reads=[restart_f], writes=[pipe])
    f.events(n_out - 1, 1, writes=[pipe])
    # W: writes; the last one releases the restart tokens
    wch.events(n_out - 1, 1, reads=[pipe], writes=[out_edge])
    wch.event(1, reads=[pipe], writes=[out_edge, restart_r, restart_f])

    # (a) fetch j needs its element in the buffer (swg_template_default.sv:129, :165)
    for j in range(n_out):
        f.add_arc(j, Arc(r.name, c[j], 0, 1))
    # (d) slot reuse (swg_template_default.sv:132-138): registers after fetch j hold
    #     Current_elem = c[j+1] and First_elem_next_window = fenw[j]; after the last fetch
    #     Fetching_done lifts the guard.
    nxt_c = [*c[1:], n_in]
    nxt_f = [*fenw[:-1], n_in]
    jstar: list[int] = []  # per k >= buf
    j = 0
    for kk in range(buf, n_in):
        dead = kk - buf
        while not (nxt_c[j] > dead and nxt_f[j] > dead):
            j += 1
        jstar.append(j)
        r.add_arc(kk, Arc(f.name, j, 0, 1))

    # history windows: how far the referenced chain can run ahead of the referenced event
    # R history for arcs (a): R.count <= Kmax(j) = max{k : j*(k) <= j-1} + 1 (>= buf)
    win_r = 0
    kmax = buf
    for jj in range(n_out):
        while kmax - buf < len(jstar) and jstar[kmax - buf] <= jj - 1:
            kmax += 1
        win_r = max(win_r, kmax - c[jj])
    # F history for arcs (d): F.count <= Jmax(k) = first j with c(j) >= k
    win_f = 0
    jmax = 0
    for idx, kk in enumerate(range(buf, n_in)):
        while jmax < n_out and c[jmax] < kk:
            jmax += 1
        win_f = max(win_f, jmax - jstar[idx])
    r.history_window = win_r + 2
    f.history_window = win_f + 2

    edges = [
        # (b)/(c) single output register, same-cycle handover (:125-126, :216)
        FIFOEdge(pipe, f.name, wch.name, depth=1, lf=1, lb=0),
        # (e) frame restart (:222-233)
        FIFOEdge(restart_r, wch.name, r.name, depth=None, lf=1, lb=1, initial_tokens=1),
        FIFOEdge(restart_f, wch.name, f.name, depth=None, lf=1, lb=1, initial_tokens=1),
    ]
    return OpModel(
        chains=[r, f, wch],
        internal_edges=edges,
        inputs=[r.name],
        outputs=[wch.name],
        notes={
            "buffer_min_size": buffer_min_size,
            "BUF": buf,
            "n_in": n_in,
            "n_out": n_out,
            "out_dim": ((h - kh) // sh + 1, (w - kw) // sw + 1),
            "history_window": (win_r, win_f),
            "depthwise": depthwise,
        },
    )


def swg_parallel_trajectory(
    h: int, w: int, kh: int, kw: int, sh: int, sw: int, cf: int
) -> tuple[list[int], int]:
    """Return ``(c, n_in)`` of the parallel SWG: ``c[j]`` is the newest input element of
    window ``j`` (``Current_elem``: ``FIRST_WRITE_ELEM = buffer_min_size - 1`` plus the
    controller's ``HEAD_INCR_KH/W/H`` per write, ``prepare_codegen_parallel:604-667``); the
    channel folds of one window are consecutive."""
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    buffer_min_size = ((kh - 1) * w + (kw - 1)) * cf + 1  # prepare_codegen_parallel:585
    c: list[int] = []
    for oy in range(out_h):
        for ox in range(out_w):
            base = (oy * sh * w + ox * sw) * cf + buffer_min_size - 1
            c.extend(base + s for s in range(cf))
    return c, h * w * cf


def swg_parallel(
    prefix: str,
    in_edge: str,
    out_edge: str,
    h: int,
    w: int,
    k: int | tuple[int, int],
    stride: int | tuple[int, int],
    cf: int,
) -> OpModel:
    """Build the chains of one parallel-template SWG instance (see the module docstring)."""
    kh, kw = k if isinstance(k, tuple) else (k, k)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    c, n_in = swg_parallel_trajectory(h, w, kh, kw, sh, sw, cf)
    n_out = len(c)
    r = Chain(f"{prefix}.R")
    wch = Chain(f"{prefix}.W")
    restart = f"{prefix}.restart"
    r.event(1, reads=[in_edge, restart])
    r.events(n_in - 1, 1, reads=[in_edge])
    wch.events(n_out - 1, 1, writes=[out_edge])
    wch.event(1, writes=[out_edge, restart])
    # W_j >= R_{c(j)} + 1 (swg_template_parallel.sv:124, :157)
    for j in range(n_out):
        wch.add_arc(j, Arc(r.name, c[j], 0, 1))
    # R_k >= W_{J(k)} + d, J(k) = max{j : c(j) <= k - 1} (:131, :134)
    j = -1
    for kk in range(n_in):
        while j + 1 < n_out and c[j + 1] <= kk - 1:
            j += 1
        if j >= 0:
            r.add_arc(kk, Arc(wch.name, j, 0, 0 if c[j] == kk - 1 else 1))
    # R may run ahead of W_j up to element c(j+1) (+1); W ahead of R_k up to J(k) + 1
    max_incr = max((c[i + 1] - c[i] for i in range(n_out - 1)), default=1)
    r.history_window = max_incr + 4
    wch.history_window = 4
    edges = [FIFOEdge(restart, wch.name, r.name, depth=None, lf=1, lb=1, initial_tokens=1)]
    return OpModel(
        chains=[r, wch],
        internal_edges=edges,
        inputs=[r.name],
        outputs=[wch.name],
        notes={
            "template": "parallel",
            "n_in": n_in,
            "n_out": n_out,
            "out_dim": ((h - kh) // sh + 1, (w - kw) // sw + 1),
        },
    )


@register("ConvolutionInputGenerator", "rtl")
def swg_rtl(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """Build the model of a ``ConvolutionInputGenerator_rtl`` node (default or parallel
    template, ``select_impl_style``)."""
    if tuple(node.get_nodeattr("Dilation")) != (1, 1):
        raise FINNUserError("SWG template: dilation not supported yet")
    if node.get_nodeattr("M") != 1:
        raise FINNUserError("SWG template: M > 1 (multiple input pixels per cycle) not supported")
    h, w = node.get_nodeattr("IFMDim")
    kh, kw = node.get_nodeattr("ConvKernelDim")
    sh, sw = node.get_nodeattr("Stride")
    cf = node.get_nodeattr("IFMChannels") // node.get_nodeattr("SIMD")
    if node.select_impl_style() == "parallel":
        return swg_parallel(prefix, in_edges[0], out_edges[0], h, w, (kh, kw), (sh, sw), cf)
    return swg_default(
        prefix,
        in_edges[0],
        out_edges[0],
        h,
        w,
        (kh, kw),
        (sh, sw),
        cf,
        node.get_buffer_depth(),
        depthwise=bool(node.get_nodeattr("depthwise")),
    )
