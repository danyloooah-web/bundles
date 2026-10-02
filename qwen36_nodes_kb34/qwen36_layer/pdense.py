"""Lane PDENSE (PRECISION CHANGE): the GDN prefill input projection [q | k | v | z] = x W_qkvz^T in INT8.

Chunks of >= MIN_ROWS rows in the recurrent layers of LAYERS run qk_inproj.cu's INT8 kernel (namespace qin8, was qk_inproj8.cu): the parent's fused projection and
conv front (qk_inproj.cu) with int8 operands and exact int32 sums. Per chunk, a per-input-channel smoothing vector
s = amax_x^ALPHA / amax_W^(1 - ALPHA) (amax_x over the chunk's rows, amax_W over the weight's output rows) moves the
residual stream's outlier channels from the activations into the weight: x W^T = (x / s)(W s)^T exactly, then x / s
is quantized per row and W s per output channel (amax / 127, round to nearest). The int8 weight is derived from the
bf16 weight at run time into a reused scratch buffer in the idle device bytes behind the i8x expert copies
(i8x_tail.py), so nothing is taken from the engine's free memory; the bf16 weight is only read.

Measured on the long check prompt (8 x 8192-row chunks, every call audited): every layer's worst MoE-output window
stayed >= 0.81 with all 30 recurrent layers on (parent 0.834-0.999 on the same layers); FP8 e4m3 per row / channel
(amax / 448) instead read 0.69-0.84 on layers 0-9 (the residual stream's outlier channels cost e4m3 ~2 % per row
where smoothed int8 costs ~1 %).
"""

import torch
import triton
import triton.language as tl

from qwen36_layer import i8x_tail
from qwen36_layer.evidence import branch_evidence

# v1 switch (lane PDENSE): False = the parent's BF16 GDN prefill in_proj (qk_inproj.cu) everywhere.
ENABLED = True
MIN_ROWS = 2048
ALPHA = 0.7
# pd38 switch. False = v1b: layer 38's prefill in_proj stays bf16 (pd3: its residual window was the weakest touched
# window on the short check set, 0.769 with INT8 at 38). True = layer 38 also runs the INT8 in_proj and, to pay for it,
# its prefill out_proj runs BF16 instead of the parent's FP8 (layer.py reads OUT_BF16_LAYERS): at 38 the recurrent
# block carries a large share of the residual row, and the FP8 out_proj error is what the INT8 in_proj error stacked on.
PD38 = True
# port (on b0e9d402): recurrent layers whose prefill in_proj stays BF16 (precision margin on top of the parent's INT8 MoE)
BF16_INPROJ_LAYERS = frozenset()
SKIP_LAYERS = (frozenset() if PD38 else frozenset({38})) | BF16_INPROJ_LAYERS
OUT_BF16_LAYERS = frozenset({38}) if (PD38 and ENABLED) else frozenset()  # port: nothing when the lane is off
LAYERS = frozenset(i for i in range(40) if i % 4 != 3 and i not in SKIP_LAYERS)
MARKER = "pdense_inproj_i8"
# kb25e: the chunk whose input norm already made the column amax (qk_moe_prefill_i8.add_norm_amax)
AMAX_MARKER = "pdense_norm_amax"
AMAX_REDO_MARKER = "pdense_norm_amax_redo"  # ... with padded rows: zeroed, _col_amax over the live rows
SEG_MARKER = "pdense_prep_segment"  # k28: the chunk whose quantized rows prepare_seg made in the graph segment
# the input norm makes it from this many rows on (isolated H100, norm + _col_amax -> add_norm_amax: 8192 rows 57.2 ->
# 48.7 us, 4096 28.6 -> 25.8, 2560 19.8 -> 17.6, 2048 14.9 -> 15.3: its end-of-kernel atomics cost what it saves)
NORM_AMAX_MIN_ROWS = 2560
K = 2048
N = 12288
ROWS_PER_PROGRAM = 1
# _col_amax launch: 512-row x 64-column blocks (isolated, 8192 x 2048: 17 us against 27 for 64 x 256)
AMAX_ROWS, AMAX_COLS = 512, 64


@triton.jit(do_not_specialize=["rows"])
def _col_amax(X, A, rows, K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr, PDL: tl.constexpr):
    """A[c] = max(A[c], max_r |X[r, c]|) over this program's row block (A starts at zero; |x| >= 0 bit-orders)."""
    if PDL:  # kb20: programmatic dependent launch (wait before the first dependent load)
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r0 = tl.program_id(0) * BLOCK_R
    c = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    r = r0 + tl.arange(0, BLOCK_R)
    x = tl.load(X + r[:, None].to(tl.int64) * K + c[None, :], mask=(r < rows)[:, None], other=0.0)
    tl.atomic_max(A + c, tl.max(tl.abs(x.to(tl.float32)), axis=0))


@triton.jit
def _smooth_vec(AX, AW, F, K: tl.constexpr, ALPHA: tl.constexpr, PDL: tl.constexpr):
    """F[0, c] = 1 / sm_c (for x), F[1, c] = sm_c (for W), sm = amax_x^a / amax_W^(1-a); then AX = 0 again (kb20: AX is
    the persistent per-device amax buffer, so no per-chunk zero fill)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    c = tl.arange(0, K)
    ax = tl.maximum(tl.load(AX + c), 1e-6)
    aw = tl.maximum(tl.load(AW + c), 1e-6)
    sm = tl.exp2(ALPHA * tl.log2(ax) - (1.0 - ALPHA) * tl.log2(aw))
    tl.store(F + c, 1.0 / sm)
    tl.store(F + K + c, sm)
    tl.store(AX + c, tl.zeros([K], dtype=tl.float32))


@triton.jit(do_not_specialize=["rows"])
def _rows_q8(X, F, Y, S, rows, K: tl.constexpr, R: tl.constexpr):
    """Y[r] = round(X[r] * f / s_r), s_r = max|X[r] * f| / 127 (round half away from zero), R rows per program."""
    c = tl.arange(0, K)
    f = tl.load(F + c)
    for i in tl.static_range(R):
        r = tl.program_id(0) * R + i
        if r < rows:
            x = tl.load(X + r.to(tl.int64) * K + c).to(tl.float32) * f
            scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-12)
            q = x * (1.0 / scale)
            q = tl.where(q >= 0, tl.floor(q + 0.5), -tl.floor(0.5 - q))
            tl.store(Y + r.to(tl.int64) * K + c, tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8))
            tl.store(S + r, scale)


@triton.jit(do_not_specialize=["rows", "rows_w"])
def _rows_q8_xw(X, W, F, YX, SX, YW, SW, rows, rows_w, K: tl.constexpr, PDL: tl.constexpr):
    """kb20: _rows_q8 of x (F[0]) and of the weight (F[1]) in one launch: program p < rows quantizes x row p, the rest
    weight row p - rows, each with _rows_q8's arithmetic (R = 1)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    c = tl.arange(0, K)
    p = tl.program_id(0)
    if p < rows:
        f = tl.load(F + c)
        x = tl.load(X + p.to(tl.int64) * K + c).to(tl.float32) * f
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-12)
        q = x * (1.0 / scale)
        q = tl.where(q >= 0, tl.floor(q + 0.5), -tl.floor(0.5 - q))
        tl.store(YX + p.to(tl.int64) * K + c, tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8))
        tl.store(SX + p, scale)
    else:
        r = p - rows
        f = tl.load(F + K + c)
        x = tl.load(W + r.to(tl.int64) * K + c).to(tl.float32) * f
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-12)
        q = x * (1.0 / scale)
        q = tl.where(q >= 0, tl.floor(q + 0.5), -tl.floor(0.5 - q))
        tl.store(YW + r.to(tl.int64) * K + c, tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8))
        tl.store(SW + r, scale)


def eligible(layer_id, x, w):
    return (ENABLED and layer_id in LAYERS and x.ndim == 2 and x.shape[0] >= MIN_ROWS and x.shape[1] == K
            and tuple(w.shape) == (N, K) and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and x.is_contiguous() and w.is_contiguous())


_aw = {}  # (data_ptr, device) -> the weight's per-input-channel amax (fp32 [K])
_scratch = {}  # device -> (int8 [N, K], fp32 [N]) reused by every layer's chunk (one stream)


def _weight_amax(w):
    key = (w.data_ptr(), w.device)
    hit = _aw.get(key)
    if hit is None:
        hit = torch.zeros(K, dtype=torch.float32, device=w.device)
        _col_amax[(triton.cdiv(N, AMAX_ROWS), K // AMAX_COLS)](w, hit, N, K=K, BLOCK_R=AMAX_ROWS, BLOCK_K=AMAX_COLS, PDL=False,
                                                                num_warps=8)
        if not torch.cuda.is_current_stream_capturing():
            _aw[key] = hit
    return hit


def _weight_scratch(device):
    hit = _scratch.get(device)
    if hit is None:
        q = i8x_tail.tensor((N, K), torch.int8, device)
        s = None if q is None else i8x_tail.tensor((N,), torch.float32, device)
        if s is None:
            return torch.empty((N, K), dtype=torch.int8, device=device), torch.empty(N, dtype=torch.float32, device=device)
        hit = _scratch[device] = (q, s)
    return hit


_ax = {}  # kb20: device -> the persistent fp32 [K] activation amax buffer (zero between chunks: _smooth_vec re-zeroes it)


def amax_buffer(device):
    """kb25e: ``device``'s persistent activation amax buffer (zero between chunks), which the layer's input norm
    (qk_moe_prefill_i8.add_norm_amax) fills with _col_amax's maxima of the normed rows for prepare(ax_ready=True);
    None under graph capture before an eager call made it (it must not come from a graph's private pool)."""
    ax = _ax.get(device)
    if ax is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ax = _ax[device] = torch.zeros(K, dtype=torch.float32, device=device)
    return ax


def discard_amax(device):
    """kb25e: zero the buffer again when the input norm filled it and no prepare() consumed it."""
    ax = _ax.get(device)
    if ax is not None:
        ax.zero_()


def prepare(x, w, live=None, ax_ready=False, prep=None):
    """(xq int8 [T, K], sa fp32 [T], wq int8 [N, K], sw fp32 [N]) with x @ w.T ~ (xq sa)(wq sw)^T; the smoothing
    vector reads the first ``live`` rows (the batch's real tokens, not a capture bucket's padding).
    kb20: three launches with programmatic dependent launch (column amax, smoothing vector, both row quantizations in
    one grid) instead of five (no zero fill, x and weight rows in one launch); the same arithmetic per element.
    kb25e: ``ax_ready`` = the input norm that wrote x already max-ed |x| of all its rows into the amax buffer
    (add_norm_amax: _col_amax's maxima, exact in any order): with every row live the _col_amax launch is skipped;
    when a capture bucket padded rows in, the buffer is zeroed and _col_amax reads the live rows as before.
    k28: ``prep`` = prepare_seg's (xq, sa, wq, sw) of every row, made in the prefill graph segment: returned as they are
    when every row is live; with padded rows the segment's maxima covered them, so the live rows run the whole pass
    again from the buffer prepare_seg's _smooth_vec left zero (kb27's padded-chunk arithmetic)."""
    rows = x.shape[0]
    live = rows if live is None else max(1, min(int(live), rows))
    if prep is not None:
        if live == rows:
            branch_evidence(SEG_MARKER, rows)
            return prep
        ax_ready = False
    aw = _weight_amax(w)
    ax = _ax.get(x.device)
    if ax is None:
        ax = _ax[x.device] = torch.zeros(K, dtype=torch.float32, device=x.device)
    if ax_ready and live == rows:
        branch_evidence(AMAX_MARKER, rows)
    else:
        if ax_ready:
            ax.zero_()
            branch_evidence(AMAX_REDO_MARKER, live)
        _col_amax[(triton.cdiv(live, AMAX_ROWS), K // AMAX_COLS)](x, ax, live, K=K, BLOCK_R=AMAX_ROWS, BLOCK_K=AMAX_COLS,
                                                                 PDL=True, num_warps=8, launch_pdl=True)
    return _quantize(x, w, ax, aw)


def _quantize(x, w, ax, aw):
    """prepare's launches after its amax step: the smoothing vector (which re-zeroes ``ax``), then x's and the
    weight's rows in one grid."""
    rows = x.shape[0]
    f = torch.empty((2, K), dtype=torch.float32, device=x.device)
    _smooth_vec[(1,)](ax, aw, f, K=K, ALPHA=ALPHA, PDL=True, num_warps=4, launch_pdl=True)
    xq = torch.empty((rows, K), dtype=torch.int8, device=x.device)
    sa = torch.empty(rows, dtype=torch.float32, device=x.device)
    wq, sw = _weight_scratch(x.device)
    _rows_q8_xw[(rows + N,)](x, w, f, xq, sa, wq, sw, rows, N, K=K, PDL=True, num_warps=4, launch_pdl=True)
    return xq, sa, wq, sw


def prepare_seg(x, w):
    """k28: prepare(x, w, rows, ax_ready=True)'s launches after its amax step - _smooth_vec and _rows_q8_xw over every
    row, the same launches and arguments - issued in the prefill graph segment right after the first layer's plain
    input norm filled the amax buffer (layer.py), so the in_proj break launches neither; returns (xq, sa, wq, sw) for
    that break's prepare(prep=...)."""
    return _quantize(x, w, _ax[x.device], _weight_amax(w))


def col_amax(x, ax):
    """k28: ax = max(ax, column maxima of |x| over every row of x): prepare's _col_amax pass for a caller that holds the
    rows in the prefill graph segment before the in_proj break (the first layer's plain input norm, layer.py), where
    add_norm_amax serves the later layers. Exact in any order, so the same maxima as prepare's launch."""
    rows = x.shape[0]
    _col_amax[(triton.cdiv(rows, AMAX_ROWS), K // AMAX_COLS)](x, ax, rows, K=K, BLOCK_R=AMAX_ROWS, BLOCK_K=AMAX_COLS,
                                                             PDL=False, num_warps=8)


def inproj(x, w_qkvz, qkv_out):
    """(mixed_qkv [T, qkv_out], z [T, rest]) of the same INT8 product unfused (torch._int_mm), bf16 out."""
    xq, sa, wq, sw = prepare(x, w_qkvz)
    y = (torch._int_mm(xq, wq.t()).float() * sa[:, None] * sw[None, :]).to(torch.bfloat16)
    return y[:, :qkv_out].contiguous(), y[:, qkv_out:].contiguous()
