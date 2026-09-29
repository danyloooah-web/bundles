"""Prefill projections in FP8 e4m3 -- a PRECISION CHANGE (the GLM crown line's prefill o_proj idea, applied to the
attention qkv / o_proj and the recurrent out_proj at prefill widths).

Each call quantizes its activation rows and the weight's output channels on the fly (amax / 448 per row, e4m3; no
persistent FP8 copy, so nothing is taken from the KV pool) and multiplies them row-wise scaled into bf16 with
torch._scaled_mm. H100, 8192 rows, cold weights: o_proj / out_proj 175 -> 156 us, qkv 439 -> 298 us; ~3.75 % row
error at the projection output against the bf16 GEMM (Gaussian inputs).
"""

import torch
import triton
import triton.language as tl

from qwen36_layer import i8x_tail

# the FP8 GEMMs' accumulation: fast (tensor-core, ~2e-3 of the output next to fp8's ~3.75 %) when True
FAST_ACCUM = False  # k2fp8y34: y18 prefill numerics (fast accumulation cost ~0.016 of the 64k-context node-audit margin: y26 live slot_audit_failed)

MIN_ROWS = 2048


@triton.jit(do_not_specialize=["rows"])
def _rowwise_fp8(X, Y, S, rows, K: tl.constexpr, BLOCK: tl.constexpr):
    """Y[r] = e4m3(X[r] / s_r), S[r] = s_r = max|X[r]| / 448: one program per row (the GLM crown line's quantizer)."""
    r = tl.program_id(0)
    base = r.to(tl.int64) * K
    amax = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(0, K, BLOCK):
        amax = tl.maximum(amax, tl.abs(tl.load(X + base + k + tl.arange(0, BLOCK)).to(tl.float32)))
    scale = tl.maximum(tl.max(amax, axis=0) / 448.0, 1e-12)
    inv = 1.0 / scale
    for k in range(0, K, BLOCK):
        v = tl.load(X + base + k + tl.arange(0, BLOCK)).to(tl.float32) * inv
        tl.store(Y + base + k + tl.arange(0, BLOCK), v.to(tl.float8e4nv))
    tl.store(S + r, scale)


def _q_rows(t):
    rows, k = t.shape
    y = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=t.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=t.device)
    _rowwise_fp8[(rows,)](t, y, s, rows, K=k, BLOCK=min(k, 2048))
    return y, s


def eligible(x, w):
    return (x.ndim == 2 and w.ndim == 2 and x.shape[0] >= MIN_ROWS and x.shape[1] == w.shape[1]
            and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_contiguous() and w.is_contiguous()
            and x.shape[1] % 16 == 0 and w.shape[0] % 16 == 0)


_weights = {}  # (data_ptr, shape, device) -> the weight's (e4m3 rows, scales), resident in an i8x idle tail


def _q_weight(w):
    """_q_rows(w), kept resident after the first eager call that finds room in the idle device bytes behind the
    i8x expert copies (i8x_tail.py): the same bytes every call, so prefill chunks skip the pass, and no free memory
    is taken (a free-memory copy starved the king's post-capture DENSE8 late map). Otherwise quantized per call."""
    key = (w.data_ptr(), tuple(w.shape), w.device)
    hit = _weights.get(key)
    if hit is not None:
        return hit
    if not torch.cuda.is_current_stream_capturing():
        rows, k = w.shape
        y = i8x_tail.tensor((rows, k), torch.float8_e4m3fn, w.device)
        s = None if y is None else i8x_tail.tensor((rows, 1), torch.float32, w.device)
        if s is not None:
            _rowwise_fp8[(rows,)](w, y, s, rows, K=k, BLOCK=min(k, 2048))
            _weights[key] = (y, s)
            return y, s
    return _q_rows(w)


def linear_q(xq, xs, w):
    """x @ w.T from x's e4m3 rows and row scales (as _q_rows makes them), bf16 out."""
    wq, ws = _q_weight(w)
    return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16, use_fast_accum=FAST_ACCUM)


def linear(x, w):
    """x @ w.T in FP8 (row-wise scales), bf16 out."""
    return linear_q(*_q_rows(x), w)
