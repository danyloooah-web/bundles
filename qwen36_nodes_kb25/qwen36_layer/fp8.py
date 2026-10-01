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

import qk_lt  # built from qk_lt.cu by the validator's CUDA build step
from qwen36_layer import i8x_tail
from qwen36_layer.evidence import branch_evidence

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


_lt_verdict = {}  # kb21: (device, M, N, K) -> True (zero-workspace plan checked equal) / False (stays on _scaled_mm);
# kb24: "fg" (qk_lt.fp8_gemm checked equal)


def linear_q(xq, xs, w):
    """x @ w.T from x's e4m3 rows and row scales (as _q_rows makes them), bf16 out.

    kb21: torch._scaled_mm bit for bit through qk_lt.fp8_linear's zero-workspace cuBLASLt plan (no per-call memsets),
    for shapes compared against _scaled_mm on a real call outside graph capture (as lt.py does for F.linear); an
    unchecked shape under capture, a mismatch or no plan keeps _scaled_mm.
    kb24: shapes where qk_lt.fp8_gemm_prefer() holds (more 128 x 256 tiles than SMs, wave efficiency within 3 % of
    cuBLAS's 128 x 128) first try qk_lt.fp8_gemm (qk_lt.cu fg11: the cuBLASLt recipe written out), checked the same
    way; a mismatch falls back to the zero-workspace plan, then _scaled_mm."""
    wq, ws = _q_weight(w)
    key = (xq.device.index, xq.shape[0], wq.shape[0], xq.shape[1])
    ok = _lt_verdict.get(key)
    if ok is None:
        ref = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
        if torch.cuda.is_current_stream_capturing() or not (xq.is_contiguous() and xs.is_contiguous() and ws.is_contiguous()):
            return ref
        if (qk_lt.fp8_gemm_prefer(xq.shape[0], wq.shape[0]) and xq.shape[1] % 128 == 0 and wq.is_contiguous()
                and xq.data_ptr() % 16 == 0 and wq.data_ptr() % 16 == 0):
            y = qk_lt.fp8_gemm(xq, xs.view(-1), wq, ws.view(-1))
            if y.numel() == ref.numel() and torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                _lt_verdict[key] = "fg"
                return ref
        y = qk_lt.fp8_linear(xq, xs.view(-1), wq, ws.view(-1))
        _lt_verdict[key] = bool(y.numel() == ref.numel() and torch.equal(y.view(torch.int16), ref.view(torch.int16)))
        return ref
    if not ok:
        return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
    if ok == "fg":
        branch_evidence("fg_fp8", xq.shape[0])
        return qk_lt.fp8_gemm(xq, xs.view(-1), wq, ws.view(-1))
    branch_evidence("lt_fp8", xq.shape[0])
    return qk_lt.fp8_linear(xq, xs.view(-1), wq, ws.view(-1))


def linear(x, w):
    """x @ w.T in FP8 (row-wise scales), bf16 out."""
    return linear_q(*_q_rows(x), w)
