"""Lane LMHEAD8: a W8A16 lm_head GEMM for Qwen3.6-35B-A3B (delivered untested -- no GPU touched writing this).

logits[M, N] = x[M, K] (bf16) @ dequant(q[N, K] int8, scale)^T, fp32 accumulate, bf16 out, where
N = 248320 vocab rows, K = 2048 hidden, M = the verify/decode row batch (8, 16, 96, 128 in the arena).

- ``quantize_rows(w)``: symmetric round-to-nearest int8 with one fp32 scale per vocab row
  (``scale = max|w_row| / 127``), or one per ``group`` columns of a row when ``group`` is given.
  One Triton program per (row, group): no temporaries beyond the int8 copy and its scales, so it
  can run lazily inside the engine's small post-pool headroom.
- ``quantize_rows_torch(w)``: the same quantizer in torch (chunked), the reference the bench checks.
- ``w8a16_linear(x, q, scale)``: the GEMM. It computes the transposed tile
  ``out^T[BLOCK_N, BLOCK_M] = q_tile @ x_tile^T`` ("swap AB"): the vocab dimension is the MMA's
  M side (>= 64 rows per tile), the tiny token batch is its N side, so 8/16-row batches waste no MMA
  width on padding rows. The grid is 1-D over vocab tiles (all SMs stream disjoint weight rows), with
  the M blocks of one vocab tile adjacent so a >128-row batch re-reads that tile from L2, not HBM.
  int8 weight tiles are loaded contiguously along K (16-byte vector loads) and converted to bf16 in
  registers; int8 values are exact in bf16, so the only rounding is fp32 accumulation, the per-row
  scale and the bf16 store. Per-row scales multiply the fp32 accumulator once at the end.

No autotune: a config is picked from ``CONFIGS`` by row count (Triton's autotuner benchmarks with
synchronizes, which is illegal under CUDA-graph capture). ``bench_lmhead8.py`` sweeps configs on the
H100 and reports the best per bucket; the table here is a first guess until that runs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    import qk_proj_decode as _qpd  # kb8: lmh16_w8a16 (qwen36_layer's qk_proj_decode.cu)
except ImportError:
    _qpd = None

# (max_rows, BLOCK_N, BLOCK_M, BLOCK_K, num_warps, num_stages): first row whose max_rows >= M wins.
# The sweep winners of bench_lmhead8.py on the H100 (2026-09-25, Triton 3.7.1, torch 2.13, the real
# [248320, 2048] lm_head; runs/20260925_qwen_king_b70160ce/evidence/q6/dev/lmhead8_1/bench_lmhead8.json),
# CUDA-graph median us, stock cuBLAS BF16 in brackets. A pure function of the row count: graph-safe.
CONFIGS = (
    (16, 64, 16, 128, 4, 3),    # 183.6/185.7/188.5/192.8 at 2/4/8/16 rows [354.9/356.4/358.7/362.6]
    (32, 64, 32, 128, 4, 3),    # 204.6/206.0 at 24/32 rows [364.8/367.1]
    (96, 64, 128, 128, 4, 3),   # 334.0 at 96 rows [386.0]; 33..64 rows (not in the arena) not timed
    (128, 64, 128, 64, 4, 3),   # 353.3 at 128 rows [393.9]
    (256, 64, 128, 64, 4, 3),   # 129..256 rows (two M blocks per vocab tile): not timed
)
MAX_ROWS = CONFIGS[-1][0]


def pick_config(rows: int) -> tuple[int, int, int, int, int]:
    for max_rows, *config in CONFIGS:
        if rows <= max_rows:
            return tuple(config)
    raise ValueError(f"w8a16_linear: {rows} rows is above the {MAX_ROWS}-row table")


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om, stride_sn, num_m,
                  BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUPED: tl.constexpr, EVEN_N: tl.constexpr, ROUND_BF16: tl.constexpr = False):
    pid = tl.program_id(0)
    pid_n = pid // num_m
    pid_m = pid % num_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_m = offs_m < M
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]  # x^T tile [BLOCK_K, BLOCK_M]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):  # the wrapper asserts K % BLOCK_K == 0
        if EVEN_N:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, mask=mask_n[:, None], other=0)
        xt = tl.load(x_ptrs, mask=mask_m[None, :], other=0.0)
        if GROUPED:
            s = tl.load(s_ptr + offs_n.to(tl.int64) * stride_sn + k // BLOCK_K, mask=mask_n, other=0.0)
            acc += tl.dot(w.to(tl.bfloat16), xt) * s[:, None]
        else:
            acc = tl.dot(w.to(tl.bfloat16), xt, acc)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    if not GROUPED:
        s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc * s[:, None]
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * stride_om + offs_n[:, None]
    if ROUND_BF16:  # fp32 output holding the bf16 logits exactly (what stock's fp32 logits buffer copy would hold)
        tl.store(out_ptrs, acc.to(tl.bfloat16).to(tl.float32), mask=mask_n[:, None] & mask_m[None, :])
    else:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_m[None, :])


@triton.jit
def _quant_rows_kernel(w_ptr, q_ptr, s_ptr, stride_wn, stride_qn, NG, GROUP: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    offs = g * GROUP + tl.arange(0, GROUP)
    w = tl.load(w_ptr + row.to(tl.int64) * stride_wn + offs).to(tl.float32)
    amax = tl.max(tl.abs(w), axis=0)
    # div_rn: IEEE round-to-nearest like torch's '/' (Triton's '/' is div.full.f32, up to 2 ulp off)
    scale = tl.div_rn(amax, 127.0)
    safe = tl.where(amax > 0.0, scale, 1.0)
    v = tl.div_rn(w, safe)
    r = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(0.5 - v))  # round half away from zero
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(q_ptr + row.to(tl.int64) * stride_qn + offs, r.to(tl.int8))
    tl.store(s_ptr + row.to(tl.int64) * NG + g, scale)


def quantize_rows(w: torch.Tensor, group: int | None = None, q: torch.Tensor | None = None,
                  scale: torch.Tensor | None = None):
    """int8 q[N, K] and fp32 scale[N] (per row) or scale[N, K // group]; w is a contiguous-rows [N, K] tensor."""
    N, K = w.shape
    group = group or K
    if K % group or group & (group - 1):
        raise ValueError(f"quantize_rows: group {group} must be a power of two dividing K={K}")
    if w.stride(1) != 1:
        raise ValueError("quantize_rows: w must be contiguous along K")
    ng = K // group
    q = torch.empty((N, K), dtype=torch.int8, device=w.device) if q is None else q
    scale = (torch.empty((N,) if ng == 1 else (N, ng), dtype=torch.float32, device=w.device)
             if scale is None else scale)
    _quant_rows_kernel[(N, ng)](w, q, scale, w.stride(0), q.stride(0), ng, GROUP=group,
                                num_warps=4 if group >= 1024 else 1)
    return q, scale


def quantize_rows_torch(w: torch.Tensor, group: int | None = None, chunk: int = 4096):
    """The torch reference of ``quantize_rows`` (round half away from zero, as the kernel does)."""
    N, K = w.shape
    group = group or K
    ng = K // group
    q = torch.empty((N, K), dtype=torch.int8, device=w.device)
    scale = torch.empty((N, ng), dtype=torch.float32, device=w.device)
    for start in range(0, N, chunk):
        wf = w[start:start + chunk].float().view(-1, ng, group)
        s = wf.abs().amax(dim=2) / 127.0
        v = wf / torch.where(s > 0, s, torch.ones_like(s)).unsqueeze(2)
        r = torch.sign(v) * torch.floor(v.abs() + 0.5)
        q[start:start + chunk] = r.clamp_(-127, 127).to(torch.int8).view(-1, K)
        scale[start:start + chunk] = s
    return q, (scale.view(N) if ng == 1 else scale)


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 weights back from (q, scale): the reference the kernel's own error is measured against."""
    N, K = q.shape
    if scale.dim() == 1:
        return q.float() * scale[:, None]
    ng = scale.shape[1]
    return (q.float().view(N, ng, K // ng) * scale[:, :, None]).view(N, K)


# kb8: 1..16 rows run qk_proj_decode.lmh16_w8a16, bit for bit the 16-row Triton config above (the same wgmma m64n16k16
# chain per output in increasing k from zero, the same epilogue) -- 174 us vs 193 at 16 rows (lmh/t_lmh16.py)
CUDA_ROWS = 16


def cuda_w8a16(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None) -> bool:
    M, K = x.shape
    N = q.shape[0]
    return (_qpd is not None and 1 <= M <= CUDA_ROWS and K == 2048 and x.dtype == torch.bfloat16 and x.is_contiguous()
            and q.dtype == torch.int8 and q.is_contiguous() and q.shape[1] == K and N % 64 == 0
            and scale.dim() == 1 and scale.dtype == torch.float32 and scale.is_contiguous() and scale.numel() == N
            and (out is None or (out.dtype in (torch.float32, torch.bfloat16) and out.is_contiguous() and tuple(out.shape) == (M, N)
                                 and out.device == x.device))
            and q.device == x.device and scale.device == x.device)


def w8a16_linear(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None,
                 config: tuple[int, int, int, int, int] | None = None) -> torch.Tensor:
    """x[M, K] bf16/fp16 (rows contiguous) @ dequant(q[N, K], scale)^T -> out[M, N] in x's dtype; a caller-given
    fp32 ``out`` for bf16 x receives the bf16 results widened exactly (stock's fp32 logits buffer contents)."""
    M, K = x.shape
    N = q.shape[0]
    if q.shape[1] != K or x.stride(1) != 1 or q.stride(1) != 1:
        raise ValueError(f"w8a16_linear: x {tuple(x.shape)} / q {tuple(q.shape)} need matching, K-contiguous rows")
    block_n, block_m, block_k, num_warps, num_stages = config or pick_config(M)
    grouped = scale.dim() == 2
    if grouped:
        block_k = K // scale.shape[1]  # one scale per K step
    if K % block_k:
        raise ValueError(f"w8a16_linear: K={K} is not a multiple of BLOCK_K={block_k}")
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if config is None and cuda_w8a16(x, q, scale, out):
        _qpd.lmh16_w8a16(x, q, scale, out)
        return out
    num_m = triton.cdiv(M, block_m)
    grid = (triton.cdiv(N, block_n) * num_m,)
    _w8a16_kernel[grid](x, q, scale, out, M, N, K, x.stride(0), q.stride(0), out.stride(0),
                        scale.stride(0), num_m, BLOCK_N=block_n, BLOCK_M=block_m, BLOCK_K=block_k,
                        GROUPED=grouped, EVEN_N=(N % block_n == 0),
                        ROUND_BF16=(out.dtype == torch.float32 and x.dtype == torch.bfloat16),
                        num_warps=num_warps, num_stages=num_stages)
    return out
