from __future__ import annotations
import torch
import triton
import triton.language as tl
CONFIGS = ((128, 128, 128, 128, 8, 4), (256, 128, 128, 128, 8, 4))
CONFIGS_TWO_TERM = ((64, 64, 64, 128, 4, 3), (96, 128, 64, 128, 4, 3), (128, 128, 128, 128, 8, 4), (256, 256, 64, 128, 8, 3))  # measured; int32 accumulation: same bits at any tile
MAX_ROWS = CONFIGS[-1][0]

def pick_config(rows: int, two_term: bool=False) -> tuple[int, int, int, int, int]:
    for max_rows, *config in CONFIGS_TWO_TERM if two_term else CONFIGS:
        if rows <= max_rows:
            return tuple(config)
    raise ValueError(f'w8a8_linear: {rows} rows is above the {MAX_ROWS}-row table')

@triton.jit
def _round_int8(x):
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.div_rn(amax, 127.0)
    safe = tl.where(amax > 0.0, scale, 1.0)
    v = tl.div_rn(x, safe)
    r = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(0.5 - v))
    return (tl.minimum(tl.maximum(r, -127.0), 127.0), scale)

@triton.jit
def _quant_act_kernel(x_ptr, q_ptr, s_ptr, stride_xm, stride_qm, K, M, BLOCK_K: tl.constexpr, TWO_TERM: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row.to(tl.int64) * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    r, scale = _round_int8(x)
    tl.store(q_ptr + row.to(tl.int64) * stride_qm + offs, r.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)
    if TWO_TERM:
        r2, scale2 = _round_int8(x - r * scale)
        tl.store(q_ptr + (M + row).to(tl.int64) * stride_qm + offs, r2.to(tl.int8), mask=mask)
        tl.store(s_ptr + M + row, scale2)

@triton.jit
def _w8a8_kernel(xq_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om, num_m, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_N: tl.constexpr, TWO_TERM: tl.constexpr, ROUND_BF16: tl.constexpr = False):
    pid = tl.program_id(0)
    pid_n = pid // num_m
    pid_m = pid % num_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_m = offs_m < M
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = xq_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        if EVEN_N:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, mask=mask_n[:, None], other=0)
        xt = tl.load(x_ptrs, mask=mask_m[None, :], other=0)
        acc = tl.dot(w, xt, acc, out_dtype=tl.int32)
        if TWO_TERM:
            xt2 = tl.load(x_ptrs + M * stride_xm, mask=mask_m[None, :], other=0)
            acc2 = tl.dot(w, xt2, acc2, out_dtype=tl.int32)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    ws = tl.load(ws_ptr + offs_n, mask=mask_n, other=0.0)
    xs = tl.load(xs_ptr + offs_m, mask=mask_m, other=0.0)
    res = acc.to(tl.float32) * xs[None, :]
    if TWO_TERM:
        xs2 = tl.load(xs_ptr + M + offs_m, mask=mask_m, other=0.0)
        res += acc2.to(tl.float32) * xs2[None, :]
    res = res * ws[:, None]
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * stride_om + offs_n[:, None]
    if ROUND_BF16:  # fp32 output holding the bf16 logits exactly (what stock's fp32 logits buffer copy would hold)
        tl.store(out_ptrs, res.to(tl.bfloat16).to(tl.float32), mask=mask_n[:, None] & mask_m[None, :])
    else:
        tl.store(out_ptrs, res.to(out_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_m[None, :])

def quantize_act(x: torch.Tensor, xq: torch.Tensor | None=None, xs: torch.Tensor | None=None, two_term: bool=False):
    M, K = x.shape
    if x.stride(1) != 1:
        raise ValueError('quantize_act: x must be contiguous along K')
    terms = 2 if two_term else 1
    xq = torch.empty((terms * M, K), dtype=torch.int8, device=x.device) if xq is None else xq
    xs = torch.empty((terms * M,), dtype=torch.float32, device=x.device) if xs is None else xs
    _quant_act_kernel[M,](x, xq, xs, x.stride(0), xq.stride(0), K, M, BLOCK_K=triton.next_power_of_2(K), TWO_TERM=two_term, num_warps=4)
    return (xq, xs)

def w8a8_gemm(xq: torch.Tensor, xs: torch.Tensor, q: torch.Tensor, w_scale: torch.Tensor, out: torch.Tensor, config: tuple[int, int, int, int, int] | None=None, two_term: bool=False) -> torch.Tensor:
    M, N = out.shape
    K = xq.shape[1]
    if q.shape[1] != K or q.shape[0] != N or xq.shape[0] != (2 if two_term else 1) * M or (xq.stride(1) != 1) or (q.stride(1) != 1) or (w_scale.dim() != 1):
        raise ValueError('w8a8_gemm: K-contiguous int8 operands, M (or 2M) activation rows and a per-row weight scale are required')
    block_n, block_m, block_k, num_warps, num_stages = config or pick_config(M, two_term)
    if K % block_k:
        raise ValueError(f'w8a8_gemm: K={K} is not a multiple of BLOCK_K={block_k}')
    num_m = triton.cdiv(M, block_m)
    grid = (triton.cdiv(N, block_n) * num_m,)
    _w8a8_kernel[grid](xq, xs, q, w_scale, out, M, N, K, xq.stride(0), q.stride(0), out.stride(0), num_m, BLOCK_N=block_n, BLOCK_M=block_m, BLOCK_K=block_k, EVEN_N=N % block_n == 0, TWO_TERM=two_term, ROUND_BF16=out.dtype == torch.float32, num_warps=num_warps, num_stages=num_stages)
    return out

def w8a8_linear(x: torch.Tensor, q: torch.Tensor, w_scale: torch.Tensor, out: torch.Tensor | None=None, config: tuple[int, int, int, int, int] | None=None, two_term: bool=False) -> torch.Tensor:
    xq, xs = quantize_act(x, two_term=two_term)
    if out is None:
        out = torch.empty((x.shape[0], q.shape[0]), dtype=x.dtype, device=x.device)
    return w8a8_gemm(xq, xs, q, w_scale, out, config, two_term)
