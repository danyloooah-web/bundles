"""Decode-width output projection fused with the residual add and Gemma RMSNorm.

Stock runs the attention/recurrent output projection as a cuBLAS split-K GEMM
plus its reduction, then the fused add-and-norm kernel. For a few decode rows
the 16.8 MB weight read is the whole cost, so this module streams it once with
a split-K Triton GEMM that writes fp32 partials, and one program per row sums
the partials, rounds the projection to bf16 as cuBLAS does, adds the residual
in fp32, stores the new residual and normalizes it with ``1 + weight`` exactly
as the stock Gemma norm does. Both launches use programmatic dependent launch
so each one's prologue overlaps its predecessor's tail.
"""

import torch
import triton
import triton.language as tl

BLOCK_T = 16
BLOCK_N = 64
BLOCK_K = 64
SPLIT_K = 8
# (max_rows, (split_k, BLOCK_K, BLOCK_T, BLOCK_N, num_warps, num_stages))
_TILES = ((16, (4, 64, 16, 32, 2, 6)), (32, (4, 64, 32, 64, 4, 6)), (48, (4, 128, 64, 64, 4, 4)),
          (96, (4, 64, 128, 64, 8, 6)), (1 << 30, (4, 64, 128, 128, 8, 4)))


@triton.jit
def _proj_kernel(
    a_ptr, w_ptr, part_ptr, M,
    K: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    K_PART: tl.constexpr = K // SPLIT_K
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_s * K_PART + tl.arange(0, BLOCK_K)
    tmask = offs_t < M
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    a_ptrs = a_ptr + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_PART, BLOCK_K):
        a = tl.load(a_ptrs, mask=tmask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    out_ptrs = part_ptr + (pid_s * M + offs_t[:, None]).to(tl.int64) * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=tmask[:, None])


@triton.jit
def _add_norm_kernel(
    part_ptr, res_ptr, w_ptr, out_ptr, M, eps,
    N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    # the residual (the previous layer's output) and the norm weight are final before the projection launched this
    # kernel: they load before the wait, the projection's partials after it
    r = tl.load(res_ptr + t * N + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    h = tl.zeros((N,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        h += tl.load(part_ptr + (s * M + t) * N + offs)
    # Stock rounds the projection to bf16 before its fp32 residual add; keep
    # that boundary so the only difference is the GEMM's accumulation order.
    z = h.to(tl.bfloat16).to(tl.float32) + r
    tl.store(res_ptr + t * N + offs, z.to(tl.bfloat16))
    var = tl.sum(z * z, axis=0) / N
    y = z * tl.rsqrt(var + eps) * (1.0 + w)
    tl.store(out_ptr + t * N + offs, y.to(tl.bfloat16))


def project_add_norm(a, weight, residual, norm_weight, eps, partial):
    """``residual += a @ weight.T`` in place; return the Gemma-normalized rows.

    ``partial`` is a reusable ``[SPLIT_K, max_rows, N]`` fp32 workspace.
    """
    M, K = a.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    # Non-bitwise launch retune (lane B of candidates/qwen36_king_nbretune_20260925, H100 2026-09-25,
    # summary.B_common): SPLIT_K 4 at every width changes the fp32 reduction order (max row error
    # vs the king 1.9e-4 on the add-norm output at 8/16 rows), saving 0.85 us per layer at 8/16 rows
    # and 2.66 us at 96/128. The partial workspace keeps the module's SPLIT_K (8) rows of capacity.
    # Tiles measured per row bucket (graphs, 40 copies); every entry is bit for bit the shipped SPLIT_K-4 launch.
    split_k, bk, bt, bn, warps, stages = next(cfg for rows, cfg in _TILES if M <= rows)
    grid = (triton.cdiv(M, bt), N // bn, split_k)
    _proj_kernel[grid](
        a, weight, partial, M, K=K, N=N, SPLIT_K=split_k,
        BLOCK_T=bt, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages, launch_pdl=True,
    )
    _add_norm_kernel[(M,)](partial, residual, norm_weight, out, M, eps, N=N, SPLIT_K=split_k, num_warps=16,
                           launch_pdl=True)
    return out, residual
