"""51f3e0f9 (5DM6k4b3): the prefill MoE up on INT8 tensor cores over the per-channel INT8 copy (PRECISION CHANGE)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

H, I, E, G = 2048, 512, 256, 128
BM = 128
WARPS = 16
STAGES = 3

@triton.jit
def _rnd(v):
    return tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(0.5 - v))

@triton.jit
def _quant_rows(x_ptr, q_ptr, s_ptr, K: tl.constexpr, G: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    x = tl.reshape(tl.load(x_ptr + r * K + tl.arange(0, K)).to(tl.float32), (K // G, G))
    s = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30) / 127.0
    tl.store(q_ptr + r * K + tl.arange(0, K), tl.reshape(_rnd(x / s[:, None]), (K,)).to(tl.int8))
    tl.store(s_ptr + r * (K // G) + tl.arange(0, K // G), s)

@triton.jit
def _up(x_ptr, xq_ptr, xs_ptr, q_ptr, s_ptr, s13_ptr, h_ptr, mt_ptr, tok_ptr, H: tl.constexpr, I: tl.constexpr,
        EB: tl.constexpr, C13: tl.constexpr, E: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr,
        TWO_LOOP: tl.constexpr):
    NB: tl.constexpr = I // G
    SPLIT: tl.constexpr = 128 // BLOCK_M
    pid = tl.program_id(0)
    j = pid // (NB * SPLIT)
    part = pid % (NB * SPLIT) // NB
    pid_n = pid % NB
    e = tl.load(mt_ptr + 4 * j)
    row0 = tl.load(mt_ptr + 4 * j + 1)
    nrows = tl.load(mt_ptr + 4 * j + 2)
    if nrows <= part * BLOCK_M:
        return
    offs_m = part * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = offs_m < nrows
    r = (row0 + offs_m).to(tl.int64)
    tok = tl.load(tok_ptr + r, mask=valid, other=0).to(tl.int64)
    offs_n = pid_n * G + tl.arange(0, G)
    offs_k = tl.arange(0, G)
    if e == E:
        offs_k2 = tl.arange(0, G // 2)
        xa_ptrs = x_ptr + tok[:, None] * H + offs_k2[None, :]
        sg_ptrs = s13_ptr + offs_n[None, :].to(tl.int64) * H + offs_k2[:, None]
        su_ptrs = sg_ptrs + I * H
        acc_g = tl.zeros((BLOCK_M, G), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, G), dtype=tl.float32)
        for kb in range(0, 2 * H // G):
            xa = tl.load(xa_ptrs, mask=valid[:, None], other=0.0)
            acc_g = tl.dot(xa, tl.load(sg_ptrs), acc_g)
            acc_u = tl.dot(xa, tl.load(su_ptrs), acc_u)
            xa_ptrs += G // 2
            sg_ptrs += G // 2
            su_ptrs += G // 2
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
    else:
        wq = q_ptr + e.to(tl.int64) * EB
        ws = s_ptr + e.to(tl.int64) * (EB // 2) + C13 // 2
        a_ptrs = xq_ptr + tok[:, None] * H + offs_k[None, :]
        as_ptrs = xs_ptr + tok * (H // G)
        g_ptrs = wq + offs_n[None, :].to(tl.int64) * H + offs_k[:, None]
        u_ptrs = g_ptrs + I * H
        acc_g = tl.zeros((BLOCK_M, G), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, G), dtype=tl.float32)
        if TWO_LOOP:
            for kb in range(0, H // G):
                a = tl.load(a_ptrs + kb * G, mask=valid[:, None], other=0)
                sa = tl.load(as_ptrs + kb, mask=valid, other=0.0)
                acc_g += tl.dot(a, tl.load(g_ptrs + kb * G), out_dtype=tl.int32).to(tl.float32) * sa[:, None]
            for kb in range(0, H // G):
                a = tl.load(a_ptrs + kb * G, mask=valid[:, None], other=0)
                sa = tl.load(as_ptrs + kb, mask=valid, other=0.0)
                acc_u += tl.dot(a, tl.load(u_ptrs + kb * G), out_dtype=tl.int32).to(tl.float32) * sa[:, None]
        else:
            for kb in range(0, H // G):
                a = tl.load(a_ptrs, mask=valid[:, None], other=0)
                sa = tl.load(as_ptrs + kb, mask=valid, other=0.0)
                dg = tl.dot(a, tl.load(g_ptrs), out_dtype=tl.int32)
                du = tl.dot(a, tl.load(u_ptrs), out_dtype=tl.int32)
                acc_g += dg.to(tl.float32) * sa[:, None]
                acc_u += du.to(tl.float32) * sa[:, None]
                a_ptrs += G
                g_ptrs += G
                u_ptrs += G
        sgs = tl.load(ws + offs_n.to(tl.int64) * (H // G)).to(tl.float32)
        sus = tl.load(ws + (I + offs_n).to(tl.int64) * (H // G)).to(tl.float32)
        g = (acc_g * sgs[None, :]).to(tl.bfloat16).to(tl.float32)
        u = (acc_u * sus[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(h_ptr + r[:, None] * I + offs_n[None, :], (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16),
             mask=valid[:, None])

def quant_g128(x: torch.Tensor):
    """``x`` rows as int8 with one scale per row and 128-column group (the up's own activation quantizer)."""
    T = x.shape[0]
    xq = torch.empty((T, H), dtype=torch.int8, device=x.device)
    xs = torch.empty((T, H // G), dtype=torch.float32, device=x.device)
    _quant_rows[(T,)](x, xq, xs, K=H, G=G, num_warps=4)
    return xq, xs

def up(x: torch.Tensor, blocks: torch.Tensor, s13: torch.Tensor, offsets, mt: torch.Tensor,
       sorted_tok: torch.Tensor, cfg=None) -> torch.Tensor:
    q2, c13, c2, eb = offsets
    bm, warps, stages, two = cfg or (BM, WARPS, STAGES, False)
    T = x.shape[0]
    xq = torch.empty((T, H), dtype=torch.int8, device=x.device)
    xs = torch.empty((T, H // G), dtype=torch.float32, device=x.device)
    _quant_rows[(T,)](x, xq, xs, K=H, G=G, num_warps=4)
    h = torch.empty((sorted_tok.shape[0], I), dtype=torch.bfloat16, device=x.device)
    grid = (mt.shape[0] * (I // G) * (128 // bm),)
    _up[grid](x, xq, xs, blocks.view(torch.int8), blocks.view(torch.float16), s13, h, mt, sorted_tok, H=H, I=I,
              EB=eb, C13=c13, E=E, G=G, BLOCK_M=bm, TWO_LOOP=two, num_warps=warps, num_stages=stages)
    return h
