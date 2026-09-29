"""Lane DENSE8: W8A16 kernels for the dense verify projections of Qwen3.6-35B-A3B (delivered untested --
no GPU touched writing this).

A candidate-owned INT8 copy of a BF16 projection weight ``w [N, K]``: ``q [N, K]`` int8 (symmetric,
round half away from zero) and one fp16 scale per 128 consecutive K elements of a row, stored
group-major ``s [K // 128, N]`` so a tile's scales for one K group are one contiguous load. Stock's
BF16 weights are only read (to build the copy); nothing here writes them.

Three consumers, all at the decode/verify row counts (1..256 token rows):

- ``gdn_in_proj8(x, q, s)``: ``x [M, 2048] @ [w_qkvz; w_ba].T`` split into the king's four outputs
  ``mixed_qkv [M, 8192]``, ``z [M, 32, 128]``, ``b [M, 32]``, ``a [M, 32]`` (bf16, contiguous), in one
  launch. ``q`` is the INT8 copy of the concatenated ``[12288 + 64, 2048]`` weight.
- ``qkv_proj8(x, q, s)``: ``x [M, 2048] @ w [9216, 2048].T`` (the attention block's fused q/k/v).
- ``proj8_partials(a, q, s, partial)``: the o_proj/out_proj GEMM ``a [M, 4096] @ w [2048, 4096].T`` as
  split-K fp32 partials ``[SPLIT_K, M, 2048]``, the operand of the king's unchanged ``_add_norm_kernel``.

The GEMMs load int8 weight tiles contiguously along K, convert them to bf16 in registers (int8 values
are exact in bf16), run a bf16 MMA with fp32 accumulation per K block and multiply the block's partial
product by its fp16 group scale before accumulating: the only roundings beyond the quantization are
fp32 accumulation and the bf16 store (in_proj/qkv) or the king's own add-norm (o_proj). The in_proj/qkv
kernel computes the transposed tile ``out^T = q_tile @ x^T`` ("swap AB", as ``lmhead8.py``): weight rows
are the MMA's M side, token rows its N side, so 8/16-row batches waste no weight-side width.

Every launch uses programmatic dependent launch (``launch_pdl=True`` + ``gdc_wait`` before the first
read of the activation). A config is a pure function of the row count (no autotune: its benchmarking
synchronizes, which is illegal under CUDA-graph capture). ``bench_dense8.py`` sweeps configs on the H100;
the tables here are first guesses until it runs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import qk_route

# v1 switch (lane dense): True = the attention qkv_proj and every decode o_proj / out_proj join DENSE8's INT8 copies and
# every DENSE8 shape at <= 32 verify rows runs qk_dense8.cu; False = 2a5abaae's DENSE8 exactly (GDN in_proj, Triton).
LANE_DENSE = True

try:
    # built from qk_proj_decode.cu (lane dense's kernels, namespace qd8) by the validator's CUDA build step
    import qk_proj_decode as _qpd

    class qk_dense8:  # noqa: N801 - the call sites below keep the lane's original module name
        seg = staticmethod(_qpd.dense8_seg)
        part = staticmethod(_qpd.dense8_part)
except ImportError:  # Triton's CPU interpreter checks (check_dense8.py) run without the CUDA units
    qk_dense8 = None
if not LANE_DENSE:
    qk_dense8 = None

GROUP = 128
HIDDEN = 2048
# GDN in_proj row batches in [NATIVE_INPROJ_MIN, NATIVE_INPROJ_MAX] take qk_route.inproj16 (bit for bit _w8_seg_kernel):
# one CTA per SM with its whole weight share streamed into shared memory before the dependency wait. It pays only
# with the previous layer's combine-norm releasing it early (i8x_decode kcombnorm `early`, set by layer.py through
# native_inproj()); the Triton kernel launched that early is ~8 us slower per layer (measured, d8p_test.py).
NATIVE_INPROJ_MIN = 1
NATIVE_INPROJ_MAX = 16
_sms = {}
PROJ_IN = 4096
QKV_OUT = 8192          # GDN [q | k | v]
Z_OUT = 4096            # GDN z = 32 heads x 128
V_HEADS = 32
V_HEAD = 128
INPROJ_N = QKV_OUT + Z_OUT + 2 * V_HEADS  # 12352 = [w_qkvz; w_ba] rows
ATTN_QKV_N = 9216
MAX_ROWS = 256
# the o_proj partial workspace the king allocates is [8, 256, 2048] fp32; a split above 8 does not fit it
MAX_SPLIT_K = 8
# True only under Triton's CPU interpreter (check_dense8.py): no programmatic dependent launch (the
# interpreter has no griddepcontrol) and an fp32 MMA (its bf16 dot returns garbage, Triton 3.7.0). On the
# GPU every GEMM launches with PDL and a bf16 MMA.
INTERPRETER = False

# swap-AB kernel (in_proj, qkv): (max_rows, BLOCK_N, BLOCK_M, BLOCK_K, num_warps, num_stages).
# First guesses (not measured): 16-row token tiles below 32 rows, one 128-row tile above; BLOCK_K is
# one scale group.
INPROJ_CONFIGS = (
    (16, 16, 16, 128, 2, 3),    # measured (graphs, 30 copies): 13.5 vs 14.4 us at 16 rows, bit for bit the first guess
    (32, 32, 32, 128, 4, 3),
    (128, 64, 128, 128, 4, 3),
    (256, 64, 128, 64, 4, 3),
)
QKV_CONFIGS = (
    (16, 32, 16, 128, 4, 5),
    (32, 32, 32, 128, 4, 4),
    (128, 64, 128, 128, 4, 3),
    (256, 64, 128, 64, 4, 3),
)
# o_proj split-K kernel: (max_rows, SWAP, SPLIT_K, BLOCK_T, BLOCK_N, BLOCK_K, num_warps, num_stages).
# SWAP 0 is the king's ``_proj_kernel`` orientation (token rows as the MMA M side), SWAP 1 the
# transposed tile. First guesses: the king's own tile shapes with BLOCK_K = one scale group.
OPROJ_CONFIGS = (
    (32, 0, 8, 16, 64, 128, 4, 4),
    (96, 0, 8, 128, 128, 128, 8, 3),
    (256, 0, 8, 64, 128, 128, 8, 3),
)


# Lane dense: the CUDA W8A16 kernel (qk_dense8.cu) serves 1..CUDA_MAX_ROWS token rows of every DENSE8 shape. It
# streams the INT8 rows with 2D TMA through one ring per CTA (weight stages requested before the programmatic-dependent
# wait) and computes the same k-slot products and fma chain as the Triton kernels below. Per shape:
# (max_rows, math warps (16 weight rows each), stages requested before the dependency wait[, K splits]).
# Measured (CUDA graphs, cold weights, H100 rig 2026-09-28): in_proj 12.2 vs Triton 12.9-13.0 us at 1-8 rows, even at 16,
# Triton faster at 24-32 (14.0 vs 15.1); qkv 10.3-11.0 vs the BF16 kernel's 14.3-14.4 us at 1-16 rows, 13.8 vs 14.4 at
# 24-32; o_proj partials + add-norm 6.2-6.8 vs BF16 8.2-8.4 us at 1-16 rows, 7.7-7.8 vs 8.8-8.9 at 24-32.
CUDA_MAX_ROWS = 32
INPROJ_CUDA = ((16, 6, 1),)
QKV_CUDA = ((32, 5, 1),)
OPROJ_CUDA = ((32, 4, 4, 1),)
OPROJ_MAX_SPLIT_K = 8


def _pick(table, rows: int):
    for max_rows, *config in table:
        if rows <= max_rows:
            return tuple(config)
    raise ValueError(f"dense8: {rows} rows is above the {table[-1][0]}-row table")


def pick_inproj(rows: int):
    return _pick(INPROJ_CONFIGS, rows)


def pick_qkv(rows: int):
    return _pick(QKV_CONFIGS, rows)


def pick_oproj(rows: int):
    return _pick(OPROJ_CONFIGS, rows)


# ------------------------------------------------------------------------------------------ quantizer

@triton.jit
def _quant_kernel(w_ptr, q_ptr, s_ptr, S_STRIDE, K: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    offs = g * GROUP + tl.arange(0, GROUP)
    w = tl.load(w_ptr + row.to(tl.int64) * K + offs).to(tl.float32)
    amax = tl.max(tl.abs(w), axis=0)
    # one fp16 ulp (2^-10) of headroom so the rounded fp16 scale is never below amax / 127 in the
    # fp16 normal range: the largest element then maps inside [-127, 127] without clamping
    s16 = (tl.div_rn(amax, 127.0) * 1.0009765625).to(tl.float16)
    s = s16.to(tl.float32)
    safe = tl.where(s > 0.0, s, 1.0)
    v = tl.div_rn(w, safe)
    r = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(0.5 - v))  # round half away from zero
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(q_ptr + row.to(tl.int64) * K + offs, r.to(tl.int8))
    tl.store(s_ptr + g.to(tl.int64) * S_STRIDE + row, s16)


def quantize(w: torch.Tensor, q: torch.Tensor | None = None, s: torch.Tensor | None = None):
    """int8 ``q [N, K]`` and fp16 ``s [K // GROUP, N]`` of a contiguous bf16 ``w [N, K]``.

    Writes into ``q``/``s`` when given (the lazily allocated resident copy): no other temporary. ``q``
    must be contiguous; ``s`` only along N, so it may be a column block of a wider scale buffer (the
    in_proj copy stacks ``w_qkvz`` and ``w_ba`` into one ``[12352, 2048]`` copy).
    """
    N, K = w.shape
    if K % GROUP or not w.is_contiguous():
        raise ValueError(f"dense8.quantize: w {tuple(w.shape)} must be contiguous with K % {GROUP} == 0")
    q = torch.empty((N, K), dtype=torch.int8, device=w.device) if q is None else q
    s = torch.empty((K // GROUP, N), dtype=torch.float16, device=w.device) if s is None else s
    if tuple(q.shape) != (N, K) or q.dtype != torch.int8 or tuple(s.shape) != (K // GROUP, N) \
            or s.dtype != torch.float16 or not q.is_contiguous() or s.stride(1) != 1:
        raise ValueError("dense8.quantize: q/s buffers have the wrong shape, dtype or layout")
    _quant_kernel[(N, K // GROUP)](w, q, s, s.stride(0), K=K, GROUP=GROUP, num_warps=1)
    return q, s


def quantize_torch(w: torch.Tensor, chunk: int = 2048):
    """The torch reference of ``quantize`` (same fp32 operations, same rounding)."""
    N, K = w.shape
    ng = K // GROUP
    q = torch.empty((N, K), dtype=torch.int8, device=w.device)
    s = torch.empty((ng, N), dtype=torch.float16, device=w.device)
    for start in range(0, N, chunk):
        wf = w[start:start + chunk].float().view(-1, ng, GROUP)
        amax = wf.abs().amax(dim=2)
        s16 = ((amax / 127.0) * 1.0009765625).to(torch.float16)
        sf = s16.float()
        v = wf / torch.where(sf > 0, sf, torch.ones_like(sf)).unsqueeze(2)
        r = torch.sign(v) * torch.floor(v.abs() + 0.5)
        q[start:start + chunk] = r.clamp_(-127, 127).to(torch.int8).view(-1, K)
        s[:, start:start + chunk] = s16.t()
    return q, s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp32 ``[N, K]`` weights back from ``(q, s)``."""
    N, K = q.shape
    return (q.float().view(N, K // GROUP, GROUP) * s.float().t().unsqueeze(2)).view(N, K)


def copy_bytes(n: int, k: int) -> int:
    """Resident bytes of one INT8 copy (int8 values + fp16 group scales)."""
    return n * k + (k // GROUP) * n * 2


# ------------------------------------------------------------------ swap-AB GEMM (in_proj, qkv_proj)

@triton.jit
def _mma(a, b, F32: tl.constexpr):
    """bf16 MMA with fp32 accumulation; ``F32`` (Triton's CPU interpreter only, whose bf16 dot is wrong)
    runs the same product in fp32 on the exact bf16 values."""
    if F32:
        d = tl.dot(a.to(tl.float32), b.to(tl.float32))
    else:
        d = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    return d


@triton.jit
def _w8_seg_kernel(x_ptr, q_ptr, s_ptr, o0, o1, o2, o3, M, num_m,
                   N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, NSEG: tl.constexpr,
                   LO1: tl.constexpr, LO2: tl.constexpr, LO3: tl.constexpr,
                   LD0: tl.constexpr, LD1: tl.constexpr, LD2: tl.constexpr, LD3: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                   EVEN_N: tl.constexpr, PDL: tl.constexpr, F32: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = pid // num_m
    pid_m = pid % num_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_m = offs_m < M
    w_ptrs = q_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :].to(tl.int64) * K + offs_k[:, None]  # x^T tile [BLOCK_K, BLOCK_M]
    s_ptrs = s_ptr + offs_n
    if PDL:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):  # the wrapper asserts GROUP % BLOCK_K == 0 and K % BLOCK_K == 0
        if EVEN_N:
            w = tl.load(w_ptrs)
            sc = tl.load(s_ptrs + (k // GROUP) * N)
        else:
            w = tl.load(w_ptrs, mask=mask_n[:, None], other=0)
            sc = tl.load(s_ptrs + (k // GROUP) * N, mask=mask_n, other=0.0)
        xt = tl.load(x_ptrs, mask=mask_m[None, :], other=0.0)
        acc += _mma(w, xt, F32) * sc.to(tl.float32)[:, None]
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    out = acc.to(tl.bfloat16)
    n2 = offs_n[:, None]
    m2 = offs_m[None, :].to(tl.int64)
    mm = mask_m[None, :]
    if NSEG == 1:
        tl.store(o0 + m2 * LD0 + n2, out, mask=mm & mask_n[:, None])
    else:
        # output column c of segment s = [LO_s, LO_{s+1}) goes to o_s[row, c - LO_s] (row stride LD_s);
        # masked-off lanes compute out-of-range addresses and never store
        tl.store(o0 + m2 * LD0 + n2, out, mask=mm & (n2 < LO1))
        tl.store(o1 + m2 * LD1 + (n2 - LO1), out, mask=mm & (n2 >= LO1) & (n2 < LO2))
        tl.store(o2 + m2 * LD2 + (n2 - LO2), out, mask=mm & (n2 >= LO2) & (n2 < LO3))
        tl.store(o3 + m2 * LD3 + (n2 - LO3), out, mask=mm & (n2 >= LO3) & (n2 < N))


def _check_x(x: torch.Tensor, k: int):
    if (x.dim() != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or not 1 <= x.shape[0] <= MAX_ROWS):
        raise ValueError(f"dense8: x must be contiguous bf16 [1..{MAX_ROWS}, {k}], got {tuple(x.shape)} {x.dtype}")


def _check_copy(q: torch.Tensor, s: torch.Tensor, n: int, k: int):
    if (tuple(q.shape) != (n, k) or q.dtype != torch.int8 or tuple(s.shape) != (k // GROUP, n)
            or s.dtype != torch.float16 or not q.is_contiguous() or not s.is_contiguous()):
        raise ValueError(f"dense8: the INT8 copy must be q [{n}, {k}] int8 and s [{k // GROUP}, {n}] fp16")


def _launch_seg(x, q, s, outs, n, lo, ld, nseg, config):
    M, K = x.shape
    block_n, block_m, block_k, warps, stages = config
    if GROUP % block_k or K % block_k:
        raise ValueError(f"dense8: BLOCK_K {block_k} must divide the scale group {GROUP} and K {K}")
    num_m = triton.cdiv(M, block_m)
    grid = (triton.cdiv(n, block_n) * num_m,)
    o = list(outs) + [outs[0]] * (4 - len(outs))
    _w8_seg_kernel[grid](
        x, q, s, o[0], o[1], o[2], o[3], M, num_m,
        N=n, K=K, GROUP=GROUP, NSEG=nseg, LO1=lo[0], LO2=lo[1], LO3=lo[2],
        LD0=ld[0], LD1=ld[1], LD2=ld[2], LD3=ld[3],
        BLOCK_N=block_n, BLOCK_M=block_m, BLOCK_K=block_k, EVEN_N=(n % block_n == 0), PDL=not INTERPRETER,
        F32=INTERPRETER, num_warps=warps, num_stages=stages, launch_pdl=not INTERPRETER)


def native_inproj(rows: int, device) -> bool:
    """True when a GDN in_proj of ``rows`` token rows runs qk_route.inproj16 (one CTA per SM, <= MAX_TILES tiles)."""
    if INTERPRETER or not NATIVE_INPROJ_MIN <= rows <= min(NATIVE_INPROJ_MAX, qk_route.INPROJ16_MAX_ROWS):
        return False
    dev = torch.device(device)
    sms = _sms.get(dev)
    if sms is None:
        sms = _sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return -(-(INPROJ_N // 16) // sms) <= qk_route.INPROJ16_MAX_TILES


def gdn_in_proj8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, config=None):
    """``(mixed_qkv [M, 8192], z [M, 32, 128], b [M, 32], a [M, 32])`` of ``x @ dequant(q, s).T``."""
    _check_x(x, HIDDEN)
    _check_copy(q, s, INPROJ_N, HIDDEN)
    M = x.shape[0]
    mixed = torch.empty((M, QKV_OUT), dtype=torch.bfloat16, device=x.device)
    z = torch.empty((M, V_HEADS, V_HEAD), dtype=torch.bfloat16, device=x.device)
    b = torch.empty((M, V_HEADS), dtype=torch.bfloat16, device=x.device)
    a = torch.empty((M, V_HEADS), dtype=torch.bfloat16, device=x.device)
    if config is None and native_inproj(M, x.device):
        lo = [0, QKV_OUT, QKV_OUT + Z_OUT, QKV_OUT + Z_OUT + V_HEADS]
        qk_route.inproj16(x, q, s, [mixed, z, b, a], lo, [QKV_OUT, Z_OUT, V_HEADS, V_HEADS])
        return mixed, z, b, a
    _launch_seg(x, q, s, (mixed, z, b, a), INPROJ_N,
                (QKV_OUT, QKV_OUT + Z_OUT, QKV_OUT + Z_OUT + V_HEADS), (QKV_OUT, Z_OUT, V_HEADS, V_HEADS), 4,
                config or pick_inproj(M))
    return mixed, z, b, a


def qkv_proj8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, config=None):
    """``x [M, 2048] @ dequant(q, s).T`` -> ``[M, 9216]`` bf16."""
    _check_x(x, HIDDEN)
    _check_copy(q, s, ATTN_QKV_N, HIDDEN)
    M = x.shape[0]
    out = torch.empty((M, ATTN_QKV_N), dtype=torch.bfloat16, device=x.device)
    if config is None and qk_dense8 is not None and M <= QKV_CUDA[-1][0]:
        warps, pre = _pick(QKV_CUDA, M)
        qk_dense8.seg(x, q, s, [out], [0], [ATTN_QKV_N], warps, pre)
        return out
    _launch_seg(x, q, s, (out,), ATTN_QKV_N, (ATTN_QKV_N, ATTN_QKV_N, ATTN_QKV_N),
                (ATTN_QKV_N, ATTN_QKV_N, ATTN_QKV_N, ATTN_QKV_N), 1, config or pick_qkv(M))
    return out


# --------------------------------------------------------------- split-K o_proj/out_proj partials

@triton.jit
def _proj8_kernel(a_ptr, q_ptr, s_ptr, part_ptr, M,
                  K: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr, GROUP: tl.constexpr,
                  SWAP: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  PDL: tl.constexpr, F32: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    K_PART: tl.constexpr = K // SPLIT_K
    k0 = pid_s * K_PART
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k0 + tl.arange(0, BLOCK_K)
    tmask = offs_t < M
    s_ptrs = s_ptr + offs_n
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if SWAP:
        w_ptrs = q_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :]     # [BLOCK_N, BLOCK_K]
        a_ptrs = a_ptr + offs_t[None, :].to(tl.int64) * K + offs_k[:, None]     # a^T [BLOCK_K, BLOCK_T]
        acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
        for kk in range(0, K_PART, BLOCK_K):
            w = tl.load(w_ptrs)
            at = tl.load(a_ptrs, mask=tmask[None, :], other=0.0)
            sc = tl.load(s_ptrs + ((k0 + kk) // GROUP) * N).to(tl.float32)
            acc += _mma(w, at, F32) * sc[:, None]
            w_ptrs += BLOCK_K
            a_ptrs += BLOCK_K
        out_ptrs = part_ptr + (pid_s * M + offs_t[None, :]).to(tl.int64) * N + offs_n[:, None]
        tl.store(out_ptrs, acc, mask=tmask[None, :])
    else:
        a_ptrs = a_ptr + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]     # [BLOCK_T, BLOCK_K]
        w_ptrs = q_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]     # w^T [BLOCK_K, BLOCK_N]
        acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
        for kk in range(0, K_PART, BLOCK_K):
            a = tl.load(a_ptrs, mask=tmask[:, None], other=0.0)
            w = tl.load(w_ptrs)
            sc = tl.load(s_ptrs + ((k0 + kk) // GROUP) * N).to(tl.float32)
            acc += _mma(a, w, F32) * sc[None, :]
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        out_ptrs = part_ptr + (pid_s * M + offs_t[:, None]).to(tl.int64) * N + offs_n[None, :]
        tl.store(out_ptrs, acc, mask=tmask[:, None])


def proj8_partials(a: torch.Tensor, q: torch.Tensor, s: torch.Tensor, partial: torch.Tensor, config=None) -> int:
    """Write ``a [M, 4096] @ dequant(q, s).T`` as fp32 split-K partials into ``partial[:SPLIT_K]``
    (viewed ``[SPLIT_K, M, N]``); return SPLIT_K for the add-norm launch that sums them."""
    _check_x(a, PROJ_IN)
    _check_copy(q, s, HIDDEN, PROJ_IN)
    M, K = a.shape
    N = q.shape[0]
    swap, split_k, bt, bn, bk, warps, stages = config or pick_oproj(M)
    k_part = K // split_k
    if (split_k > MAX_SPLIT_K or K % split_k or k_part % bk or GROUP % bk or N % bn
            or partial.dtype != torch.float32 or partial.numel() < split_k * M * N or not partial.is_contiguous()):
        raise ValueError(f"dense8: o_proj config {config} does not fit K={K}, N={N} or the partial workspace")
    grid = (triton.cdiv(M, bt), N // bn, split_k)
    _proj8_kernel[grid](a, q, s, partial, M, K=K, N=N, SPLIT_K=split_k, GROUP=GROUP, SWAP=swap,
                        BLOCK_T=bt, BLOCK_N=bn, BLOCK_K=bk, PDL=not INTERPRETER, F32=INTERPRETER,
                        num_warps=warps, num_stages=stages, launch_pdl=not INTERPRETER)
    return split_k


def oproj8_partials(a: torch.Tensor, q: torch.Tensor, s: torch.Tensor, partial: torch.Tensor) -> int:
    """Lane dense: ``a [M, 4096] @ dequant(q, s).T`` as fp32 split-K partials ``[SPLIT_K, M, 2048]`` in ``partial``
    (the o_proj / out_proj operand of proj.py's add-norm) through qk_dense8.cu; returns SPLIT_K."""
    M = a.shape[0]
    if (qk_dense8 is None or a.dim() != 2 or a.shape[1] != PROJ_IN or a.dtype != torch.bfloat16
            or not a.is_contiguous() or not 1 <= M <= OPROJ_CUDA[-1][0]):
        raise ValueError(f"dense8: o_proj rows must be contiguous bf16 [1..{CUDA_MAX_ROWS}, {PROJ_IN}]")
    _check_copy(q, s, HIDDEN, PROJ_IN)
    warps, split_k, pre = _pick(OPROJ_CUDA, M)
    if (split_k > OPROJ_MAX_SPLIT_K or partial.dtype != torch.float32 or not partial.is_contiguous()
            or partial.numel() < split_k * M * HIDDEN):
        raise ValueError("dense8: the o_proj partial workspace does not hold the split")
    qk_dense8.part(a, q, s, partial, split_k, warps, pre)
    return split_k
