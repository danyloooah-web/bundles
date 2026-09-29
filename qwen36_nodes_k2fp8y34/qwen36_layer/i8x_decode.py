from __future__ import annotations
import torch
import triton
import triton.language as tl
import qk_route
import qk_moe_dn16
import qk_moe_up16
TOPK = 8
MAX_T = 256
BLOCK_T = 16
GATE_SPLIT_K = 4
GATE_BLOCK_E = 16
GATE_BLOCK_K = 256
SHARED_UP_N = 16
GROUP = 128
INTERPRETER = False
ROUTE = 'native'
SHARED_IN_ROUTE = True
EARLY_W = True
# row batches in [NATIVE_DOWN_MIN, NATIVE_DOWN_MAX] take the native down (qk_moe_dn16.cu: bit for bit i8x_dec_down,
# each warp streaming its own weight rows through a private cp.async ring); it needs the native route's lists and g128
# scales. Measured per decode MoE layer with the recorded expert popularity (CUDA graphs): 1-5 rows +0.5..+3 us, 6-16
# rows -1..-9 us, 17-48 rows -0.2..-3 us, 64+ rows slower; decode batches are multiples of 4 rows.
# the up's shared-expert blocks (positions 0 .. ceil(M / 16) - 1 of the block list) skip the dependency wait and run
# under the native route: they read only x (final once the route, which waits on x's producer, has launched the up),
# the static shared weights and their fixed slots M * TOPK + t; the routed blocks still wait for the route's lists
EARLY_SH = True
EARLY_SH_MAX = 32  # above, the shared blocks crowd the (longer) route: +1.6 us/layer at 128 rows, -1.6 at 32
# L2 eviction priority of the Triton up / down's streamed expert weights ('' is the plain load; 'evict_first' measured
# neutral in the layer flow, +-0.3 us at 4-128 rows)
WEIGHT_EVICT = ''
# Triton up / down: a CTA's fp16 group scales load once as a [groups, BLOCK] tile instead of one load per K step (the
# selected entry is the same scale: bit for bit), for row batches in SCALE_TILE_ROWS
SCALE_TILE_ROWS = (9, 128)
# row batches in [NATIVE_UP_MIN, NATIVE_UP_MAX] take the native up (qk_moe_up16.cu: bit for bit i8x_dec_up over the
# route's 16-token blocks; in the engine's decode-only lane -2.8 % per 8-row step; at 16 rows it ties the Triton up per
# kernel but the whole step ran ~1 % slower, so 9..16 rows stay on Triton)
NATIVE_UP_MIN = 5
NATIVE_UP_MAX = 8
NATIVE_DOWN_MIN = 6
NATIVE_DOWN_MAX = min(48, qk_moe_dn16.MAX_ROWS)
CONVERT = 'cvt'
TILES = {'g128': ((8, (64, 128, 8, 4), (64, 128, 4, 3), 4), (16, (32, 128, 4, 4), (64, 128, 4, 3), 16), (48, (32, 128, 4, 3), (64, 128, 4, 3), 1), (MAX_T, (32, 128, 4, 3), (128, 128, 4, 3), 2)), 'channel': ((MAX_T, (32, 256, 4, 3), (64, 128, 4, 3), 16),)}

def pick(rows: int, g128: bool, tiles=None):
    for max_rows, up, down, group_m in (tiles or TILES)['g128' if g128 else 'channel']:
        if rows <= max_rows:
            return (up, down, group_m)
    raise ValueError(f'i8x_decode: no tile for {rows} rows')

@triton.jit
def _i8_bf16(w, PRMT: tl.constexpr):
    if PRMT:
        out = tl.inline_asm_elementwise(asm='{\n            .reg .b32 a, m, f0, f1, f2, f3;\n            mov.b32 m, 0x4B000000;\n            xor.b32 a, $2, 0x80808080;\n            prmt.b32 f0, a, m, 0x7650;\n            prmt.b32 f1, a, m, 0x7651;\n            prmt.b32 f2, a, m, 0x7652;\n            prmt.b32 f3, a, m, 0x7653;\n            sub.f32 f0, f0, 0f4B000080;\n            sub.f32 f1, f1, 0f4B000080;\n            sub.f32 f2, f2, 0f4B000080;\n            sub.f32 f3, f3, 0f4B000080;\n            prmt.b32 $0, f0, f1, 0x7632;\n            prmt.b32 $1, f2, f3, 0x7632;\n            }', constraints='=r,=r,r', args=[w], dtype=tl.bfloat16, is_pure=True, pack=4)
    else:
        out = w.to(tl.bfloat16)
    return out

@triton.jit
def _mma(a, b, F32: tl.constexpr, PRMT: tl.constexpr):
    if F32:
        d = tl.dot(a.to(tl.float32), b.to(tl.float32))
    elif b.dtype == tl.int8:
        d = tl.dot(a, _i8_bf16(b, PRMT))
    else:
        d = tl.dot(a, b)
    return d

@triton.jit
def _mma_acc(a, b, acc, F32: tl.constexpr, PRMT: tl.constexpr):
    if F32:
        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
    elif b.dtype == tl.int8:
        acc = tl.dot(a, _i8_bf16(b, PRMT), acc)
    else:
        acc = tl.dot(a, b, acc)
    return acc

@triton.jit
def _swizzle(pid, n_tiles, GROUP_M: tl.constexpr):
    per_group = GROUP_M * n_tiles
    group = pid // per_group
    first = group * GROUP_M
    pid_b = first + pid % per_group % GROUP_M
    pid_n = pid % per_group // GROUP_M
    return (pid_b, pid_n)

@triton.jit
def _top8(logits, offs_e, E: tl.constexpr, TOPK: tl.constexpr):
    x = logits.to(tl.bfloat16).to(tl.float32)
    ex = tl.exp(x - tl.max(x, axis=1)[:, None])
    prob = ex / tl.sum(ex, axis=1)[:, None]
    cur = x
    offs_k = tl.arange(0, TOPK)
    ids = tl.zeros((x.shape[0], TOPK), dtype=tl.int32)
    wts = tl.zeros((x.shape[0], TOPK), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        mx = tl.max(cur, axis=1)
        win = tl.minimum(tl.min(tl.where(cur == mx[:, None], offs_e[None, :], E + 1), axis=1), E - 1).to(tl.int32)
        hit = offs_e[None, :] == win[:, None]
        w = tl.sum(tl.where(hit, prob, 0.0), axis=1)
        ids = tl.where(offs_k[None, :] == k, win[:, None], ids)
        wts = tl.where(offs_k[None, :] == k, w[:, None], wts)
        cur = tl.where(hit, -float('inf'), cur)
    total = tl.sum(wts, axis=1)
    wts = wts / tl.where(total > 0.0, total, 1.0)[:, None]
    return (ids, wts)

@triton.jit
def _up_tile(a_ptrs, tmask, g_ptrs, u_ptrs, sg_ptrs, su_ptrs, H: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr, INT8: tl.constexpr, G128: tl.constexpr, F32: tl.constexpr, PRMT: tl.constexpr, EVICT: tl.constexpr, SCALE_TILE: tl.constexpr=False):
    acc_g = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    if INT8 and G128 and SCALE_TILE:
        NG: tl.constexpr = H // GROUP
        offs_g = tl.arange(0, NG)
        sg_tile = tl.load(sg_ptrs[None, :] + offs_g[:, None]).to(tl.float32)
        su_tile = tl.load(su_ptrs[None, :] + offs_g[:, None]).to(tl.float32)
    for kb in range(0, H // BLOCK_K):
        a = tl.load(a_ptrs, mask=tmask[:, None], other=0.0)
        g = tl.load(g_ptrs, eviction_policy=EVICT)
        u = tl.load(u_ptrs, eviction_policy=EVICT)
        if INT8 and G128 and SCALE_TILE:
            sel = offs_g[:, None] == kb * BLOCK_K // GROUP
            acc_g += _mma(a, g, F32, PRMT) * tl.sum(tl.where(sel, sg_tile, 0.0), axis=0)[None, :]
            acc_u += _mma(a, u, F32, PRMT) * tl.sum(tl.where(sel, su_tile, 0.0), axis=0)[None, :]
        elif INT8 and G128:
            grp = kb * BLOCK_K // GROUP
            acc_g += _mma(a, g, F32, PRMT) * tl.load(sg_ptrs + grp).to(tl.float32)[None, :]
            acc_u += _mma(a, u, F32, PRMT) * tl.load(su_ptrs + grp).to(tl.float32)[None, :]
        else:
            acc_g = _mma_acc(a, g, acc_g, F32, PRMT)
            acc_u = _mma_acc(a, u, acc_u, F32, PRMT)
        a_ptrs += BLOCK_K
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K
    if INT8 and (not G128):
        acc_g = acc_g * tl.load(sg_ptrs).to(tl.float32)[None, :]
        acc_u = acc_u * tl.load(su_ptrs).to(tl.float32)[None, :]
    return (acc_g, acc_u)

@triton.jit
def _swiglu_store(acc_g, acc_u, h_ptr, slot, offs_n, tmask, I: tl.constexpr):
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    out_ptrs = h_ptr + slot[:, None].to(tl.int64) * I + offs_n[None, :]
    tl.store(out_ptrs, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=tmask[:, None])

@triton.jit
def i8x_dec_route(x_ptr, rw_ptr, pos_ptr, s13_ptr, part_ptr, tick_ptr, ids_ptr, wts_ptr, counts_ptr, tokens_ptr, slots_ptr, nblocks_ptr, bexp_ptr, bt0_ptr, h_ptr, M, K: tl.constexpr, I: tl.constexpr, E: tl.constexpr, SPLIT_K: tl.constexpr, TOPK: tl.constexpr, MAX_T: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr, SH_N: tl.constexpr, SH_K: tl.constexpr, SHARED: tl.constexpr, PDL: tl.constexpr, F32: tl.constexpr):
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    n_tb = tl.num_programs(0)
    K_PART: tl.constexpr = K // SPLIT_K
    PER_TB: tl.constexpr = E // BLOCK_E * SPLIT_K
    MAX_BPE: tl.constexpr = MAX_T // BLOCK_T
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < M
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if j >= PER_TB:
        offs_n = (j - PER_TB) * SH_N + tl.arange(0, SH_N)
        offs_k = tl.arange(0, SH_K)
        a_ptrs = x_ptr + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]
        g_ptrs = s13_ptr + offs_k[:, None] + offs_n[None, :].to(tl.int64) * K
        acc_g, acc_u = _up_tile(a_ptrs, tmask, g_ptrs, g_ptrs + I * K, g_ptrs, g_ptrs, K, BLOCK_T, SH_N, SH_K, 128, False, False, F32, False)
        _swiglu_store(acc_g, acc_u, h_ptr, M * TOPK + offs_t, offs_n, tmask, I)
        return
    pid_e = j // SPLIT_K
    pid_s = j % SPLIT_K
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = pid_s * K_PART + tl.arange(0, BLOCK_K)
    h_ptrs = x_ptr + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]
    w_ptrs = rw_ptr + offs_e[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
    for _ in range(0, K_PART, BLOCK_K):
        a = tl.load(h_ptrs, mask=tmask[:, None], other=0.0)
        acc = _mma_acc(a, tl.load(w_ptrs), acc, F32, False)
        h_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    tl.store(part_ptr + (pid_s * MAX_T + offs_t[:, None]).to(tl.int64) * E + offs_e[None, :], acc)
    tl.debug_barrier()
    ticket = tl.atomic_add(tick_ptr + pid_t, 1, sem='acq_rel', scope='gpu')
    if ticket == PER_TB - 1:
        tl.store(tick_ptr + pid_t, 0)
        offs_all = tl.arange(0, E)
        logits = tl.zeros((BLOCK_T, E), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            logits += tl.load(part_ptr + (s * MAX_T + offs_t[:, None]).to(tl.int64) * E + offs_all[None, :], cache_modifier='.cg')
        ids, wts = _top8(logits, offs_all, E, TOPK)
        offs_k8 = tl.arange(0, TOPK)
        slot = offs_t[:, None] * TOPK + offs_k8[None, :]
        tl.store(ids_ptr + slot, ids, mask=tmask[:, None])
        tl.store(wts_ptr + slot, wts, mask=tmask[:, None])
        valid = tmask & (tl.load(pos_ptr + offs_t, mask=tmask, other=0) != 0)
        vmask = valid[:, None] & (offs_k8[None, :] < TOPK)
        where = tl.atomic_add(counts_ptr + ids, 1, mask=vmask, sem='relaxed', scope='gpu')
        tl.store(tokens_ptr + ids * MAX_T + where, offs_t[:, None] + 0 * offs_k8[None, :], mask=vmask)
        tl.store(slots_ptr + ids * MAX_T + where, slot, mask=vmask)
        tl.debug_barrier()
        last = tl.atomic_add(tick_ptr + MAX_T // BLOCK_T, 1, sem='acq_rel', scope='gpu')
        if last == n_tb - 1:
            tl.store(tick_ptr + MAX_T // BLOCK_T, 0)
            n_sh = (M + BLOCK_T - 1) // BLOCK_T
            offs_b = tl.arange(0, MAX_BPE)
            tl.store(bexp_ptr + offs_b, E + 0 * offs_b, mask=offs_b < n_sh)
            tl.store(bt0_ptr + offs_b, offs_b * BLOCK_T, mask=offs_b < n_sh)
            cnt = tl.load(counts_ptr + offs_all, cache_modifier='.cg')
            nb = (cnt + BLOCK_T - 1) // BLOCK_T
            base = n_sh + tl.cumsum(nb, axis=0) - nb
            bm = offs_b[None, :] < nb[:, None]
            tl.store(bexp_ptr + base[:, None] + offs_b[None, :], offs_all[:, None] + 0 * offs_b[None, :], mask=bm)
            tl.store(bt0_ptr + base[:, None] + offs_b[None, :], offs_b[None, :] * BLOCK_T + 0 * offs_all[:, None], mask=bm)
            tl.store(nblocks_ptr, n_sh + tl.sum(nb, axis=0))

@triton.jit
def i8x_dec_gate(x_ptr, rw_ptr, part_ptr, M, K: tl.constexpr, E: tl.constexpr, SPLIT_K: tl.constexpr, MAX_T: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr, PDL: tl.constexpr, F32: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    pid_s = tl.program_id(2)
    K_PART: tl.constexpr = K // SPLIT_K
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = pid_s * K_PART + tl.arange(0, BLOCK_K)
    tmask = offs_t < M
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    h_ptrs = x_ptr + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]
    w_ptrs = rw_ptr + offs_e[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
    for _ in range(0, K_PART, BLOCK_K):
        a = tl.load(h_ptrs, mask=tmask[:, None], other=0.0)
        acc = _mma_acc(a, tl.load(w_ptrs), acc, F32, False)
        h_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    tl.store(part_ptr + (pid_s * MAX_T + offs_t[:, None]).to(tl.int64) * E + offs_e[None, :], acc, mask=tmask[:, None])

@triton.jit
def i8x_dec_topk(part_ptr, ids_ptr, wts_ptr, E: tl.constexpr, SPLIT_K: tl.constexpr, TOPK: tl.constexpr, MAX_T: tl.constexpr, PDL: tl.constexpr):
    t = tl.program_id(0)
    offs_e = tl.arange(0, E)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.zeros((1, E), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        x += tl.load(part_ptr + (s * MAX_T + t).to(tl.int64) * E + offs_e[None, :])
    ids, wts = _top8(x, offs_e, E, TOPK)
    offs_k = tl.arange(0, TOPK)
    tl.store(ids_ptr + t * TOPK + offs_k[None, :], ids)
    tl.store(wts_ptr + t * TOPK + offs_k[None, :], wts)

@triton.jit
def i8x_dec_lists(ids_ptr, pos_ptr, counts_ptr, tokens_ptr, slots_ptr, nblocks_ptr, bexp_ptr, bt0_ptr, M, E: tl.constexpr, TOPK: tl.constexpr, MAX_T: tl.constexpr, BLOCK: tl.constexpr, BLOCK_T: tl.constexpr, PDL: tl.constexpr):
    e = tl.program_id(0)
    total = M * TOPK
    offs = tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if e == E:
        count = M
    else:
        ids = tl.load(ids_ptr + offs, mask=offs < total, other=-1)
        rv = tl.load(pos_ptr + offs // TOPK, mask=offs < total, other=0) != 0
        hit = (ids == e) & rv
        hit_i = hit.to(tl.int32)
        where = tl.cumsum(hit_i, axis=0) - hit_i
        tl.store(tokens_ptr + e * MAX_T + where, offs // TOPK, mask=hit)
        tl.store(slots_ptr + e * MAX_T + where, offs, mask=hit)
        count = tl.sum(hit_i, axis=0)
        tl.store(counts_ptr + e, count)
    nb = (count + BLOCK_T - 1) // BLOCK_T
    base = tl.atomic_add(nblocks_ptr, nb)
    offs_b = tl.arange(0, MAX_T // BLOCK_T)
    bmask = offs_b < nb
    tl.store(bexp_ptr + base + offs_b, e + 0 * offs_b, mask=bmask)
    tl.store(bt0_ptr + base + offs_b, offs_b * BLOCK_T, mask=bmask)

@triton.jit
def _block_rows(e, t0, counts_ptr, tokens_ptr, slots_ptr, M, E: tl.constexpr, TOPK: tl.constexpr, MAX_T: tl.constexpr, BLOCK_T: tl.constexpr):
    offs_t = t0 + tl.arange(0, BLOCK_T)
    if e == E:
        tmask = offs_t < M
        tok = offs_t
        slot = M * TOPK + offs_t
    else:
        count = tl.load(counts_ptr + e)
        tmask = offs_t < count
        tok = tl.load(tokens_ptr + e * MAX_T + offs_t, mask=tmask, other=0)
        slot = tl.load(slots_ptr + e * MAX_T + offs_t, mask=tmask, other=0)
    return (tok, slot, tmask)

@triton.jit
def i8x_dec_up(x_ptr, q_ptr, s16_ptr, s13_ptr, h_ptr, counts_ptr, tokens_ptr, slots_ptr, nblocks_ptr, bexp_ptr, bt0_ptr, M, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, EB: tl.constexpr, C13: tl.constexpr, MAX_T: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, G128: tl.constexpr, SHARED_DONE: tl.constexpr, EARLY_SH: tl.constexpr, EVICT: tl.constexpr, PDL: tl.constexpr, F32: tl.constexpr, PRMT: tl.constexpr, SCALE_TILE: tl.constexpr=False):
    pid_b, pn = _swizzle(tl.program_id(0), I // BLOCK_N, GROUP_M)
    if EARLY_SH:
        sh = pid_b < (M + BLOCK_T - 1) // BLOCK_T
        if PDL:
            if not sh:
                tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        # a shared block's list entries are known without the route: (E, 16 * pid_b)
        if pid_b >= tl.where(sh, pid_b + 1, tl.load(nblocks_ptr)):
            return
        e = tl.where(sh, E, tl.load(bexp_ptr + pid_b))
        t0 = tl.where(sh, pid_b * BLOCK_T, tl.load(bt0_ptr + pid_b))
    else:
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        if pid_b >= tl.load(nblocks_ptr):
            return
        e = tl.load(bexp_ptr + pid_b)
        if SHARED_DONE and e == E:
            return
        t0 = tl.load(bt0_ptr + pid_b)
    tok, slot, tmask = _block_rows(e, t0, counts_ptr, tokens_ptr, slots_ptr, M, E, TOPK, MAX_T, BLOCK_T)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = x_ptr + tok[:, None].to(tl.int64) * H + offs_k[None, :]
    if e == E:
        gs_ptrs = s13_ptr + offs_k[:, None] + offs_n[None, :].to(tl.int64) * H
        acc_g, acc_u = _up_tile(a_ptrs, tmask, gs_ptrs, gs_ptrs + I * H, gs_ptrs, gs_ptrs, H, BLOCK_T, BLOCK_N, BLOCK_K, GROUP, False, G128, F32, PRMT, EVICT)
    else:
        base = e.to(tl.int64) * EB
        gq_ptrs = q_ptr + base + offs_k[:, None] + offs_n[None, :].to(tl.int64) * H
        sg_ptrs = s16_ptr + (base + C13) // 2 + offs_n.to(tl.int64) * (H // GROUP)
        acc_g, acc_u = _up_tile(a_ptrs, tmask, gq_ptrs, gq_ptrs + I * H, sg_ptrs, sg_ptrs + I * (H // GROUP), H, BLOCK_T, BLOCK_N, BLOCK_K, GROUP, True, G128, F32, PRMT, EVICT, SCALE_TILE)
    _swiglu_store(acc_g, acc_u, h_ptr, slot, offs_n, tmask, I)

@triton.jit
def i8x_dec_down(h_ptr, q_ptr, s16_ptr, s2_ptr, wts_ptr, cache_ptr, counts_ptr, tokens_ptr, slots_ptr, nblocks_ptr, bexp_ptr, bt0_ptr, M, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, EB: tl.constexpr, Q2: tl.constexpr, C2: tl.constexpr, MAX_T: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, G128: tl.constexpr, EARLY_W: tl.constexpr, EVICT: tl.constexpr, PDL: tl.constexpr, F32: tl.constexpr, PRMT: tl.constexpr, SCALE_TILE: tl.constexpr=False):
    pid_b, ph = _swizzle(tl.program_id(0), H // BLOCK_H, GROUP_M)
    offs_h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    if EARLY_W:
        tl.static_assert(I // BLOCK_K == 4, 'EARLY_W holds the tile as four K steps')
        n_blocks = tl.load(nblocks_ptr)
        e_early = tl.load(bexp_ptr + pid_b, mask=pid_b < n_blocks, other=E)
        wq_early = q_ptr + e_early.to(tl.int64) * EB + Q2 + offs_k[:, None] + offs_h[None, :].to(tl.int64) * I
        wm = (pid_b < n_blocks) & (e_early < E)
        w0 = tl.load(wq_early, mask=wm, other=0, eviction_policy=EVICT)
        w1 = tl.load(wq_early + BLOCK_K, mask=wm, other=0, eviction_policy=EVICT)
        w2 = tl.load(wq_early + 2 * BLOCK_K, mask=wm, other=0, eviction_policy=EVICT)
        w3 = tl.load(wq_early + 3 * BLOCK_K, mask=wm, other=0, eviction_policy=EVICT)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
    else:
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        n_blocks = tl.load(nblocks_ptr)
    if pid_b >= n_blocks:
        return
    e = tl.load(bexp_ptr + pid_b)
    t0 = tl.load(bt0_ptr + pid_b)
    tok, slot, tmask = _block_rows(e, t0, counts_ptr, tokens_ptr, slots_ptr, M, E, TOPK, MAX_T, BLOCK_T)
    a_ptrs = h_ptr + slot[:, None].to(tl.int64) * I + offs_k[None, :]
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    if e == E:
        ws_ptrs = s2_ptr + offs_k[:, None] + offs_h[None, :].to(tl.int64) * I
        as_ptrs = a_ptrs
        for _ in range(0, I, BLOCK_K):
            a = tl.load(as_ptrs, mask=tmask[:, None], other=0.0)
            acc = _mma_acc(a, tl.load(ws_ptrs, eviction_policy=EVICT), acc, F32, PRMT)
            as_ptrs += BLOCK_K
            ws_ptrs += BLOCK_K
        wgt = tl.where(tmask, 1.0, 0.0)
    else:
        base = e.to(tl.int64) * EB
        s_ptrs = s16_ptr + (base + C2) // 2 + offs_h.to(tl.int64) * (I // GROUP)
        if EARLY_W:
            a0 = tl.load(a_ptrs, mask=tmask[:, None], other=0.0)
            a1 = tl.load(a_ptrs + BLOCK_K, mask=tmask[:, None], other=0.0)
            a2 = tl.load(a_ptrs + 2 * BLOCK_K, mask=tmask[:, None], other=0.0)
            a3 = tl.load(a_ptrs + 3 * BLOCK_K, mask=tmask[:, None], other=0.0)
            if G128:
                acc += _mma(a0, w0, F32, PRMT) * tl.load(s_ptrs + 0 * BLOCK_K // GROUP).to(tl.float32)[None, :]
                acc += _mma(a1, w1, F32, PRMT) * tl.load(s_ptrs + 1 * BLOCK_K // GROUP).to(tl.float32)[None, :]
                acc += _mma(a2, w2, F32, PRMT) * tl.load(s_ptrs + 2 * BLOCK_K // GROUP).to(tl.float32)[None, :]
                acc += _mma(a3, w3, F32, PRMT) * tl.load(s_ptrs + 3 * BLOCK_K // GROUP).to(tl.float32)[None, :]
            else:
                acc = _mma_acc(a0, w0, acc, F32, PRMT)
                acc = _mma_acc(a1, w1, acc, F32, PRMT)
                acc = _mma_acc(a2, w2, acc, F32, PRMT)
                acc = _mma_acc(a3, w3, acc, F32, PRMT)
        else:
            w_ptrs = q_ptr + base + Q2 + offs_k[:, None] + offs_h[None, :].to(tl.int64) * I
            if G128 and SCALE_TILE:
                NG: tl.constexpr = I // GROUP
                offs_g = tl.arange(0, NG)
                s_tile = tl.load(s_ptrs[None, :] + offs_g[:, None]).to(tl.float32)
            for kb in range(0, I // BLOCK_K):
                a = tl.load(a_ptrs, mask=tmask[:, None], other=0.0)
                w = tl.load(w_ptrs, eviction_policy=EVICT)
                if G128 and SCALE_TILE:
                    sc = tl.sum(tl.where(offs_g[:, None] == kb * BLOCK_K // GROUP, s_tile, 0.0), axis=0)
                    acc += _mma(a, w, F32, PRMT) * sc[None, :]
                elif G128:
                    acc += _mma(a, w, F32, PRMT) * tl.load(s_ptrs + kb * BLOCK_K // GROUP).to(tl.float32)[None, :]
                else:
                    acc = _mma_acc(a, w, acc, F32, PRMT)
                a_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
        if not G128:
            acc = acc * tl.load(s_ptrs).to(tl.float32)[None, :]
        wgt = tl.load(wts_ptr + slot, mask=tmask, other=0.0)
    out_ptrs = cache_ptr + slot[:, None].to(tl.int64) * H + offs_h[None, :]
    tl.store(out_ptrs, (acc * wgt[:, None]).to(tl.bfloat16), mask=tmask[:, None])

@triton.jit
def i8x_dec_combine(cache_ptr, x_ptr, gate_w_ptr, pos_ptr, out_ptr, counts_ptr, nblocks_ptr, M, E: tl.constexpr, TOPK: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr, PDL: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    if PDL:
        tl.extra.cuda.gdc_wait()
    if t == 0 and tl.program_id(1) == 0:
        offs_e = tl.arange(0, E)
        tl.store(counts_ptr + offs_e, 0 * offs_e)
        tl.store(nblocks_ptr, 0)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if tl.load(pos_ptr + t) != 0:
        offs_x = tl.arange(0, H)
        h = tl.load(x_ptr + t * H + offs_x).to(tl.float32)
        w = tl.load(gate_w_ptr + offs_x).to(tl.float32)
        gate = tl.sigmoid(tl.sum(h * w, axis=0))
        for j in tl.static_range(TOPK):
            acc += tl.load(cache_ptr + (t * TOPK + j) * H + offs).to(tl.float32)
        acc += gate * tl.load(cache_ptr + (M * TOPK + t) * H + offs).to(tl.float32)
    tl.store(out_ptr + t * H + offs, acc.to(tl.bfloat16))

class Workspace:

    def __init__(self, experts, hidden, inter, device):
        self.max_blocks = experts + MAX_T * TOPK // BLOCK_T + MAX_T // BLOCK_T
        self.part = torch.empty(GATE_SPLIT_K, MAX_T, experts, dtype=torch.float32, device=device)
        self.tickets = torch.zeros(MAX_T // BLOCK_T + 1, dtype=torch.int32, device=device)
        self.ids = torch.empty(MAX_T * TOPK, dtype=torch.int32, device=device)
        self.wts = torch.empty(MAX_T * TOPK, dtype=torch.float32, device=device)
        self.counts = torch.zeros(experts, dtype=torch.int32, device=device)
        self.tokens = torch.empty(experts * MAX_T, dtype=torch.int32, device=device)
        self.slots = torch.empty(experts * MAX_T, dtype=torch.int32, device=device)
        self.nblocks = torch.zeros(1, dtype=torch.int32, device=device)
        self.bexp = torch.empty(self.max_blocks, dtype=torch.int32, device=device)
        self.bt0 = torch.empty(self.max_blocks, dtype=torch.int32, device=device)
        self.inter = torch.empty(MAX_T * (TOPK + 1), inter, dtype=torch.bfloat16, device=device)
        self.cache = torch.empty(MAX_T * (TOPK + 1), hidden, dtype=torch.bfloat16, device=device)

    def nbytes(self):
        tensors = (self.part, self.tickets, self.ids, self.wts, self.counts, self.tokens, self.slots, self.nblocks, self.bexp, self.bt0, self.inter, self.cache)
        return sum((t.numel() * t.element_size() for t in tensors))
_workspaces = {}

def workspace(device, experts=256, hidden=2048, inter=512):
    key = (device, experts, hidden, inter)
    ws = _workspaces.get(key)
    if ws is None:
        ws = _workspaces[key] = Workspace(experts, hidden, inter, device)
    return ws

def moe_row_batch(x, positions, router_w, gate_w, blocks, offsets, s13, s2, g128=True, route=None, tiles=None, ws=None, shared_in_route=None, early_w=None, convert=None, norm=None):
    M, H = x.shape
    E, eb = blocks.shape
    q2, c13, c2, eb_ = offsets
    inter = s2.shape[1]
    if not 1 <= M <= MAX_T or x.dtype != torch.bfloat16 or (not x.is_contiguous()) or (eb != eb_):
        raise ValueError(f'i8x_decode: x must be contiguous bf16 [1..{MAX_T}, H], got {tuple(x.shape)} {x.dtype}')
    route = route or ROUTE
    shared_in_route = SHARED_IN_ROUTE if shared_in_route is None else shared_in_route
    shared_in_route = shared_in_route and route == 'fused'
    early_w = EARLY_W if early_w is None else early_w
    ws = ws or workspace(x.device, E, H, inter)
    up, down, group_m = pick(M, g128, tiles)
    pdl, f32 = (not INTERPRETER, INTERPRETER)
    prmt = (CONVERT if convert is None else convert) == 'prmt' and (not INTERPRETER)
    q8, s16 = (blocks.view(torch.int8), blocks.view(torch.float16))
    n_tb = triton.cdiv(M, BLOCK_T)
    gate_bk = min(GATE_BLOCK_K, H // GATE_SPLIT_K)
    if route == 'native':
        qk_route.route(x, router_w, positions, ws.part, ws.tickets, ws.ids, ws.wts, ws.counts, ws.tokens, ws.slots, ws.nblocks, ws.bexp, ws.bt0, M > BLOCK_T)
    elif route == 'fused':
        per_tb = E // GATE_BLOCK_E * GATE_SPLIT_K
        sh_n = min(SHARED_UP_N, inter)
        i8x_dec_route[n_tb, per_tb + (inter // sh_n if shared_in_route else 0)](x, router_w, positions, s13, ws.part, ws.tickets, ws.ids, ws.wts, ws.counts, ws.tokens, ws.slots, ws.nblocks, ws.bexp, ws.bt0, ws.inter, M, K=H, I=inter, E=E, SPLIT_K=GATE_SPLIT_K, TOPK=TOPK, MAX_T=MAX_T, BLOCK_T=BLOCK_T, BLOCK_E=GATE_BLOCK_E, BLOCK_K=gate_bk, SH_N=sh_n, SH_K=min(128, H), SHARED=shared_in_route, PDL=pdl, F32=f32, num_warps=4, num_stages=2, launch_pdl=pdl)
    elif route == 'an':
        i8x_dec_gate[n_tb, E // GATE_BLOCK_E, GATE_SPLIT_K](x, router_w, ws.part, M, K=H, E=E, SPLIT_K=GATE_SPLIT_K, MAX_T=MAX_T, BLOCK_T=BLOCK_T, BLOCK_E=GATE_BLOCK_E, BLOCK_K=gate_bk, PDL=pdl, F32=f32, num_warps=4, num_stages=2, launch_pdl=pdl)
        i8x_dec_topk[M,](ws.part, ws.ids, ws.wts, E=E, SPLIT_K=GATE_SPLIT_K, TOPK=TOPK, MAX_T=MAX_T, PDL=pdl, num_warps=1, launch_pdl=pdl)
        i8x_dec_lists[E + 1,](ws.ids, positions, ws.counts, ws.tokens, ws.slots, ws.nblocks, ws.bexp, ws.bt0, M, E=E, TOPK=TOPK, MAX_T=MAX_T, BLOCK=triton.next_power_of_2(M * TOPK), BLOCK_T=BLOCK_T, PDL=pdl, num_warps=2 if M * TOPK <= 512 else 8, launch_pdl=pdl)
    else:
        raise ValueError(f'i8x_decode: unknown ROUTE {route!r}')
    blocks_n = min(E, M * TOPK) + M * TOPK // BLOCK_T + n_tb
    blocks_n = triton.cdiv(blocks_n, group_m) * group_m
    bn, bk, warps, stages = up
    if NATIVE_UP_MIN <= M <= min(NATIVE_UP_MAX, qk_moe_up16.MAX_ROWS) and route == 'native' and g128 and pdl and not shared_in_route and (E, H, inter) == (256, 2048, 512):
        qk_moe_up16.up(x, blocks, s13, ws.counts, ws.tokens, ws.slots, ws.nblocks, ws.bexp, ws.bt0, ws.inter, eb, c13, int(EARLY_SH and M <= EARLY_SH_MAX))
    else:
        i8x_dec_up[blocks_n * (inter // bn),](x, q8, s16, s13, ws.inter, ws.counts, ws.tokens, ws.slots, ws.nblocks, ws.bexp, ws.bt0, M, H=H, I=inter, E=E, EB=eb, C13=c13, MAX_T=MAX_T, TOPK=TOPK, GROUP=GROUP, BLOCK_T=BLOCK_T, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m, G128=g128, SHARED_DONE=shared_in_route, EARLY_SH=EARLY_SH and M <= EARLY_SH_MAX and route == 'native' and not shared_in_route, EVICT=WEIGHT_EVICT, SCALE_TILE=SCALE_TILE_ROWS[0] <= M <= SCALE_TILE_ROWS[1], PDL=pdl, F32=f32, PRMT=prmt, num_warps=warps, num_stages=stages, launch_pdl=pdl)
    if NATIVE_DOWN_MIN <= M <= NATIVE_DOWN_MAX and route == 'native' and g128 and pdl and (E, H, inter) == (256, 2048, 512):
        qk_moe_dn16.down(ws.inter, blocks, s2, ws.wts, ws.counts, ws.slots, ws.nblocks, ws.bexp, ws.bt0, ws.cache, eb, q2, c2, M)
    else:
        bh, bk, warps, stages = down
        i8x_dec_down[blocks_n * (H // bh),](ws.inter, q8, s16, s2, ws.wts, ws.cache, ws.counts, ws.tokens, ws.slots, ws.nblocks, ws.bexp, ws.bt0, M, H=H, I=inter, E=E, EB=eb, Q2=q2, C2=c2, MAX_T=MAX_T, TOPK=TOPK, GROUP=GROUP, BLOCK_T=BLOCK_T, BLOCK_H=bh, BLOCK_K=bk, GROUP_M=group_m, G128=g128, EARLY_W=early_w and inter // bk == 4, EVICT=WEIGHT_EVICT, SCALE_TILE=SCALE_TILE_ROWS[0] <= M <= SCALE_TILE_ROWS[1], PDL=pdl, F32=f32, PRMT=prmt, num_warps=warps, num_stages=stages, launch_pdl=pdl)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    if norm is not None:
        # kcombnorm: the combine also writes layer i+1's stock input norm of (out, residual) into y / r_next.
        # A 4th entry `early` releases the next launch at the combine's start (layer i+1's native in_proj,
        # qk_route.inproj16, streams its static weights into shared memory until its own wait).
        residual, norm_w, eps = norm[:3]
        early = len(norm) > 3 and bool(norm[3])
        y, r_next = torch.empty_like(out), torch.empty_like(out)
        qk_route.combine_norm(ws.cache, x, gate_w, positions, out, ws.counts, ws.nblocks, residual, norm_w, eps, y, r_next, early)
        return out, y, r_next
    block_h = min(512, H)
    i8x_dec_combine[M, H // block_h](ws.cache, x, gate_w, positions, out, ws.counts, ws.nblocks, M, E=E, TOPK=TOPK, H=H, BLOCK_H=block_h, PDL=pdl, num_warps=4, launch_pdl=pdl)
    return out

def check_tiles(tiles, hidden, inter):
    for g128 in (True, False):
        for _, up, down, group_m in tiles['g128' if g128 else 'channel']:
            bn, bk_u, _, _ = up
            bh, bk_d, _, _ = down
            if inter % bn or hidden % bk_u or hidden % bh or inter % bk_d or (group_m < 1):
                return False
            if g128 and (GROUP % bk_u or GROUP % bk_d):
                return False
    return True
