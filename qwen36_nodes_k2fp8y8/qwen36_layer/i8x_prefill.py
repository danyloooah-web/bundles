from __future__ import annotations
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
GROUP = 128
TOPK = 8
INTERPRETER = False
KIND = 'swap'
FOLD = 'bf16'
FUSED_COMBINE = True
MAX_TOKENS = 16384
FOLDS = {'acc': 0, 'f32': 1, 'bf16': 2}
TILES = ((64, 16, (64, 128, 4, 3), (64, 128, 4, 3), 1), (1 << 30, 64, (64, 64, 8, 3), (128, 64, 4, 3), 8))
SWAP_TILES = ((64, 16, (32, 128, 4, 3), (64, 128, 4, 3), 1), (1 << 30, 128, (128, 32, 8, 4), (256, 32, 8, 4), 8))

def pick(tokens: int, kind: str | None=None, tiles=None):
    for max_tokens, block_m, up, down, group_m in tiles or (SWAP_TILES if (kind or KIND) == 'swap' else TILES):
        if tokens <= max_tokens:
            return (block_m, up, down, group_m)
    raise ValueError(f'i8x_prefill: no tile for {tokens} tokens')

@triton.jit
def _mma(a, b, F32: tl.constexpr):
    if F32:
        d = tl.dot(a.to(tl.float32), b.to(tl.float32))
    else:
        d = tl.dot(a, b.to(tl.bfloat16))
    return d

@triton.jit
def _tile(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    return (first_m + pid % in_group % size_m, pid % in_group // size_m)

@triton.jit(do_not_specialize=['EM', 'num_valid'])
def i8x_pf_up_tri(x_ptr, q_ptr, s_ptr, h_ptr, sorted_ptr, expert_ptr, ntpp_ptr, EM, num_valid, stride_x, H: tl.constexpr, I: tl.constexpr, EB: tl.constexpr, C13: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, F32: tl.constexpr):
    pid_m, pid_n = _tile(tl.program_id(0), tl.cdiv(EM, BLOCK_M), I // BLOCK_N, GROUP_M)
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    slot = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    live = slot < num_valid
    e = tl.load(expert_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    h_ptrs = h_ptr + slot[:, None].to(tl.int64) * I + offs_n[None, :]
    if e < 0:
        tl.store(h_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16), mask=live[:, None])
        return
    a_ptrs = x_ptr + (slot // TOPK).to(tl.int64)[:, None] * stride_x + offs_k[None, :]
    g_ptrs = q_ptr + e * EB + offs_n[None, :].to(tl.int64) * H + offs_k[:, None]
    u_ptrs = g_ptrs + I * H
    sg_ptrs = s_ptr + e * (EB // 2) + C13 // 2 + offs_n.to(tl.int64) * (H // GROUP)
    su_ptrs = sg_ptrs + I * (H // GROUP)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, H // BLOCK_K):
        grp = kb * BLOCK_K // GROUP
        a = tl.load(a_ptrs, mask=live[:, None], other=0.0)
        acc_g += _mma(a, tl.load(g_ptrs), F32) * tl.load(sg_ptrs + grp).to(tl.float32)[None, :]
        acc_u += _mma(a, tl.load(u_ptrs), F32) * tl.load(su_ptrs + grp).to(tl.float32)[None, :]
        a_ptrs += BLOCK_K
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    tl.store(h_ptrs, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=live[:, None])

@triton.jit(do_not_specialize=['EM', 'num_valid'])
def i8x_pf_down_tri(h_ptr, q_ptr, s_ptr, y_ptr, w_ptr, sorted_ptr, expert_ptr, ntpp_ptr, EM, num_valid, H: tl.constexpr, I: tl.constexpr, EB: tl.constexpr, Q2: tl.constexpr, C2: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, F32: tl.constexpr):
    pid_m, pid_n = _tile(tl.program_id(0), tl.cdiv(EM, BLOCK_M), H // BLOCK_N, GROUP_M)
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    slot = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    live = slot < num_valid
    e = tl.load(expert_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    y_ptrs = y_ptr + slot[:, None].to(tl.int64) * H + offs_n[None, :]
    if e < 0:
        tl.store(y_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16), mask=live[:, None])
        return
    a_ptrs = h_ptr + slot[:, None].to(tl.int64) * I + offs_k[None, :]
    w_ptrs = q_ptr + e * EB + Q2 + offs_n[None, :].to(tl.int64) * I + offs_k[:, None]
    s_ptrs = s_ptr + e * (EB // 2) + C2 // 2 + offs_n.to(tl.int64) * (I // GROUP)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, I // BLOCK_K):
        grp = kb * BLOCK_K // GROUP
        a = tl.load(a_ptrs, mask=live[:, None], other=0.0)
        acc += _mma(a, tl.load(w_ptrs), F32) * tl.load(s_ptrs + grp).to(tl.float32)[None, :]
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    rw = tl.load(w_ptr + slot, mask=live, other=0.0)
    tl.store(y_ptrs, (acc * rw[:, None]).to(tl.bfloat16), mask=live[:, None])

@triton.jit
def _swap_step(acc, w, bt, s_ptrs, grp, G128: tl.constexpr, FOLD: tl.constexpr, F32: tl.constexpr):
    if F32:
        bt = bt.to(tl.float32)
    if G128 and FOLD == 2:
        sc = tl.load(s_ptrs + grp).to(tl.bfloat16)
        if F32:
            wa = (w.to(tl.float32) * sc.to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
        else:
            wa = w.to(tl.bfloat16) * sc[:, None]
        acc = tl.dot(wa, bt, acc)
    elif G128 and FOLD == 1:
        wa = (w.to(tl.float32) * tl.load(s_ptrs + grp).to(tl.float32)[:, None]).to(tl.bfloat16)
        if F32:
            wa = wa.to(tl.float32)
        acc = tl.dot(wa, bt, acc)
    else:
        if F32:
            wa = w.to(tl.float32)
        else:
            wa = w.to(tl.bfloat16)
        if G128:
            acc += tl.dot(wa, bt) * tl.load(s_ptrs + grp).to(tl.float32)[:, None]
        else:
            acc = tl.dot(wa, bt, acc)
    return acc

@triton.jit(do_not_specialize=['EM', 'num_valid'])
def i8x_pf_up_swap(x_ptr, q_ptr, s_ptr, h_ptr, sorted_ptr, expert_ptr, ntpp_ptr, EM, num_valid, stride_x, H: tl.constexpr, I: tl.constexpr, EB: tl.constexpr, C13: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, G128: tl.constexpr, FOLD: tl.constexpr, F32: tl.constexpr):
    pid_m, pid_n = _tile(tl.program_id(0), tl.cdiv(EM, BLOCK_M), I // BLOCK_N, GROUP_M)
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    slot = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    live = slot < num_valid
    e = tl.load(expert_ptr + pid_m).to(tl.int64)
    offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_ptrs = h_ptr + slot[None, :].to(tl.int64) * I + offs_c[:, None]
    if e < 0:
        tl.store(h_ptrs, tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.bfloat16), mask=live[None, :])
        return
    offs_r = tl.arange(0, 2 * BLOCK_N)
    wrow = tl.where(offs_r < BLOCK_N, pid_n * BLOCK_N + offs_r, I + pid_n * BLOCK_N + offs_r - BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = q_ptr + e * EB + wrow[:, None].to(tl.int64) * H + offs_k[None, :]
    s_ptrs = s_ptr + e * (EB // 2) + C13 // 2 + wrow.to(tl.int64) * (H // GROUP)
    xt_ptrs = x_ptr + (slot // TOPK).to(tl.int64)[None, :] * stride_x + offs_k[:, None]
    acc = tl.zeros((2 * BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kb in range(0, H // BLOCK_K):
        xt = tl.load(xt_ptrs, mask=live[None, :], other=0.0)
        acc = _swap_step(acc, tl.load(w_ptrs), xt, s_ptrs, kb * BLOCK_K // GROUP, G128, FOLD, F32)
        w_ptrs += BLOCK_K
        xt_ptrs += BLOCK_K
    if not G128:
        acc = acc * tl.load(s_ptrs).to(tl.float32)[:, None]
    gu = acc.to(tl.bfloat16)
    gate, up = tl.split(tl.permute(tl.reshape(gu, (2, BLOCK_N, BLOCK_M)), (1, 2, 0)))
    gate = gate.to(tl.float32)
    tl.store(h_ptrs, (gate / (1.0 + tl.exp(-gate)) * up.to(tl.float32)).to(tl.bfloat16), mask=live[None, :])

@triton.jit(do_not_specialize=['EM', 'num_valid'])
def i8x_pf_down_swap(h_ptr, q_ptr, s_ptr, y_ptr, w_ptr, sorted_ptr, expert_ptr, ntpp_ptr, sh_ptr, out_ptr, cnt_ptr, EM, num_valid, H: tl.constexpr, I: tl.constexpr, EB: tl.constexpr, Q2: tl.constexpr, C2: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, G128: tl.constexpr, FOLD: tl.constexpr, FUSED: tl.constexpr, F32: tl.constexpr):
    N_TILES: tl.constexpr = H // BLOCK_N
    pid_m, pid_n = _tile(tl.program_id(0), tl.cdiv(EM, BLOCK_M), N_TILES, GROUP_M)
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    slot = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    live = slot < num_valid
    e = tl.load(expert_ptr + pid_m).to(tl.int64)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + slot[None, :].to(tl.int64) * H + rows[:, None]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    if e >= 0:
        offs_k = tl.arange(0, BLOCK_K)
        w_ptrs = q_ptr + e * EB + Q2 + rows[:, None].to(tl.int64) * I + offs_k[None, :]
        s_ptrs = s_ptr + e * (EB // 2) + C2 // 2 + rows.to(tl.int64) * (I // GROUP)
        ht_ptrs = h_ptr + slot[None, :].to(tl.int64) * I + offs_k[:, None]
        for kb in range(0, I // BLOCK_K):
            ht = tl.load(ht_ptrs, mask=live[None, :], other=0.0)
            acc = _swap_step(acc, tl.load(w_ptrs), ht, s_ptrs, kb * BLOCK_K // GROUP, G128, FOLD, F32)
            w_ptrs += BLOCK_K
            ht_ptrs += BLOCK_K
        if not G128:
            acc = acc * tl.load(s_ptrs).to(tl.float32)[:, None]
        rw = tl.load(w_ptr + slot, mask=live, other=0.0)
        acc = acc * rw[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=live[None, :])
    if FUSED:
        tok = (slot // TOPK).to(tl.int64)
        tl.debug_barrier()
        seen = tl.atomic_add(cnt_ptr + tok * N_TILES + pid_n, 1, mask=live, sem='acq_rel', scope='gpu')
        last = live & (seen == TOPK - 1)
        tl.debug_barrier()
        CB: tl.constexpr = 64 if BLOCK_N > 64 else BLOCK_N
        for c in tl.static_range(BLOCK_N // CB):
            rc = pid_n * BLOCK_N + c * CB + tl.arange(0, CB)
            comb = tl.zeros((CB, BLOCK_M), dtype=tl.float32)
            for j in tl.static_range(TOPK):
                comb += tl.load(y_ptr + (tok * TOPK + j)[None, :] * H + rc[:, None], mask=last[None, :], other=0.0, cache_modifier='.cg').to(tl.float32)
            sh = tl.load(sh_ptr + tok[None, :] * H + rc[:, None], mask=last[None, :], other=0.0).to(tl.float32)
            res = comb.to(tl.bfloat16).to(tl.float32) + sh
            tl.store(out_ptr + tok[None, :] * H + rc[:, None], res.to(tl.bfloat16), mask=last[None, :])
        tl.store(cnt_ptr + tok * N_TILES + pid_n, 0 * tok.to(tl.int32), mask=last)

@triton.jit
def i8x_pf_combine(y_ptr, sh_ptr, out_ptr, H: tl.constexpr, TOPK: tl.constexpr, BLOCK_H: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        acc += tl.load(y_ptr + (t * TOPK + j) * H + offs).to(tl.float32)
    r = acc.to(tl.bfloat16).to(tl.float32) + tl.load(sh_ptr + t * H + offs).to(tl.float32)
    tl.store(out_ptr + t * H + offs, r.to(tl.bfloat16))
_counts = {}

def _combine_counts(device, n_tiles):
    key = (device, n_tiles)
    cnt = _counts.get(key)
    if cnt is None:
        cnt = _counts[key] = torch.zeros(MAX_TOKENS * n_tiles, dtype=torch.int32, device=device)
    return cnt

def _stock_align():
    from sglang.srt.layers.moe.moe_runner.triton_utils import moe_align_block_size
    return moe_align_block_size

def moe_prefill(x, router_w, blocks, hidden, inter, offsets, topk, shared, align=None, kind=None, g128=True, tiles=None, fold=None, fused=None):
    st = _setup(x, router_w, blocks, hidden, inter, offsets, topk, shared, align, kind, g128, tiles, fold, fused)
    _launch_up(st)
    _launch_down(st)
    if not st['fused']:
        _launch_combine(st)
    return st['out']

def moe_prefill_parts(x, router_w, blocks, hidden, inter, offsets, topk, shared, align=None, kind=None, g128=True, tiles=None, fold=None, fused=None):
    st = _setup(x, router_w, blocks, hidden, inter, offsets, topk, shared, align, kind, g128, tiles, fold, fused)
    launches = {'up': lambda: _launch_up(st), 'down': lambda: _launch_down(st)}
    if not st['fused']:
        launches['combine'] = lambda: _launch_combine(st)
    return (st['out'], launches)

def _setup(x, router_w, blocks, hidden, inter, offsets, topk, shared, align, kind, g128, tiles, fold, fused):
    kind = kind or KIND
    fold = FOLDS[fold or FOLD]
    fused = (FUSED_COMBINE if fused is None else fused) and kind == 'swap'
    tokens = x.shape[0]
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or x.shape[1] != hidden:
        raise ValueError(f'i8x_prefill: x must be bf16 [T, {hidden}] with unit column stride')
    q2, c13, c2, eb = offsets
    experts = blocks.shape[0]
    if blocks.dtype != torch.uint8 or tuple(blocks.shape) != (experts, eb) or blocks.stride() != (eb, 1):
        raise ValueError(f'i8x_prefill: the copy must be contiguous uint8 [E, {eb}]')
    logits = F.linear(x, router_w)
    topk_w, topk_ids = topk(x, logits)
    topk_w = topk_w.to(torch.float32).contiguous()
    shared_out = shared(x).contiguous()
    block_m, up, down, group_m = pick(tokens, kind, tiles)
    sorted_ids, expert_ids, ntpp = (align or _stock_align())(topk_ids, block_m, experts)
    slots = tokens * TOPK
    q8, s16 = (blocks.view(torch.int8), blocks.view(torch.float16))
    h = torch.empty((slots, inter), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((slots, hidden), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device)
    if kind not in ('swap', 'tri'):
        raise ValueError(f'i8x_prefill: unknown KIND {kind!r}')
    fused = fused and tokens <= MAX_TOKENS
    cnt = _combine_counts(x.device, hidden // down[0]) if fused else topk_ids
    return dict(x=x, q8=q8, s16=s16, h=h, y=y, out=out, topk_w=topk_w, sorted_ids=sorted_ids, expert_ids=expert_ids, ntpp=ntpp, shared_out=shared_out, cnt=cnt, em=sorted_ids.shape[0], slots=slots, tokens=tokens, hidden=hidden, inter=inter, offsets=offsets, block_m=block_m, up=up, down=down, group_m=group_m, kind=kind, g128=g128, fold=fold, fused=fused, f32=INTERPRETER)

def _launch_up(st):
    bn, bk, warps, stages = st['up']
    q2, c13, c2, eb = st['offsets']
    grid = (triton.cdiv(st['em'], st['block_m']) * (st['inter'] // bn),)
    args = (st['x'], st['q8'], st['s16'], st['h'], st['sorted_ids'], st['expert_ids'], st['ntpp'], st['em'], st['slots'], st['x'].stride(0))
    common = dict(H=st['hidden'], I=st['inter'], EB=eb, C13=c13, TOPK=TOPK, GROUP=GROUP, BLOCK_M=st['block_m'], BLOCK_N=bn, BLOCK_K=bk, GROUP_M=st['group_m'], F32=st['f32'], num_warps=warps, num_stages=stages)
    if st['kind'] == 'swap':
        i8x_pf_up_swap[grid](*args, G128=st['g128'], FOLD=st['fold'], **common)
    else:
        i8x_pf_up_tri[grid](*args, **common)

def _launch_down(st):
    bn, bk, warps, stages = st['down']
    q2, c13, c2, eb = st['offsets']
    grid = (triton.cdiv(st['em'], st['block_m']) * (st['hidden'] // bn),)
    common = dict(H=st['hidden'], I=st['inter'], EB=eb, Q2=q2, C2=c2, GROUP=GROUP, BLOCK_M=st['block_m'], BLOCK_N=bn, BLOCK_K=bk, GROUP_M=st['group_m'], F32=st['f32'], num_warps=warps, num_stages=stages)
    if st['kind'] == 'swap':
        i8x_pf_down_swap[grid](st['h'], st['q8'], st['s16'], st['y'], st['topk_w'], st['sorted_ids'], st['expert_ids'], st['ntpp'], st['shared_out'], st['out'], st['cnt'], st['em'], st['slots'], TOPK=TOPK, G128=st['g128'], FOLD=st['fold'], FUSED=st['fused'], **common)
    else:
        i8x_pf_down_tri[grid](st['h'], st['q8'], st['s16'], st['y'], st['topk_w'], st['sorted_ids'], st['expert_ids'], st['ntpp'], st['em'], st['slots'], **common)

def _launch_combine(st):
    block_h = min(1024, st['hidden'])
    i8x_pf_combine[st['tokens'], st['hidden'] // block_h](st['y'], st['shared_out'], st['out'], H=st['hidden'], TOPK=TOPK, BLOCK_H=block_h, num_warps=4)
