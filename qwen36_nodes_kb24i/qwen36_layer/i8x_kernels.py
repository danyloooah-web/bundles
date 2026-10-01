from __future__ import annotations
import os
import sys
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from qwen36_layer import i8x_decode, i8x_prefill, i8x_up8
from qwen36_layer.dense8 import GROUP, dequantize, quantize
from qwen36_layer.evidence import branch_evidence
from qwen36_layer.lt import linear as lt_linear
LAYOUT = 'q8-block-v1'
SHAPE = (256, 2048, 512)
# PRECISION CHANGE (after 51f3e0f9): per-channel INT8 expert scales, for the INT8-tensor-core prefill up (i8x_up8.py)
SCALES = 'channel'
DECODE = ((16, 'tri8_an'), (256, 'tri8_late'))
TRI8_VARIANTS = {'tri8': {}, 'tri8_an': {'route': 'native', 'shared_in_route': False}, 'tri8_late': {'early_w': False}}
FP16_MIN_NORMAL = 2.0 ** (-14)
_active = {'fallback': None, 'warmed': False, 'warm_error': None}
PREFILL = 'king_i8'
PREFILL_DOWN = 'held'
# ---- v2f port switches (on b9d3c95a). Every switch at its OFF value ('up8' / False) serves b9d3c95a's prefill exactly.
# Shape classes of a prefill chunk (layer.py, prefill_shape): 'long' = some request of the chunk already holds more than
# LONG_SEQ_MIN tokens (the 64k/4k cell's chunks 2..8), 'short' = everything else (the 8k/1k cell, and each 64k request's
# first chunk, which looks the same as an 8k prompt to the layer).
LONG_SEQ_MIN = 16384
# Lane P up (PRECISION CHANGE, a different one from the parent's up8): which INT8 prefill up serves a chunk of >= UP8_MIN
# rows. 'up8' = the parent's up_i8g (x per row with 256-column integer multipliers, the shared expert INT8 per channel);
# 'lanep' = our W8A8 up (qk_moe_prefill_i8.up_w8a8: x per token and 128-column group, per-group fp32 promotion over
# the copy's scales, the shared expert's gate/up in BF16). The down (held BF16 w2 + the parent's y8) is unchanged.
UP_PATH = {'short': 'up8', 'long': 'up8'}
# Per-layer override: layers whose prefill up is always 'lanep' (its MoE output error is several times smaller than
# up8's; used to pay for another lane's precision change on a layer the node audit reads thin). Empty = none.
LANEP_LAYERS = frozenset({10, 12, 22})
# Lane P ping-pong promotion (bit for bit the plain lanep up): two s32 accumulators, the next K block's wgmmas issued
# before the previous one is promoted.
PREFILL_UP_PP = True
# Lane PD2 (PRECISION CHANGE): the routed down on INT8 tensor cores after either up (qk_moe_prefill_i8.down_w8a8: h
# quantized per row and 128-column group, the copy's w2 rows, ping-pong promotion); the shared down and the combine stay
# the king's BF16 kernel. On y8 layers its routed y is stored int8 with a scale per row and 128 columns (stacks with the
# parent's y8: the combine reads it through kY8 + kY128); elsewhere bf16 y.
DOWN_W8A8 = {'short': False, 'long': False}
W8A8_MARKER = 'prefill_w8a8'
W8A8_DN_MARKER = 'prefill_w8a8_dn'
# warm-up refusal of the port paths: the MoE block against king_i8 on the warm-up rows (median row, relative)
PORT_WARM_MAX_P50 = 0.05
PREFILL_FOLD = 'bf16'
# prefill chunks of at least UP8_MIN rows run the MoE up on INT8 tensor cores (i8x_up8.py, PRECISION CHANGE)
UP8_MIN = 1024
# the prefill MoE up as qk_moe_prefill_i8.cu up_i8g (native, the same per-group INT8 rows and per-channel INT8 copy as
# i8x_up8.py; the shared expert's gate/up rows quantized per channel per call: a PRECISION CHANGE for that expert)
UP_NATIVE = True
UP_NATIVE_OK = True
FOLD_MODES = {'exact': 0, 'bf16': 8}
# kb1 (on 3d6e3bef; bit for bit its up8 prefill): the up8 chunk's routing and x / shared gate-up quantization in one
# topk launch (qk_moe_prefill_i8.route_i8q = route_i8 + i8x_up8.quant_img + quantize_channel's values, our k2fp8y19),
# and chunks of >= UP_WS_MIN rows on the warp-specialized up with the consumer warpgroups taking turns on the tensor
# cores (up_i8gp, our k2fp8y55k / y59k: h bit for bit up_i8g's). Knobs at the parent's values: y scale amax / 127,
# IMG multiplier cap 64, tanh silu.
KNOB_YDIV, KNOB_QM, KNOB_TANH = 127.0, 64.0, 1
UP_WS_MIN = 6144
UP_FUSED_QUANT = True
# kb23n (bit for bit): the post-attention norm of a chunk route_pre_ready() admits runs add_norm_route (layer.py), which
# leaves route_i8q's per-token extras (shared gate sg, IMG rows xq / xs / xm) of the normed rows; prefill_up8 then routes
# with route_i8q_pre over them instead of re-reading x in route_i8q.
ROUTE_PRE_MARKER = 'route_pre'
_prefill_state = {'fallback': None, 'warmed': False, 'error': None, 'counters': {}, 'up8': None, 'port': None}

def offsets(hidden=SHAPE[1], inter=SHAPE[2]):
    q2 = 2 * inter * hidden
    c13 = q2 + hidden * inter
    c2 = c13 + 2 * inter * (hidden // GROUP) * 2
    return (q2, c13, c2, c2 + hidden * (inter // GROUP) * 2)
Q2_OFF, C13_OFF, C2_OFF, EXPERT_BYTES = offsets()
LAYER_BYTES = SHAPE[0] * EXPERT_BYTES

class Copy:
    __slots__ = ('blocks', 'table', 'hidden', 'inter', 'scales', 'w2', 'w2_host')

    def __init__(self, blocks, table, hidden, inter, scales=None):
        self.blocks, self.table, self.hidden, self.inter = (blocks, table, hidden, inter)
        self.scales = scales or SCALES
        self.w2 = None
        self.w2_host = None

def g128(copy):
    # the per-channel quantizer writes each row's scale into all of its 128-column group slots, so the copy keeps the
    # group layout: every group-scaled decode kernel (native and Triton) reads it unchanged
    return True


def channel(copy):
    """The copy is quantized per channel (one scale per row, replicated per group): the prefill up8 path needs it."""
    return copy.scales == 'channel'

def region_bytes(experts=SHAPE[0], hidden=SHAPE[1], inter=SHAPE[2]):
    return experts * offsets(hidden, inter)[3]

def make_copy(region, experts=SHAPE[0], hidden=SHAPE[1], inter=SHAPE[2]):
    eb = offsets(hidden, inter)[3]
    if region.dtype != torch.uint8 or region.dim() != 1 or region.numel() < experts * eb or region.data_ptr() % 16:
        raise ValueError(f'i8x: the copy needs {experts * eb} bytes of 16-byte-aligned uint8, got {tuple(region.shape)} {region.dtype}')
    blocks = region[:experts * eb].view(experts, eb)
    table = torch.full((experts + 1,), -1, dtype=torch.int32, device=region.device)
    table[:experts] = torch.arange(experts, dtype=torch.int32, device=region.device)
    return Copy(blocks, table, hidden, inter)

def block_parts(block, hidden=SHAPE[1], inter=SHAPE[2]):
    q2, c13, c2, eb = offsets(hidden, inter)
    return (block[:q2].view(torch.int8).view(2 * inter, hidden), block[q2:c13].view(torch.int8).view(hidden, inter), block[c13:c2].view(torch.float16).view(2 * inter, hidden // GROUP), block[c2:eb].view(torch.float16).view(hidden, inter // GROUP))

@triton.jit
def _quant_row_kernel(w_ptr, q_ptr, c_ptr, K: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, K)
    w = tl.load(w_ptr + row.to(tl.int64) * K + offs).to(tl.float32)
    amax = tl.max(tl.abs(w), axis=0)
    s16 = (tl.div_rn(amax, 127.0) * 1.0009765625).to(tl.float16)
    s = s16.to(tl.float32)
    v = tl.div_rn(w, tl.where(s > 0.0, s, 1.0))
    r = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(0.5 - v))
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(q_ptr + row.to(tl.int64) * K + offs, r.to(tl.int8))
    offs_g = tl.arange(0, K // GROUP)
    tl.store(c_ptr + row.to(tl.int64) * (K // GROUP) + offs_g, s16 + tl.zeros((K // GROUP,), dtype=tl.float16))

def quantize_channel(w, q, c):
    n, k = w.shape
    if not w.is_contiguous() or k % GROUP or tuple(q.shape) != (n, k) or (tuple(c.shape) != (n, k // GROUP)) or (q.dtype != torch.int8) or (c.dtype != torch.float16) or (not c.is_contiguous()):
        raise ValueError('i8x: quantize_channel needs contiguous w [N, K], q [N, K] int8, c [N, K/128] fp16')
    _quant_row_kernel[n,](w, q, c, K=k, GROUP=GROUP, num_warps=4)

def fill_expert(block, w13_e, w2_e, scales=None):
    two_i, hidden = w13_e.shape
    q13, q2, c13, c2 = block_parts(block, hidden, two_i // 2)
    for w, q, c in ((w13_e, q13, c13), (w2_e, q2, c2)):
        if (scales or SCALES) == 'channel':
            quantize_channel(w, q, c)
        else:
            _, s = quantize(w, q)
            c.copy_(s.t())

def fill_layer(copy, w13, w2):
    if tuple(w13.shape) != (copy.blocks.shape[0], 2 * copy.inter, copy.hidden) or tuple(w2.shape) != (copy.blocks.shape[0], copy.hidden, copy.inter):
        raise ValueError(f'i8x: weights {tuple(w13.shape)} / {tuple(w2.shape)} do not match the copy')
    for e in range(copy.blocks.shape[0]):
        fill_expert(copy.blocks[e], w13[e], w2[e], copy.scales)
    return scale_census(copy)

def scale_census(copy):
    _, c13, _, eb = offsets(copy.hidden, copy.inter)
    s = copy.blocks[:, c13:eb].reshape(-1).view(torch.float16).float().abs()
    zero = int((s == 0).sum())
    return (int((s < FP16_MIN_NORMAL).sum()) - zero, zero)

def dequantize_block(block, hidden=SHAPE[1], inter=SHAPE[2]):
    q13, q2, c13, c2 = block_parts(block, hidden, inter)
    return (dequantize(q13, c13.t()), dequantize(q2, c2.t()))

def check_kernel(kernel):
    if getattr(kernel, 'HOT8', 0) != 1 or getattr(kernel, 'Q8_EXPERT', None) != EXPERT_BYTES or getattr(kernel, 'HOT8_MAX_N', 0) < getattr(kernel, 'MAX_T', 1 << 30):
        raise ValueError(f'qk_moe_decode is not the HOT8 build with {EXPERT_BYTES}-byte blocks and QK_HOT8_MAX_N >= MAX_T: a relocated layer would stream some experts from host memory')

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def _entry(rows):
    for max_rows, kind in DECODE:
        if rows <= max_rows:
            return kind
    return 'hot8'

def decode_kind(rows=None):
    if _active['fallback']:
        return _active['fallback']
    if rows is None:
        return ','.join((f'{kind}<={max_rows}' for max_rows, kind in DECODE))
    return 'tri8' if _entry(rows) in TRI8_VARIANTS else 'hot8'

def _uses_tri8():
    return any((kind in TRI8_VARIANTS for _, kind in DECODE))

def warm(copy, router_w, gate_w, s13, s2):
    warm_prefill_king(copy, router_w, gate_w, s13, s2)
    if not _uses_tri8() or _active['warmed'] or _active['fallback']:
        return None
    try:
        low = 1
        for max_rows, kind in DECODE:
            if kind in TRI8_VARIANTS:
                x = torch.zeros(low, copy.hidden, dtype=torch.bfloat16, device=copy.blocks.device)
                positions = torch.ones(low, dtype=torch.int64, device=copy.blocks.device)
                row_batch(copy, x, positions, router_w, gate_w, s13, s2)
            low = max_rows + 1
        _sync(copy.blocks.device)
    except Exception as exc:
        _active['fallback'], _active['warm_error'] = ('hot8', exc)
        return exc
    _active['warmed'] = True
    return None

def row_batch(copy, x, positions, router_w, gate_w, s13, s2, norm=None):
    return i8x_decode.moe_row_batch(x, positions, router_w, gate_w, copy.blocks, offsets(copy.hidden, copy.inter), s13, s2, g128=g128(copy), norm=norm, **TRI8_VARIANTS.get(_entry(x.shape[0]), {}))

def row_batch_args(copy):
    return (copy.table, copy.blocks, True)

# the prefill MoE down stores its routed outputs y as int8 (per row x 256-column scale; a PRECISION CHANGE: ~0.55 % of
# the MoE output on real rows) on these layers (all)
Y8_MIN_LAYER = 0
Y8_SKIP_LAYERS = frozenset()

def y8_layer(layer_id):
    return isinstance(layer_id, int) and layer_id >= Y8_MIN_LAYER and layer_id not in Y8_SKIP_LAYERS

def port_paths(shape, layer_id=None):
    """(up, dn8) for a chunk of this shape class / layer: the parent's ('up8', False) unless a port switch picks ours."""
    if _prefill_state.get('port') is not None:
        return 'up8', False
    up = 'lanep' if layer_id in LANEP_LAYERS else UP_PATH.get(shape, 'up8')
    return up, bool(DOWN_W8A8.get(shape, False))

def prefill(x, router_w, copy, topk, shared, gate_w=None, s13=None, s2=None, layer_id=None, shape=None, pre=None):
    if prefill_kind() == 'king_i8' and gate_w is not None:
        if (x.shape[0] >= UP8_MIN and _prefill_state['up8'] is None and copy.w2 is not None and channel(copy)
                and PREFILL_DOWN == 'held'):
            up, dn8 = port_paths(shape, layer_id)
            return prefill_up8(x, router_w, gate_w, copy, s13, s2, y8=y8_layer(layer_id), up=up, dn8=dn8, pre=pre)
        return prefill_king_i8(x, router_w, gate_w, copy, s13, s2)
    return i8x_prefill.moe_prefill(x, router_w, copy.blocks, copy.hidden, copy.inter, offsets(copy.hidden, copy.inter), topk, shared, g128=g128(copy))
KING_I8_SHAPE = (256, 2048, 512)
HELD_VERIFY = True

def held_intact(copy, chunk=8):
    for e0 in range(0, copy.w2.shape[0], chunk):
        if not torch.equal(copy.w2[e0:e0 + chunk].view(torch.int16), copy.w2_host[e0:e0 + chunk].view(torch.int16)):
            return False
    return True

def prefill_kind():
    return _prefill_state['fallback'] or PREFILL

def prefill_label():
    if prefill_kind() == 'king_i8':
        return f'king_i8(up=q8,down={PREFILL_DOWN})'
    return f'triton({i8x_prefill.KIND})'

def _prefill_ext():
    import qk_moe_prefill_i8
    return qk_moe_prefill_i8

def _prefill_counters(device, n):
    cnt = _prefill_state['counters'].get(device)
    if cnt is None:
        cnt = _prefill_state['counters'][device] = torch.zeros(n, dtype=torch.int32, device=device)
    return cnt

def prefill_king_i8(x, router_w, gate_w, copy, s13, s2, down=None, fold=None, parts=7, diag=0):
    ext = _prefill_ext()
    down_q8 = (down or PREFILL_DOWN) == 'q8'
    if not down_q8 and copy.w2 is None:
        raise ValueError("i8x: the 'held' down needs the layer's held BF16 w2 (Copy.w2)")
    empty = x.new_empty(0)
    logits = lt_linear(x, router_w)  # kb24: F.linear's bits (lt.py verdict), pinned plan per chunk width
    cnt = _prefill_counters(x.device, ext.COUNTERS)
    mode = FOLD_MODES[fold or PREFILL_FOLD] | diag
    return ext.forward_i8(x, logits, gate_w, copy.blocks, empty, empty if down_q8 else copy.w2, s13, s2, cnt, 1, int(down_q8), parts, mode)[0]

def prefill_up8(x, router_w, gate_w, copy, s13, s2, native=True, y8=False, up='up8', dn8=False, pre=None):
    """The king's i8 prefill with its up GEMM on INT8 tensor cores (routing and the held-BF16 down unchanged).
    Port: up='lanep' runs our W8A8 up instead of up_i8g, dn8 our W8A8 routed down instead of the held-BF16 one.
    kb23n: pre = add_norm_route's (sg, xq, xs, xm) of this x, or None."""
    ext = _prefill_ext()
    logits = lt_linear(x, router_w)  # kb24: F.linear's bits (lt.py verdict), pinned plan per chunk width
    cnt = _prefill_counters(x.device, ext.COUNTERS)
    fused = up == 'up8' and not dn8 and UP_NATIVE and UP_NATIVE_OK and native and UP_FUSED_QUANT
    if fused and pre is not None:
        _, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, xq, xs, xm, sq, ss = ext.route_i8q_pre(x, logits, cnt, s13, *pre)
        branch_evidence(ROUTE_PRE_MARKER, x.shape[0])
    elif fused:
        _, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, xq, xs, xm, sq, ss = ext.route_i8q(x, logits, gate_w, cnt, s13, KNOB_QM)
    elif up == 'up8' and not dn8:
        _, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok = ext.route_i8(x, logits, gate_w, cnt)
    else:
        _, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, offs = ext.route_i8o(x, logits, gate_w, cnt)
    if up == 'lanep':
        h = ext.up_w8a8(x, copy.blocks, s13, mt, pairs, n_pairs, sorted_tok, offs, PREFILL_UP_PP)
        branch_evidence(W8A8_MARKER, x.shape[0])
    elif fused:
        up_kernel = ext.up_i8gp if x.shape[0] >= UP_WS_MIN else ext.up_i8g
        h = up_kernel(x, xq, xs, xm, copy.blocks, sq, ss, mt, sorted_tok, KNOB_TANH)
    elif UP_NATIVE and UP_NATIVE_OK and native:
        xq, xs, xm = i8x_up8.quant_img(x)
        n, k = s13.shape
        sq = torch.empty((n, k), dtype=torch.int8, device=x.device)
        sc = torch.empty((n, k // GROUP), dtype=torch.float16, device=x.device)
        quantize_channel(s13, sq, sc)
        h = ext.up_i8g(x, xq, xs, xm, copy.blocks, sq, sc[:, 0].float(), mt, sorted_tok, KNOB_TANH)
    else:
        h = i8x_up8.up(x, copy.blocks, s13, offsets(copy.hidden, copy.inter), mt, sorted_tok)
    if dn8:
        branch_evidence(W8A8_DN_MARKER, x.shape[0])
        return ext.down_w8a8(x, copy.blocks, copy.w2, s2, cnt, h, offs, topk_w, pos_tk, sg, n_pairs, mt, pairs,
                             sorted_tok, y8)
    return ext.down_i8(x, copy.w2, s2, cnt, h, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, y8, KNOB_YDIV)

def route_pre_ready(copy, rows, shape=None, layer_id=None):
    """kb23n: True when prefill() routes a chunk of `rows` rows of this layer through route_i8q (prefill_up8's fused
    path), so its post-attention norm may run add_norm_route."""
    if (copy is None or prefill_kind() != 'king_i8' or rows < UP8_MIN or _prefill_state['up8'] is not None
            or copy.w2 is None or not channel(copy) or PREFILL_DOWN != 'held'):
        return False
    up, dn8 = port_paths(shape, layer_id)
    return up == 'up8' and not dn8 and UP_NATIVE and UP_NATIVE_OK and UP_FUSED_QUANT

def add_norm_route(x, residual, weight, eps, gate_w):
    """kb23n: flashinfer's Gemma fused add-RMSNorm of (x, residual) in place (x <- normed, residual <- x + residual), bit
    for bit, plus route_i8q's (sg, xq, xs, xm) of the normed rows."""
    return tuple(_prefill_ext().add_norm_route(x, residual, weight, eps, gate_w, KNOB_QM))

def _warm_port(x, router_w, gate_w, copy, s13, s2, ref):
    """Warm the port paths the switches select (each shape class) against king_i8; a failure turns them all off."""
    st = _prefill_state
    combos = sorted(({port_paths(c) for c in UP_PATH} | {port_paths(c, l) for c in UP_PATH for l in LANEP_LAYERS})
                    - {('up8', False)})
    if not combos:
        return
    try:
        for up, dn8 in combos:
            for y8 in (False, True):
                for _ in range(2):
                    out = prefill_up8(x, router_w, gate_w, copy, s13, s2, True, y8, up, dn8)
                _sync(x.device)
                if not bool(torch.isfinite(out).all()):
                    raise RuntimeError(f'i8x: the port prefill ({up}, dn8={dn8}, y8={y8}) returned non-finite values')
                err = (out.float() - ref.float()).norm(dim=1) / ref.float().norm(dim=1).clamp_min(1e-12)
                if float(err.median()) > PORT_WARM_MAX_P50:
                    raise RuntimeError(f'i8x: the port prefill ({up}, dn8={dn8}, y8={y8}) is {float(err.median()):.4f} '
                                       'from king_i8 at the warm-up (median row)')
    except Exception as exc:  # the parent's up8 path then serves every chunk
        st['port'] = exc
        line = f'CACHEON-I8X: ERROR stage=warm_port error={exc!r} result=up8 pid={os.getpid()}'
        print(line, flush=True)
        print(line, file=sys.stderr, flush=True)

def warm_prefill_king(copy, router_w, gate_w, s13, s2):
    st = _prefill_state
    if PREFILL != 'king_i8' or st['warmed'] or st['fallback'] or ((copy.blocks.shape[0], copy.hidden, copy.inter) != KING_I8_SHAPE):
        return None
    try:
        device = copy.blocks.device
        gen = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(64, copy.hidden, generator=gen, device=device).to(torch.bfloat16)
        for _ in range(2):
            out = prefill_king_i8(x, router_w, gate_w, copy, s13, s2)
        _sync(device)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError('i8x: the king_i8 prefill warm-up returned non-finite values')
        try:
            for _ in range(2):
                out8 = prefill_up8(x, router_w, gate_w, copy, s13, s2, False)
            _sync(device)
            if not bool(torch.isfinite(out8).all()):
                raise RuntimeError('i8x: the up8 prefill warm-up returned non-finite values')
            if UP_NATIVE:
                global UP_NATIVE_OK
                try:
                    for y8 in (False, True):
                        outn = prefill_up8(x, router_w, gate_w, copy, s13, s2, True, y8)
                    _sync(device)
                    if not bool(torch.isfinite(outn).all()):
                        raise RuntimeError('i8x: the native up warm-up returned non-finite values')
                except Exception as exc:  # i8x_up8.py's Triton up then serves every chunk
                    UP_NATIVE_OK = False
                    line = f'CACHEON-I8X: ERROR stage=warm_up_native error={exc!r} result=up8 pid={os.getpid()}'
                    print(line, flush=True)
                    print(line, file=sys.stderr, flush=True)
            _warm_port(x, router_w, gate_w, copy, s13, s2, out)
        except Exception as exc:  # the up8 path is then skipped; the king_i8 prefill serves every chunk
            st['up8'] = exc
            line = f'CACHEON-I8X: ERROR stage=warm_up8 error={exc!r} result=king_i8 pid={os.getpid()}'
            print(line, flush=True)
            print(line, file=sys.stderr, flush=True)
        if HELD_VERIFY and PREFILL_DOWN == 'held' and (copy.w2_host is not None) and (not held_intact(copy)):
            raise RuntimeError("i8x: the held BF16 w2 no longer equals stock's values in host memory")
    except Exception as exc:
        st['fallback'], st['error'] = ('triton', exc)
        line = f'CACHEON-I8X: ERROR stage=warm_prefill error={exc!r} result=triton pid={os.getpid()}'
        print(line, flush=True)
        print(line, file=sys.stderr, flush=True)
        return exc
    st['warmed'] = True
    return None

def _topk_torch(x, logits):
    w, ids = torch.topk(torch.softmax(logits.float(), dim=-1), 8, dim=-1)
    return (w / w.sum(dim=-1, keepdim=True), ids.to(torch.int32))

def warm_tokens():
    bounds = {t[0] for t in i8x_prefill.SWAP_TILES + i8x_prefill.TILES if t[0] < 1 << 20}
    return sorted({16} | bounds | {b + 1 for b in bounds})

def warm_prefill(device, experts=8, hidden=SHAPE[1], inter=SHAPE[2], align=None):
    eb = offsets(hidden, inter)[3]
    blocks = torch.zeros(experts, eb, dtype=torch.uint8, device=device)
    router_w = torch.zeros(experts, hidden, dtype=torch.bfloat16, device=device)
    errors = []
    for kind in dict.fromkeys((i8x_prefill.KIND, 'tri')):
        try:
            for tokens in warm_tokens():
                x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device=device)
                i8x_prefill.moe_prefill(x, router_w, blocks, hidden, inter, offsets(hidden, inter), _topk_torch, torch.zeros_like, align=align, kind=kind, g128=SCALES != 'channel')
            _sync(device)
        except Exception as exc:
            errors.append(f'{kind}: {exc!r}'[:400])
            continue
        i8x_prefill.KIND = kind
        return (kind, errors)
    raise RuntimeError('i8x: no prefill kernel compiled: ' + ' | '.join(errors))
