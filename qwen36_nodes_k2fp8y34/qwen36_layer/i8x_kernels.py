from __future__ import annotations
import os
import sys
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from qwen36_layer import i8x_decode, i8x_prefill, i8x_up8
from qwen36_layer.dense8 import GROUP, dequantize, quantize
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
PREFILL_FOLD = 'bf16'
# prefill chunks of at least UP8_MIN rows run the MoE up on INT8 tensor cores (i8x_up8.py, PRECISION CHANGE)
UP8_MIN = 1024  # k2fp8y34: y18 (small prefill chunks on king_i8)
# the prefill MoE up as qk_moe_prefill_i8.cu up_i8g (native, the same per-group INT8 rows and per-channel INT8 copy as
# i8x_up8.py; the shared expert's gate/up rows quantized per channel per call: a PRECISION CHANGE for that expert)
UP_NATIVE = True
UP_NATIVE_OK = True
FOLD_MODES = {'exact': 0, 'bf16': 8}
_prefill_state = {'fallback': None, 'warmed': False, 'error': None, 'counters': {}, 'up8': None}

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

def prefill(x, router_w, copy, topk, shared, gate_w=None, s13=None, s2=None, layer_id=None):
    if prefill_kind() == 'king_i8' and gate_w is not None:
        if (x.shape[0] >= UP8_MIN and _prefill_state['up8'] is None and copy.w2 is not None and channel(copy)
                and PREFILL_DOWN == 'held'):
            return prefill_up8(x, router_w, gate_w, copy, s13, s2, y8=y8_layer(layer_id))
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
    logits = F.linear(x, router_w)
    cnt = _prefill_counters(x.device, ext.COUNTERS)
    mode = FOLD_MODES[fold or PREFILL_FOLD] | diag
    return ext.forward_i8(x, logits, gate_w, copy.blocks, empty, empty if down_q8 else copy.w2, s13, s2, cnt, 1, int(down_q8), parts, mode)[0]

# the prefill MoE's numerics: int8 y scale amax / KNOB_YDIV, IMG multiplier cap KNOB_QM (<= 64: the s32 bound),
# silu through tanh.approx (1) or __expf (0); chosen by full-lane verify counts (the MTP acceptance of each numerics)
KNOB_YDIV, KNOB_QM, KNOB_TANH = 127.0, 64.0, 1

def prefill_up8(x, router_w, gate_w, copy, s13, s2, native=True, y8=False):
    """The king's i8 prefill with its up GEMM on INT8 tensor cores (routing and the held-BF16 down unchanged)."""
    ext = _prefill_ext()
    logits = F.linear(x, router_w)
    cnt = _prefill_counters(x.device, ext.COUNTERS)
    if UP_NATIVE and UP_NATIVE_OK and native:
        # one topk launch also quantizes x (i8x_up8.quant_img's values) and the shared gate/up rows (quantize_channel's)
        _, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, xq, xs, xm, sq, ss = ext.route_i8q(x, logits, gate_w, cnt, s13, KNOB_QM)
        h = ext.up_i8g(x, xq, xs, xm, copy.blocks, sq, ss, mt, sorted_tok, KNOB_TANH)
    else:
        _, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok = ext.route_i8(x, logits, gate_w, cnt)
        h = i8x_up8.up(x, copy.blocks, s13, offsets(copy.hidden, copy.inter), mt, sorted_tok)
    return ext.down_i8(x, copy.w2, s2, cnt, h, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, y8, KNOB_YDIV)

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
