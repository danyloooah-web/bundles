from __future__ import annotations
import os
import sys
import time
import torch
from qwen36_layer import i8x_kernels
MARKER = 'king_i8x'
ENABLED = True
VIEW = 'cai'
VERIFY = True
VERIFY_CHUNK = 8
LAYERS = 40
_entries = []
_state = {'closed': None, 'seen_capture': False, 'preflight': False, 'logged': False, 'prefill': None}
_totals = {'layers': 0, 'moved': 0, 'host': 0, 'copy': 0, 'held': 0, 'ms': 0.0, 'before': None, 'subnormal': 0, 'zero': 0}

class Entry:
    __slots__ = ('layer', 'w13', 'w2', 'state', 'reason', 'copy', 'host', 'held')

    def __init__(self, layer, w13, w2):
        self.layer, self.w13, self.w2 = (layer, w13, w2)
        self.state, self.reason = ('bf16', None)
        self.copy = self.host = self.held = None

def _capturing():
    return torch.cuda.is_current_stream_capturing()

def _is_cuda(t):
    return t.is_cuda

def _mem(device):
    free, _ = torch.cuda.mem_get_info(device)
    return (int(free), int(torch.cuda.memory_reserved(device)))

def _sync(device):
    torch.cuda.synchronize(device)

def _pin(shape):
    return torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)

def _kernel():
    import qk_moe_decode
    i8x_kernels.check_kernel(qk_moe_decode)

class _HostSpan:

    def __init__(self, host):
        self.host = host
        self.__cuda_array_interface__ = {'shape': tuple(host.shape), 'typestr': '<i2', 'data': (int(host.data_ptr()), False), 'version': 3, 'strides': None}

def host_view(host, device, method=None):
    method = method or VIEW
    if host.dtype != torch.bfloat16 or not host.is_contiguous():
        raise ValueError('i8x: host_view needs a contiguous bf16 host tensor')
    if method == 'cai':
        view = torch.as_tensor(_HostSpan(host)).view(torch.bfloat16)
    elif method == 'storage':
        storage = torch._C._construct_storage_from_data_pointer(int(host.data_ptr()), device, host.numel() * host.element_size())
        view = torch.empty(0, dtype=torch.bfloat16, device=device).set_(storage, 0, tuple(host.shape))
    else:
        raise ValueError(f'i8x: unknown VIEW {method!r}')
    if view.data_ptr() != host.data_ptr() or view.device != device or tuple(view.shape) != tuple(host.shape) or (view.dtype != torch.bfloat16) or (not view.is_contiguous()):
        raise RuntimeError(f'i8x: the {method} view is not the host buffer on {device}: {view.device} {hex(view.data_ptr())} vs host {hex(host.data_ptr())}')
    return view

def _same(view, dev):
    for e0 in range(0, dev.shape[0], VERIFY_CHUNK):
        if not torch.equal(view[e0:e0 + VERIFY_CHUNK].view(torch.int16), dev[e0:e0 + VERIFY_CHUNK].view(torch.int16)):
            return False
    return True

def _quantize_rows(w):
    n, k = w.shape
    if i8x_kernels.SCALES == 'channel':
        q = torch.empty((n, k), dtype=torch.int8, device=w.device)
        c = torch.empty((n, k // 128), dtype=torch.float16, device=w.device)
        i8x_kernels.quantize_channel(w, q, c)
        return (q, c)
    q, s = i8x_kernels.quantize(w)
    return (q, s.t())

def _preflight(device):
    _, hidden, inter = i8x_kernels.SHAPE
    patterns = _pin(((1 << 16) // hidden, hidden))
    patterns.view(torch.int16).copy_((torch.arange(1 << 16, dtype=torch.int32) - (1 << 15)).to(torch.int16).view(patterns.shape))
    if not torch.equal(host_view(patterns, device).view(torch.int16), patterns.to(device).view(torch.int16)):
        raise RuntimeError('i8x: a host view does not read back its buffer')
    gen = torch.Generator().manual_seed(0)
    for k in (hidden, inter):
        host = _pin((16, k))
        host.copy_(torch.randn(16, k, generator=gen).to(torch.bfloat16))
        q_view, s_view = _quantize_rows(host_view(host, device))
        q_dev, s_dev = _quantize_rows(host.to(device))
        if not (torch.equal(q_view, q_dev) and torch.equal(s_view.view(torch.int16), s_dev.view(torch.int16))):
            raise RuntimeError('i8x: the quantizer reads a host view differently from device memory')
    _state['prefill'], errors = _warm_prefill(device)
    for err in errors:
        _log(f'stage=warm_prefill fallback error={err}')

def _warm_prefill(device):
    return i8x_kernels.warm_prefill(device)

def _log(line):
    print(f'CACHEON-I8X: {line} pid={os.getpid()}', flush=True)

def _log_error(stage, exc, entry):
    line = f'CACHEON-I8X: ERROR stage={stage} layer={entry.layer} error={exc!r} result={entry.state} pid={os.getpid()}'
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)

def log_warm_failure(exc):
    line = f'CACHEON-I8X: ERROR stage=warm_decode error={exc!r} result=hot8 pid={os.getpid()}'
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)

def summary(device=None):
    out = {'layers': _totals['layers'], 'seen': len(_entries), 'bytes_moved': _totals['moved'], 'host_bytes': _totals['host'], 'copy_bytes': _totals['copy'], 'held_idle': _totals['held'], 'ms': round(_totals['ms'], 1), 'view': VIEW, 'closed': _state['closed'], 'scales': i8x_kernels.SCALES, 'decode': i8x_kernels.decode_kind(), 'prefill': _state['prefill'], 'prefill_path': i8x_kernels.prefill_label(), 'subnormal_scales': _totals['subnormal'], 'zero_scales': _totals['zero']}
    before = _totals['before']
    if before is not None:
        out['free_before'], out['reserved_before'] = before
        if device is not None:
            out['free_after'], out['reserved_after'] = _mem(device)
    return out

def _settle(device, force=False):
    if _state['logged'] or not (force or len(_entries) >= LAYERS):
        return
    _state['logged'] = True
    s = summary(device if _totals['before'] is not None else None)
    result = 'off' if _state['closed'] == 'disabled' else 'closed' if _state['closed'] else 'done'
    fields = [f'result={result}', f"reason={_state['closed']}"] if _state['closed'] else [f'result={result}']
    fields += [f"layers={s['layers']}/{s['seen']}"] + [f'{k}={s[k]}' for k in ('bytes_moved', 'host_bytes', 'copy_bytes', 'held_idle', 'free_before', 'free_after', 'reserved_before', 'reserved_after', 'ms', 'view', 'scales', 'decode', 'prefill', 'prefill_path', 'subnormal_scales', 'zero_scales') if k in s]
    _log('stage=relocate ' + ' '.join(fields))

def _close(reason, device):
    if _state['closed'] is None:
        _state['closed'] = reason
    _settle(device, force=True)

def _fail(entry, stage, exc, device):
    if entry.state != 'host_bf16':
        entry.state = 'bf16'
    entry.reason = f'{stage}_failed'
    _log_error(stage, exc, entry)
    _close(entry.reason, device)
    return entry

def _refusal(w13, w2, servable):
    if not ENABLED:
        return 'disabled'
    if _state['closed'] is not None:
        return _state['closed']
    if _capturing():
        _state['seen_capture'] = True
        return 'capture'
    if _state['seen_capture']:
        return 'after_capture'
    if not servable:
        return 'stock_api'
    experts, hidden, inter = i8x_kernels.SHAPE
    for t, shape in ((w13, (experts, 2 * inter, hidden)), (w2, (experts, hidden, inter))):
        if not isinstance(t, torch.nn.Parameter) or tuple(t.shape) != shape or t.dtype != torch.bfloat16 or (not _is_cuda(t)) or (not t.is_contiguous()) or (t.device != w13.device):
            return 'not_served_weights'
    return None

def relocate(layer, w13, w2, servable=True):
    for entry in _entries:
        if entry.w13 is w13 and entry.w2 is w2:
            return entry
    entry = Entry(layer, w13, w2)
    _entries.append(entry)
    device = w13.device
    reason = _refusal(w13, w2, servable)
    if reason is None:
        with torch.inference_mode(False), torch.no_grad():
            try:
                _kernel()
            except Exception as exc:
                return _fail(entry, 'kernel', exc, device)
            if not _state['preflight']:
                try:
                    _preflight(device)
                except Exception as exc:
                    return _fail(entry, 'preflight', exc, device)
                _state['preflight'] = True
            _move(entry, device)
    else:
        entry.reason = reason
        _close(reason, device)
    _settle(device)
    return entry

def _move(entry, device):
    w13, w2 = (entry.w13, entry.w2)
    t0 = time.perf_counter()
    if _totals['before'] is None:
        _totals['before'] = _mem(device)
    stage = 'pin'
    try:
        h13, h2 = (_pin(tuple(w13.shape)), _pin(tuple(w2.shape)))
        stage = 'd2h'
        h13.copy_(w13.detach())
        h2.copy_(w2.detach())
        stage = 'view'
        v13, v2 = (host_view(h13, device), host_view(h2, device))
        stage = 'verify'
        if VERIFY and (not (_same(v13, w13) and _same(v2, w2))):
            raise RuntimeError("the host view does not read back stock's device bytes")
    except Exception as exc:
        return _fail(entry, stage, exc, device)
    old13, old2 = (w13.data, w2.data)
    w13.data, w2.data = (v13, v2)
    stage = 'copy'
    try:
        copy = i8x_kernels.make_copy(old13.view(torch.uint8).reshape(-1), w13.shape[0], w13.shape[2], w2.shape[2])
        stage = 'fill'
        census = i8x_kernels.fill_layer(copy, w13, w2)
        _sync(device)
    except Exception as exc:
        try:
            old13.copy_(h13)
            _sync(device)
            w13.data, w2.data = (old13, old2)
        except Exception as again:
            entry.state = 'host_bf16'
            _log_error('restore', again, entry)
        return _fail(entry, stage, exc, device)
    entry.state, entry.copy, entry.host, entry.held = ('i8x', copy, (h13, h2), old2)
    copy.w2, copy.w2_host = (old2, v2)
    moved = w13.numel() * 2 + w2.numel() * 2
    copy_bytes = copy.blocks.numel()
    _totals['layers'] += 1
    _totals['moved'] += moved
    _totals['host'] += h13.numel() * 2 + h2.numel() * 2
    _totals['copy'] += copy_bytes
    _totals['held'] += moved - copy_bytes
    subnormal, zero = census or (0, 0)
    _totals['subnormal'] += subnormal
    _totals['zero'] += zero
    _totals['ms'] += (time.perf_counter() - t0) * 1000.0
    return entry

def serve(entry):
    if not _state['seen_capture'] and _capturing():
        _state['seen_capture'] = True
    return entry.copy if entry is not None and entry.state == 'i8x' else None

def stock_only(entry):
    return entry is not None and entry.state == 'host_bf16'
