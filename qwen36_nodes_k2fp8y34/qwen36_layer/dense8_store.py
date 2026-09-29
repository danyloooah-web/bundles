import os
import sys
import time
import torch
from qwen36_layer.dense8 import GROUP, LANE_DENSE, copy_bytes, quantize
from qwen36_layer import i8x_reloc, i8x_tail
MARKER = 'king_dense8'
# Lane dense: the attention qkv_proj and every decode o_proj / out_proj join the GDN in_proj (INT8 copies in the i8x
# tails, candidate-owned; the served BF16 weights are only read).
KINDS = ('inproj', 'qkv', 'oproj') if LANE_DENSE else ('inproj',)
SHAPES = {'inproj': (12352, 2048), 'oproj': (2048, 4096), 'qkv': (9216, 2048)}
MAX_ROWS_INT8 = 32
MAX_ROWS = {'inproj': MAX_ROWS_INT8, 'qkv': 32, 'oproj': 32}
_entries = {kind: [] for kind in KINDS}
_state = {kind: 'pending' for kind in KINDS}
_flags = {'built': False, 'seen_capture': False, 'captured_int8': False, 'prefill_seen': False}
_vmm = {}
_driver = {'mod': None}

class Entry:
    __slots__ = ('kind', 'sources', 'n', 'k', 'key', 'q', 's')

    def __init__(self, kind, sources):
        self.kind = kind
        self.sources = tuple(sources)
        self.n = sum((int(t.shape[0]) for t in self.sources))
        self.k = int(self.sources[0].shape[1])
        self.key = None
        self.q = self.s = None

def _source_key(entry):
    return tuple(((t.data_ptr(), tuple(t.shape)) for t in entry.sources))

def _is_cuda(t):
    return t.is_cuda

def register(kind, sources):
    sources = tuple(sources)
    if kind not in KINDS or kind not in SHAPES or (not sources) or (not all((isinstance(t, torch.Tensor) for t in sources))):
        return None
    k = sources[0].shape[1] if sources[0].dim() == 2 else -1
    for t in sources:
        if t.dim() != 2 or t.dtype != torch.bfloat16 or (not _is_cuda(t)) or (not t.is_contiguous()) or (t.shape[1] != k) or (t.device != sources[0].device):
            return None
    if (sum((int(t.shape[0]) for t in sources)), k) != SHAPES[kind] or k % GROUP:
        return None
    key = tuple(((t.data_ptr(), tuple(t.shape)) for t in sources))
    for entry in _entries[kind]:
        if _source_key(entry) == key:
            return entry
    entry = Entry(kind, sources)
    _entries[kind].append(entry)
    return entry

def _log(kind, **fields):
    print('CACHEON-DENSE8: ' + ' '.join([f'kind={kind}'] + [f'{k}={v}' for k, v in fields.items()]) + f' pid={os.getpid()}', flush=True)

def _log_latemap(kind, **fields):
    print('CACHEON-LATEMAP: ' + ' '.join([f'kind={kind}'] + [f'{k}={v}' for k, v in fields.items()]) + f' pid={os.getpid()}', flush=True)

def _log_error(kind, stage, exc, **fields):
    line = 'CACHEON-LATEMAP: ERROR ' + ' '.join([f'kind={kind}', f'stage={stage}'] + [f'{k}={v}' for k, v in fields.items()]) + f' error={exc!r} pid={os.getpid()}'
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)

def _fill(entry):
    r0 = 0
    for t in entry.sources:
        n = int(t.shape[0])
        quantize(t, entry.q[r0:r0 + n], entry.s[:, r0:r0 + n])
        r0 += n
    entry.key = _source_key(entry)

def _cu():
    if _driver['mod'] is None:
        try:
            from cuda.bindings import driver
        except ImportError:
            from cuda import cuda as driver
        _driver['mod'] = driver
    return _driver['mod']

class DriverError(RuntimeError):

    def __init__(self, what, code):
        super().__init__(f"{what}: {getattr(code, 'name', code)} ({int(code)})")
        self.code = code

def _ok(result, what):
    cu = _cu()
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise DriverError(what, err)
    return rest[0] if len(rest) == 1 else None

def _is_oom(exc):
    return isinstance(exc, DriverError) and exc.code == _cu().CUresult.CUDA_ERROR_OUT_OF_MEMORY

class _Range:

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {'shape': (nbytes,), 'typestr': '|i1', 'data': (int(ptr), False), 'version': 3, 'strides': None}

def _view(va, nbytes, device):
    return torch.as_tensor(_Range(va, nbytes), device=device)

def _prop_and_access(index):
    cu = _cu()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = index
    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = index
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return (prop, access)

def _device_index(device):
    return device.index if device.index is not None else torch.cuda.current_device()

def _map_alias(rec):
    cu = _cu()
    va, gran = (rec['va'], rec['gran'])
    for i in range(rec['size'] // gran):
        _ok(cu.cuMemMap(va + i * gran, gran, 0, rec['alias'], 0), 'cuMemMap(alias)')
        rec['slots'] = i + 1
    _ok(cu.cuMemSetAccess(va, rec['size'], [rec['access']], 1), 'cuMemSetAccess(alias)')

def _undo_reserve(rec):
    cu = _cu()
    calls = []
    if rec.get('slots'):
        calls.append(lambda: cu.cuMemUnmap(rec['va'], rec['slots'] * rec['gran']))
    if rec.get('alias') is not None:
        calls.append(lambda: cu.cuMemRelease(rec['alias']))
    if rec.get('va'):
        calls.append(lambda: cu.cuMemAddressFree(rec['va'], rec['size']))
    for call in calls:
        try:
            call()
        except Exception:
            pass

def _reserve(device):
    _flags['built'] = True
    for kind in KINDS:
        entries = _entries[kind]
        if not entries:
            _state[kind] = 'none'
            continue
        q_bytes = sum((e.n * e.k for e in entries))
        s_bytes = sum((e.k // GROUP * e.n * 2 for e in entries))
        nbytes = sum((copy_bytes(e.n, e.k) for e in entries))
        free_before = torch.cuda.mem_get_info(device)[0]
        rec = {'va': 0, 'size': 0, 'gran': 0, 'slots': 0, 'alias': None, 'real': None, 'device': device}
        stage = 'import'
        try:
            cu = _cu()
            stage = 'init'
            _ok(cu.cuInit(0), 'cuInit')
            rec['prop'], rec['access'] = _prop_and_access(_device_index(device))
            stage = 'granularity'
            gran = int(_ok(cu.cuMemGetAllocationGranularity(rec['prop'], cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM), 'cuMemGetAllocationGranularity'))
            rec['gran'], rec['size'] = (gran, -(-nbytes // gran) * gran)
            stage = 'reserve'
            rec['va'] = int(_ok(cu.cuMemAddressReserve(rec['size'], 0, 0, 0), 'cuMemAddressReserve'))
            stage = 'alias_create'
            rec['alias'] = _ok(cu.cuMemCreate(gran, rec['prop'], 0), 'cuMemCreate(alias)')
            stage = 'alias_map'
            _map_alias(rec)
            stage = 'alias_zero'
            base = _view(rec['va'], rec['size'], device)
            base[:gran].zero_()
            torch.cuda.synchronize(device)
        except Exception as exc:
            _undo_reserve(rec)
            _state[kind] = 'skipped'
            _log_error(kind, stage, exc, layers=len(entries), bytes=nbytes, result='skipped')
            _log(kind, layers=len(entries), bytes=nbytes, free_before=free_before, result='skipped', reason=f'latemap_{stage}_failed')
            continue
        _vmm[kind] = rec
        s_all = base[q_bytes:q_bytes + s_bytes].view(torch.float16)
        qo = so = 0
        for e in entries:
            e.q = base[qo:qo + e.n * e.k].view(e.n, e.k)
            e.s = s_all[so:so + e.k // GROUP * e.n].view(e.k // GROUP, e.n)
            qo += e.n * e.k
            so += e.k // GROUP * e.n
        _state[kind] = 'aliased'
        _log(kind, layers=len(entries), bytes=nbytes, reserved=rec['size'], granularity=gran, free_before=free_before, free_after=torch.cuda.mem_get_info(device)[0], result='aliased')

def _build_tails(device):
    """Place every kind's INT8 copy in the idle tails behind the i8x expert copies (i8x_tail.py) and fill it now,
    before any graph that reads it is captured: no free memory is taken and no post-capture late map is needed."""
    _flags['built'] = True
    for kind in KINDS:
        entries = _entries[kind]
        if not entries:
            _state[kind] = 'none'
            continue
        nbytes = sum((copy_bytes(e.n, e.k) for e in entries))
        # one (q, s) pair per entry: a tail holds ~244 MB, a kind's stacked copies up to 771 MB
        placed = [(i8x_tail.tensor((e.n, e.k), torch.int8, device), i8x_tail.tensor((e.k // GROUP, e.n), torch.float16, device))
                  for e in entries]
        if any((q is None or s is None for q, s in placed)):
            _state[kind] = 'skipped'
            _log(kind, layers=len(entries), bytes=nbytes, result='skipped', reason='no_tail')
            continue
        with torch.no_grad():
            for e, (q, s) in zip(entries, placed):
                e.q, e.s = q, s
                _fill(e)
        torch.cuda.synchronize(device)
        _state[kind] = 'taken'
        _log(kind, layers=len(entries), bytes=nbytes, free=torch.cuda.mem_get_info(device)[0], result='tail')

def _create_real(rec):
    cu = _cu()
    try:
        return _ok(cu.cuMemCreate(rec['size'], rec['prop'], 0), 'cuMemCreate(real)')
    except DriverError as exc:
        if not _is_oom(exc):
            raise
    torch.cuda.empty_cache()
    return _ok(cu.cuMemCreate(rec['size'], rec['prop'], 0), 'cuMemCreate(real, after empty_cache)')

def _restore_alias(rec, real, unmapped, mapped):
    cu = _cu()
    steps = []
    if mapped:
        steps.append(lambda: _ok(cu.cuMemUnmap(rec['va'], rec['size']), 'cuMemUnmap(real)'))
    if real is not None:
        steps.append(lambda: _ok(cu.cuMemRelease(real), 'cuMemRelease(real)'))
    if unmapped:
        rec['slots'] = 0
        steps.append(lambda: _map_alias(rec))
    for step in steps:
        try:
            step()
        except Exception:
            pass

def _fail(kind, stage, exc, **fields):
    safe = not _flags['captured_int8']
    _log_error(kind, stage, exc, captured_int8=int(not safe), result='skipped' if safe else 'failed', **fields)
    if safe:
        _state[kind] = 'skipped'
        for e in _entries[kind]:
            e.q = e.s = None
        return
    _state[kind] = 'failed'
    raise RuntimeError(f'CACHEON-LATEMAP: {kind} late map failed at {stage} after CUDA graphs captured the INT8 path; replays would read unfilled weights, refusing to serve: {exc!r}')

def _late_map(kind, trigger):
    rec = _vmm[kind]
    cu = _cu()
    device = rec['device']
    entries = [e for e in _entries[kind] if e.q is not None]
    nbytes = sum((copy_bytes(e.n, e.k) for e in entries))
    free_before = torch.cuda.mem_get_info(device)[0]
    t0 = time.perf_counter()
    real, unmapped, mapped, stage = (None, False, False, 'create')
    try:
        real = _create_real(rec)
        stage = 'synchronize'
        torch.cuda.synchronize(device)
        stage = 'unmap_alias'
        _ok(cu.cuMemUnmap(rec['va'], rec['size']), 'cuMemUnmap(alias)')
        unmapped = True
        stage = 'map'
        _ok(cu.cuMemMap(rec['va'], rec['size'], 0, real, 0), 'cuMemMap(real)')
        mapped = True
        stage = 'access'
        _ok(cu.cuMemSetAccess(rec['va'], rec['size'], [rec['access']], 1), 'cuMemSetAccess(real)')
    except Exception as exc:
        _restore_alias(rec, real, unmapped, mapped)
        _fail(kind, stage, exc, bytes=nbytes, reserved=rec['size'], free_before=free_before)
        return
    rec['real'] = real
    try:
        _ok(cu.cuMemRelease(rec['alias']), 'cuMemRelease(alias)')
        rec['alias'] = None
    except Exception as exc:
        _log_error(kind, 'release_alias', exc, result='continuing')
    map_ms = (time.perf_counter() - t0) * 1000.0
    try:
        with torch.no_grad():
            for e in entries:
                _fill(e)
        torch.cuda.synchronize(device)
    except Exception as exc:
        _fail(kind, 'fill', exc, bytes=nbytes, reserved=rec['size'], free_before=free_before)
        return
    _state[kind] = 'taken'
    _log_latemap(kind, layers=len(entries), bytes=nbytes, reserved=rec['size'], free_before=free_before, free_after=torch.cuda.mem_get_info(device)[0], map_ms=round(map_ms, 3), ms=round((time.perf_counter() - t0) * 1000.0, 3), trigger=trigger, result='taken')

def late_map_armed():
    return any((_state[kind] in ('pending', 'aliased', 'failed') for kind in KINDS))

def late_map_hook(prefill):
    if not prefill or torch.cuda.is_current_stream_capturing():
        return False
    _flags['prefill_seen'] = True
    if not _flags['seen_capture']:
        return False
    done = False
    for kind in KINDS:
        if _state[kind] == 'failed':
            raise RuntimeError(f'CACHEON-LATEMAP: {kind} late map failed earlier; refusing to serve')
        if _state[kind] == 'aliased':
            _late_map(kind, 'prefill_logits')
            done = _state[kind] == 'taken' or done
    return done

def get(entry, x):
    if entry is None:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        _flags['seen_capture'] = True
    elif not _flags['built'] and isinstance(x, torch.Tensor) and (x.device == entry.sources[0].device):
        if i8x_tail.ready():
            _build_tails(x.device)
        elif i8x_reloc._state['closed'] is not None:
            _reserve(x.device)
    if not capturing and _state[entry.kind] == 'aliased' and _flags['prefill_seen'] and (not _flags['seen_capture']):
        _late_map(entry.kind, 'eager_verify')
    state = _state[entry.kind]
    if state not in ('aliased', 'taken') or entry.q is None:
        return None
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.bfloat16 or (not x.is_contiguous()) or (not 0 < x.shape[0] <= MAX_ROWS[entry.kind]) or (x.shape[1] != entry.k) or (x.device != entry.sources[0].device):
        return None
    if state == 'taken' and entry.key != _source_key(entry):
        if capturing:
            return None
        with torch.no_grad():
            _fill(entry)
    if capturing:
        _flags['captured_int8'] = True
    return (entry.q, entry.s)


def peek(entry, rows):
    """Lane dense: the INT8 rows ``get`` would hand a ``rows``-row call right now, else None (no build, no refill).
    The GDN verify kernel L2-prefetches them in place of the BF16 out_proj weight the projection no longer reads."""
    if entry is None or entry.q is None or _state[entry.kind] not in ('aliased', 'taken') or not 0 < rows <= MAX_ROWS[entry.kind]:
        return None
    if _state[entry.kind] == 'taken' and entry.key != _source_key(entry):
        return None
    return entry.q
