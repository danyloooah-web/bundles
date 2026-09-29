from __future__ import annotations
import os
import torch
from qwen36_layer import dense8_store
from qwen36_lmhead8 import lmhead8, w8a8
from qwen36_lmhead8.lmhead8 import quantize_rows, w8a16_linear
from qwen36_lmhead8.w8a8 import w8a8_linear
NODE = 'logits_processor'
MAX_ROWS = min(lmhead8.MAX_ROWS, w8a8.MAX_ROWS)
W8A8_MIN_ROWS = 64
W8A8_TWO_TERM = True

LOGITS_BUF_MIN_ROWS = 16
_SEEN = set()

def _evidence(branch, rows):
    try:
        captured = bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        captured = False
    rank = -1
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
    except Exception:
        pass
    key = (os.getpid(), rank, branch, captured)
    if key in _SEEN:
        return
    _SEEN.add(key)
    print(f'CACHEON-AUTHORED-BRANCH: {branch} pid={key[0]} rank={rank} node={NODE} rows={rows} captured={int(captured)}', flush=True)

def _plain_head(module, lm_head) -> bool:
    if getattr(module, 'use_fp32_lm_head', False) or getattr(module, 'rl_on_policy_target', None) is not None:
        return False
    if getattr(module, 'do_tensor_parallel_all_gather', False) or getattr(module, 'do_tensor_parallel_all_gather_dp_attn', False):
        return False
    if hasattr(lm_head, 'set_lora') and hasattr(lm_head, 'apply_lora'):
        return False
    weight = getattr(lm_head, 'weight', None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or (not weight.is_cuda):
        return False
    if weight.dtype != torch.bfloat16 or weight.stride(1) != 1:
        return False
    try:
        from sglang.srt.layers import logits_processor as stock_logits
        if stock_logits.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, 'quant_method', None)):
            return False
        amx = getattr(stock_logits, 'use_intel_amx_backend', None)
        if amx is not None and amx(lm_head):
            return False
    except Exception:
        return False
    return True

def _is_prefill(args, kwargs) -> bool:
    meta = kwargs['logits_metadata'] if 'logits_metadata' in kwargs else args[3] if len(args) > 3 else None
    mode = getattr(meta, 'forward_mode', None)
    is_extend = getattr(mode, 'is_extend', None)
    is_verify = getattr(mode, 'is_target_verify', None)
    if callable(is_extend) and callable(is_verify):
        return bool(is_extend()) and (not bool(is_verify()))
    hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[1] if len(args) > 1 else None
    rows = int(hidden.shape[0]) if isinstance(hidden, torch.Tensor) and hidden.dim() >= 1 else 0
    return rows > 0 and (rows < 4 or rows % 4 != 0)

def _logits_buffer(module, args, kwargs):
    """Stock's fp32 ``next_token_logits_buffer`` for a decode / target-verify call, else None.

    In those modes stock's ``_get_logits`` runs once and copies the bf16 logits into this buffer when the shapes
    match (``_copy_logits_to_buffer``). Writing the bf16-rounded logits straight into it makes that copy a no-op
    (``copy_`` of a tensor onto itself returns at once) with the same bytes in the buffer."""
    meta = kwargs['logits_metadata'] if 'logits_metadata' in kwargs else args[3] if len(args) > 3 else None
    mode = getattr(meta, 'forward_mode', None)
    is_decode = getattr(mode, 'is_decode', None)
    is_verify = getattr(mode, 'is_target_verify', None)
    if not (callable(is_decode) and callable(is_verify)) or not (is_decode() or is_verify()):
        return None
    if getattr(module, 'logit_scale', None) is not None:
        return None
    buf = getattr(meta, 'next_token_logits_buffer', None)
    if not isinstance(buf, torch.Tensor) or buf.dtype != torch.float32 or buf.dim() != 2 or not buf.is_contiguous():
        return None
    return buf

def _in_autotune_dummy_run() -> bool:
    try:
        from sglang.srt.layers import logits_processor as stock_logits
        probe = getattr(stock_logits, 'get_in_autotune_dummy_run', None)
        return bool(probe()) if callable(probe) else False
    except Exception:
        return False

def _late_map(args, kwargs) -> None:
    if dense8_store.late_map_armed() and (not _in_autotune_dummy_run()):
        dense8_store.late_map_hook(_is_prefill(args, kwargs))

class _Apply:
    __slots__ = ('node', 'lm_head')

    def __init__(self, node, lm_head):
        self.node, self.lm_head = (node, lm_head)

    def apply(self, _stand_in, hidden_states, embedding_bias=None):
        return self.node.logits(self.lm_head, hidden_states, embedding_bias)

class _StandInHead:
    __slots__ = ('quant_method',)

    def __init__(self, apply):
        self.quant_method = apply

class _Node:

    def __init__(self, module):
        self.module = module
        self.stock_forward = type(module).forward
        self.q = self.scale = None
        self.source = None
        self.out_buf = None

    def _ready(self, weight) -> bool:
        key = (weight.data_ptr(), tuple(weight.shape), weight.dtype, weight.device)
        if self.q is not None and self.source == key:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        with torch.no_grad():
            self.q = self.scale = None
            self.q, self.scale = quantize_rows(weight)
        self.source = key
        _evidence('lmhead8_build', int(weight.shape[0]))
        return True

    def logits(self, lm_head, hidden_states, embedding_bias):
        weight = lm_head.weight
        rows = hidden_states.shape[0] if hidden_states.dim() == 2 else 0
        buf, self.out_buf = self.out_buf, None
        if embedding_bias is None and hidden_states.dtype == torch.bfloat16 and (0 < rows <= MAX_ROWS) and (hidden_states.shape[1] == weight.shape[1]) and (hidden_states.stride(1) == 1) and (hidden_states.device == weight.device) and self._ready(weight):
            # the fp32 logits buffer only when it is exactly what stock would copy these logits into
            out = buf if buf is not None and rows >= LOGITS_BUF_MIN_ROWS and tuple(buf.shape) == (rows, weight.shape[0]) and weight.shape[0] == getattr(self.module, 'vocab_size', -1) and buf.device == weight.device else None
            tag = '_f32buf' if out is not None else ''
            if rows >= W8A8_MIN_ROWS:
                _evidence(('lmhead8_w8a8x2' if W8A8_TWO_TERM else 'lmhead8_w8a8') + tag, rows)
                return w8a8_linear(hidden_states, self.q, self.scale, out=out, two_term=W8A8_TWO_TERM)
            _evidence('lmhead8_int8' + tag, rows)
            return w8a16_linear(hidden_states, self.q, self.scale, out=out)
        _evidence('lmhead8_stock_matmul', rows)
        return torch.matmul(hidden_states.to(weight.dtype), weight.T)

    def __call__(self, *args, **kwargs):
        _late_map(args, kwargs)
        if 'lm_head' in kwargs:
            lm_head = kwargs['lm_head']
            if _plain_head(self.module, lm_head):
                kwargs = {**kwargs, 'lm_head': _StandInHead(_Apply(self, lm_head))}
        elif len(args) > 2:
            lm_head = args[2]
            if _plain_head(self.module, lm_head):
                args = (*args[:2], _StandInHead(_Apply(self, lm_head)), *args[3:])
        self.out_buf = _logits_buffer(self.module, args, kwargs)
        try:
            return self.stock_forward(self.module, *args, **kwargs)
        finally:
            self.out_buf = None

def prepare(module):
    return _Node(module)

def forward(prepared, *args, **kwargs):
    return prepared(*args, **kwargs)
