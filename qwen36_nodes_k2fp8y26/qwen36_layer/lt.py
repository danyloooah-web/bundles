"""Prefill projections without cuBLAS's per-call memset (``qk_lt.cu``).

``linear(x, w)`` is ``F.linear(x, w)`` bit for bit. It runs the zero-workspace cuBLASLt plan only for shapes
that have been compared against ``F.linear`` on a real call outside graph capture (the engine's warm-up runs
each prefill width eagerly before capturing it); an unchecked shape under capture, a mismatch, or no plan
keeps that shape on ``F.linear``.
"""

import torch
import torch.nn.functional as F

import qk_lt  # built from qk_lt.cu by the validator's CUDA build step
from qwen36_layer.evidence import branch_evidence

MIN_ROWS = 64
# (rows, N, K) -> cuBLASLt tile ids to prefer, in order (H100, cuBLAS 13 of the arena image; 30 cold weights in a
# graph chain, every listed algorithm bit-identical): attention qkv at a full 8192-row chunk 421-424 us against
# F.linear's 440 (the first-listed tile 446); in_proj_ba 8.3 us against 9.5.
_TILES = {(8192, 9216, 2048): (201, 197), (8192, 64, 2048): (14,)}
_verdict = {}  # (device, M, N, K) -> True (plan checked equal) / False (stays on F.linear)


def linear(x, w):
    if not (x.ndim == 2 and w.ndim == 2 and x.shape[0] >= MIN_ROWS and x.shape[1] == w.shape[1]
            and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_contiguous() and w.is_contiguous()
            and x.is_cuda):
        return F.linear(x, w)
    key = (x.device.index, x.shape[0], w.shape[0], x.shape[1])
    ok = _verdict.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return F.linear(x, w)
        y = qk_lt.linear(x, w, _TILES.get(key[1:], ()))
        ref = F.linear(x, w)
        _verdict[key] = ok = bool(y.numel() == ref.numel() and torch.equal(y.view(torch.int16), ref.view(torch.int16)))
        return ref
    if not ok:
        return F.linear(x, w)
    branch_evidence("lt_linear", x.shape[0])
    return qk_lt.linear(x, w, _TILES.get(key[1:], ()))
