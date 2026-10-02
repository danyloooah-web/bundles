"""Prefill projections without cuBLAS's per-call memset (``qk_lt.cu``).

``linear(x, w)`` is ``F.linear(x, w)`` bit for bit. It runs the zero-workspace cuBLASLt plan only for shapes
that have been compared against ``F.linear`` on a real call outside graph capture (the engine's warm-up runs
each prefill width eagerly before capturing it); an unchecked shape under capture, a mismatch, or no plan
keeps that shape on ``F.linear``.

kb24 (ltsweep): shapes in ``_CFG`` first try a pinned cuBLASLt configuration (every algo id / tile / stages / swizzle /
custom option / cluster shape at split-K 1 enumerated and compared bit for bit against F.linear on real weights;
the fastest bit-identical zero-workspace one, timed in a CUDA graph with cold weights); the same first-call check
decides, and a pin that is rejected or mismatches falls back to the tile-preference plan below, then F.linear.
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
# (rows, N, K) -> pinned configuration (algo id, tile id, stages id, cta swizzling, custom option, cluster shape id,
# inner shape id): the fastest zero-workspace one of the sweep, bit-identical to F.linear on real weights over 3
# activation seeds (H100, cuBLAS 13.1; CUDA graph, cold weights: kb23's call -> pinned). Only the large wins are pinned:
# at 8192 rows the router / ba pins (-2 % / -6 % in that bench) ran 2-4 % slower inside the engine's prefill graph
# (c48 profile), so 8192 keeps kb23's plans; k/v, ba at 2500 and the bf16 out_proj at 4096 (-1.5..-4.6 %) are not pinned.
_CFG = {
    (4096, 256, 2048): (66, 17, 35, 0, 2, 2, 0),   # router logits (i8x_kernels)  10.8 -> 8.0 us
    (2500, 256, 2048): (66, 26, 35, 0, 1, 4, 0),   #                               8.5 -> 6.8
    (4096, 64, 2048): (66, 12, 35, 0, 2, 2, 0),    # GDN in_proj_ba                6.4 -> 5.4
}
_verdict = {}  # (device, M, N, K) -> "pin" (pinned plan checked equal) / True (tile plan checked equal) / False


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
        ref = F.linear(x, w)
        cfg = _CFG.get(key[1:])
        if cfg is not None:
            y = qk_lt.linear_cfg(x, w, list(cfg))
            if y.numel() == ref.numel() and torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                _verdict[key] = "pin"
                return ref
        y = qk_lt.linear(x, w, _TILES.get(key[1:], ()))
        _verdict[key] = bool(y.numel() == ref.numel() and torch.equal(y.view(torch.int16), ref.view(torch.int16)))
        return ref
    if not ok:
        return F.linear(x, w)
    if ok == "pin":
        branch_evidence(f"lt_pin_{w.shape[0]}x{x.shape[1]}", x.shape[0])
        return qk_lt.linear_cfg(x, w, list(_CFG[key[1:]]))
    branch_evidence("lt_linear", x.shape[0])
    return qk_lt.linear(x, w, _TILES.get(key[1:], ()))
