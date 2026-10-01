"""Idle device bytes behind each layer's INT8 expert copy, handed out as persistent buffers.

i8x (i8x_reloc.py) writes a layer's INT8 experts (256 x 3,194,880 B = 818 MB) into the front of the served w13's
old device storage (1,074 MB) and keeps that storage alive through the copy's view; the last ~256 MB of every layer's
storage stays allocated and unused (10.2 GB over 40 layers). This module bump-allocates 256-byte-aligned uint8 views
from those tails once every layer has relocated, so resident copies (DENSE8's INT8 dense weights, the FP8 weight
rows) take no memory the engine still has free: allocating them from free memory after the KV pool's plan left the
king's post-capture DENSE8 late map without room (CUDA OOM, refused to serve).

``alloc`` returns None until all layers are relocated, when no tail has room, or under graph capture; callers keep
their own path then. Nothing here frees: views live as long as the copies (the process).
"""

import torch

from qwen36_layer import i8x_reloc

ALIGN = 256
_cursor = {}  # device -> [tails, tail index, offset within tail]
_stats = {"bytes": 0, "views": 0}


def _tails(device):
    """(storage, first free byte, end byte) of every relocated layer's old w13 storage on ``device``."""
    out = []
    for entry in i8x_reloc._entries:
        blocks = getattr(entry.copy, "blocks", None) if entry.state == "i8x" else None
        if blocks is None or blocks.device != device or blocks.dtype != torch.uint8:
            continue
        storage = blocks.untyped_storage()
        start = -(-(blocks.storage_offset() + blocks.numel()) // ALIGN) * ALIGN
        if start < storage.nbytes():
            out.append((storage, start, storage.nbytes()))
    return out


def ready():
    """Every layer relocated (the tails exist and nothing else will be carved from them)."""
    return i8x_reloc._state["closed"] is None and i8x_reloc._totals["layers"] >= i8x_reloc.LAYERS


def alloc(nbytes, device):
    """A persistent uint8 [nbytes] view into an idle tail on ``device``, or None."""
    if nbytes <= 0 or not ready() or torch.cuda.is_current_stream_capturing():
        return None
    cur = _cursor.get(device)
    if cur is None:
        cur = _cursor[device] = [_tails(device), 0, None]
    tails = cur[0]
    while cur[1] < len(tails):
        storage, start, end = tails[cur[1]]
        off = start if cur[2] is None else cur[2]
        if off + nbytes <= end:
            view = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, off, (nbytes,))
            cur[2] = -(-(off + nbytes) // ALIGN) * ALIGN
            _stats["bytes"] += nbytes
            _stats["views"] += 1
            return view
        cur[1], cur[2] = cur[1] + 1, None
    return None


def tensor(shape, dtype, device):
    """A persistent tensor of ``shape`` / ``dtype`` in an idle tail, or None."""
    n = 1
    for d in shape:
        n *= int(d)
    view = alloc(n * torch.empty(0, dtype=dtype).element_size(), device)
    return None if view is None else view.view(dtype).view(*shape)
