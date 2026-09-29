"""Branch evidence: one parseable line per process, rank, branch and capture state.

``CACHEON-AUTHORED-BRANCH: <branch> pid=<pid> rank=<rank> node=model.layers.* rows=<rows> captured=<0|1>``

One line per (process, rank, branch, captured), not per layer and not per
call: the question a reader asks is which RANKS took a path, and a per-layer
emission lets one rank print forty lines that read as coverage. Emission is
host-side only -- a print records nothing into a CUDA graph -- and a line
printed while a graph is being recorded proves the path is IN that graph,
which is the only way it runs during replay.
"""

import os

import torch

_BRANCH_SEEN = set()


def branch_identity():
    """(pid, rank) as this process can read them now.

    Rank comes from the live process group. -1 means it was not readable, which
    is not coverage; the bracket gate refuses a marker that carries it.
    """
    pid = os.getpid()
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return pid, int(dist.get_rank())
    except Exception:  # noqa: BLE001 - evidence must never break execution
        pass
    return pid, -1


def branch_evidence(branch, rows):
    """Record that ``branch`` ran on ``rows`` rows, and whether under capture."""
    try:
        captured = bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        captured = False
    pid, rank = branch_identity()
    key = (pid, rank, branch, captured)
    if key in _BRANCH_SEEN:
        return
    _BRANCH_SEEN.add(key)
    print(
        f"CACHEON-AUTHORED-BRANCH: {branch} pid={pid} rank={rank} "
        f"node=model.layers.* rows={rows} captured={int(captured)}",
        flush=True,
    )
