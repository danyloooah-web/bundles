"""The decode-width projections with our CUDA kernels (``qk_proj_decode.cu``).

Row batches of 1 to 256 token rows (decode and MTP target-verify) of the Qwen3.6-35B-A3B BF16
decoder layer: the recurrent block's in_proj_qkvz + in_proj_ba and the attention block's fused
q/k/v projection. Any other shape, dtype or layout raises; nothing falls back to another
kernel. Every kernel is launched on the current stream with programmatic dependent launch.
"""

import qk_proj_decode  # built from qk_proj_decode.cu by the validator's CUDA build step


def gdn_in_proj(x, w_qkvz, w_ba, *, qkv, nv):
    """``mixed_qkv [M, qkv]``, ``z [M, nv, 128]``, ``b [M, nv]``, ``a [M, nv]`` of
    ``x [M, 2048] @ [w_qkvz; w_ba].T``: the same bf16 bits as the Triton kernel it replaces."""
    return tuple(qk_proj_decode.gdn_in_proj(x, w_qkvz, w_ba, qkv, nv))


def qkv_proj(x, w):
    """``x [M, 2048] @ w [9216, 2048].T``: the same bf16 bits as cuBLAS."""
    return qk_proj_decode.qkv_proj(x, w)
