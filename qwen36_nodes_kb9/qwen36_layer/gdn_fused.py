"""The recurrent block's target-verify step between the two projections, as one CUDA kernel.

For a chain verify batch, ``qk_gdn_verify.cu`` replaces stock's four launches (the conv
update with its per-draft windows and conv-state roll, the packed q/k/v split, the
ReplaySSM verify recurrence with its ring appends, and the gated RMSNorm). It reads the
in_proj outputs and returns the out_proj operand. The kernel is specialized to this
model's recurrent block and to the served ring: 4-token drafts on a 16-entry circular
(d, k, g) ring with bf16 high/low parts. Anything else raises on the layer's first call.
"""

import qk_gdn_verify  # built from qk_gdn_verify.cu by the validator's CUDA build step
from sglang.srt.model_executor.forward_context import get_attn_backend

DRAFT = 4
RING = 16
_checked = set()


def _require_served_layout(lin, layer, norm, params, draft):
    pool = lin.req_to_token_pool.mamba_pool
    if getattr(lin, "mis_metadata", None) is not None:
        raise RuntimeError("GDN multi-item scoring is not served by this bundle")
    if (getattr(params, "replayssm_d", None) is None or getattr(pool, "replayssm_cache_base", None) is None
            or getattr(pool, "replayssm_spec_fold", True) or getattr(pool, "replayssm_is_kda", False)
            or params.replayssm_rawv is None or params.replayssm_rawk is None
            or getattr(params, "replayssm_beta", None) is not None or params.replayssm_d.shape[-2] != RING):
        raise RuntimeError("the fused verify kernel serves the 16-entry compact circular ReplaySSM ring only")
    if draft != DRAFT:
        raise RuntimeError(f"the fused verify kernel serves {DRAFT}-token drafts, not {draft}")
    if layer.bias is not None or layer.activation not in ("silu", "swish"):
        raise RuntimeError("the fused verify kernel serves a bias-free SiLU conv only")
    if (norm.norm_before_gate is not True or norm.group_size not in (None, norm.weight.numel())
            or norm.activation not in ("silu", "swish") or getattr(norm, "bias", None) is not None):
        raise RuntimeError("the fused verify kernel serves the swish-gated RMSNorm only")


def gdn_verify_fused(layer, norm, forward_batch, mixed_qkv, z, a, b, pf_weight):
    """``out_proj``'s input ``[tokens, 32 * 128]`` for a target-verify batch.

    kgdnpf: ``pf_weight`` (the layer's ``out_proj`` weight) is only L2-prefetched by idle-SM CTAs.
    """
    lin = get_attn_backend().linear_attn_backend
    fm = lin.forward_metadata
    params = lin.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    if layer.layer_id not in _checked:
        _require_served_layout(lin, layer, norm, params, forward_batch.spec_info.draft_token_num)
        _checked.add(layer.layer_id)
    if fm.retrieve_next_token is not None or fm.retrieve_next_sibling is not None:
        raise RuntimeError("this bundle serves chain speculation (top-k one) only")
    pool = lin.req_to_token_pool.mamba_pool
    return qk_gdn_verify.gdn_verify_fused(
        mixed_qkv, z, a, b, params.conv[0], layer.conv_weights, params.intermediate_conv_window[0],
        layer.A_log, layer.dt_bias, params.temporal, params.replayssm_d, params.replayssm_rawv,
        params.replayssm_k, params.replayssm_rawk, params.replayssm_g, fm.query_start_loc,
        fm.mamba_cache_indices, forward_batch.req_pool_indices, lin.verify_intermediate_state_indices,
        pool.replayssm_spec_write_pos, pool.replayssm_cache_base, norm.weight, norm.eps,
        layer.head_k_dim ** -0.5, pf_weight)  # kgdnpf: prefetch target
