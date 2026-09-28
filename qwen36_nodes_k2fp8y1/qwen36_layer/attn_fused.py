
import torch

import qk_attn_verify
import qk_kv_pages
import qk_gated_norm
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import get_tc_piecewise_forward_context

from qwen36_layer.proj_fused import qkv_proj
from qwen36_layer import dense8_store
from qwen36_layer.dense8 import qkv_proj8
from qwen36_layer.evidence import branch_evidence
from qwen36_layer.lt import linear as lt_linear
from qwen36_layer import fp8

# PRECISION CHANGE: the prefill q / gate projection runs in FP8 from this attention layer on (0: every one).
FP8_QG_MIN_LAYER = 0
# Layers kept in bf16: the audit's weakest window of the FP8 build was layer 35's residual (0.7837 of rows passing
# against 0.75; every other layer >= 0.867).
FP8_SKIP_LAYERS = frozenset({35})
# PRECISION CHANGE: prefill o_proj / out_proj in FP8 from this layer on (the node audit grades every layer's outputs
# within 2 % per row; layer 0's FP8 out_proj read 0.211 of rows passing, its residual being small).
FP8_PROJ_MIN_LAYER = 8

DRAFT = 4
HEADS, KV_HEADS, HEAD_DIM, ROTARY = 16, 2, 256, 64
PAGE = qk_kv_pages.PAGE
# The re-layout pays from about 2k query rows (an 8192-row chunk with no prefix saves 28 us,
# one after 57k cached rows 1.2 ms per layer; three rows lose 60 us to its launches). The page
# buffers hold every row the batch attends to, K and V at 512 B each, capped at 2 x 64 MB.
PAGED_MIN_ROWS = 2048
PAGED_MAX_ROWS = 1 << 17
_checked = set()
_workspaces = {}


def _require_served_layout(layer, backend, k_cache):
    attn = layer.attn
    if (layer.num_heads != HEADS or layer.num_kv_heads != KV_HEADS or layer.head_dim != HEAD_DIM
            or layer.rotary_emb.rotary_dim != ROTARY or not layer.attn_output_gate):
        raise RuntimeError("the fused attention serves 16 x 256 query heads over 2 KV heads with a gate")
    if layer.q_norm.variance_epsilon != layer.k_norm.variance_epsilon:
        raise RuntimeError("the fused attention assumes one q/k norm epsilon")
    if k_cache.dtype != torch.float8_e4m3fn or attn.k_scale is not None or attn.v_scale is not None:
        raise RuntimeError("the fused attention serves an unscaled E4M3 KV cache only")
    if (backend.page_size != 1 or backend.topk > 1 or getattr(backend, "use_mla", False)
            or (attn.sliding_window_size is not None and attn.sliding_window_size > -1)
            or attn.is_cross_attention or attn.logit_cap not in (None, 0, 0.0)):
        raise RuntimeError("the fused attention serves one-token pages, chain drafts and full causal attention")


def _cache(layer, forward_batch):
    backend = get_attn_backend().full_attn_backend
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.attn.layer_id)
    if layer.attn.layer_id not in _checked:
        _require_served_layout(layer, backend, k_cache)
        _checked.add(layer.attn.layer_id)
    return backend, k_cache, v_cache


def _rope(layer, positions):
    return layer.rotary_emb.cos_sin_cache, layer.rotary_emb.axis_map if positions.dim() == 2 else None


def _workspace(device):
    """Split partials for the largest grid (one CTA per SM) and the split counters."""
    ws = _workspaces.get(device)
    if ws is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ws = _workspaces[device] = (
            sms,
            torch.empty(sms * 32 * HEAD_DIM, dtype=torch.float32, device=device),
            torch.empty(sms * 2 * 32, dtype=torch.float32, device=device),
            torch.zeros(2 * 128, dtype=torch.int32, device=device),
        )
    return ws


def attn_verify_fused(layer, forward_batch, positions, qkv):
    """``o_proj``'s input ``[tokens, 16 * 256]`` for a target-verify batch."""
    backend, k_cache, v_cache = _cache(layer, forward_batch)
    if forward_batch.spec_info.draft_token_num != DRAFT:
        raise RuntimeError(f"the fused verify attention serves {DRAFT}-token drafts")
    md = backend.forward_metadata
    sms, ws_o, ws_ml, counters = _workspace(qkv.device)
    batch = qkv.shape[0] // DRAFT
    splits = max(1, min(64, sms // (batch * KV_HEADS)))
    cos_sin, axis_map = _rope(layer, positions)
    return qk_attn_verify.attn_verify_fused(
        qkv, layer.q_norm.weight, layer.k_norm.weight, layer.q_norm.variance_epsilon, cos_sin, positions, axis_map,
        k_cache, v_cache, forward_batch.out_cache_loc, md.page_table, md.cache_seqlens_int32,
        layer.attn.scaling, splits, ws_o, ws_ml, counters)


def _paged_attention(forward_batch, radix, q8, out):
    """Stock's FA3 extend call into ``out``, over 128-row pages of the batch's K/V rows;
    ``False`` when the batch is too small to gain or too large for the page buffers."""
    backend = get_attn_backend().full_attn_backend
    md = backend.forward_metadata
    seq_lens = forward_batch.seq_lens_cpu
    if seq_lens is None or q8.shape[0] < PAGED_MIN_ROWS:
        return False
    pages = (seq_lens + (PAGE - 1)) // PAGE
    total = int(pages.sum())
    if total * PAGE > PAGED_MAX_ROWS:
        return False
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(radix.layer_id)
    lens = md.cache_seqlens_int32
    first = (lens + (PAGE - 1)) // PAGE
    first = torch.cumsum(first, 0, dtype=torch.int32) - first
    widest = int(pages.max())
    k_pages, v_pages = qk_kv_pages.kv_pages(k_cache, v_cache, md.page_table, lens, first, total, widest)
    table = first[:, None] + torch.arange(widest, dtype=torch.int32, device=q8.device)
    flash_attn_with_kvcache(
        q=q8.view(-1, HEADS, HEAD_DIM), k_cache=k_pages, v_cache=v_pages, page_table=table, cache_seqlens=lens,
        cu_seqlens_q=md.cu_seqlens_q, cu_seqlens_k_new=md.cu_seqlens_k, max_seqlen_q=md.max_seq_len_q,
        softmax_scale=radix.scaling, causal=True, window_size=(-1, -1), softcap=0.0, num_splits=backend.num_splits,
        pack_gqa=False, ver=backend.fa_impl_ver, out=out.view(-1, HEADS, HEAD_DIM))
    return True


def _prefill_attention(q8, out, layer_id):
    """Graph-break body: the prefill attention on the live batch, into ``out``.

    Like stock's attention op it takes the batch and layer from the prefill graph context,
    so a replay sees the live metadata, and it zeroes the rows a capture bucket padded;
    stock's own body runs the batches the page re-layout does not pay for.
    """
    from sglang.srt.layers.radix_attention import _unified_attention_with_output_impl, _zero_padded_pcg_tail

    context = get_tc_piecewise_forward_context()
    forward_batch, radix = context.forward_batch, context.attention_layers[layer_id]
    n = forward_batch.global_num_token_non_padded_cpu
    if not n or not _paged_attention(forward_batch, radix, q8[:n], out[:n]):
        _unified_attention_with_output_impl(q8, None, None, out, False, layer_id, False, False)
        return
    _zero_padded_pcg_tail(out, context)


_prefill_attention_break = eager_on_graph(True)(_prefill_attention)


def attn_prefill_fused(layer, forward_batch, positions, qkv, kv=None, fp8_rows=False):
    """``o_proj``'s input for a plain prefill batch: our front, stock FA3 on 128-row pages,
    stock's gate; with ``fp8_rows`` the gated rows come back as fp8.py's (e4m3 rows, scales)."""
    _, k_cache, v_cache = _cache(layer, forward_batch)
    cos_sin, axis_map = _rope(layer, positions)
    if kv is None:
        q8 = qk_attn_verify.attn_prefill_front(
            qkv, layer.q_norm.weight, layer.k_norm.weight, layer.q_norm.variance_epsilon, cos_sin, positions, axis_map,
            k_cache, v_cache, forward_batch.out_cache_loc)
    else:  # qkv is the q / gate block [T, 8192] and kv the k / v block [T, 1024]
        q8 = qk_attn_verify.attn_prefill_front_split(
            qkv, kv, layer.q_norm.weight, layer.k_norm.weight, layer.q_norm.variance_epsilon, cos_sin, positions,
            axis_map, k_cache, v_cache, forward_batch.out_cache_loc)
    attn = torch.empty_like(q8, dtype=torch.bfloat16)
    if get_tc_piecewise_forward_context() is not None:
        _prefill_attention_break(q8, attn, layer.attn.layer_id)
    elif not _paged_attention(forward_batch, layer.attn, q8, attn):
        attn = layer.attn(q8, None, None, forward_batch, save_kv_cache=False)
    tokens = qkv.shape[0]
    if fp8_rows:
        # stock's gate then fp8.py's quantizer in one kernel (qk_gated_norm.cu), bit for bit the two launches.
        return qk_gated_norm.gate_mul_fp8(attn, qkv)
    gate = qkv[:, :2 * HEADS * HEAD_DIM].view(tokens, HEADS, 2 * HEAD_DIM)[:, :, HEAD_DIM:]
    return fused_sigmoid_mul(attn, gate, inplace=True)


def install(module, replica):
    """Route the replica's target-verify and plain-prefill attention through our kernels."""
    stock = type(module).self_attention
    layer_id = int(getattr(getattr(module, "attn", None), "layer_id", -1))
    # DENSE8: registered only (no allocation); the INT8 copy is built at the first eligible call
    dense8_qkv = dense8_store.register("qkv", (getattr(getattr(module, "qkv_proj", None), "weight", None),))
    o_proj = getattr(module, "o_proj", None)
    fp8_o = (layer_id >= FP8_PROJ_MIN_LAYER and layer_id not in FP8_SKIP_LAYERS and o_proj is not None
             and getattr(o_proj, "bias", None) is None and getattr(o_proj, "tp_size", 1) == 1
             and tuple(o_proj.weight.shape) == (2048, HEADS * HEAD_DIM) and o_proj.weight.dtype == torch.bfloat16
             and o_proj.weight.is_contiguous())

    def self_attention(positions, hidden_states, forward_batch):
        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            copy8 = dense8_store.get(dense8_qkv, hidden_states)
            if copy8 is not None:
                qkv = qkv_proj8(hidden_states, copy8[0], copy8[1])
                branch_evidence(dense8_store.MARKER, hidden_states.shape[0])
            else:
                qkv = qkv_proj(hidden_states, replica.qkv_proj.weight)
            fused = attn_verify_fused
        elif mode == ForwardMode.EXTEND:
            w_qkv = replica.qkv_proj.weight
            if (replica.qkv_proj.bias is None and hidden_states.ndim == 2 and hidden_states.dtype == torch.bfloat16
                    and w_qkv.dtype == torch.bfloat16 and getattr(replica.qkv_proj, "tp_size", 1) == 1):
                if (layer_id >= FP8_QG_MIN_LAYER and layer_id not in FP8_SKIP_LAYERS
                        and fp8.eligible(hidden_states, w_qkv)):
                    # PRECISION CHANGE (fp8.py): q / gate [T, 8192] in FP8; k / v [T, 1024] stay bf16 (F.linear's
                    # bits), so the KV-cache rows the node audit grades are stock's. The front reads the two blocks.
                    branch_evidence("fp8_qg", hidden_states.shape[0])
                    x_c = hidden_states.contiguous()
                    qg, kv = fp8.linear(x_c, w_qkv[:8192]), lt_linear(x_c, w_qkv[8192:])
                    if fp8_o:
                        # PRECISION CHANGE (fp8.py): the gate writes o_proj's e4m3 rows and scales directly.
                        q8, scale = attn_prefill_fused(replica, forward_batch, positions, qg, kv, fp8_rows=True)
                        branch_evidence("fp8_proj", q8.shape[0])
                        return fp8.linear_q(q8, scale, module.o_proj.weight)
                    output, _ = replica.o_proj(attn_prefill_fused(replica, forward_batch, positions, qg, kv))
                    return output
                qkv = lt_linear(hidden_states.contiguous(), w_qkv)  # F.linear's bits, no memset (lt.py)
            else:
                qkv, _ = replica.qkv_proj(hidden_states)
            fused = attn_prefill_fused
        else:
            return stock(replica, positions, hidden_states, forward_batch)
        output, _ = replica.o_proj(fused(replica, forward_batch, positions, qkv))
        return output

    replica.self_attention = self_attention
