"""Prefill front of the recurrent block in one launch: causal conv, q/k/v split, gating.

Stock runs four kernels between the input projection and the chunked delta-rule
kernel: the causal conv over the concatenated ``[q | k | v]`` channels, a copy
that splits them into per-head tensors, the gate ``g``/``beta`` computation and
(inside the dispatcher) the q/k L2 norm. The first three move the same 134 MB
per 8192-token chunk two or three times. This kernel computes the conv per
128-channel head and writes each head straight into the split layouts the
chunked kernel consumes, and the value-head programs also produce ``g`` and
``beta``. The arithmetic is stock's: bf16 tap products accumulated in fp32
oldest tap first, ``acc / (1 + exp(-acc))``, one bf16 rounding; the conv state
keeps its meaning (the raw last three input rows of each sequence, or the
shifted merge when a sequence is shorter), read and written by the programs
that own a sequence's first token, as stock's first-chunk program does.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph

import qk_conv_gate  # built from qk_conv_gate.cu by the validator's CUDA build step
import qk_inproj  # built from qk_inproj.cu by the validator's CUDA build step

BLOCK_T = 8
BLOCK_C = 512
HEAD = 128
WIDTH = 4


@triton.jit
def _conv_split_gate_kernel(
    x_ptr, w_ptr, cs_ptr, qsl_ptr, idx_ptr, hinit_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    D: tl.constexpr, HEAD: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, L2NORM: tl.constexpr, ALPHA: tl.constexpr, EPS: tl.constexpr,
):
    NH: tl.constexpr = BLOCK_C // HEAD
    # Channel blocks vary fastest so the programs in flight cover whole rows.
    h0 = tl.program_id(0) * NH
    t0 = tl.program_id(1) * BLOCK_T
    seq = tl.program_id(2)
    start = tl.load(qsl_ptr + seq)
    end = tl.load(qsl_ptr + seq + 1)
    L = end - start
    if t0 < L:
        n = min(BLOCK_T, L - t0)
        slot = tl.load(idx_ptr + seq)
        hinit = tl.load(hinit_ptr + seq) > 0
        # Channels as [head, dim] so the per-head norm is a plain row reduction.
        offs_c = h0 * HEAD + tl.arange(0, NH)[:, None] * HEAD + tl.arange(0, HEAD)[None, :]
        w0 = tl.load(w_ptr + offs_c * 4 + 0)
        w1 = tl.load(w_ptr + offs_c * 4 + 1)
        w2 = tl.load(w_ptr + offs_c * 4 + 2)
        w3 = tl.load(w_ptr + offs_c * 4 + 3)
        # The three rows before this block: the sequence's own, or its conv state
        # (column 2 is the most recent) for the first block.
        if t0 == 0:
            use = hinit & (slot >= 0)
            sbase = cs_ptr + tl.where(use, slot, 0) * (D * 3) + offs_c * 3
            col0 = tl.load(sbase + 0, mask=use, other=0.0)
            col1 = tl.load(sbase + 1, mask=use, other=0.0)
            col2 = tl.load(sbase + 2, mask=use, other=0.0)
        else:
            xb = x_ptr + (start + t0) * D + offs_c
            col0 = tl.load(xb - 3 * D)
            col1 = tl.load(xb - 2 * D)
            col2 = tl.load(xb - 1 * D)
        offs_h = tl.arange(0, NH)
        offs_o = offs_c - h0 * HEAD
        # Unrolled so every row load of the block is in flight at once.
        for i in tl.static_range(BLOCK_T):
            live = i < n
            t = start + t0 + i
            xr = tl.load(x_ptr + t * D + offs_c, mask=live, other=0.0)
            acc = tl.zeros((NH, HEAD), dtype=tl.float32)
            acc += col0 * w0
            acc += col1 * w1
            acc += col2 * w2
            acc += xr * w3
            col0 = col1
            col1 = col2
            col2 = xr
            y = (acc / (1 + tl.exp(-acc))).to(tl.bfloat16)
            if L2NORM and h0 < NQ + NK:
                # Stock's l2norm of the bf16 q/k rows, per head, in fp32.
                yh = y.to(tl.float32)
                y = (yh / tl.sqrt(tl.sum(yh * yh, axis=1) + EPS)[:, None]).to(tl.bfloat16)
            # This block's heads are contiguous within the token's row of q, k or v.
            if h0 < NQ:
                tl.store(q_ptr + (t * NQ + h0) * HEAD + offs_o, y, mask=live)
            elif h0 < NQ + NK:
                tl.store(k_ptr + (t * NK + h0 - NQ) * HEAD + offs_o, y, mask=live)
            else:
                tl.store(v_ptr + (t * NV + h0 - NQ - NK) * HEAD + offs_o, y, mask=live)
        if h0 >= NQ + NK:
            # Gate and beta for this block's value heads, as stock's gating kernel computes them.
            hv = h0 - NQ - NK + offs_h
            offs_t = start + t0 + tl.arange(0, BLOCK_T)
            m = (tl.arange(0, BLOCK_T) < n)[:, None]
            o = offs_t[:, None] * NV + hv[None, :]
            av = tl.load(a_ptr + o, mask=m, other=0.0).to(tl.float32)
            bv = tl.load(b_ptr + o, mask=m, other=0.0).to(tl.float32)
            alog = tl.load(alog_ptr + hv).to(tl.float32)
            dtb = tl.load(dtb_ptr + hv).to(tl.float32)
            xg = av + dtb[None, :]
            sp = tl.where(1.0 * xg <= 20.0, (1 / 1.0) * tl.log(1 + tl.exp(1.0 * xg)), xg)
            gv = -tl.exp(alog)[None, :] * sp
            if ALPHA:
                # The chunked kernel takes exp(g); stock's torch.exp is libdevice's.
                gv = libdevice.exp(gv)
            tl.store(g_ptr + o, gv, mask=m)
            # Stock rounds beta through b's dtype (bf16) before its fp32 store.
            tl.store(beta_ptr + o, tl.sigmoid(bv).to(tl.bfloat16).to(tl.float32), mask=m)
        if t0 == 0 and slot >= 0:
            # New conv state: the raw last three input rows, or the old state
            # shifted by the sequence length when it is shorter. The state reads
            # above must land first (other threads may own the same columns).
            tl.debug_barrier()
            sbase = cs_ptr + slot * (D * 3) + offs_c * 3
            for col in tl.static_range(3):
                tt = end - 3 + col
                if tt >= start:
                    val = tl.load(x_ptr + tt * D + offs_c)
                else:
                    ocol = col + L
                    val = tl.load(sbase + ocol, mask=hinit & (ocol < 3), other=0.0)
                tl.store(sbase + col, val)


def conv_split_gate(x, conv_weights, conv_states, query_start_loc, cache_indices, prefix_lens, a, b, A_log, dt_bias,
                    num_q_heads, num_k_heads, num_v_heads, max_len, l2norm=False, alpha=False,
                    block_t=BLOCK_T, block_c=BLOCK_C, warps=4, stages=2):
    """Return ``(q, k, v, g, beta)`` in the dispatcher's layouts; updates ``conv_states`` in place.

    A sequence starts from its conv state when its ``prefix_lens`` entry is
    positive. With ``l2norm`` the q/k heads come out normalized and with
    ``alpha`` the gate comes out as ``exp(g)``, the chunked kernel's own inputs.
    """
    T, D = x.shape
    B = query_start_loc.shape[0] - 1
    dev = x.device
    # 32-bit addressing inside the kernel.
    assert T * D < 2**31 and conv_states.numel() < 2**31
    q = torch.empty((1, T, num_q_heads, HEAD), dtype=x.dtype, device=dev)
    k = torch.empty((1, T, num_k_heads, HEAD), dtype=x.dtype, device=dev)
    v = torch.empty((1, T, num_v_heads, HEAD), dtype=x.dtype, device=dev)
    g = torch.empty((1, T, num_v_heads), dtype=torch.float32, device=dev)
    beta = torch.empty((1, T, num_v_heads), dtype=torch.float32, device=dev)
    _conv_split_gate_kernel[(D // block_c, triton.cdiv(max_len, block_t), B)](
        x, conv_weights, conv_states, query_start_loc, cache_indices, prefix_lens, a, b, A_log, dt_bias,
        q, k, v, g, beta,
        D=D, HEAD=HEAD, NQ=num_q_heads, NK=num_k_heads, NV=num_v_heads,
        BLOCK_T=block_t, BLOCK_C=block_c, L2NORM=l2norm, ALPHA=alpha, EPS=1e-6, num_warps=warps, num_stages=stages,
    )
    return q, k, v, g, beta


_gate_cache = {}


def _gate_params(attn):
    """``A_log`` and ``dt_bias`` in fp32 (exact from bf16), made once per layer."""
    key = (id(attn.A_log), id(attn.dt_bias))
    hit = _gate_cache.get(key)
    if hit is None or hit[0] is not attn.A_log or hit[1] is not attn.dt_bias:
        hit = _gate_cache[key] = (attn.A_log, attn.dt_bias, attn.A_log.detach().float().contiguous(),
                                  attn.dt_bias.detach().float().contiguous())
    return hit[2], hit[3]


def _native_ok(attn, mixed_qkv, conv_states, md, prefix):
    """The CUDA front serves the 16 / 16 / 32-head layout (int32 or int64 metadata)."""
    ints = (torch.int32, torch.int64)
    return (mixed_qkv.shape[1] == 8192 and attn.num_q_heads == 16 and attn.num_k_heads == 16 and attn.num_v_heads == 32
            and md.query_start_loc.dtype in ints and md.mamba_cache_indices.dtype in ints and prefix.dtype in ints
            and prefix.is_cuda and md.mamba_cache_indices.is_contiguous() and md.query_start_loc.is_contiguous()
            and prefix.is_contiguous())


def _fused_extend(attn, forward_batch, mixed_qkv, a, b, output):
    """Run the recurrent block's extend with its front fused, into ``output``; ``False`` when stock must run."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    T = mixed_qkv.shape[0]
    backend = getattr(get_attn_backend(), "linear_attn_backend", None)
    md = getattr(backend, "forward_metadata", None)
    lens = forward_batch.extend_seq_lens_cpu
    prefix = forward_batch.extend_prefix_lens
    if (T == 0 or md is None or md.has_mamba_track_mask or md.query_start_loc is None or not lens
            or len(lens) + 1 != md.query_start_loc.shape[0] or sum(lens) != T
            or not isinstance(prefix, torch.Tensor) or prefix.shape[0] != len(lens)):
        return False
    cache = backend.req_to_token_pool.mamba2_layer_cache(attn.layer_id)
    conv_states, ssm_states = cache.conv[0], cache.temporal
    if (T * mixed_qkv.shape[1] >= 2**31 or conv_states.numel() >= 2**31
            or not conv_states.is_contiguous() or not ssm_states.is_contiguous() or conv_states.shape[-1] != WIDTH - 1
            or conv_states.dtype != mixed_qkv.dtype or attn.bias is not None or attn.activation != "silu"
            or tuple(attn.conv_weights.shape) != (mixed_qkv.shape[1], WIDTH) or not attn.conv_weights.is_contiguous()
            or not mixed_qkv.is_contiguous() or a.stride(-1) != 1 or b.stride(-1) != 1):
        return False
    dispatcher = backend.kernel_dispatcher
    kernel = getattr(dispatcher, "extend_kernel", None)
    prefill_fn = getattr(kernel, "_prefill_fn", None)
    # Stock's SM90 FlashInfer extend is replicated below; anything else keeps its own path.
    direct = (type(kernel).__name__ == "FlashInferGDNKernel" and prefill_fn is not None
              and not getattr(kernel, "use_state_pool", True) and not md.num_state_checkpoints
              and ssm_states.dtype == torch.float32)
    if direct and _native_ok(attn, mixed_qkv, conv_states, md, prefix):
        # The same front in CUDA (qk_conv_gate.cu), bit for bit the Triton kernel below with l2norm and alpha.
        alog, dtb = _gate_params(attn)
        q, k, v, g, beta = qk_conv_gate.conv_gate(
            mixed_qkv, attn.conv_weights, conv_states, md.query_start_loc, md.mamba_cache_indices, prefix,
            a, b, alog, dtb, max(lens))
    else:
        q, k, v, g, beta = conv_split_gate(
            mixed_qkv, attn.conv_weights, conv_states, md.query_start_loc, md.mamba_cache_indices, prefix,
            a.contiguous(), b.contiguous(), attn.A_log, attn.dt_bias, attn.num_q_heads, attn.num_k_heads,
            attn.num_v_heads, max(lens), l2norm=direct, alpha=direct)
    if not direct:
        dispatcher.extend(
            q=q, k=k, v=v, g=g, beta=beta, ssm_states=ssm_states, cache_indices=md.mamba_cache_indices,
            query_start_loc=md.query_start_loc, state_checkpoint_cu_starts=md.state_checkpoint_cu_starts,
            num_state_checkpoints=md.num_state_checkpoints,
            state_checkpoint_every_n_tokens=md.state_checkpoint_every_n_tokens, output=output)
        return True
    # Stock's SM90 dispatcher call with its q/k l2norm and exp(g) already done above.
    if md.mamba_cache_indices.dtype in (torch.int32, torch.int64) and md.query_start_loc.dtype in (torch.int32, torch.int64):
        # One launch (qk_conv_gate.cu) for stock's where / casts / gather: the same values.
        slots, cu_seqlens, initial_state = qk_conv_gate.gdn_prep(md.mamba_cache_indices.contiguous(),
                                                                 md.query_start_loc.contiguous(), ssm_states)
    else:
        slots = torch.where(md.mamba_cache_indices >= 0, md.mamba_cache_indices, ssm_states.shape[0] - 1).to(torch.int64)
        initial_state = ssm_states[slots]
        cu_seqlens = md.query_start_loc.to(torch.int64)
    final_state = torch.empty_like(initial_state)
    prefill_fn(
        q=q[0], k=k[0], v=v[0], g=g[0], beta=beta[0], scale=None, initial_state=initial_state,
        output_final_state=True, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False,
        output=output[0], output_state=final_state, state_checkpoints=None,
        checkpoint_cu_starts=md.state_checkpoint_cu_starts,
        checkpoint_every_n_tokens=md.state_checkpoint_every_n_tokens)
    ssm_states.index_copy_(0, slots, final_state)
    return True


def _prefill_with_output(mixed_qkv, a, b, output, layer_id):
    """Graph-break body: the fused extend on the batch's real rows, else stock's own break body.

    Like stock's op it takes the batch and layer from the prefill graph context,
    so a replay sees the live metadata, and it zeroes rows a capture bucket padded.
    """
    from sglang.srt.layers.radix_linear_attention import _unified_linear_attention_with_output_impl
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import get_tc_piecewise_forward_context

    context = get_tc_piecewise_forward_context()
    forward_batch, attn = context.forward_batch, context.attention_layers[layer_id]
    T = mixed_qkv.shape[0]
    real = getattr(forward_batch, "num_token_non_padded_cpu", None)
    n = T if real is None else min(int(real), T)
    if (forward_batch.forward_mode.is_target_verify() or n <= 0
            or not _fused_extend(attn, forward_batch, mixed_qkv[:n], a[:n], b[:n], output[:, :n])):
        return _unified_linear_attention_with_output_impl(mixed_qkv, a.contiguous(), b.contiguous(), output, layer_id)
    if n < T:
        output[:, n:].zero_()


def linear_attention_prefill(attn, forward_batch, mixed_qkv, a, b):
    """The recurrent block's prefill output for an extend batch; ``None`` when stock's call must run.

    Under a prefill CUDA graph the work is a graph break, as stock's is: the
    captured segment only allocates the output and records the break.
    """
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import get_tc_piecewise_forward_context

    mode = forward_batch.forward_mode
    if not mode.is_extend() or mode.is_target_verify():
        return None
    T = mixed_qkv.shape[0]
    if get_tc_piecewise_forward_context() is not None:
        output = torch.empty((1, T, attn.num_v_heads, attn.head_v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        _prefill_break(mixed_qkv, a, b, output, attn.layer_id)
        return output
    real = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if real is not None and real < T:
        return None
    output = torch.empty((1, T, attn.num_v_heads, attn.head_v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    return output if _fused_extend(attn, forward_batch, mixed_qkv, a, b, output) else None


_prefill_break = eager_on_graph(True)(_prefill_with_output)


# The fused projection (qk_inproj.cu) serves chunks of at least this many rows; below it the three cuBLAS
# GEMMs and the conv kernel run as before.
INPROJ_MIN_ROWS = 2048
Z_OUT = 4096
_inproj_weights = {}  # layer_id -> the layer's in_proj_qkvz weight, for the prefill graph break


def register_inproj(layer_id, w_qkvz):
    _inproj_weights[layer_id] = w_qkvz


def _fused_inproj_extend(attn, forward_batch, x, w_qkvz, a, b, output, z):
    """The recurrent block's extend straight from its input rows, into ``output`` and ``z``; ``False`` (nothing
    written) when this path does not serve the batch.

    One kernel (qk_inproj.cu) computes the ``[q | k | v]`` and ``z`` projections with cuBLAS's arithmetic and
    runs the conv front on the ``[q | k | v]`` tiles in place of storing them, bit for bit the three GEMMs plus
    qk_conv_gate; ``front_prep`` makes the recurrence's slots, cu_seqlens and initial states as ``gdn_prep``
    does and copies the batch's old conv states for the front to read. The recurrence is ``_fused_extend``'s.
    """
    from sglang.srt.model_executor.forward_context import get_attn_backend

    n = sum(forward_batch.extend_seq_lens_cpu or [])
    backend = getattr(get_attn_backend(), "linear_attn_backend", None)
    md = getattr(backend, "forward_metadata", None)
    lens = forward_batch.extend_seq_lens_cpu
    prefix = forward_batch.extend_prefix_lens
    if (n == 0 or n > x.shape[0] or md is None or md.has_mamba_track_mask or md.query_start_loc is None or not lens
            or len(lens) + 1 != md.query_start_loc.shape[0] or len(lens) > 256
            or not isinstance(prefix, torch.Tensor) or prefix.shape[0] != len(lens)):
        return False
    cache = backend.req_to_token_pool.mamba2_layer_cache(attn.layer_id)
    conv_states, ssm_states = cache.conv[0], cache.temporal
    if (x.shape[0] * 8192 >= 2**31 or conv_states.numel() >= 2**31 or not conv_states.is_contiguous()
            or not ssm_states.is_contiguous() or conv_states.shape[-1] != WIDTH - 1 or conv_states.shape[-2] != 8192
            or conv_states.dtype != x.dtype or attn.bias is not None or attn.activation != "silu"
            or tuple(attn.conv_weights.shape) != (8192, WIDTH) or not attn.conv_weights.is_contiguous()
            or a.stride(-1) != 1 or b.stride(-1) != 1 or a.shape[0] != x.shape[0]):
        return False
    kernel = getattr(backend.kernel_dispatcher, "extend_kernel", None)
    prefill_fn = getattr(kernel, "_prefill_fn", None)
    ints = (torch.int32, torch.int64)
    if not (type(kernel).__name__ == "FlashInferGDNKernel" and prefill_fn is not None
            and not getattr(kernel, "use_state_pool", True) and not md.num_state_checkpoints
            and ssm_states.dtype == torch.float32 and attn.num_q_heads == 16 and attn.num_k_heads == 16
            and attn.num_v_heads == 32 and md.query_start_loc.dtype in ints and md.mamba_cache_indices.dtype in ints
            and prefix.dtype in ints and prefix.is_cuda and prefix.is_contiguous()):
        return False
    alog, dtb = _gate_params(attn)
    qsl, idx = md.query_start_loc.contiguous(), md.mamba_cache_indices.contiguous()
    slots, cu_seqlens, initial_state, snap = qk_inproj.front_prep(idx, qsl, prefix, ssm_states, conv_states)
    q, k, v, _, g, beta = qk_inproj.inproj_front(x, w_qkvz, attn.conv_weights, conv_states, snap, qsl, idx, prefix,
                                                 a, b, alog, dtb, z)
    final_state = torch.empty_like(initial_state)
    prefill_fn(
        q=q[0, :n], k=k[0, :n], v=v[0, :n], g=g[0, :n], beta=beta[0, :n], scale=None, initial_state=initial_state,
        output_final_state=True, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False,
        output=output[0, :n], output_state=final_state, state_checkpoints=None,
        checkpoint_cu_starts=md.state_checkpoint_cu_starts,
        checkpoint_every_n_tokens=md.state_checkpoint_every_n_tokens)
    ssm_states.index_copy_(0, slots, final_state)
    return True


def _inproj_with_output(x, a, b, z, output, layer_id):
    """Graph-break body of the fused projection: the fused extend on the live batch, else the three GEMMs and the
    previous break body; rows a capture bucket padded come back zero in ``output`` (``z`` covers every row)."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import get_tc_piecewise_forward_context

    context = get_tc_piecewise_forward_context()
    forward_batch, attn = context.forward_batch, context.attention_layers[layer_id]
    w_qkvz = _inproj_weights[layer_id]
    T = x.shape[0]
    real = getattr(forward_batch, "num_token_non_padded_cpu", None)
    n = T if real is None else min(int(real), T)
    if (not forward_batch.forward_mode.is_target_verify() and 0 < n
            and sum(forward_batch.extend_seq_lens_cpu or []) == n
            and _fused_inproj_extend(attn, forward_batch, x, w_qkvz, a, b, output, z)):
        if n < T:
            output[:, n:].zero_()
        return
    mixed_qkv = torch.nn.functional.linear(x, w_qkvz[:8192])
    z.copy_(torch.nn.functional.linear(x, w_qkvz[8192:]))
    _prefill_with_output(mixed_qkv, a, b, output, layer_id)


_inproj_break = eager_on_graph(True)(_inproj_with_output)


def inproj_prefill(attn, forward_batch, x, w_qkvz, a, b):
    """``(core, z)`` of the recurrent block for an extend batch from its input rows; ``None`` when the separate
    projections must run. Under a prefill CUDA graph the work is one graph break."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import get_tc_piecewise_forward_context

    mode = forward_batch.forward_mode
    T = x.shape[0]
    if (not mode.is_extend() or mode.is_target_verify() or T < INPROJ_MIN_ROWS or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.dim() != 2 or x.shape[1] != 2048 or tuple(w_qkvz.shape) != (8192 + Z_OUT, 2048)
            or not w_qkvz.is_contiguous()):
        return None
    output = torch.empty((1, T, attn.num_v_heads, attn.head_v_dim), dtype=x.dtype, device=x.device)
    z = torch.empty((T, Z_OUT), dtype=x.dtype, device=x.device)
    if get_tc_piecewise_forward_context() is not None:
        _inproj_break(x, a, b, z, output, attn.layer_id)
        return output, z
    real = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if (real is not None and real < T) or sum(forward_batch.extend_seq_lens_cpu or []) != T:
        return None
    return (output, z) if _fused_inproj_extend(attn, forward_batch, x, w_qkvz, a, b, output, z) else None
