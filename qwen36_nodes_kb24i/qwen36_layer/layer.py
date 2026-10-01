"""Node entry for ``model.layers.*`` of Qwen3.6-35B-A3B on one H100.

The stock decoder layer keeps its sequence, attention and recurrent state; a
private shallow replica swaps its MoE block for the CUDA kernels of
``moe_fused.py`` (decode and target-verify rows in ``qk_moe_decode.cu``, prefill
and mixed batches in ``qk_moe_prefill.cu`` after stock's router GEMM) and defers
its decode output projection into the fused projection-add-norm of ``proj.py``;
idle batches and deferred finalize run the stock code through the replica. Both
kernels take the router's arithmetic (fp32 accumulate, bf16 logits) and stock's
selection rule, so the audited expert choice is the router's. Original modules and their methods are
untouched and stay available to the node auditor.

Lane DENSE8 (NOT bitwise): a target-verify batch of <= 32 rows projects the recurrent block's input
through a candidate-owned INT8 copy of ``[in_proj_qkvz; in_proj_ba]`` (group-128 fp16 scales along K,
``dense8.py``), built once at the first non-capture eligible call under a memory budget
(``dense8_store.py``); wider batches, and every batch when the copy was not built, run the king's BF16
kernel on stock's weight. Stock's weights are only read.
"""

import copy

import torch

from qwen36_layer.evidence import branch_evidence
from qwen36_layer.gdn import inproj_prefill, linear_attention_prefill, register_inproj
import qk_gated_norm
from qwen36_layer.attn_fused import install as install_attention_verify
from qwen36_layer.gdn_fused import gdn_verify_fused
from qwen36_layer.proj_fused import gdn_in_proj
from qwen36_layer.moe_fused import TOPK, hot_experts, moe_decode_fused, moe_decode_norm_fused, moe_prefill_fused
from qwen36_layer.proj import SPLIT_K, project_add_norm
from qwen36_layer.lt import linear as lt_linear
from qwen36_layer import fp8
from qwen36_layer import pdense

from qwen36_layer.attn_fused import FP8_PROJ_MIN_LAYER, FP8_SKIP_LAYERS
from qwen36_layer import dense8_store
from qwen36_layer.dense8 import gdn_in_proj8
from qwen36_layer import dense8
# The crowned fd0012a2 line's MoE: INT8 expert copies in the served weights' device storage (the bf16 originals
# move to pinned host memory, read through host views by stock), used by decode and prefill (i8x_*.py).
from qwen36_layer import i8x_kernels, i8x_reloc

# The arena serves EAGLE MTP with speculative_num_draft_tokens=4, so a target-verify
# call carries four rows per request and the widest captured decode bucket (48
# requests) is 192 rows. The old 64 was the non-speculative decode width. The cap is
# rounded up to a power of two because the routing kernel indexes its per-expert
# token list with tl.arange(0, MAX_T), and Triton refuses a non-power-of-two arange
# (CompilationError on the first routed call, 2026-09-23).
MAX_TOKENS = 256
EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512
PROJ_IN = 4096
QKVZ_OUT = 12288
QKV_OUT = 8192
V_HEADS = 32
V_HEAD = 128
KING_MARKER = "king_base_b70160ce"
# Lane dense: INT8 o_proj / out_proj receipts, and the layers that keep the BF16 projection (audit margin)
DENSE8_OPROJ_MARKER = "dense8_oproj"
OPROJ8_SKIP_LAYERS = frozenset()
STACK_MARKER = "king_stack"
_partials = {}

# ---- kcombnorm: the next layer's input norm folded into this layer's MoE combine ----------
#
# Every layer still returns exactly stock's (hidden, residual). On a row batch the MoE's last
# launch also writes, into a per-device side slot, what layer i+1's stock input norm
# (flashinfer's CuTe-DSL fused add-RMSNorm, reproduced bit for bit) will compute from that
# pair: the normalized rows and the updated residual. The handoff is keyed on the identity of
# the returned tensors (data_ptr, shape, stride, dtype, device, _version) and holds them, so
# their storage cannot be reused while it exists. Layer i+1's input-norm call takes it only on
# an exact match (the same unchanged tensors, row-batch mode, the norm the slot was computed
# with); every other call runs stock. Each layer call drops any handoff it does not consume, so
# a handoff never outlives the next layer call. Layer 0's own norm (no residual), layer 39
# (no next layer; the model's final norm stays stock) and every non-row-batch path are stock.
COMBNORM_MARKER = "king_combnorm"
NUM_LAYERS = 40
_AMBIGUOUS = object()
_by_id = {}  # layer_id -> the one prepared decoder layer with that id, or _AMBIGUOUS
_hooked = set()  # layer_ids whose replica input norm can take a handoff
_handoffs = {}  # device -> _Handoff, set when a layer returns, consumed or dropped by the next call
_cute_norm = {}  # device -> stock's fused add-RMSNorm is flashinfer's CuTe-DSL kernel


class _Handoff:
    """kcombnorm: layer i's side rows for layer i+1's input norm, keyed on the pair layer i returned."""

    def __init__(self, consumer, norm, h, r, y, r_next):
        self.consumer, self.norm, self.y, self.r_next = consumer, norm, y, r_next
        self.h, self.r = h, r  # held: the keyed storage cannot be freed and reused meanwhile
        self.h_key, self.r_key = _tensor_key(h), _tensor_key(r)


def _tensor_version(t):
    try:
        return t._version
    except RuntimeError:  # an inference tensor tracks no version; the rest of the key still binds it
        return None


def _tensor_key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, _tensor_version(t))


def _stock_norm_is_cute(device):
    """True when stock's GemmaRMSNorm fused add runs the CuTe-DSL kernel this lane reproduces
    (sgl_kernel -> flashinfer.norm.gemma_fused_add_rmsnorm without FLASHINFER_USE_CUDA_NORM)."""
    ok = _cute_norm.get(device)
    if ok is None:
        try:
            import flashinfer.norm
            import sgl_kernel  # noqa: F401 - GemmaRMSNorm's CUDA path imports its norm from here

            ok = torch.version.hip is None and not flashinfer.norm._use_cuda_norm(device)
        except Exception:  # noqa: BLE001 - an unknown dispatch means stock only
            ok = False
        _cute_norm[device] = ok
    return ok


def _register_layer(module):
    """kcombnorm: index the decoder layers by layer_id; a repeated id (a draft layer) disables it."""
    layer_id = getattr(module, "layer_id", None)
    if isinstance(layer_id, int):
        _by_id[layer_id] = module if _by_id.get(layer_id, module) is module else _AMBIGUOUS
    return layer_id


def _next_input_norm(module, layer_id, x):
    """kcombnorm: layer layer_id + 1's stock input norm when this layer may fold it, else None."""
    if not isinstance(layer_id, int) or not 0 <= layer_id < NUM_LAYERS - 1 or _by_id.get(layer_id) is not module:
        return None
    nxt = _by_id.get(layer_id + 1)
    if nxt is None or nxt is _AMBIGUOUS or layer_id + 1 not in _hooked:
        return None
    norm = getattr(nxt, "input_layernorm", None)
    weight = getattr(norm, "weight", None)
    if (type(norm).__name__ != "GemmaRMSNorm" or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (HIDDEN,) or not weight.is_contiguous()
            or weight.device != x.device or getattr(nxt.layer_communicator, "input_layernorm", None) is not norm
            or not _stock_norm_is_cute(x.device)):
        return None
    return norm


def prefill_shape(forward_batch):
    """Port: the prefill chunk's shape class for i8x_kernels' per-class switches ('long' once some request of the batch
    holds more than LONG_SEQ_MIN tokens, else 'short'); host-side lengths only, no device sync."""
    lens = getattr(forward_batch, "seq_lens_cpu", None)
    try:
        return "long" if lens is not None and len(lens) and int(lens.max()) > i8x_kernels.LONG_SEQ_MIN else "short"
    except Exception:
        return "short"


def _next_native_inproj(layer_id, rows, device):
    """True when layer layer_id + 1 is a recurrent layer whose decode in_proj runs qk_route.inproj16 (DENSE8 copy served,
    native row range): the combine-norm then releases it at its start."""
    nxt = _by_id.get(layer_id + 1) if isinstance(layer_id, int) else None
    return (nxt is not None and nxt is not _AMBIGUOUS and hasattr(nxt, "linear_attn")
            and dense8_store._state.get("inproj") in ("aliased", "taken") and rows <= dense8_store.MAX_ROWS_INT8
            and dense8.native_inproj(rows, device))


class _Deferred:
    """Projection input held until the layer's post-attention norm consumes it."""

    def __init__(self, x):
        self.x = x


def _row_batch(forward_batch):
    """True for the engine's per-token calls: plain decode and MTP target-verify.

    ``ForwardMode.is_decode()`` is False for TARGET_VERIFY and ``is_extend()`` is
    True for it, so a predicate written for the non-speculative engine does not
    merely miss the fused path here — it routes every verify call into the
    prefill path, which is built for 8k chunks and, in the recurrent block, for
    the chunked delta rule rather than the ReplaySSM verify writes.
    """
    mode = forward_batch.forward_mode
    return mode.is_decode() or mode.is_target_verify()


def _decode_rows(forward_batch, x):
    return (forward_batch is not None and _row_batch(forward_batch)
            and 0 < x.shape[0] <= MAX_TOKENS and x.dtype == torch.bfloat16)


def _replace_projection(layer, replica, owner, owner_name, projection, proj_name):
    """Defer the decode output projection into the fused add-and-norm kernel.

    ``owner`` holds ``projection`` under ``proj_name``: the layer itself (then
    ``owner_name`` is None) for full attention, ``linear_attn`` for the
    recurrent layers.
    """
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    norm = layer.post_attention_layernorm
    if (not isinstance(norm, GemmaRMSNorm) or tuple(projection.weight.shape) != (HIDDEN, PROJ_IN)
            or projection.weight.dtype != torch.bfloat16 or projection.bias is not None
            or tuple(norm.weight.shape) != (HIDDEN,)):
        raise RuntimeError(
            f"{proj_name} or its norm is not the BF16 Qwen3.6-35B-A3B shape: "
            f"{type(norm).__name__} weight {tuple(projection.weight.shape)} {projection.weight.dtype} "
            f"bias {projection.bias is not None} norm {tuple(norm.weight.shape)}")
    weight, norm_weight, eps = projection.weight, norm.weight, norm.variance_epsilon
    layer_id = int(getattr(layer, "layer_id", -1))
    # Lane dense: the decode o_proj / out_proj from a candidate-owned INT8 copy at <= 32 verify rows (registered only;
    # built with DENSE8's other copies before capture). OPROJ8_SKIP_LAYERS keep the BF16 projection.
    dense8_o = None if layer_id in OPROJ8_SKIP_LAYERS else dense8_store.register("oproj", (weight,))
    partial = _partials.get(weight.device)
    if partial is None:
        partial = _partials[weight.device] = torch.empty(
            (SPLIT_K, MAX_TOKENS, HIDDEN), dtype=torch.float32, device=weight.device)
    batch = [None]

    def project(x, *args, **kwargs):
        if (not args and not kwargs and x.ndim == 2 and x.shape[1] == PROJ_IN
                and x.is_contiguous() and _decode_rows(batch[0], x)):
            return _Deferred(x), None
        if (not args and not kwargs and x.ndim == 2 and x.shape[1] == PROJ_IN and x.dtype == torch.bfloat16
                and getattr(projection, "tp_size", 1) == 1):
            # TP1, no bias: the projection is F.linear; prefill widths skip cuBLAS's memset (lt.py).
            # PRECISION CHANGE (fp8.py) from layer FP8_PROJ_MIN_LAYER on, where the residual dilutes its error.
            # pd38 (pdense.OUT_BF16_LAYERS): a layer whose prefill in_proj runs INT8 there keeps this projection BF16.
            if (layer_id >= FP8_PROJ_MIN_LAYER and layer_id not in FP8_SKIP_LAYERS and layer_id not in pdense.OUT_BF16_LAYERS
                    and fp8.eligible(x, weight)):
                branch_evidence("fp8_proj", x.shape[0])
                return fp8.linear(x.contiguous(), weight), None
            return lt_linear(x.contiguous(), weight), None
        return projection.forward(x, *args, **kwargs)

    private = copy.copy(projection)
    private.forward = project
    if owner_name is None:
        replica._modules[proj_name] = private
    else:
        owner_replica = copy.copy(owner)
        owner_replica._modules = dict(owner._modules)
        owner_replica._modules[proj_name] = private
        replica._modules[owner_name] = owner_replica
        _replace_gdn_prefill(owner, owner_replica)
    communicator = copy.copy(layer.layer_communicator)
    stock_prepare_mlp = type(communicator).prepare_mlp

    def prepare_mlp(hidden_states, residual, forward_batch, cache=None):
        if isinstance(hidden_states, _Deferred):
            x = hidden_states.x
            if (residual is None or residual.shape != (x.shape[0], HIDDEN)
                    or residual.dtype != torch.bfloat16 or not residual.is_contiguous()):
                hidden_states, _ = projection.forward(x)
                return stock_prepare_mlp(communicator, hidden_states, residual, forward_batch, cache)
            copy8 = dense8_store.get(dense8_o, x)
            if copy8 is not None:
                branch_evidence(DENSE8_OPROJ_MARKER, x.shape[0])
            return project_add_norm(x, weight, residual, norm_weight, eps, partial, copy8=copy8)
        return stock_prepare_mlp(communicator, hidden_states, residual, forward_batch, cache)

    communicator.prepare_mlp = prepare_mlp
    # The communicator is a plain attribute, not a registered submodule.
    replica.layer_communicator = communicator
    return batch


def _replace_gdn_prefill(gdn, replica):
    """Prefill through the recurrent block without stock's split copy.

    Stock projects ``[q | k | v | z]`` with one GEMM and then copies it apart with
    a 400 MB-per-chunk kernel. Two GEMMs on row slices of the same weight write
    the ``[q | k | v]`` and ``z`` buffers directly, one kernel replaces stock's
    conv, split copy and gating, and the rest of the block is the stock
    sequence. Decode keeps stock's fused projection-and-conv path.
    """
    w_qkvz, w_ba = gdn.in_proj_qkvz.weight, gdn.in_proj_ba.weight
    if (tuple(w_qkvz.shape) != (QKVZ_OUT, HIDDEN) or tuple(w_ba.shape) != (2 * V_HEADS, HIDDEN)
            or gdn.in_proj_qkvz.bias is not None or gdn.in_proj_ba.bias is not None
            or gdn.num_v_heads != V_HEADS or gdn.head_v_dim != V_HEAD or gdn.attn_tp_size != 1
            or gdn.num_v_heads // gdn.num_k_heads not in (1, 2, 4, 8)):
        raise RuntimeError("this bundle serves the BF16 Qwen3.6-35B-A3B recurrent block at TP1 only")
    w_qkv, w_z = w_qkvz[:QKV_OUT], w_qkvz[QKV_OUT:]
    register_inproj(gdn.layer_id, w_qkvz)
    # DENSE8: registered only (no allocation); the INT8 copy is built at the first eligible call
    dense8_in = dense8_store.register("inproj", (w_qkvz, w_ba))
    # Lane dense: the same entry _replace_projection registers (keyed on the weight), for the L2 prefetch target
    dense8_out = (None if gdn.layer_id in OPROJ8_SKIP_LAYERS
                  else dense8_store.register("oproj", (gdn.out_proj.weight,)))
    fp8_out = (gdn.layer_id >= FP8_PROJ_MIN_LAYER and gdn.layer_id not in FP8_SKIP_LAYERS
               and gdn.layer_id not in pdense.OUT_BF16_LAYERS)
    stock_forward = type(gdn).forward

    def forward(hidden_states, forward_batch):
        mode = forward_batch.forward_mode
        if (not isinstance(hidden_states, torch.Tensor) or hidden_states.shape[0] == 0
                or mode.is_decode() or mode.is_idle()):
            return stock_forward(replica, hidden_states, forward_batch)
        x = hidden_states
        if mode.is_target_verify():
            # Target-verify: one stream stores the projections' split operands,
            # then one CUDA kernel (qk_gdn_verify.cu) runs the conv update, the
            # ReplaySSM recurrence and the gated norm, writing the state stock's
            # launches write (conv state, per-draft windows, ring appends).
            if x.shape[0] > MAX_TOKENS or x.dtype != torch.bfloat16 or not x.is_contiguous():
                return stock_forward(replica, hidden_states, forward_batch)
            copy8 = dense8_store.get(dense8_in, x)
            if copy8 is not None:
                mixed_qkv, z, b, a = gdn_in_proj8(x, copy8[0], copy8[1])
                branch_evidence(dense8_store.MARKER, x.shape[0])
            else:
                mixed_qkv, z, b, a = gdn_in_proj(x, w_qkvz, w_ba, qkv=QKV_OUT, nv=V_HEADS)
            # kgdnpf: the verify kernel L2-prefetches this layer's out_proj weight (the tensor
            # project_add_norm reads next) on SMs its grid leaves idle.
            # Lane dense: when out_proj will read its INT8 copy, those rows are the prefetch target.
            pf = dense8_store.peek(dense8_out, x.shape[0])
            core = gdn_verify_fused(replica.attn, replica.norm, forward_batch, mixed_qkv, z, a, b,
                                    gdn.out_proj.weight if pf is None else pf)
            branch_evidence(KING_MARKER, x.shape[0])
            branch_evidence(STACK_MARKER, x.shape[0])
            output, _ = replica.out_proj(core)
            return output
        else:
            ba = lt_linear(x, w_ba) if x.is_contiguous() else torch.nn.functional.linear(x, w_ba)
            # Views: the CUDA fronts read them strided; other paths copy them apart.
            b = ba[:, :V_HEADS]
            a = ba[:, V_HEADS:]
            # [q | k | v] and z projections with the conv front in one kernel (qk_inproj.cu), else three GEMMs.
            fused = inproj_prefill(replica.attn, forward_batch, x, w_qkvz, a, b)
            if fused is not None:
                core, z = fused
                z = z.view(-1, V_HEADS, V_HEAD)
            else:
                mixed_qkv = torch.nn.functional.linear(x, w_qkv)
                z = torch.nn.functional.linear(x, w_z).view(-1, V_HEADS, V_HEAD)
                core = linear_attention_prefill(replica.attn, forward_batch, mixed_qkv, a, b)
                if core is None:
                    core = replica.attn(forward_batch, mixed_qkv=mixed_qkv, a=a.contiguous(), b=b.contiguous())
        z_shape = z.shape
        core = core.reshape(-1, core.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core.shape != z.shape:
            padded = torch.zeros_like(z)
            padded[: core.shape[0], :] = core
            core = padded
        norm = replica.norm
        if (type(norm).__name__ == "RMSNorm" and getattr(norm, "norm_before_gate", False)
                and getattr(norm, "activation", None) in ("swish", "silu") and getattr(norm, "bias", None) is None
                and getattr(norm, "group_size", None) in (None, V_HEAD) and norm.weight.dtype == torch.bfloat16
                and norm.weight.shape == (V_HEAD,) and norm.weight.is_contiguous() and core.dtype == torch.bfloat16
                and z.dtype == torch.bfloat16 and core.is_contiguous() and z.is_contiguous() and core.shape[-1] == V_HEAD):
            if (fp8_out and z_shape[0] >= fp8.MIN_ROWS and getattr(gdn.out_proj, "tp_size", 1) == 1
                    and tuple(z_shape[1:]) == (V_HEADS, V_HEAD)):
                # PRECISION CHANGE (fp8.py): the gated norm writes out_proj's e4m3 rows and scales directly
                # (qk_gated_norm.cu), bit for bit gated_norm then fp8's quantizer; the bf16 rows skip HBM.
                q8, scale = qk_gated_norm.gated_norm_fp8(core, z, norm.weight, norm.eps)
                branch_evidence("fp8_proj", q8.shape[0])
                return fp8.linear_q(q8, scale, gdn.out_proj.weight)
            # The same gated RMSNorm in CUDA (qk_gated_norm.cu), bit for bit stock's Triton kernel.
            core = qk_gated_norm.gated_norm(core, z, norm.weight, norm.eps).reshape(z_shape)
        else:
            core = norm(core, z).reshape(z_shape)
        core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
        output, _ = replica.out_proj(core)
        return output

    replica.forward = forward


def _check(name, tensor, shape):
    if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        raise RuntimeError(
            f"{name} is {tuple(tensor.shape)} {tensor.dtype}; this bundle serves the "
            f"BF16 Qwen3.6-35B-A3B block {shape} only"
        )


def _i8x_evidence(h8, x):
    if h8 is not None:
        branch_evidence(i8x_reloc.MARKER, x.shape[0])


def _replace_moe(layer, replica, state=None):
    """Give the replica a private MoE block whose decode path is the fused sequence."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.srt.runtime_context import get_parallel

    block = layer.mlp
    if not isinstance(block, Qwen2MoeSparseMoeBlock):
        raise RuntimeError("this bundle serves the Qwen sparse MoE decoder layer only")
    topk = block.topk.topk_config
    if (get_parallel().tp_size != 1 or block.tp_size != 1 or block.enable_shared_expert_fusion
            or block.shared_expert is None or topk.top_k != TOPK or not topk.renormalize
            or topk.scoring_func != "softmax" or topk.correction_bias is not None
            or topk.use_grouped_topk or topk.custom_routing_function is not None
            or topk.num_fused_shared_experts or topk.apply_routed_scaling_factor_on_output
            or tuple(block.gate.weight.shape) != (EXPERTS, HIDDEN)):
        raise RuntimeError("this bundle serves TP1 with a separate shared expert and plain softmax top-8 routing")
    experts, shared = block.experts, block.shared_expert
    w13, w2 = experts.w13_weight, experts.w2_weight
    s13, s2 = shared.gate_up_proj.weight, shared.down_proj.weight
    gate_w = block.shared_expert_gate.weight.reshape(-1)
    router_w = block.gate.weight
    if block.gate.bias is not None:
        raise RuntimeError("this bundle serves a bias-free router")
    _check("gate.weight", router_w, (EXPERTS, HIDDEN))
    _check("experts.w13_weight", w13, (EXPERTS, 2 * INTERMEDIATE, HIDDEN))
    _check("experts.w2_weight", w2, (EXPERTS, HIDDEN, INTERMEDIATE))
    _check("shared_expert.gate_up_proj.weight", s13, (2 * INTERMEDIATE, HIDDEN))
    _check("shared_expert.down_proj.weight", s2, (HIDDEN, INTERMEDIATE))
    _check("shared_expert_gate.weight", gate_w, (HIDDEN,))
    private = copy.copy(block)
    hot = hot_experts(w13.device)
    stock_topk = block.topk
    stock_shared = getattr(block, "_forward_shared_experts", None)
    i8x = i8x_reloc.relocate(getattr(layer, "layer_id", None), w13, w2,
                             servable=callable(stock_topk) and callable(stock_shared))
    copy_ready = i8x_reloc.serve(i8x)
    if copy_ready is not None:
        failed = i8x_kernels.warm(copy_ready, router_w, gate_w, s13, s2)
        if failed is not None:
            i8x_reloc.log_warm_failure(failed)
    if state is not None:
        state["moe_route"] = (i8x, gate_w)

    def topk(x, logits):
        out = stock_topk(x, logits)
        return out.topk_weights, out.topk_ids

    def mlp(hidden_states, forward_batch=None, defer_finalize=False):
        num_tokens = hidden_states.shape[0]
        copy8 = i8x_reloc.serve(i8x)
        if defer_finalize or forward_batch is None or num_tokens == 0 or i8x_reloc.stock_only(i8x):
            return block.forward(hidden_states, forward_batch, defer_finalize=defer_finalize)
        if (not _row_batch(forward_batch) or num_tokens > MAX_TOKENS
                or forward_batch.positions is None or forward_batch.positions.shape[0] != num_tokens):
            if copy8 is not None:
                x = hidden_states.view(-1, HIDDEN).contiguous()
                out = i8x_kernels.prefill(x, router_w, copy8, topk, stock_shared, gate_w, s13, s2,
                                          layer_id=state["layer_id"] if state is not None else None,
                                          shape=prefill_shape(forward_batch), pre=_take_route_pre(state, x))
                branch_evidence(i8x_reloc.MARKER, num_tokens)
                return out
            # Prefill and mixed batches: stock's router GEMM, then our routing, grouped
            # expert GEMMs and combine (qk_moe_prefill.cu).
            return moe_prefill_fused(hidden_states.view(-1, HIDDEN), router_w, gate_w, w13, w2, s13, s2)
        x = hidden_states.view(-1, HIDDEN)
        # kcombnorm: on a row batch whose residual is the one this layer returns, the last
        # launch also computes layer i+1's stock input norm into a side slot; `out` is the same.
        residual = state["residual"] if state is not None else None
        norm = _next_input_norm(layer, state["layer_id"], x) if residual is not None else None
        norm_ok = (norm is not None and residual.shape == x.shape and residual.dtype == torch.bfloat16
                   and residual.is_contiguous())
        if copy8 is not None and i8x_kernels.decode_kind(num_tokens) == "tri8":
            if norm_ok:
                early = _next_native_inproj(state["layer_id"], num_tokens, x.device)
                out, y_next, r_next = i8x_kernels.row_batch(copy8, x.contiguous(), forward_batch.positions, router_w, gate_w,
                                                            s13, s2, norm=(residual, norm.weight, norm.variance_epsilon, early))
                state["fused"] = (norm, residual, out, y_next, r_next)
            else:
                out = i8x_kernels.row_batch(copy8, x.contiguous(), forward_batch.positions, router_w, gate_w, s13, s2)
            branch_evidence(i8x_reloc.MARKER, num_tokens)
            return out
        h8 = None if copy8 is None else i8x_kernels.row_batch_args(copy8)
        if norm_ok:
            fused = moe_decode_norm_fused(x.contiguous(), forward_batch.positions, router_w, gate_w, w13, w2, s13, s2,
                                          hot, residual, norm.weight, norm.variance_epsilon, hot8=h8)
            if fused is not None:
                _i8x_evidence(h8, x)
                state["fused"] = (norm, residual) + fused
                return fused[0]
        # Decode rows a CUDA-graph replay padded in carry position 0; a real decode
        # token always follows at least one prompt token.
        out = moe_decode_fused(x.contiguous(), forward_batch.positions, router_w, gate_w, w13, w2, s13, s2, hot, hot8=h8)
        _i8x_evidence(h8, x)
        return out

    private.forward = mlp
    replica._modules["mlp"] = private


def prepare(module):
    """Compose the replica layer without modifying the served tree."""
    replica = copy.copy(module)
    replica._modules = dict(module._modules)
    # kcombnorm: per-layer call state (the residual this layer returns, the fused side rows,
    # the handoff offered to this layer's input norm) and the layer_id index.
    state = {"layer_id": _register_layer(module), "residual": None, "fused": None, "incoming": None,
             "moe_route": None, "route_pre": None}
    _replace_moe(module, replica, state)
    if hasattr(module, "linear_attn"):
        batch = _replace_projection(
            module, replica, module.linear_attn, "linear_attn", module.linear_attn.out_proj, "out_proj")
    else:
        batch = _replace_projection(module, replica, module, None, module.o_proj, "o_proj")
        install_attention_verify(module, replica)
    _install_combnorm(module, replica, state, batch)
    _install_route_pre_norm(module, replica, state, batch)
    stock_forward = type(module).forward

    def layer(*args, **kwargs):
        batch[0] = kwargs.get("forward_batch", args[3] if len(args) > 3 else None)
        # kcombnorm: this call consumes or drops the previous layer's handoff.
        hidden_in = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
        device = hidden_in.device if isinstance(hidden_in, torch.Tensor) else None
        state["incoming"] = _handoffs.pop(device, None)
        if device is None:
            _handoffs.clear()
        state["residual"] = state["fused"] = state["route_pre"] = None
        try:
            result = stock_forward(replica, *args, **kwargs)
            fused = state["fused"]
            # kcombnorm: offer the side rows to layer i+1 only when this layer returns exactly the
            # pair they were computed from: the combine output and the residual the kernel read.
            if (fused is not None and device is not None and isinstance(result, tuple) and len(result) == 2
                    and result[0] is fused[2] and result[1] is fused[1]):
                norm, residual, out, y_next, r_next = fused
                _handoffs[device] = _Handoff(state["layer_id"] + 1, norm, out, residual, y_next, r_next)
            return result
        finally:
            batch[0] = None
            state["residual"] = state["fused"] = state["incoming"] = state["route_pre"] = None

    return layer


def _install_combnorm(module, replica, state, batch):
    """kcombnorm: record the residual the layer returns, and serve this layer's input norm
    from the previous layer's handoff on an exact match. Both hooks live on the replica's
    private communicator copy; the served module and its communicator are untouched."""
    communicator = replica.layer_communicator
    stock_norm = module.input_layernorm
    if communicator is module.layer_communicator or getattr(communicator, "input_layernorm", None) is not stock_norm:
        return
    inner_prepare_mlp = communicator.prepare_mlp

    def prepare_mlp(*args, **kwargs):
        # The stock layer returns this residual unchanged after the MoE (TP1: postprocess_layer
        # passes the pair through); the layer wrapper re-checks the identity before any handoff.
        prepared = inner_prepare_mlp(*args, **kwargs)
        pair = isinstance(prepared, tuple) and len(prepared) == 2 and isinstance(prepared[1], torch.Tensor)
        state["residual"] = prepared[1] if pair else None
        return prepared

    communicator.prepare_mlp = prepare_mlp
    private_norm = copy.copy(stock_norm)

    def input_norm(*args, **kwargs):
        handoff, state["incoming"] = state["incoming"], None
        # Only prepare_attn's plain fused-add call qualifies: (x, residual[, None]) on a row batch,
        # the same tensors layer i returned, unchanged, and the norm the side rows were computed with.
        if (handoff is not None and not kwargs and len(args) in (2, 3) and (len(args) == 2 or args[2] is None)
                and handoff.consumer == state["layer_id"] and handoff.norm is stock_norm
                and batch[0] is not None and _row_batch(batch[0])
                and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor)
                and _tensor_key(args[0]) == handoff.h_key and _tensor_key(args[1]) == handoff.r_key):
            branch_evidence(COMBNORM_MARKER, args[0].shape[0])
            return handoff.y, handoff.r_next
        return stock_norm(*args, **kwargs)

    private_norm.forward = input_norm
    communicator.input_layernorm = private_norm
    if isinstance(state["layer_id"], int):
        _hooked.add(state["layer_id"])


# ---- kb23n: the prefill post-attention norm with route_i8q's per-token extras ----------
#
# On a prefill chunk whose MoE will route through route_i8q (i8x_kernels.route_pre_ready), the
# post-attention norm runs qk_moe_prefill_i8.add_norm_route instead of stock's flashinfer call: the
# same in-place writes bit for bit (residual <- x + residual, x <- the normed rows), plus the shared
# gate and IMG int8 rows of the normed rows, which route_i8q would otherwise compute by re-reading x.
# They wait in this layer's state keyed on the normed tensor (_tensor_key); the MoE takes them only
# for that exact tensor (route_i8q_pre), the layer call drops them on exit, and every other call
# (decode, short chunks, other paths, any mismatch) runs stock's norm and route_i8q.
def _take_route_pre(state, x):
    entry = state["route_pre"] if state is not None else None
    if entry is None:
        return None
    state["route_pre"] = None
    key, extras = entry
    return extras if key == _tensor_key(x) else None


def _install_route_pre_norm(module, replica, state, batch):
    communicator = replica.layer_communicator
    stock_norm = module.post_attention_layernorm
    if (communicator is module.layer_communicator or state.get("moe_route") is None
            or getattr(communicator, "post_attention_layernorm", None) is not stock_norm
            or type(stock_norm).__name__ != "GemmaRMSNorm"):
        return
    i8x, gate_w = state["moe_route"]
    weight, eps = stock_norm.weight, stock_norm.variance_epsilon
    if (weight.dtype != torch.bfloat16 or tuple(weight.shape) != (HIDDEN,) or not weight.is_contiguous()
            or not gate_w.is_contiguous() or gate_w.data_ptr() % 16 or weight.data_ptr() % 16):
        return
    private_norm = copy.copy(stock_norm)

    def post_norm(*args, **kwargs):
        fb = batch[0]
        if (not kwargs and len(args) == 2 and fb is not None and not _row_batch(fb)
                and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor)):
            x, residual = args
            rows = x.shape[0] if x.dim() == 2 else 0
            if (rows and x.shape == (rows, HIDDEN) and residual.shape == x.shape
                    and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
                    and x.is_contiguous() and residual.is_contiguous()
                    and x.data_ptr() % 16 == 0 and residual.data_ptr() % 16 == 0
                    and x.device == weight.device and residual.device == weight.device
                    and not i8x_reloc.stock_only(i8x) and _stock_norm_is_cute(x.device)
                    and i8x_kernels.route_pre_ready(i8x_reloc.serve(i8x), rows, prefill_shape(fb), state["layer_id"])):
                extras = i8x_kernels.add_norm_route(x, residual, weight, eps, gate_w)
                state["route_pre"] = (_tensor_key(x), extras)
                return x, residual
        return stock_norm(*args, **kwargs)

    private_norm.forward = post_norm
    communicator.post_attention_layernorm = private_norm


def forward(prepared, *args, **kwargs):
    """Return the stock node structure from its prepared computation."""
    return prepared(*args, **kwargs)
