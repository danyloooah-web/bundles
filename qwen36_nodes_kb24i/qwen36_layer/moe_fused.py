import torch
import torch.nn.functional as F
import qk_moe_decode
import qk_moe_prefill
TOPK = 8
_workspaces = {}
_prefill_counters = {}

def _workspace(device):
    ws = _workspaces.get(device)
    if ws is None:
        slots, rows = (qk_moe_decode.SLOTS, qk_moe_decode.MAX_T)
        ws = _workspaces[device] = (torch.empty(rows, 256, dtype=torch.float32, device=device), torch.zeros(qk_moe_decode.COUNTERS, dtype=torch.int32, device=device), torch.empty(rows * 8, dtype=torch.int32, device=device), torch.empty(slots, dtype=torch.float32, device=device), torch.empty(slots, 512, dtype=torch.bfloat16, device=device), torch.empty(slots, 2048, dtype=torch.float32, device=device))
    return ws

def _hot8_args(hot8):
    return (None, None) if hot8 is None else (hot8[0], hot8[1])

def moe_decode_fused(x, positions, router_w, gate_w, w13, w2, s13, s2, hot=None, hot8=None):
    return qk_moe_decode.moe_decode(x, positions, router_w, gate_w, w13, w2, s13, s2, *_workspace(x.device), hot, *_hot8_args(hot8))
_sides = {}

def _side_slot(device, avoid, rows):
    slots = _sides.get(device)
    if slots is None:
        rows_max, hidden = (qk_moe_decode.MAX_T, 2048)
        slots = _sides[device] = [(torch.empty(rows_max, hidden, dtype=torch.bfloat16, device=device), torch.empty(rows_max, hidden, dtype=torch.bfloat16, device=device)) for _ in range(2)]

    def overlaps(a, b):
        a0, b0 = (a.data_ptr(), b.data_ptr())
        return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()
    for y, r in slots:
        if not any((overlaps(buf, t) for buf in (y, r) for t in avoid)):
            return (y[:rows], r[:rows])
    return None

def moe_decode_norm_fused(x, positions, router_w, gate_w, w13, w2, s13, s2, hot, residual, norm_w, eps, hot8=None):
    side = _side_slot(x.device, (residual, x), x.shape[0])
    if side is None:
        return None
    y_next, r_next = side
    out = qk_moe_decode.moe_decode_norm(x, positions, router_w, gate_w, w13, w2, s13, s2, *_workspace(x.device), hot, residual, norm_w, float(eps), y_next, r_next, *_hot8_args(hot8))
    return (out, y_next, r_next)

def hot_experts(device):
    return torch.full((qk_moe_decode.HOT,), -1, dtype=torch.int32, device=device)

def moe_prefill_fused(x, router_w, gate_w, w13, w2, s13, s2):
    logits = F.linear(x, router_w)
    cnt = _prefill_counters.get(x.device)
    if cnt is None:
        cnt = _prefill_counters[x.device] = torch.zeros(qk_moe_prefill.COUNTERS, dtype=torch.int32, device=x.device)
    return qk_moe_prefill.forward(x, logits, gate_w, w13, w2, s13, s2, cnt)[0]
