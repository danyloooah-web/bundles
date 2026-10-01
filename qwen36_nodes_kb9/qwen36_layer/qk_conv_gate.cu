// Prefill front of the recurrent block (gdn.py's conv_split_gate with l2norm and alpha, the served
// SM90 FlashInfer path), bit for bit the Triton _conv_split_gate_kernel it replaces:
//   causal conv over the 8192 [q | k | v] channels, width 4: acc = 0 + p0 + p1 + p2 + p3 in fp32, the tap
//     products p_i = mul.rn.bf16(x_i, w_i) oldest tap first; y = bf16(acc / (1 + ex2.approx(-acc * log2 e)))
//   q / k heads: per head, lane l of the head's warp holding dims 4 l .. 4 l + 3: s = y1 * y1, fma y0, y2, y3;
//     butterfly add over lanes 16 .. 1; y = bf16(y / sqrt.approx.ftz(s + 1e-6))
//   value heads also give g = libdevice expf(-ex2.approx(A_log * log2 e) * softplus(a + dt_bias)), softplus =
//     libdevice logf(1 + ex2.approx(x * log2 e)) up to 20, and beta = f32(bf16(1 / (1 + ex2.approx(-b * log2 e))))
//   the conv state keeps the raw last three input rows of each sequence (or the shifted merge when shorter).
// Every fp32 operation is inline PTX, either in the Triton kernel's form or in a cheaper form proved to give the
// same bits for every fp32 input (exh.cu, all 2^32 values): silu = FMUL.D4(acc, rcp.ftz(fma(ex2.ftz(-acc log2 e),
// 1/4, 1/4))) (ptxas fuses the two multiplies into one rounding), the l2norm divisions = y * rcp.ftz(r), pairs
// rounded with cvt.rn.bf16x2, and the products as fma.rn.bf16x2(x, w, +0), which turns -0 products into +0 so the
// sum needs no leading 0 +. One CTA per (4 heads, 64 tokens, sequence): warp w holds head h0 + w,
// as Triton's layout does, so the l2norm reduction runs in the same order.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

using bf16 = __nv_bfloat16;
constexpr int D = 8192, HEAD = 128, NQ = 16, NK = 16, NV = 32, NH = 4, BT = 32, PF = 8;
constexpr float kLog2eF = 1.44269502162933349609375f;

__device__ __forceinline__ float f_add(float a, float b) { float d; asm("add.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_sub(float a, float b) { float d; asm("sub.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_mul(float a, float b) { float d; asm("mul.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_fma(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float f_div(float a, float b) { float d; asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_ex2(float a) { float d; asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_sqrt_ftz(float a) { float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ uint16_t bf_mul(uint16_t a, uint16_t b) { uint16_t d; asm("mul.rn.bf16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b)); return d; }
// Two mul.rn.bf16 at once: each half rounds exactly as the scalar instruction does.
__device__ __forceinline__ uint32_t bf_mul2(uint32_t a, uint32_t b) { uint32_t d; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
// x * w + 0 per half: mul.rn.bf16x2 except that a -0 product becomes +0.
__device__ __forceinline__ uint32_t bf_fma2z(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0u));
  return d;
}
__device__ __forceinline__ float f_ex2_ftz(float a) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_rcp_ftz(float a) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
// (bf16(lo), bf16(hi)) packed; each half rounds as cvt.rn.bf16.f32 does.
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
// acc / (1 + ex2(-acc log2 e)) as div.full computes it, for every fp32 acc.
__device__ __forceinline__ float silu(float acc) {
  const float r4 = f_rcp_ftz(f_fma(f_ex2_ftz(f_mul(acc, -kLog2eF)), 0.25f, 0.25f));
  return f_mul(f_mul(acc, r4), 0.25f);
}
__device__ __forceinline__ float lo2f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi2f(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float v) { uint16_t d; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(v)); return d; }
constexpr float kLog2e = 1.44269502162933349609375f;  // 0f3FB8AA3B
// exp(0 - x) as the Triton kernel computes it: ex2.approx(mul(0 - x, log2 e)). 0 - x is -x exactly (both
// zeros go to +0 or -0, and ex2 of either is 1), and mul(-x, c) = mul(x, -c) under round-to-nearest.
__device__ __forceinline__ float t_exp_neg(float x) { return f_ex2(f_mul(x, -kLog2e)); }

// Triton tl.exp(x): ex2.approx.f32(x * log2 e).
__device__ __forceinline__ float t_exp(float x) { return f_ex2(f_mul(x, kLog2e)); }

// libdevice __nv_logf as the Triton kernel inlines it (same instructions and constants).
__device__ __forceinline__ float nv_logf(float v) {
  float r;
  asm("{\n"
      ".reg .pred pd, pinf, pz;\n"
      ".reg .f32 m, ea, e, f, p, t;\n"
      ".reg .b32 bi, ex, mi;\n"
      "setp.lt.f32 pd, %1, 0f00800000;\n"
      "mul.f32 t, %1, 0f4B000000;\n"
      "selp.f32 m, t, %1, pd;\n"
      "selp.f32 ea, 0fC1B80000, 0f00000000, pd;\n"
      "mov.b32 bi, m;\n"
      "add.s32 ex, bi, -1059760811;\n"
      "and.b32 ex, ex, -8388608;\n"
      "sub.s32 mi, bi, ex;\n"
      "cvt.rn.f32.s32 e, ex;\n"
      "fma.rn.ftz.f32 e, e, 0f34000000, ea;\n"
      "mov.b32 f, mi;\n"
      "add.f32 f, f, 0fBF800000;\n"
      "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n"
      "mul.f32 p, f, p;\n"
      "fma.rn.ftz.f32 p, p, f, f;\n"
      "fma.rn.ftz.f32 %0, e, 0f3F317218, p;\n"
      "setp.lt.u32 pinf, bi, 2139095040;\n"
      "@!pinf fma.rn.ftz.f32 %0, m, 0f7F800000, 0f7F800000;\n"
      "setp.eq.f32 pz, m, 0f00000000;\n"
      "@pz mov.f32 %0, 0fFF800000;\n"
      "}\n"
      : "=f"(r)
      : "f"(v));
  return r;
}

// libdevice __nv_expf as the Triton kernel inlines it.
__device__ __forceinline__ float nv_expf(float x) {
  float r;
  asm("{\n"
      ".reg .f32 t, j, k, q, e;\n"
      ".reg .b32 s;\n"
      "fma.rn.ftz.f32 t, %1, 0f3BBB989D, 0f3F000000;\n"
      "cvt.ftz.sat.f32.f32 t, t;\n"
      "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n"
      "add.f32 k, j, 0fCB40007F;\n"
      "neg.f32 k, k;\n"
      "fma.rn.ftz.f32 q, %1, 0f3FB8AA3B, k;\n"
      "fma.rn.ftz.f32 q, %1, 0f32A57060, q;\n"
      "mov.b32 s, j;\n"
      "shl.b32 s, s, 23;\n"
      "ex2.approx.ftz.f32 e, q;\n"
      "mov.b32 j, s;\n"
      "mul.f32 %0, e, j;\n"
      "}\n"
      : "=f"(r)
      : "f"(x));
  return r;
}

struct Args {
  const bf16* x;        // [T, D]
  const bf16* w;        // [D, 4]
  bf16* cs;             // [slots, D, 3]
  const void* qsl;      // [B + 1]
  const void* idx;      // [B] conv-state slot, < 0: none
  const void* prefix;   // [B] > 0: start from the conv state
  int wide;             // bit 0 / 1 / 2: qsl / idx / prefix are int64 (else int32)
  const bf16* a;        // [T, NV], rows `ab_stride` apart (a view of in_proj_ba's output)
  const bf16* b;        // [T, NV], rows `ab_stride` apart
  int ab_stride;
  const float* alog;    // [NV] (fp32)
  const float* dtb;     // [NV] (fp32)
  bf16* q;              // [T, NQ, HEAD]
  bf16* k;              // [T, NK, HEAD]
  bf16* v;              // [T, NV, HEAD]
  float* g;             // [T, NV]
  float* beta;          // [T, NV]
};

__device__ __forceinline__ int ld_meta(const void* ptr, int i, bool is64) {
  return is64 ? static_cast<int>(static_cast<const int64_t*>(ptr)[i]) : static_cast<const int*>(ptr)[i];
}

__global__ void __launch_bounds__(128) conv_gate_kernel(const Args p) {
  const int h0 = blockIdx.x * NH, t0 = blockIdx.y * BT, seq = blockIdx.z;
  const int start = ld_meta(p.qsl, seq, p.wide & 1), end = ld_meta(p.qsl, seq + 1, p.wide & 1), L = end - start;
  if (t0 >= L) return;
  const int n = min(BT, L - t0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c = (h0 + warp) * HEAD + 4 * lane;  // this thread's 4 channels
  const int slot = ld_meta(p.idx, seq, p.wide & 2);
  const bool hinit = ld_meta(p.prefix, seq, p.wide & 4) > 0;
  // Taps: w[c + j][i].
  uint16_t w[4][4];
  {
    const uint2* wp = reinterpret_cast<const uint2*>(p.w + static_cast<int64_t>(c) * 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint2 u = wp[j];
      w[j][0] = u.x & 0xffff, w[j][1] = u.x >> 16, w[j][2] = u.y & 0xffff, w[j][3] = u.y >> 16;
    }
  }
  // Tap i of channels (0, 1) and (2, 3) as bf16x2 pairs.
  uint32_t wp2[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    wp2[i][0] = static_cast<uint32_t>(w[0][i]) | (static_cast<uint32_t>(w[1][i]) << 16);
    wp2[i][1] = static_cast<uint32_t>(w[2][i]) | (static_cast<uint32_t>(w[3][i]) << 16);
  }
  // The three rows before this block (column 2 the most recent): the sequence's own, else its conv state.
  uint16_t col[3][4];
  if (t0 == 0) {
    const bool use = hinit && slot >= 0;
    const bf16* sb = p.cs + static_cast<int64_t>(use ? slot : 0) * (D * 3) + static_cast<int64_t>(c) * 3;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 3; ++i) col[i][j] = use ? __bfloat16_as_ushort(sb[j * 3 + i]) : 0;
  } else {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const uint2 u = *reinterpret_cast<const uint2*>(p.x + static_cast<int64_t>(start + t0 - 3 + i) * D + c);
      col[i][0] = u.x & 0xffff, col[i][1] = u.x >> 16, col[i][2] = u.y & 0xffff, col[i][3] = u.y >> 16;
    }
  }
  const bool is_q = h0 < NQ, is_k = !is_q && h0 < NQ + NK;
  bf16* dst = is_q ? p.q + h0 * HEAD : is_k ? p.k + (h0 - NQ) * HEAD : p.v + (h0 - NQ - NK) * HEAD;
  const int dst_heads = is_q ? NQ : is_k ? NK : NV;
  const int doff = warp * HEAD + 4 * lane;
  // The window as bf16x2 pairs (channels 0, 1 and 2, 3), shifted one row per token.
  uint32_t cp[3][2];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    cp[i][0] = static_cast<uint32_t>(col[i][0]) | (static_cast<uint32_t>(col[i][1]) << 16);
    cp[i][1] = static_cast<uint32_t>(col[i][2]) | (static_cast<uint32_t>(col[i][3]) << 16);
  }

  const bf16* xp = p.x + static_cast<int64_t>(start + t0) * D + c;
  bf16* op = dst + static_cast<int64_t>(start + t0) * dst_heads * HEAD + doff;
  const int ostride = dst_heads * HEAD;
  // Row ring: slot u holds row i0 + u; its next row is requested as soon as it is taken.
  uint2 xr[PF];
#pragma unroll
  for (int u = 0; u < PF; ++u)
    xr[u] = u < n ? *reinterpret_cast<const uint2*>(xp + static_cast<int64_t>(u) * D) : make_uint2(0, 0);
  for (int i0 = 0; i0 < n; i0 += PF) {
#pragma unroll
    for (int u = 0; u < PF; ++u) {
      if (i0 + u >= n) break;
      const uint32_t x2[2] = {xr[u].x, xr[u].y};
      const int nx = i0 + PF + u;
      xr[u] = nx < n ? *reinterpret_cast<const uint2*>(xp + static_cast<int64_t>(nx) * D) : make_uint2(0, 0);
      float y[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t q0 = bf_fma2z(cp[0][h], wp2[0][h]), q1 = bf_fma2z(cp[1][h], wp2[1][h]);
        const uint32_t q2 = bf_fma2z(cp[2][h], wp2[2][h]), q3 = bf_fma2z(x2[h], wp2[3][h]);
        float lo = f_add(lo2f(q0), lo2f(q1)), hi = f_add(hi2f(q0), hi2f(q1));
        lo = f_add(lo, lo2f(q2)), hi = f_add(hi, hi2f(q2));
        lo = f_add(lo, lo2f(q3)), hi = f_add(hi, hi2f(q3));
        y[2 * h] = silu(lo);
        y[2 * h + 1] = silu(hi);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) cp[0][h] = cp[1][h], cp[1][h] = cp[2][h], cp[2][h] = x2[h];
      uint2 o = make_uint2(f2bf2(y[0], y[1]), f2bf2(y[2], y[3]));
      if (is_q || is_k) {
        const float z0 = lo2f(o.x), z1 = hi2f(o.x), z2 = lo2f(o.y), z3 = hi2f(o.y);
        float s = f_mul(z1, z1);
        s = f_fma(z0, z0, s);
        s = f_fma(z2, z2, s);
        s = f_fma(z3, z3, s);
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) s = f_add(s, __shfl_xor_sync(0xffffffffu, s, off));
        const float rr = f_rcp_ftz(f_sqrt_ftz(f_add(s, 1e-6f)));
        o = make_uint2(f2bf2(f_mul(z0, rr), f_mul(z1, rr)), f2bf2(f_mul(z2, rr), f_mul(z3, rr)));
      }
      *reinterpret_cast<uint2*>(op + static_cast<int64_t>(i0 + u) * ostride) = o;
    }
  }

  if (h0 >= NQ + NK) {
    // g and beta of this block's value heads: one (token, head) per thread per pass.
    for (int e = threadIdx.x; e < n * NH; e += blockDim.x) {
      const int i = e / NH, hh = e % NH, hv = h0 - NQ - NK + hh, t = start + t0 + i;
      const float av = bf2f(__bfloat16_as_ushort(p.a[static_cast<int64_t>(t) * p.ab_stride + hv]));
      const float bv = bf2f(__bfloat16_as_ushort(p.b[static_cast<int64_t>(t) * p.ab_stride + hv]));
      const float xg = f_add(av, p.dtb[hv]);
      const float lg = nv_logf(f_add(t_exp(xg), 1.f));
      const float sp = xg <= 20.f ? lg : xg;
      const float gv = f_mul(f_sub(0.f, t_exp(p.alog[hv])), sp);
      p.g[static_cast<int64_t>(t) * NV + hv] = nv_expf(gv);
      const float sg = f_div(1.f, f_add(t_exp(f_sub(0.f, bv)), 1.f));
      p.beta[static_cast<int64_t>(t) * NV + hv] = bf2f(f2bf(sg));
    }
  }

  if (t0 == 0 && slot >= 0) {
    // New conv state: the raw last three input rows, or the old state shifted by the sequence length when
    // shorter. This thread read its own channels' state above and now writes them.
    bf16* sb = p.cs + static_cast<int64_t>(slot) * (D * 3) + static_cast<int64_t>(c) * 3;
    uint16_t nv[3][4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int tt = end - 3 + i;
      if (tt >= start) {
        const uint2 u = *reinterpret_cast<const uint2*>(p.x + static_cast<int64_t>(tt) * D + c);
        nv[i][0] = u.x & 0xffff, nv[i][1] = u.x >> 16, nv[i][2] = u.y & 0xffff, nv[i][3] = u.y >> 16;
      } else {
        const int oc = i + L;
#pragma unroll
        for (int j = 0; j < 4; ++j) nv[i][j] = hinit && oc < 3 ? __bfloat16_as_ushort(sb[j * 3 + oc]) : 0;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 3; ++i) sb[j * 3 + i] = __ushort_as_bfloat16(nv[i][j]);
  }
}

// The FlashInfer extend's inputs around the recurrence in one launch (stock: a compare, a where, two int64
// casts and an index gather): slots = cache index, or the last state slot for a sequence without one (int64),
// cu_seqlens as int64, and initial_state = ssm_states[slots] (a copy).
__global__ void __launch_bounds__(256) gdn_prep_kernel(const void* idx, int idx64, const void* qsl, int qsl64, int B,
                                                       int64_t last_slot, const float4* __restrict__ states,
                                                       int64_t per_slot4, int64_t* __restrict__ slots_out,
                                                       int64_t* __restrict__ cu_out, float4* __restrict__ init) {
  if (blockIdx.x == 0) {
    for (int b = threadIdx.x; b <= B; b += blockDim.x) {
      cu_out[b] = ld_meta(qsl, b, qsl64);
      if (b < B) {
        const int s = ld_meta(idx, b, idx64);
        slots_out[b] = s >= 0 ? static_cast<int64_t>(s) : last_slot;
      }
    }
  }
  const int64_t total = static_cast<int64_t>(B) * per_slot4;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(i / per_slot4);
    const int s = ld_meta(idx, b, idx64);
    const int64_t slot = s >= 0 ? static_cast<int64_t>(s) : last_slot;
    init[i] = __ldcs(states + slot * per_slot4 + (i - static_cast<int64_t>(b) * per_slot4));
  }
}

}  // namespace

// (slots int64 [B], cu_seqlens int64 [B + 1], initial_state [B, ...] = ssm_states[slots]) for the extend.
std::vector<torch::Tensor> gdn_prep(torch::Tensor idx, torch::Tensor qsl, torch::Tensor ssm_states) {
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && (idx.scalar_type() == at::kInt || idx.scalar_type() == at::kLong), "idx");
  TORCH_CHECK(qsl.is_cuda() && qsl.is_contiguous() && (qsl.scalar_type() == at::kInt || qsl.scalar_type() == at::kLong), "qsl");
  TORCH_CHECK(ssm_states.is_cuda() && ssm_states.is_contiguous() && ssm_states.scalar_type() == at::kFloat, "ssm_states fp32");
  const int64_t B = idx.numel(), per_slot = ssm_states.numel() / ssm_states.size(0);
  TORCH_CHECK(qsl.numel() == B + 1 && per_slot % 4 == 0, "shapes");
  const at::cuda::CUDAGuard guard(ssm_states.device());
  auto slots = torch::empty({B}, idx.options().dtype(at::kLong));
  auto cu = torch::empty({B + 1}, idx.options().dtype(at::kLong));
  std::vector<int64_t> shape(ssm_states.sizes().begin(), ssm_states.sizes().end());
  shape[0] = B;
  auto init = torch::empty(shape, ssm_states.options());
  const int64_t total4 = B * (per_slot / 4);
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>((total4 + 255) / 256, 1024)));
  gdn_prep_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      idx.data_ptr(), idx.scalar_type() == at::kLong, qsl.data_ptr(), qsl.scalar_type() == at::kLong, static_cast<int>(B),
      ssm_states.size(0) - 1, reinterpret_cast<const float4*>(ssm_states.data_ptr<float>()), per_slot / 4,
      slots.data_ptr<int64_t>(), cu.data_ptr<int64_t>(), reinterpret_cast<float4*>(init.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {slots, cu, init};
}

// conv_split_gate(l2norm=True, alpha=True): returns (q, k, v, g, beta); updates conv_states in place.
std::vector<torch::Tensor> conv_gate(torch::Tensor x, torch::Tensor w, torch::Tensor cs, torch::Tensor qsl,
                                     torch::Tensor idx, torch::Tensor prefix, torch::Tensor a, torch::Tensor b,
                                     torch::Tensor alog, torch::Tensor dtb, int64_t max_len) {
  const int64_t T = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == D, "x");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == D * 4, "conv weights");
  TORCH_CHECK(cs.scalar_type() == at::kBFloat16 && cs.is_contiguous() && cs.size(-1) == 3 && cs.size(-2) == D, "conv state");
  auto meta = [](const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong), name,
                " must be a contiguous CUDA int32 / int64 tensor");
    return t.scalar_type() == at::kLong;
  };
  const int wide = (meta(qsl, "query_start_loc") ? 1 : 0) | (meta(idx, "cache indices") ? 2 : 0) |
                   (meta(prefix, "prefix lens") ? 4 : 0);
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && a.dim() == 2 && b.dim() == 2 &&
                  a.size(1) == NV && b.size(1) == NV && a.stride(1) == 1 && b.stride(1) == 1 && a.stride(0) == b.stride(0),
              "a, b: [T, 32] rows with one stride");
  TORCH_CHECK(alog.scalar_type() == at::kFloat && dtb.scalar_type() == at::kFloat, "A_log / dt_bias as fp32");
  const int64_t B = qsl.size(0) - 1;
  const at::cuda::CUDAGuard guard(x.device());
  auto q = torch::empty({1, T, NQ, HEAD}, x.options());
  auto k = torch::empty({1, T, NK, HEAD}, x.options());
  auto v = torch::empty({1, T, NV, HEAD}, x.options());
  auto g = torch::empty({1, T, NV}, x.options().dtype(at::kFloat));
  auto beta = torch::empty({1, T, NV}, x.options().dtype(at::kFloat));
  if (T == 0 || B == 0) return {q, k, v, g, beta};
  Args p{reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<const bf16*>(w.data_ptr()),
         reinterpret_cast<bf16*>(cs.data_ptr()), qsl.data_ptr(), idx.data_ptr(), prefix.data_ptr(), wide,
         reinterpret_cast<const bf16*>(a.data_ptr()), reinterpret_cast<const bf16*>(b.data_ptr()),
         static_cast<int>(a.stride(0)), alog.data_ptr<float>(),
         dtb.data_ptr<float>(), reinterpret_cast<bf16*>(q.data_ptr()), reinterpret_cast<bf16*>(k.data_ptr()),
         reinterpret_cast<bf16*>(v.data_ptr()), g.data_ptr<float>(), beta.data_ptr<float>()};
  const dim3 grid(D / (NH * HEAD), static_cast<unsigned>((max_len + BT - 1) / BT), static_cast<unsigned>(B));
  conv_gate_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, v, g, beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_prep", &gdn_prep, "GDN extend: int64 slots and cu_seqlens, initial states gathered");
  m.def("conv_gate", &conv_gate, "GDN prefill causal conv + q/k/v split + l2norm + gating (gdn.py conv_split_gate)");
}
