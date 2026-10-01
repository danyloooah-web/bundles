// The recurrent block's gated RMSNorm for prefill (sglang layernorm_gated RMSNorm, norm_before_gate, swish
// gate, group = row of 128), bit for bit the Triton _layer_norm_fwd_1pass_kernel it replaces (BLOCK_N 128,
// 4 rows per program, one warp, layout [1, 8] x [2, 16]: a half warp per row, lane l % 16 holding columns
// 8 (l % 16) .. + 7):
//   x = f32(core); s = x1 * x1, fma x0, x2 .. x7; butterfly add over lanes 8, 4, 2, 1; var = div.full(s, 128)
//   rstd = rsqrt.approx.ftz(var + eps); y = (x * rstd) * w; y = y * (z * sigmoid(z)),
//   sigmoid(z) = div.full(1, ex2.approx(-z * log2 e) + 1); out = bf16(y)
// Every fp32 operation is inline PTX in the Triton kernel's form, so --use_fast_math cannot change it. A half
// warp per row, RB row pairs in flight per warp; the per-row rstd the Triton kernel also stores is never read
// and is not written.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int D = 128, WARPS = 8, RB = 4;  // RB row pairs per warp per pass
constexpr float kLog2e = 1.44269502162933349609375f;

__device__ __forceinline__ float f_add(float a, float b) { float d; asm("add.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_sub(float a, float b) { float d; asm("sub.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_mul(float a, float b) { float d; asm("mul.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_fma(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float f_div(float a, float b) { float d; asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_ex2(float a) { float d; asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_rsqrt_ftz(float a) { float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ uint16_t f2bf(float v) { uint16_t d; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(v)); return d; }
__device__ __forceinline__ float lo2f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi2f(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__global__ void __launch_bounds__(32 * WARPS) gated_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                                                const __nv_bfloat16* __restrict__ z,
                                                                const __nv_bfloat16* __restrict__ w,
                                                                __nv_bfloat16* __restrict__ out, int64_t M, float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, half = lane >> 4, c = 8 * (lane & 15);
  float wf[8];
  {
    const uint4 wv = *reinterpret_cast<const uint4*>(w + c);
    wf[0] = lo2f(wv.x), wf[1] = hi2f(wv.x), wf[2] = lo2f(wv.y), wf[3] = hi2f(wv.y);
    wf[4] = lo2f(wv.z), wf[5] = hi2f(wv.z), wf[6] = lo2f(wv.w), wf[7] = hi2f(wv.w);
  }
  const int64_t warps_total = static_cast<int64_t>(gridDim.x) * WARPS;
  for (int64_t r0 = (static_cast<int64_t>(blockIdx.x) * WARPS + warp) * (2 * RB); r0 < M; r0 += warps_total * 2 * RB) {
    uint4 xv[RB], zv[RB];
#pragma unroll
    for (int u = 0; u < RB; ++u) {
      const int64_t r = r0 + 2 * u + half;
      xv[u] = r < M ? __ldcs(reinterpret_cast<const uint4*>(x + r * D + c)) : make_uint4(0, 0, 0, 0);
      zv[u] = r < M ? __ldcs(reinterpret_cast<const uint4*>(z + r * D + c)) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < RB; ++u) {
      const int64_t r = r0 + 2 * u + half;
      const float xs[8] = {lo2f(xv[u].x), hi2f(xv[u].x), lo2f(xv[u].y), hi2f(xv[u].y),
                           lo2f(xv[u].z), hi2f(xv[u].z), lo2f(xv[u].w), hi2f(xv[u].w)};
      float s = f_mul(xs[1], xs[1]);
      s = f_fma(xs[0], xs[0], s);
#pragma unroll
      for (int j = 2; j < 8; ++j) s = f_fma(xs[j], xs[j], s);
#pragma unroll
      for (int o = 8; o >= 1; o >>= 1) s = f_add(s, __shfl_xor_sync(0xffffffffu, s, o));
      if (r >= M) continue;
      const float rstd = f_rsqrt_ftz(f_add(eps, f_div(s, 128.f)));
      const float zs[8] = {lo2f(zv[u].x), hi2f(zv[u].x), lo2f(zv[u].y), hi2f(zv[u].y),
                           lo2f(zv[u].z), hi2f(zv[u].z), lo2f(zv[u].w), hi2f(zv[u].w)};
      uint16_t y[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float yj = f_mul(f_mul(xs[j], rstd), wf[j]);
        const float sg = f_div(1.f, f_add(f_ex2(f_mul(f_sub(0.f, zs[j]), kLog2e)), 1.f));
        y[j] = f2bf(f_mul(yj, f_mul(zs[j], sg)));
      }
      *reinterpret_cast<uint4*>(out + r * D + c) =
          make_uint4(static_cast<uint32_t>(y[0]) | (static_cast<uint32_t>(y[1]) << 16),
                     static_cast<uint32_t>(y[2]) | (static_cast<uint32_t>(y[3]) << 16),
                     static_cast<uint32_t>(y[4]) | (static_cast<uint32_t>(y[5]) << 16),
                     static_cast<uint32_t>(y[6]) | (static_cast<uint32_t>(y[7]) << 16));
    }
  }
}

// gated_norm_fp8: the same per-row arithmetic for all HEADS rows of one token in one CTA (16 half warps x 2 passes),
// then fp8.py's Triton _rowwise_fp8 on the token's bf16 row of HEADS x 128: amax = max |y|, scale =
// max(div.full(amax, 448), 1e-12), inv = div.full(1, scale), q = cvt.rn.satfinite.e4m3(y * inv). The bf16 row
// never goes to HBM.
constexpr int QHEADS = 32, QPASS = QHEADS / (2 * WARPS);

__device__ __forceinline__ float f_max(float a, float b) { float d; asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ uint16_t f2e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}

__global__ void __launch_bounds__(32 * WARPS) gated_norm_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                                                    const __nv_bfloat16* __restrict__ z,
                                                                    const __nv_bfloat16* __restrict__ w,
                                                                    uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                    float eps) {
  __shared__ float red[WARPS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, half = lane >> 4, c = 8 * (lane & 15);
  const int64_t t = blockIdx.x;
  float wf[8];
  {
    const uint4 wv = *reinterpret_cast<const uint4*>(w + c);
    wf[0] = lo2f(wv.x), wf[1] = hi2f(wv.x), wf[2] = lo2f(wv.y), wf[3] = hi2f(wv.y);
    wf[4] = lo2f(wv.z), wf[5] = hi2f(wv.z), wf[6] = lo2f(wv.w), wf[7] = hi2f(wv.w);
  }
  uint4 xv[QPASS], zv[QPASS];
#pragma unroll
  for (int p = 0; p < QPASS; ++p) {
    const int64_t r = t * QHEADS + p * 2 * WARPS + 2 * warp + half;
    xv[p] = __ldcs(reinterpret_cast<const uint4*>(x + r * D + c));
    zv[p] = __ldcs(reinterpret_cast<const uint4*>(z + r * D + c));
  }
  float yv[QPASS][8];
  float amax = 0.f;
#pragma unroll
  for (int p = 0; p < QPASS; ++p) {
    const float xs[8] = {lo2f(xv[p].x), hi2f(xv[p].x), lo2f(xv[p].y), hi2f(xv[p].y),
                         lo2f(xv[p].z), hi2f(xv[p].z), lo2f(xv[p].w), hi2f(xv[p].w)};
    float s = f_mul(xs[1], xs[1]);
    s = f_fma(xs[0], xs[0], s);
#pragma unroll
    for (int j = 2; j < 8; ++j) s = f_fma(xs[j], xs[j], s);
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) s = f_add(s, __shfl_xor_sync(0xffffffffu, s, o));
    const float rstd = f_rsqrt_ftz(f_add(eps, f_div(s, 128.f)));
    const float zs[8] = {lo2f(zv[p].x), hi2f(zv[p].x), lo2f(zv[p].y), hi2f(zv[p].y),
                         lo2f(zv[p].z), hi2f(zv[p].z), lo2f(zv[p].w), hi2f(zv[p].w)};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float yj = f_mul(f_mul(xs[j], rstd), wf[j]);
      const float sg = f_div(1.f, f_add(f_ex2(f_mul(f_sub(0.f, zs[j]), kLog2e)), 1.f));
      yv[p][j] = __uint_as_float(static_cast<uint32_t>(f2bf(f_mul(yj, f_mul(zs[j], sg)))) << 16);
      amax = f_max(amax, __uint_as_float(__float_as_uint(yv[p][j]) & 0x7fffffffu));
    }
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) amax = f_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (lane == 0) red[warp] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int i = 1; i < WARPS; ++i) amax = f_max(amax, red[i]);
  const float sc = f_max(f_div(amax, 448.f), 1e-12f);
  const float inv = f_div(1.f, sc);
#pragma unroll
  for (int p = 0; p < QPASS; ++p) {
    const int64_t r = t * QHEADS + p * 2 * WARPS + 2 * warp + half;
    uint16_t b[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) b[j] = f2e4m3x2(f_mul(yv[p][2 * j], inv), f_mul(yv[p][2 * j + 1], inv));
    __stcs(reinterpret_cast<uint2*>(q + r * D + c),
           make_uint2(static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 16),
                      static_cast<uint32_t>(b[2]) | (static_cast<uint32_t>(b[3]) << 16)));
  }
  if (threadIdx.x == 0) scale[t] = sc;
}

// gate_mul_fp8: the attention output gate for o_proj, stock's _fused_sigmoid_mul_kernel (sglang elementwise.py:
// attn * div.full(1, ex2.approx(-g * log2 e) + 1), bf16) for one token's GHEADS x GDIM row in one CTA, then
// _rowwise_fp8 as above. The gate is head h's second half of the [q | gate] block, qkv[t, h * 2 GDIM + GDIM + d].
constexpr int GHEADS = 16, GDIM = 256, GCHUNKS = GHEADS * GDIM / 8 / (32 * WARPS);

__global__ void __launch_bounds__(32 * WARPS) gate_mul_fp8_kernel(const __nv_bfloat16* __restrict__ attn,
                                                                  const __nv_bfloat16* __restrict__ qkv, int64_t ld,
                                                                  uint8_t* __restrict__ q, float* __restrict__ scale) {
  __shared__ float red[WARPS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t t = blockIdx.x;
  uint4 av[GCHUNKS], gv[GCHUNKS];
#pragma unroll
  for (int p = 0; p < GCHUNKS; ++p) {
    const int e = 8 * (p * 32 * WARPS + threadIdx.x), h = e / GDIM;
    av[p] = __ldcs(reinterpret_cast<const uint4*>(attn + t * GHEADS * GDIM + e));
    gv[p] = __ldcs(reinterpret_cast<const uint4*>(qkv + t * ld + h * 2 * GDIM + GDIM + (e - h * GDIM)));
  }
  float yv[GCHUNKS][8];
  float amax = 0.f;
#pragma unroll
  for (int p = 0; p < GCHUNKS; ++p) {
    const float as[8] = {lo2f(av[p].x), hi2f(av[p].x), lo2f(av[p].y), hi2f(av[p].y),
                         lo2f(av[p].z), hi2f(av[p].z), lo2f(av[p].w), hi2f(av[p].w)};
    const float gs[8] = {lo2f(gv[p].x), hi2f(gv[p].x), lo2f(gv[p].y), hi2f(gv[p].y),
                         lo2f(gv[p].z), hi2f(gv[p].z), lo2f(gv[p].w), hi2f(gv[p].w)};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float sg = f_div(1.f, f_add(f_ex2(f_mul(f_sub(0.f, gs[j]), kLog2e)), 1.f));
      yv[p][j] = __uint_as_float(static_cast<uint32_t>(f2bf(f_mul(sg, as[j]))) << 16);
      amax = f_max(amax, __uint_as_float(__float_as_uint(yv[p][j]) & 0x7fffffffu));
    }
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) amax = f_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (lane == 0) red[warp] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int i = 1; i < WARPS; ++i) amax = f_max(amax, red[i]);
  const float sc = f_max(f_div(amax, 448.f), 1e-12f);
  const float inv = f_div(1.f, sc);
#pragma unroll
  for (int p = 0; p < GCHUNKS; ++p) {
    const int e = 8 * (p * 32 * WARPS + threadIdx.x);
    uint16_t b[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) b[j] = f2e4m3x2(f_mul(yv[p][2 * j], inv), f_mul(yv[p][2 * j + 1], inv));
    __stcs(reinterpret_cast<uint2*>(q + t * GHEADS * GDIM + e),
           make_uint2(static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 16),
                      static_cast<uint32_t>(b[2]) | (static_cast<uint32_t>(b[3]) << 16)));
  }
  if (threadIdx.x == 0) scale[t] = sc;
}

}  // namespace

// out = RMSNorm(core) * w * silu(z) per 128-wide row (norm_before_gate), bf16.
torch::Tensor gated_norm(torch::Tensor core, torch::Tensor z, torch::Tensor w, double eps) {
  TORCH_CHECK(core.is_cuda() && core.scalar_type() == at::kBFloat16 && core.is_contiguous() && core.dim() == 2 &&
                  core.size(1) == D, "core [M, 128] bf16");
  TORCH_CHECK(z.scalar_type() == at::kBFloat16 && z.is_contiguous() && z.sizes() == core.sizes(), "z like core");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == D, "weight [128] bf16");
  const int64_t M = core.size(0);
  const at::cuda::CUDAGuard guard(core.device());
  auto out = torch::empty_like(core);
  if (M == 0) return out;
  const int64_t rows_per_cta = static_cast<int64_t>(WARPS) * 2 * RB;
  int sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, core.get_device()));
  const int64_t ctas = std::min<int64_t>((M + rows_per_cta - 1) / rows_per_cta, static_cast<int64_t>(sms) * 8);
  gated_norm_kernel<<<static_cast<unsigned>(ctas), 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(core.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The token rows [T, HEADS * 128] of gated_norm's output quantized as fp8.py's _q_rows: (e4m3 [T, HEADS * 128],
// fp32 scale [T, 1]), bit for bit gated_norm then _rowwise_fp8.
std::vector<torch::Tensor> gated_norm_fp8(torch::Tensor core, torch::Tensor z, torch::Tensor w, double eps) {
  TORCH_CHECK(core.is_cuda() && core.scalar_type() == at::kBFloat16 && core.is_contiguous() && core.dim() == 2 &&
                  core.size(1) == D && core.size(0) % QHEADS == 0, "core [T * 32, 128] bf16");
  TORCH_CHECK(z.scalar_type() == at::kBFloat16 && z.is_contiguous() && z.sizes() == core.sizes(), "z like core");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == D, "weight [128] bf16");
  const int64_t T = core.size(0) / QHEADS;
  const at::cuda::CUDAGuard guard(core.device());
  auto q = torch::empty({T, QHEADS * D}, core.options().dtype(at::kFloat8_e4m3fn));
  auto s = torch::empty({T, 1}, core.options().dtype(at::kFloat));
  if (T == 0) return {q, s};
  gated_norm_fp8_kernel<<<static_cast<unsigned>(T), 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(core.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<uint8_t*>(q.data_ptr()),
      s.data_ptr<float>(), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, s};
}

// fused_sigmoid_mul(attn, gate) then fp8.py's _q_rows in one kernel: (e4m3 [T, 4096], fp32 scale [T, 1]); qkv is the
// [q | gate] block (or the whole projection) whose rows hold the gate at h * 512 + 256.
std::vector<torch::Tensor> gate_mul_fp8(torch::Tensor attn, torch::Tensor qkv) {
  TORCH_CHECK(attn.is_cuda() && attn.scalar_type() == at::kBFloat16 && attn.is_contiguous() && attn.dim() == 2 &&
                  attn.size(1) == GHEADS * GDIM, "attn [T, 4096] bf16");
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(0) == attn.size(0) &&
                  qkv.size(1) >= 2 * GHEADS * GDIM && qkv.stride(1) == 1 && qkv.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "qkv [T, >= 8192] bf16, rows 16-byte aligned");
  const int64_t T = attn.size(0);
  const at::cuda::CUDAGuard guard(attn.device());
  auto q = torch::empty({T, GHEADS * GDIM}, attn.options().dtype(at::kFloat8_e4m3fn));
  auto s = torch::empty({T, 1}, attn.options().dtype(at::kFloat));
  if (T == 0) return {q, s};
  gate_mul_fp8_kernel<<<static_cast<unsigned>(T), 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(attn.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      qkv.stride(0), reinterpret_cast<uint8_t*>(q.data_ptr()), s.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, s};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gated_norm", &gated_norm, "GDN prefill gated RMSNorm (norm before swish gate), bit for bit the Triton kernel");
  m.def("gated_norm_fp8", &gated_norm_fp8, "gated_norm then fp8.py's per-token e4m3 row quantizer, one kernel");
  m.def("gate_mul_fp8", &gate_mul_fp8, "attention output gate then fp8.py's per-token e4m3 row quantizer, one kernel");
}
