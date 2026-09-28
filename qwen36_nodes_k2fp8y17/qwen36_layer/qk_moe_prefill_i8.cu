// The crowned fd0012a2 qk_moe_prefill_i8.cu with this line's bitwise prefill routing (v60 gate-first top-k; v68
// top-8 as eight warp max-reductions of packed (value order, id) keys and the shared expert's m-tiles /
// pairs written by the whole scan block).
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#ifndef QMOE_DN_STAGES4
#define QMOE_DN_STAGES4 1
#endif
#ifndef QMOE_Q8
#define QMOE_Q8 1
#endif
#if QMOE_Q8 && !defined(QMOE_Q8_LA)
#define QMOE_Q8_LA 1
#endif
#if QMOE_Q8 && !defined(QMOE_Q8_PAIR)
#define QMOE_Q8_PAIR 0
#endif
// QMOE_Q8_TMA: the up GEMM's INT8 weight slices and the unit's scales arrive by TMA into shared memory (kQR slots
// ahead) instead of per-thread global loads; the producer converts them from shared memory with the same
// q8_slice_to_smem into the same bf16 tile, so the MMA inputs and every output bit are unchanged.
#if QMOE_Q8 && !defined(QMOE_Q8_TMA)
#define QMOE_Q8_TMA 1
#endif
namespace qmoe {
constexpr int kH = 2048;
constexpr int kI = 512;
constexpr int kE = 256;
constexpr int kTopK = 8, kStripes = 8;
constexpr int kBM = 128;
constexpr int kBK = 64;
constexpr int kRowAlign = 64;
constexpr int kUpNB = kI / 128;
constexpr int kDnNB = kH / 256;
constexpr int kUpKB = kH / kBK;
constexpr int kDnKB = kI / kBK;
constexpr uint32_t kABytes = kBM * kBK * 2;
constexpr uint32_t kBBytes = 256 * kBK * 2;
constexpr uint64_t kEvictNormal = 0x1000000000000000ull;
#if QMOE_Q8
constexpr int kQ8Q2 = 2 * kI * kH;
constexpr int kQ8C13 = kQ8Q2 + kH * kI;
constexpr int kQ8C2 = kQ8C13 + 2 * kI * (kH / 128) * 2;
constexpr int kQ8Expert = kQ8C2 + kH * (kI / 128) * 2;
static_assert(kQ8Q2 == 2097152 && kQ8C13 == 3145728 && kQ8C2 == 3178496 && kQ8Expert == 3194880,
              "the INT8 copy's block layout is i8x_kernels.offsets()");
static_assert(kBK * 2 == 128, "one K64 stage is half of a 128-wide scale group");
constexpr int kModeNoMma = 1;
constexpr int kModeNoLoad = 2;
constexpr int kModeNoCvt = 4;
constexpr int kModeFastFold = 8;
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(phase), "r"(0x989680)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                            uint64_t hint) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(hint)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                            int c2, uint64_t hint) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "l"(hint)
      : "memory");
}
__device__ __forceinline__ unsigned short lds_u16(uint32_t addr) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(
                   reinterpret_cast<uint64_t>(map)),
               "r"(src), "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void stsm_x2(uint32_t v0, uint32_t v1, uint32_t addr) {
  asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};" ::"r"(addr), "r"(v0), "r"(v1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
#if QMOE_Q8
__device__ __forceinline__ uint2 q8x4_to_bf16(uint32_t w, float s) {
  const uint32_t u = w ^ 0x80808080u;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    v[j] = __fmul_rn(__fsub_rn(__uint_as_float(__byte_perm(u, 0x4B000000u, 0x7440u | j)), 8388736.f), s);
  return make_uint2(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]));
}
__device__ __forceinline__ uint2 q8x4_to_bf16_fast(uint32_t p, uint32_t s2) {
  uint2 o;
  asm("{\n"
      " .reg .b32 t0, t1, a0, a1, c0, c1;\n"
      " prmt.b32 t0, %2, 0x43434343, 0x4140;\n"
      " prmt.b32 t1, %2, 0x43434343, 0x4342;\n"
      " and.b32 a0, t0, 0xff7fff7f;\n"
      " and.b32 a1, t1, 0xff7fff7f;\n"
      " and.b32 c0, t0, 0xff80ff80;\n"
      " and.b32 c1, t1, 0xff80ff80;\n"
      " sub.rn.bf16x2 a0, a0, c0;\n"
      " sub.rn.bf16x2 a1, a1, c1;\n"
      " mul.rn.bf16x2 %0, a0, %3;\n"
      " mul.rn.bf16x2 %1, a1, %3;\n"
      "}\n"
      : "=&r"(o.x), "=&r"(o.y)
      : "r"(p), "r"(s2));
  return o;
}
__device__ __forceinline__ void sts16(uint32_t addr, uint2 a, uint2 b) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y)
               : "memory");
}
__device__ __forceinline__ void q8_row_to_smem(const uint4 (&raw)[4], float s, uint32_t row_addr, int r7, bool fast) {
  if (fast) {
    const uint32_t s2 = pack_bf16(s, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sts16(row_addr + (((2 * j) ^ r7) << 4), q8x4_to_bf16_fast(raw[j].x, s2), q8x4_to_bf16_fast(raw[j].y, s2));
      sts16(row_addr + (((2 * j + 1) ^ r7) << 4), q8x4_to_bf16_fast(raw[j].z, s2), q8x4_to_bf16_fast(raw[j].w, s2));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sts16(row_addr + (((2 * j) ^ r7) << 4), q8x4_to_bf16(raw[j].x, s), q8x4_to_bf16(raw[j].y, s));
      sts16(row_addr + (((2 * j + 1) ^ r7) << 4), q8x4_to_bf16(raw[j].z, s), q8x4_to_bf16(raw[j].w, s));
    }
  }
}
__device__ __forceinline__ void q8_row_load(const uint8_t* row, const unsigned short* scales, int kb, uint4 (&raw)[4],
                                            float& s) {
  const uint4* p = reinterpret_cast<const uint4*>(row + kb * kBK);
#pragma unroll
  for (int j = 0; j < 4; ++j) raw[j] = __ldg(p + j);
  s = __half2float(__ushort_as_half(__ldg(scales + (kb * kBK) / 128)));
}
__device__ __forceinline__ void q8_slice_load(const uint8_t* row, const unsigned short* scales, int kb, uint4& raw,
                                              float& s) {
  raw = __ldg(reinterpret_cast<const uint4*>(row + kb * kBK));
  s = __half2float(__ushort_as_half(__ldg(scales + (kb * kBK) / 128)));
}
__device__ __forceinline__ void q8_slice_to_smem(uint4 raw, float s, uint32_t row_addr, int j, int r7, bool fast) {
  if (fast) {
    const uint32_t s2 = pack_bf16(s, s);
    sts16(row_addr + (((2 * j) ^ r7) << 4), q8x4_to_bf16_fast(raw.x, s2), q8x4_to_bf16_fast(raw.y, s2));
    sts16(row_addr + (((2 * j + 1) ^ r7) << 4), q8x4_to_bf16_fast(raw.z, s2), q8x4_to_bf16_fast(raw.w, s2));
  } else {
    sts16(row_addr + (((2 * j) ^ r7) << 4), q8x4_to_bf16(raw.x, s), q8x4_to_bf16(raw.y, s));
    sts16(row_addr + (((2 * j + 1) ^ r7) << 4), q8x4_to_bf16(raw.z, s), q8x4_to_bf16(raw.w, s));
  }
}
#endif
__device__ __forceinline__ uint64_t gmma_desc(uint32_t smem_addr) {
  return static_cast<uint64_t>((smem_addr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(1024 >> 4) << 32) |
         (static_cast<uint64_t>(1) << 62);
}
// wgmma.mma_async m64n128k32 s32 (+)= s8 x s8, both operands K-major in shared memory; scale_d 0 starts a fresh sum.
__device__ __forceinline__ void wgmma_m64n128k32_s8(int (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, p;\n"
      "}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void fence_operand(int& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void wgmma_m64n256k16(float (&d)[128], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
      "}, "
      "%128, %129, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(da), "l"(db), "r"(1));
}
constexpr uint32_t kNanKey17 = 0x1d6b;
__device__ __forceinline__ uint32_t sel_key(uint32_t b, int id) {
  uint32_t k17;
  if ((b & 0x7fffu) > 0x7f80u) {
    k17 = kNanKey17;
  } else if (b == 0xff80u) {
    return 0u;
  } else {
    if (b == 0x8000u) b = 0u;
    k17 = 2u * ((b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u));
  }
  return (k17 << 8) | static_cast<uint32_t>(255 - id);
}
__device__ __forceinline__ float key_value(uint32_t key) {
  const uint32_t k17 = key >> 8;
  if (k17 & 1u) return -1e30f;
  const uint32_t k16 = k17 >> 1;
  return __uint_as_float(((k16 & 0x8000u) ? (k16 & 0x7fffu) : (~k16 & 0xffffu)) << 16);
}
__global__ void __launch_bounds__(256, 4) route_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                                         const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ gw, int T,
                                                         int* __restrict__ cnt, int* __restrict__ topk_ids,
                                                         float* __restrict__ topk_w, int* __restrict__ rank,
                                                         float* __restrict__ sg) {
  __shared__ uint4 s_gw[kH / 8];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int t = blockIdx.x * 8 + warp;
  s_gw[threadIdx.x] = reinterpret_cast<const uint4*>(gw)[threadIdx.x];
  uint4 xa[kH / 8 / 32];
  uint4 lraw = make_uint4(0, 0, 0, 0);
  if (t < T) {
    const uint4* xr = reinterpret_cast<const uint4*>(x + static_cast<size_t>(t) * kH);
#pragma unroll
    for (int j = 0; j < kH / 8 / 32; ++j) xa[j] = xr[lane + 32 * j];
    lraw = *reinterpret_cast<const uint4*>(logits + static_cast<size_t>(t) * kE + lane * 8);
  }
  __syncthreads();
  if (t >= T) return;
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < kH / 8 / 32; ++j) {
    const uint4 bq = s_gw[lane + 32 * j];
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&xa[j]);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bq);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 fa = __bfloat1622float2(a2[q]), fb = __bfloat1622float2(b2[q]);
      acc = fmaf(fa.x, fb.x, acc);
      acc = fmaf(fa.y, fb.y, acc);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) sg[t] = 1.f / (1.f + expf(-acc));
  uint32_t key[8];
  {
    const uint32_t w4[4] = {lraw.x, lraw.y, lraw.z, lraw.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) key[j] = sel_key((w4[j >> 1] >> (16 * (j & 1))) & 0xffffu, lane * 8 + j);
  }
  float sel_v[kTopK];
  int sel_i[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    uint32_t best = key[0];
#pragma unroll
    for (int j = 1; j < 8; ++j) best = max(best, key[j]);
    const uint32_t win = __reduce_max_sync(0xffffffffu, best);
    sel_v[k] = win == 0u ? -INFINITY : key_value(win);
    sel_i[k] = win == 0u ? 0x7fffffff : 255 - static_cast<int>(win & 0xffu);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (key[j] == win) key[j] = 0u;
  }
  float p[kTopK], s = 0.f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) p[k] = expf(sel_v[k] - sel_v[0]), s += p[k];
  float pk = p[0];
  int ik = sel_i[0];
#pragma unroll
  for (int k = 1; k < kTopK; ++k)
    if (lane == k) pk = p[k], ik = sel_i[k];
  const int r = lane < kTopK ? atomicAdd(cnt + (blockIdx.x % kStripes) * kE + ik, 1) : 0;
  if (lane < kTopK) {
    const int idx = t * kTopK + lane;
    topk_ids[idx] = ik;
    topk_w[idx] = pk / s;
    rank[idx] = r;
  }
}
__device__ __forceinline__ int block_excl_scan(int v, int* s) {
  const int e = threadIdx.x;
  s[e] = v;
  __syncthreads();
  for (int off = 1; off < 512; off <<= 1) {
    const int r = e >= off ? s[e - off] : 0;
    __syncthreads();
    s[e] += r;
    __syncthreads();
  }
  const int incl = s[e];
  __syncthreads();
  return incl - v;
}
__global__ void __launch_bounds__(512) route_scan_kernel(int T, int* __restrict__ cnt, int* __restrict__ sbase,
                                                         int* __restrict__ offs,
                                                         int4* __restrict__ mt, int4* __restrict__ pairs,
                                                         int* __restrict__ n_pairs) {
  __shared__ int s_scan[512], s_tail_mt[512];
  const int e = threadIdx.x;
  int c = e == kE ? T : 0;
  if (e < kE) {
#pragma unroll
    for (int st = 0; st < kStripes; ++st) {
      const int v = cnt[st * kE + e];
      sbase[st * kE + e] = c;
      cnt[st * kE + e] = 0;
      c += v;
    }
  }
  const int rows = (c + kRowAlign - 1) / kRowAlign * kRowAlign;
  const int nm = (c + kBM - 1) / kBM;
  const int np = nm / 2, tail = nm & 1;
  const int row0 = block_excl_scan(rows, s_scan);
  const int mt0 = block_excl_scan(nm, s_scan);
  const bool routed = e < kE;
  const int tj = block_excl_scan(routed ? tail : 0, s_scan);
  const int slots = routed ? np + ((tail && (tj & 1)) ? 1 : 0) : 0;
  const int p0 = block_excl_scan(slots, s_scan);
  __shared__ int s_sh[5];
  if (e <= kE) offs[e] = row0;
  if (e == kE) {
    offs[kE + 1] = row0 + rows;
    s_sh[0] = mt0, s_sh[1] = row0, s_sh[2] = c, s_sh[3] = p0, s_sh[4] = tj;
  }
  if (routed)
    for (int j = 0; j < nm; ++j) mt[mt0 + j] = make_int4(e, row0 + j * kBM, min(kBM, c - j * kBM), 0);
  if (routed && tail) s_tail_mt[tj] = mt0 + nm - 1;
  __syncthreads();
  if (routed) {
    for (int i = 0; i < np; ++i) pairs[p0 + i] = make_int4(mt0 + 2 * i, mt0 + 2 * i + 1, 1, 0);
    if (tail && (tj & 1)) pairs[p0 + np] = make_int4(s_tail_mt[tj - 1], mt0 + nm - 1, 0, 0);
  }
  {
    const int smt0 = s_sh[0], srow0 = s_sh[1], sc = s_sh[2], sp0 = s_sh[3], stj = s_sh[4];
    const int snm = (sc + kBM - 1) / kBM, snp = snm / 2, stail = snm & 1;
    for (int j = e; j < snm; j += blockDim.x) mt[smt0 + j] = make_int4(kE, srow0 + j * kBM, min(kBM, sc - j * kBM), 0);
    const int first = sp0 + ((stj & 1) ? 1 : 0);
    if (e == 0 && (stj & 1)) pairs[sp0] = make_int4(s_tail_mt[stj - 1], -1, 0, 0);
    for (int i = e; i < snp; i += blockDim.x) pairs[first + i] = make_int4(smt0 + 2 * i, smt0 + 2 * i + 1, 1, 0);
    if (e == 0 && stail) pairs[first + snp] = make_int4(smt0 + snm - 1, -1, 0, 0);
    if (e == 0) {
      *n_pairs = first + snp + stail;
      cnt[kStripes * kE] = 0;
    }
  }
}
__global__ void route_scatter_kernel(int T, const int* __restrict__ topk_ids, const int* __restrict__ rank,
                                     const int* __restrict__ sbase, const int* __restrict__ offs,
                                     int* __restrict__ sorted_tok,
                                     int* __restrict__ pos_tk) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T * kTopK) {
    const int e = topk_ids[i], t = i / kTopK;
    const int pos = offs[e] + sbase[(t / 8 % kStripes) * kE + e] + rank[i];
    sorted_tok[pos] = t;
    pos_tk[i] = pos;
  } else if (i < T * (kTopK + 1)) {
    const int t = i - T * kTopK;
    sorted_tok[offs[kE] + t] = t;
  }
}
struct GemmArgs {
  const int4* mt;
  const int4* pairs;
  const int* n_pairs;
  const int* sorted_tok;
  const __nv_bfloat16* x;
  const __nv_bfloat16* y;  // kY8: the int8 bytes of y, with ysc
  float* ysc;              // kY8: y's scale per routed row and 256-column n-block [rows][kDnNB]
  const int* pos_tk;
  const float* topk_w;
  const float* sg;
  __nv_bfloat16* out;
  int* done;
#if QMOE_Q8
  const uint8_t* q8;
  int mode;
#endif
};
template <bool kUp, bool kQ8 = false>
struct GemmCfg {
#if QMOE_Q8 && QMOE_Q8_TMA
  // the TMA'd INT8 up: 3 bf16 stages + kQR INT8 slots (256 rows x 64 B) + the unit's scales (256 rows x 16 fp16)
  static constexpr bool kTma = kUp && kQ8;
  static constexpr int kQR = kTma ? 2 : 0;
  static constexpr uint32_t kQSlot = 256 * kBK;
  static constexpr uint32_t kQBytes = kTma ? kQR * kQSlot + 256 * (kH / 128) * 2 : 0;
#else
  static constexpr bool kTma = false;
  static constexpr int kQR = 0;
  static constexpr uint32_t kQSlot = 0, kQBytes = 0;
#endif
#if QMOE_DN_STAGES4
  static constexpr int kStages = kTma ? 3 : 4;
#else
  static constexpr int kStages = kTma ? 3 : (kUp ? 4 : 3);
#endif
  static constexpr int NB = kUp ? kUpNB : kDnNB;
  static constexpr int NKB = kUp ? kUpKB : kDnKB;
#if QMOE_DN_STAGES4
  static constexpr uint32_t kDBytesWG = 64 * 128 * 2;
#else
  static constexpr uint32_t kDBytesWG = kUp ? 64 * 128 * 2 : 64 * 256 * 2;
#endif
  static constexpr uint32_t kSmem = kStages * (kABytes + kBBytes) + 2 * kDBytesWG + kQBytes + 2 * kStages * 8 +
                                     (kTma ? (kQR + 1) * 8 : 0);
#if QMOE_DN_STAGES4
  static_assert(kTma || kSmem == 229440, "4-stage MoE GEMM smem is 229440 B");
#endif
  static_assert(kSmem <= 232448, "MoE GEMM smem must fit the 227 KiB opt-in");
};
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t cluster_id_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t n_clusters_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t* bar, uint32_t cta) {
  asm volatile(
      "{\n"
      ".reg .b32 ra;\n"
      "mapa.shared::cluster.u32 ra, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64 _, [ra];\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(cta)
      : "memory");
}
#if QMOE_Q8
__device__ __forceinline__ void bulk_s2s_to_peer(uint32_t local_src, uint32_t bytes, uint64_t* bar, uint32_t cta) {
  asm volatile(
      "{\n"
      ".reg .b32 rd, rb;\n"
      "mapa.shared::cluster.u32 rd, %0, %3;\n"
      "mapa.shared::cluster.u32 rb, %1, %3;\n"
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd], [%0], %2, [rb];\n"
      "}\n" ::"r"(local_src),
      "r"(smem_u32(bar)), "r"(bytes), "r"(cta)
      : "memory");
}
#endif
__device__ __forceinline__ void tma_load_2d_mc(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask, uint64_t hint) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "h"(mask), "r"(c0), "r"(c1), "l"(hint)
      : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(policy)
               : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 lds16(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}
// kY8 (the down of down_i8, a PRECISION CHANGE): routed tiles store y as int8 with one fp32 scale (amax / 127) per row
// and 256-column n-block, and the combine reads those back: half the bytes of y's write and read. (e4m3 y fails the
// node audit: its ~2 % MoE-output error is not diluted, tensor 0 is the MLP output; int8 y adds ~0.55 % on real rows.)
#if QMOE_Q8
template <bool kUp, bool kQ8 = false, bool kY8 = false>
#else
template <bool kUp>
#endif
__global__ void __launch_bounds__(384, 1)
    moe_gemm_kernel(const GemmArgs args, const __grid_constant__ CUtensorMap tm_b,
                    const __grid_constant__ CUtensorMap tm_bs, const __grid_constant__ CUtensorMap tm_a,
                    const __grid_constant__ CUtensorMap tm_d, const __grid_constant__ CUtensorMap tm_q8,
                    const __grid_constant__ CUtensorMap tm_q8s) {
#if QMOE_Q8
  using Cfg = GemmCfg<kUp, kQ8>;
#else
  using Cfg = GemmCfg<kUp>;
#endif
  constexpr int kStages = Cfg::kStages, NB = Cfg::NB, NKB = Cfg::NKB;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + kStages * kABytes;
  uint8_t* sD = sB + kStages * kBBytes;
  uint8_t* sQ = sD + 2 * Cfg::kDBytesWG;  // TMA'd INT8 up only: kQR INT8 slots, then the unit's scales
  uint64_t* full = reinterpret_cast<uint64_t*>(sQ + Cfg::kQBytes);
  uint64_t* empty = full + kStages;
  uint64_t* qfull = empty + kStages;  // kQR INT8 slots + the scales
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t rank = cluster_rank();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
#if QMOE_Q8
      mbar_init(&full[s], (kUp ? 129 : 1) + (kQ8 ? 128 : 0));
#else
      mbar_init(&full[s], kUp ? 129 : 1);
#endif
      mbar_init(&empty[s], 16);
    }
    if constexpr (Cfg::kTma) {
#pragma unroll
      for (int s = 0; s <= Cfg::kQR; ++s) mbar_init(&qfull[s], 1);
    }
    fence_barrier_init();
  }
  if (threadIdx.x == 32) {
    prefetch_tmap(&tm_b);
    prefetch_tmap(&tm_bs);
    if constexpr (!kUp) prefetch_tmap(&tm_a);
    prefetch_tmap(&tm_d);
    if constexpr (Cfg::kTma) {
      prefetch_tmap(&tm_q8);
      prefetch_tmap(&tm_q8s);
    }
  }
  cluster_sync();
  const int n_units = *args.n_pairs * NB;
  const int unit0 = static_cast<int>(cluster_id_x()), unit_step = static_cast<int>(n_clusters_x());
  auto resolve = [&](int u, int4& info, bool& shared, int& n) {
    const int4 pr = args.pairs[u / NB];
    n = u % NB;
    const int m = rank == 0 ? pr.x : pr.y;
    info = m >= 0 ? args.mt[m] : make_int4(0, 0, 0, 0);
    shared = pr.z != 0;
  };
  if (threadIdx.x >= 256) {
    const int pt = threadIdx.x - 256;
    int stage = 0;
    uint32_t phase = 0;
#if QMOE_Q8
    if constexpr (Cfg::kTma) {
      const uint64_t pol_x = policy_evict_last();
      const int c = pt & 7, r0 = pt >> 3;
      const uint32_t dst_off = r0 * 128 + ((c ^ (r0 & 7)) << 4);
      const int q8g = pt >> 2, q8j = pt & 3;
      const bool fast = args.mode & kModeFastFold;
      constexpr int kQR = Cfg::kQR;
      constexpr uint32_t kQSlot = Cfg::kQSlot, kGroups = kH / 128;
      const uint32_t q_base = smem_u32(sQ), s_base = q_base + kQR * kQSlot;
      uint32_t qi = 0, s_phase = 0;  // INT8 stages consumed (all units) and scale loads consumed
      for (int u = unit0; u < n_units; u += unit_step) {
        int4 info;
        bool shared;
        int n;
        resolve(u, info, shared, n);
        const bool has = info.z > 0;
        const CUtensorMap* bm = info.x == kE ? &tm_bs : &tm_b;
        const int brow = info.x == kE ? 0 : info.x * (2 * kI);
        const __nv_bfloat16* src[8];
        uint32_t mask = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = r0 + 16 * i;
          src[i] = args.x + c * 8;
          if (r < info.z) {
            mask |= 1u << i;
            src[i] += static_cast<size_t>(args.sorted_tok[info.y + r]) * kH;
          }
        }
        const bool q8 = info.x != kE;
        const bool ld = q8 && has && !(args.mode & kModeNoLoad);
        const bool cvt = ld && !(args.mode & kModeNoCvt);
        if (ld && pt == 0) {
          // every producer thread read the previous unit's last slots and scales before its last stage barrier
          mbar_arrive_expect_tx(&qfull[kQR], 256 * kGroups * 2);
          tma_load_3d(s_base, &tm_q8s, &qfull[kQR], 0, n * 128, info.x, kEvictNormal);
          tma_load_3d(s_base + 128 * kGroups * 2, &tm_q8s, &qfull[kQR], 0, kI + n * 128, info.x, kEvictNormal);
#pragma unroll
          for (int p = 0; p < kQR; ++p) {
            const uint32_t slot = (qi + p) % kQR;
            mbar_arrive_expect_tx(&qfull[slot], kQSlot);
            tma_load_3d(q_base + slot * kQSlot, &tm_q8, &qfull[slot], p * kBK, n * 128, info.x, kEvictNormal);
            tma_load_3d(q_base + slot * kQSlot + kQSlot / 2, &tm_q8, &qfull[slot], p * kBK, kI + n * 128, info.x,
                        kEvictNormal);
          }
        }
        if (ld) {
          mbar_wait(&qfull[kQR], s_phase);
          s_phase ^= 1;
        }
        for (int kb = 0; kb < NKB; ++kb) {
          uint4 raw[8];
          float sc[8];
          if (ld) {
            const uint32_t slot = qi % kQR;
            mbar_wait(&qfull[slot], (qi / kQR) & 1);
            const uint32_t qs = q_base + slot * kQSlot + q8g * kBK + q8j * 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              raw[i] = lds16(qs + i * 32 * kBK);
              sc[i] = __half2float(__ushort_as_half(
                  lds_u16(s_base + ((q8g + 32 * i) * kGroups + (kb * kBK) / 128) * 2)));
            }
            named_bar_sync(4, 128);  // the slot is read by all 128 producer threads: refill it kQR stages ahead
            if (pt == 0 && kb + kQR < NKB) {
              mbar_arrive_expect_tx(&qfull[slot], kQSlot);
              tma_load_3d(q_base + slot * kQSlot, &tm_q8, &qfull[slot], (kb + kQR) * kBK, n * 128, info.x,
                          kEvictNormal);
              tma_load_3d(q_base + slot * kQSlot + kQSlot / 2, &tm_q8, &qfull[slot], (kb + kQR) * kBK,
                          kI + n * 128, info.x, kEvictNormal);
            }
            ++qi;
          }
          mbar_wait(&empty[stage], phase ^ 1);
          const uint32_t a_dst = smem_u32(sA + stage * kABytes) + dst_off;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            if (has && (mask >> i & 1))
              cp_async16(a_dst + i * 2048, src[i] + kb * kBK, pol_x);
          if (pt == 0) {
            if (has && !q8) {
              mbar_arrive_expect_tx(&full[stage], kBBytes);
              const uint32_t b_dst = smem_u32(sB + stage * kBBytes);
              if (!shared) {
                tma_load_2d(b_dst, bm, &full[stage], kb * kBK, brow + n * 128, kEvictNormal);
                tma_load_2d(b_dst + kBBytes / 2, bm, &full[stage], kb * kBK, brow + kI + n * 128, kEvictNormal);
              } else if (rank == 0) {
                tma_load_2d_mc(b_dst, bm, &full[stage], kb * kBK, brow + n * 128, 3, kEvictNormal);
                tma_load_2d_mc(b_dst + kBBytes / 2, bm, &full[stage], kb * kBK, brow + kI + n * 128, 3,
                               kEvictNormal);
              }
            } else {
              mbar_arrive(&full[stage]);
            }
          }
          if (cvt) {
            const uint32_t b_row = smem_u32(sB + stage * kBBytes) + q8g * 128;
#pragma unroll
            for (int i = 0; i < 8; ++i) q8_slice_to_smem(raw[i], sc[i], b_row + i * 32 * 128, q8j, q8g & 7, fast);
            fence_async_shared();
          }
          mbar_arrive(&full[stage]);
          cp_async_arrive_noinc(&full[stage]);
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    } else if constexpr (kUp) {
      const uint64_t pol_x = policy_evict_last();
      const int c = pt & 7, r0 = pt >> 3;
      const uint32_t dst_off = r0 * 128 + ((c ^ (r0 & 7)) << 4);
      for (int u = unit0; u < n_units; u += unit_step) {
        int4 info;
        bool shared;
        int n;
        resolve(u, info, shared, n);
        const bool has = info.z > 0;
        const CUtensorMap* bm = info.x == kE ? &tm_bs : &tm_b;
        const int brow = info.x == kE ? 0 : info.x * (2 * kI);
        const __nv_bfloat16* src[8];
        uint32_t mask = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = r0 + 16 * i;
          src[i] = args.x + c * 8;
          if (r < info.z) {
            mask |= 1u << i;
            src[i] += static_cast<size_t>(args.sorted_tok[info.y + r]) * kH;
          }
        }
        const bool q8 = kQ8 && info.x != kE;
        const bool ld = q8 && has && !(args.mode & kModeNoLoad);
        const bool cvt = ld && !(args.mode & kModeNoCvt);
        const bool fast = args.mode & kModeFastFold;
        const bool pair = QMOE_Q8_PAIR && q8 && shared;
        const uint32_t peer = rank ^ 1u;
        const int own_lo = pair ? 4 * static_cast<int>(rank) : 0, own_hi = pair ? own_lo + 4 : 8;
        const uint8_t* qrow[8];
        const unsigned short* srow[8];
        constexpr int kLA = QMOE_Q8_LA, kNS = kLA + 1;
        uint4 raw[kNS][8];
        float sc[kNS][8];
        const int q8g = pt >> 2, q8j = pt & 3;
        if constexpr (kQ8) {
          if (ld) {
            const uint8_t* blk = args.q8 + static_cast<size_t>(info.x) * kQ8Expert;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const int row = (i < 4 ? 0 : kI - 128) + n * 128 + q8g + 32 * i;
              qrow[i] = blk + static_cast<size_t>(row) * kH + q8j * 16;
              srow[i] = reinterpret_cast<const unsigned short*>(blk + kQ8C13) + row * (kH / 128);
            }
#pragma unroll
            for (int p = 0; p < kLA; ++p)
              if (p < NKB) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                  if (i >= own_lo && i < own_hi) q8_slice_load(qrow[i], srow[i], p, raw[p][i], sc[p][i]);
              }
          }
        }
        for (int kb0 = 0; kb0 < NKB; kb0 += kNS) {
#pragma unroll
        for (int t = 0; t < kNS; ++t) {
          const int kb = kb0 + t;
          if (kb >= NKB) break;
          if constexpr (kQ8) {
            if (ld && kb + kLA < NKB) {
#pragma unroll
              for (int i = 0; i < 8; ++i)
                if (i >= own_lo && i < own_hi)
                  q8_slice_load(qrow[i], srow[i], kb + kLA, raw[(t + kLA) % kNS][i], sc[(t + kLA) % kNS][i]);
            }
          }
          mbar_wait(&empty[stage], phase ^ 1);
          const uint32_t a_dst = smem_u32(sA + stage * kABytes) + dst_off;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            if (has && (mask >> i & 1))
              cp_async16(a_dst + i * 2048, src[i] + kb * kBK, pol_x);
          if (pt == 0) {
            if (has && !q8) {
              mbar_arrive_expect_tx(&full[stage], kBBytes);
              const uint32_t b_dst = smem_u32(sB + stage * kBBytes);
              if (!shared) {
                tma_load_2d(b_dst, bm, &full[stage], kb * kBK, brow + n * 128, kEvictNormal);
                tma_load_2d(b_dst + kBBytes / 2, bm, &full[stage], kb * kBK, brow + kI + n * 128, kEvictNormal);
              } else if (rank == 0) {
                tma_load_2d_mc(b_dst, bm, &full[stage], kb * kBK, brow + n * 128, 3, kEvictNormal);
                tma_load_2d_mc(b_dst + kBBytes / 2, bm, &full[stage], kb * kBK, brow + kI + n * 128, 3,
                               kEvictNormal);
              }
            } else if (pair) {
              mbar_arrive_expect_tx(&full[stage], kBBytes / 2);
            } else {
              mbar_arrive(&full[stage]);
            }
          }
          if constexpr (kQ8) {
            if (cvt) {
              const uint32_t b_row = smem_u32(sB + stage * kBBytes) + q8g * 128;
#pragma unroll
              for (int i = 0; i < 8; ++i)
                if (i >= own_lo && i < own_hi)
                  q8_slice_to_smem(raw[t][i], sc[t][i], b_row + i * 32 * 128, q8j, q8g & 7, fast);
              fence_async_shared();
            }
#if QMOE_Q8_PAIR
            if (pair) {
              named_bar_sync(3, 128);
              if (pt == 0)
                bulk_s2s_to_peer(smem_u32(sB + stage * kBBytes) + rank * (kBBytes / 2), kBBytes / 2, &full[stage],
                                 peer);
            }
#endif
            mbar_arrive(&full[stage]);
          }
          cp_async_arrive_noinc(&full[stage]);
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
        }
      }
    } else if constexpr (kQ8) {
      for (int u = unit0; u < n_units; u += unit_step) {
        int4 info;
        bool shared;
        int n;
        resolve(u, info, shared, n);
        const bool has = info.z > 0;
        const bool q8 = info.x != kE;
        const bool ld = q8 && has && !(args.mode & kModeNoLoad);
        const bool cvt = ld && !(args.mode & kModeNoCvt);
        const bool fast = args.mode & kModeFastFold;
        const CUtensorMap* bm = info.x == kE ? &tm_bs : &tm_b;
        const int brow = info.x == kE ? 0 : info.x * kH;
        const uint8_t* qrow[2] = {nullptr, nullptr};
        const unsigned short* srow[2] = {nullptr, nullptr};
        uint4 raw[2][4];
        float sc[2] = {0.f, 0.f};
        if (ld) {
          const uint8_t* blk = args.q8 + static_cast<size_t>(info.x) * kQ8Expert;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = n * 256 + h * 128 + pt;
            qrow[h] = blk + kQ8Q2 + static_cast<size_t>(row) * kI;
            srow[h] = reinterpret_cast<const unsigned short*>(blk + kQ8C2) + row * (kI / 128);
            q8_row_load(qrow[h], srow[h], 0, raw[h], sc[h]);
          }
        }
        for (int kb = 0; kb < NKB; ++kb) {
          uint4 nraw[2][4];
          float nsc[2] = {0.f, 0.f};
          if (ld && kb + 1 < NKB) {
#pragma unroll
            for (int h = 0; h < 2; ++h) q8_row_load(qrow[h], srow[h], kb + 1, nraw[h], nsc[h]);
          }
          mbar_wait(&empty[stage], phase ^ 1);
          if (pt == 0) {
            if (has) {
              mbar_arrive_expect_tx(&full[stage], q8 ? kABytes : kABytes + kBBytes);
              tma_load_2d(smem_u32(sA + stage * kABytes), &tm_a, &full[stage], kb * kBK, info.y, kEvictNormal);
              const uint32_t b_dst = smem_u32(sB + stage * kBBytes);
              if (!q8 && !shared)
                tma_load_2d(b_dst, bm, &full[stage], kb * kBK, brow + n * 256, kEvictNormal);
              else if (!q8 && rank == 0)
                tma_load_2d_mc(b_dst, bm, &full[stage], kb * kBK, brow + n * 256, 3, kEvictNormal);
            } else {
              mbar_arrive(&full[stage]);
            }
          }
          if (cvt) {
            const uint32_t b_row = smem_u32(sB + stage * kBBytes) + pt * 128;
            q8_row_to_smem(raw[0], sc[0], b_row, pt & 7, fast);
            q8_row_to_smem(raw[1], sc[1], b_row + 128 * 128, pt & 7, fast);
            fence_async_shared();
          }
          if (ld) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
              for (int j = 0; j < 4; ++j) raw[h][j] = nraw[h][j];
              sc[h] = nsc[h];
            }
          }
          mbar_arrive(&full[stage]);
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    } else if (pt == 0) {
      for (int u = unit0; u < n_units; u += unit_step) {
        int4 info;
        bool shared;
        int n;
        resolve(u, info, shared, n);
        const bool has = info.z > 0;
        const CUtensorMap* bm = info.x == kE ? &tm_bs : &tm_b;
        const int brow = info.x == kE ? 0 : info.x * kH;
        for (int kb = 0; kb < NKB; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          if (has) {
            mbar_arrive_expect_tx(&full[stage], kABytes + kBBytes);
            tma_load_2d(smem_u32(sA + stage * kABytes), &tm_a, &full[stage], kb * kBK, info.y, kEvictNormal);
            const uint32_t b_dst = smem_u32(sB + stage * kBBytes);
            if (!shared)
              tma_load_2d(b_dst, bm, &full[stage], kb * kBK, brow + n * 256, kEvictNormal);
            else if (rank == 0)
              tma_load_2d_mc(b_dst, bm, &full[stage], kb * kBK, brow + n * 256, 3, kEvictNormal);
          } else {
            mbar_arrive(&full[stage]);
          }
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    }
#else
    if constexpr (kUp) {
      const uint64_t pol_x = policy_evict_last();
      const int c = pt & 7, r0 = pt >> 3;
      const uint32_t dst_off = r0 * 128 + ((c ^ (r0 & 7)) << 4);
      for (int u = unit0; u < n_units; u += unit_step) {
        int4 info;
        bool shared;
        int n;
        resolve(u, info, shared, n);
        const bool has = info.z > 0;
        const CUtensorMap* bm = info.x == kE ? &tm_bs : &tm_b;
        const int brow = info.x == kE ? 0 : info.x * (2 * kI);
        const __nv_bfloat16* src[8];
        uint32_t mask = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = r0 + 16 * i;
          src[i] = args.x + c * 8;
          if (r < info.z) {
            mask |= 1u << i;
            src[i] += static_cast<size_t>(args.sorted_tok[info.y + r]) * kH;
          }
        }
        for (int kb = 0; kb < NKB; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          const uint32_t a_dst = smem_u32(sA + stage * kABytes) + dst_off;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            if (has && (mask >> i & 1))
              cp_async16(a_dst + i * 2048, src[i] + kb * kBK, pol_x);
          if (pt == 0) {
            if (has) {
              mbar_arrive_expect_tx(&full[stage], kBBytes);
              const uint32_t b_dst = smem_u32(sB + stage * kBBytes);
              if (!shared) {
                tma_load_2d(b_dst, bm, &full[stage], kb * kBK, brow + n * 128, kEvictNormal);
                tma_load_2d(b_dst + kBBytes / 2, bm, &full[stage], kb * kBK, brow + kI + n * 128, kEvictNormal);
              } else if (rank == 0) {
                tma_load_2d_mc(b_dst, bm, &full[stage], kb * kBK, brow + n * 128, 3, kEvictNormal);
                tma_load_2d_mc(b_dst + kBBytes / 2, bm, &full[stage], kb * kBK, brow + kI + n * 128, 3,
                               kEvictNormal);
              }
            } else {
              mbar_arrive(&full[stage]);
            }
          }
          cp_async_arrive_noinc(&full[stage]);
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    } else if (pt == 0) {
      for (int u = unit0; u < n_units; u += unit_step) {
        int4 info;
        bool shared;
        int n;
        resolve(u, info, shared, n);
        const bool has = info.z > 0;
        const CUtensorMap* bm = info.x == kE ? &tm_bs : &tm_b;
        const int brow = info.x == kE ? 0 : info.x * kH;
        for (int kb = 0; kb < NKB; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          if (has) {
            mbar_arrive_expect_tx(&full[stage], kABytes + kBBytes);
            tma_load_2d(smem_u32(sA + stage * kABytes), &tm_a, &full[stage], kb * kBK, info.y, kEvictNormal);
            const uint32_t b_dst = smem_u32(sB + stage * kBBytes);
            if (!shared)
              tma_load_2d(b_dst, bm, &full[stage], kb * kBK, brow + n * 256, kEvictNormal);
            else if (rank == 0)
              tma_load_2d_mc(b_dst, bm, &full[stage], kb * kBK, brow + n * 256, 3, kEvictNormal);
          } else {
            mbar_arrive(&full[stage]);
          }
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    }
#endif
  } else {
    const int wg = threadIdx.x / 128, wi = warp % 4;
    const uint32_t a_base = smem_u32(sA) + wg * 64 * 128;
    const uint32_t b_base = smem_u32(sB);
    const uint32_t d_base = smem_u32(sD) + wg * Cfg::kDBytesWG;
    int stage = 0;
    uint32_t phase = 0;
    bool signaled = kUp;
    auto signal_routed_done = [&]() {
      if (threadIdx.x % 128 == 0) {
        tma_store_wait_all();
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        atomicAdd(args.done, 1);
      }
      signaled = true;
    };
    for (int u = unit0; u < n_units; u += unit_step) {
      int4 info;
      bool shared;
      int n;
      resolve(u, info, shared, n);
      const bool active = info.z > wg * 64;
      const bool shared_tile = !kUp && info.x == kE && info.z > 0;
      if (shared_tile && !signaled) signal_routed_done();
      float acc[128];
#pragma unroll
      for (int i = 0; i < 128; ++i) acc[i] = 0.f;
      // QMOE_MMA_LAG: one wgmma group stays in flight across the stage boundary; a stage is released once the
      // group reading it completes (the next stage's wait<1>). Same wgmmas on the same accumulator in the same
      // order: same bits.
      int prev = -1;
      for (int kb = 0; kb < NKB; ++kb) {
        mbar_wait(&full[stage], phase);
#if QMOE_Q8
        const bool mma = active && !(kQ8 && (args.mode & kModeNoMma));
#else
        const bool mma = active;
#endif
        if (mma) {
#pragma unroll
          for (int i = 0; i < 128; ++i) fence_operand(acc[i]);
          wgmma_fence();
#pragma unroll
          for (int k = 0; k < kBK / 16; ++k)
            wgmma_m64n256k16(acc, gmma_desc(a_base + stage * kABytes + k * 32),
                             gmma_desc(b_base + stage * kBBytes + k * 32));
          wgmma_commit();
#pragma unroll
          for (int i = 0; i < 128; ++i) fence_operand(acc[i]);
          wgmma_wait<1>();
        }
        if (prev >= 0 && lane < 2) mbar_arrive_cluster(&empty[prev], lane);
        prev = stage;
        stage = stage + 1 == kStages ? 0 : stage + 1;
        phase ^= stage == 0;
      }
      wgmma_wait<0>();
#pragma unroll
      for (int i = 0; i < 128; ++i) fence_operand(acc[i]);
      if (lane < 2) mbar_arrive_cluster(&empty[prev], lane);
      if (!active) continue;
#if QMOE_DN_STAGES4
      if constexpr (!kUp) {
        const uint32_t row_addr = d_base + wi * 2048 + (lane % 16) * 128;
        float ga = 1.f, gb = 1.f;
        if (shared_tile) {
          const int ra = wg * 64 + wi * 16 + lane / 4, rb = ra + 8;
          ga = ra < info.z ? args.sg[args.sorted_tok[info.y + ra]] : 0.f;
          gb = rb < info.z ? args.sg[args.sorted_tok[info.y + rb]] : 0.f;
        }
        if (!shared_tile && kY8) {
          // int8 rows: scale = amax / 127 over the row's 256 columns of this tile (four lanes hold a row)
          float ma = 0.f, mb = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            ma = fmaxf(ma, fmaxf(fabsf(acc[4 * i]), fabsf(acc[4 * i + 1])));
            mb = fmaxf(mb, fmaxf(fabsf(acc[4 * i + 2]), fabsf(acc[4 * i + 3])));
          }
#pragma unroll
          for (int o = 1; o < 4; o <<= 1) {
            ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
            mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
          }
          const float sa8 = __fdiv_rn(ma, 127.f), sb8 = __fdiv_rn(mb, 127.f);
          const float ia = sa8 > 0.f ? __fdiv_rn(1.f, sa8) : 0.f, ib = sb8 > 0.f ? __fdiv_rn(1.f, sb8) : 0.f;
          const int la = wi * 16 + lane / 4, lb = la + 8;  // rows of this warpgroup's 64-row box
          if (threadIdx.x % 128 == 0) tma_store_wait_read();
          named_bar_sync(1 + wg, 128);
#pragma unroll
          for (int i = 0; i < 32; ++i) {  // column 8 i + 2 (lane % 4) + (q & 1): 128-byte atom i / 16, chunk (i % 16) / 2
            const uint32_t atom = d_base + (i / 16) * 8192, cb = 8 * (i % 2) + 2 * (lane % 4), ch = (i % 16) / 2;
            // round(v) for |v| <= 127: v + 1.5 * 2^23 has round(v) in its low mantissa bits (low byte = the int8 pattern)
            const unsigned short va = static_cast<unsigned short>(__byte_perm(
                __float_as_uint(fmaf(acc[4 * i], ia, 12582912.f)), __float_as_uint(fmaf(acc[4 * i + 1], ia, 12582912.f)), 0x0040));
            const unsigned short vb = static_cast<unsigned short>(__byte_perm(
                __float_as_uint(fmaf(acc[4 * i + 2], ib, 12582912.f)), __float_as_uint(fmaf(acc[4 * i + 3], ib, 12582912.f)), 0x0040));
            asm volatile("st.shared.u16 [%0], %1;" ::"r"(atom + la * 128 + ((ch ^ (la & 7)) << 4) + cb), "h"(va) : "memory");
            asm volatile("st.shared.u16 [%0], %1;" ::"r"(atom + lb * 128 + ((ch ^ (lb & 7)) << 4) + cb), "h"(vb) : "memory");
          }
          if (lane % 4 == 0) {
            args.ysc[static_cast<size_t>(info.y + wg * 64 + la) * kDnNB + n] = sa8;
            args.ysc[static_cast<size_t>(info.y + wg * 64 + lb) * kDnNB + n] = sb8;
          }
          fence_async_shared();
          named_bar_sync(1 + wg, 128);
          if (threadIdx.x % 128 == 0) {
#pragma unroll
            for (int a = 0; a < 2; ++a) tma_store_2d(&tm_d, d_base + a * 8192, n * 256 + a * 128, info.y + wg * 64);
            tma_store_commit();
          }
        } else if (!shared_tile) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            if (threadIdx.x % 128 == 0) tma_store_wait_read();
            named_bar_sync(1 + wg, 128);
#pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
              const int i = 16 * half + ii;
              stsm_x2(pack_bf16(ga * acc[4 * i], ga * acc[4 * i + 1]),
                      pack_bf16(gb * acc[4 * i + 2], gb * acc[4 * i + 3]),
                      row_addr + (ii / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
            }
            fence_async_shared();
            named_bar_sync(1 + wg, 128);
            if (threadIdx.x % 128 == 0) {
#pragma unroll
              for (int a = 0; a < 2; ++a)
                tma_store_2d(&tm_d, d_base + a * 8192, n * 256 + (2 * half + a) * 64, info.y + wg * 64);
              tma_store_commit();
            }
          }
        } else {
          const int my_row = wg * 64 + wi * 16 + lane / 2;
          const int my_tok = my_row < info.z ? args.sorted_tok[info.y + my_row] : 0;
          int my_pos[4];
          float my_w[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int kk = 4 * (lane % 2) + j;
            my_pos[j] = my_row < info.z ? args.pos_tk[my_tok * kTopK + kk] : 0;
            my_w[j] = my_row < info.z ? args.topk_w[my_tok * kTopK + kk] : 0.f;
          }
          uint32_t hi[32];
#pragma unroll
          for (int i = 16; i < 32; ++i) {
            hi[2 * (i - 16)] = pack_bf16(ga * acc[4 * i], ga * acc[4 * i + 1]);
            hi[2 * (i - 16) + 1] = pack_bf16(gb * acc[4 * i + 2], gb * acc[4 * i + 3]);
          }
          if (threadIdx.x % 128 == 0) tma_store_wait_read();
          named_bar_sync(1 + wg, 128);
#pragma unroll
          for (int i = 0; i < 16; ++i)
            stsm_x2(pack_bf16(ga * acc[4 * i], ga * acc[4 * i + 1]),
                    pack_bf16(gb * acc[4 * i + 2], gb * acc[4 * i + 3]),
                    row_addr + (i / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
          fence_async_shared();
          named_bar_sync(1 + wg, 128);
          if (threadIdx.x % 128 == 0)
            while (ld_acquire(args.done) < 2 * static_cast<int>(gridDim.x)) __nanosleep(64);
          named_bar_sync(1 + wg, 128);
          const int lh = lane / 16, lc = lane % 16;
#pragma unroll 1
          for (int half = 0; half < 2; ++half) {
            if (half == 1) {
              named_bar_sync(1 + wg, 128);
#pragma unroll
              for (int i = 16; i < 32; ++i)
                stsm_x2(hi[2 * (i - 16)], hi[2 * (i - 16) + 1],
                        row_addr + (i / 8 - 2) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
              named_bar_sync(1 + wg, 128);
            }
            const int col = n * 256 + half * 128 + lc * 8;
#pragma unroll 1
            for (int r4 = 0; r4 < 16; r4 += 4) {
              uint4 yv[2][kTopK];  // kY8: .x / .y hold the 8 int8 bytes
              float w[2][kTopK], ysv[2][kTopK];
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const int lr = r4 + 2 * j + lh;
                const bool ok = wg * 64 + wi * 16 + lr < info.z;
#pragma unroll
                for (int k = 0; k < kTopK; ++k) {
                  const int src = 2 * lr + k / 4;
                  const int pos = __shfl_sync(0xffffffffu, my_pos[k % 4], src);
                  w[j][k] = __shfl_sync(0xffffffffu, my_w[k % 4], src);
                  if constexpr (kY8) {
                    const uint2 b8 = ok ? __ldcg(reinterpret_cast<const uint2*>(
                                              reinterpret_cast<const uint8_t*>(args.y) + static_cast<size_t>(pos) * kH + col))
                                        : make_uint2(0, 0);
                    yv[j][k] = make_uint4(b8.x, b8.y, 0, 0);
                    ysv[j][k] = ok ? __ldcg(args.ysc + static_cast<size_t>(pos) * kDnNB + n) : 0.f;
                  } else {
                    yv[j][k] = ok ? __ldcg(reinterpret_cast<const uint4*>(args.y + static_cast<size_t>(pos) * kH + col))
                                  : make_uint4(0, 0, 0, 0);
                  }
                }
              }
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const int lr = r4 + 2 * j + lh;
                const int r = wi * 16 + lr;
                const int tok = __shfl_sync(0xffffffffu, my_tok, 2 * lr);
                if (wg * 64 + r < info.z) {
                  const uint4 sv = lds16(d_base + (lc / 8) * 8192 + r * 128 + (((lc % 8) ^ (r & 7)) << 4));
                  float a[8];
                  {
                    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&sv);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                      const float2 f = __bfloat1622float2(b[q]);
                      a[2 * q] = f.x, a[2 * q + 1] = f.y;
                    }
                  }
#pragma unroll
                  for (int k = 0; k < kTopK; ++k) {
                    if constexpr (kY8) {
                      // int8 b -> float exactly: 0x4B0000(b ^ 0x80) is 2^23 + b + 128
                      const float ws = w[j][k] * ysv[j][k];
                      const uint32_t wd[2] = {yv[j][k].x ^ 0x80808080u, yv[j][k].y ^ 0x80808080u};
#pragma unroll
                      for (int q = 0; q < 8; ++q)
                        a[q] = fmaf(ws, __uint_as_float(__byte_perm(wd[q / 4], 0x4B000000u, 0x7540 | (q % 4))) - 8388736.f,
                                    a[q]);
                    } else {
                    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&yv[j][k]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                      const float2 f = __bfloat1622float2(b[q]);
                      a[2 * q] = fmaf(w[j][k], f.x, a[2 * q]);
                      a[2 * q + 1] = fmaf(w[j][k], f.y, a[2 * q + 1]);
                    }
                    }
                  }
                  *reinterpret_cast<uint4*>(args.out + static_cast<size_t>(tok) * kH + col) =
                      make_uint4(pack_bf16(a[0], a[1]), pack_bf16(a[2], a[3]), pack_bf16(a[4], a[5]),
                                 pack_bf16(a[6], a[7]));
                }
              }
            }
          }
        }
      } else {
#endif
      if (threadIdx.x % 128 == 0) tma_store_wait_read();
      named_bar_sync(1 + wg, 128);
      const uint32_t row_addr = d_base + wi * 2048 + (lane % 16) * 128;
      if constexpr (kUp) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          float h[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float g = acc[4 * i + q], up = acc[4 * (i + 16) + q];
            h[q] = g / (1.f + __expf(-g)) * up;
          }
          stsm_x2(pack_bf16(h[0], h[1]), pack_bf16(h[2], h[3]),
                  row_addr + (i / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
        }
      } else {
        float ga = 1.f, gb = 1.f;
        if (shared_tile) {
          const int ra = wg * 64 + wi * 16 + lane / 4, rb = ra + 8;
          ga = ra < info.z ? args.sg[args.sorted_tok[info.y + ra]] : 0.f;
          gb = rb < info.z ? args.sg[args.sorted_tok[info.y + rb]] : 0.f;
        }
#pragma unroll
        for (int i = 0; i < 32; ++i)
          stsm_x2(pack_bf16(ga * acc[4 * i], ga * acc[4 * i + 1]), pack_bf16(gb * acc[4 * i + 2], gb * acc[4 * i + 3]),
                  row_addr + (i / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
      }
      fence_async_shared();
      named_bar_sync(1 + wg, 128);
      if (shared_tile) {
        const int my_row = wg * 64 + wi * 16 + lane / 2;
        const int my_tok = my_row < info.z ? args.sorted_tok[info.y + my_row] : 0;
        int my_pos[4];
        float my_w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int kk = 4 * (lane % 2) + j;
          my_pos[j] = my_row < info.z ? args.pos_tk[my_tok * kTopK + kk] : 0;
          my_w[j] = my_row < info.z ? args.topk_w[my_tok * kTopK + kk] : 0.f;
        }
        if (threadIdx.x % 128 == 0)
          while (ld_acquire(args.done) < 2 * static_cast<int>(gridDim.x)) __nanosleep(64);
        named_bar_sync(1 + wg, 128);
        const int col = n * 256 + lane * 8;
#pragma unroll 1
        for (int r2 = 0; r2 < 16; r2 += 2) {
          uint4 yv[2][kTopK];
          float w[2][kTopK];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const bool ok = wg * 64 + wi * 16 + r2 + j < info.z;
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
              const int src = 2 * (r2 + j) + k / 4;
              const int pos = __shfl_sync(0xffffffffu, my_pos[k % 4], src);
              w[j][k] = __shfl_sync(0xffffffffu, my_w[k % 4], src);
              yv[j][k] = ok ? __ldcg(reinterpret_cast<const uint4*>(args.y + static_cast<size_t>(pos) * kH + col))
                            : make_uint4(0, 0, 0, 0);
            }
          }
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int r = wi * 16 + r2 + j;
            const int tok = __shfl_sync(0xffffffffu, my_tok, 2 * (r2 + j));
            if (wg * 64 + r >= info.z) continue;
            const uint4 sv = lds16(d_base + (lane / 8) * 8192 + r * 128 + (((lane % 8) ^ (r & 7)) << 4));
            float a[8];
            {
              const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&sv);
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(b[q]);
                a[2 * q] = f.x, a[2 * q + 1] = f.y;
              }
            }
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
              const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&yv[j][k]);
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(b[q]);
                a[2 * q] = fmaf(w[j][k], f.x, a[2 * q]);
                a[2 * q + 1] = fmaf(w[j][k], f.y, a[2 * q + 1]);
              }
            }
            *reinterpret_cast<uint4*>(args.out + static_cast<size_t>(tok) * kH + col) =
                make_uint4(pack_bf16(a[0], a[1]), pack_bf16(a[2], a[3]), pack_bf16(a[4], a[5]), pack_bf16(a[6], a[7]));
          }
        }
      } else if (threadIdx.x % 128 == 0) {
        constexpr int kAtoms = kUp ? 2 : 4;
#pragma unroll
        for (int a = 0; a < kAtoms; ++a)
          tma_store_2d(&tm_d, d_base + a * 8192, n * (kUp ? 128 : 256) + a * 64, info.y + wg * 64);
        tma_store_commit();
      }
#if QMOE_DN_STAGES4
      }
#endif
    }
    if (!signaled) signal_routed_done();
    if (threadIdx.x % 128 == 0) tma_store_wait_all();
  }
  cluster_sync();
}

#if QMOE_Q8
// ---------------------------------------------------------------- up_i8g: the prefill MoE up on INT8 tensor cores
// A PRECISION CHANGE with i8x_up8.py's arithmetic class: x rows quantized per 128-column group (i8x_up8._quant_rows),
// the expert rows the per-channel INT8 copy, the shared expert's rows quantized per channel per call. Each k block
// (128 columns = one group) is an s32 sum from zero per n128 half (wgmma m64n128k32 s8), folded into fp32 accumulators
// with the row's group scale; the channel scale, silu(gate) * up and the bf16 h store follow as in the BF16 up.
// x groups are 256 columns with integer group scales (i8x_up8.quant_img): a row's group g has the scale xs[row] * m_g,
// m_g an integer in [1, 64] (m_g = ceil(64 amax_g / amax_row), so the scale is within a factor 1 + 1 / m_g of the group's
// own amax / 127). The two k blocks of a group sum into one s32 partial t_g, folded as acc += m_g * t_g (one IMAD per
// value: s32 conversions run at a quarter of the integer rate and bounded the fold); |acc| <= 8 * 64 * 256 * 127^2 <
// 2^31. The epilogue converts acc once and applies xs[row] with the channel scale.
// 256 threads = two math warpgroups (255 registers: 128 + 64 s32 live values), each gathering its own 64 A rows
// kGAhead k blocks ahead (cp.async, 4 x 16 B per thread per stage); thread 0 issues the B tiles by TMA. A stage is
// refilled once all eight warps released it (empty barrier), so the warpgroups drift apart and one folds while the
// other's wgmmas run. Units are (m-tile, n-block) of route_i8's zero-filled m-tile list, m-tile major.
constexpr int kGStages = 4, kGAhead = 2;
constexpr uint32_t kGA = 128 * 128, kGB = 256 * 128, kGD = 64 * 128 * 2;
constexpr uint32_t kGSmem = kGStages * (kGA + kGB) + 2 * kGD + 2 * 256 * 4 + 2 * kGStages * 8;
static_assert(kGSmem <= 232448, "up_i8g smem must fit the 227 KiB opt-in");
struct UpGArgs {
  const int4* mt;
  int n_units;  // m-tiles x 4 n-blocks
  const int* sorted_tok;
  const uint8_t* xq;   // [T, kH] int8
  const float* xs;     // [T] the row's base scale
  const int* xm;       // [T, kH / 256] the row's integer group multipliers
  const uint8_t* q8;   // the INT8 copy [kE][kQ8Expert]
  const float* ss13;   // [2 kI] shared expert channel scales
};
__global__ void __launch_bounds__(256, 1) up_g128_kernel(const UpGArgs a, const __grid_constant__ CUtensorMap tm_w,
                                                       const __grid_constant__ CUtensorMap tm_sw,
                                                       const __grid_constant__ CUtensorMap tm_h) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = sA + kGStages * kGA;
  uint8_t* sD = sB + kGStages * kGB;
  float* sScale = reinterpret_cast<float*>(sD + 2 * kGD);  // per warpgroup: the unit's [gate 128 | up 128] scales
  uint64_t* full = reinterpret_cast<uint64_t*>(sScale + 512);
  uint64_t* empty = full + kGStages;
  const int tid = threadIdx.x, wg = tid / 128, wt = tid % 128, warp = tid / 32, wi = warp % 4, lane = tid % 32;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kGStages; ++s) {
      mbar_init(&full[s], 256 + 1);  // 256 cp.async arrivals + thread 0's expect_tx
      mbar_init(&empty[s], 8);       // one per warp
    }
    fence_barrier_init();
  }
  if (tid == 32) {
    prefetch_tmap(&tm_w);
    prefetch_tmap(&tm_sw);
    prefetch_tmap(&tm_h);
  }
  __syncthreads();
  const uint64_t pol_x = kEvictNormal;  // evict_last here costs the down that follows ~5 %
  const int step = static_cast<int>(gridDim.x);
  auto next_active = [&](int u) {
    while (u < a.n_units && __ldg(&a.mt[u >> 2].z) <= 0) u += step;
    return u;
  };
  // producer: slot (pu, pkb) into pstage; this thread's A chunks: rows 64 wg + wt / 8 + 16 i, 16-byte chunk wt % 8
  int pu = next_active(static_cast<int>(blockIdx.x)), pkb = 0, pstage = 0;
  uint32_t pphase = 0;
  int4 pinfo = make_int4(0, 0, 0, 0);
  const uint8_t* psrc[4];
  auto load_unit = [&]() {
    pinfo = pu < a.n_units ? a.mt[pu >> 2] : make_int4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = 64 * wg + wt / 8 + 16 * i;
      psrc[i] = r < pinfo.z ? a.xq + static_cast<size_t>(a.sorted_tok[pinfo.y + r]) * kH + (wt % 8) * 16 : nullptr;
    }
  };
  load_unit();
  auto issue = [&]() {
    if (pu >= a.n_units) return;
    mbar_wait(&empty[pstage], pphase ^ 1);
    uint64_t* bar = &full[pstage];
    const uint32_t a_dst = smem_u32(sA + pstage * kGA);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = 64 * wg + wt / 8 + 16 * i;
      if (psrc[i] != nullptr) cp_async16(a_dst + r * 128 + (((wt % 8) ^ (r & 7)) << 4), psrc[i] + pkb * 128, pol_x);
    }
    cp_async_arrive_noinc(bar);
    if (tid == 0) {
      const int n = pu & 3;
      mbar_arrive_expect_tx(bar, kGB);
      const uint32_t b_dst = smem_u32(sB + pstage * kGB);
      const bool sh = pinfo.x == kE;
      const CUtensorMap* m = sh ? &tm_sw : &tm_w;
      const int brow = sh ? 0 : pinfo.x * (kQ8Expert / kH);
      tma_load_2d(b_dst, m, bar, pkb * 128, brow + n * 128, kEvictNormal);
      tma_load_2d(b_dst + kGB / 2, m, bar, pkb * 128, brow + kI + n * 128, kEvictNormal);
    }
    pstage = pstage + 1 == kGStages ? 0 : pstage + 1;
    pphase ^= pstage == 0;
    if (++pkb == kH / 128) {
      pkb = 0;
      pu = next_active(pu + step);
      load_unit();
    }
  };
#pragma unroll 1
  for (int s = 0; s < kGAhead; ++s) issue();
  int cstage = 0;
  uint32_t cphase = 0;
  const uint32_t a_base = smem_u32(sA) + wg * 64 * 128, b_base = smem_u32(sB);
  const uint32_t d_base = smem_u32(sD) + wg * kGD;
  float* sc = sScale + wg * 256;
  int acc[128];
#pragma unroll 1
  for (int cu = next_active(static_cast<int>(blockIdx.x)); cu < a.n_units; cu = next_active(cu + step)) {
    const int4 info = a.mt[cu >> 2];
    const int n = cu & 3;
    const bool active = info.z > wg * 64;
    const int ra = wg * 64 + wi * 16 + lane / 4, rb = ra + 8;
    const bool va = ra < info.z, vb = rb < info.z;
    const int ta = va ? a.sorted_tok[info.y + ra] : 0, tb = vb ? a.sorted_tok[info.y + rb] : 0;
    const int* ma_row = a.xm + static_cast<size_t>(ta) * (kH / 256);
    const int* mb_row = a.xm + static_cast<size_t>(tb) * (kH / 256);
    float csc[2];  // this thread's two of the unit's 256 channel scales, stored after the mainloop
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int t = wt + 128 * j, ch = (t < 128 ? 0 : kI - 128) + n * 128 + t;
      csc[j] = info.x == kE ? __ldg(a.ss13 + ch)
                            : __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(
                                  a.q8 + static_cast<size_t>(info.x) * kQ8Expert + kQ8C13) + ch * (kH / 128))));
    }
#pragma unroll
    for (int i = 0; i < 128; ++i) acc[i] = 0;
#pragma unroll 1
    for (int kp = 0; kp < kH / 256; ++kp) {  // k blocks 2 kp and 2 kp + 1: one 256-column x group
      issue();
      issue();
      mbar_wait(&full[cstage], cphase);
      const int st0 = cstage;
      cstage = cstage + 1 == kGStages ? 0 : cstage + 1;
      cphase ^= cstage == 0;
      mbar_wait(&full[cstage], cphase);
      const int st1 = cstage;
      if (active) {
        const int ma = va ? __ldg(ma_row + kp) : 0, mb = vb ? __ldg(mb_row + kp) : 0;
        const uint32_t a_s0 = a_base + st0 * kGA, b_s0 = b_base + st0 * kGB;
        const uint32_t a_s1 = a_base + st1 * kGA, b_s1 = b_base + st1 * kGB;
#pragma unroll
        for (int half = 0; half < 2; ++half) {  // B rows [gate 128 | up 128]
          int t[64];  // the first wgmma (scale_d 0) ignores its prior contents
          wgmma_fence();
#pragma unroll
          for (int k = 0; k < 4; ++k)
            wgmma_m64n128k32_s8(t, gmma_desc(a_s0 + k * 32), gmma_desc(b_s0 + half * (kGB / 2) + k * 32), k);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            wgmma_m64n128k32_s8(t, gmma_desc(a_s1 + k * 32), gmma_desc(b_s1 + half * (kGB / 2) + k * 32), 1);
          wgmma_commit();
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand(t[i]);
          wgmma_wait<0>();
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            int* f = acc + 64 * half + 4 * j;
            f[0] += t[4 * j] * ma;
            f[1] += t[4 * j + 1] * ma;
            f[2] += t[4 * j + 2] * mb;
            f[3] += t[4 * j + 3] * mb;
          }
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand(acc[64 * half + i]);
        }
      }
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(&empty[st0]);
        mbar_arrive(&empty[st1]);
      }
      cstage = cstage + 1 == kGStages ? 0 : cstage + 1;
      cphase ^= cstage == 0;
    }
    if (!active) continue;
    // acc[4 i + q]: row ra (q < 2) / rb, column 8 i + 2 (lane % 4) + (q & 1) of [gate 128 | up 128]
    const float xa = va ? __ldg(a.xs + ta) : 0.f, xb = vb ? __ldg(a.xs + tb) : 0.f;
    if (wt == 0) tma_store_wait_read();
    sc[wt] = csc[0], sc[wt + 128] = csc[1];
    named_bar_sync(1 + wg, 128);  // sc written, sD free
    const uint32_t row_addr = d_base + wi * 2048 + (lane % 16) * 128;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float2 wg2 = *reinterpret_cast<const float2*>(sc + 8 * i + 2 * (lane % 4));
      const float2 wu2 = *reinterpret_cast<const float2*>(sc + 128 + 8 * i + 2 * (lane % 4));
      float h[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float xr = q < 2 ? xa : xb;
        const float g = static_cast<float>(acc[4 * i + q]) * xr * ((q & 1) ? wg2.y : wg2.x);
        const float up = static_cast<float>(acc[4 * (i + 16) + q]) * xr * ((q & 1) ? wu2.y : wu2.x);
        h[q] = g / (1.f + __expf(-g)) * up;
      }
      stsm_x2(pack_bf16(h[0], h[1]), pack_bf16(h[2], h[3]), row_addr + (i / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
    }
    fence_async_shared();
    named_bar_sync(1 + wg, 128);  // sD staged, sc read
    if (wt == 0) {
#pragma unroll
      for (int at = 0; at < 2; ++at) tma_store_2d(&tm_h, d_base + at * 8192, n * 128 + at * 64, info.y + wg * 64);
      tma_store_commit();
    }
  }
  if (wt == 0) tma_store_wait_all();
}
#endif
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode_fn() {
  static EncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q));
    TORCH_CHECK(p != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled is unavailable");
    fn = reinterpret_cast<EncodeFn>(p);
  }
  return fn;
}
static CUtensorMap make_map(const void* base, uint64_t inner, uint64_t outer, uint32_t box_outer) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner * 2};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kBK), box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides,
                                 box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}
#if QMOE_Q8
// The INT8 copy [kE][kQ8Expert] bytes as 3D tensors: the up rows' int8 [kE][2 kI][kH] (box 64 B x 128 rows) and their
// fp16 group scales [kE][2 kI][kH / 128] at kQ8C13 (box 16 x 128 rows), no swizzle.
static CUtensorMap make_map_q8(const uint8_t* q8, bool scales) {
  CUtensorMap map;
  const cuuint64_t dims[3] = {scales ? static_cast<cuuint64_t>(kH / 128) : static_cast<cuuint64_t>(kH),
                              static_cast<cuuint64_t>(2 * kI), static_cast<cuuint64_t>(kE)};
  const cuuint64_t strides[2] = {scales ? static_cast<cuuint64_t>(kH / 128 * 2) : static_cast<cuuint64_t>(kH),
                                 static_cast<cuuint64_t>(kQ8Expert)};
  const cuuint32_t box[3] = {scales ? static_cast<cuuint32_t>(kH / 128) : static_cast<cuuint32_t>(kBK), 128, 1};
  const cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = encode_fn()(&map, scales ? CU_TENSOR_MAP_DATA_TYPE_UINT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                                 const_cast<uint8_t*>(scales ? q8 + kQ8C13 : q8), dims, strides, box, estr,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (INT8 copy) failed: ", static_cast<int>(r));
  return map;
}
#endif
#if QMOE_Q8
template <bool kUp, bool kQ8 = false, bool kY8 = false>
#else
template <bool kUp>
#endif
static void launch_gemm(const GemmArgs& args, const CUtensorMap& b, const CUtensorMap& bs, const CUtensorMap& a,
                        const CUtensorMap& d, cudaStream_t stream, const CUtensorMap* q8 = nullptr,
                        const CUtensorMap* q8s = nullptr) {
#if QMOE_Q8
  using Cfg = GemmCfg<kUp, kQ8>;
  TORCH_CHECK(!Cfg::kTma || (q8 != nullptr && q8s != nullptr), "the TMA'd INT8 up GEMM needs its tensor maps");
#else
  using Cfg = GemmCfg<kUp>;
#endif
#if QMOE_Q8
  auto kernel = moe_gemm_kernel<kUp, kQ8, kY8>;
#else
  auto kernel = moe_gemm_kernel<kUp>;
#endif
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeCooperative;
  attr[1].val.cooperative = kUp ? 0 : 1;
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = Cfg::kSmem;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  static int clusters_by_device[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index ", dev, " out of range");
  int& clusters = clusters_by_device[dev];
  if (clusters == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmem));
    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    cfg.gridDim = dim3(num_sms / 2 * 2);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, kernel, &cfg));
    TORCH_CHECK(clusters > 0, "the MoE GEMM cannot be resident as a 2-CTA cluster");
  }
  cfg.gridDim = dim3(2 * clusters);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args, b, bs, a, d, q8 ? *q8 : a, q8s ? *q8s : a));
}
static void check_bf16(const torch::Tensor& t, std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bf16 tensor");
  TORCH_CHECK(t.sizes().vec() == std::vector<int64_t>(shape), name, " has shape ", t.sizes(), ", expected ",
              std::vector<int64_t>(shape));
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}
std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor logits, torch::Tensor gw, torch::Tensor w13,
                                   torch::Tensor w2, torch::Tensor s13, torch::Tensor s2, torch::Tensor cnt) {
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range: ", T);
  check_bf16(x, {T, kH}, "x");
  check_bf16(logits, {T, kE}, "router logits");
  check_bf16(gw, {kH}, "shared_expert_gate weight");
  check_bf16(w13, {kE, 2 * kI, kH}, "w13");
  check_bf16(w2, {kE, kH, kI}, "w2");
  check_bf16(s13, {2 * kI, kH}, "shared gate_up weight");
  check_bf16(s2, {kH, kI}, "shared down weight");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == torch::kInt32 && cnt.numel() == kStripes * kE + 1,
              "cnt must be int32 [2049]");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  auto i32 = x.options().dtype(torch::kInt32);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto topk_ids = torch::empty({T, kTopK}, i32);
  auto topk_w = torch::empty({T, kTopK}, f32);
  auto rank = torch::empty({T * kTopK}, i32);
  auto pos_tk = torch::empty({T * kTopK}, i32);
  auto sg = torch::empty({T}, f32);
  auto offs = torch::empty({kE + 2}, i32);
  auto sbase = torch::empty({kStripes * kE}, i32);
  auto n_pairs = torch::empty({1}, i32);
  auto mt = torch::empty({max_mt, 4}, i32);
  auto pairs = torch::empty({max_mt, 4}, i32);
  auto sorted_tok = torch::empty({max_rows}, i32);
  auto h = torch::empty({max_rows, kI}, x.options());
  const int64_t max_routed_rows = T * kTopK + kE * (kRowAlign - 1);
  auto y = torch::empty({max_routed_rows, kH}, x.options());
  auto out = torch::empty({T, kH}, x.options());
  route_topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr()), T, cnt.data_ptr<int>(), topk_ids.data_ptr<int>(),
      topk_w.data_ptr<float>(), rank.data_ptr<int>(), sg.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  route_scan_kernel<<<1, 512, 0, stream>>>(T, cnt.data_ptr<int>(), sbase.data_ptr<int>(), offs.data_ptr<int>(),
                                           reinterpret_cast<int4*>(mt.data_ptr<int>()),
                                           reinterpret_cast<int4*>(pairs.data_ptr<int>()), n_pairs.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t n_sc = T * (kTopK + 1);
  route_scatter_kernel<<<(n_sc + 255) / 256, 256, 0, stream>>>(T, topk_ids.data_ptr<int>(), rank.data_ptr<int>(),
                                                               sbase.data_ptr<int>(), offs.data_ptr<int>(),
                                                               sorted_tok.data_ptr<int>(),
                                                               pos_tk.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GemmArgs args;
  args.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
  args.pairs = reinterpret_cast<const int4*>(pairs.data_ptr<int>());
  args.n_pairs = n_pairs.data_ptr<int>();
  args.sorted_tok = sorted_tok.data_ptr<int>();
  args.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  args.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
  args.pos_tk = pos_tk.data_ptr<int>();
  args.topk_w = topk_w.data_ptr<float>();
  args.sg = sg.data_ptr<float>();
  args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.done = cnt.data_ptr<int>() + kStripes * kE;
  const CUtensorMap m_w13 = make_map(w13.data_ptr(), kH, static_cast<uint64_t>(kE) * 2 * kI, 128);
  const CUtensorMap m_s13 = make_map(s13.data_ptr(), kH, 2 * kI, 128);
  const CUtensorMap m_h_st = make_map(h.data_ptr(), kI, max_rows, 64);
  launch_gemm<true>(args, m_w13, m_s13, m_h_st, m_h_st, stream);
  const CUtensorMap m_w2 = make_map(w2.data_ptr(), kI, static_cast<uint64_t>(kE) * kH, 256);
  const CUtensorMap m_s2 = make_map(s2.data_ptr(), kI, kH, 256);
  const CUtensorMap m_h_ld = make_map(h.data_ptr(), kI, max_rows, kBM);
  const CUtensorMap m_y = make_map(y.data_ptr(), kH, max_routed_rows, 64);
  launch_gemm<false>(args, m_w2, m_s2, m_h_ld, m_y, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, topk_ids, topk_w};
}
#if QMOE_Q8
std::vector<torch::Tensor> forward_i8(torch::Tensor x, torch::Tensor logits, torch::Tensor gw, torch::Tensor q8,
                                      torch::Tensor w13, torch::Tensor w2, torch::Tensor s13, torch::Tensor s2,
                                      torch::Tensor cnt, int64_t up_q8, int64_t down_q8, int64_t parts,
                                      int64_t mode) {
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range: ", T);
  TORCH_CHECK(parts == 1 || parts == 3 || parts == 7, "parts must be 1, 3 or 7");
  TORCH_CHECK(mode >= 0 && mode < 16, "mode must be kMode* bits");
  check_bf16(x, {T, kH}, "x");
  check_bf16(logits, {T, kE}, "router logits");
  check_bf16(gw, {kH}, "shared_expert_gate weight");
  if (up_q8 || down_q8) {
    TORCH_CHECK(q8.is_cuda() && q8.scalar_type() == torch::kUInt8 && q8.is_contiguous() && q8.dim() == 2 &&
                    q8.size(0) == kE && q8.size(1) == kQ8Expert,
                "q8 must be the contiguous CUDA uint8 [256, ", kQ8Expert, "] INT8 copy");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q8.data_ptr()) % 16 == 0, "q8 must be 16-byte aligned");
    TORCH_CHECK(q8.device() == x.device(), "q8 must be on x's device");
  }
  if (!up_q8) check_bf16(w13, {kE, 2 * kI, kH}, "w13");
  if (!down_q8) check_bf16(w2, {kE, kH, kI}, "w2");
  check_bf16(s13, {2 * kI, kH}, "shared gate_up weight");
  check_bf16(s2, {kH, kI}, "shared down weight");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == torch::kInt32 && cnt.numel() == kStripes * kE + 1,
              "cnt must be int32 [2049]");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  auto i32 = x.options().dtype(torch::kInt32);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto topk_ids = torch::empty({T, kTopK}, i32);
  auto topk_w = torch::empty({T, kTopK}, f32);
  auto rank = torch::empty({T * kTopK}, i32);
  auto pos_tk = torch::empty({T * kTopK}, i32);
  auto sg = torch::empty({T}, f32);
  auto offs = torch::empty({kE + 2}, i32);
  auto sbase = torch::empty({kStripes * kE}, i32);
  auto n_pairs = torch::empty({1}, i32);
  auto mt = torch::empty({max_mt, 4}, i32);
  auto pairs = torch::empty({max_mt, 4}, i32);
  auto sorted_tok = torch::empty({max_rows}, i32);
  auto h = torch::empty({max_rows, kI}, x.options());
  const int64_t max_routed_rows = T * kTopK + kE * (kRowAlign - 1);
  auto y = torch::empty({max_routed_rows, kH}, x.options());
  auto out = torch::empty({T, kH}, x.options());
  route_topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr()), T, cnt.data_ptr<int>(), topk_ids.data_ptr<int>(),
      topk_w.data_ptr<float>(), rank.data_ptr<int>(), sg.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  route_scan_kernel<<<1, 512, 0, stream>>>(T, cnt.data_ptr<int>(), sbase.data_ptr<int>(), offs.data_ptr<int>(),
                                           reinterpret_cast<int4*>(mt.data_ptr<int>()),
                                           reinterpret_cast<int4*>(pairs.data_ptr<int>()), n_pairs.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t n_sc = T * (kTopK + 1);
  route_scatter_kernel<<<(n_sc + 255) / 256, 256, 0, stream>>>(T, topk_ids.data_ptr<int>(), rank.data_ptr<int>(),
                                                               sbase.data_ptr<int>(), offs.data_ptr<int>(),
                                                               sorted_tok.data_ptr<int>(),
                                                               pos_tk.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GemmArgs args;
  args.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
  args.pairs = reinterpret_cast<const int4*>(pairs.data_ptr<int>());
  args.n_pairs = n_pairs.data_ptr<int>();
  args.sorted_tok = sorted_tok.data_ptr<int>();
  args.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  args.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
  args.pos_tk = pos_tk.data_ptr<int>();
  args.topk_w = topk_w.data_ptr<float>();
  args.sg = sg.data_ptr<float>();
  args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.done = cnt.data_ptr<int>() + kStripes * kE;
  args.q8 = (up_q8 || down_q8) ? reinterpret_cast<const uint8_t*>(q8.data_ptr()) : nullptr;
  args.mode = static_cast<int>(mode);
  if (parts & 2) {
    const CUtensorMap m_s13 = make_map(s13.data_ptr(), kH, 2 * kI, 128);
    const CUtensorMap m_h_st = make_map(h.data_ptr(), kI, max_rows, 64);
    if (up_q8) {
      const CUtensorMap m_q8 = make_map_q8(reinterpret_cast<const uint8_t*>(q8.data_ptr()), false);
      const CUtensorMap m_q8s = make_map_q8(reinterpret_cast<const uint8_t*>(q8.data_ptr()), true);
      launch_gemm<true, true>(args, m_s13, m_s13, m_h_st, m_h_st, stream, &m_q8, &m_q8s);
    } else {
      const CUtensorMap m_w13 = make_map(w13.data_ptr(), kH, static_cast<uint64_t>(kE) * 2 * kI, 128);
      launch_gemm<true, false>(args, m_w13, m_s13, m_h_st, m_h_st, stream);
    }
  }
  if (parts & 4) {
    const CUtensorMap m_s2 = make_map(s2.data_ptr(), kI, kH, 256);
    const CUtensorMap m_h_ld = make_map(h.data_ptr(), kI, max_rows, kBM);
    const CUtensorMap m_y = make_map(y.data_ptr(), kH, max_routed_rows, 64);
    if (down_q8) {
      launch_gemm<false, true>(args, m_s2, m_s2, m_h_ld, m_y, stream);
    } else {
      const CUtensorMap m_w2 = make_map(w2.data_ptr(), kI, static_cast<uint64_t>(kE) * kH, 256);
      launch_gemm<false, false>(args, m_w2, m_s2, m_h_ld, m_y, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, topk_ids, topk_w};
}
// forward_i8 split around an external up GEMM (after 51f3e0f9's route_i8 / down_i8): the block routing alone (mt
// zero-filled: the external up reads every row of it), then the held-BF16 down GEMM + combine over its h.
std::vector<torch::Tensor> route_i8(torch::Tensor x, torch::Tensor logits, torch::Tensor gw, torch::Tensor cnt) {
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range: ", T);
  check_bf16(x, {T, kH}, "x");
  check_bf16(logits, {T, kE}, "router logits");
  check_bf16(gw, {kH}, "shared_expert_gate weight");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == torch::kInt32 && cnt.numel() == kStripes * kE + 1,
              "cnt must be int32 [2049]");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  auto i32 = x.options().dtype(torch::kInt32);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto topk_ids = torch::empty({T, kTopK}, i32);
  auto topk_w = torch::empty({T, kTopK}, f32);
  auto rank = torch::empty({T * kTopK}, i32);
  auto pos_tk = torch::empty({T * kTopK}, i32);
  auto sg = torch::empty({T}, f32);
  auto offs = torch::empty({kE + 2}, i32);
  auto sbase = torch::empty({kStripes * kE}, i32);
  auto n_pairs = torch::empty({1}, i32);
  auto mt = torch::zeros({max_mt, 4}, i32);
  auto pairs = torch::empty({max_mt, 4}, i32);
  auto sorted_tok = torch::empty({max_rows}, i32);
  route_topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr()), T, cnt.data_ptr<int>(), topk_ids.data_ptr<int>(),
      topk_w.data_ptr<float>(), rank.data_ptr<int>(), sg.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  route_scan_kernel<<<1, 512, 0, stream>>>(T, cnt.data_ptr<int>(), sbase.data_ptr<int>(), offs.data_ptr<int>(),
                                           reinterpret_cast<int4*>(mt.data_ptr<int>()),
                                           reinterpret_cast<int4*>(pairs.data_ptr<int>()), n_pairs.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t n_sc = T * (kTopK + 1);
  route_scatter_kernel<<<(n_sc + 255) / 256, 256, 0, stream>>>(T, topk_ids.data_ptr<int>(), rank.data_ptr<int>(),
                                                               sbase.data_ptr<int>(), offs.data_ptr<int>(),
                                                               sorted_tok.data_ptr<int>(),
                                                               pos_tk.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {topk_ids, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok};
}
// Row-major uint8 [outer, inner] in boxes of [box_outer rows, 128 bytes], 128B swizzle.
static CUtensorMap make_map_u8(const void* base, uint64_t inner, uint64_t outer, uint32_t box_outer) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner};
  const cuuint32_t box[2] = {128, box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (uint8) failed: ", static_cast<int>(r));
  return map;
}
// route_i8's up GEMM on INT8 operands (up_g128_kernel): xq / xs / xm the rows of x as int8 with a base scale per row and
// an integer multiplier per 256-column group (i8x_up8.quant_img), q8 the per-channel INT8 copy, sq / ss the shared expert's gate/up rows as int8
// with one fp32 scale per row. Returns h [max_rows, kI] where the BF16 up writes it.
torch::Tensor up_i8g(torch::Tensor x, torch::Tensor xq, torch::Tensor xs, torch::Tensor xm, torch::Tensor q8, torch::Tensor sq,
                     torch::Tensor ss, torch::Tensor mt, torch::Tensor sorted_tok) {
  const int64_t T = x.size(0);
  check_bf16(x, {T, kH}, "x");
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kInt8 && xq.is_contiguous() && xq.size(0) == T &&
                  xq.size(1) == kH, "xq int8 [T, 2048]");
  TORCH_CHECK(xs.is_cuda() && xs.scalar_type() == torch::kFloat32 && xs.is_contiguous() && xs.numel() == T,
              "xs fp32 [T]");
  TORCH_CHECK(xm.is_cuda() && xm.scalar_type() == torch::kInt32 && xm.is_contiguous() && xm.numel() == T * (kH / 256),
              "xm int32 [T, 8] (256-column groups)");
  TORCH_CHECK(q8.is_cuda() && q8.scalar_type() == torch::kUInt8 && q8.is_contiguous() && q8.dim() == 2 &&
                  q8.size(0) == kE && q8.size(1) == kQ8Expert, "q8 uint8 [256, Q8_EXPERT]");
  TORCH_CHECK(sq.is_cuda() && sq.scalar_type() == torch::kInt8 && sq.is_contiguous() && sq.size(0) == 2 * kI &&
                  sq.size(1) == kH, "sq int8 [1024, 2048]");
  TORCH_CHECK(ss.is_cuda() && ss.scalar_type() == torch::kFloat32 && ss.is_contiguous() && ss.numel() == 2 * kI,
              "ss fp32 [1024]");
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  TORCH_CHECK(sorted_tok.numel() == max_rows && mt.numel() == max_mt * 4, "route_i8's sorted_tok / mt");
  static_assert(kQ8Expert % kH == 0, "an expert block is a whole number of 2048-byte rows");
  const c10::cuda::CUDAGuard guard(x.device());
  auto h = torch::empty({max_rows, kI}, x.options());
  UpGArgs args;
  args.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
  args.n_units = static_cast<int>(max_mt * 4);
  args.sorted_tok = sorted_tok.data_ptr<int>();
  args.xq = reinterpret_cast<const uint8_t*>(xq.data_ptr());
  args.xs = xs.data_ptr<float>();
  args.xm = xm.data_ptr<int>();
  args.q8 = reinterpret_cast<const uint8_t*>(q8.data_ptr());
  args.ss13 = ss.data_ptr<float>();
  const CUtensorMap m_w = make_map_u8(q8.data_ptr(), kH, static_cast<uint64_t>(kE) * (kQ8Expert / kH), 128);
  const CUtensorMap m_sw = make_map_u8(sq.data_ptr(), kH, 2 * kI, 128);
  const CUtensorMap m_h = make_map(h.data_ptr(), kI, max_rows, 64);
  static bool attr[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index out of range");
  if (!attr[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(up_g128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kGSmem));
    attr[dev] = true;
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  up_g128_kernel<<<sms, 256, kGSmem, at::cuda::getCurrentCUDAStream()>>>(args, m_w, m_sw, m_h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
// y8: y in int8 with per-row x 256-column scales (kY8, a PRECISION CHANGE: the caller gates it by layer), else bf16.
torch::Tensor down_i8(torch::Tensor x, torch::Tensor w2, torch::Tensor s2, torch::Tensor cnt, torch::Tensor h,
                      torch::Tensor topk_w, torch::Tensor pos_tk, torch::Tensor sg, torch::Tensor n_pairs,
                      torch::Tensor mt, torch::Tensor pairs, torch::Tensor sorted_tok, bool y8) {
  const int64_t T = x.size(0);
  check_bf16(x, {T, kH}, "x");
  check_bf16(w2, {kE, kH, kI}, "w2");
  check_bf16(s2, {kH, kI}, "shared down weight");
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  check_bf16(h, {max_rows, kI}, "h");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == torch::kInt32 && cnt.numel() == kStripes * kE + 1,
              "cnt must be int32 [2049]");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t max_routed_rows = T * kTopK + kE * (kRowAlign - 1);
  auto y = y8 ? torch::empty({max_routed_rows, kH}, x.options().dtype(torch::kUInt8))  // int8 bytes
              : torch::empty({max_routed_rows, kH}, x.options());
  auto ysc = torch::empty({y8 ? max_routed_rows : 0, kDnNB}, x.options().dtype(torch::kFloat32));
  auto out = torch::empty({T, kH}, x.options());
  GemmArgs args;
  args.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
  args.pairs = reinterpret_cast<const int4*>(pairs.data_ptr<int>());
  args.n_pairs = n_pairs.data_ptr<int>();
  args.sorted_tok = sorted_tok.data_ptr<int>();
  args.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  args.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
  args.ysc = ysc.data_ptr<float>();
  args.pos_tk = pos_tk.data_ptr<int>();
  args.topk_w = topk_w.data_ptr<float>();
  args.sg = sg.data_ptr<float>();
  args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.done = cnt.data_ptr<int>() + kStripes * kE;
  args.q8 = nullptr;
  args.mode = 0;
  const CUtensorMap m_w2 = make_map(w2.data_ptr(), kI, static_cast<uint64_t>(kE) * kH, 256);
  const CUtensorMap m_s2 = make_map(s2.data_ptr(), kI, kH, 256);
  const CUtensorMap m_h_ld = make_map(h.data_ptr(), kI, max_rows, kBM);
  if (y8) {
    const CUtensorMap m_y = make_map_u8(y.data_ptr(), kH, max_routed_rows, 64);
    launch_gemm<false, false, true>(args, m_w2, m_s2, m_h_ld, m_y, stream);
  } else {
    const CUtensorMap m_y = make_map(y.data_ptr(), kH, max_routed_rows, 64);
    launch_gemm<false, false, false>(args, m_w2, m_s2, m_h_ld, m_y, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
#endif
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qmoe::forward, "Qwen3.6 MoE prefill block (routing + gate/up + down + combine)");
  m.attr("COUNTERS") = qmoe::kStripes * qmoe::kE + 1;
#if QMOE_Q8
  m.def("forward_i8", &qmoe::forward_i8,
        "the MoE prefill block of a relocated layer: INT8 copy up and/or down (lane I8X), else the king's BF16");
  m.attr("Q8_EXPERT") = qmoe::kQ8Expert;
  m.attr("MODE_NO_MMA") = qmoe::kModeNoMma;
  m.attr("MODE_NO_LOAD") = qmoe::kModeNoLoad;
  m.attr("MODE_NO_CVT") = qmoe::kModeNoCvt;
  m.attr("MODE_FAST_FOLD") = qmoe::kModeFastFold;
  m.def("route_i8", &qmoe::route_i8, "the block routing alone (topk, scan, scatter), mt zero-filled");
  m.def("down_i8", &qmoe::down_i8, "the down GEMM + combine over an external h");
  m.def("up_i8g", &qmoe::up_i8g, "route_i8's up GEMM on per-group INT8 rows and the per-channel INT8 copy");
#endif
}
