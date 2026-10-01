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
__device__ __forceinline__ void named_bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ float tanh_approx(float v) {  // tanh.approx.f32: max relative error ~2^-11
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
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
// route_i8q's extras, computed in the routing launch (a warp holds a token's whole x row for the shared gate anyway):
// x as int8 for up_i8g with i8x_up8._quant_img's arithmetic (lane chunk j = 256-column group j), and the shared
// expert's gate/up rows as int8 per channel with i8x_kernels._quant_row_kernel's arithmetic (the grid's first
// 2 kI / 8 blocks, so they run beside the token blocks rather than after them).
struct RouteQuant {
  int8_t* xq;                // [T, kH]
  float* xs;                 // [T] the row's base scale
  int* xm;                   // [T, kH / 256] the row's integer group multipliers
  const __nv_bfloat16* s13;  // [2 kI, kH]
  int8_t* sq;                // [2 kI, kH]
  float* ss;                 // [2 kI] (fp16-rounded channel scales)
  int w_blocks;  // leading blocks that quantize the shared rows (8 rows each)
  float qm;      // KNOB: the IMG multiplier cap M (64 = y19)
};
__device__ __forceinline__ float div_rn(float a, float b) {
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float div_tr(float a, float b) {  // Triton's default fp32 '/' (bitwise with quant_img)
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rnd_away(float v) { return v >= 0.f ? floorf(v + 0.5f) : -floorf(0.5f - v); }
__device__ __forceinline__ uint32_t q8_byte(float v) {  // v integral in [-127, 127]
  return static_cast<uint32_t>(static_cast<int>(v)) & 0xffu;
}
template <bool kQuant = false>
__global__ void __launch_bounds__(256, kQuant ? 2 : 4) route_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                                         const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ gw, int T,
                                                         int* __restrict__ cnt, int* __restrict__ topk_ids,
                                                         float* __restrict__ topk_w, int* __restrict__ rank,
                                                         float* __restrict__ sg, const RouteQuant rq = RouteQuant{}) {
  __shared__ uint4 s_gw[kH / 8];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if constexpr (kQuant) {
    if (static_cast<int>(blockIdx.x) < rq.w_blocks) {  // one shared-expert row per warp
      const int row = static_cast<int>(blockIdx.x) * 8 + warp;
      if (row >= 2 * kI) return;
      const uint4* wr = reinterpret_cast<const uint4*>(rq.s13 + static_cast<size_t>(row) * kH);
      uint4 wa[kH / 8 / 32];
      float m = 0.f;
#pragma unroll
      for (int j = 0; j < kH / 8 / 32; ++j) {
        wa[j] = wr[lane + 32 * j];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&wa[j]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = __bfloat1622float2(a2[q]);
          m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
      }
      const float amax = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(m)));
      const float sc = __half2float(__float2half_rn(__fmul_rn(div_rn(amax, 127.f), 1.0009765625f)));
      const float sd = sc > 0.f ? sc : 1.f;
      uint2* dst = reinterpret_cast<uint2*>(rq.sq + static_cast<size_t>(row) * kH);
#pragma unroll
      for (int j = 0; j < kH / 8 / 32; ++j) {
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&wa[j]);
        uint32_t wd[2] = {0u, 0u};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = __bfloat1622float2(a2[q]);
          const float r0 = fminf(fmaxf(rnd_away(div_rn(f.x, sd)), -127.f), 127.f);
          const float r1 = fminf(fmaxf(rnd_away(div_rn(f.y, sd)), -127.f), 127.f);
          wd[q / 2] |= (q8_byte(r0) | (q8_byte(r1) << 8)) << (16 * (q % 2));
        }
        dst[lane + 32 * j] = make_uint2(wd[0], wd[1]);
      }
      if (lane == 0) rq.ss[row] = sc;
      return;
    }
  }
  const int t = (static_cast<int>(blockIdx.x) - (kQuant ? rq.w_blocks : 0)) * 8 + warp;
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
  if constexpr (kQuant) {  // group g's scale is xs * m_g, m_g = ceil(64 amax_g / amax_row) in [1, 64]
    float ga[kH / 8 / 32];
#pragma unroll
    for (int j = 0; j < kH / 8 / 32; ++j) {
      const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&xa[j]);
      float m = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(a2[q]);
        m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
      ga[j] = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(m)));
    }
    float am = ga[0];
#pragma unroll
    for (int j = 1; j < kH / 8 / 32; ++j) am = fmaxf(am, ga[j]);
    am = fmaxf(am, 1e-30f);
    const float xsv = div_tr(am, 127.f * rq.qm);
    uint2* dst = reinterpret_cast<uint2*>(rq.xq + static_cast<size_t>(t) * kH);
    int mine = 0;
#pragma unroll
    for (int j = 0; j < kH / 8 / 32; ++j) {
      const float mj = fminf(fmaxf(ceilf(div_tr(ga[j] * rq.qm, am)), 1.f), rq.qm);
      if (lane == j) mine = static_cast<int>(mj);
      const float sm = xsv * mj;
      const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&xa[j]);
      uint32_t wd[2] = {0u, 0u};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(a2[q]);
        const float r0 = fminf(fmaxf(rnd_away(div_tr(f.x, sm)), -127.f), 127.f);
        const float r1 = fminf(fmaxf(rnd_away(div_tr(f.y, sm)), -127.f), 127.f);
        wd[q / 2] |= (q8_byte(r0) | (q8_byte(r1) << 8)) << (16 * (q % 2));
      }
      dst[lane + 32 * j] = make_uint2(wd[0], wd[1]);
    }
    if (lane == 0) rq.xs[t] = xsv;
    if (lane < kH / 256) rq.xm[static_cast<size_t>(t) * (kH / 256) + lane] = mine;
  }
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
  float ydiv = 127.f;      // KNOB: kY8's scale = amax / ydiv (127 = y19; the default keeps every launcher that does not set it on y18's /127)
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
template <bool kUp, bool kQ8 = false, bool kY8 = false, bool kY128 = false>
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
          const float sa8 = __fdiv_rn(ma, args.ydiv), sb8 = __fdiv_rn(mb, args.ydiv);
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
                    // kY128 (port, our W8A8 down's y): one scale per row and 128-column block, [rows][2 kDnNB]
                    ysv[j][k] = ok ? __ldcg(kY128 ? args.ysc + static_cast<size_t>(pos) * (2 * kDnNB) + 2 * n + half
                                                  : args.ysc + static_cast<size_t>(pos) * kDnNB + n)
                                   : 0.f;
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
template <bool kTanh>
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
        if constexpr (kTanh) h[q] = 0.5f * g * (1.f + tanh_approx(0.5f * g)) * up;  // silu(g) = g sigmoid(g), one MUFU op
        else h[q] = g / (1.f + __expf(-g)) * up;  // KNOB: y17's silu
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
// up_i8gp (k2fp8y54k, bit for bit up_i8g): warp-specialized. WG2 (threads 256..383) is the producer: every A row
// gather (cp.async, 8 x 16 B per thread per stage) and the B tiles (TMA, its thread 0), up to kGStages stages ahead;
// WG0 / WG1 only run the wgmmas, the integer group fold and the epilogue (setmaxnreg: producer 56, consumers 224;
// 40 / 232 spills the producer, 72 / 216 the consumers: T 8192 -4.7 %, T 4096 -0.3 % vs up_i8g, h bit for bit).
// The s32 group sums are exact and the fold / epilogue are up_i8g's, so h is identical.
template <bool kTanh>
__global__ void __launch_bounds__(384, 1) up_g128p_kernel(const UpGArgs a, const __grid_constant__ CUtensorMap tm_w,
                                                        const __grid_constant__ CUtensorMap tm_sw,
                                                        const __grid_constant__ CUtensorMap tm_h) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = sA + kGStages * kGA;
  uint8_t* sD = sB + kGStages * kGB;
  float* sScale = reinterpret_cast<float*>(sD + 2 * kGD);
  uint64_t* full = reinterpret_cast<uint64_t*>(sScale + 512);
  uint64_t* empty = full + kGStages;
  const int tid = threadIdx.x, wg = tid / 128, wt = tid % 128, warp = tid / 32, wi = warp % 4, lane = tid % 32;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kGStages; ++s) {
      mbar_init(&full[s], 128 + 1);  // the producer's 128 cp.async arrivals + its thread 0's expect_tx
      mbar_init(&empty[s], 8);       // one per consumer warp
    }
    fence_barrier_init();
  }
  if (tid == 32) {
    prefetch_tmap(&tm_w);
    prefetch_tmap(&tm_sw);
    prefetch_tmap(&tm_h);
  }
  __syncthreads();
  const int step = static_cast<int>(gridDim.x);
  auto next_active = [&](int u) {
    while (u < a.n_units && __ldg(&a.mt[u >> 2].z) <= 0) u += step;
    return u;
  };
  if (wg == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");
    const uint64_t pol_x = kEvictNormal;
    const int pt = wt, chunk = pt % 8;
    int stage = 0;
    uint32_t phase = 0;
#pragma unroll 1
    for (int u = next_active(static_cast<int>(blockIdx.x)); u < a.n_units; u = next_active(u + step)) {
      const int4 info = a.mt[u >> 2];
      const int n = u & 3;
      const uint8_t* src[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = pt / 8 + 16 * i;
        src[i] = r < info.z ? a.xq + static_cast<size_t>(a.sorted_tok[info.y + r]) * kH + chunk * 16 : nullptr;
      }
      const bool sh = info.x == kE;
      const CUtensorMap* m = sh ? &tm_sw : &tm_w;
      const int brow = sh ? 0 : info.x * (kQ8Expert / kH);
#pragma unroll 1
      for (int kb = 0; kb < kH / 128; ++kb) {
        mbar_wait(&empty[stage], phase ^ 1);
        uint64_t* bar = &full[stage];
        const uint32_t a_dst = smem_u32(sA + stage * kGA);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = pt / 8 + 16 * i;
          if (src[i] != nullptr) cp_async16(a_dst + r * 128 + ((chunk ^ (r & 7)) << 4), src[i] + kb * 128, pol_x);
        }
        cp_async_arrive_noinc(bar);
        if (pt == 0) {
          mbar_arrive_expect_tx(bar, kGB);
          const uint32_t b_dst = smem_u32(sB + stage * kGB);
          tma_load_2d(b_dst, m, bar, kb * 128, brow + n * 128, kEvictNormal);
          tma_load_2d(b_dst + kGB / 2, m, bar, kb * 128, brow + kI + n * 128, kEvictNormal);
        }
        stage = stage + 1 == kGStages ? 0 : stage + 1;
        phase ^= stage == 0;
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n" ::: "memory");
  int cstage = 0;
  uint32_t cphase = 0;
  const uint32_t a_base = smem_u32(sA) + wg * 64 * 128, b_base = smem_u32(sB);
  const uint32_t d_base = smem_u32(sD) + wg * kGD;
  float* sc = sScale + wg * 256;
  int acc[128];
  bool first = true;  // k2fp8y59k: WG0's first batch needs no go
#pragma unroll 1
  for (int cu = next_active(static_cast<int>(blockIdx.x)); cu < a.n_units; cu = next_active(cu + step)) {
    const int4 info = a.mt[cu >> 2];
    const int n = cu & 3;
    const bool active = info.z > wg * 64;  // a warpgroup past the m-tile's rows still takes its turns (no wgmmas)
    const int ra = wg * 64 + wi * 16 + lane / 4, rb = ra + 8;
    const bool va = ra < info.z, vb = rb < info.z;
    const int ta = va ? a.sorted_tok[info.y + ra] : 0, tb = vb ? a.sorted_tok[info.y + rb] : 0;
    const int* ma_row = a.xm + static_cast<size_t>(ta) * (kH / 256);
    const int* mb_row = a.xm + static_cast<size_t>(tb) * (kH / 256);
    float csc[2];
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
      mbar_wait(&full[cstage], cphase);
      const int st0 = cstage;
      cstage = cstage + 1 == kGStages ? 0 : cstage + 1;
      cphase ^= cstage == 0;
      mbar_wait(&full[cstage], cphase);
      const int st1 = cstage;
      {
        const int ma = va ? __ldg(ma_row + kp) : 0, mb = vb ? __ldg(mb_row + kp) : 0;
        const uint32_t a_s0 = a_base + st0 * kGA, b_s0 = b_base + st0 * kGB;
        const uint32_t a_s1 = a_base + st1 * kGA, b_s1 = b_base + st1 * kGB;
#pragma unroll
        for (int half = 0; half < 2; ++half) {  // B rows [gate 128 | up 128]
          int t[64];
          // k2fp8y59k: the two consumer warpgroups take turns issuing their 8-wgmma batches (named barriers 5 / 6,
          // go given right after the issue): the tensor cores then run the batches back to back instead of
          // interleaving both, so each warpgroup folds while the other's batch runs (T8192 -5..-8 %). Same wgmmas and
          // fold per warpgroup: h bit for bit.
          if (wg == 0) {
            if (!first) named_bar_sync(6, 256);
          } else {
            named_bar_sync(5, 256);
          }
          first = false;
          if (active) {
            wgmma_fence();
#pragma unroll
            for (int k = 0; k < 4; ++k)
              wgmma_m64n128k32_s8(t, gmma_desc(a_s0 + k * 32), gmma_desc(b_s0 + half * (kGB / 2) + k * 32), k);
#pragma unroll
            for (int k = 0; k < 4; ++k)
              wgmma_m64n128k32_s8(t, gmma_desc(a_s1 + k * 32), gmma_desc(b_s1 + half * (kGB / 2) + k * 32), 1);
            wgmma_commit();
          }
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand(t[i]);
          named_bar_arrive(wg == 0 ? 5 : 6, 256);
          wgmma_wait<0>();
          // unconditional fold (a branch here costs 1-5 %): past the m-tile's rows ma = mb = 0 and acc is never stored
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
    const float xa = va ? __ldg(a.xs + ta) : 0.f, xb = vb ? __ldg(a.xs + tb) : 0.f;
    if (wt == 0) tma_store_wait_read();
    sc[wt] = csc[0], sc[wt + 128] = csc[1];
    named_bar_sync(1 + wg, 128);
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
        if constexpr (kTanh) h[q] = 0.5f * g * (1.f + tanh_approx(0.5f * g)) * up;
        else h[q] = g / (1.f + __expf(-g)) * up;
      }
      stsm_x2(pack_bf16(h[0], h[1]), pack_bf16(h[2], h[3]), row_addr + (i / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
    }
    fence_async_shared();
    named_bar_sync(1 + wg, 128);
    if (wt == 0) {
#pragma unroll
      for (int at = 0; at < 2; ++at) tma_store_2d(&tm_h, d_base + at * 8192, n * 128 + at * 64, info.y + wg * 64);
      tma_store_commit();
    }
  }
  if (wg == 0 && !first) named_bar_sync(6, 256);  // the last go of WG1
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
template <bool kUp, bool kQ8 = false, bool kY8 = false, bool kY128 = false>
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
  auto kernel = moe_gemm_kernel<kUp, kQ8, kY8, kY128>;
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
// route_i8 plus, in the same topk launch, up_i8g's operands: x as int8 with a base scale and integer multiplier per
// 256-column group (i8x_up8.quant_img's values) and the shared gate/up rows as int8 per channel (quantize_channel's).
std::vector<torch::Tensor> route_i8q(torch::Tensor x, torch::Tensor logits, torch::Tensor gw, torch::Tensor cnt,
                                     torch::Tensor s13, double qm) {
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range: ", T);
  check_bf16(x, {T, kH}, "x");
  check_bf16(logits, {T, kE}, "router logits");
  check_bf16(gw, {kH}, "shared_expert_gate weight");
  check_bf16(s13, {2 * kI, kH}, "shared gate_up weight");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == torch::kInt32 && cnt.numel() == kStripes * kE + 1,
              "cnt must be int32 [2049]");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  auto i32 = x.options().dtype(torch::kInt32);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto i8 = x.options().dtype(torch::kInt8);
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
  auto xq = torch::empty({T, kH}, i8);
  auto xs = torch::empty({T}, f32);
  auto xm = torch::empty({T, kH / 256}, i32);
  auto sq = torch::empty({2 * kI, kH}, i8);
  auto ss = torch::empty({2 * kI}, f32);
  RouteQuant rq;
  rq.xq = xq.data_ptr<int8_t>();
  rq.xs = xs.data_ptr<float>();
  rq.xm = xm.data_ptr<int>();
  rq.s13 = reinterpret_cast<const __nv_bfloat16*>(s13.data_ptr());
  rq.sq = sq.data_ptr<int8_t>();
  rq.ss = ss.data_ptr<float>();
  rq.w_blocks = 2 * kI / 8;
  TORCH_CHECK(qm >= 1.0 && qm <= 64.0, "the IMG multiplier cap must be in [1, 64] (s32 bound)");
  rq.qm = static_cast<float>(qm);
  route_topk_kernel<true><<<rq.w_blocks + (T + 7) / 8, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr()), T, cnt.data_ptr<int>(), topk_ids.data_ptr<int>(),
      topk_w.data_ptr<float>(), rank.data_ptr<int>(), sg.data_ptr<float>(), rq);
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
  return {topk_ids, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, xq, xs, xm, sq, ss};
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
                     torch::Tensor ss, torch::Tensor mt, torch::Tensor sorted_tok, int64_t tanh_silu) {
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
    C10_CUDA_CHECK(cudaFuncSetAttribute(up_g128_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGSmem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(up_g128_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGSmem));
    attr[dev] = true;
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (tanh_silu)
    up_g128_kernel<true><<<sms, 256, kGSmem, at::cuda::getCurrentCUDAStream()>>>(args, m_w, m_sw, m_h);
  else
    up_g128_kernel<false><<<sms, 256, kGSmem, at::cuda::getCurrentCUDAStream()>>>(args, m_w, m_sw, m_h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
// up_i8g on the warp-specialized kernel (up_g128p): bit for bit
torch::Tensor up_i8gp(torch::Tensor x, torch::Tensor xq, torch::Tensor xs, torch::Tensor xm, torch::Tensor q8, torch::Tensor sq,
                     torch::Tensor ss, torch::Tensor mt, torch::Tensor sorted_tok, int64_t tanh_silu) {
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
  static bool attr_p[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index out of range");
  if (!attr_p[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(up_g128p_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGSmem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(up_g128p_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGSmem));
    attr_p[dev] = true;
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (tanh_silu)
    up_g128p_kernel<true><<<sms, 384, kGSmem, at::cuda::getCurrentCUDAStream()>>>(args, m_w, m_sw, m_h);
  else
    up_g128p_kernel<false><<<sms, 384, kGSmem, at::cuda::getCurrentCUDAStream()>>>(args, m_w, m_sw, m_h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
// y8: y in int8 with per-row x 256-column scales (kY8, a PRECISION CHANGE: the caller gates it by layer), else bf16.
torch::Tensor down_i8(torch::Tensor x, torch::Tensor w2, torch::Tensor s2, torch::Tensor cnt, torch::Tensor h,
                      torch::Tensor topk_w, torch::Tensor pos_tk, torch::Tensor sg, torch::Tensor n_pairs,
                      torch::Tensor mt, torch::Tensor pairs, torch::Tensor sorted_tok, bool y8, double ydiv) {
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
  args.ydiv = static_cast<float>(ydiv);
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
// ==== LANE P W8A8 BEGIN
// Lane P (prefill routed-MoE up on INT8 tensor cores, W8A8). x is quantized per token and 128-column group
// (symmetric, fp32 scale = amax / 127, round to nearest) into xq / xs; the INT8 copy's g128 weight rows and fp16
// scales are used as they are (no new weight copy). One 128-wide K block = one scale group: four wgmma m64n128k32 s8
// into an s32 tile, then acc += float(s32) * (xs[row][g] * ws[col][g]) in fp32. Tile: 128 routed rows (two consumer
// warpgroups x 64) x 128 weight rows (64 gate + 64 up rows of one expert), so the SiLU-mul epilogue writes 64 h
// columns per tile. The shared expert stays BF16 (cuBLAS + silu_mul into its h rows) and the down is the king's BF16
// held-w2 GEMM with the fused combine, unchanged. Router, top-8 and the routed-row layout are the king's kernels.
constexpr int kW8BK = 128, kW8Stages = 5, kW8Groups = kH / 128, kW8NB = kI / 64;
constexpr uint32_t kW8ABytes = kBM * kW8BK;
constexpr uint32_t kW8BBytes = 128 * kW8BK;
constexpr uint32_t kW8ScaleBytes = 2 * kW8Groups * 128 * 4;
constexpr uint32_t kW8DBytesWG = 64 * 64 * 2;
constexpr uint32_t kW8Smem = 1024 + kW8Stages * (kW8ABytes + kW8BBytes) + 2 * kW8ScaleBytes + 2 * kW8DBytesWG +
                             (2 * kW8Stages + 4) * 8;
static_assert(kW8Smem <= 232448, "W8A8 up smem must fit the 227 KiB opt-in");
struct W8Args {
  const int4* mt;
  const int4* pairs;
  const int* n_pairs;
  const int* sorted_tok;
  const int8_t* xq;
  const float* xs;
  const uint8_t* q8;
};
__global__ void __launch_bounds__(256) w8_quant_kernel(const __nv_bfloat16* __restrict__ x, int T,
                                                       int8_t* __restrict__ xq, float* __restrict__ xs) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int t = blockIdx.x * 8 + warp;
  if (t >= T) return;
  const int hf = lane >> 4, l16 = lane & 15;
  const uint4* xr = reinterpret_cast<const uint4*>(x + static_cast<size_t>(t) * kH);
  uint2* qr = reinterpret_cast<uint2*>(xq + static_cast<size_t>(t) * kH);
  uint4 v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __ldg(xr + (2 * i + hf) * 16 + l16);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
    float f[8];
    float m = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 p = __bfloat1622float2(b[q]);
      f[2 * q] = p.x, f[2 * q + 1] = p.y;
      m = fmaxf(m, fmaxf(fabsf(p.x), fabsf(p.y)));
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    const float s = __fdiv_rn(m, 127.f);
    const float inv = m > 0.f ? __fdiv_rn(127.f, m) : 0.f;
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int qv = max(-127, min(127, __float2int_rn(f[j] * inv)));
      w[j >> 2] |= (static_cast<uint32_t>(qv) & 0xffu) << (8 * (j & 3));
    }
    qr[(2 * i + hf) * 16 + l16] = make_uint2(w[0], w[1]);
    if (l16 == 0) xs[static_cast<size_t>(t) * kW8Groups + 2 * i + hf] = s;
  }
}
__global__ void __launch_bounds__(128) w8_shared_silu_kernel(const __nv_bfloat16* __restrict__ gu, int T,
                                                             const int* __restrict__ offs,
                                                             __nv_bfloat16* __restrict__ h) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const int row = offs[kE] + t;
  const int j = threadIdx.x * 4;
  const uint2 g2 = *reinterpret_cast<const uint2*>(gu + static_cast<size_t>(t) * (2 * kI) + j);
  const uint2 u2 = *reinterpret_cast<const uint2*>(gu + static_cast<size_t>(t) * (2 * kI) + kI + j);
  const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&g2);
  const __nv_bfloat162* u = reinterpret_cast<const __nv_bfloat162*>(&u2);
  float o[4];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const float2 gf = __bfloat1622float2(g[q]), uf = __bfloat1622float2(u[q]);
    o[2 * q] = gf.x / (1.f + __expf(-gf.x)) * uf.x;
    o[2 * q + 1] = gf.y / (1.f + __expf(-gf.y)) * uf.y;
  }
  *reinterpret_cast<uint2*>(h + static_cast<size_t>(row) * kI + j) = make_uint2(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]));
}
__device__ __forceinline__ void wgmma_s8_m64n128k32(int (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p;\n"
      "}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void fence_operand_i(int& r) { asm volatile("" : "+r"(r)::"memory"); }
// kQ64 (Lane PD2): the epilogue rounds h to bf16 as before, then quantizes each row's 64 columns of this unit
// (symmetric, amax / 127, round to nearest) and stores int8 h through tm_h (an int8 map) plus the fp32 scale in hs.
__device__ __forceinline__ void sts_u16(uint32_t addr, unsigned short v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(v) : "memory");
}
__device__ __forceinline__ void sts_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ unsigned short q8_pair(float a, float b, float inv) {
  const int qa = max(-127, min(127, __float2int_rn(a * inv)));
  const int qb = max(-127, min(127, __float2int_rn(b * inv)));
  return static_cast<unsigned short>((static_cast<uint32_t>(qa) & 0xffu) | ((static_cast<uint32_t>(qb) & 0xffu) << 8));
}
// kPP (Lane PD2 follow-up, bit for bit): two s32 accumulators, K block kb + 1's wgmmas run while K block kb is promoted
// (same wgmmas, same promotion order and fma chain); no branch around the async wgmmas (an idle WG multiplies rows
// whose scales are 0); the producer gives registers to the consumers (setmaxnreg).
template <bool kQ64, bool kPP = false>
__global__ void __launch_bounds__(384, 1)
    w8_up_kernel(const W8Args args, const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ CUtensorMap tm_h,
                 float* __restrict__ hs) {
  extern __shared__ __align__(1024) uint8_t w8_smem[];
  const uint32_t raw = smem_u32(w8_smem);
  const uint32_t base = (raw + 1023u) & ~1023u;
  uint8_t* gbase = w8_smem + (base - raw);
  const uint32_t sA = base;
  const uint32_t sB = sA + kW8Stages * kW8ABytes;
  const uint32_t sS = sB + kW8Stages * kW8BBytes;
  const uint32_t sD = sS + 2 * kW8ScaleBytes;
  float* sSg = reinterpret_cast<float*>(gbase + (sS - base));
  uint64_t* full = reinterpret_cast<uint64_t*>(gbase + (sD + 2 * kW8DBytesWG - base));
  uint64_t* empty = full + kW8Stages;
  uint64_t* sfull = empty + kW8Stages;
  uint64_t* sempty = sfull + 2;
  const int tid = threadIdx.x;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kW8Stages; ++s) {
      mbar_init(&full[s], 129);
      mbar_init(&empty[s], 8);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      mbar_init(&sfull[s], 128);
      mbar_init(&sempty[s], 8);
    }
    fence_barrier_init();
  }
  if (tid == 32) {
    prefetch_tmap(&tm_w);
    prefetch_tmap(&tm_h);
  }
  __syncthreads();
  const int n_units = *args.n_pairs * 2 * kW8NB;
  auto resolve = [&](int u, int4& info) -> bool {
    const int4 pr = args.pairs[u / (2 * kW8NB)];
    const int m = ((u / kW8NB) & 1) ? pr.y : pr.x;
    if (m < 0) return false;
    info = args.mt[m];
    return info.x < kE && info.z > 0;
  };
  if (tid >= 256) {
    if constexpr (kPP) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");
    const int pt = tid - 256;
    const int c = pt & 7, r0 = pt >> 3;
    const uint32_t dst_off = r0 * 128 + ((c ^ (r0 & 7)) << 4);
    const uint64_t pol_x = policy_evict_last();
    int stage = 0, tile = 0;
    uint32_t phase = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x) {
      int4 info;
      if (!resolve(u, info)) continue;
      const int n = u % kW8NB;
      const int slot = tile & 1;
      mbar_wait(&sempty[slot], ((tile >> 1) & 1) ^ 1);
      {
        float* sa = sSg + slot * (kW8ScaleBytes / 4);
        float* sb = sa + kW8Groups * 128;
        if (pt < info.z) {
          const float4* xs4 = reinterpret_cast<const float4*>(args.xs + static_cast<size_t>(args.sorted_tok[info.y + pt]) * kW8Groups);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float4 v = __ldg(xs4 + j);
            sa[(4 * j) * 128 + pt] = v.x;
            sa[(4 * j + 1) * 128 + pt] = v.y;
            sa[(4 * j + 2) * 128 + pt] = v.z;
            sa[(4 * j + 3) * 128 + pt] = v.w;
          }
        } else {
#pragma unroll
          for (int g = 0; g < kW8Groups; ++g) sa[g * 128 + pt] = 0.f;
        }
        const int wrow = pt < 64 ? n * 64 + pt : kI + n * 64 + (pt - 64);
        const uint4* sp = reinterpret_cast<const uint4*>(args.q8 + static_cast<size_t>(info.x) * kQ8Expert + kQ8C13 +
                                                         static_cast<size_t>(wrow) * kW8Groups * 2);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint4 v = __ldg(sp + j);
          const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __half22float2(h2[q]);
            sb[(8 * j + 2 * q) * 128 + pt] = f.x;
            sb[(8 * j + 2 * q + 1) * 128 + pt] = f.y;
          }
        }
        mbar_arrive(&sfull[slot]);
      }
      const int8_t* src[8];
      uint32_t mask = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = r0 + 16 * i;
        src[i] = args.xq + c * 16;
        if (r < info.z) {
          mask |= 1u << i;
          src[i] += static_cast<size_t>(args.sorted_tok[info.y + r]) * kH;
        }
      }
      for (int kb = 0; kb < kW8Groups; ++kb) {
        mbar_wait(&empty[stage], phase ^ 1);
        const uint32_t a_dst = sA + stage * kW8ABytes + dst_off;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (mask >> i & 1) cp_async16(a_dst + i * 2048, src[i] + kb * kW8BK, pol_x);
        if (pt == 0) {
          mbar_arrive_expect_tx(&full[stage], kW8BBytes);
          const uint32_t b_dst = sB + stage * kW8BBytes;
          tma_load_3d(b_dst, &tm_w, &full[stage], kb * kW8BK, n * 64, info.x, kEvictNormal);
          tma_load_3d(b_dst + kW8BBytes / 2, &tm_w, &full[stage], kb * kW8BK, kI + n * 64, info.x, kEvictNormal);
        }
        cp_async_arrive_noinc(&full[stage]);
        stage = stage + 1 == kW8Stages ? 0 : stage + 1;
        phase ^= stage == 0;
      }
      ++tile;
    }
  } else {
    if constexpr (kPP) asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n" ::: "memory");
    const int wg = tid / 128, warp = tid / 32, wi = warp % 4, lane = tid % 32;
    const uint32_t a_base = sA + wg * 64 * 128;
    const uint32_t d_base = sD + wg * kW8DBytesWG;
    const int rr = wg * 64 + wi * 16 + lane / 4, q = lane % 4;
    int stage = 0, tile = 0;
    uint32_t phase = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x) {
      int4 info;
      if (!resolve(u, info)) continue;
      const int n = u % kW8NB;
      const int slot = tile & 1;
      const bool active = info.z > wg * 64;
      mbar_wait(&sfull[slot], (tile >> 1) & 1);
      const float* sa = sSg + slot * (kW8ScaleBytes / 4);
      const float* sb = sa + kW8Groups * 128;
      float acc[64];
      if constexpr (kPP) {
#pragma unroll
        for (int i = 0; i < 64; ++i) acc[i] = 0.f;
        int accA[64], accB[64];
        auto issue = [&](int (&d)[64]) -> int {
          const int st = stage;
          mbar_wait(&full[st], phase);
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(d[i]);
          wgmma_fence();
#pragma unroll
          for (int kk = 0; kk < kW8BK / 32; ++kk)
            wgmma_s8_m64n128k32(d, gmma_desc(a_base + st * kW8ABytes + kk * 32), gmma_desc(sB + st * kW8BBytes + kk * 32),
                                kk);
          wgmma_commit();
          stage = stage + 1 == kW8Stages ? 0 : stage + 1;
          phase ^= stage == 0;
          return st;
        };
        auto promote = [&](int (&d)[64], int kb, int st) {
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(d[i]);
          if (lane == 0) mbar_arrive(&empty[st]);
          const float sa0 = sa[kb * 128 + rr], sa1 = sa[kb * 128 + rr + 8];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const float2 s2 = *reinterpret_cast<const float2*>(sb + kb * 128 + 8 * i + 2 * q);
            acc[4 * i] = fmaf(static_cast<float>(d[4 * i]), sa0 * s2.x, acc[4 * i]);
            acc[4 * i + 1] = fmaf(static_cast<float>(d[4 * i + 1]), sa0 * s2.y, acc[4 * i + 1]);
            acc[4 * i + 2] = fmaf(static_cast<float>(d[4 * i + 2]), sa1 * s2.x, acc[4 * i + 2]);
            acc[4 * i + 3] = fmaf(static_cast<float>(d[4 * i + 3]), sa1 * s2.y, acc[4 * i + 3]);
          }
        };
        static_assert(kW8Groups % 2 == 0, "the up ping-pong pairs K blocks");
        int stA = issue(accA), stB;
#pragma unroll
        for (int kb = 1; kb < kW8Groups; kb += 2) {
          stB = issue(accB);
          wgmma_wait<1>();
          promote(accA, kb - 1, stA);
          if (kb + 1 < kW8Groups) {
            stA = issue(accA);
            wgmma_wait<1>();
            promote(accB, kb, stB);
          }
        }
        wgmma_wait<0>();
        promote(accB, kW8Groups - 1, stB);
      } else {
      int accs[64];
#pragma unroll
      for (int i = 0; i < 64; ++i) acc[i] = 0.f, accs[i] = 0;
      for (int kb = 0; kb < kW8Groups; ++kb) {
        mbar_wait(&full[stage], phase);
        if (active) {
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(accs[i]);
          wgmma_fence();
#pragma unroll
          for (int kk = 0; kk < kW8BK / 32; ++kk)
            wgmma_s8_m64n128k32(accs, gmma_desc(a_base + stage * kW8ABytes + kk * 32),
                                gmma_desc(sB + stage * kW8BBytes + kk * 32), kk);
          wgmma_commit();
          wgmma_wait<0>();
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(accs[i]);
        }
        if (lane == 0) mbar_arrive(&empty[stage]);
        stage = stage + 1 == kW8Stages ? 0 : stage + 1;
        phase ^= stage == 0;
        if (active) {
          const float sa0 = sa[kb * 128 + rr], sa1 = sa[kb * 128 + rr + 8];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const float2 s2 = *reinterpret_cast<const float2*>(sb + kb * 128 + 8 * i + 2 * q);
            acc[4 * i] = fmaf(static_cast<float>(accs[4 * i]), sa0 * s2.x, acc[4 * i]);
            acc[4 * i + 1] = fmaf(static_cast<float>(accs[4 * i + 1]), sa0 * s2.y, acc[4 * i + 1]);
            acc[4 * i + 2] = fmaf(static_cast<float>(accs[4 * i + 2]), sa1 * s2.x, acc[4 * i + 2]);
            acc[4 * i + 3] = fmaf(static_cast<float>(accs[4 * i + 3]), sa1 * s2.y, acc[4 * i + 3]);
          }
        }
      }
      }
      if (lane == 0) mbar_arrive(&sempty[slot]);
      ++tile;
      if (!active) continue;
      if (tid % 128 == 0) tma_store_wait_read();
      named_bar_sync(1 + wg, 128);
      if constexpr (kQ64) {
        uint32_t hp[16];
        float m0 = 0.f, m1 = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float h[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float g = acc[4 * i + k], up = acc[4 * (i + 8) + k];
            h[k] = g / (1.f + __expf(-g)) * up;
          }
          hp[2 * i] = pack_bf16(h[0], h[1]);
          hp[2 * i + 1] = pack_bf16(h[2], h[3]);
          const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hp[2 * i]));
          const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hp[2 * i + 1]));
          m0 = fmaxf(m0, fmaxf(fabsf(a.x), fabsf(a.y)));
          m1 = fmaxf(m1, fmaxf(fabsf(b.x), fabsf(b.y)));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
          m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, off));
          m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, off));
        }
        const float inv0 = m0 > 0.f ? __fdiv_rn(127.f, m0) : 0.f;
        const float inv1 = m1 > 0.f ? __fdiv_rn(127.f, m1) : 0.f;
        const int rl = wi * 16 + lane / 4;
        const uint32_t rb0 = d_base + rl * 64, rb1 = rb0 + 8 * 64;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hp[2 * i]));
          const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hp[2 * i + 1]));
          sts_u16(rb0 + 8 * i + 2 * q, q8_pair(a.x, a.y, inv0));
          sts_u16(rb1 + 8 * i + 2 * q, q8_pair(b.x, b.y, inv1));
        }
        if (q == 0) {
          const size_t row = static_cast<size_t>(info.y + wg * 64 + rl);
          hs[row * kW8NB + n] = __fdiv_rn(m0, 127.f);
          hs[(row + 8) * kW8NB + n] = __fdiv_rn(m1, 127.f);
        }
      } else {
      const uint32_t row_addr = d_base + wi * 2048 + (lane % 16) * 128;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float h[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float g = acc[4 * i + k], up = acc[4 * (i + 8) + k];
          h[k] = g / (1.f + __expf(-g)) * up;
        }
        stsm_x2(pack_bf16(h[0], h[1]), pack_bf16(h[2], h[3]), row_addr + (((i % 8) ^ (lane % 8)) << 4));
      }
      }
      fence_async_shared();
      named_bar_sync(1 + wg, 128);
      if (tid % 128 == 0) {
        tma_store_2d(&tm_h, d_base, n * 64, info.y + wg * 64);
        tma_store_commit();
      }
    }
    if (tid % 128 == 0) tma_store_wait_all();
  }
}
// The INT8 copy's up rows as a 3D int8 tensor [kE][2 kI][kH], box 128 B x 64 rows, 128B-swizzled for wgmma.
static CUtensorMap make_map_w8(const uint8_t* q8) {
  CUtensorMap map;
  const cuuint64_t dims[3] = {static_cast<cuuint64_t>(kH), static_cast<cuuint64_t>(2 * kI), static_cast<cuuint64_t>(kE)};
  const cuuint64_t strides[2] = {static_cast<cuuint64_t>(kH), static_cast<cuuint64_t>(kQ8Expert)};
  const cuuint32_t box[3] = {static_cast<cuuint32_t>(kW8BK), 64, 1};
  const cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<uint8_t*>(q8), dims, strides, box,
                                 estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (W8A8 up) failed: ", static_cast<int>(r));
  return map;
}
// ==== LANE PD2 (W8A8 routed down) BEGIN
// The routed down on INT8 tensor cores. The up (w8_upq_kernel) is the Lane P up with one change of schedule: a CTA
// takes both 64-column halves of one 128-column group of h in turn, so each row's h group is whole on chip. It rounds
// h to bf16 exactly as the BF16 path stores it, then quantizes each (row, 128-column group) symmetrically (fp32 scale =
// amax / 127, round to nearest, the x quantizer's rule) into hq / hs instead of writing bf16 h. The down
// (w8_dn_kernel) is the up's pipeline on K = 512: A = hq rows by TMA (routed rows are contiguous per expert), B = the
// INT8 copy's w2 rows by TMA, four wgmma m64n128k32 s8 per 128-column group into s32, then acc += float(s32) *
// (hs[row][g] * ws[col][g]) in fp32, and the unweighted routed y rows in bf16 where the king's routed tiles put them.
// The shared expert's down (BF16) and the fused combine are the king's kernel run over the shared pairs only; CTA 0 of
// the down lists them. The shared up, router, top-8 and row layout are unchanged.
constexpr int kD8BK = 128, kD8Stages = 5, kD8Groups = kI / 128, kD8NB = kH / 128;
constexpr uint32_t kD8ABytes = kBM * kD8BK;
constexpr uint32_t kD8BBytes = 128 * kD8BK;
constexpr int kD8SaG = 8;  // A scale slots per row: 4 (g128 h) or 8 (g64 h)
constexpr uint32_t kD8ScaleBytes = (kD8SaG + kD8Groups) * 128 * 4;
constexpr uint32_t kD8DBytesWG = 64 * 128 * 2;
constexpr uint32_t kD8Smem = 1024 + kD8Stages * (kD8ABytes + kD8BBytes) + 2 * kD8ScaleBytes + 2 * kD8DBytesWG +
                             (2 * kD8Stages + 4) * 8;
static_assert(kD8Smem <= 232448, "W8A8 down smem must fit the 227 KiB opt-in");
static_assert(kW8DBytesWG == 64 * 128, "one WG's quantized h staging is 64 rows x 128 int8");
__global__ void __launch_bounds__(384, 1)
    w8_upq_kernel(const W8Args args, const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ CUtensorMap tm_hq,
                  float* __restrict__ hs) {
  extern __shared__ __align__(1024) uint8_t w8_smem[];
  const uint32_t raw = smem_u32(w8_smem);
  const uint32_t base = (raw + 1023u) & ~1023u;
  uint8_t* gbase = w8_smem + (base - raw);
  const uint32_t sA = base;
  const uint32_t sB = sA + kW8Stages * kW8ABytes;
  const uint32_t sS = sB + kW8Stages * kW8BBytes;
  const uint32_t sD = sS + 2 * kW8ScaleBytes;
  float* sSg = reinterpret_cast<float*>(gbase + (sS - base));
  uint64_t* full = reinterpret_cast<uint64_t*>(gbase + (sD + 2 * kW8DBytesWG - base));
  uint64_t* empty = full + kW8Stages;
  uint64_t* sfull = empty + kW8Stages;
  uint64_t* sempty = sfull + 2;
  const int tid = threadIdx.x;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kW8Stages; ++s) {
      mbar_init(&full[s], 129);
      mbar_init(&empty[s], 8);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      mbar_init(&sfull[s], 128);
      mbar_init(&sempty[s], 8);
    }
    fence_barrier_init();
  }
  if (tid == 32) {
    prefetch_tmap(&tm_w);
    prefetch_tmap(&tm_hq);
  }
  __syncthreads();
  // double units: (pair, m-select, 128-column group of h) = the Lane P units 2 du and 2 du + 1
  const int n_du = *args.n_pairs * kW8NB;
  auto resolve = [&](int du, int4& info) -> bool {
    const int u = 2 * du;
    const int4 pr = args.pairs[u / (2 * kW8NB)];
    const int m = ((u / kW8NB) & 1) ? pr.y : pr.x;
    if (m < 0) return false;
    info = args.mt[m];
    return info.x < kE && info.z > 0;
  };
  if (tid >= 256) {
    const int pt = tid - 256;
    const int c = pt & 7, r0 = pt >> 3;
    const uint32_t dst_off = r0 * 128 + ((c ^ (r0 & 7)) << 4);
    const uint64_t pol_x = policy_evict_last();
    int stage = 0, tile = 0;
    uint32_t phase = 0;
    for (int du = blockIdx.x; du < n_du; du += gridDim.x) {
      int4 info;
      if (!resolve(du, info)) continue;
      const int n0 = (2 * du) % kW8NB;
      const int8_t* src[8];
      uint32_t mask = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = r0 + 16 * i;
        src[i] = args.xq + c * 16;
        if (r < info.z) {
          mask |= 1u << i;
          src[i] += static_cast<size_t>(args.sorted_tok[info.y + r]) * kH;
        }
      }
#pragma unroll 1
      for (int sub = 0; sub < 2; ++sub) {
        const int n = n0 + sub;
        const int slot = tile & 1;
        mbar_wait(&sempty[slot], ((tile >> 1) & 1) ^ 1);
        {
          float* sa = sSg + slot * (kW8ScaleBytes / 4);
          float* sb = sa + kW8Groups * 128;
          if (pt < info.z) {
            const float4* xs4 =
                reinterpret_cast<const float4*>(args.xs + static_cast<size_t>(args.sorted_tok[info.y + pt]) * kW8Groups);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float4 v = __ldg(xs4 + j);
              sa[(4 * j) * 128 + pt] = v.x;
              sa[(4 * j + 1) * 128 + pt] = v.y;
              sa[(4 * j + 2) * 128 + pt] = v.z;
              sa[(4 * j + 3) * 128 + pt] = v.w;
            }
          } else {
#pragma unroll
            for (int g = 0; g < kW8Groups; ++g) sa[g * 128 + pt] = 0.f;
          }
          const int wrow = pt < 64 ? n * 64 + pt : kI + n * 64 + (pt - 64);
          const uint4* sp = reinterpret_cast<const uint4*>(args.q8 + static_cast<size_t>(info.x) * kQ8Expert + kQ8C13 +
                                                           static_cast<size_t>(wrow) * kW8Groups * 2);
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint4 v = __ldg(sp + j);
            const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const float2 f = __half22float2(h2[q]);
              sb[(8 * j + 2 * q) * 128 + pt] = f.x;
              sb[(8 * j + 2 * q + 1) * 128 + pt] = f.y;
            }
          }
          mbar_arrive(&sfull[slot]);
        }
        for (int kb = 0; kb < kW8Groups; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          const uint32_t a_dst = sA + stage * kW8ABytes + dst_off;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            if (mask >> i & 1) cp_async16(a_dst + i * 2048, src[i] + kb * kW8BK, pol_x);
          if (pt == 0) {
            mbar_arrive_expect_tx(&full[stage], kW8BBytes);
            const uint32_t b_dst = sB + stage * kW8BBytes;
            tma_load_3d(b_dst, &tm_w, &full[stage], kb * kW8BK, n * 64, info.x, kEvictNormal);
            tma_load_3d(b_dst + kW8BBytes / 2, &tm_w, &full[stage], kb * kW8BK, kI + n * 64, info.x, kEvictNormal);
          }
          cp_async_arrive_noinc(&full[stage]);
          stage = stage + 1 == kW8Stages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
        ++tile;
      }
    }
  } else {
    const int wg = tid / 128, warp = tid / 32, wi = warp % 4, lane = tid % 32;
    const uint32_t a_base = sA + wg * 64 * 128;
    const uint32_t d_base = sD + wg * kW8DBytesWG;
    const int rr = wg * 64 + wi * 16 + lane / 4, q = lane % 4;
    int stage = 0, tile = 0;
    uint32_t phase = 0;
    for (int du = blockIdx.x; du < n_du; du += gridDim.x) {
      int4 info;
      if (!resolve(du, info)) continue;
      const int j128 = (2 * du) % kW8NB / 2;
      const bool active = info.z > wg * 64;
      // The first half's bf16 h pairs wait in this WG's staging buffer (thread-private words) while the second half's
      // accumulators are live; the second half quantizes both. One copy of the MMA body serves both halves.
      const uint32_t stash = d_base + 4u * static_cast<uint32_t>(tid % 128);
#pragma unroll 1
      for (int sub = 0; sub < 2; ++sub) {
        const int slot = tile & 1;
        mbar_wait(&sfull[slot], (tile >> 1) & 1);
        const float* sa = sSg + slot * (kW8ScaleBytes / 4);
        const float* sb = sa + kW8Groups * 128;
        float acc[64];
        int accs[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) acc[i] = 0.f, accs[i] = 0;
#pragma unroll 1
        for (int kb = 0; kb < kW8Groups; ++kb) {
          mbar_wait(&full[stage], phase);
          if (active) {
#pragma unroll
            for (int i = 0; i < 64; ++i) fence_operand_i(accs[i]);
            wgmma_fence();
#pragma unroll
            for (int kk = 0; kk < kW8BK / 32; ++kk)
              wgmma_s8_m64n128k32(accs, gmma_desc(a_base + stage * kW8ABytes + kk * 32),
                                  gmma_desc(sB + stage * kW8BBytes + kk * 32), kk);
            wgmma_commit();
            wgmma_wait<0>();
#pragma unroll
            for (int i = 0; i < 64; ++i) fence_operand_i(accs[i]);
          }
          if (lane == 0) mbar_arrive(&empty[stage]);
          stage = stage + 1 == kW8Stages ? 0 : stage + 1;
          phase ^= stage == 0;
          if (active) {
            const float sa0 = sa[kb * 128 + rr], sa1 = sa[kb * 128 + rr + 8];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              const float2 s2 = *reinterpret_cast<const float2*>(sb + kb * 128 + 8 * i + 2 * q);
              acc[4 * i] = fmaf(static_cast<float>(accs[4 * i]), sa0 * s2.x, acc[4 * i]);
              acc[4 * i + 1] = fmaf(static_cast<float>(accs[4 * i + 1]), sa0 * s2.y, acc[4 * i + 1]);
              acc[4 * i + 2] = fmaf(static_cast<float>(accs[4 * i + 2]), sa1 * s2.x, acc[4 * i + 2]);
              acc[4 * i + 3] = fmaf(static_cast<float>(accs[4 * i + 3]), sa1 * s2.y, acc[4 * i + 3]);
            }
          }
        }
        if (lane == 0) mbar_arrive(&sempty[slot]);
        ++tile;
        if (!active) continue;
        uint32_t hc[16];  // [2 i] row rr, [2 i + 1] row rr + 8
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float h[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float g = acc[4 * i + k], up = acc[4 * (i + 8) + k];
            h[k] = g / (1.f + __expf(-g)) * up;
          }
          hc[2 * i] = pack_bf16(h[0], h[1]);
          hc[2 * i + 1] = pack_bf16(h[2], h[3]);
        }
        if (sub == 0) {
          if (tid % 128 == 0) tma_store_wait_read();
          named_bar_sync(1 + wg, 128);
#pragma unroll
          for (int k = 0; k < 16; ++k) sts_u32(stash + 512u * k, hc[k]);
          continue;
        }
        uint32_t hp[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) hp[k] = lds_u32(stash + 512u * k);
        float m0 = 0.f, m1 = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hp[2 * i]));
          const float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hp[2 * i + 1]));
          const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hc[2 * i]));
          const float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hc[2 * i + 1]));
          m0 = fmaxf(m0, fmaxf(fmaxf(fabsf(a0.x), fabsf(a0.y)), fmaxf(fabsf(a1.x), fabsf(a1.y))));
          m1 = fmaxf(m1, fmaxf(fmaxf(fabsf(b0.x), fabsf(b0.y)), fmaxf(fabsf(b1.x), fabsf(b1.y))));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
          m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, off));
          m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, off));
        }
        const float inv0 = m0 > 0.f ? __fdiv_rn(127.f, m0) : 0.f;
        const float inv1 = m1 > 0.f ? __fdiv_rn(127.f, m1) : 0.f;
        named_bar_sync(1 + wg, 128);
        const int rl = wi * 16 + lane / 4;
        const uint32_t rb0 = d_base + rl * 128, rb1 = rb0 + 8 * 128;
        const uint32_t sw = static_cast<uint32_t>(rl & 7);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf)
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int col = hf * 64 + 8 * i + 2 * q;
            const uint32_t off = (((static_cast<uint32_t>(col) >> 4) ^ sw) << 4) + (col & 15);
            const uint32_t va = hf ? hc[2 * i] : hp[2 * i], vb = hf ? hc[2 * i + 1] : hp[2 * i + 1];
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&va));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vb));
            sts_u16(rb0 + off, q8_pair(a.x, a.y, inv0));
            sts_u16(rb1 + off, q8_pair(b.x, b.y, inv1));
          }
        fence_async_shared();
        named_bar_sync(1 + wg, 128);
        if (tid % 128 == 0) {
          tma_store_2d(&tm_hq, d_base, j128 * 128, info.y + wg * 64);
          tma_store_commit();
        }
        if (q == 0) {
          const size_t row = static_cast<size_t>(info.y + wg * 64 + rl);
          hs[row * kD8Groups + j128] = __fdiv_rn(m0, 127.f);
          hs[(row + 8) * kD8Groups + j128] = __fdiv_rn(m1, 127.f);
        }
      }
    }
    if (tid % 128 == 0) tma_store_wait_all();
  }
}
struct D8Args {
  const int4* mt;
  const int4* pairs;
  const int* n_pairs;
  const float* hs;
  const uint8_t* q8;
  int4* spairs;
  int* n_spairs;
  int mode;  // bench ablations only (0 in service): 1 no promotion, 2 no MMA, 4 no y store
  float* ysc;  // kYQ (port): y's fp32 scale per routed row and 128-column block [rows][kD8NB]
};
// kYQ (port: our W8A8 down stacked with b9d3c95a's y8): the routed y tile is stored as int8 with one fp32 scale
// (amax / 127 of the fp32 sums, round to nearest) per row and 128-column block, the y8 epilogue's rule on this kernel's
// 128-column units; the king's combine reads it back with kY8 + kY128.
template <bool kPP, bool kG64, bool kYQ = false>
__global__ void __launch_bounds__(384, 1)
    w8_dn_kernel(const D8Args args, const __grid_constant__ CUtensorMap tm_w2, const __grid_constant__ CUtensorMap tm_hq,
                 const __grid_constant__ CUtensorMap tm_y) {
  extern __shared__ __align__(1024) uint8_t d8_smem[];
  __shared__ int s_first;
  const uint32_t raw = smem_u32(d8_smem);
  const uint32_t base = (raw + 1023u) & ~1023u;
  uint8_t* gbase = d8_smem + (base - raw);
  const uint32_t sA = base;
  const uint32_t sB = sA + kD8Stages * kD8ABytes;
  const uint32_t sS = sB + kD8Stages * kD8BBytes;
  const uint32_t sD = sS + 2 * kD8ScaleBytes;
  float* sSg = reinterpret_cast<float*>(gbase + (sS - base));
  uint64_t* full = reinterpret_cast<uint64_t*>(gbase + (sD + 2 * kD8DBytesWG - base));
  uint64_t* empty = full + kD8Stages;
  uint64_t* sfull = empty + kD8Stages;
  uint64_t* sempty = sfull + 2;
  const int tid = threadIdx.x;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kD8Stages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      mbar_init(&sfull[s], 128);
      mbar_init(&sempty[s], 8);
    }
    fence_barrier_init();
  }
  if (tid == 32) {
    prefetch_tmap(&tm_w2);
    prefetch_tmap(&tm_hq);
    prefetch_tmap(&tm_y);
  }
  __syncthreads();
  const int n_units = *args.n_pairs * 2 * kD8NB;
  auto resolve = [&](int u, int4& info) -> bool {
    const int4 pr = args.pairs[u / (2 * kD8NB)];
    const int m = ((u / kD8NB) & 1) ? pr.y : pr.x;
    if (m < 0) return false;
    info = args.mt[m];
    return info.x < kE && info.z > 0;
  };
  if (tid >= 256) {
    if constexpr (kPP) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    const int pt = tid - 256;
    if (blockIdx.x == 0) {
      // the shared expert's pairs (they follow every routed pair) for the king's BF16 shared down + combine
      const int np = *args.n_pairs;
      if (pt == 0) s_first = np;
      named_bar_sync(3, 128);
      for (int p = pt; p < np; p += 128) {
        const int4 pr = args.pairs[p];
        if (pr.x >= 0 && args.mt[pr.x].x == kE) atomicMin(&s_first, p);
      }
      named_bar_sync(3, 128);
      const int f = s_first;
      for (int p = f + pt; p < np; p += 128) args.spairs[p - f] = args.pairs[p];
      if (pt == 0) *args.n_spairs = np - f;
    }
    int stage = 0, tile = 0;
    uint32_t phase = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x) {
      int4 info;
      if (!resolve(u, info)) continue;
      const int n = u % kD8NB;
      const int slot = tile & 1;
      mbar_wait(&sempty[slot], ((tile >> 1) & 1) ^ 1);
      {
        float* sa = sSg + slot * (kD8ScaleBytes / 4);
        float* sb = sa + kD8SaG * 128;
        constexpr int kSaG = kG64 ? 8 : kD8Groups;
        if (pt < info.z) {
#pragma unroll
          for (int j = 0; j < kSaG / 4; ++j) {
            const float4 v = __ldg(reinterpret_cast<const float4*>(args.hs + static_cast<size_t>(info.y + pt) * kSaG) + j);
            sa[(4 * j) * 128 + pt] = v.x;
            sa[(4 * j + 1) * 128 + pt] = v.y;
            sa[(4 * j + 2) * 128 + pt] = v.z;
            sa[(4 * j + 3) * 128 + pt] = v.w;
          }
        } else {
#pragma unroll
          for (int g = 0; g < kSaG; ++g) sa[g * 128 + pt] = 0.f;
        }
        const uint2 w = __ldg(reinterpret_cast<const uint2*>(args.q8 + static_cast<size_t>(info.x) * kQ8Expert + kQ8C2 +
                                                              static_cast<size_t>(n * 128 + pt) * kD8Groups * 2));
        const __half2* h2 = reinterpret_cast<const __half2*>(&w);
        const float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]);
        sb[pt] = f0.x;
        sb[128 + pt] = f0.y;
        sb[256 + pt] = f1.x;
        sb[384 + pt] = f1.y;
        mbar_arrive(&sfull[slot]);
      }
      if (pt == 0) {
        for (int kb = 0; kb < kD8Groups; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          mbar_arrive_expect_tx(&full[stage], kD8ABytes + kD8BBytes);
          tma_load_2d(sA + stage * kD8ABytes, &tm_hq, &full[stage], kb * kD8BK, info.y, kEvictNormal);
          tma_load_3d(sB + stage * kD8BBytes, &tm_w2, &full[stage], kb * kD8BK, n * 128, info.x, kEvictNormal);
          stage = stage + 1 == kD8Stages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
      ++tile;
    }
  } else {
    if constexpr (kPP) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const int wg = tid / 128, warp = tid / 32, wi = warp % 4, lane = tid % 32;
    const uint32_t a_base = sA + wg * 64 * 128;
    const uint32_t d_base = sD + wg * kD8DBytesWG;
    const int rr = wg * 64 + wi * 16 + lane / 4, q = lane % 4;
    int stage = 0, tile = 0;
    uint32_t phase = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x) {
      int4 info;
      if (!resolve(u, info)) continue;
      const int n = u % kD8NB;
      const int slot = tile & 1;
      const bool active = info.z > wg * 64;
      mbar_wait(&sfull[slot], (tile >> 1) & 1);
      const float* sa = sSg + slot * (kD8ScaleBytes / 4);
      const float* sb = sa + kD8SaG * 128;
      float acc[64];
#pragma unroll
      for (int i = 0; i < 64; ++i) acc[i] = 0.f;
      if constexpr (kPP && kG64) {
        // g64 h: eight 64-wide K halves, A scale per half, B scale per 128 group; two s32 accumulators ping-pong
        static_assert(kD8Groups == 4, "the g64 sequence is written for four 128-wide K blocks");
        int accA[64], accB[64];
        int stg[kD8Groups];
        auto issueH = [&](int (&d)[64], int hh) {
          if ((hh & 1) == 0) {
            stg[hh / 2] = stage;
            mbar_wait(&full[stage], phase);
          }
          const int st = stg[hh / 2];
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(d[i]);
          wgmma_fence();
#pragma unroll
          for (int kk = 0; kk < 2; ++kk)
            wgmma_s8_m64n128k32(d, gmma_desc(a_base + st * kD8ABytes + (2 * (hh & 1) + kk) * 32),
                                gmma_desc(sB + st * kD8BBytes + (2 * (hh & 1) + kk) * 32), kk);
          wgmma_commit();
          if (hh & 1) {
            stage = stage + 1 == kD8Stages ? 0 : stage + 1;
            phase ^= stage == 0;
          }
        };
        auto promoteG = [&](int (&d)[64], int hh) {
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(d[i]);
          if ((hh & 1) && lane == 0) mbar_arrive(&empty[stg[hh / 2]]);
          const float sa0 = sa[hh * 128 + rr], sa1 = sa[hh * 128 + rr + 8];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const float2 s2 = *reinterpret_cast<const float2*>(sb + (hh / 2) * 128 + 8 * i + 2 * q);
            acc[4 * i] = fmaf(static_cast<float>(d[4 * i]), sa0 * s2.x, acc[4 * i]);
            acc[4 * i + 1] = fmaf(static_cast<float>(d[4 * i + 1]), sa0 * s2.y, acc[4 * i + 1]);
            acc[4 * i + 2] = fmaf(static_cast<float>(d[4 * i + 2]), sa1 * s2.x, acc[4 * i + 2]);
            acc[4 * i + 3] = fmaf(static_cast<float>(d[4 * i + 3]), sa1 * s2.y, acc[4 * i + 3]);
          }
        };
        issueH(accA, 0);
        issueH(accB, 1);
        wgmma_wait<1>();
        promoteG(accA, 0);
        issueH(accA, 2);
        wgmma_wait<1>();
        promoteG(accB, 1);
        issueH(accB, 3);
        wgmma_wait<1>();
        promoteG(accA, 2);
        issueH(accA, 4);
        wgmma_wait<1>();
        promoteG(accB, 3);
        issueH(accB, 5);
        wgmma_wait<1>();
        promoteG(accA, 4);
        issueH(accA, 6);
        wgmma_wait<1>();
        promoteG(accB, 5);
        issueH(accB, 7);
        wgmma_wait<1>();
        promoteG(accA, 6);
        wgmma_wait<0>();
        promoteG(accB, 7);
      } else if constexpr (kPP) {
        // two s32 accumulators: K block kb + 1's wgmmas run while K block kb is promoted
        static_assert(kD8Groups == 4, "the ping-pong sequence is written for four K blocks");
        int accA[64], accB[64];
        int stg[kD8Groups];
        auto issue = [&](int (&d)[64], int kb) {
          stg[kb] = stage;
          mbar_wait(&full[stage], phase);
          // no branch around the async wgmmas (a divergent path serializes them): an idle WG multiplies rows whose
          // scales are 0
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(d[i]);
          wgmma_fence();
#pragma unroll
          for (int kk = 0; kk < kD8BK / 32; ++kk)
            wgmma_s8_m64n128k32(d, gmma_desc(a_base + stage * kD8ABytes + kk * 32),
                                gmma_desc(sB + stage * kD8BBytes + kk * 32), kk);
          wgmma_commit();
          stage = stage + 1 == kD8Stages ? 0 : stage + 1;
          phase ^= stage == 0;
        };
        auto promote = [&](int (&d)[64], int kb) {
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(d[i]);
          if (lane == 0) mbar_arrive(&empty[stg[kb]]);
          {
            const float sa0 = sa[kb * 128 + rr], sa1 = sa[kb * 128 + rr + 8];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              const float2 s2 = *reinterpret_cast<const float2*>(sb + kb * 128 + 8 * i + 2 * q);
              acc[4 * i] = fmaf(static_cast<float>(d[4 * i]), sa0 * s2.x, acc[4 * i]);
              acc[4 * i + 1] = fmaf(static_cast<float>(d[4 * i + 1]), sa0 * s2.y, acc[4 * i + 1]);
              acc[4 * i + 2] = fmaf(static_cast<float>(d[4 * i + 2]), sa1 * s2.x, acc[4 * i + 2]);
              acc[4 * i + 3] = fmaf(static_cast<float>(d[4 * i + 3]), sa1 * s2.y, acc[4 * i + 3]);
            }
          }
        };
        auto wait1 = [&]() { wgmma_wait<1>(); };
        issue(accA, 0);
        issue(accB, 1);
        wait1();
        promote(accA, 0);
        issue(accA, 2);
        wait1();
        promote(accB, 1);
        issue(accB, 3);
        wait1();
        promote(accA, 2);
        wgmma_wait<0>();
        promote(accB, 3);
      } else {
        int accs[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) accs[i] = 0;
#pragma unroll 1
        for (int kb = 0; kb < kD8Groups; ++kb) {
        mbar_wait(&full[stage], phase);
        if (active && !(args.mode & 2)) {
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(accs[i]);
          wgmma_fence();
#pragma unroll
          for (int kk = 0; kk < kD8BK / 32; ++kk)
            wgmma_s8_m64n128k32(accs, gmma_desc(a_base + stage * kD8ABytes + kk * 32),
                                gmma_desc(sB + stage * kD8BBytes + kk * 32), kk);
          wgmma_commit();
          wgmma_wait<0>();
#pragma unroll
          for (int i = 0; i < 64; ++i) fence_operand_i(accs[i]);
        }
        if (lane == 0) mbar_arrive(&empty[stage]);
        stage = stage + 1 == kD8Stages ? 0 : stage + 1;
        phase ^= stage == 0;
        if (active && !(args.mode & 1)) {
          const float sa0 = sa[kb * 128 + rr], sa1 = sa[kb * 128 + rr + 8];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const float2 s2 = *reinterpret_cast<const float2*>(sb + kb * 128 + 8 * i + 2 * q);
            acc[4 * i] = fmaf(static_cast<float>(accs[4 * i]), sa0 * s2.x, acc[4 * i]);
            acc[4 * i + 1] = fmaf(static_cast<float>(accs[4 * i + 1]), sa0 * s2.y, acc[4 * i + 1]);
            acc[4 * i + 2] = fmaf(static_cast<float>(accs[4 * i + 2]), sa1 * s2.x, acc[4 * i + 2]);
            acc[4 * i + 3] = fmaf(static_cast<float>(accs[4 * i + 3]), sa1 * s2.y, acc[4 * i + 3]);
          }
        }
      }
      }
      if (lane == 0) mbar_arrive(&sempty[slot]);
      ++tile;
      if (!active) continue;
      if constexpr (kYQ) {
        float ma = 0.f, mb = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
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
        const int la = wi * 16 + lane / 4, lb = la + 8;
        if (tid % 128 == 0) tma_store_wait_read();
        named_bar_sync(1 + wg, 128);
#pragma unroll
        for (int i = 0; i < 16; ++i) {  // column 8 i + 2 q + {0, 1}: 16-byte chunk i / 2, byte 8 (i % 2) + 2 q
          const uint32_t cb = 8 * (i % 2) + 2 * q, ch = i / 2;
          const unsigned short va = static_cast<unsigned short>(__byte_perm(
              __float_as_uint(fmaf(acc[4 * i], ia, 12582912.f)), __float_as_uint(fmaf(acc[4 * i + 1], ia, 12582912.f)), 0x0040));
          const unsigned short vb = static_cast<unsigned short>(__byte_perm(
              __float_as_uint(fmaf(acc[4 * i + 2], ib, 12582912.f)), __float_as_uint(fmaf(acc[4 * i + 3], ib, 12582912.f)), 0x0040));
          sts_u16(d_base + la * 128 + ((ch ^ (la & 7)) << 4) + cb, va);
          sts_u16(d_base + lb * 128 + ((ch ^ (lb & 7)) << 4) + cb, vb);
        }
        if (q == 0) {
          args.ysc[static_cast<size_t>(info.y + wg * 64 + la) * kD8NB + n] = sa8;
          args.ysc[static_cast<size_t>(info.y + wg * 64 + lb) * kD8NB + n] = sb8;
        }
        fence_async_shared();
        named_bar_sync(1 + wg, 128);
        if (tid % 128 == 0 && !(args.mode & 4)) {
          tma_store_2d(&tm_y, d_base, n * 128, info.y + wg * 64);
          tma_store_commit();
        }
      } else {
      if (tid % 128 == 0) tma_store_wait_read();
      named_bar_sync(1 + wg, 128);
      const uint32_t row_addr = d_base + wi * 2048 + (lane % 16) * 128;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        stsm_x2(pack_bf16(acc[4 * i], acc[4 * i + 1]), pack_bf16(acc[4 * i + 2], acc[4 * i + 3]),
                row_addr + (i / 8) * 8192 + (((i % 8) ^ (lane % 8)) << 4));
      fence_async_shared();
      named_bar_sync(1 + wg, 128);
      if (tid % 128 == 0 && !(args.mode & 4)) {
#pragma unroll
        for (int a = 0; a < 2; ++a) tma_store_2d(&tm_y, d_base + a * 8192, n * 128 + a * 64, info.y + wg * 64);
        tma_store_commit();
      }
      }
    }
    if (tid % 128 == 0) tma_store_wait_all();
  }
}
// Lane PD2, separate-quant path: the kept up writes bf16 h; one warp per routed row quantizes it per 128-column group
// with the upq epilogue's rule (same bf16 inputs, same arithmetic -> the same hq / hs bytes).
__global__ void __launch_bounds__(256) w8_hquant_kernel(const __nv_bfloat16* __restrict__ h, const int* __restrict__ offs,
                                                        int max_rows, int8_t* __restrict__ hq, float* __restrict__ hs) {
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= max_rows || row >= __ldg(offs + kE)) return;
  const uint4* src = reinterpret_cast<const uint4*>(h + static_cast<size_t>(row) * kI) + lane * 2;
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  float f[16];
  const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 a = __bfloat1622float2(b0[k]), b = __bfloat1622float2(b1[k]);
    f[2 * k] = a.x, f[2 * k + 1] = a.y, f[8 + 2 * k] = b.x, f[8 + 2 * k + 1] = b.y;
    m = fmaxf(m, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
  }
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  const float inv = m > 0.f ? __fdiv_rn(127.f, m) : 0.f;
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t x = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int qv = max(-127, min(127, __float2int_rn(f[4 * k + j] * inv)));
      x |= (static_cast<uint32_t>(qv) & 0xffu) << (8 * j);
    }
    w[k] = x;
  }
  reinterpret_cast<uint4*>(hq + static_cast<size_t>(row) * kI)[lane] = make_uint4(w[0], w[1], w[2], w[3]);
  if ((lane & 7) == 0) hs[static_cast<size_t>(row) * kD8Groups + lane / 8] = __fdiv_rn(m, 127.f);
}
// int8 [outer][inner] rows, box 128 B x box_outer rows, 128B-swizzled (quantized h: stored by the up, read by the down)
static CUtensorMap make_map_i8(const void* base, uint64_t inner, uint64_t outer, uint32_t box_outer,
                               uint32_t box_inner = 128) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 box_inner == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (int8 h) failed: ", static_cast<int>(r));
  return map;
}
// The INT8 copy's w2 rows as a 3D int8 tensor [kE][kH][kI], box 128 B x 128 rows, 128B-swizzled for wgmma.
static CUtensorMap make_map_d8w(const uint8_t* q8) {
  CUtensorMap map;
  const cuuint64_t dims[3] = {static_cast<cuuint64_t>(kI), static_cast<cuuint64_t>(kH), static_cast<cuuint64_t>(kE)};
  const cuuint64_t strides[2] = {static_cast<cuuint64_t>(kI), static_cast<cuuint64_t>(kQ8Expert)};
  const cuuint32_t box[3] = {static_cast<cuuint32_t>(kD8BK), 128, 1};
  const cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<uint8_t*>(q8 + kQ8Q2), dims, strides,
                                 box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (W8A8 down) failed: ", static_cast<int>(r));
  return map;
}
// ==== LANE PD2 KERNELS END
// parts: 2 = the up (quantize, INT8 routed up, BF16 shared up), 4 = the king's BF16 down + combine;
// 16 (Lane PD2) = the up writes quantized h (w8_upq_kernel) and the routed down runs W8A8 (w8_dn_kernel), then the
// king's kernel over the shared pairs only (BF16 shared down + combine).
std::vector<torch::Tensor> forward_w8a8(torch::Tensor x, torch::Tensor logits, torch::Tensor gw, torch::Tensor q8,
                                        torch::Tensor w2, torch::Tensor s13, torch::Tensor s2, torch::Tensor cnt,
                                        int64_t parts) {
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range: ", T);
  TORCH_CHECK(parts >= 0 && parts < 512 && (parts & 8) == 0, "parts must be bits 2 | 4 | 16 | 32 | 64 | 128 | 256");
  const bool up_pp = (parts & 256) != 0;  // the kept W8A8 up with the ping-pong promotion (bit for bit)
  const bool dn8 = (parts & 16) != 0;
  const bool hq_sep = (parts & 32) != 0;  // PD2: the kept up + w8_hquant_kernel instead of w8_upq_kernel
  const bool dn_pp = (parts & 64) != 0;   // PD2: the ping-pong down
  const bool g64 = (parts & 128) != 0;    // PD2: h quantized per 64 columns in the kept up's epilogue (ping-pong down)
  TORCH_CHECK(!g64 || (dn8 && !hq_sep), "the g64 h path is parts 2 | 4 | 16 | 128");
  check_bf16(x, {T, kH}, "x");
  check_bf16(logits, {T, kE}, "router logits");
  check_bf16(gw, {kH}, "shared_expert_gate weight");
  TORCH_CHECK(q8.is_cuda() && q8.scalar_type() == torch::kUInt8 && q8.is_contiguous() && q8.dim() == 2 &&
                  q8.size(0) == kE && q8.size(1) == kQ8Expert,
              "q8 must be the contiguous CUDA uint8 [256, ", kQ8Expert, "] INT8 copy");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q8.data_ptr()) % 16 == 0 && q8.device() == x.device(), "q8 placement");
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
  torch::Tensor hq, hs;
  if (dn8) {
    hq = torch::empty({max_rows, kI}, x.options().dtype(torch::kInt8));
    hs = torch::empty({max_rows, g64 ? kW8NB : kD8Groups}, f32);
  }
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
  if (parts & 2) {
    auto xq = torch::empty({T, kH}, x.options().dtype(torch::kInt8));
    auto xs = torch::empty({T, kW8Groups}, f32);
    w8_quant_kernel<<<(T + 7) / 8, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                     static_cast<int>(T), xq.data_ptr<int8_t>(), xs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    W8Args wa;
    wa.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
    wa.pairs = reinterpret_cast<const int4*>(pairs.data_ptr<int>());
    wa.n_pairs = n_pairs.data_ptr<int>();
    wa.sorted_tok = sorted_tok.data_ptr<int>();
    wa.xq = xq.data_ptr<int8_t>();
    wa.xs = xs.data_ptr<float>();
    wa.q8 = reinterpret_cast<const uint8_t*>(q8.data_ptr());
    const CUtensorMap m_w = make_map_w8(wa.q8);
    const CUtensorMap m_h_st = make_map(h.data_ptr(), kI, max_rows, 64);
    static int w8_grid[64] = {};
    const int dev = at::cuda::current_device();
    TORCH_CHECK(dev < 64, "device index ", dev, " out of range");
    if (w8_grid[dev] == 0) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(w8_up_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8Smem));
      C10_CUDA_CHECK(cudaFuncSetAttribute(w8_up_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8Smem));
      C10_CUDA_CHECK(cudaFuncSetAttribute(w8_up_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8Smem));
      w8_grid[dev] = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    }
    if (g64) {
      const CUtensorMap m_hq64 = make_map_i8(hq.data_ptr(), kI, max_rows, 64, 64);
      w8_up_kernel<true><<<w8_grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_hq64, hs.data_ptr<float>());
    } else if (dn8 && hq_sep) {
      if (up_pp)
        w8_up_kernel<false, true><<<w8_grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_h_st, nullptr);
      else
        w8_up_kernel<false><<<w8_grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_h_st, nullptr);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      w8_hquant_kernel<<<(max_rows + 7) / 8, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
                                                              offs.data_ptr<int>(), static_cast<int>(max_rows),
                                                              hq.data_ptr<int8_t>(), hs.data_ptr<float>());
    } else if (dn8) {
      static bool upq_attr[64] = {};
      if (!upq_attr[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(w8_upq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8Smem));
        upq_attr[dev] = true;
      }
      const CUtensorMap m_hq_st = make_map_i8(hq.data_ptr(), kI, max_rows, 64);
      w8_upq_kernel<<<w8_grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_hq_st, hs.data_ptr<float>());
    } else {
      if (up_pp)
        w8_up_kernel<false, true><<<w8_grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_h_st, nullptr);
      else
        w8_up_kernel<false><<<w8_grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_h_st, nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gu = at::mm(x, s13.t());
    w8_shared_silu_kernel<<<T, kI / 4, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr()),
                                                    static_cast<int>(T), offs.data_ptr<int>(),
                                                    reinterpret_cast<__nv_bfloat16*>(h.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (parts & 4) {
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
    args.q8 = nullptr;
    args.mode = 0;
    const CUtensorMap m_s2 = make_map(s2.data_ptr(), kI, kH, 256);
    const CUtensorMap m_h_ld = make_map(h.data_ptr(), kI, max_rows, kBM);
    const CUtensorMap m_y = make_map(y.data_ptr(), kH, max_routed_rows, 64);
    const CUtensorMap m_w2 = make_map(w2.data_ptr(), kI, static_cast<uint64_t>(kE) * kH, 256);
    if (dn8) {
      TORCH_CHECK(parts & 2, "the W8A8 down reads the quantized h its up writes (parts 2 | 4 | 16)");
      auto spairs = torch::empty({max_mt, 4}, i32);
      auto n_spairs = torch::empty({1}, i32);
      D8Args da;
      da.mt = args.mt;
      da.pairs = args.pairs;
      da.n_pairs = args.n_pairs;
      da.hs = hs.data_ptr<float>();
      da.q8 = reinterpret_cast<const uint8_t*>(q8.data_ptr());
      da.spairs = reinterpret_cast<int4*>(spairs.data_ptr<int>());
      da.n_spairs = n_spairs.data_ptr<int>();
      da.mode = 0;  // ablation bits are for the isolated bench build only
      da.ysc = nullptr;
      static int d8_grid[64] = {};
      const int dev = at::cuda::current_device();
      TORCH_CHECK(dev < 64, "device index ", dev, " out of range");
      if (d8_grid[dev] == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(w8_dn_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kD8Smem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(w8_dn_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kD8Smem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(w8_dn_kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kD8Smem));
        d8_grid[dev] = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
      }
      const CUtensorMap m_d8w = make_map_d8w(da.q8);
      const CUtensorMap m_hq_ld = make_map_i8(hq.data_ptr(), kI, max_rows, kBM);
      if (g64)
        w8_dn_kernel<true, true><<<d8_grid[dev], 384, kD8Smem, stream>>>(da, m_d8w, m_hq_ld, m_y);
      else if (dn_pp)
        w8_dn_kernel<true, false><<<d8_grid[dev], 384, kD8Smem, stream>>>(da, m_d8w, m_hq_ld, m_y);
      else
        w8_dn_kernel<false, false><<<d8_grid[dev], 384, kD8Smem, stream>>>(da, m_d8w, m_hq_ld, m_y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      args.pairs = da.spairs;
      args.n_pairs = da.n_spairs;
      launch_gemm<false, false>(args, m_w2, m_s2, m_h_ld, m_y, stream);
    } else {
      launch_gemm<false, false>(args, m_w2, m_s2, m_h_ld, m_y, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, topk_ids, topk_w, h};
}
// ==== LANE PD2 END
// ==== LANE P W8A8 END
// ==== PORT ENTRIES (v2f port onto b9d3c95a): our lane P up and PD2 down as parts composable with route_i8 / down_i8
// route_i8o: route_i8 (the same three kernels, mt zero-filled) that also returns offs [kE + 2] (row offsets per
// expert; offs[kE] = the shared expert's first h row, needed by up_w8a8's shared silu and down_w8a8's h quantizer).
std::vector<torch::Tensor> route_i8o(torch::Tensor x, torch::Tensor logits, torch::Tensor gw, torch::Tensor cnt) {
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
  return {topk_ids, topk_w, pos_tk, sg, n_pairs, mt, pairs, sorted_tok, offs};
}
// up_w8a8: our lane P up over route_i8o's layout, writing h [max_rows, kI] bf16 where up_i8g writes it: x per token and
// 128-column group (w8_quant_kernel), the routed up on INT8 tensor cores over the copy's per-group scales as they are
// (w8_up_kernel; pp = the bit-for-bit ping-pong promotion), the shared expert's gate/up in BF16 (cuBLAS + silu).
torch::Tensor up_w8a8(torch::Tensor x, torch::Tensor q8, torch::Tensor s13, torch::Tensor mt, torch::Tensor pairs,
                      torch::Tensor n_pairs, torch::Tensor sorted_tok, torch::Tensor offs, bool pp) {
  const int64_t T = x.size(0);
  check_bf16(x, {T, kH}, "x");
  check_bf16(s13, {2 * kI, kH}, "shared gate_up weight");
  TORCH_CHECK(q8.is_cuda() && q8.scalar_type() == torch::kUInt8 && q8.is_contiguous() && q8.dim() == 2 &&
                  q8.size(0) == kE && q8.size(1) == kQ8Expert && reinterpret_cast<uintptr_t>(q8.data_ptr()) % 16 == 0,
              "q8 uint8 [256, Q8_EXPERT]");
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  TORCH_CHECK(sorted_tok.numel() == max_rows && mt.numel() == max_mt * 4 && pairs.numel() == max_mt * 4 &&
                  offs.numel() == kE + 2, "route_i8o's layout");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto h = torch::empty({max_rows, kI}, x.options());
  auto xq = torch::empty({T, kH}, x.options().dtype(torch::kInt8));
  auto xs = torch::empty({T, kW8Groups}, x.options().dtype(torch::kFloat32));
  w8_quant_kernel<<<(T + 7) / 8, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                   static_cast<int>(T), xq.data_ptr<int8_t>(), xs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  W8Args wa;
  wa.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
  wa.pairs = reinterpret_cast<const int4*>(pairs.data_ptr<int>());
  wa.n_pairs = n_pairs.data_ptr<int>();
  wa.sorted_tok = sorted_tok.data_ptr<int>();
  wa.xq = xq.data_ptr<int8_t>();
  wa.xs = xs.data_ptr<float>();
  wa.q8 = reinterpret_cast<const uint8_t*>(q8.data_ptr());
  const CUtensorMap m_w = make_map_w8(wa.q8);
  const CUtensorMap m_h_st = make_map(h.data_ptr(), kI, max_rows, 64);
  static int grid[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index ", dev, " out of range");
  if (grid[dev] == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(w8_up_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8Smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(w8_up_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8Smem));
    grid[dev] = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  }
  if (pp)
    w8_up_kernel<false, true><<<grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_h_st, nullptr);
  else
    w8_up_kernel<false><<<grid[dev], 384, kW8Smem, stream>>>(wa, m_w, m_h_st, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto gu = at::mm(x, s13.t());
  w8_shared_silu_kernel<<<T, kI / 4, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr()),
                                                  static_cast<int>(T), offs.data_ptr<int>(),
                                                  reinterpret_cast<__nv_bfloat16*>(h.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
// down_w8a8: our PD2 routed down over any up's bf16 h (route_i8o's layout): h's routed rows quantized per row and
// 128-column group (w8_hquant_kernel), the routed down on INT8 tensor cores over the copy's w2 rows and per-group scales
// (w8_dn_kernel, ping-pong promotion), then the king's kernel over the shared pairs only (BF16 shared down + combine).
// y8 (stacks with b9d3c95a's y8): the routed y is stored int8 with a scale per row and 128-column block (kYQ) and the
// combine reads it (kY8 + kY128); else bf16 y.
torch::Tensor down_w8a8(torch::Tensor x, torch::Tensor q8, torch::Tensor w2, torch::Tensor s2, torch::Tensor cnt,
                        torch::Tensor h, torch::Tensor offs, torch::Tensor topk_w, torch::Tensor pos_tk,
                        torch::Tensor sg, torch::Tensor n_pairs, torch::Tensor mt, torch::Tensor pairs,
                        torch::Tensor sorted_tok, bool y8) {
  const int64_t T = x.size(0);
  check_bf16(x, {T, kH}, "x");
  check_bf16(w2, {kE, kH, kI}, "w2");
  check_bf16(s2, {kH, kI}, "shared down weight");
  const int64_t max_rows = T * (kTopK + 1) + (kE + 1) * (kRowAlign - 1);
  const int64_t max_mt = (T * (kTopK + 1) + kBM - 1) / kBM + kE + 1;
  check_bf16(h, {max_rows, kI}, "h");
  TORCH_CHECK(q8.is_cuda() && q8.scalar_type() == torch::kUInt8 && q8.is_contiguous() && q8.dim() == 2 &&
                  q8.size(0) == kE && q8.size(1) == kQ8Expert && reinterpret_cast<uintptr_t>(q8.data_ptr()) % 16 == 0,
              "q8 uint8 [256, Q8_EXPERT]");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == torch::kInt32 && cnt.numel() == kStripes * kE + 1,
              "cnt must be int32 [2049]");
  TORCH_CHECK(sorted_tok.numel() == max_rows && mt.numel() == max_mt * 4 && pairs.numel() == max_mt * 4 &&
                  offs.numel() == kE + 2, "route_i8o's layout");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto i32 = x.options().dtype(torch::kInt32);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto hq = torch::empty({max_rows, kI}, x.options().dtype(torch::kInt8));
  auto hs = torch::empty({max_rows, kD8Groups}, f32);
  w8_hquant_kernel<<<(max_rows + 7) / 8, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
                                                          offs.data_ptr<int>(), static_cast<int>(max_rows),
                                                          hq.data_ptr<int8_t>(), hs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t max_routed_rows = T * kTopK + kE * (kRowAlign - 1);
  auto y = y8 ? torch::empty({max_routed_rows, kH}, x.options().dtype(torch::kUInt8))
              : torch::empty({max_routed_rows, kH}, x.options());
  auto ysc = torch::empty({y8 ? max_routed_rows : 0, kD8NB}, f32);
  auto out = torch::empty({T, kH}, x.options());
  auto spairs = torch::empty({max_mt, 4}, i32);
  auto n_spairs = torch::empty({1}, i32);
  D8Args da;
  da.mt = reinterpret_cast<const int4*>(mt.data_ptr<int>());
  da.pairs = reinterpret_cast<const int4*>(pairs.data_ptr<int>());
  da.n_pairs = n_pairs.data_ptr<int>();
  da.hs = hs.data_ptr<float>();
  da.q8 = reinterpret_cast<const uint8_t*>(q8.data_ptr());
  da.spairs = reinterpret_cast<int4*>(spairs.data_ptr<int>());
  da.n_spairs = n_spairs.data_ptr<int>();
  da.mode = 0;
  da.ysc = y8 ? ysc.data_ptr<float>() : nullptr;
  static int grid[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index ", dev, " out of range");
  if (grid[dev] == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(w8_dn_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kD8Smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(w8_dn_kernel<true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kD8Smem));
    grid[dev] = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  }
  const CUtensorMap m_d8w = make_map_d8w(da.q8);
  const CUtensorMap m_hq_ld = make_map_i8(hq.data_ptr(), kI, max_rows, kBM);
  const CUtensorMap m_y = y8 ? make_map_u8(y.data_ptr(), kH, max_routed_rows, 64)
                             : make_map(y.data_ptr(), kH, max_routed_rows, 64);
  if (y8)
    w8_dn_kernel<true, false, true><<<grid[dev], 384, kD8Smem, stream>>>(da, m_d8w, m_hq_ld, m_y);
  else
    w8_dn_kernel<true, false><<<grid[dev], 384, kD8Smem, stream>>>(da, m_d8w, m_hq_ld, m_y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GemmArgs args;
  args.mt = da.mt;
  args.pairs = da.spairs;
  args.n_pairs = da.n_spairs;
  args.sorted_tok = sorted_tok.data_ptr<int>();
  args.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  args.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
  args.ysc = y8 ? ysc.data_ptr<float>() : nullptr;
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
  if (y8)
    launch_gemm<false, false, true, true>(args, m_w2, m_s2, m_h_ld, m_y, stream);
  else
    launch_gemm<false, false>(args, m_w2, m_s2, m_h_ld, m_y, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
// ==== PORT ENTRIES END
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
  m.def("route_i8q", &qmoe::route_i8q, "route_i8 plus up_i8g's int8 x (IMG) and shared gate/up rows, one topk launch");
  m.def("up_i8g", &qmoe::up_i8g, "route_i8's up GEMM on per-group INT8 rows and the per-channel INT8 copy");
  m.def("forward_w8a8", &qmoe::forward_w8a8,
        "lane P: the MoE prefill block with the routed up on INT8 tensor cores (W8A8), shared up and down BF16");
  m.attr("PD2_DOWN_W8A8") = 16;
  m.def("route_i8o", &qmoe::route_i8o, "route_i8 plus the per-expert row offsets (port)");
  m.def("up_w8a8", &qmoe::up_w8a8, "lane P: the W8A8 routed up (g128 x, per-group fp32 promotion) + BF16 shared up over route_i8o");
  m.def("down_w8a8", &qmoe::down_w8a8, "lane PD2: the W8A8 routed down + the king's shared down / combine (y8: int8 y per 128 columns)");
  m.def("up_i8gp", &qmoe::up_i8gp, "up_i8g bit for bit, warp-specialized (producer warpgroup)");
#endif
}
