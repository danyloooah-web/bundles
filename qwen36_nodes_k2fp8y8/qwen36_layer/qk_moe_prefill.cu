// The crowned fd0012a2 qk_moe_prefill.cu with this line's bitwise prefill routing (v60 gate-first top-k; v68
// top-8 as eight warp max-reductions of packed (value order, id) keys and the shared expert's m-tiles /
// pairs written by the whole scan block).
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#ifndef QMOE_DN_STAGES4
#define QMOE_DN_STAGES4 1
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
__device__ __forceinline__ uint64_t gmma_desc(uint32_t smem_addr) {
  return static_cast<uint64_t>((smem_addr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(1024 >> 4) << 32) |
         (static_cast<uint64_t>(1) << 62);
}
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
  const __nv_bfloat16* y;
  const int* pos_tk;
  const float* topk_w;
  const float* sg;
  __nv_bfloat16* out;
  int* done;
};
template <bool kUp>
struct GemmCfg {
#if QMOE_DN_STAGES4
  static constexpr int kStages = 4;
#else
  static constexpr int kStages = kUp ? 4 : 3;
#endif
  static constexpr int NB = kUp ? kUpNB : kDnNB;
  static constexpr int NKB = kUp ? kUpKB : kDnKB;
#if QMOE_DN_STAGES4
  static constexpr uint32_t kDBytesWG = 64 * 128 * 2;
#else
  static constexpr uint32_t kDBytesWG = kUp ? 64 * 128 * 2 : 64 * 256 * 2;
#endif
  static constexpr uint32_t kSmem = kStages * (kABytes + kBBytes) + 2 * kDBytesWG + 2 * kStages * 8;
#if QMOE_DN_STAGES4
  static_assert(kSmem == 229440 && kSmem <= 232448, "4-stage MoE GEMM smem must fit the 227 KiB opt-in");
#endif
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
template <bool kUp>
__global__ void __launch_bounds__(384, 1)
    moe_gemm_kernel(const GemmArgs args, const __grid_constant__ CUtensorMap tm_b,
                    const __grid_constant__ CUtensorMap tm_bs, const __grid_constant__ CUtensorMap tm_a,
                    const __grid_constant__ CUtensorMap tm_d) {
  using Cfg = GemmCfg<kUp>;
  constexpr int kStages = Cfg::kStages, NB = Cfg::NB, NKB = Cfg::NKB;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + kStages * kABytes;
  uint8_t* sD = sB + kStages * kBBytes;
  uint64_t* full = reinterpret_cast<uint64_t*>(sD + 2 * Cfg::kDBytesWG);
  uint64_t* empty = full + kStages;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t rank = cluster_rank();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full[s], kUp ? 129 : 1);
      mbar_init(&empty[s], 16);
    }
    fence_barrier_init();
  }
  if (threadIdx.x == 32) {
    prefetch_tmap(&tm_b);
    prefetch_tmap(&tm_bs);
    if constexpr (!kUp) prefetch_tmap(&tm_a);
    prefetch_tmap(&tm_d);
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
      for (int kb = 0; kb < NKB; ++kb) {
        mbar_wait(&full[stage], phase);
        if (active) {
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
          wgmma_wait<0>();
        }
        if (lane < 2) mbar_arrive_cluster(&empty[stage], lane);
        stage = stage + 1 == kStages ? 0 : stage + 1;
        phase ^= stage == 0;
      }
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
        if (!shared_tile) {
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
              uint4 yv[2][kTopK];
              float w[2][kTopK];
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const int lr = r4 + 2 * j + lh;
                const bool ok = wg * 64 + wi * 16 + lr < info.z;
#pragma unroll
                for (int k = 0; k < kTopK; ++k) {
                  const int src = 2 * lr + k / 4;
                  const int pos = __shfl_sync(0xffffffffu, my_pos[k % 4], src);
                  w[j][k] = __shfl_sync(0xffffffffu, my_w[k % 4], src);
                  yv[j][k] = ok ? __ldcg(reinterpret_cast<const uint4*>(args.y + static_cast<size_t>(pos) * kH + col))
                                : make_uint4(0, 0, 0, 0);
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
                    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&yv[j][k]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                      const float2 f = __bfloat1622float2(b[q]);
                      a[2 * q] = fmaf(w[j][k], f.x, a[2 * q]);
                      a[2 * q + 1] = fmaf(w[j][k], f.y, a[2 * q + 1]);
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
template <bool kUp>
static void launch_gemm(const GemmArgs& args, const CUtensorMap& b, const CUtensorMap& bs, const CUtensorMap& a,
                        const CUtensorMap& d, cudaStream_t stream) {
  using Cfg = GemmCfg<kUp>;
  auto kernel = moe_gemm_kernel<kUp>;
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
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args, b, bs, a, d));
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
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qmoe::forward, "Qwen3.6 MoE prefill block (routing + gate/up + down + combine)");
  m.attr("COUNTERS") = qmoe::kStripes * qmoe::kE + 1;
}
