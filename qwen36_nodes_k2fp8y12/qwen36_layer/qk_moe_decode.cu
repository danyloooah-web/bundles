// Decode / MTP-verify MoE block (router, experts, shared expert, combine) for Qwen3.6-35B-A3B: the crowned fd0012a2
// kernel (hot8: INT8 expert units streamed from the i8x copy) with this line's bitwise units: the combine and the
// next layer's input norm folded into the ffn kernel's tail (v63) and the route as one 16-CTA cluster per token
// block exchanging logits through DSMEM (v65).
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace {
using bf16 = __nv_bfloat16;
constexpr int H = 2048, I = 512, E = 256, NE = E + 1, TOPK = 8, MAX_T = 256, SLOTS = (TOPK + 1) * MAX_T;
constexpr int GROUP = 8;
constexpr int UP_COLS = 8, UP_BLKS = I / UP_COLS;
constexpr int DN_ROWS = 64, DN_BLKS = H / DN_ROWS;
constexpr int STAGE_BYTES = 16384, USTAGES = 4, STAGES = 12, USLOTS = STAGES / USTAGES;
constexpr int CWARPS = 4, CTHREADS = CWARPS * 32, PRODUCER = CWARPS, PUBLISHER = CWARPS + 1;
constexpr int FFN_WARPS = CWARPS + 2, FFN_THREADS = FFN_WARPS * 32;
constexpr int RED_BYTES = 2 * CWARPS * 32 * 16;
constexpr int FFN_SMEM = STAGES * STAGE_BYTES + RED_BYTES;
constexpr int PUBQ = 8;
#ifndef QK_HOT
#define QK_HOT 10
#endif
constexpr int HOT = QK_HOT;
constexpr int HOT_CHUNKS = 3 * I * H * 2 / 16384;
static_assert(FFN_WARPS * NE * 4 <= USTAGES * STAGE_BYTES, "the sort histograms live in the ring's last unit slot");
#ifndef QK_HOT8
#define QK_HOT8 1
#endif
#if QK_HOT8
#ifndef QK_HOT8_MAX_N
#define QK_HOT8_MAX_N 256
#endif
#ifndef QK_HOT8_CHAINS
#define QK_HOT8_CHAINS 2
#endif
static_assert(QK_HOT8_CHAINS == 1 || QK_HOT8_CHAINS == 2 || QK_HOT8_CHAINS == 4, "chains divide a group's 4 steps");
constexpr int QG = 128;
constexpr int Q8_STAGE = STAGE_BYTES / 2;
constexpr int Q8_SCALES = 128;
constexpr int Q8_Q2 = 2 * I * H, Q8_C13 = Q8_Q2 + H * I, Q8_C2 = Q8_C13 + 2 * I * (H / QG) * 2;
constexpr int Q8_EXPERT = Q8_C2 + H * (I / QG) * 2;
constexpr int Q8_CHUNKS = Q8_EXPERT / 16384;
static_assert(4 * (H / QG) * 2 == Q8_SCALES && 16 * (I / QG) * 2 == Q8_SCALES, "one 128 B scale copy per stage");
static_assert(Q8_STAGE + Q8_SCALES <= STAGE_BYTES && Q8_EXPERT == 3194880 && Q8_EXPERT % 16384 == 0, "HOT8 layout");
#endif
constexpr int RT_EXP = 16, RT_TOK = 16, RT_THREADS = 256, RT_WARPS = RT_THREADS / 32, RT_XBLK = E / RT_EXP;
constexpr int C_TICKET = 0, C_UPDONE = MAX_T / RT_TOK, C_QUEUE = C_UPDONE + NE, C_ARRIVE = C_QUEUE + 1,
              C_TOTAL = C_ARRIVE + 2;
static_assert(C_ARRIVE % 2 == 0, "the 64-bit arrival count needs 8-byte alignment");
constexpr unsigned FULL = 0xffffffffu;
constexpr int K_UP = 0, K_DOWN = 1, K_DONE = 2;
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_n(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "W_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra W_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 lds128(const unsigned char* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint4 ldg128(const bf16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint4 ldcg128(const bf16* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_k32(float (&d)[4], const uint4& ag, const uint4& ag8, const uint4& b) {
  mma_bf16(d, ag.x, ag8.x, ag.y, ag8.y, b.x, b.y);
  mma_bf16(d, ag.z, ag8.z, ag.w, ag8.w, b.z, b.w);
}
__device__ __forceinline__ float4 sum_chains(const float (&a)[4][4]) {
  return make_float4(a[0][0] + a[1][0] + a[2][0] + a[3][0], a[0][1] + a[1][1] + a[2][1] + a[3][1],
                     a[0][2] + a[1][2] + a[2][2] + a[3][2], a[0][3] + a[1][3] + a[2][3] + a[3][3]);
}
struct RouteArgs {
  const bf16* x;
  const bf16* wr;
  const bf16* gw;
  const bf16* s13;
  const bf16* s2;
  const int64_t* valid;
  float* logits;
  int* ctr;
  int* ids;
  float* wts;
  int M;
};
__device__ __forceinline__ uint32_t logit_key(float v, int e) {
  const uint32_t b = v == 0.f ? 0u : __float_as_uint(v) >> 16;
  const uint32_t o = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
  return (o << 16) | static_cast<uint32_t>(255 - e);
}
__device__ __forceinline__ float key_logit(uint32_t k) {
  const uint32_t o = k >> 16;
  const uint32_t b = (o & 0x8000u) ? (o & 0x7fffu) : (~o & 0xffffu);
  return __uint_as_float(b << 16);
}
__device__ __forceinline__ uint32_t mapa_rank(uint32_t addr, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(addr), "r"(rank));
  return d;
}
__device__ __forceinline__ void st_cluster_f2(uint32_t addr, float a, float b) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void bulk_s2c(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "r"(src), "r"(bytes), "r"(bar)
               : "memory");
}
constexpr int RT_CL = 16;
static_assert(RT_CL == RT_TOK && RT_CL == 2 * RT_WARPS && E == 32 * RT_WARPS, "cluster layout");
__device__ __forceinline__ void route_token(const RouteArgs& p, int t, int M, int warp, int lane, int64_t vt,
                                            const float* tok, const bf16* gx, const bf16* gwv) {
  if (warp == 1) {
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncwarp();
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 a = *reinterpret_cast<const uint4*>(gx + 256 * i + 8 * lane);
      const uint4 w = *reinterpret_cast<const uint4*>(gwv + 256 * i + 8 * lane);
      d += bf_lo(a.x) * bf_lo(w.x) + bf_hi(a.x) * bf_hi(w.x) + bf_lo(a.y) * bf_lo(w.y) + bf_hi(a.y) * bf_hi(w.y) +
           bf_lo(a.z) * bf_lo(w.z) + bf_hi(a.z) * bf_hi(w.z) + bf_lo(a.w) * bf_lo(w.w) + bf_hi(a.w) * bf_hi(w.w);
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) d += __shfl_xor_sync(FULL, d, off);
    if (lane == 0) p.wts[TOPK * M + t] = 1.f / (1.f + __expf(-d));
    return;
  }
  if (vt == 0) {
    if (lane < TOPK) p.ids[t * TOPK + lane] = -1;
    return;
  }
  const float4 v0 = *reinterpret_cast<const float4*>(&tok[8 * lane]);
  const float4 v1 = *reinterpret_cast<const float4*>(&tok[8 * lane + 4]);
  uint32_t key[8] = {logit_key(v0.x, 8 * lane), logit_key(v0.y, 8 * lane + 1), logit_key(v0.z, 8 * lane + 2),
                     logit_key(v0.w, 8 * lane + 3), logit_key(v1.x, 8 * lane + 4), logit_key(v1.y, 8 * lane + 5),
                     logit_key(v1.z, 8 * lane + 6), logit_key(v1.w, 8 * lane + 7)};
  float mx = 0.f, ex_k = 0.f, sum = 0.f;
  int id_k = 0;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    uint32_t best = key[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) best = max(best, key[i]);
    best = __reduce_max_sync(FULL, best);
    const float v = key_logit(best);
    if (k == 0) mx = v;
    const float ex = __expf(v - mx);
    if (lane == k) ex_k = ex, id_k = 255 - static_cast<int>(best & 0xffu);
    sum += ex;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (key[i] == best) key[i] = 0;
  }
  if (lane < TOPK) {
    p.ids[t * TOPK + lane] = id_k;
    p.wts[t * TOPK + lane] = ex_k / sum;
  }
}
__global__ void __launch_bounds__(RT_THREADS, 2) moe_route_kernel(const RouteArgs p) {
  __shared__ __align__(128) float part[RT_CL][E];
  __shared__ __align__(128) float stage[RT_TOK][E + 4];
  __shared__ __align__(16) float tok[E];
  __shared__ __align__(8) uint64_t inbar;
  __shared__ __align__(16) bf16 gx[H], gwv[H];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int r = blockIdx.x, slice = r >> 1, chain = r & 1;
  const int t0 = blockIdx.y * RT_TOK, M = p.M, t_end = min(M, t0 + RT_TOK);
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    prefetch_l2(p.wr + static_cast<int64_t>(r) * RT_EXP * H, RT_EXP * H * 2);
    if (r == 0) prefetch_l2(p.gw, H * 2);
  }
  if (threadIdx.x == 0) {
    mbar_init(smem_u32(&inbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync_all();
  const int ta = t0 + g, tb = ta + 8;
  const bool oka = ta < M, okb = tb < M;
  const int kb = 256 * slice + 32 * chain + 8 * c;
  const bf16* xa = p.x + static_cast<int64_t>(oka ? ta : 0) * H + kb;
  const bf16* xb = p.x + static_cast<int64_t>(okb ? tb : 0) * H + kb;
  const bf16* wrow = p.wr + static_cast<int64_t>(32 * warp + g) * H + kb;
  const int t = t0 + r;
  uint4 a0[4], a1[4], b[4][4];
  int64_t vt = 0;
  griddep_wait();
  griddep_launch();
  if (t < t_end) {
    if (warp == 0) vt = p.valid[t];
    if (warp == 1) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(gx + 256 * i + 8 * lane)),
                     "l"(p.x + static_cast<int64_t>(t) * H + 256 * i + 8 * lane)
                     : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(gwv + 256 * i + 8 * lane)),
                     "l"(p.gw + 256 * i + 8 * lane)
                     : "memory");
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int n = 0; n < 4; ++n) b[n][j] = ldg128(wrow + static_cast<int64_t>(8 * n) * H + 64 * j);
    a0[j] = oka ? ldg128(xa + 64 * j) : make_uint4(0, 0, 0, 0);
    a1[j] = okb ? ldg128(xb + 64 * j) : make_uint4(0, 0, 0, 0);
  }
  float acc[4][4] = {};
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < 4; ++n) mma_k32(acc[n], a0[j], a1[j], b[n][j]);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    *reinterpret_cast<float2*>(&stage[g][32 * warp + 8 * n + 2 * c]) = make_float2(acc[n][0], acc[n][1]);
    *reinterpret_cast<float2*>(&stage[g + 8][32 * warp + 8 * n + 2 * c]) = make_float2(acc[n][2], acc[n][3]);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x < RT_TOK && t0 + static_cast<int>(threadIdx.x) < t_end) {
    const uint32_t i = threadIdx.x;
    bulk_s2c(mapa_rank(smem_u32(&part[r][0]), i), smem_u32(&stage[i][0]), E * 4, mapa_rank(smem_u32(&inbar), i));
  }
  const bool ranks = t < t_end;
  if (ranks) {
    if (threadIdx.x == 0) mbar_expect(smem_u32(&inbar), RT_CL * E * 4);
    mbar_wait(smem_u32(&inbar), 0);
  }
  cluster_arrive_relaxed();
  if (ranks) {
    const int e = threadIdx.x;
    float s = part[0][e] + part[1][e];
#pragma unroll
    for (int w = 1; w < RT_WARPS; ++w) s = s + (part[2 * w][e] + part[2 * w + 1][e]);
    tok[e] = rbf(s);
  }
  __syncthreads();
  if (ranks && warp < 2) route_token(p, t, M, warp, lane, vt, tok, gx, gwv);
  cluster_wait();
}
struct NextNorm {
  const bf16* resid;
  const bf16* w;
  float eps;
  bf16* normed;
  bf16* next_resid;
};
__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float d;
  asm("mul.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
struct FfnArgs {
  const bf16* x;
  const int64_t* valid;
  const bf16* w13;
  const bf16* w2;
  const bf16* s13;
  const bf16* s2;
  const int* ids;
  const float* wts;
  int* ctr;
  bf16* hbuf;
  float* ybuf;
  int* hot;
  int M;
  bf16* out;
  NextNorm nn;
#if QK_HOT8
  const int* slot8;
  const unsigned char* q8;
  int64_t q8_stride;
  int q8_slots;
#endif
};
struct Unit {
  int kind, e, blk, n, off;
#if QK_HOT8
  int q8;
#endif
};
struct Tables {
  int list[SLOTS];
  float wts[SLOTS];
  int cnt[NE], off[NE], order[NE];
  int groups;
};
__device__ __forceinline__ Unit decode_unit(const Tables& T, int u) {
  Unit un{};
  const int n_up = UP_BLKS * T.groups;
  if (u >= (UP_BLKS + DN_BLKS) * T.groups) {
    un.kind = K_DONE;
    return un;
  }
  const bool up = u < n_up;
  const int v = up ? u : u - n_up, gi = up ? v / UP_BLKS : v / DN_BLKS;
  un.kind = up ? K_UP : K_DOWN;
  un.e = T.order[gi];
  un.blk = up ? v % UP_BLKS : v % DN_BLKS;
  un.n = T.cnt[un.e];
  un.off = T.off[un.e];
  return un;
}
__device__ __forceinline__ int n_groups(const Unit& un) { return (un.n + GROUP - 1) / GROUP; }
__device__ __forceinline__ int slot_token(int slot, int routed) { return slot < routed ? slot / TOPK : slot - routed; }
__device__ __forceinline__ bool load_b(const FfnArgs& p, const Tables& T, const Unit& un, int grp, bool known,
                                       uint4 (&b)[16], int warp, int lane) {
  const int g = lane >> 2, c = lane & 3, j = GROUP * grp + g;
  const bool has = j < un.n;
  const int slot = has ? T.list[un.off + j] : 0;
  if (un.kind == K_UP) {
    const bf16* row = p.x + static_cast<int64_t>(slot_token(slot, TOPK * p.M)) * H + 512 * warp + 8 * c;
#pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = has ? ldg128(row + 32 * i) : make_uint4(0, 0, 0, 0);
    return true;
  }
  if (!known) {
    int ready = 0;
    if (lane == 0) ready = ld_acquire(p.ctr + C_UPDONE + un.e) >= UP_BLKS;
    if (!__shfl_sync(FULL, ready, 0)) return false;
    __syncwarp();
  }
  const bf16* row = p.hbuf + static_cast<int64_t>(slot) * I + 8 * c;
#pragma unroll
  for (int i = 0; i < 16; ++i) b[i] = has ? ldcg128(row + 32 * i) : make_uint4(0, 0, 0, 0);
  return true;
}
#if QK_HOT8
__device__ __forceinline__ float i8f(uint32_t biased, uint32_t sel) {
  return __uint_as_float(__byte_perm(biased, 0x4B000000u, sel)) - 8388736.f;
}
__device__ __forceinline__ uint32_t bf16x2_hi(float lo, float hi) {
  return __byte_perm(__float_as_uint(lo), __float_as_uint(hi), 0x7632);
}
__device__ __forceinline__ uint2 lds64(const unsigned char* p) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ uint4 deq8(uint2 v) {
  const uint32_t a = v.x ^ 0x80808080u, b = v.y ^ 0x80808080u;
  return make_uint4(bf16x2_hi(i8f(a, 0x7650), i8f(a, 0x7651)), bf16x2_hi(i8f(a, 0x7652), i8f(a, 0x7653)),
                    bf16x2_hi(i8f(b, 0x7650), i8f(b, 0x7651)), bf16x2_hi(i8f(b, 0x7652), i8f(b, 0x7653)));
}
__device__ __forceinline__ float h2f(uint32_t pair, int half) {
  const unsigned short h = static_cast<unsigned short>(half ? pair >> 16 : pair & 0xffffu);
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
template <int N>
__device__ __forceinline__ float4 sum_q8(const float (&a)[N][4]) {
  float4 s = make_float4(a[0][0], a[0][1], a[0][2], a[0][3]);
#pragma unroll
  for (int k = 1; k < N; ++k) s = add4(s, make_float4(a[k][0], a[k][1], a[k][2], a[k][3]));
  return s;
}
__device__ __forceinline__ int q8_slot(const FfnArgs& p, const Unit& un) {
  if (p.slot8 == nullptr || un.kind == K_DONE || un.e < 0 || un.e >= E || un.n > QK_HOT8_MAX_N) return 0;
  const int s = __ldg(p.slot8 + un.e);
  return s >= 0 && s < p.q8_slots ? s + 1 : 0;
}
__device__ __forceinline__ void issue_unit8(const FfnArgs& p, const Unit& un, unsigned char* ring, uint64_t* full,
                                            int slot0, uint64_t pol) {
  const unsigned char* blk = p.q8 + static_cast<int64_t>(un.q8 - 1) * p.q8_stride;
  for (int j = 0; j < USTAGES; ++j) {
    const uint32_t fb = smem_u32(&full[slot0 + j]);
    const unsigned char *q, *s;
    if (un.kind == K_UP) {
      const int row = (j < 2 ? 0 : I) + UP_COLS * un.blk + 4 * (j & 1);
      q = blk + static_cast<int64_t>(row) * H;
      s = blk + Q8_C13 + row * (H / QG) * 2;
    } else {
      const int row = DN_ROWS * un.blk + 16 * j;
      q = blk + Q8_Q2 + static_cast<int64_t>(row) * I;
      s = blk + Q8_C2 + row * (I / QG) * 2;
    }
    unsigned char* dst = ring + (slot0 + j) * STAGE_BYTES;
    mbar_expect(fb, Q8_STAGE + Q8_SCALES);
    bulk_g2s(smem_u32(dst), q, Q8_STAGE, fb, pol);
    bulk_g2s(smem_u32(dst + Q8_STAGE), s, Q8_SCALES, fb, pol);
  }
}
__device__ __forceinline__ float4 up_q8(const unsigned char* st, const uint4 (&b)[16], int warp, int g, int c) {
  const unsigned char* qg = st + (g >> 2) * STAGE_BYTES + (g & 3) * H;
  const unsigned char* qu = qg + 2 * STAGE_BYTES;
  const int sc = Q8_STAGE + ((g & 3) * (H / QG) + 4 * warp) * 2;
  const uint2 sg = lds64(st + (g >> 2) * STAGE_BYTES + sc), su = lds64(st + (2 + (g >> 2)) * STAGE_BYTES + sc);
  float4 tot = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float acc[QK_HOT8_CHAINS][4] = {};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = 4 * q + r, kk = 512 * warp + 32 * i + 8 * c;
      mma_k32(acc[r % QK_HOT8_CHAINS], deq8(lds64(qg + kk)), deq8(lds64(qu + kk)), b[i]);
    }
    const float4 part = sum_q8(acc);
    const float fg = h2f(q < 2 ? sg.x : sg.y, q & 1), fu = h2f(q < 2 ? su.x : su.y, q & 1);
    tot.x = __fmaf_rn(fg, part.x, tot.x);
    tot.y = __fmaf_rn(fg, part.y, tot.y);
    tot.z = __fmaf_rn(fu, part.z, tot.z);
    tot.w = __fmaf_rn(fu, part.w, tot.w);
  }
  return tot;
}
__device__ __forceinline__ float4 down_q8(const unsigned char* sw, const uint4 (&b)[16], int g, int c) {
  const uint2 s0 = lds64(sw + Q8_STAGE + g * (I / QG) * 2), s1 = lds64(sw + Q8_STAGE + (g + 8) * (I / QG) * 2);
  float4 tot = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float acc[QK_HOT8_CHAINS][4] = {};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = 4 * q + r, kk = 32 * i + 8 * c;
      mma_k32(acc[r % QK_HOT8_CHAINS], deq8(lds64(sw + g * I + kk)), deq8(lds64(sw + (g + 8) * I + kk)), b[i]);
    }
    const float4 part = sum_q8(acc);
    const float f0 = h2f(q < 2 ? s0.x : s0.y, q & 1), f1 = h2f(q < 2 ? s1.x : s1.y, q & 1);
    tot.x = __fmaf_rn(f0, part.x, tot.x);
    tot.y = __fmaf_rn(f0, part.y, tot.y);
    tot.z = __fmaf_rn(f1, part.z, tot.z);
    tot.w = __fmaf_rn(f1, part.w, tot.w);
  }
  return tot;
}
#endif
__device__ __forceinline__ void issue_unit(const FfnArgs& p, const Unit& un, unsigned char* ring, uint64_t* full,
                                           int slot0, uint64_t pol) {
#if QK_HOT8
  if (un.q8) {
    issue_unit8(p, un, ring, full, slot0, pol);
    return;
  }
#endif
  for (int j = 0; j < USTAGES; ++j) {
    const uint32_t fb = smem_u32(&full[slot0 + j]);
    const bf16* src;
    if (un.kind == K_UP) {
      const int row = (j < 2 ? 0 : I) + UP_COLS * un.blk + 4 * (j & 1);
      src = (un.e == E ? p.s13 : p.w13 + static_cast<int64_t>(un.e) * 2 * I * H) + static_cast<int64_t>(row) * H;
    } else {
      src = (un.e == E ? p.s2 : p.w2 + static_cast<int64_t>(un.e) * H * I) +
            static_cast<int64_t>(DN_ROWS * un.blk + 16 * j) * I;
    }
    mbar_expect(fb, STAGE_BYTES);
    bulk_g2s(smem_u32(ring + (slot0 + j) * STAGE_BYTES), src, STAGE_BYTES, fb, pol);
  }
}
__device__ __forceinline__ void combine(const FfnArgs& p, float* hs, int tid, int warp, int lane) {
  const int M = p.M, grid = gridDim.x;
  const NextNorm& nn = p.nn;
  uint4 rv[2], wv[2];
  if (nn.resid != nullptr && static_cast<int>(blockIdx.x) < M) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      rv[hf] = *reinterpret_cast<const uint4*>(nn.resid + static_cast<int64_t>(blockIdx.x) * H + 1024 * hf + 8 * tid);
      wv[hf] = *reinterpret_cast<const uint4*>(nn.w + 1024 * hf + 8 * tid);
    }
  }
  named_sync(1, CTHREADS);
  griddep_launch();
  const bool rows = static_cast<int>(blockIdx.x) < M;
  if (tid == 0) {
    unsigned long long* arrive = reinterpret_cast<unsigned long long*>(p.ctr + C_ARRIVE);
    unsigned long long old, now;
    asm volatile("atom.add.release.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(arrive) : "memory");
    const unsigned long long target = (old / grid + 1) * grid;
    if (!rows) {
    } else if (old + 1 != target) {
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(now) : "l"(arrive) : "memory");
      } while (now < target);
    } else {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  if (!rows) return;
  named_sync(1, CTHREADS);
  if (blockIdx.x == 0)
    for (int i = C_UPDONE + tid; i <= C_QUEUE; i += CTHREADS) p.ctr[i] = 0;
  for (int t = blockIdx.x; t < M; t += grid) {
    if (nn.resid != nullptr && t != static_cast<int>(blockIdx.x)) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf)
        rv[hf] = *reinterpret_cast<const uint4*>(nn.resid + static_cast<int64_t>(t) * H + 1024 * hf + 8 * tid);
    }
    float a[2][8] = {};
    if (p.valid[t] != 0) {
      float4 v[TOPK + 1][2][2];
#pragma unroll
      for (int k = 0; k <= TOPK; ++k) {
        const int slot = k < TOPK ? t * TOPK + k : TOPK * M + t;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const float4* r = reinterpret_cast<const float4*>(p.ybuf + static_cast<int64_t>(slot) * H + 1024 * hf + 8 * tid);
          v[k][hf][0] = __ldcg(r);
          v[k][hf][1] = __ldcg(r + 1);
        }
      }
#pragma unroll
      for (int k = 0; k <= TOPK; ++k)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const float4 v0 = v[k][hf][0], v1 = v[k][hf][1];
          a[hf][0] += v0.x, a[hf][1] += v0.y, a[hf][2] += v0.z, a[hf][3] += v0.w, a[hf][4] += v1.x, a[hf][5] += v1.y,
              a[hf][6] += v1.z, a[hf][7] += v1.w;
        }
    }
    __nv_bfloat162 o[2][4];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
      for (int i = 0; i < 4; ++i) o[hf][i] = __floats2bfloat162_rn(a[hf][2 * i], a[hf][2 * i + 1]);
      *reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(t) * H + 1024 * hf + 8 * tid) =
          *reinterpret_cast<const uint4*>(o[hf]);
    }
    if (nn.resid == nullptr) continue;
    float h[2][8];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&rv[hf]);
      __nv_bfloat162 ro[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 x = __bfloat1622float2(o[hf][q]), r = __bfloat1622float2(rb[q]);
        h[hf][2 * q] = add_rn(x.x, r.x);
        h[hf][2 * q + 1] = add_rn(x.y, r.y);
        ro[q] = __floats2bfloat162_rn(h[hf][2 * q], h[hf][2 * q + 1]);
      }
      *reinterpret_cast<uint4*>(nn.next_resid + static_cast<int64_t>(t) * H + 1024 * hf + 8 * tid) =
          *reinterpret_cast<const uint4*>(ro);
#pragma unroll
      for (int q = 0; q < 8; ++q) hs[1024 * hf + 8 * tid + q] = h[hf][q];
    }
    named_sync(1, CTHREADS);
    if (warp == 0) {
      float s = 0.f;
#pragma unroll
      for (int v1 = 0; v1 < 8; ++v1)
#pragma unroll
        for (int v0 = 0; v0 < 8; ++v0) {
          const float v = hs[256 * v1 + 8 * lane + v0];
          s = fma_rn(v, v, s);
        }
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) s = add_rn(s, __shfl_xor_sync(FULL, s, off));
      if (lane == 0) {
        float r;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(fma_rn(s, 1.f / 2048.f, nn.eps)));
        hs[H] = r;
      }
    }
    named_sync(1, CTHREADS);
    const float rstd = hs[H];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv[hf]);
      __nv_bfloat162 y[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 wf = __bfloat1622float2(wb[q]);
        y[q] = __floats2bfloat162_rn(mul_rn(mul_rn(h[hf][2 * q], rstd), add_rn(wf.x, 1.f)),
                                     mul_rn(mul_rn(h[hf][2 * q + 1], rstd), add_rn(wf.y, 1.f)));
      }
      *reinterpret_cast<uint4*>(nn.normed + static_cast<int64_t>(t) * H + 1024 * hf + 8 * tid) =
          *reinterpret_cast<const uint4*>(y);
    }
  }
}
__global__ void __maxnreg__(216) moe_ffn_kernel(const FfnArgs p) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES], hbar[USLOTS], pub_full[PUBQ], pub_empty[PUBQ];
  __shared__ Unit hdr[USLOTS];
  __shared__ int pub_e[PUBQ];
  __shared__ Tables T;
  unsigned char* ring = smem;
  float4* red = reinterpret_cast<float4*>(smem + STAGES * STAGE_BYTES);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(smem_u32(&full[i]), 1);
      mbar_init(smem_u32(&empty[i]), CWARPS);
    }
    for (int i = 0; i < USLOTS; ++i) mbar_init(smem_u32(&hbar[i]), 1);
    for (int i = 0; i < PUBQ; ++i) {
      mbar_init(smem_u32(&pub_full[i]), 1);
      mbar_init(smem_u32(&pub_empty[i]), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int M = p.M;
  const bool ahead = M > 96;
  int c0 = 0, c1 = 0;
  bool early = false;
  uint64_t pol = 0;
  if (warp == PRODUCER) {
    int nsh = 0;
    for (int t = lane; t < M; t += 32) nsh += p.valid[t] != 0 ? 1 : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) nsh += __shfl_xor_sync(FULL, nsh, o);
    if (lane == 0) {
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      int* queue = p.ctr + C_QUEUE;
      c0 = atomicAdd(queue, 1);
      c1 = ahead ? atomicAdd(queue, 1) : 0;
      early = nsh > 0 && c0 < UP_BLKS;
      if (early) {
        const Unit un{K_UP, E, c0, nsh, 0};
        hdr[0] = un;
        mbar_arrive(smem_u32(&hbar[0]));
        issue_unit(p, un, ring, full, 0, pol);
      }
    }
  }
  const int routed = TOPK * M, nslots = routed + M, R = (nslots + FFN_WARPS - 1) / FFN_WARPS;
  const int lo = warp * R, hi = min(nslots, lo + R);
  unsigned char* hist_base = ring + (USLOTS - 1) * USTAGES * STAGE_BYTES;
  int* hist = reinterpret_cast<int*>(hist_base) + warp * NE;
  for (int i = lane; i < NE; i += 32) hist[i] = 0;
  if (warp == PUBLISHER && lane == 0) {
    const int64_t n13 = 2 * I * H * 2, n2 = H * I * 2;
    const int64_t c13 = (n13 / gridDim.x + 255) & ~int64_t{255}, c2 = (n2 / gridDim.x + 255) & ~int64_t{255};
    const int64_t o13 = c13 * blockIdx.x, o2 = c2 * blockIdx.x;
    if (o13 < n13) prefetch_l2(reinterpret_cast<const unsigned char*>(p.s13) + o13, static_cast<uint32_t>(min(c13, n13 - o13)));
    if (o2 < n2) prefetch_l2(reinterpret_cast<const unsigned char*>(p.s2) + o2, static_cast<uint32_t>(min(c2, n2 - o2)));
  }
  __syncwarp();
  if (warp == PUBLISHER && p.hot != nullptr) {
    const int nhot = M <= 8 ? 1 : HOT;
    for (int k = 0; k < nhot; ++k) {
      const int e = __ldcg(p.hot + k);
      if (e < 0 || e >= E) continue;
#if QK_HOT8
      const int s8 = p.slot8 != nullptr && M <= QK_HOT8_MAX_N ? __ldg(p.slot8 + e) : -1;
      if (s8 >= 0 && s8 < p.q8_slots) {
        const unsigned char* q = p.q8 + static_cast<int64_t>(s8) * p.q8_stride;
        for (int c = blockIdx.x + gridDim.x * lane; c < Q8_CHUNKS; c += gridDim.x * 32)
          prefetch_l2(q + static_cast<int64_t>(c) * 16384, 16384);
        continue;
      }
#endif
      const unsigned char* a = reinterpret_cast<const unsigned char*>(p.w13 + static_cast<int64_t>(e) * 2 * I * H);
      const unsigned char* b = reinterpret_cast<const unsigned char*>(p.w2 + static_cast<int64_t>(e) * H * I);
      for (int c = blockIdx.x + gridDim.x * lane; c < HOT_CHUNKS; c += gridDim.x * 32)
        prefetch_l2(c < 2 * HOT_CHUNKS / 3 ? a + static_cast<int64_t>(c) * 16384
                                           : b + static_cast<int64_t>(c - 2 * HOT_CHUNKS / 3) * 16384, 16384);
    }
  }
  griddep_wait();
  constexpr int SPAN = (SLOTS + FFN_WARPS * 32 - 1) / (FFN_WARPS * 32);
  int ex[SPAN];
  float wx[SPAN];
#pragma unroll
  for (int j = 0; j < SPAN; ++j) {
    const int i = lo + 32 * j + lane;
    ex[j] = i >= hi ? -1 : i < routed ? __ldcg(p.ids + i) : (p.valid[i - routed] != 0 ? E : -1);
    wx[j] = i < hi ? __ldcg(p.wts + i) : 0.f;
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < SPAN; ++j)
    if (ex[j] >= 0) atomicAdd(&hist[ex[j]], 1);
  __syncthreads();
  for (int e = tid; e < NE; e += FFN_THREADS) {
    int* h = reinterpret_cast<int*>(hist_base) + e;
    int run = 0;
    for (int w = 0; w < FFN_WARPS; ++w) {
      const int n = h[w * NE];
      h[w * NE] = run;
      run += n;
    }
    T.cnt[e] = run;
  }
  __syncthreads();
  if (warp == 0) {
    const int nsh = T.cnt[E];
    int nlist = nsh, gi = nsh > 0 ? 1 : 0;
    if (lane == 0) {
      T.order[0] = E;
      T.off[E] = 0;
    }
    for (int i = 0; i < E / 32; ++i) {
      const int e = 32 * i + lane, n = T.cnt[e];
      const unsigned m = __ballot_sync(FULL, n > 0);
      int sl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(FULL, sl, o);
        if (lane >= o) sl += a;
      }
      if (n > 0) T.order[gi + __popc(m & ((1u << lane) - 1))] = e;
      T.off[e] = nlist + sl - n;
      nlist += __shfl_sync(FULL, sl, 31);
      gi += __popc(m);
    }
    if (lane == 0) T.groups = gi;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < SPAN; ++j) {
    if (lo + 32 * j < hi) {
      const int i = lo + 32 * j + lane, e = ex[j];
      const unsigned m = __match_any_sync(FULL, e), below = m & ((1u << lane) - 1);
      const int at = e >= 0 ? T.off[e] + hist[e] + __popc(below) : 0;
      __syncwarp();
      if (e >= 0) {
        T.list[at] = i;
        T.wts[i] = wx[j];
        if (below == 0) hist[e] += __popc(m);
      }
      __syncwarp();
    }
  }
  __syncthreads();
  if (warp == PRODUCER) {
    if (lane != 0) return;
    int* queue = p.ctr + C_QUEUE;
    for (int us = 0;; ++us) {
      if (us > 0 || !early) {
        const int hs = us % USLOTS, slot0 = USTAGES * hs;
        const uint32_t ph = (us / USLOTS) & 1;
        for (int j = 0; j < USTAGES; ++j) mbar_wait(smem_u32(&empty[slot0 + j]), ph ^ 1);
#if QK_HOT8
        Unit un = decode_unit(T, ahead || us == 0 ? c0 : atomicAdd(queue, 1));
        un.q8 = q8_slot(p, un);
#else
        const Unit un = decode_unit(T, ahead || us == 0 ? c0 : atomicAdd(queue, 1));
#endif
        hdr[hs] = un;
        mbar_arrive(smem_u32(&hbar[hs]));
        if (un.kind == K_DONE) return;
        issue_unit(p, un, ring, full, slot0, pol);
      }
      if (ahead) {
        c0 = c1;
        c1 = atomicAdd(queue, 1);
      }
    }
  }
  if (warp == PUBLISHER) {
    if (blockIdx.x == 0 && p.hot != nullptr) {
      int cnt[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) cnt[i] = T.cnt[8 * lane + i];
      for (int k = 0; k < HOT; ++k) {
        uint32_t best = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (cnt[i] > 0) best = max(best, (static_cast<uint32_t>(cnt[i]) << 8) | static_cast<uint32_t>(255 - 8 * lane - i));
        best = __reduce_max_sync(FULL, best);
        const int e = best != 0 ? 255 - static_cast<int>(best & 0xffu) : -1;
        if (lane == 0) p.hot[k] = e;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (8 * lane + i == e) cnt[i] = 0;
      }
    }
    if (lane != 0) return;
    for (int q = 0;; ++q) {
      const int s = q % PUBQ;
      mbar_wait(smem_u32(&pub_full[s]), (q / PUBQ) & 1);
      const int e = pub_e[s];
      if (e < 0) return;
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" ::"l"(p.ctr + C_UPDONE + e) : "memory");
      mbar_arrive(smem_u32(&pub_empty[s]));
    }
  }
  const int g = lane >> 2, c = lane & 3;
  int us = 0, grp = 0, gs = 0, pq = 0;
  mbar_wait(smem_u32(&hbar[0]), 0);
  Unit cur = hdr[0];
  uint4 b0[16], b1[16];
  bool r0 = cur.kind != K_DONE && load_b(p, T, cur, 0, false, b0, warp, lane), r1 = false;
  auto publish = [&](int e) {
    const int s = pq % PUBQ;
    mbar_wait(smem_u32(&pub_empty[s]), ((pq / PUBQ) & 1) ^ 1);
    pub_e[s] = e;
    mbar_arrive(smem_u32(&pub_full[s]));
    ++pq;
  };
  auto step = [&](uint4(&bc)[16], bool rc, uint4(&bn)[16], bool& rn) -> bool {
    if (cur.kind == K_DONE) {
      if (warp == 0 && lane == 0) publish(-1);
      return true;
    }
    const int G = n_groups(cur);
    Unit nx = cur;
    int ngrp = grp + 1, nus = us;
    if (ngrp == G) {
      nus = us + 1;
      ngrp = 0;
      const int hn = nus % USLOTS;
      mbar_wait(smem_u32(&hbar[hn]), (nus / USLOTS) & 1);
      nx = hdr[hn];
    }
    rn = nx.kind != K_DONE && load_b(p, T, nx, ngrp, ngrp != 0 && rc, bn, warp, lane);
    const int slot0 = USTAGES * (us % USLOTS);
    const uint32_t ph = (us / USLOTS) & 1;
    const unsigned char* st = ring + slot0 * STAGE_BYTES;
    const int j0 = GROUP * grp + 2 * c;
    if (cur.kind == K_UP) {
      if (grp == 0)
        for (int j = 0; j < USTAGES; ++j) mbar_wait(smem_u32(&full[slot0 + j]), ph);
      const unsigned char* rg = st + (g >> 2) * STAGE_BYTES + (g & 3) * (2 * H);
      const unsigned char* ru = rg + 2 * STAGE_BYTES;
#if QK_HOT8
      float4 sq;
      if (cur.q8) {
        sq = up_q8(st, bc, warp, g, c);
      } else {
#endif
      float acc[4][4] = {};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int kk = 512 * warp + 32 * i + 8 * c;
        mma_k32(acc[i & 3], lds128(rg + 2 * kk), lds128(ru + 2 * kk), bc[i]);
      }
#if QK_HOT8
      sq = sum_chains(acc);
      }
#endif
      float4* rb = red + (gs & 1) * CTHREADS;
#if QK_HOT8
      rb[warp * 32 + lane] = sq;
#else
      rb[warp * 32 + lane] = sum_chains(acc);
#endif
      named_sync(1, CTHREADS);
      if (warp == 0) {
        const float4 s = add4(add4(rb[lane], rb[32 + lane]), add4(rb[64 + lane], rb[96 + lane]));
        const int col = UP_COLS * cur.blk + g;
        if (j0 < cur.n)
          p.hbuf[T.list[cur.off + j0] * I + col] = __float2bfloat16_rn(s.x / (1.f + __expf(-s.x)) * s.z);
        if (j0 + 1 < cur.n)
          p.hbuf[T.list[cur.off + j0 + 1] * I + col] = __float2bfloat16_rn(s.y / (1.f + __expf(-s.y)) * s.w);
        if (grp == G - 1) {
          __syncwarp();
          if (lane == 0) publish(cur.e);
        }
      }
      ++gs;
      if (grp == G - 1) {
        __syncwarp();
        if (lane == 0)
          for (int j = 0; j < USTAGES; ++j) mbar_arrive(smem_u32(&empty[slot0 + j]));
      }
    } else {
      if (grp == 0) mbar_wait(smem_u32(&full[slot0 + warp]), ph);
      if (!rc) {
        if (lane == 0)
          while (ld_acquire(p.ctr + C_UPDONE + cur.e) < UP_BLKS) {
          }
        __syncwarp();
        const int j = GROUP * grp + g;
        const bool has = j < cur.n;
        const bf16* row = p.hbuf + static_cast<int64_t>(has ? T.list[cur.off + j] : 0) * I + 8 * c;
#pragma unroll
        for (int i = 0; i < 16; ++i) bc[i] = has ? ldcg128(row + 32 * i) : make_uint4(0, 0, 0, 0);
      }
      const unsigned char* sw = st + warp * STAGE_BYTES;
#if QK_HOT8
      float4 sq;
      if (cur.q8) {
        sq = down_q8(sw, bc, g, c);
      } else {
#endif
      float acc[4][4] = {};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int kk = 32 * i + 8 * c;
        mma_k32(acc[i & 3], lds128(sw + g * (2 * I) + 2 * kk), lds128(sw + (g + 8) * (2 * I) + 2 * kk), bc[i]);
      }
#if QK_HOT8
      sq = sum_chains(acc);
      }
#endif
#if QK_HOT8
      const float4 s = sq;
#else
      const float4 s = sum_chains(acc);
#endif
      const int r = DN_ROWS * cur.blk + 16 * warp + g;
      if (j0 < cur.n) {
        const int slot = T.list[cur.off + j0];
        const float w = T.wts[slot];
        p.ybuf[static_cast<int64_t>(slot) * H + r] = w * s.x;
        p.ybuf[static_cast<int64_t>(slot) * H + r + 8] = w * s.z;
      }
      if (j0 + 1 < cur.n) {
        const int slot = T.list[cur.off + j0 + 1];
        const float w = T.wts[slot];
        p.ybuf[static_cast<int64_t>(slot) * H + r] = w * s.y;
        p.ybuf[static_cast<int64_t>(slot) * H + r + 8] = w * s.w;
      }
      if (grp == G - 1) {
        __syncwarp();
        if (lane == 0) mbar_arrive_n(smem_u32(&empty[slot0 + warp]), CWARPS);
      }
    }
    cur = nx;
    grp = ngrp;
    us = nus;
    return false;
  };
  for (;;) {
    if (step(b0, r0, b1, r1)) break;
    if (step(b1, r1, b0, r0)) break;
  }
  combine(p, reinterpret_cast<float*>(ring), tid, warp, lane);
}
template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}
void check(const torch::Tensor& t, const char* name, at::ScalarType dtype, std::initializer_list<int64_t> shape) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.is_contiguous(), name, " must be a contiguous CUDA ",
              c10::toString(dtype), " tensor");
  TORCH_CHECK(t.sizes() == c10::IntArrayRef(shape), name, " has shape ", t.sizes(), ", expected ",
              c10::IntArrayRef(shape));
}
torch::Tensor moe_decode_impl(torch::Tensor x, torch::Tensor valid, torch::Tensor router_w, torch::Tensor gate_w,
                              torch::Tensor w13, torch::Tensor w2, torch::Tensor s13, torch::Tensor s2,
                              torch::Tensor logits, torch::Tensor counters, torch::Tensor ids, torch::Tensor wts,
                              torch::Tensor hbuf, torch::Tensor ybuf, c10::optional<torch::Tensor> hot,
#if QK_HOT8
                              const NextNorm* next, const c10::optional<torch::Tensor>& slot8,
                              const c10::optional<torch::Tensor>& copy8) {
#else
                              const NextNorm* next) {
#endif
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_T, "the decode MoE serves 1 to ", MAX_T, " rows, got ", M);
  check(x, "x", at::kBFloat16, {M, H});
  check(valid, "valid", at::kLong, {M});
  check(router_w, "router_w", at::kBFloat16, {E, H});
  check(gate_w, "gate_w", at::kBFloat16, {H});
  check(w13, "w13", at::kBFloat16, {E, 2 * I, H});
  check(w2, "w2", at::kBFloat16, {E, H, I});
  check(s13, "s13", at::kBFloat16, {2 * I, H});
  check(s2, "s2", at::kBFloat16, {H, I});
  check(logits, "logits", at::kFloat, {MAX_T, E});
  check(counters, "counters", at::kInt, {C_TOTAL});
  check(ids, "ids", at::kInt, {MAX_T * TOPK});
  check(wts, "wts", at::kFloat, {SLOTS});
  check(hbuf, "hbuf", at::kBFloat16, {SLOTS, I});
  check(ybuf, "ybuf", at::kFloat, {SLOTS, H});
  if (hot.has_value()) check(*hot, "hot", at::kInt, {HOT});
#if QK_HOT8
  TORCH_CHECK(slot8.has_value() == copy8.has_value(), "hot8: slot8 and copy8 are passed together or not at all");
  if (slot8.has_value()) {
    check(*slot8, "slot8", at::kInt, {NE});
    const torch::Tensor& c8 = *copy8;
    TORCH_CHECK(c8.is_cuda() && c8.scalar_type() == at::kByte && c8.dim() == 2 && c8.size(0) >= 1 &&
                    c8.size(1) == Q8_EXPERT && c8.stride(1) == 1 && (c8.size(0) == 1 || c8.stride(0) >= Q8_EXPERT) &&
                    c8.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(c8.data_ptr()) % 16 == 0,
                "hot8: copy8 must be a CUDA uint8 [slots, ", Q8_EXPERT,
                "] view: unit inner stride, disjoint 16-byte-aligned slot blocks");
    TORCH_CHECK(slot8->get_device() == x.get_device() && c8.get_device() == x.get_device(),
                "hot8: slot8 and copy8 must be on x's device");
  }
#endif
  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty({M, H}, x.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static int sms = 0;
  if (sms == 0) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe_ffn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FFN_SMEM));
  }
  const int m = static_cast<int>(M);
  const auto* xp = reinterpret_cast<const bf16*>(x.data_ptr());
  const int64_t* vp = valid.data_ptr<int64_t>();
  RouteArgs ra{xp, reinterpret_cast<const bf16*>(router_w.data_ptr()), reinterpret_cast<const bf16*>(gate_w.data_ptr()),
               reinterpret_cast<const bf16*>(s13.data_ptr()), reinterpret_cast<const bf16*>(s2.data_ptr()), vp, logits.data_ptr<float>(), counters.data_ptr<int>(), ids.data_ptr<int>(), wts.data_ptr<float>(), m};
  const NextNorm nn = next != nullptr ? *next : NextNorm{};
  {
    static bool cluster_ok = false;
    if (!cluster_ok) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(moe_route_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      cluster_ok = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(RT_CL, (m + RT_TOK - 1) / RT_TOK);
    cfg.blockDim = dim3(RT_THREADS);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = RT_CL;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_route_kernel, ra));
  }
  FfnArgs fa{xp, vp, reinterpret_cast<const bf16*>(w13.data_ptr()), reinterpret_cast<const bf16*>(w2.data_ptr()),
             reinterpret_cast<const bf16*>(s13.data_ptr()), reinterpret_cast<const bf16*>(s2.data_ptr()),
             ids.data_ptr<int>(), wts.data_ptr<float>(), counters.data_ptr<int>(),
             reinterpret_cast<bf16*>(hbuf.data_ptr()), ybuf.data_ptr<float>(),
             hot.has_value() ? hot->data_ptr<int>() : nullptr, m, reinterpret_cast<bf16*>(out.data_ptr()), nn};
#if QK_HOT8
  if (slot8.has_value()) {
    fa.slot8 = slot8->data_ptr<int>();
    fa.q8 = reinterpret_cast<const unsigned char*>(copy8->data_ptr());
    fa.q8_stride = copy8->size(0) > 1 ? copy8->stride(0) : 0;
    fa.q8_slots = static_cast<int>(copy8->size(0));
  }
#endif
  launch_pdl(moe_ffn_kernel, dim3(sms), dim3(FFN_THREADS), FFN_SMEM, stream, fa);
  return out;
}
bool overlaps(const torch::Tensor& a, const torch::Tensor& b) {
  const auto* a0 = static_cast<const char*>(a.data_ptr());
  const auto* b0 = static_cast<const char*>(b.data_ptr());
  return a0 < b0 + b.nbytes() && b0 < a0 + a.nbytes();
}
}
torch::Tensor moe_decode(torch::Tensor x, torch::Tensor valid, torch::Tensor router_w, torch::Tensor gate_w,
                         torch::Tensor w13, torch::Tensor w2, torch::Tensor s13, torch::Tensor s2, torch::Tensor logits,
                         torch::Tensor counters, torch::Tensor ids, torch::Tensor wts, torch::Tensor hbuf,
#if QK_HOT8
                         torch::Tensor ybuf, c10::optional<torch::Tensor> hot, c10::optional<torch::Tensor> slot8,
                         c10::optional<torch::Tensor> copy8) {
  return moe_decode_impl(x, valid, router_w, gate_w, w13, w2, s13, s2, logits, counters, ids, wts, hbuf, ybuf, hot,
                         nullptr, slot8, copy8);
#else
                         torch::Tensor ybuf, c10::optional<torch::Tensor> hot) {
  return moe_decode_impl(x, valid, router_w, gate_w, w13, w2, s13, s2, logits, counters, ids, wts, hbuf, ybuf, hot,
                         nullptr);
#endif
}
torch::Tensor moe_decode_norm(torch::Tensor x, torch::Tensor valid, torch::Tensor router_w, torch::Tensor gate_w,
                              torch::Tensor w13, torch::Tensor w2, torch::Tensor s13, torch::Tensor s2,
                              torch::Tensor logits, torch::Tensor counters, torch::Tensor ids, torch::Tensor wts,
                              torch::Tensor hbuf, torch::Tensor ybuf, c10::optional<torch::Tensor> hot,
                              torch::Tensor residual, torch::Tensor norm_w, double eps, torch::Tensor y_next,
#if QK_HOT8
                              torch::Tensor r_next, c10::optional<torch::Tensor> slot8,
                              c10::optional<torch::Tensor> copy8) {
#else
                              torch::Tensor r_next) {
#endif
  const int64_t M = x.size(0);
  check(residual, "residual", at::kBFloat16, {M, H});
  check(norm_w, "norm_w", at::kBFloat16, {H});
  check(y_next, "y_next", at::kBFloat16, {M, H});
  check(r_next, "r_next", at::kBFloat16, {M, H});
  TORCH_CHECK(!overlaps(y_next, residual) && !overlaps(r_next, residual) && !overlaps(y_next, r_next) &&
                  !overlaps(y_next, x) && !overlaps(r_next, x),
              "moe_decode_norm: the side buffers must not overlap residual, x or each other");
  const NextNorm n{reinterpret_cast<const bf16*>(residual.data_ptr()), reinterpret_cast<const bf16*>(norm_w.data_ptr()),
                   static_cast<float>(eps), reinterpret_cast<bf16*>(y_next.data_ptr()),
                   reinterpret_cast<bf16*>(r_next.data_ptr())};
#if QK_HOT8
  return moe_decode_impl(x, valid, router_w, gate_w, w13, w2, s13, s2, logits, counters, ids, wts, hbuf, ybuf, hot, &n,
                         slot8, copy8);
#else
  return moe_decode_impl(x, valid, router_w, gate_w, w13, w2, s13, s2, logits, counters, ids, wts, hbuf, ybuf, hot, &n);
#endif
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_decode", &moe_decode, "Qwen3.6-35B-A3B decode-width MoE block (router, experts, shared expert, combine)");
  m.def("moe_decode_norm", &moe_decode_norm,
        "moe_decode plus the next layer's Gemma fused add-RMSNorm of (out, residual) into side buffers (kcombnorm)");
  m.attr("COUNTERS") = C_TOTAL;
  m.attr("MAX_T") = MAX_T;
  m.attr("SLOTS") = SLOTS;
  m.attr("HOT") = HOT;
#if QK_HOT8
  m.attr("HOT8") = QK_HOT8;
  m.attr("Q8_EXPERT") = Q8_EXPERT;
  m.attr("HOT8_MAX_N") = QK_HOT8_MAX_N;
#endif
}
