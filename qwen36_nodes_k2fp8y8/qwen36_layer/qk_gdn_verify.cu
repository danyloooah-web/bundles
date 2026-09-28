// Fused GDN target-verify block for Qwen3.6-35B-A3B (TP1, MTP draft window 4) on sm_90a.
//
// One launch replaces stock's verify sequence for a recurrent layer: the conv update
// (with its per-draft windows and the conv-state roll), the q/k/v split, the ReplaySSM
// verify recurrence (fp32 checkpoint plus the circular (d, k, g) ring, and this window's
// ring writes) and the gated RMSNorm. Its input is the winner bundle's in_proj output;
// its output feeds the winner's deferred out_proj unchanged.
//
// Persistent: one CTA per SM walks units (request, key head). A unit owns both value heads of
// its key head, so the q/k conv, the conv-state roll and the key-ring append stay CTA-local (a
// CTA per value head and request left HBM idle for most of each CTA's life). A producer warp
// streams the fp32 checkpoint (128 KB per unit) with TMA through four 32 KB stages. A prefetch
// warp copies each unit's small operands (q/k/v rows, z, the a/b bytes of its heads, the conv
// state and only the committed rows of the rings) into one of three input blocks a unit ahead
// with 16-byte cp.async, so the eight consumer warps never wait on a dependent global load.
// Every dot product (checkpoint rows, ring keys and the window's own keys against [q | k]) runs
// on TF32 tensor cores, stock's precision. At 32 requests a layer moves ~83 MB (67 MB of
// checkpoint, ~7 MB of operands, ~9 MB of state and ring writes): HBM sets its time. Dependents
// are released only at exit: the projection after this kernel waits before loading anything,
// so an early launch only parked its CTAs on these SMs (slower in the whole-layer graph).

#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int K = 128, V = 128, HV = 32, HK = 16, NT = 4, L = 16;
constexpr int QKV = 8192, QDIM = 2048, ZDIM = HV * V;
constexpr int CONSUMERS = 256, THREADS = CONSUMERS + 64;  // eight consumer warps, the producer, the prefetcher
// A fill is 64 value rows (32 KB); four stages hold one unit. One fill in flight per SM: more
// only queued every other read of the SM behind the stream (unit inputs took 8-12 us to land
// at 32 requests with four in flight; 34.5 vs 37 us at 32 requests for one vs four).
constexpr int STAGES = 4, INBUF = 3, STAGE_ROWS = 64, BOX_COLS = 32;
// A unit's fill j always lands in stage (n0 + j) % STAGES and is read by warp group j & 1;
// an even stage count pins every stage to one group. With an odd count a group can wait on
// a stage's next fill while the other group's fill is still in flight, and try_wait.parity
// then reads the preceding phase as complete (reproduced: corrupt ring, launch failure).
static_assert(STAGES % 2 == 0, "each stage must belong to one consumer warp group");
constexpr int MAXU = 64;  // units per CTA the index table holds (B <= 528 on 132 SMs)
// King-stack variant kgdnpf (rental 2026-09-25): when the grid leaves SMs idle, up to PF_CTAS extra
// prefetch-only CTAs pull the first PF_BYTES of the layer's out_proj weight into L2 while the verify
// runs; the projection after this kernel then reads them from L2 (a read-only cache hint, no result change).
constexpr int PF_CTAS = 8;
constexpr int64_t PF_CHUNK = 16384, PF_BYTES = 256 * PF_CHUNK;  // 4 MiB
constexpr uint32_t BOX_BYTES = STAGE_ROWS * BOX_COLS * sizeof(float);  // 64 rows x 128 B
constexpr uint32_t STAGE_BYTES = 4 * BOX_BYTES;                       // 64 rows x 128 columns
constexpr uint32_t XBOX_BYTES = (L + NT) * BOX_COLS * sizeof(float);    // ring + window keys: 20 rows x 128 B
constexpr int QK_STRIDE = K + 4;  // padded rows: conflict-free B-fragment loads
constexpr int UCH = 2 * K + 2 * V;  // a unit's conv channels: q | k | both value heads' v
constexpr unsigned FULL = 0xffffffffu;

typedef __nv_bfloat16 bf16;

// One unit's operands, bulk-copied by the prefetch warp; the rings keep physical slot order.
struct __align__(16) UnitIn {
  bf16 mixed[NT][UCH];
  bf16 kring[L][K];     // the key head's ring keys (high parts)
  bf16 dring[2][L][V];  // both value heads' d entries (high parts)
  bf16 cst[3 * UCH];    // conv state of the channels above, three taps each
  bf16 z[NT][2 * V];
  bf16 a[NT][HV];
  bf16 b[NT][HV];
  float gring[2][L];
  long long slot, bos, replay, wslot;  // stored by the prefetch thread
  int wp, cb;
};
constexpr uint32_t IN_BYTES = (NT * UCH + L * K + 2 * L * V + 3 * UCH + NT * 2 * V + 2 * NT * HV) * sizeof(bf16) +
                              2 * L * sizeof(float);
static_assert(sizeof(UnitIn) >= IN_BYTES + 40, "UnitIn layout");
constexpr uint32_t DYN_BYTES = STAGES * STAGE_BYTES + 4 * XBOX_BYTES + INBUF * sizeof(UnitIn) + 1024;  // + alignment

struct Args {
  const bf16* mixed;
  const bf16* z;
  const bf16* a;
  const bf16* b;
  bf16* conv_state;
  int64_t cs_slot;
  const bf16* conv_w;
  bf16* window;
  int64_t win_seq;
  const float* A_log;
  const void* dt_bias;
  int dt_f32;
  bf16* d_ring;
  bf16* dlo_ring;
  int64_t d_slot, dlo_slot;
  bf16* k_ring;
  bf16* klo_ring;
  int64_t k_slot, klo_slot;
  float* g_ring;
  int64_t g_slot;
  const int32_t* qsl;
  const void* state_idx;
  int state_i64;
  const void* replay_idx;
  int replay_i64;
  const void* win_idx;
  int win_i64;
  const int32_t* write_pos;
  const int32_t* cache_base;
  const bf16* norm_w;
  float eps, scale;
  bf16* out;
  int units;  // requests x key heads
  int ctas;   // kgdnpf: compute CTAs (the unit stride); CTAs from here on only prefetch
  const unsigned char* pf;  // kgdnpf: out_proj weight bytes to prefetch into L2
  int64_t pf_chunks;        // kgdnpf: PF_CHUNK-byte chunks to prefetch (0: no extra CTAs)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int64_t ld_idx(const void* p, int i64, int i) {
  return i64 ? reinterpret_cast<const int64_t*>(p)[i] : static_cast<int64_t>(reinterpret_cast<const int32_t*>(p)[i]);
}
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 tobf(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// The consumer warps' own barrier; the producer and prefetch warps never join it.
__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS) : "memory"); }

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(bar),
               "r"(parity)
               : "memory");
}
// One 16-byte cp.async (L2 only); cp_arrive makes the barrier track the thread's pending ones.
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
// A warp copies bytes (a multiple of 16) in 16-byte pieces, lane-strided.
__device__ __forceinline__ void warp_copy(void* dst, const void* src, uint32_t bytes, int lane) {
  const uint32_t d = smem_u32(dst);
  const char* s = static_cast<const char*>(src);
  for (uint32_t off = 16 * lane; off < bytes; off += 32 * 16) cp16(d + off, s + off);
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(
          dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void sts4(uint32_t addr, float4 x) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(x.x), "f"(x.y), "f"(x.z), "f"(x.w) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Byte offset of fp32 element (row, col) in a set of 128B-swizzled boxes of 32 columns
// (the layout TMA's SWIZZLE_128B writes), boxes box_bytes apart.
__device__ __forceinline__ uint32_t swz(int row, int col, uint32_t box_bytes) {
  return (col >> 5) * box_bytes + row * 128 + ((((col & 31) >> 2) ^ (row & 7)) << 4) + ((col & 3) << 2);
}


// One 16-row tile of A (swizzled fp32 rows) against the eight [q | k] columns, K = 128:
// sixteen m16n8k8 TF32 MMAs.
__device__ __forceinline__ void tile_dots(uint32_t base, uint32_t box_bytes, int m0, const uint32_t (&bq)[32],
                                          int lane, float (&acc)[4]) {
  // Two accumulator chains halve the dependent MMA sequence.
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
  const int mi = lane >> 3, ri = lane & 7;
  const int row = m0 + ri + 8 * (mi & 1);
#pragma unroll
  for (int kk = 0; kk < K / 8; ++kk) {
    const int chunk = 2 * (kk & 3) + (mi >> 1);
    uint32_t a[4];
    ldsm_x4(base + (kk >> 2) * box_bytes + row * 128 + ((chunk ^ (row & 7)) << 4), a);
    if (kk & 1)
      mma_tf32(acc2, a, bq[2 * kk], bq[2 * kk + 1]);
    else
      mma_tf32(acc, a, bq[2 * kk], bq[2 * kk + 1]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] += acc2[i];
}

// Four per-token partial sums reduced across the warp; the total returned is for token
// ((lane >> 4) & 1) * 2 + ((lane >> 3) & 1).
__device__ __forceinline__ float sum4(float (&s)[NT], int lane) {
  bool hi = lane & 16;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const float send = hi ? s[m] : s[m + 2], keep = hi ? s[m + 2] : s[m];
    s[m] = keep + __shfl_xor_sync(FULL, send, 16);
  }
  hi = lane & 8;
  {
    const float send = hi ? s[0] : s[1], keep = hi ? s[1] : s[0];
    s[0] = keep + __shfl_xor_sync(FULL, send, 8);
  }
  s[0] += __shfl_xor_sync(FULL, s[0], 4);
  s[0] += __shfl_xor_sync(FULL, s[0], 2);
  s[0] += __shfl_xor_sync(FULL, s[0], 1);
  return s[0];
}

// bf16 lanes of a packed register pair.
__device__ __forceinline__ bf16 lo16(uint32_t w) { return __ushort_as_bfloat16(static_cast<unsigned short>(w & 0xffffu)); }
__device__ __forceinline__ bf16 hi16(uint32_t w) { return __ushort_as_bfloat16(static_cast<unsigned short>(w >> 16)); }
__device__ __forceinline__ uint32_t pack2(bf16 a, bf16 b) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(a)) | (static_cast<uint32_t>(__bfloat16_as_ushort(b)) << 16);
}

// Stock's causal conv arithmetic for one channel and token t: each tap's product rounded to
// bf16, oldest tap first in fp32, SiLU, then the bf16 store the split kernel reads back.
template <int T>
__device__ __forceinline__ float conv_tap(const bf16 (&win)[NT + 3], bf16 w0, bf16 w1, bf16 w2, bf16 w3) {
  float acc = bf(__hmul(win[T], w0));
  acc += bf(__hmul(win[T + 1], w1));
  acc += bf(__hmul(win[T + 2], w2));
  acc += bf(__hmul(win[T + 3], w3));
  return bf(tobf(acc / (1.f + __expf(-acc))));
}

// Four consecutive channels: windows [s0 s1 s2 x0 x1 x2 x3] from the conv state (3 taps per
// channel, channel-major) and the four draft tokens, then token t's conv output.
struct Conv4 {
  bf16 w[4][NT + 3];
  __device__ __forceinline__ void load(const uint2 (&st)[3], const uint2 (&x)[NT]) {
    const uint32_t s32[6] = {st[0].x, st[0].y, st[1].x, st[1].y, st[2].x, st[2].y};
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int k3 = 0; k3 < 3; ++k3) {
        const int e = c * 3 + k3;
        w[c][k3] = (e & 1) ? hi16(s32[e >> 1]) : lo16(s32[e >> 1]);
      }
#pragma unroll
    for (int s = 0; s < NT; ++s) {
      w[0][3 + s] = lo16(x[s].x);
      w[1][3 + s] = hi16(x[s].x);
      w[2][3 + s] = lo16(x[s].y);
      w[3][3 + s] = hi16(x[s].y);
    }
  }
  template <int T>
  __device__ __forceinline__ void conv(const uint4 (&wt)[2], float (&u)[4]) const {
    const uint32_t w32[8] = {wt[0].x, wt[0].y, wt[0].z, wt[0].w, wt[1].x, wt[1].y, wt[1].z, wt[1].w};
#pragma unroll
    for (int c = 0; c < 4; ++c)
      u[c] = conv_tap<T>(w[c], lo16(w32[2 * c]), hi16(w32[2 * c]), lo16(w32[2 * c + 1]), hi16(w32[2 * c + 1]));
  }
  // The pool's deduplicated window record [s1 s2 x0 x1 x2 x3] and the rolled state [x1 x2 x3].
  __device__ __forceinline__ void store(bf16* win_row, bf16* state) const {
#pragma unroll
    for (int c = 0; c < 4; c += 2) {
      *reinterpret_cast<uint2*>(win_row + 6 * c) = make_uint2(pack2(w[c][1], w[c][2]), pack2(w[c][3], w[c][4]));
      *reinterpret_cast<uint2*>(win_row + 6 * c + 4) =
          make_uint2(pack2(w[c][5], w[c][6]), pack2(w[c + 1][1], w[c + 1][2]));
      *reinterpret_cast<uint2*>(win_row + 6 * c + 8) =
          make_uint2(pack2(w[c + 1][3], w[c + 1][4]), pack2(w[c + 1][5], w[c + 1][6]));
    }
    *reinterpret_cast<uint2*>(state) = make_uint2(pack2(w[0][4], w[0][5]), pack2(w[0][6], w[1][4]));
    *reinterpret_cast<uint2*>(state + 4) = make_uint2(pack2(w[1][5], w[1][6]), pack2(w[2][4], w[2][5]));
    *reinterpret_cast<uint2*>(state + 8) = make_uint2(pack2(w[2][6], w[3][4]), pack2(w[3][5], w[3][6]));
  }
};

__global__ void __launch_bounds__(THREADS, 1) gdn_verify_kernel(const Args p, const __grid_constant__ CUtensorMap h0map) {
  extern __shared__ __align__(16) unsigned char dyn[];
  __shared__ __align__(16) float qk[8][QK_STRIDE];  // rows 0-3 scaled unit q, 4-7 unit k
  __shared__ __align__(16) float vsh[2][NT][V];
  __shared__ __align__(16) float hw[2][V][8];       // checkpoint rows . [q | k], per value head
  __shared__ __align__(16) float xr[L + NT][8];     // rows 0-15 ring keys, 16-19 window keys . [q | k]
  __shared__ float decay[2][L];
  __shared__ float gate[2][4][NT];                  // g, beta, cumulative g, exp(cumulative g)
  __shared__ float red[8][NT];
  __shared__ float tdecay[2];
  __shared__ long long tab[MAXU][4];                // this CTA's units: slot, bos, replay, window slot
  __shared__ int tabc[MAXU][2];                     // and ring cursors: write_pos, cache_base
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES], in_full[INBUF], in_empty[INBUF], tab_ready, ops_issued;

  const uint32_t sbase = (smem_u32(dyn) + 1023u) & ~1023u;  // TMA 128B swizzle wants 1024-B alignment
  const uint32_t xbase = sbase + STAGES * STAGE_BYTES;       // ring and window keys, rows 0-19
  UnitIn* const inb =
      reinterpret_cast<UnitIn*>(dyn + (sbase - smem_u32(dyn)) + STAGES * STAGE_BYTES + 4 * XBOX_BYTES);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (static_cast<int>(blockIdx.x) >= p.ctas) {  // kgdnpf: a prefetch-only CTA (reads nothing, writes nothing)
    if (warp == 0) {
      const int64_t n = static_cast<int64_t>(gridDim.x) - p.ctas;
      for (int64_t c = (static_cast<int64_t>(blockIdx.x) - p.ctas) + n * lane; c < p.pf_chunks; c += n * 32)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.pf + c * PF_CHUNK),
                     "r"(static_cast<uint32_t>(PF_CHUNK)) : "memory");
    }
    return;
  }
  const int units = (p.units - 1 - static_cast<int>(blockIdx.x)) / p.ctas + 1;  // this CTA's (kgdnpf: p.ctas)
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(smem_u32(&full[s]), 1);
      mbar_init(smem_u32(&empty[s]), 4);  // the four consumer warps that read a stage
    }
#pragma unroll
    for (int i = 0; i < INBUF; ++i) {
      mbar_init(smem_u32(&in_full[i]), 33);  // each prefetch lane's copies, then the index store
      mbar_init(smem_u32(&in_empty[i]), 1);
    }
    mbar_init(smem_u32(&tab_ready), 1);
    mbar_init(smem_u32(&ops_issued), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == CONSUMERS / 32) {
    // Producer: stage j of a unit holds value rows (j & 1) * 64 .. + 63 of value head
    // 2 * hk + (j >> 1). Fill n goes to stage n % STAGES; the consumers release it.
    if (lane == 0) {
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      int n = 0;
      bool dep = false;
      for (int k = 0; k < units; ++k) {
        const int u = blockIdx.x + k * p.ctas;  // kgdnpf: stride over compute CTAs only
        int64_t slot;
        if (k == 0) {
          slot = ld_idx(p.state_idx, p.state_i64, u / HK);
        } else {
          if (k == 1) mbar_wait(smem_u32(&tab_ready), 0);
          slot = tab[k][0];
        }
        if (slot < 0) continue;
        // The checkpoint copy is issued only after the dependency wait: the projection before
        // this kernel streams its weights at the bandwidth limit, so copying during it only
        // slowed it down (measured in the whole-layer graph: 3.3 us/layer at 16 rows).
        if (!dep) {
          griddep_wait();
          dep = true;
        }
        const int row0 = static_cast<int>((slot * HV + 2 * (u % HK)) * V);
        for (int j = 0; j < 4; ++j, ++n) {
          const int s = n % STAGES;
          if (n >= STAGES) mbar_wait(smem_u32(&empty[s]), ((n / STAGES) - 1) & 1);
          if (units == 1) {
            // A CTA with one unit has no later unit to delay: after the first fill, its other three
            // go together, but only once the unit's operands are queued ahead of them (queued
            // behind 96 KB of checkpoint they landed 3 us late, measured).
            if (n == 1) mbar_wait(smem_u32(&ops_issued), 0);
          } else if (n >= 1) {
            mbar_wait(smem_u32(&full[(n - 1) % STAGES]), ((n - 1) / STAGES) & 1);  // one in flight
          }
          mbar_expect(smem_u32(&full[s]), STAGE_BYTES);
#pragma unroll
          for (int c = 0; c < 4; ++c)
            tma_2d(sbase + s * STAGE_BYTES + c * BOX_BYTES, &h0map, smem_u32(&full[s]), c * BOX_COLS,
                   row0 + j * STAGE_ROWS, pol);
        }
      }
    }
    return;
  }

  if (warp == CONSUMERS / 32 + 1) {
    // Prefetcher: the k-th unit of this CTA goes to input block k % INBUF, up to INBUF - 1 units
    // ahead of the consumers. The whole warp copies it in 16-byte cp.async pieces (TMA bulk
    // copies measured the same). in_full completes on the 32 lanes' copies and on lane 0's index
    // store. Every unit's indices and ring cursors are loaded once up front.
    bool dep = false;
    auto copies = [&](int k, int u, int64_t slot, int64_t bos, int64_t replay, int wp, int cb) {
      if (slot < 0) return;
      const int hk = u % HK;
      UnitIn& in = inb[k % INBUF];
      // Operands the preceding projection does not write are copied before the dependency wait.
      const bf16* cs = p.conv_state + slot * p.cs_slot;
      warp_copy(&in.cst[0], cs + static_cast<int64_t>(hk) * K * 3, K * 3 * sizeof(bf16), lane);
      warp_copy(&in.cst[3 * K], cs + static_cast<int64_t>(QDIM + hk * K) * 3, K * 3 * sizeof(bf16), lane);
      warp_copy(&in.cst[6 * K], cs + static_cast<int64_t>(2 * QDIM + 2 * hk * V) * 3, 2 * V * 3 * sizeof(bf16), lane);
      {
        // Only the committed history of the rings (logical entries below wp): the physical runs
        // [cb, cb + n0) and [0, n1). The consumers never read the other rows.
        const int c0 = cb & (L - 1), nv = min(max(wp, 0), L), n0 = min(nv, L - c0), n1 = nv - n0;
        const bf16* kr = p.k_ring + replay * p.k_slot + static_cast<int64_t>(hk) * L * K;
        warp_copy(&in.kring[c0][0], kr + c0 * K, n0 * K * sizeof(bf16), lane);
        warp_copy(&in.kring[0][0], kr, n1 * K * sizeof(bf16), lane);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const bf16* dr = p.d_ring + replay * p.d_slot + static_cast<int64_t>(2 * hk + h) * L * V;
          warp_copy(&in.dring[h][c0][0], dr + c0 * V, n0 * V * sizeof(bf16), lane);
          warp_copy(&in.dring[h][0][0], dr, n1 * V * sizeof(bf16), lane);
        }
      }
      warp_copy(&in.gring[0][0], p.g_ring + replay * p.g_slot + 2 * hk * L, 2 * L * sizeof(float), lane);
      if (!dep) {
        griddep_wait();
        dep = true;
      }
#pragma unroll
      for (int s = 0; s < NT; ++s) {
        const bf16* row = p.mixed + (bos + s) * QKV;
        warp_copy(&in.mixed[s][0], row + hk * K, K * sizeof(bf16), lane);
        warp_copy(&in.mixed[s][K], row + QDIM + hk * K, K * sizeof(bf16), lane);
        warp_copy(&in.mixed[s][2 * K], row + 2 * QDIM + 2 * hk * V, 2 * V * sizeof(bf16), lane);
        warp_copy(&in.z[s][0], p.z + (bos + s) * ZDIM + 2 * hk * V, 2 * V * sizeof(bf16), lane);
      }
      if (lane < 2 * NT) {  // the 16 bytes of a and b rows that hold this unit's two heads
        const int s = lane % NT, c = (2 * hk) & ~7;
        const bf16* src = (lane < NT ? p.a : p.b) + (bos + s) * HV + c;
        cp16(smem_u32(lane < NT ? &in.a[s][c] : &in.b[s][c]), src);
      }
    };
    for (int i0 = 0; i0 < units; i0 += 32) {
      const int i = i0 + lane;
      int64_t slot = -1, bos = 0, replay = 0, wslot = 0;
      int wp = 0, cb = 0;
      if (i < units) {
        const int r = (blockIdx.x + i * p.ctas) / HK;  // kgdnpf: p.ctas
        slot = ld_idx(p.state_idx, p.state_i64, r);
        bos = p.qsl[r];
        replay = ld_idx(p.replay_idx, p.replay_i64, r);
        wslot = ld_idx(p.win_idx, p.win_i64, r);
        if (slot >= 0) {
          wp = p.write_pos[replay];
          cb = p.cache_base[replay];
        }
      }
      if (i0 == 0) {  // the first unit's copies wait for nothing else
        copies(0, blockIdx.x, __shfl_sync(FULL, slot, 0), __shfl_sync(FULL, bos, 0), __shfl_sync(FULL, replay, 0),
               __shfl_sync(FULL, wp, 0), __shfl_sync(FULL, cb, 0));
      }
      if (i < units) {
        tab[i][0] = slot;
        tab[i][1] = bos;
        tab[i][2] = replay;
        tab[i][3] = wslot;
        tabc[i][0] = wp;
        tabc[i][1] = cb;
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(&tab_ready));
    for (int k = 0; k < units; ++k) {
      const int b = k % INBUF;
      if (k >= INBUF) mbar_wait(smem_u32(&in_empty[b]), ((k / INBUF) - 1) & 1);
      // One unit's copies in flight at a time: they queue behind the checkpoint stream anyway.
      if (k > 0) mbar_wait(smem_u32(&in_full[(k - 1) % INBUF]), ((k - 1) / INBUF) & 1);
      // kgdnpf: p.ctas
      if (k > 0) copies(k, blockIdx.x + k * p.ctas, tab[k][0], tab[k][1], tab[k][2], tabc[k][0], tabc[k][1]);
      cp_arrive(smem_u32(&in_full[b]));
      if (k == 0 && lane == 0) mbar_arrive(smem_u32(&ops_issued));
      if (lane == 0) {
        UnitIn& in = inb[b];
        in.slot = tab[k][0];
        in.bos = tab[k][1];
        in.replay = tab[k][2];
        in.wslot = tab[k][3];
        in.wp = tabc[k][0];
        in.cb = tabc[k][1];
        mbar_arrive(smem_u32(&in_full[b]));
      }
    }
    return;
  }

  // Consumers. Conv: warp (vec, t) convolves token t of four q (vec 0) or k (vec 1) channels
  // per lane, so each vector's L2 norm is one warp reduction, and the same token of four v
  // channels per lane of value head vec. Epilogue: thread (h = vec, v = tid & 127).
  const int vec = warp >> 2, t = warp & 3, h = vec, v = tid & 127;
  const float nw = bf(p.norm_w[v]);
  // Warp 2 lane i holds value head i's gate constants for the whole launch.
  float dtb_i = 0.f, alog_i = 0.f;
  if (warp == 2) {
    dtb_i = p.dt_f32 ? reinterpret_cast<const float*>(p.dt_bias)[lane] : bf(reinterpret_cast<const bf16*>(p.dt_bias)[lane]);
    alog_i = p.A_log[lane];
  }
  // A unit's conv weights load one unit ahead: issued at the unit's own start they came from
  // HBM behind the checkpoint stream and landed after its inputs.
  uint4 wqn[2], wvn[2];
  auto load_w = [&](int u) {
    const uint4* q = reinterpret_cast<const uint4*>(p.conv_w + static_cast<int64_t>(vec * QDIM + (u % HK) * K + 4 * lane) * 4);
    const uint4* w = reinterpret_cast<const uint4*>(p.conv_w + static_cast<int64_t>(2 * QDIM + (2 * (u % HK) + h) * V + 4 * lane) * 4);
    wqn[0] = q[0];
    wqn[1] = q[1];
    wvn[0] = w[0];
    wvn[1] = w[1];
  };
  load_w(blockIdx.x);
  int n0 = 0;  // the producer's fill count at this unit's first stage
  bool dep = false;
  for (int k = 0; k < units; ++k) {
    // kgdnpf: p.ctas (here and in load_w below)
    const int u = blockIdx.x + k * p.ctas, hk = u % HK, bsel = k % INBUF, hv = 2 * hk + h;
    const UnitIn& in = inb[bsel];
// Trace slots 0-7: start, inputs, barrier A, stages 0 and 2 (3, 4), barriers B and C, end.
    const int cq = vec * QDIM + hk * K + 4 * lane;  // first of four q/k channels
    const int cv = 2 * QDIM + hv * V + 4 * lane;    // first of four v channels
    const uint4 wq[2] = {wqn[0], wqn[1]}, wv[2] = {wvn[0], wvn[1]};
    if (k + 1 < units) load_w(u + p.ctas);
    const int ghv = (2 * hk + (lane >> 2)) & (HV - 1);  // gate lanes 0-7 of warp 2: head (lane >> 2), token lane & 3
    mbar_wait(smem_u32(&in_full[bsel]), (k / INBUF) & 1);
    const int64_t slot = in.slot, bos = in.bos;
    if (!dep) {  // before this kernel's first global write
      griddep_wait();
      dep = true;
    }
    if (slot < 0) {
      // A padded graph row: stock's recurrence writes zeros, which the norm keeps zero.
#pragma unroll
      for (int s = 0; s < NT; ++s) p.out[(bos + s) * ZDIM + hv * V + v] = tobf(0.f);
      consumer_sync();
      if (tid == 0) mbar_arrive(smem_u32(&in_empty[bsel]));
      continue;
    }
    const int64_t replay = in.replay, wslot = in.wslot;
    // The commit folds a nearly full ring and clears its flush flag before the next
    // verify (L = 16 >= 2 x the window), so a verify row always appends to its ring.
    const int wp = in.wp, cb = in.cb;

    uint2 xq[NT], xv[NT], sq[3], sv[3];
#pragma unroll
    for (int s = 0; s < NT; ++s) {
      if (s <= t || t == NT - 1) {
        xq[s] = *reinterpret_cast<const uint2*>(&in.mixed[s][vec * K + 4 * lane]);
        xv[s] = *reinterpret_cast<const uint2*>(&in.mixed[s][2 * K + vec * V + 4 * lane]);
      } else {
        xq[s] = make_uint2(0u, 0u);
        xv[s] = make_uint2(0u, 0u);
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      sq[i] = *reinterpret_cast<const uint2*>(&in.cst[vec * 3 * K + 12 * lane + 4 * i]);
      sv[i] = *reinterpret_cast<const uint2*>(&in.cst[6 * K + vec * 3 * V + 12 * lane + 4 * i]);
    }
    Conv4 wiq, wiv;
    wiq.load(sq, xq);
    wiv.load(sv, xv);
    float uq[4], uv[4];
    // The token is warp-uniform. One switch around all eight channels lets their dependent
    // chains interleave; a switch per channel serialized them (~1900 cycles per unit).
    switch (t) {
      case 0: wiq.conv<0>(wq, uq); wiv.conv<0>(wv, uv); break;
      case 1: wiq.conv<1>(wq, uq); wiv.conv<1>(wv, uv); break;
      case 2: wiq.conv<2>(wq, uq); wiv.conv<2>(wv, uv); break;
      default: wiq.conv<3>(wq, uq); wiv.conv<3>(wv, uv); break;
    }
    {
      float n = uq[0] * uq[0] + uq[1] * uq[1] + uq[2] * uq[2] + uq[3] * uq[3];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(FULL, n, o);
      const float r = (1.f / sqrtf(n + 1e-6f)) * (vec == 0 ? p.scale : 1.f);
      const float4 unit = make_float4(uq[0] * r, uq[1] * r, uq[2] * r, uq[3] * r);
      *reinterpret_cast<float4*>(&qk[4 * vec + t][4 * lane]) = unit;
      if (vec == 1) {
        // This window's unit keys are rows 16-19 of the key tile, and join the key ring as
        // bf16 high and low parts (slots past the history: disjoint from the ring reads).
        sts4(xbase + swz(L + t, 4 * lane, XBOX_BYTES), unit);
        if (wp + t < L) {
          const int64_t ph = static_cast<int64_t>(hk) * L * K + ((cb + wp + t) & (L - 1)) * K + 4 * lane;
          const bf16 h0 = tobf(unit.x), h1 = tobf(unit.y), h2 = tobf(unit.z), h3 = tobf(unit.w);
          *reinterpret_cast<uint2*>(p.k_ring + replay * p.k_slot + ph) = make_uint2(pack2(h0, h1), pack2(h2, h3));
          *reinterpret_cast<uint2*>(p.klo_ring + replay * p.klo_slot + ph) =
              make_uint2(pack2(tobf(unit.x - bf(h0)), tobf(unit.y - bf(h1))),
                         pack2(tobf(unit.z - bf(h2)), tobf(unit.w - bf(h3))));
        }
      }
      *reinterpret_cast<float4*>(&vsh[vec][t][4 * lane]) = make_float4(uv[0], uv[1], uv[2], uv[3]);
    }
    {
      // Committed ring keys (zero past the history) are rows 0-15 of the key tile.
      const int kr = tid >> 4, kc = (tid & 15) * 8;
      uint4 raw = make_uint4(0u, 0u, 0u, 0u);
      if (kr < wp) raw = *reinterpret_cast<const uint4*>(&in.kring[(cb + kr) & (L - 1)][kc]);
      sts4(xbase + swz(kr, kc, XBOX_BYTES), make_float4(bf(lo16(raw.x)), bf(hi16(raw.x)), bf(lo16(raw.y)), bf(hi16(raw.y))));
      sts4(xbase + swz(kr, kc + 4, XBOX_BYTES),
           make_float4(bf(lo16(raw.z)), bf(hi16(raw.z)), bf(lo16(raw.w)), bf(hi16(raw.w))));
    }
    consumer_sync();
    // Every warp has read the old conv state: roll it and write the draft windows.
    if (t == NT - 1) {
      bf16* wrow = p.window + wslot * p.win_seq;
      bf16* cs = p.conv_state + slot * p.cs_slot;
      wiq.store(wrow + static_cast<int64_t>(cq) * 6, cs + static_cast<int64_t>(cq) * 3);
      wiv.store(wrow + static_cast<int64_t>(cv) * 6, cs + static_cast<int64_t>(cv) * 3);
    }
    // Gates and decay are read only by the epilogue: two warps compute them while the stage MMAs wait.
    const float dtb = __shfl_sync(FULL, dtb_i, ghv), alog = __shfl_sync(FULL, alog_i, ghv);
    if (warp == 2 && lane < 2 * NT) {
      const int gh = lane >> 2, s = lane & 3;
      const float av = bf(in.a[s][ghv]), bv = bf(in.b[s][ghv]);
      const float xg = av + dtb;
      const float sp = xg <= 20.f ? __logf(1.f + __expf(xg)) : xg;
      const float g = -__expf(alog) * sp;
      const float beta = 1.f / (1.f + __expf(-bv));
      float G = g, n = __shfl_up_sync(0xffu, G, 1, 4);
      if (s >= 1) G += n;
      n = __shfl_up_sync(0xffu, G, 2, 4);
      if (s >= 2) G += n;
      gate[gh][0][s] = g;
      gate[gh][1][s] = beta;
      gate[gh][2][s] = G;
      gate[gh][3][s] = __expf(G);
      if (wp + s < L) p.g_ring[replay * p.g_slot + ghv * L + ((cb + wp + s) & (L - 1))] = g;
    }
    if (warp == 6) {
      // Committed-history decay per value head (lanes 0-15, 16-31): ring entries older than
      // this window, in logical order.
      const int c = lane & 15;
      const bool valid = c < wp;
      float pre = valid ? in.gring[lane >> 4][(cb + c) & (L - 1)] : 0.f;
#pragma unroll
      for (int o = 1; o < L; o <<= 1) {
        const float n = __shfl_up_sync(FULL, pre, o, L);
        if (c >= o) pre += n;
      }
      const float total = __shfl_sync(FULL, pre, L - 1, L);
      decay[lane >> 4][c] = valid ? __expf(total - pre) : 0.f;
      if (c == 0) tdecay[lane >> 4] = __expf(total);
    }
    // B fragments: column n = lane / 4 of [q | k], rows kk * 8 + lane % 4 (+ 4).
    uint32_t bq[32];
    {
      const int n = lane >> 2, tt = lane & 3;
#pragma unroll
      for (int kk = 0; kk < K / 8; ++kk) {
        bq[2 * kk] = __float_as_uint(qk[n][kk * 8 + tt]);
        bq[2 * kk + 1] = __float_as_uint(qk[n][kk * 8 + tt + 4]);
      }
    }
    float o[NT], D[NT], zz[NT];
    {
      const int g = lane >> 2, tt = lane & 3;
      float acc[4];
      if (warp < 2) {
        // The ring and window keys first: their operands are already resident. Warp 0 takes
        // rows 0-15, warp 1 rows 4-19 and keeps 16-19 (the tile holds exactly 20 rows).
        tile_dots(xbase, XBOX_BYTES, warp * 4, bq, lane, acc);
        if (warp == 0) {
          *reinterpret_cast<float2*>(&xr[g][2 * tt]) = make_float2(acc[0], acc[1]);
          *reinterpret_cast<float2*>(&xr[g + 8][2 * tt]) = make_float2(acc[2], acc[3]);
        } else if (g >= 4) {
          *reinterpret_cast<float2*>(&xr[g + 12][2 * tt]) = make_float2(acc[2], acc[3]);
        }
      }
      // The ring and window dots, the gates and the decay are all the epilogue needs besides the
      // checkpoint rows: everything that does not read them is computed while the stages land.
      consumer_sync();
      float bt[NT], G[NT], eG[NT];
#pragma unroll
      for (int s = 0; s < NT; ++s) {
        bt[s] = gate[h][1][s];
        G[s] = gate[h][2][s];
        eG[s] = gate[h][3][s];
        zz[s] = bf(in.z[s][h * V + v]);
      }
      // kx[i][n] = k_i . [q | k]_n. A = strictly lower beta_i exp(G_i - G_j) k_i.k_j,
      // T = (I + A)^-1 by stock's row-by-row substitution; F = upper exp(G_j - G_i) k_i.q_j.
      const float(*kx)[8] = xr + L;
      // The substitution coefficients, spelled as the served kernel's compiled instructions (ptxas
      // fuses these products into the adds; left implicit, the fusion follows scheduling and moved
      // once this block moved ahead of the stage waits): a_ij = (beta_i exp(G_i - G_j)) k_i.k_j,
      // t20 = fma(a21, a10, -a20), t30 = fma(-(beta_3 exp(G_3 - G_0)), k_3.k_0, fma(a31, a10, -a32 t20)),
      // t31 = fma(a32, a21, -a31).
      const float a10 = __fmul_rn(__fmul_rn(bt[1], __expf(G[1] - G[0])), kx[1][4]);
      const float a20 = __fmul_rn(__fmul_rn(bt[2], __expf(G[2] - G[0])), kx[2][4]);
      const float a21 = __fmul_rn(__fmul_rn(bt[2], __expf(G[2] - G[1])), kx[2][5]);
      const float b30 = __fmul_rn(bt[3], __expf(G[3] - G[0]));
      const float a31 = __fmul_rn(__fmul_rn(bt[3], __expf(G[3] - G[1])), kx[3][5]);
      const float a32 = __fmul_rn(__fmul_rn(bt[3], __expf(G[3] - G[2])), kx[3][6]);
      const float t10 = -a10, t21 = -a21, t32 = -a32;
      const float t20 = __fmaf_rn(a21, a10, -a20);
      const float t30 = __fmaf_rn(-b30, kx[3][4], __fmaf_rn(a31, a10, -__fmul_rn(a32, t20)));
      const float t31 = __fmaf_rn(a32, a21, -a31);
      float F[NT][NT];
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int jj = 0; jj < NT; ++jj) F[i][jj] = i <= jj ? __expf(G[jj] - G[i]) * kx[i][jj] : 0.f;
      float aq[NT] = {0.f, 0.f, 0.f, 0.f}, ak[NT] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < L; ++c) {
        // Selects, not a branch per entry, so the sixteen entries' loads pipeline.
        const bool on = c < wp;
        const float dsc = bf(in.dring[h][(cb + c) & (L - 1)][v]) * decay[h][c];
        const float4 s0 = *reinterpret_cast<const float4*>(&xr[c][0]), s1 = *reinterpret_cast<const float4*>(&xr[c][4]);
        const float xq[NT] = {s0.x, s0.y, s0.z, s0.w}, xk[NT] = {s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int s = 0; s < NT; ++s) {
          aq[s] = on ? fmaf(dsc, xq[s], aq[s]) : aq[s];
          ak[s] = on ? fmaf(dsc, xk[s], ak[s]) : ak[s];
        }
      }
      const float td = tdecay[h];
      float vv[NT];
#pragma unroll
      for (int s = 0; s < NT; ++s) vv[s] = vsh[h][s][v];

      // Warps 0-3 take stages 0 and 2 of the unit (rows 0-63 of each head), warps 4-7 stages
      // 1 and 3 (rows 64-127); warp t & 3 multiplies rows 16 t .. 16 t + 15 of each.
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int j = vec + 2 * jj, nn = n0 + j, s = nn % STAGES;
        mbar_wait(smem_u32(&full[s]), (nn / STAGES) & 1);
        tile_dots(sbase + s * STAGE_BYTES, BOX_BYTES, t * 16, bq, lane, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&empty[s]));
        const int row = (j & 1) * STAGE_ROWS + t * 16 + g;
        *reinterpret_cast<float2*>(&hw[j >> 1][row][2 * tt]) = make_float2(acc[0], acc[1]);
        *reinterpret_cast<float2*>(&hw[j >> 1][row + 8][2 * tt]) = make_float2(acc[2], acc[3]);
      }
      n0 += 4;
      consumer_sync();

      const float4 hq4 = *reinterpret_cast<const float4*>(&hw[h][v][0]);
      const float4 hk4 = *reinterpret_cast<const float4*>(&hw[h][v][4]);
      float hq[NT] = {hq4.x, hq4.y, hq4.z, hq4.w}, hkk[NT] = {hk4.x, hk4.y, hk4.z, hk4.w};
      float R[NT];
#pragma unroll
      for (int s = 0; s < NT; ++s) {
        hq[s] = fmaf(td, hq[s], aq[s]);
        hkk[s] = fmaf(td, hkk[s], ak[s]);
        R[s] = bt[s] * (vv[s] - eG[s] * hkk[s]);
      }
      D[0] = R[0];
      D[1] = fmaf(R[0], t10, R[1]);
      D[2] = fmaf(R[1], t21, fmaf(R[0], t20, R[2]));
      D[3] = fmaf(R[2], t32, fmaf(R[1], t31, fmaf(R[0], t30, R[3])));
#pragma unroll
      for (int jj = 0; jj < NT; ++jj) {
        float df = 0.f;
#pragma unroll
        for (int i = 0; i <= jj; ++i) df = fmaf(D[i], F[i][jj], df);
        o[jj] = bf(tobf(fmaf(eG[jj], hq[jj], df)));  // stock stores the core output in bf16
      }
      float s4[NT];
#pragma unroll
      for (int s = 0; s < NT; ++s) s4[s] = o[s] * o[s];
      const float tot = sum4(s4, lane);
      if ((lane & 7) == 0) red[warp][((lane >> 4) & 1) * 2 + ((lane >> 3) & 1)] = tot;
    }
    {
      bf16* dh = p.d_ring + replay * p.d_slot + static_cast<int64_t>(hv) * L * V + v;
      bf16* dl = p.dlo_ring + replay * p.dlo_slot + static_cast<int64_t>(hv) * L * V + v;
#pragma unroll
      for (int s = 0; s < NT; ++s) {
        if (wp + s < L) {
          const int ph = (cb + wp + s) & (L - 1);
          const bf16 hi = tobf(D[s]);
          dh[ph * V] = hi;
          dl[ph * V] = tobf(D[s] - bf(hi));
        }
      }
    }
    // Past this barrier the unit's input block is dead (the prefetcher may refill it) and only
    // red is read before the next unit's second barrier.
    consumer_sync();
    if (tid == 0) mbar_arrive(smem_u32(&in_empty[bsel]));
#pragma unroll
    for (int s = 0; s < NT; ++s) {
      const float var = (red[4 * h][s] + red[4 * h + 1][s] + red[4 * h + 2][s] + red[4 * h + 3][s]) * (1.f / V);
      const float rstd = rsqrtf(var + p.eps);
      p.out[(bos + s) * ZDIM + hv * V + v] = tobf(o[s] * rstd * nw * (zz[s] * (1.f / (1.f + __expf(-zz[s])))));
    }
#undef TR
  }
}

void check(const torch::Tensor& t, const char* name, at::ScalarType dtype, int dims) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.dim() == dims, name, " must be a ", dims,
              "-d CUDA tensor of ", c10::toString(dtype), ", got ", t.sizes(), " ", t.scalar_type());
}

int idx_flag(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 1 && t.numel() >= n && t.is_contiguous() &&
                  (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong),
              name, " must hold at least ", n, " int32/int64 indices");
  return t.scalar_type() == at::kLong;
}

// The prefetch warp's 16-byte copies need 16-byte aligned sources.
void check_aligned(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must start 16-byte aligned");
}

// A 2D view of the checkpoint pool: one 128-column fp32 row per (slot, head, value row),
// boxes of 32 columns x 64 rows with the 128-byte swizzle.
CUtensorMap h0_map(const torch::Tensor& h0) {
  using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static Encode encode = nullptr;
  if (encode == nullptr) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q));
    TORCH_CHECK(fn != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled is unavailable");
    encode = reinterpret_cast<Encode>(fn);
  }
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(h0.size(0)) * HV * V};
  const cuuint64_t strides[1] = {K * sizeof(float)};
  const cuuint32_t box[2] = {BOX_COLS, STAGE_ROWS};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, h0.data_ptr(), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "checkpoint tensor map encoding failed: ", static_cast<int>(r));
  return map;
}

}  // namespace

torch::Tensor gdn_verify_fused(torch::Tensor mixed, torch::Tensor z, torch::Tensor a, torch::Tensor b,
                               torch::Tensor conv_state, torch::Tensor conv_w, torch::Tensor window,
                               torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor h0, torch::Tensor d_ring,
                               torch::Tensor dlo_ring, torch::Tensor k_ring, torch::Tensor klo_ring,
                               torch::Tensor g_ring, torch::Tensor qsl, torch::Tensor state_idx,
                               torch::Tensor replay_idx, torch::Tensor win_idx, torch::Tensor write_pos,
                               torch::Tensor cache_base, torch::Tensor norm_w, double eps,
                               double scale, torch::Tensor pf_weight  // kgdnpf: the layer's out_proj weight
) {
  const int64_t T = mixed.size(0);
  TORCH_CHECK(T % NT == 0, "verify rows must be a multiple of the draft window ", NT);
  const int B = static_cast<int>(T / NT);
  check(mixed, "mixed_qkv", at::kBFloat16, 2);
  TORCH_CHECK(mixed.size(1) == QKV && mixed.is_contiguous(), "mixed_qkv must be contiguous [T, 8192]");
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kBFloat16 && z.numel() == T * ZDIM && z.is_contiguous(),
              "z must be contiguous bf16 [T, 32, 128]");
  check(a, "a", at::kBFloat16, 2);
  check(b, "b", at::kBFloat16, 2);
  TORCH_CHECK(a.size(1) == HV && b.size(1) == HV && a.is_contiguous() && b.is_contiguous() && a.size(0) == T &&
                  b.size(0) == T,
              "a and b must be contiguous [T, 32]");
  check(conv_state, "conv_state", at::kBFloat16, 3);
  TORCH_CHECK(conv_state.size(1) == QKV && conv_state.size(2) == 3 && conv_state.stride(2) == 1 &&
                  conv_state.stride(1) == 3 && conv_state.stride(0) % 8 == 0,
              "conv_state must be [slots, 8192, 3] with contiguous channels");
  check(conv_w, "conv_weights", at::kBFloat16, 2);
  TORCH_CHECK(conv_w.size(0) == QKV && conv_w.size(1) == 4 && conv_w.is_contiguous(),
              "conv weights must be contiguous [8192, 4]");
  check(window, "intermediate_conv_window", at::kBFloat16, 4);
  // The pool's deduplicated sliding view: draft t's window is row [t, t + 3) of a
  // per-channel [s1 s2 x0 x1 x2 x3] record.
  TORCH_CHECK(window.size(1) == NT && window.size(2) == QKV && window.size(3) == 3 && window.stride(3) == 1 &&
                  window.stride(2) == NT + 2 && window.stride(1) == 1 && window.stride(0) % 8 == 0,
              "conv windows must be the deduplicated [slots, 4, 8192, 3] view of [slots, 8192, 6]");
  check(A_log, "A_log", at::kFloat, 1);
  TORCH_CHECK(A_log.numel() == HV && dt_bias.numel() == HV && dt_bias.is_contiguous() &&
                  (dt_bias.scalar_type() == at::kFloat || dt_bias.scalar_type() == at::kBFloat16),
              "A_log/dt_bias must hold 32 values");
  check(h0, "checkpoint", at::kFloat, 4);
  TORCH_CHECK(h0.size(1) == HV && h0.size(2) == V && h0.size(3) == K && h0.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(h0.data_ptr()) % 16 == 0 && h0.size(0) * HV * V < (int64_t{1} << 31),
              "checkpoint must be a contiguous fp32 [slots, 32, 128, 128] pool");
  for (auto* r : {&d_ring, &dlo_ring}) {
    check(*r, "d ring", at::kBFloat16, 4);
    TORCH_CHECK(r->size(1) == HV && r->size(2) == L && r->size(3) == V && r->stride(3) == 1 && r->stride(2) == V &&
                    r->stride(1) == L * V && r->stride(0) % 8 == 0,
                "d rings must be [slots, 32, 16, 128] with contiguous heads");
  }
  for (auto* r : {&k_ring, &klo_ring}) {
    check(*r, "k ring", at::kBFloat16, 4);
    TORCH_CHECK(r->size(1) == HK && r->size(2) == L && r->size(3) == K && r->stride(3) == 1 && r->stride(2) == K &&
                    r->stride(1) == L * K && r->stride(0) % 8 == 0,
                "k rings must be [slots, 16, 16, 128] with contiguous heads");
  }
  check(g_ring, "g ring", at::kFloat, 3);
  TORCH_CHECK(g_ring.size(1) == HV && g_ring.size(2) == L && g_ring.stride(2) == 1 && g_ring.stride(1) == L &&
                  g_ring.stride(0) % 4 == 0,
              "g ring must be [slots, 32, 16]");
  check_aligned(mixed, "mixed_qkv");
  check_aligned(z, "z");
  check_aligned(a, "a");
  check_aligned(b, "b");
  check_aligned(conv_state, "conv_state");
  check_aligned(d_ring, "d ring");
  check_aligned(k_ring, "k ring");
  check_aligned(g_ring, "g ring");
  TORCH_CHECK(qsl.is_cuda() && qsl.scalar_type() == at::kInt && qsl.numel() >= B + 1 && qsl.is_contiguous(),
              "query_start_loc must hold B + 1 int32 offsets");
  const int state_i64 = idx_flag(state_idx, "mamba_cache_indices", B);
  const int replay_i64 = idx_flag(replay_idx, "req_pool_indices", B);
  const int win_i64 = idx_flag(win_idx, "verify_intermediate_state_indices", B);
  TORCH_CHECK(write_pos.scalar_type() == at::kInt && cache_base.scalar_type() == at::kInt &&
                  write_pos.is_contiguous() && cache_base.is_contiguous(),
              "ring cursors must be contiguous int32");
  check(norm_w, "norm weight", at::kBFloat16, 1);
  TORCH_CHECK(norm_w.numel() == V, "norm weight must hold 128 values");
  // kgdnpf: the prefetch target is only read by the cache hint, never by the arithmetic
  TORCH_CHECK(pf_weight.is_cuda() && pf_weight.device() == mixed.device() && pf_weight.is_contiguous(),
              "pf_weight must be a contiguous tensor on the verify device");

  auto out = torch::empty({T, ZDIM}, mixed.options());
  if (B == 0) return out;
  const c10::cuda::CUDAGuard guard(mixed.device());
  Args p{};
  p.mixed = reinterpret_cast<const bf16*>(mixed.data_ptr());
  p.z = reinterpret_cast<const bf16*>(z.data_ptr());
  p.a = reinterpret_cast<const bf16*>(a.data_ptr());
  p.b = reinterpret_cast<const bf16*>(b.data_ptr());
  p.conv_state = reinterpret_cast<bf16*>(conv_state.data_ptr());
  p.cs_slot = conv_state.stride(0);
  p.conv_w = reinterpret_cast<const bf16*>(conv_w.data_ptr());
  p.window = reinterpret_cast<bf16*>(window.data_ptr());
  p.win_seq = window.stride(0);
  p.A_log = A_log.data_ptr<float>();
  p.dt_bias = dt_bias.data_ptr();
  p.dt_f32 = dt_bias.scalar_type() == at::kFloat;
  p.d_ring = reinterpret_cast<bf16*>(d_ring.data_ptr());
  p.dlo_ring = reinterpret_cast<bf16*>(dlo_ring.data_ptr());
  p.d_slot = d_ring.stride(0);
  p.dlo_slot = dlo_ring.stride(0);
  p.k_ring = reinterpret_cast<bf16*>(k_ring.data_ptr());
  p.klo_ring = reinterpret_cast<bf16*>(klo_ring.data_ptr());
  p.k_slot = k_ring.stride(0);
  p.klo_slot = klo_ring.stride(0);
  p.g_ring = g_ring.data_ptr<float>();
  p.g_slot = g_ring.stride(0);
  p.qsl = qsl.data_ptr<int32_t>();
  p.state_idx = state_idx.data_ptr();
  p.state_i64 = state_i64;
  p.replay_idx = replay_idx.data_ptr();
  p.replay_i64 = replay_i64;
  p.win_idx = win_idx.data_ptr();
  p.win_i64 = win_i64;
  p.write_pos = write_pos.data_ptr<int32_t>();
  p.cache_base = cache_base.data_ptr<int32_t>();
  p.norm_w = reinterpret_cast<const bf16*>(norm_w.data_ptr());
  p.eps = static_cast<float>(eps);
  p.scale = static_cast<float>(scale);
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.units = B * HK;
  const CUtensorMap map = h0_map(h0);

  static bool configured = false;
  static int sms = 0;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_verify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_BYTES));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, mixed.device().index()));
    configured = true;
  }
  const int grid = std::min(p.units, sms);  // one resident CTA per SM walks the units
  TORCH_CHECK((p.units + grid - 1) / grid <= MAXU, "at most ", MAXU * sms / HK, " verify requests per launch, got ", B);
  // kgdnpf: extra prefetch-only CTAs on the SMs this grid leaves idle (none at a full grid)
  const int64_t pf_bytes = std::min<int64_t>(PF_BYTES, pf_weight.numel() * pf_weight.element_size());
  const int extra = grid < sms && pf_bytes >= PF_CHUNK ? std::min(PF_CTAS, sms - grid) : 0;
  p.ctas = grid;
  p.pf = reinterpret_cast<const unsigned char*>(pf_weight.data_ptr());
  p.pf_chunks = extra > 0 ? pf_bytes / PF_CHUNK : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid + extra);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = DYN_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, gdn_verify_kernel, p, map));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("gdn_verify_fused", &gdn_verify_fused, py::arg("mixed"), py::arg("z"), py::arg("a"), py::arg("b"),
        py::arg("conv_state"), py::arg("conv_w"), py::arg("window"), py::arg("A_log"), py::arg("dt_bias"),
        py::arg("h0"), py::arg("d_ring"), py::arg("dlo_ring"), py::arg("k_ring"), py::arg("klo_ring"),
        py::arg("g_ring"), py::arg("qsl"), py::arg("state_idx"), py::arg("replay_idx"), py::arg("win_idx"),
        py::arg("write_pos"), py::arg("cache_base"), py::arg("norm_w"), py::arg("eps"),
        py::arg("scale"), py::arg("pf_weight")  // kgdnpf
  );
}
