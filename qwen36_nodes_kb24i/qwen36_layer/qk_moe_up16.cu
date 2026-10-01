// The decode MoE up projection + SwiGLU (i8x_dec_up's work) for row batches of up to 256 as a weight-streaming native
// kernel over the route's block list (every block: one expert and at most 16 of its tokens), bit for bit the king's
// Triton kernel:
//   routed slot s (token t) of expert e, channel n:  acc_g = fma(d_15, c_15, ... fma(d_0, c_0, 0)), d_j = one chain of
//     eight m16n8k16 (bf16, fp32) steps from zero over k = 128 j .. 128 j + 127 in Triton's kWidth-4 k order,
//     c_j = the gate row's fp16 group scale; acc_u the same on row 512 + n;
//   shared expert (bf16 s13): one continuous chain over K = 2048 in the natural k order;
//   h[s, n] = bf16(g / (1 + ex2(-g * log2 e)) * u), g = bf16(acc_g), u = bf16(acc_u) (the Triton kernel's PTX).
// The products are computed transposed (weight rows as the m16 operand, tokens as the n8 operand). Each warp owns 16
// channels (16 gate + 16 up rows) and streams them through a private cp.async ring of 4 KB stages (both rows' 128 k of
// one scale group, or 64 k of the bf16 shared expert; 16-byte chunks XOR-swizzled by row for conflict-free ldmatrix).
// The CTA's token rows go through a double-buffered smem ring of kXK-wide k slabs (one CTA barrier per slab), which
// keeps the CTA small enough for three per SM. Grid: the shared expert's ceil(M / 16) blocks' chunks, then the route's
// block list (per expert ceil(count / 16) blocks), sized for the most blocks M rows can reach. With early_sh the shared
// chunks never wait: they read only x (final once the route, which waits on x's producer, has launched us), the static
// shared weights and their fixed slots M * 8 + t, so they run under the route; the routed chunks learn their block
// after the wait. Weights
// only one block reads (a routed expert with <= 16 tokens, the shared expert at <= 16 rows) stream through L2 as
// evict_first (the layer's reused buffers stay resident).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef UP_STAGES
#define UP_STAGES 4
#endif
#ifndef UP_WARPS
#define UP_WARPS 2
#endif
#ifndef UP_XK
#define UP_XK 512
#endif
#ifndef UP_MINB
#define UP_MINB 1
#endif

namespace {
constexpr int kH = 2048, kI = 512, kE = 256, kTopK = 8, kMaxT = 256, kGroups = kH / 128, kMaxRows = 16;
constexpr int kWarps = UP_WARPS, kThreads = 32 * kWarps, kStages = UP_STAGES;
constexpr int kCols = 16 * kWarps, kChunks = kI / kCols;  // CTA: (expert, 16 channels per warp)
constexpr int kStageBytes = 32 * 128;                     // 16 gate + 16 up rows x 128 B
constexpr int kRtStages = kH / 128, kShStages = kH * 2 / 128;
constexpr int kXK = UP_XK, kXRow = kXK * 2 + 32, kXSlab = kMaxRows * kXRow;  // row pitch = 32 mod 128: lds.64 at 8q
constexpr int kRtSPS = kXK / 128, kShSPS = kXK / 64;                        // ring stages per x slab
static_assert(kStages - 1 <= kRtSPS, "a slab's loads ride in the ring group issued one slab ahead");
constexpr int kSmemRing = kWarps * kStages * kStageBytes, kSmemSc = kCols * 2 * kGroups * 2;
constexpr int kSmemBytes = kSmemRing + kSmemSc + 2 * kXSlab + 2 * kMaxRows * 4;

extern __shared__ __align__(128) uint8_t smem[];
__device__ __forceinline__ uint32_t sbase() { return static_cast<uint32_t>(__cvta_generic_to_shared(smem)); }
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t off, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(off + sbase()), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16_ef(uint32_t off, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(off + sbase()), "l"(src), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wait_groups() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
// four 8x8 b16 matrices: lane (g, q) gets word q of row g of each; lanes 8m .. 8m + 7 give matrix m's row addresses
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t off) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(off + sbase())
               : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t off) { return *reinterpret_cast<const uint32_t*>(smem + off); }
__device__ __forceinline__ uint2 lds64(uint32_t off) { return *reinterpret_cast<const uint2*>(smem + off); }
// four int8 (k order) -> bf16x2 {b0, b1}, {b2, b3}, exact
__device__ __forceinline__ void i8x4_bf16x2(uint32_t v, uint32_t& lo, uint32_t& hi) {
  asm("{\n"
      ".reg .b32 l0, h0, l1, h1, l2, h2;\n"
      "prmt.b32 l0, %2, 0x43, 0x4140;\n"
      "prmt.b32 h0, %2, 0x43, 0x4342;\n"
      "and.b32 l1, l0, 0xff7fff7f;\n"
      "and.b32 h1, h0, 0xff7fff7f;\n"
      "and.b32 l2, l0, 0xff80ff80;\n"
      "and.b32 h2, h0, 0xff80ff80;\n"
      "sub.bf16x2 %0, l1, l2;\n"
      "sub.bf16x2 %1, h1, h2;\n"
      "}"
      : "=r"(lo), "=r"(hi)
      : "r"(v));
}
__device__ __forceinline__ float ffma(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float bf_round(float v) {
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(v));
  return __uint_as_float(static_cast<uint32_t>(h) << 16);
}
__device__ __forceinline__ uint16_t swiglu_bf16(float gs, float us) {
  const float g = bf_round(gs), u = bf_round(us);
  float t, e, den, q, o;
  asm("sub.f32 %0, 0f00000000, %1;" : "=f"(t) : "f"(g));
  asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(t) : "f"(t));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(den) : "f"(e));
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(g), "f"(den));
  asm("mul.f32 %0, %1, %2;" : "=f"(o) : "f"(q), "f"(u));
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(o));
  return h;
}

struct UpArgs {
  const __nv_bfloat16* x;    // [M, 2048]
  const uint8_t* q8;         // i8x blocks
  const __nv_bfloat16* s13;  // shared gate/up [1024, 2048]
  const int* counts;
  const int* tokens;
  const int* slots;
  const int* nblocks;        // the route's block list: blocks 0 .. ceil(M / 16) - 1 are the shared expert's, then
  const int* bexp;           // per expert with tokens ceil(count / 16) blocks
  const int* bt0;            // a block's first token in its expert's list
  __nv_bfloat16* h;          // [slots, 512]
  int64_t eb, c13_off;
  int M, early_sh;
};

__global__ void __launch_bounds__(kThreads, UP_MINB) up_kernel(const UpArgs a) {
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int n_tb = (a.M + 15) / 16;
  const bool shared = b < kChunks * n_tb;
  const int n0 = (b % kChunks) * kCols + 16 * warp;
  const uint32_t ring = warp * kStages * kStageBytes, s_sc = kSmemRing, s_x = kSmemRing + kSmemSc;
  int* s_tok = reinterpret_cast<int*>(smem + kSmemRing + kSmemSc + 2 * kXSlab);
  int* s_slot = s_tok + kMaxRows;
  const int row_b = shared ? kH * 2 : kH, n_st = shared ? kShStages : kRtStages, sps = shared ? kShSPS : kRtSPS;
  int e = kE, t_first = shared ? 16 * (b / kChunks) : 0, n_tok = shared ? min(16, a.M - t_first) : 0;
  const uint8_t* wg = reinterpret_cast<const uint8_t*>(a.s13) + static_cast<int64_t>(n0) * row_b;
  const uint8_t* wu = wg + static_cast<int64_t>(kI) * row_b;
  // ring stage st: 128 bytes of k of the warp's 16 gate rows (stage rows 0..15) and up rows (16..31); chunk c of row r
  // sits at chunk c ^ (r & 7)
  // weights only this block reads go through L2 as evict_first; an expert split into several blocks (count > 16, or
  // the shared expert above 16 rows) keeps the plain policy, so its other blocks find its rows in L2
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  bool ef = !shared || a.M <= 16;
  auto issue_w = [&](int st) {
    if (st < n_st) {
      const uint32_t dst = ring + (st % kStages) * kStageBytes;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int c = lane + 32 * i, r = c >> 3, k16 = c & 7;
        const uint8_t* src = (r < 16 ? wg + static_cast<int64_t>(r) * row_b : wu + static_cast<int64_t>(r - 16) * row_b) +
                             st * 128 + k16 * 16;
        if (ef) cp16_ef(dst + r * 128 + ((k16 ^ (r & 7)) << 4), src, pol);
        else cp16(dst + r * 128 + ((k16 ^ (r & 7)) << 4), src);
      }
    }
  };
  // x slab j: k = kXK j .. kXK j + kXK - 1 of the CTA's token rows into buffer j & 1 (slab 0 of a routed CTA reads
  // the token list from global: it goes out before the list reaches smem)
  auto issue_x = [&](int j) {
    if (j * kXK < kH) {
      for (int i = tid; i < n_tok * (kXK / 8); i += kThreads) {
        const int r = i / (kXK / 8), c = i % (kXK / 8);
        const int t = shared ? t_first + r : j == 0 ? a.tokens[e * kMaxT + t_first + r] : s_tok[r];
        cp16(s_x + (j & 1) * kXSlab + r * kXRow + c * 16,
             reinterpret_cast<const uint8_t*>(a.x) + (static_cast<int64_t>(t) * kH + j * kXK) * 2 + c * 16);
      }
    }
  };
  // commit group st carries ring stage st; group 0 also the group scales and x slab 0
  if (shared) {  // x, the tokens and the shared weights are all known before the wait
#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
      issue_w(st);
      if (st == 0) issue_x(0);
      commit();
    }
  }
  if (!shared || !a.early_sh) asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  if (!shared) {
    const int blk = n_tb + (b - kChunks * n_tb) / kChunks;
    if (blk >= *a.nblocks) return;
    e = a.bexp[blk];
    t_first = a.bt0[blk];
    n_tok = min(16, a.counts[e] - t_first);
    ef = a.counts[e] <= 16;
    wg = a.q8 + e * a.eb + static_cast<int64_t>(n0) * row_b;
    wu = wg + static_cast<int64_t>(kI) * row_b;
    // ring stage 0, then this CTA's group scales ([kCols][16] fp16 per gate / up row, 32 B per row) and x slab 0
    issue_w(0);
    const uint8_t* sc = a.q8 + e * a.eb + a.c13_off;
    const int cn0 = n0 - 16 * warp;
    for (int i = tid; i < kCols * 2 * 2; i += kThreads) {
      const int r = i >> 1, h16 = i & 1, row = r < kCols ? cn0 + r : kI + cn0 + (r - kCols);
      cp16(s_sc + r * 32 + h16 * 16, sc + static_cast<int64_t>(row) * 32 + h16 * 16);
    }
    issue_x(0);
    commit();
#pragma unroll
    for (int st = 1; st < kStages - 1; ++st) {
      issue_w(st);
      commit();
    }
    for (int i = tid; i < n_tok; i += kThreads) {  // read by the later slabs (after the loop's first barrier)
      s_tok[i] = a.tokens[e * kMaxT + t_first + i];
      s_slot[i] = a.slots[e * kMaxT + t_first + i];
    }
  } else {
    for (int i = tid; i < n_tok; i += kThreads) s_slot[i] = a.M * kTopK + t_first + i;
  }
  const uint32_t lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lsw = lane & 7, lhi = lane >> 4;
  uint16_t* out = reinterpret_cast<uint16_t*>(a.h);
  const int n = n0 + g, n_tt = (n_tok + 7) >> 3;
  float accg[2][4], accu[2][4];
#pragma unroll
  for (int tt = 0; tt < 2; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) accg[tt][i] = accu[tt][i] = 0.f;
  for (int st = 0; st < n_st; ++st) {
    // groups in flight: st .. st + kStages - 2; group st (and every older one, with the slabs riding in them) lands
    wait_groups<kStages - 2>();
    const int in_slab = st % sps;
    if (in_slab == 0) {
      __syncthreads();       // this slab's x is visible to all; every warp is done with the previous slab's buffer
      issue_x(st / sps + 1);  // lands by stage st + sps: its group is st + kStages - 1 <= st + sps
    }
    issue_w(st + kStages - 1);
    commit();
    __syncwarp();
    const uint32_t stage = ring + (st % kStages) * kStageBytes;
    const uint32_t xs = s_x + ((st / sps) & 1) * kXSlab;
    const uint32_t mg = stage + lrow * 128, mu = mg + 16 * 128;  // ldmatrix rows: gate 0..15, up 16..31
    if (shared) {
      // natural k order; stage st covers k = 64 st .. 64 st + 63 (4 steps), slab-local k = 64 in_slab + ...
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t rg[4], ru[4];
        const uint32_t sw = ((2 * s + lhi) ^ lsw) << 4;
        ldsm4(rg, mg + sw);
        ldsm4(ru, mu + sw);
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          if (tt < n_tt) {
            const uint32_t xw = xs + (8 * tt + g) * kXRow + (64 * in_slab + 16 * s + 2 * q) * 2;
            const uint32_t b0 = lds32(xw), b1 = lds32(xw + 16);
            mma16816(accg[tt], rg[0], rg[1], rg[2], rg[3], b0, b1);
            mma16816(accu[tt], ru[0], ru[1], ru[2], ru[3], b0, b1);
          }
        }
      }
    } else {
      uint32_t glo[4][4], ghi[4][4], ulo[4][4], uhi[4][4];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        uint32_t rg[4], ru[4];
        const uint32_t sw = ((2 * jj + lhi) ^ lsw) << 4;
        ldsm4(rg, mg + sw);
        ldsm4(ru, mu + sw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          i8x4_bf16x2(rg[i], glo[jj][i], ghi[jj][i]);
          i8x4_bf16x2(ru[i], ulo[jj][i], uhi[jj][i]);
        }
      }
      // group st scales of rows g / g + 8 (gate, up)
      const float sg0 = __half2float(*reinterpret_cast<const __half*>(smem + s_sc + (16 * warp + g) * 32 + st * 2));
      const float sg1 = __half2float(*reinterpret_cast<const __half*>(smem + s_sc + (16 * warp + g + 8) * 32 + st * 2));
      const float su0 = __half2float(*reinterpret_cast<const __half*>(smem + s_sc + (kCols + 16 * warp + g) * 32 + st * 2));
      const float su1 = __half2float(*reinterpret_cast<const __half*>(smem + s_sc + (kCols + 16 * warp + g + 8) * 32 + st * 2));
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        if (tt < n_tt) {
          float dg[4] = {0.f, 0.f, 0.f, 0.f}, du[4] = {0.f, 0.f, 0.f, 0.f};
          const uint32_t xrow = xs + (8 * tt + g) * kXRow + (128 * in_slab) * 2 + 8 * q;  // k = 128 st + 4q
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const uint2 x0 = lds64(xrow + 64 * jj), x1 = lds64(xrow + 64 * jj + 32);  // k + 32 jj, + 16
            mma16816(dg, glo[jj][0], glo[jj][1], glo[jj][2], glo[jj][3], x0.x, x1.x);
            mma16816(du, ulo[jj][0], ulo[jj][1], ulo[jj][2], ulo[jj][3], x0.x, x1.x);
            mma16816(dg, ghi[jj][0], ghi[jj][1], ghi[jj][2], ghi[jj][3], x0.y, x1.y);
            mma16816(du, uhi[jj][0], uhi[jj][1], uhi[jj][2], uhi[jj][3], x0.y, x1.y);
          }
          // D: (row g, tok 2q), (g, 2q + 1), (g + 8, 2q), (g + 8, 2q + 1)
          accg[tt][0] = ffma(dg[0], sg0, accg[tt][0]);
          accg[tt][1] = ffma(dg[1], sg0, accg[tt][1]);
          accg[tt][2] = ffma(dg[2], sg1, accg[tt][2]);
          accg[tt][3] = ffma(dg[3], sg1, accg[tt][3]);
          accu[tt][0] = ffma(du[0], su0, accu[tt][0]);
          accu[tt][1] = ffma(du[1], su0, accu[tt][1]);
          accu[tt][2] = ffma(du[2], su1, accu[tt][2]);
          accu[tt][3] = ffma(du[3], su1, accu[tt][3]);
        }
      }
    }
    __syncwarp();  // this ring slot is refilled by a later stage's issue
  }
#pragma unroll
  for (int tt = 0; tt < 2; ++tt) {
    const int t = 8 * tt + 2 * q;
    if (t < n_tok) {
      const int64_t o = static_cast<int64_t>(s_slot[t]) * kI + n;
      out[o] = swiglu_bf16(accg[tt][0], accu[tt][0]);
      out[o + 8] = swiglu_bf16(accg[tt][2], accu[tt][2]);
    }
    if (t + 1 < n_tok) {
      const int64_t o = static_cast<int64_t>(s_slot[t + 1]) * kI + n;
      out[o] = swiglu_bf16(accg[tt][1], accu[tt][1]);
      out[o + 8] = swiglu_bf16(accg[tt][3], accu[tt][3]);
    }
  }
  wait_groups<0>();
}
}  // namespace

void up(torch::Tensor x, torch::Tensor q8, torch::Tensor s13, torch::Tensor counts, torch::Tensor tokens,
        torch::Tensor slots, torch::Tensor nblocks, torch::Tensor bexp, torch::Tensor bt0, torch::Tensor h, int64_t eb,
        int64_t c13_off, int64_t early_sh) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && h.scalar_type() == at::kBFloat16 && s13.scalar_type() == at::kBFloat16,
              "bf16 tensors");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= kMaxT, "the native decode up takes 1..256 rows");
  const at::cuda::CUDAGuard guard(x.device());
  // blocks: the shared expert's ceil(M / 16), then at most min(E, 8 M) experts' first blocks and 8 M / 16 further ones
  const int m = static_cast<int>(x.size(0)), n_tb = (m + 15) / 16;
  const int blocks = n_tb + (kTopK * m < kE ? kTopK * m : kE) + kTopK * m / 16;
  UpArgs a{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint8_t*>(q8.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(s13.data_ptr()), counts.data_ptr<int>(), tokens.data_ptr<int>(),
           slots.data_ptr<int>(), nblocks.data_ptr<int>(), bexp.data_ptr<int>(), bt0.data_ptr<int>(),
           reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), eb, c13_off, m, static_cast<int>(early_sh)};
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kChunks * blocks);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, up_kernel, a));
}


// Persistent fused decode MoE for row batches of 1..16 (qk_moe_up16.fused): the shared + routed up projection, SwiGLU
// and down projection of one layer in one launch, bit for bit the pair it replaces (up16 / dn16 / i8x_dec_up / _down):
//   routed up (slot s, channel n):   acc = fma(d_15, c_15, ... fma(d_0, c_0, 0)), d_j = eight m16n8k16 steps from zero
//                                    over the 128-deep group j in Triton's kWidth-4 k order; h = swiglu(bf16 g, bf16 u)
//   routed down (slot s, column n):  acc = fma(d_3, c_3, ... fma(d_0, c_0, 0)); cache = bf16(acc * wts[s])
//   shared up / down (bf16):         one chain over K in the natural k order; h = swiglu(...), cache = bf16(acc)
// Every output element is its own row's dot product, so the m16 tiles may group rows freely (the shared up puts 8 gate
// rows and the matching 8 up rows in one tile).
// One CTA per SM of producer / consumer warp pairs. A producer claims tasks from a global counter (the next one only a
// few stages before its current task ends, so no pair hoards work) and streams each task's stages -- 4 KB of weights
// (32 INT8 rows x 128 k, or 16 bf16 rows x 128 k) plus the stage's 128-k slice of the task's activation rows (x for
// up, h for down) -- with cp.async into its consumer's ring; mbarriers track completion (full: the producer lanes'
// cp.async arrivals + one release arrival that also covers the task copy; empty: the consumer's release). The consumer
// only computes, so the weight stream never waits on the math.
// Task queue: shared up (64 x 8 channels), then after the route: routed up (per active expert 32 x 16 channels), shared
// down (128 x 16 columns), routed down (per active expert 64 x 32 columns). The shared up needs only x (final once the
// route that launched us waited on its producer) and runs under the route. A down task's producer waits until the
// expert's up tasks have all published their h rows (per-expert counters, release / acquire). The last CTA to exit
// resets the counters.
#ifndef PM_PAIRS16
#define PM_PAIRS16 8  // producer / consumer pairs per CTA for 9..16 rows (kb18: 7 -> 8, fm/t_fm9.py -0.9 us/layer at 16 rows)
#endif
#ifndef PM_STAGES16
#define PM_STAGES16 3
#endif
#ifndef PM_PAIRS8
#define PM_PAIRS8 7   // for 1..8 rows (half the activation area per stage)
#endif
#ifndef PM_STAGES8
#define PM_STAGES8 3
#endif
#ifndef PM_CPS
#define PM_CPS 1  // CTAs per SM (smaller CTAs free their SM share progressively at the end, for the next kernels)
#endif
#ifndef PM_SHD_LATE
#define PM_SHD_LATE 1  // the shared down tasks follow the routed up ones (they never wait for the shared up)
#endif
#ifndef PM_LATE_Q
#define PM_LATE_Q 3  // claim the next task PM_LATE_Q stages before the current one's last (no early hoarding)
#endif
static_assert(PM_LATE_Q >= 1, "late claiming");
#ifndef PM_TRACE
#define PM_TRACE 0
#endif

namespace pmw {
constexpr int kH = 2048, kI = 512, kE = 256, kTopK = 8, kMaxT = 256;

constexpr int kWB = 4096, kAP = 272;
constexpr int kShU = kI / 8, kShD = kH / 16, kRtU = kI / 16, kRtD = kH / 32;
constexpr int kNSh = PM_SHD_LATE ? kShU : kShU + kShD;
enum { END = 0, SHU = 1, SHD = 2, RTU = 3, RTD = 4 };
constexpr int kCtlDone = 8, kCtlInts = kCtlDone + kE + 1;
template <int TC>
struct Lay {
  static constexpr int kP = TC > 8 ? PM_PAIRS16 : PM_PAIRS8, kNS = TC > 8 ? PM_STAGES16 : PM_STAGES8;
  static_assert(kNS >= 2 && kNS <= 4, "at most two tasks in flight per pair (the shortest task has 4 stages)");
  static constexpr int kStage = kWB + TC * kAP;
  static constexpr int kRing = kNS * kStage;
  static constexpr int kSc = 1024, kDesc = 256;
  static constexpr int kPairB = kRing + 2 * kSc + 2 * kDesc + 128;  // + full / empty mbarriers
  static constexpr int kSmem = kP * kPairB;
  static constexpr int kThreads = 64 * kP;
  static_assert(kStage % 128 == 0 && kPairB % 128 == 0, "128-byte aligned stages");
};
extern __shared__ __align__(128) uint8_t smem[];
__device__ __forceinline__ uint32_t sbase() { return static_cast<uint32_t>(__cvta_generic_to_shared(smem)); }
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t off, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(off + sbase()), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16_ef(uint32_t off, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(off + sbase()), "l"(src), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void wait_n(int n) {  // this thread's copies: all but the n most recent groups landed
  if (n <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else if (n == 2) asm volatile("cp.async.wait_group 2;" ::: "memory");
  else asm volatile("cp.async.wait_group 3;" ::: "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t off) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(off + sbase())
               : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t off) { return *reinterpret_cast<const uint32_t*>(smem + off); }
__device__ __forceinline__ uint2 lds64(uint32_t off) { return *reinterpret_cast<const uint2*>(smem + off); }
__device__ __forceinline__ float lds_h(uint32_t off) { return __half2float(*reinterpret_cast<const __half*>(smem + off)); }
__device__ __forceinline__ void i8x4_bf16x2(uint32_t v, uint32_t& lo, uint32_t& hi) {
  asm("{\n"
      ".reg .b32 l0, h0, l1, h1, l2, h2;\n"
      "prmt.b32 l0, %2, 0x43, 0x4140;\n"
      "prmt.b32 h0, %2, 0x43, 0x4342;\n"
      "and.b32 l1, l0, 0xff7fff7f;\n"
      "and.b32 h1, h0, 0xff7fff7f;\n"
      "and.b32 l2, l0, 0xff80ff80;\n"
      "and.b32 h2, h0, 0xff80ff80;\n"
      "sub.bf16x2 %0, l1, l2;\n"
      "sub.bf16x2 %1, h1, h2;\n"
      "}"
      : "=r"(lo), "=r"(hi)
      : "r"(v));
}
__device__ __forceinline__ float ffma(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float fmul(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint16_t bf16_bits(float v) {
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(v));
  return h;
}
__device__ __forceinline__ float bf_round(float v) { return __uint_as_float(static_cast<uint32_t>(bf16_bits(v)) << 16); }
__device__ __forceinline__ uint16_t swiglu_bf16(float gs, float us) {
  const float g = bf_round(gs), u = bf_round(us);
  float t, e, den, q, o;
  asm("sub.f32 %0, 0f00000000, %1;" : "=f"(t) : "f"(g));
  asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(t) : "f"(t));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(den) : "f"(e));
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(g), "f"(den));
  asm("mul.f32 %0, %1, %2;" : "=f"(o) : "f"(q), "f"(u));
  return bf16_bits(o);
}
__device__ __forceinline__ void mbar_init(uint32_t bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar + sbase()) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar + sbase()), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}" ::"r"(bar + sbase()), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst + sbase()), "l"(src), "r"(bytes), "r"(bar + sbase()) : "memory");
}
__device__ __forceinline__ void bulk_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(dst + sbase()), "l"(src), "r"(bytes), "r"(bar + sbase()), "l"(pol) : "memory");
}
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}


__device__ __forceinline__ void mbar_init_n(uint32_t bar, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar + sbase()), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar + sbase()) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t bar) {  // arrives once this thread's prior cp.async have landed
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar + sbase()) : "memory");
}

struct Args {
  const __nv_bfloat16* x;    // [M, 2048]
  const uint8_t* q8;         // i8x blocks [256, eb]
  const __nv_bfloat16* s13;  // shared gate/up [1024, 2048]
  const __nv_bfloat16* s2;   // shared down [2048, 512]
  const float* wts;
  const int* counts;
  const int* tokens;
  const int* slots;
  const int* nblocks;
  const int* bexp;
  __nv_bfloat16* h;          // [>= M * 9, 512]
  __nv_bfloat16* cache;      // [>= M * 9, 2048]
  int* ctl;                  // [0] queue head, [1] exit ticket, [8 + e] up tasks published per expert (256: shared)
  int64_t eb, c13, q2, c2;
  int M;
  long long* trace;
};

struct Task {
  int kind, e, c0, n_tok, n_st, rid, slot, pitch, need;
  const uint8_t* wa;
  const uint8_t* wb;
  const uint8_t* sa;
  const uint8_t* sb;
};

template <int TC>
__device__ __forceinline__ void producer(const Args& a, int p, int lane, uint32_t ring, uint32_t scb, uint32_t dsb,
                                         uint32_t full, uint32_t empty) {
  using LY = Lay<TC>;
  int* const head = a.ctl;
  int* const done = a.ctl + kCtlDone;
  const int M = a.M;
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  bool route_ok = false;
  int n_rt = 0, n_up = 0, total = kNSh;
  int q_next = 0, q_pre = 0;  // the task being decoded; the next one, claimed PM_LATE_Q stages before this one ends
  if (lane == 0) q_next = atomicAdd(head, 1);
  q_next = __shfl_sync(0xffffffffu, q_next, 0);
  int gs = 0, task = 0;
  while (true) {
    const int qq = q_next;
    if (qq >= kNSh && !route_ok) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      asm volatile("griddepcontrol.launch_dependents;");
      route_ok = true;
      n_rt = *a.nblocks - 1;
      n_up = n_rt * kRtU;
      total = kNSh + n_up + n_rt * kRtD + (PM_SHD_LATE ? kShD : 0);
    }
    const int slot0 = gs % LY::kNS;
    if (qq >= total) {  // the consumer's end marker: a stage whose task copy says END
      if (gs >= LY::kNS) mbar_wait(empty + 8 * slot0, ((gs / LY::kNS) - 1) & 1);
      int* d = reinterpret_cast<int*>(smem + dsb + (task & 1) * LY::kDesc);
      if (lane == 0) d[0] = END;
      __syncwarp();
      cp_arrive(full + 8 * slot0);
      if (lane == 0) mbar_arrive(full + 8 * slot0);
      break;
    }
    Task t{};
    if (qq < kShU) {
      t.kind = SHU, t.e = kE, t.c0 = 8 * qq, t.n_tok = M, t.n_st = kH / 128, t.pitch = kH * 2;
      t.wa = reinterpret_cast<const uint8_t*>(a.s13) + static_cast<int64_t>(t.c0) * kH * 2;
      t.wb = t.wa + static_cast<int64_t>(kI) * kH * 2;
      t.rid = lane, t.slot = M * kTopK + lane;
    } else if (!PM_SHD_LATE && qq < kNSh) {
      t.kind = SHD, t.c0 = 16 * (qq - kShU);
    } else if (PM_SHD_LATE && qq - kNSh >= n_up && qq - kNSh - n_up < kShD) {
      t.kind = SHD, t.c0 = 16 * (qq - kNSh - n_up);
    } else {
      int r = qq - kNSh, i;
      if (r < n_up) {
        i = r / kRtU, t.kind = RTU, t.c0 = 16 * (r % kRtU);
      } else {
        r -= n_up + (PM_SHD_LATE ? kShD : 0);
        i = r / kRtD, t.kind = RTD, t.c0 = 32 * (r % kRtD);
      }
      t.need = kRtU;
      const int e = a.bexp[1 + i];
      t.n_tok = a.counts[e];
      const int sl = a.slots[e * kMaxT + (lane & 15)], tk = a.tokens[e * kMaxT + (lane & 15)];
      t.e = e;
      t.slot = sl;
      const uint8_t* blk = a.q8 + static_cast<int64_t>(e) * a.eb;
      if (t.kind == RTU) {
        t.rid = tk;
        t.n_st = kH / 128, t.pitch = kH;
        t.wa = blk + static_cast<int64_t>(t.c0) * kH;
        t.wb = blk + static_cast<int64_t>(kI + t.c0) * kH;
        t.sa = blk + a.c13 + t.c0 * 32;
        t.sb = blk + a.c13 + (kI + t.c0) * 32;
      } else {
        t.rid = sl;
        t.n_st = kI / 128, t.pitch = kI;
        t.wa = blk + a.q2 + static_cast<int64_t>(t.c0) * kI;
        t.wb = t.wa + 16 * kI;
        t.sa = blk + a.c2 + t.c0 * 8;
      }
    }
    if (t.kind == SHD) {
      t.need = kShU;
      t.e = kE, t.n_tok = M, t.n_st = kI / 128, t.pitch = kI * 2;
      t.wa = reinterpret_cast<const uint8_t*>(a.s2) + static_cast<int64_t>(t.c0) * kI * 2;
      t.wb = t.wa + 8 * kI * 2;
      t.rid = t.slot = M * kTopK + lane;
    }
    if (t.kind == SHD || t.kind == RTD) {  // the expert's up side must have published its h rows
      if (lane == 0)
        while (ld_acquire(done + t.e) < t.need) __nanosleep(32);
      __syncwarp();
    }
    const bool i8 = t.kind >= RTU, up = t.kind == SHU || t.kind == RTU;
    const int chunk = i8 ? (lane & 7) : (lane & 15);
    const uint8_t* act = up ? reinterpret_cast<const uint8_t*>(a.x) : reinterpret_cast<const uint8_t*>(a.h);
    const int apitch = up ? kH * 2 : kI * 2;
    for (int st = 0; st < t.n_st; ++st, ++gs) {
      const int slot = gs % LY::kNS;
      if (gs >= LY::kNS) mbar_wait(empty + 8 * slot, ((gs / LY::kNS) - 1) & 1);
      if (st == 0) {  // the consumer's copy of the task
        int* d = reinterpret_cast<int*>(smem + dsb + (task & 1) * LY::kDesc);
        if (lane == 0) d[0] = t.kind, d[1] = t.e, d[2] = t.c0, d[3] = t.n_tok, d[4] = t.n_st, d[5] = qq;
        if (PM_TRACE && lane == 0) *reinterpret_cast<long long*>(d + 6) = gtime();
        if (lane < 16) d[8 + lane] = t.slot;
        __syncwarp();
      }
      const uint32_t sb = ring + slot * LY::kStage;
      const int64_t koff = static_cast<int64_t>(st) * (i8 ? 128 : 256) + chunk * 16;
#pragma unroll
      for (int i = 0; i < 8; ++i) {  // INT8: rows 0..15 gate / first 16 columns, 16..31 up / last; bf16: 0..7, 8..15
        const int rl = i8 ? (lane >> 3) + 4 * (i & 3) : (lane >> 4) + 2 * (i & 3);
        const uint8_t* src = (i < 4 ? t.wa : t.wb) + static_cast<int64_t>(rl) * t.pitch + koff;
        const int rf = rl + (i < 4 ? 0 : (i8 ? 16 : 8));
        cp16_ef(sb + rf * (i8 ? 128 : 256) + ((chunk ^ (rl & 7)) << 4), src, pol);
      }
      const int n_it = (t.n_tok + 1) >> 1;
      for (int m = 0; m < n_it; ++m) {
        const int tk = 2 * m + (lane >> 4);
        const int rid = __shfl_sync(0xffffffffu, t.rid, tk & 15);
        if (tk < t.n_tok)
          cp16(sb + kWB + tk * kAP + (lane & 15) * 16, act + static_cast<int64_t>(rid) * apitch + st * 256 + (lane & 15) * 16);
      }
      if (st == 0 && i8) {
        const uint32_t sc = scb + (task & 1) * LY::kSc;
        if (t.kind == RTU) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int j = lane + 32 * i, r = j >> 1, h16 = j & 1;
            cp16(sc + r * 32 + h16 * 16, (r < 16 ? t.sa + r * 32 : t.sb + (r - 16) * 32) + h16 * 16);
          }
        } else if (lane < 16) {
          cp16(sc + lane * 16, t.sa + lane * 16);
        }
      }
      cp_arrive(full + 8 * slot);
      if (lane == 0) mbar_arrive(full + 8 * slot);
      if (st == (t.n_st > PM_LATE_Q ? t.n_st - PM_LATE_Q : 0) && lane == 0) q_pre = atomicAdd(head, 1);
    }
    ++task;
    q_next = __shfl_sync(0xffffffffu, q_pre, 0);
  }
  if (!route_ok) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
}

template <int TC>
__device__ __forceinline__ void consumer(const Args& a, int lane, uint32_t ring, uint32_t scb, uint32_t dsb, uint32_t full,
                                         uint32_t empty) {
  using LY = Lay<TC>;
  int* const done = a.ctl + kCtlDone;
  const int M = a.M, g = lane >> 2, q = lane & 3;
  const uint32_t lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lsw = lane & 7, lhi = lane >> 4;
  int c_kind = END, c_e = 0, c_c0 = 0, c_ntok = 0, c_nst = 1, c_st = 0, c_task = 0, c_slot = 0;
  float c_wt = 0.f;
  float acc[2][2][4];
  uint16_t* hout = reinterpret_cast<uint16_t*>(a.h);
  uint16_t* cout = reinterpret_cast<uint16_t*>(a.cache);
  for (int gs = 0;; ++gs) {
    const int slot = gs % LY::kNS;
    mbar_wait(full + 8 * slot, (gs / LY::kNS) & 1);
    if (c_st == 0) {
      const int* d = reinterpret_cast<const int*>(smem + dsb + (c_task & 1) * LY::kDesc);
      c_kind = d[0];
      if (c_kind == END) break;
      c_e = d[1], c_c0 = d[2], c_ntok = d[3], c_nst = d[4];
      c_slot = d[8 + (lane & 15)];
      if (PM_TRACE && lane == 0 && a.trace) {
        long long* tr = a.trace + 8 * d[5];
        unsigned sm;
        asm("mov.u32 %0, %%smid;" : "=r"(sm));
        tr[0] = d[5], tr[1] = c_kind, tr[2] = c_e, tr[3] = blockIdx.x, tr[4] = sm;
        tr[5] = *reinterpret_cast<const long long*>(d + 6), tr[6] = gtime();
      }
      if (c_kind == RTD) c_wt = lane < c_ntok ? a.wts[c_slot] : 0.f;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int tt = 0; tt < 2; ++tt)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[t][tt][i] = 0.f;
    }
    const uint32_t sb = ring + slot * LY::kStage, act = sb + kWB;
    const int n_tt = (c_ntok + 7) >> 3;
    if (c_kind >= RTU) {
      const uint32_t sc = scb + (c_task & 1) * LY::kSc;
      const int sp = c_kind == RTD ? 8 : 32;
      float s[2][2];
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        s[t][0] = lds_h(sc + (16 * t + g) * sp + c_st * 2);
        s[t][1] = lds_h(sc + (16 * t + g + 8) * sp + c_st * 2);
      }
      float d[2][2][4];  // [tt][tile]
#pragma unroll
      for (int tt = 0; tt < 2; ++tt)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i) d[tt][t][i] = 0.f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        uint32_t lo[2][4], hi[2][4];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          uint32_t r[4];
          ldsm4(r, sb + (16 * t + lrow) * 128 + (((2 * jj + lhi) ^ lsw) << 4));
#pragma unroll
          for (int i = 0; i < 4; ++i) i8x4_bf16x2(r[i], lo[t][i], hi[t][i]);
        }
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          if (tt < n_tt) {
            const uint32_t xrow = act + (8 * tt + g) * kAP + 8 * q + 64 * jj;
            const uint2 x0 = lds64(xrow), x1 = lds64(xrow + 32);
            mma16816(d[tt][0], lo[0], x0.x, x1.x);
            mma16816(d[tt][1], lo[1], x0.x, x1.x);
            mma16816(d[tt][0], hi[0], x0.y, x1.y);
            mma16816(d[tt][1], hi[1], x0.y, x1.y);
          }
        }
      }
#pragma unroll
      for (int tt = 0; tt < 2; ++tt)
        if (tt < n_tt)
#pragma unroll
          for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[t][tt][i] = ffma(d[tt][t][i], s[t][i >> 1], acc[t][tt][i]);
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t r[4];
        ldsm4(r, sb + lrow * 256 + (((2 * s + lhi) ^ lsw) << 4));
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          if (tt < n_tt) {
            const uint32_t xw = act + (8 * tt + g) * kAP + 32 * s + 4 * q;
            mma16816(acc[0][tt], r, lds32(xw), lds32(xw + 16));
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + 8 * slot);  // the stage's smem is consumed (the math below uses registers)
    if (++c_st == c_nst) {
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const int t0 = 8 * tt + 2 * q, t1 = t0 + 1;
        const int s0 = __shfl_sync(0xffffffffu, c_slot, t0), s1 = __shfl_sync(0xffffffffu, c_slot, t1);
        const float w0 = __shfl_sync(0xffffffffu, c_wt, t0), w1 = __shfl_sync(0xffffffffu, c_wt, t1);
        if (c_kind == RTU) {
          if (t0 < c_ntok) {
            const int64_t o = static_cast<int64_t>(s0) * kI + c_c0 + g;
            hout[o] = swiglu_bf16(acc[0][tt][0], acc[1][tt][0]);
            hout[o + 8] = swiglu_bf16(acc[0][tt][2], acc[1][tt][2]);
          }
          if (t1 < c_ntok) {
            const int64_t o = static_cast<int64_t>(s1) * kI + c_c0 + g;
            hout[o] = swiglu_bf16(acc[0][tt][1], acc[1][tt][1]);
            hout[o + 8] = swiglu_bf16(acc[0][tt][3], acc[1][tt][3]);
          }
        } else if (c_kind == SHU) {  // rows g: gate channel c0 + g, g + 8: its up channel
          if (t0 < c_ntok) hout[static_cast<int64_t>(s0) * kI + c_c0 + g] = swiglu_bf16(acc[0][tt][0], acc[0][tt][2]);
          if (t1 < c_ntok) hout[static_cast<int64_t>(s1) * kI + c_c0 + g] = swiglu_bf16(acc[0][tt][1], acc[0][tt][3]);
        } else if (c_kind == RTD) {
          if (t0 < c_ntok) {
            const int64_t o = static_cast<int64_t>(s0) * kH + c_c0 + g;
            cout[o] = bf16_bits(fmul(acc[0][tt][0], w0));
            cout[o + 8] = bf16_bits(fmul(acc[0][tt][2], w0));
            cout[o + 16] = bf16_bits(fmul(acc[1][tt][0], w0));
            cout[o + 24] = bf16_bits(fmul(acc[1][tt][2], w0));
          }
          if (t1 < c_ntok) {
            const int64_t o = static_cast<int64_t>(s1) * kH + c_c0 + g;
            cout[o] = bf16_bits(fmul(acc[0][tt][1], w1));
            cout[o + 8] = bf16_bits(fmul(acc[0][tt][3], w1));
            cout[o + 16] = bf16_bits(fmul(acc[1][tt][1], w1));
            cout[o + 24] = bf16_bits(fmul(acc[1][tt][3], w1));
          }
        } else {  // SHD
          if (t0 < c_ntok) {
            const int64_t o = static_cast<int64_t>(s0) * kH + c_c0 + g;
            cout[o] = bf16_bits(acc[0][tt][0]);
            cout[o + 8] = bf16_bits(acc[0][tt][2]);
          }
          if (t1 < c_ntok) {
            const int64_t o = static_cast<int64_t>(s1) * kH + c_c0 + g;
            cout[o] = bf16_bits(acc[0][tt][1]);
            cout[o + 8] = bf16_bits(acc[0][tt][3]);
          }
        }
      }
      if (c_kind == RTU || c_kind == SHU) {  // publish: the warp's h stores, then the counter (release)
        __syncwarp();
        if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(done + c_e) : "memory");
      }
      if (PM_TRACE && lane == 0 && a.trace) {
        const int* d = reinterpret_cast<const int*>(smem + dsb + (c_task & 1) * LY::kDesc);
        a.trace[8 * d[5] + 7] = gtime();
      }
      c_st = 0;
      ++c_task;
    }
  }
}

template <int TC>
__global__ void __launch_bounds__(Lay<TC>::kThreads, PM_CPS) pmoe_ws_kernel(const Args a) {
  using LY = Lay<TC>;
  constexpr int kP = LY::kP;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int p = warp % kP;
  const uint32_t ring = p * LY::kPairB, scb = ring + LY::kRing, dsb = scb + 2 * LY::kSc, full = dsb + 2 * LY::kDesc,
                 empty = full + 8 * LY::kNS;
  if (warp < kP && lane < LY::kNS) {
    mbar_init_n(full + 8 * lane, 33);
    mbar_init_n(empty + 8 * lane, 1);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  if (warp < kP) producer<TC>(a, p, lane, ring, scb, dsb, full, empty);
  else consumer<TC>(a, lane, ring, scb, dsb, full, empty);
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(a.ctl + 1, 1) == static_cast<int>(gridDim.x) - 1) {
      int* done = a.ctl + kCtlDone;
      for (int i = 0; i <= kE; ++i) done[i] = 0;
      a.ctl[0] = 0;
      a.ctl[1] = 0;
      __threadfence();
    }
  }
}
}  // namespace pmw

static int pmw_sms() {
  static int n = 0;
  if (!n) {
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  }
  return n;
}

void fused(torch::Tensor x, torch::Tensor q8, torch::Tensor s13, torch::Tensor s2, torch::Tensor wts, torch::Tensor counts,
           torch::Tensor tokens, torch::Tensor slots, torch::Tensor nblocks, torch::Tensor bexp, torch::Tensor h,
           torch::Tensor cache, torch::Tensor ctl, int64_t eb, int64_t c13, int64_t q2, int64_t c2, int64_t grid,
           c10::optional<torch::Tensor> trace) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == pmw::kH,
              "x [M, 2048] bf16");
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "the fused decode MoE takes 1..16 rows");
  TORCH_CHECK(h.scalar_type() == at::kBFloat16 && h.is_contiguous() && h.size(1) == pmw::kI && h.size(0) >= M * (pmw::kTopK + 1), "h");
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.is_contiguous() && cache.size(1) == pmw::kH &&
                  cache.size(0) >= M * (pmw::kTopK + 1), "cache");
  TORCH_CHECK(s13.scalar_type() == at::kBFloat16 && s13.is_contiguous() && s13.size(0) == 2 * pmw::kI && s13.size(1) == pmw::kH, "s13");
  TORCH_CHECK(s2.scalar_type() == at::kBFloat16 && s2.is_contiguous() && s2.size(0) == pmw::kH && s2.size(1) == pmw::kI, "s2");
  TORCH_CHECK(q8.is_contiguous() && q8.dim() == 2 && q8.size(0) == pmw::kE && q8.size(1) == eb && q8.element_size() == 1, "q8");
  TORCH_CHECK(ctl.scalar_type() == at::kInt && ctl.numel() >= pmw::kCtlInts && ctl.is_contiguous(), "ctl");
  TORCH_CHECK(wts.scalar_type() == at::kFloat && counts.scalar_type() == at::kInt && slots.scalar_type() == at::kInt &&
                  tokens.scalar_type() == at::kInt && nblocks.scalar_type() == at::kInt && bexp.scalar_type() == at::kInt,
              "routing lists");
  for (const auto* t : {&x, &q8, &s13, &s2, &h})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "16-byte aligned operands");
  const at::cuda::CUDAGuard guard(x.device());
  pmw::Args a{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint8_t*>(q8.data_ptr()),
         reinterpret_cast<const __nv_bfloat16*>(s13.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(s2.data_ptr()),
         wts.data_ptr<float>(), counts.data_ptr<int>(), tokens.data_ptr<int>(), slots.data_ptr<int>(),
         nblocks.data_ptr<int>(), bexp.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(h.data_ptr()),
         reinterpret_cast<__nv_bfloat16*>(cache.data_ptr()), ctl.data_ptr<int>(), eb, c13, q2, c2, static_cast<int>(M),
         trace.has_value() ? reinterpret_cast<long long*>(trace->data_ptr<int64_t>()) : nullptr};
  const bool small = M <= 8;
  auto kernel = small ? pmw::pmoe_ws_kernel<8> : pmw::pmoe_ws_kernel<16>;
  const int smem = small ? pmw::Lay<8>::kSmem : pmw::Lay<16>::kSmem;
  static bool attr[2] = {false, false};
  if (!attr[small]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr[small] = true;
  }
  cudaLaunchConfig_t cfg = {};
  // grid > 0: that many CTAs; grid <= 0: PM_CPS per SM minus -grid (SMs left to the next kernels' weight prefetch)
  cfg.gridDim = dim3(static_cast<unsigned>(grid > 0 ? grid : std::max(1, PM_CPS * pmw_sms() + static_cast<int>(grid))));
  cfg.blockDim = dim3(small ? pmw::Lay<8>::kThreads : pmw::Lay<16>::kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("up", &up, "decode MoE up projection + SwiGLU over the route's block list (routed INT8 + shared bf16), bit for bit i8x_dec_up");
  m.def("fused", &fused, py::arg("x"), py::arg("q8"), py::arg("s13"), py::arg("s2"), py::arg("wts"), py::arg("counts"),
        py::arg("tokens"), py::arg("slots"), py::arg("nblocks"), py::arg("bexp"), py::arg("h"), py::arg("cache"),
        py::arg("ctl"), py::arg("eb"), py::arg("c13"), py::arg("q2"), py::arg("c2"), py::arg("grid"),
        py::arg("trace") = py::none(), "persistent fused decode MoE (1..16 rows), producer / consumer warp pairs");
  m.attr("MAX_ROWS") = kMaxT;
  m.attr("FUSED_MAX_ROWS") = 16;
  m.attr("FUSED_CTL_INTS") = pmw::kCtlInts;
}
