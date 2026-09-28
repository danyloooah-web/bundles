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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("up", &up, "decode MoE up projection + SwiGLU over the route's block list (routed INT8 + shared bf16), bit for bit i8x_dec_up");
  m.attr("MAX_ROWS") = kMaxT;
}
