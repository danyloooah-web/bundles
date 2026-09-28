// The decode MoE down projection (i8x_decode.i8x_dec_down's work) for the tri8 decode path, bit for bit the king's
// Triton kernel:
//   routed slot s of expert e, output column n:  acc = fma(d_3, c_3, fma(d_2, c_2, fma(d_1, c_1, fma(d_0, c_0, 0))))
//     over the four 128-deep scale groups, d_g = one chain of eight m16n8k16 (bf16, fp32) steps from zero with
//     Triton's kWidth-4 k order (32-deep chunk j: step 2j takes k = 32j + 4q + {0, 1} and 32j + 16 + 4q + {0, 1},
//     step 2j + 1 the same plus 2), c_g = the row's fp16 group scale; cache[s, n] = bf16(acc * wts[s]);
//   shared expert (bf16 s2): one continuous chain over K = 512 in the natural k order; cache = bf16(acc).
// The products are computed transposed (weight rows as the m16 operand, tokens as the n8 operand): every output is the
// same k-slot dot product, so the bits are the Triton kernel's. A CTA takes (route block: an expert and at most 16 of
// its tokens, 128 columns) or (shared expert, 16-token group, 128 columns); each of its warps streams its own 32
// weight rows through a private cp.async ring of 2 KB stages (16 rows x 128 k INT8, or x 64 k bf16) and
// synchronises only itself, so the weight stream never waits on a CTA barrier. The ring's first stages, the group
// scales and the routing slots are issued before the dependency wait (weights, scales and routing lists are final
// once the up kernel is running); the tokens' h rows are staged after it. Routed CTAs follow the route's 16-token
// block list (grid sized for the most blocks M rows can reach).
// Used for decode row batches of 6..48 (measured: faster there, slower below and above).
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>


namespace {
constexpr int kH = 2048, kI = 512, kE = 256, kTopK = 8, kMaxT = 256, kGroup = 128, kGroups = kI / kGroup;
#ifndef DN_WARPS
#define DN_WARPS 4
#endif
#ifndef DN_STAGES
#define DN_STAGES 4
#endif
#ifndef DN_COLS
#define DN_COLS 128
#endif
constexpr int kWarps = DN_WARPS, kThreads = 32 * kWarps, kStages = DN_STAGES, kTokCap = 16, kMaxRows = 48;
constexpr int kCols = DN_COLS, kChunks = kH / kCols;         // CTA: (expert or shared token group, kCols columns)
static_assert(kCols % (16 * kWarps) == 0, "whole m16 row tiles per warp");
constexpr int kTilesPerWarp = kCols / 16 / kWarps;           // m16 row tiles per warp
constexpr int kStageRowB = 128, kStagePad = kStageRowB + 16; // a stage row: 128 INT8 k or 64 bf16 k
constexpr int kStageBytes = 16 * kStagePad;
constexpr int kRtStagesPerTile = kI / 128, kShStagesPerTile = kI * 2 / kStageRowB;
constexpr int kAPad = kI * 2 + 32;
constexpr int kSmemRing = kWarps * kStages * kStageBytes, kSmemA = kTokCap * kAPad, kSmemSc = kCols * kGroups * 2;
constexpr int kSmemBytes = kSmemRing + kSmemA + kSmemSc + kTokCap * 8;

// plain shared-memory loads the compiler may schedule (ordered by the memory-clobbering waits and warp syncs)
extern __shared__ __align__(128) uint8_t smem[];
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {  // dst: offset into smem
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "l"(src) : "memory");
}
// streamed weights only one pass reads (a routed expert with <= 16 tokens, the shared expert at <= 16 rows) go through
// L2 as evict_first: the layer's small reused buffers (h, lists, cache rows) stay resident
#ifndef DN_EVF
#define DN_EVF 1
#endif
__device__ __forceinline__ void cp16_ef(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst + static_cast<uint32_t>(__cvta_generic_to_shared(smem))),
               "l"(src), "l"(pol) : "memory");
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
               : "r"(off + static_cast<uint32_t>(__cvta_generic_to_shared(smem))) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  return *reinterpret_cast<const uint32_t*>(smem + addr);
}
__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  return *reinterpret_cast<const uint2*>(smem + addr);
}
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

struct DnArgs {
  const __nv_bfloat16* h;  // [slots, 512]
  const uint8_t* q8;       // i8x blocks
  const __nv_bfloat16* s2; // shared down [2048, 512]
  const float* wts;
  const int* counts;
  const int* slots;
  const int* nblocks;      // the route's 16-token block list: [0, n_tb) shared, then (expert, t0) in expert order
  const int* bexp;
  const int* bt0;
  __nv_bfloat16* cache;    // [slots, 2048]
  int64_t eb, q2_off, c2_off;
  int M, sh_ctas, n_tb;
};

__global__ void __launch_bounds__(kThreads) down_kernel(const DnArgs a) {
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const bool shared = b < a.sh_ctas;
  int e, col0, t_first, n_tok;
  if (shared) {
    e = kE, col0 = (b % kChunks) * kCols, t_first = (b / kChunks) * kTokCap;
    n_tok = a.M - t_first < kTokCap ? a.M - t_first : kTokCap;
  } else {
    // one CTA per (routed 16-token block, 128 columns); the block list is final before the wait (the route finished
    // before up's wait)
    const int blk = a.n_tb + (b - a.sh_ctas) / kChunks;
    if (blk >= *a.nblocks) return;
    e = a.bexp[blk], col0 = ((b - a.sh_ctas) % kChunks) * kCols, t_first = a.bt0[blk];
    n_tok = a.counts[e] - t_first < kTokCap ? a.counts[e] - t_first : kTokCap;
  }
  if (n_tok <= 0) return;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t ring = warp * kStages * kStageBytes;  // offsets into smem; cp.async adds sbase
  const uint32_t s_a = kSmemRing, s_sc = s_a + kSmemA;
  int* s_slot = reinterpret_cast<int*>(smem + kSmemRing + kSmemA + kSmemSc);
  float* s_wt = reinterpret_cast<float*>(s_slot + kTokCap);
  // this warp's rows: col0 + 16 * (warp * kTilesPerWarp + i) + r
  const int row_b = shared ? kI * 2 : kI, st_per_tile = shared ? kShStagesPerTile : kRtStagesPerTile;
  const int n_st = kTilesPerWarp * st_per_tile;
  const uint8_t* w = (shared ? reinterpret_cast<const uint8_t*>(a.s2) : a.q8 + e * a.eb + a.q2_off) +
                     static_cast<int64_t>(col0 + 16 * warp * kTilesPerWarp) * row_b;
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const bool ef = shared ? a.M <= kTokCap : a.counts[e] <= kTokCap;  // a single-block expert
  auto issue = [&](int st) {  // stage st: tile st / st_per_tile, k bytes (st % st_per_tile) * 128, 16 rows
    if (st < n_st) {
      const uint8_t* src = w + static_cast<int64_t>(st / st_per_tile) * 16 * row_b + (st % st_per_tile) * kStageRowB;
      const uint32_t dst = ring + (st % kStages) * kStageBytes;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int c = lane + 32 * i, r = c >> 3, k16 = c & 7;
        if (DN_EVF && ef) cp16_ef(dst + r * kStagePad + k16 * 16, src + static_cast<int64_t>(r) * row_b + k16 * 16, pol);
        else cp16(dst + r * kStagePad + k16 * 16, src + static_cast<int64_t>(r) * row_b + k16 * 16);
      }
    }
    commit();
  };
  if (!shared) {  // group scales of the 256 rows: [row][4] fp16, 2 KB contiguous (their own commit group)
    const uint8_t* sc = a.q8 + e * a.eb + a.c2_off + static_cast<int64_t>(col0) * kGroups * 2;
    for (int i = tid; i < kSmemSc / 16; i += kThreads) cp16(s_sc + i * 16, sc + i * 16);
  }
  commit();
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) issue(st);
  // the routing lists are final before the wait (the route finished before up's wait): stage this batch's slots
  for (int i = tid; i < (n_tok < kTokCap ? n_tok : kTokCap); i += kThreads) {
    const int s = shared ? a.M * kTopK + t_first + i : a.slots[e * kMaxT + t_first + i];
    s_slot[i] = s;
    s_wt[i] = shared ? 1.f : a.wts[s];
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  uint16_t* out = reinterpret_cast<uint16_t*>(a.cache);
  for (int t0 = 0; t0 < n_tok; t0 += kTokCap) {
    const int nb = n_tok - t0 < kTokCap ? n_tok - t0 : kTokCap;
    if (t0 > 0) {  // another token batch: every warp's ring restarts on its first stages
      __syncthreads();
#pragma unroll
      for (int st = 0; st < kStages - 1; ++st) issue(st);
      for (int i = tid; i < nb; i += kThreads) {
        const int s = shared ? a.M * kTopK + t_first + i : a.slots[e * kMaxT + t0 + i];
        s_slot[i] = s;
        s_wt[i] = shared ? 1.f : a.wts[s];
      }
    }
    __syncthreads();
    for (int i = tid; i < nb * (kI * 2 / 16); i += kThreads) {
      const int r = i >> 6, c = i & 63;
      cp16(s_a + r * kAPad + c * 16, reinterpret_cast<const uint8_t*>(a.h) + static_cast<int64_t>(s_slot[r]) * kI * 2 + c * 16);
    }
    commit();
    wait_groups<0>();
    __syncthreads();
    const int n_tt = (nb + 7) >> 3;
    float acc[2][4];
    for (int st = 0; st < n_st; ++st) {
      issue(st + kStages - 1);
      wait_groups<kStages - 1>();
      __syncwarp();
      const int tile = st / st_per_tile, ks = st % st_per_tile;
      const uint32_t w0 = ring + (st % kStages) * kStageBytes + g * kStagePad + 4 * q, w1 = w0 + 8 * kStagePad;
      // ldmatrix row address of this lane: matrix m = lane / 8 -> rows (lane & 7) + 8 (m & 1), bytes + 16 (m >> 1)
      const uint32_t wm = ring + (st % kStages) * kStageBytes + ((lane & 7) + 8 * ((lane >> 3) & 1)) * kStagePad + 16 * (lane >> 4);
      if (ks == 0) {
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
      }
      if (shared) {
        // natural k order, one chain over K = 512 continued across the tile's eight 64-k stages
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          if (tt < n_tt) {
            const uint32_t arow = s_a + (8 * tt + g) * kAPad + 4 * q + ks * kStageRowB;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              const int kb = 32 * s;
              uint32_t r[4];
              ldsm4(r, wm + kb);
              mma16816(acc[tt], r[0], r[1], r[2], r[3], lds32(arow + kb), lds32(arow + kb + 16));
            }
          }
        }
      } else {
        const int lrow = 16 * (warp * kTilesPerWarp + tile) + g;
        const uint32_t v0 = lds32(s_sc + lrow * kGroups * 2 + (ks >> 1) * 4), v1 = lds32(s_sc + (lrow + 8) * kGroups * 2 + (ks >> 1) * 4);
        const __half2 h0 = *reinterpret_cast<const __half2*>(&v0), h1 = *reinterpret_cast<const __half2*>(&v1);
        const float sc0 = (ks & 1) ? __high2float(h0) : __low2float(h0), sc1 = (ks & 1) ? __high2float(h1) : __low2float(h1);
        uint32_t lo[4][4], hi[4][4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          uint32_t r[4];  // W[g][32jj + 4q], W[g + 8][..], W[g][32jj + 16 + 4q], W[g + 8][..]
          ldsm4(r, wm + 32 * jj);
#pragma unroll
          for (int i = 0; i < 4; ++i) i8x4_bf16x2(r[i], lo[jj][i], hi[jj][i]);
        }
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          if (tt < n_tt) {
            const uint32_t arow = s_a + (8 * tt + g) * kAPad + 8 * q + ks * kGroup * 2;
            float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
              const uint2 x0 = lds64(arow + 64 * jj), x1 = lds64(arow + 64 * jj + 32);
              mma16816(d, lo[jj][0], lo[jj][1], lo[jj][2], lo[jj][3], x0.x, x1.x);
              mma16816(d, hi[jj][0], hi[jj][1], hi[jj][2], hi[jj][3], x0.y, x1.y);
            }
            // D: (row g, tok 2q), (g, 2q + 1), (g + 8, 2q), (g + 8, 2q + 1)
            acc[tt][0] = ffma(d[0], sc0, acc[tt][0]);
            acc[tt][1] = ffma(d[1], sc0, acc[tt][1]);
            acc[tt][2] = ffma(d[2], sc1, acc[tt][2]);
            acc[tt][3] = ffma(d[3], sc1, acc[tt][3]);
          }
        }
      }
      if (ks == st_per_tile - 1) {
        const int col = col0 + 16 * (warp * kTilesPerWarp + tile) + g;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = 8 * tt + 2 * q;
          if (t < nb) {
            const float wv = s_wt[t];
            const int64_t o = static_cast<int64_t>(s_slot[t]) * kH + col;
            out[o] = bf16_bits(shared ? acc[tt][0] : fmul(acc[tt][0], wv));
            out[o + 8] = bf16_bits(shared ? acc[tt][2] : fmul(acc[tt][2], wv));
          }
          if (t + 1 < nb) {
            const float wv = s_wt[t + 1];
            const int64_t o = static_cast<int64_t>(s_slot[t + 1]) * kH + col;
            out[o] = bf16_bits(shared ? acc[tt][1] : fmul(acc[tt][1], wv));
            out[o + 8] = bf16_bits(shared ? acc[tt][3] : fmul(acc[tt][3], wv));
          }
        }
      }
      __syncwarp();  // this ring slot is refilled by the next stage's issue
    }
    wait_groups<0>();
  }
}
}  // namespace

void down(torch::Tensor h, torch::Tensor q8, torch::Tensor s2, torch::Tensor wts, torch::Tensor counts, torch::Tensor slots,
          torch::Tensor nblocks, torch::Tensor bexp, torch::Tensor bt0, torch::Tensor cache, int64_t eb, int64_t q2_off,
          int64_t c2_off, int64_t M) {
  TORCH_CHECK(M >= 1 && M <= kMaxRows, "the native decode down takes 1..48 rows, got ", M);
  TORCH_CHECK(h.is_cuda() && h.scalar_type() == at::kBFloat16 && h.is_contiguous() && h.dim() == 2 && h.size(1) == kI &&
                  h.size(0) >= M * (kTopK + 1), "h [>= M * 9, 512] bf16");
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.is_contiguous() && cache.dim() == 2 && cache.size(1) == kH &&
                  cache.size(0) >= M * (kTopK + 1), "cache [>= M * 9, 2048] bf16");
  TORCH_CHECK(s2.scalar_type() == at::kBFloat16 && s2.is_contiguous() && s2.dim() == 2 && s2.size(0) == kH &&
                  s2.size(1) == kI, "s2 [2048, 512] bf16");
  TORCH_CHECK(q8.is_contiguous() && q8.dim() == 2 && q8.size(0) == kE && q8.size(1) == eb && q8.element_size() == 1,
              "q8 [256, EB] bytes");
  TORCH_CHECK(wts.scalar_type() == at::kFloat && counts.scalar_type() == at::kInt && slots.scalar_type() == at::kInt &&
                  counts.numel() >= kE && slots.numel() >= kE * kMaxT, "routing lists");
  const at::cuda::CUDAGuard guard(h.device());
  TORCH_CHECK(nblocks.scalar_type() == at::kInt && bexp.scalar_type() == at::kInt && bt0.scalar_type() == at::kInt,
              "block list int32");
  const int n_tb = static_cast<int>((M + kTokCap - 1) / kTokCap);  // the route's shared blocks (16 tokens each)
  const int sh_ctas = kChunks * n_tb;
  // routed blocks: at most one per active expert plus one per further 16 tokens of any expert (i8x_decode's blocks_n)
  const int rt_blocks = static_cast<int>(std::min<int64_t>(kE, M * kTopK) + M * kTopK / kTokCap);
  TORCH_CHECK(bexp.numel() >= n_tb + rt_blocks && bt0.numel() >= n_tb + rt_blocks, "block list capacity");
  DnArgs a{reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()), reinterpret_cast<const uint8_t*>(q8.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(s2.data_ptr()), wts.data_ptr<float>(), counts.data_ptr<int>(),
           slots.data_ptr<int>(), nblocks.data_ptr<int>(), bexp.data_ptr<int>(), bt0.data_ptr<int>(),
           reinterpret_cast<__nv_bfloat16*>(cache.data_ptr()), eb, q2_off, c2_off, static_cast<int>(M), sh_ctas, n_tb};
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(sh_ctas + rt_blocks * kChunks);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, down_kernel, a));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("down", &down, "decode MoE down projection (routed INT8 + shared bf16), bit for bit i8x_dec_down");
  m.attr("MAX_ROWS") = kMaxRows;
}
