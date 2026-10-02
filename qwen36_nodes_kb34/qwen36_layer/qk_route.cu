// The decode MoE's routing for a row batch (Qwen3.6-35B-A3B, TP1) in one launch: router gate, softmax top-8 and the
// per-expert token and block lists that i8x_dec_up / i8x_dec_down read. Bit for bit the king's Triton routes:
//   * gate: each 16-token block x 16-expert tile x 512-deep K quarter is one chain of mma.sync.m16n8k16 (bf16, fp32)
//     steps in increasing k from zero, and logits = (((0 + p0) + p1) + p2) + p3 over the four quarters;
//   * top-8 (i8x_decode._top8): x = bf16(logits), e = ex2.approx((x - max) * log2 e), prob = div.full(e, sum), ids by
//     max value then min id, total over the eight weights, div.full(w, total). The sums' associations are the ones
//     Triton's runtime layouts compile to, which differ between the fused route (rows > 16) and the 'an' route's
//     i8x_dec_topk (rows <= 16): see col_of and the weight total; `fused` picks which one to reproduce;
//   * lists: counts / tokens / slots (the order inside an expert's list is free: every consumer indexes by slot),
//     then the block list with the shared expert's n_tb blocks first, as the fused route writes it. The shared
//     expert itself is left to i8x_dec_up (SHARED_DONE false), which computes the same bits.
// The last gate CTA of a 16-token block runs that block's top-8, one warp per row: each lane sorts its eight 24-bit
// keys (bf16 value order, -0 = +0, then 255 - id) once, so each pick is one redux.sync of the lanes' heads.
#include <cstdint>
#include <type_traits>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kH = 2048, kE = 256, kTopK = 8, kMaxT = 256, kBT = 16, kSplit = 4, kKPart = kH / kSplit;
constexpr int kThreads = 512;              // the ticket winner runs one warp per row of its 16-token block
constexpr int kChunks = kKPart / 8;        // 16-byte chunks per smem row
constexpr int kStages = 4;                 // cp.async groups over the K quarter
constexpr int kPreStages = kStages;        // router column groups issued before griddepcontrol.wait (static rows)
template <int kEPC>
constexpr uint32_t smem_bytes() { return (kBT + kEPC) * kKPart * 2; }
constexpr uint32_t kNegInfKey = 0x007Fu;   // monotone key of -inf's bf16 bits

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ float fadd(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fdiv_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float exp_tri(float x, float mx) {  // Triton's tl.exp(x - mx): sub, mul log2 e, ex2.approx
  float t, e;
  asm("sub.f32 %0, %1, %2;" : "=f"(t) : "f"(x), "f"(mx));
  asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(t) : "f"(t));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  return e;
}
__device__ __forceinline__ float bf_round(float v) {
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(v));
  return __uint_as_float(static_cast<uint32_t>(h) << 16);
}
__device__ __forceinline__ uint32_t key16(float x) {  // bf16-rounded x -> monotone 16-bit order; -0 = +0, NaN lowest
  const uint32_t b = __float_as_uint(fadd(x, 0.f));
  const uint32_t k = (b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u)) >> 16;
  return x != x ? 0u : k;
}
__device__ __forceinline__ float key_value(uint32_t key) {  // a key's bf16 value (key16 inverted; -0 comes back +0)
  const uint32_t k = key >> 8;
  return __uint_as_float(k & 0x8000u ? (k ^ 0x8000u) << 16 : (~k & 0xFFFFu) << 16);
}
__device__ __forceinline__ uint32_t redux_max(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ int ld_relaxed(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float bfly_sum(float v) {  // xor 16, 8, 4, 2, 1 as Triton's warp reduce
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v = fadd(v, __shfl_xor_sync(0xffffffffu, v, m));
  return v;
}

// Column ownership of Triton's runtime layouts (16-byte aligned pointers vectorize them): the fused route's 16 x 256
// tile gives lane l columns 8l .. 8l + 7; i8x_dec_topk's 1 x 256 row gives lane l columns 4l .. 4l + 3 and
// 128 + 4l .. 128 + 4l + 3. Each lane sums its eight in that order, then the butterfly.
template <bool kFused>
__device__ __forceinline__ int col_of(int lane, int j) {
  return kFused ? 8 * lane + j : (j < 4 ? 4 * lane + j : 128 + 4 * lane + (j - 4));
}
template <bool kFused>
__device__ __forceinline__ int owner_of(int c) {
  return kFused ? c >> 3 : (c & 127) >> 2;
}
template <bool kFused>
__device__ __forceinline__ int slot_of(int c) {
  return kFused ? c & 7 : ((c >> 7) << 2) | (c & 3);
}

// One warp: row t's top-8 from the four gate partials.
template <bool kFused>
__device__ __forceinline__ void top8_row(const float* __restrict__ part, int t, int lane, int& id_out, float& w_out) {
  float x[8];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int c0 = col_of<kFused>(lane, 4 * h);
    float4 v = __ldcg(reinterpret_cast<const float4*>(part + static_cast<int64_t>(t) * kE + c0));
    float l0 = fadd(0.f, v.x), l1 = fadd(0.f, v.y), l2 = fadd(0.f, v.z), l3 = fadd(0.f, v.w);
#pragma unroll
    for (int s = 1; s < kSplit; ++s) {
      v = __ldcg(reinterpret_cast<const float4*>(part + (static_cast<int64_t>(s) * kMaxT + t) * kE + c0));
      l0 = fadd(l0, v.x);
      l1 = fadd(l1, v.y);
      l2 = fadd(l2, v.z);
      l3 = fadd(l3, v.w);
    }
    x[4 * h] = bf_round(l0);
    x[4 * h + 1] = bf_round(l1);
    x[4 * h + 2] = bf_round(l2);
    x[4 * h + 3] = bf_round(l3);
  }
  float e[8];
  uint32_t key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) key[j] = (key16(x[j]) << 8) | static_cast<uint32_t>(255 - col_of<kFused>(lane, j));
  // sort the lane's keys descending (19-comparator network): each pick is then one redux of the heads
#define QK_CS(i, j) { const uint32_t hi = max(key[i], key[j]), lo = min(key[i], key[j]); key[i] = hi; key[j] = lo; }
  QK_CS(0, 2) QK_CS(1, 3) QK_CS(4, 6) QK_CS(5, 7)
  QK_CS(0, 4) QK_CS(1, 5) QK_CS(2, 6) QK_CS(3, 7)
  QK_CS(0, 1) QK_CS(2, 3) QK_CS(4, 5) QK_CS(6, 7)
  QK_CS(2, 4) QK_CS(3, 5)
  QK_CS(1, 4) QK_CS(3, 6)
  QK_CS(1, 2) QK_CS(3, 4) QK_CS(5, 6)
#undef QK_CS
  uint32_t picked[kTopK];
  picked[0] = redux_max(key[0]);
  const float mx = key_value(picked[0]);  // the row max (+0 / -0 give the same exponentials)
#pragma unroll
  for (int j = 0; j < 8; ++j) e[j] = exp_tri(x[j], mx);
  float sum = e[0];
#pragma unroll
  for (int j = 1; j < 8; ++j) sum = fadd(sum, e[j]);
  sum = bfly_sum(sum);
  // the winner's lane drops its head (Triton's cur = -inf at the pick: a -inf key only matters for rows with fewer
  // than eight finite logits, where it stays at the tail)
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    if (k > 0) picked[k] = redux_max(key[0]);
    if (key[0] == picked[k]) {
#pragma unroll
      for (int j = 0; j < 7; ++j) key[j] = key[j + 1];
      key[7] = (kNegInfKey << 8) | (picked[k] & 255u);
    }
  }
  // lane k: pick k's probability from its key (every lane holds every pick's key), then the weight total
  uint32_t mine = picked[0];
#pragma unroll
  for (int k = 1; k < kTopK; ++k)
    if (lane == k) mine = picked[k];
  const float wk = fdiv_full(exp_tri(key_value(mine), mx), sum);
  float w[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) w[k] = __shfl_sync(0xffffffffu, wk, k);
  float tv;
  if (kFused) {  // every lane holds all eight: in order
    tv = w[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) tv = fadd(tv, w[k]);
  } else {  // lane halves 0..3 and 4..7 in order, then xor 1
    tv = fadd(fadd(fadd(w[0], w[1]), w[2]), w[3]);
    tv = fadd(tv, fadd(fadd(fadd(w[4], w[5]), w[6]), w[7]));
  }
  w_out = fdiv_full(wk, tv > 0.f ? tv : 1.f);
  id_out = 255 - static_cast<int>(mine & 255u);
}

struct Args {
  const __nv_bfloat16* x;
  const __nv_bfloat16* rw;
  const int64_t* pos;
  float* part;
  int* tick;
  int* ids;
  float* wts;
  int* counts;
  int* tokens;
  int* slots;
  int* nblocks;
  int* bexp;
  int* bt0;
  int M;
};

// grid (n_tb, 4 * 256 / kEPC): CTA (tb, j) computes experts (j / 4) * kEPC.. (kEPC / 8 warps, one n8 tile each) over K
// quarter j % 4; 16 experts per CTA at <= 16 rows (latency), 32 above (half the CTAs re-reading the token rows)
template <bool kFused, int kEPC>
__global__ void __launch_bounds__(kThreads) route_kernel(const Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_flag;
  __shared__ int s_scan[kThreads / 32];
  __shared__ int s_counts[kE];
  const int tb = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n_tb = gridDim.x;
  const int e_base = (j / kSplit) * kEPC, split = j % kSplit, k_base = split * kKPart;
  const uint32_t xs = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t wsm = xs + kBT * kKPart * 2;
  constexpr int per = kChunks / kStages, pre = kPreStages * per;
  // this warp's row of the top-8 below (kThreads / 32 == kBT): whether it is a real token (padded graph rows carry
  // position 0). The positions are the step's input, final before this launch: read before the wait.
  static_assert(kThreads / 32 == kBT, "one top-8 warp per row");
  const int t_row = tb * kBT + warp;
  const bool row_live = t_row < a.M && __ldg(a.pos + t_row) != 0;
  // the router rows are static: their first kPreStages column groups go out before the wait, one group older than
  // every stage group below, under the previous kernel's tail (x is that kernel's output: only after the wait)
  for (int i = tid; i < kEPC * pre; i += kThreads) {
    const int re = i / pre, c = i % pre;
    cp16(wsm + re * kKPart * 2 + ((c ^ (re & 7)) << 4), a.rw + static_cast<int64_t>(e_base + re) * kH + k_base + c * 8,
         true);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  // stage x rows tb*16.. (zero past M) and the router columns left, K quarter `split`, in kStages column groups
#pragma unroll
  for (int g = 0; g < kStages; ++g) {
    for (int i = tid; i < (kBT + (g < kPreStages ? 0 : kEPC)) * per; i += kThreads) {
      const int r = i / per, c = g * per + i % per;
      if (r < kBT) {
        const int t = tb * kBT + r;
        const bool v = t < a.M;
        cp16(xs + r * kKPart * 2 + ((c ^ (r & 7)) << 4), a.x + static_cast<int64_t>(v ? t : 0) * kH + k_base + c * 8, v);
      } else {
        const int re = r - kBT;
        cp16(wsm + re * kKPart * 2 + ((c ^ (re & 7)) << 4), a.rw + static_cast<int64_t>(e_base + re) * kH + k_base + c * 8,
             true);
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  // cp.async.wait_group covers only the calling thread's copies: every stage ends in a full barrier
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  const bool mma_warp = warp < kEPC / 8;
  const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, ahi = lane >> 4;
  const int brow = warp * 8 + (lane & 7), bsel = lane >> 3;
#pragma unroll
  for (int g = 0; g < kStages; ++g) {
    if (g == 0) asm volatile("cp.async.wait_group 3;" ::: "memory");
    if (g == 1) asm volatile("cp.async.wait_group 2;" ::: "memory");
    if (g == 2) asm volatile("cp.async.wait_group 1;" ::: "memory");
    if (g == 3) asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (mma_warp) {
#pragma unroll
      for (int kp = 0; kp < kKPart / kStages / 32; ++kp) {  // pairs of k16 steps
        const int ks = g * (kKPart / kStages / 16) + 2 * kp;
        uint32_t fa0[4], fa1[4], fb[4];
        ldsm_x4(fa0, xs + arow * kKPart * 2 + (((2 * ks + ahi) ^ (arow & 7)) << 4));
        ldsm_x4(fa1, xs + arow * kKPart * 2 + (((2 * ks + 2 + ahi) ^ (arow & 7)) << 4));
        ldsm_x4(fb, wsm + brow * kKPart * 2 + (((2 * ks + bsel) ^ (brow & 7)) << 4));
        mma16816(d, fa0, fb[0], fb[1]);
        mma16816(d, fa1, fb[2], fb[3]);
      }
    }
  }
  if (mma_warp) {
    const int g8 = lane >> 2, q = lane & 3;
    float* p = a.part + (static_cast<int64_t>(split) * kMaxT + tb * kBT + g8) * kE + e_base + warp * 8 + 2 * q;
    *reinterpret_cast<float2*>(p) = make_float2(d[0], d[1]);
    *reinterpret_cast<float2*>(p + 8 * kE) = make_float2(d[2], d[3]);
  }
  // arrival: the barrier orders the CTA's partial stores before thread 0's release (the CUTLASS semaphore pattern)
  __syncthreads();
  if (tid == 0) s_flag = atom_add_acq_rel(a.tick + tb, 1) == kSplit * kE / kEPC - 1;
  __syncthreads();
  if (!s_flag) return;
  if (tid == 0) a.tick[tb] = 0;
  const bool single = n_tb == 1;  // the block winner is also the last: lists from shared memory, no second ticket
  if (single && tid < kE) s_counts[tid] = 0;
  if (single) __syncthreads();
  for (int r = warp; r < kBT; r += kThreads / 32) {
    const int t = tb * kBT + r;
    if (t < a.M) {
      int id;
      float w;
      top8_row<kFused>(a.part, t, lane, id, w);
      if (lane < kTopK) {
        const int slot = t * kTopK + lane;
        a.ids[slot] = id;
        a.wts[slot] = w;
        if (row_live) {  // r == warp: this warp's row
          const int where = single ? atomicAdd(s_counts + id, 1) : atomicAdd(a.counts + id, 1);
          a.tokens[id * kMaxT + where] = t;
          a.slots[id * kMaxT + where] = slot;
        }
      }
    }
  }
  __syncthreads();
  if (!single) {
    if (tid == 0) s_flag = atom_add_acq_rel(a.tick + kMaxT / kBT, 1) == n_tb - 1;
    __syncthreads();
    if (!s_flag) return;
    if (tid == 0) a.tick[kMaxT / kBT] = 0;
  }
  const int n_sh = n_tb;
  if (tid < n_sh) {
    a.bexp[tid] = kE;
    a.bt0[tid] = tid * kBT;
  }
  // one expert per thread: block counts, then a block-wide exclusive scan
  int cnt = 0;
  if (tid < kE) {
    if (single) {
      cnt = s_counts[tid];
      a.counts[tid] = cnt;
    } else {
      cnt = ld_relaxed(a.counts + tid);
    }
  }
  const int nb = (cnt + kBT - 1) / kBT;
  int incl = nb;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s_scan[warp] = incl;
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) {
    before += w < warp ? s_scan[w] : 0;
    total += s_scan[w];
  }
  const int base = n_sh + before + incl - nb;
  for (int b = 0; b < nb; ++b) {
    a.bexp[base + b] = tid;
    a.bt0[base + b] = b * kBT;
  }
  if (tid == 0) *a.nblocks = n_sh + total;
}

// kb27: top8_row's arithmetic from the bf16-rounded logits x (fused layout: lane l owns columns 8 l .. 8 l + 7);
// a copy so that route_kernel's own code stays as it was.
__device__ __forceinline__ void top8_x48(const float (&x)[8], int lane, int& id_out, float& w_out) {
  constexpr bool kFused = true;
  float e[8];
  uint32_t key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) key[j] = (key16(x[j]) << 8) | static_cast<uint32_t>(255 - col_of<kFused>(lane, j));
  // sort the lane's keys descending (19-comparator network): each pick is then one redux of the heads
#define QK_CS(i, j) { const uint32_t hi = max(key[i], key[j]), lo = min(key[i], key[j]); key[i] = hi; key[j] = lo; }
  QK_CS(0, 2) QK_CS(1, 3) QK_CS(4, 6) QK_CS(5, 7)
  QK_CS(0, 4) QK_CS(1, 5) QK_CS(2, 6) QK_CS(3, 7)
  QK_CS(0, 1) QK_CS(2, 3) QK_CS(4, 5) QK_CS(6, 7)
  QK_CS(2, 4) QK_CS(3, 5)
  QK_CS(1, 4) QK_CS(3, 6)
  QK_CS(1, 2) QK_CS(3, 4) QK_CS(5, 6)
#undef QK_CS
  uint32_t picked[kTopK];
  picked[0] = redux_max(key[0]);
  const float mx = key_value(picked[0]);  // the row max (+0 / -0 give the same exponentials)
#pragma unroll
  for (int j = 0; j < 8; ++j) e[j] = exp_tri(x[j], mx);
  float sum = e[0];
#pragma unroll
  for (int j = 1; j < 8; ++j) sum = fadd(sum, e[j]);
  sum = bfly_sum(sum);
  // the winner's lane drops its head (Triton's cur = -inf at the pick: a -inf key only matters for rows with fewer
  // than eight finite logits, where it stays at the tail)
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    if (k > 0) picked[k] = redux_max(key[0]);
    if (key[0] == picked[k]) {
#pragma unroll
      for (int j = 0; j < 7; ++j) key[j] = key[j + 1];
      key[7] = (kNegInfKey << 8) | (picked[k] & 255u);
    }
  }
  // lane k: pick k's probability from its key (every lane holds every pick's key), then the weight total
  uint32_t mine = picked[0];
#pragma unroll
  for (int k = 1; k < kTopK; ++k)
    if (lane == k) mine = picked[k];
  const float wk = fdiv_full(exp_tri(key_value(mine), mx), sum);
  float w[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) w[k] = __shfl_sync(0xffffffffu, wk, k);
  float tv;
  if (kFused) {  // every lane holds all eight: in order
    tv = w[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) tv = fadd(tv, w[k]);
  } else {  // lane halves 0..3 and 4..7 in order, then xor 1
    tv = fadd(fadd(fadd(w[0], w[1]), w[2]), w[3]);
    tv = fadd(tv, fadd(fadd(fadd(w[4], w[5]), w[6]), w[7]));
  }
  w_out = fdiv_full(wk, tv > 0.f ? tv : 1.f);
  id_out = 255 - static_cast<int>(mine & 255u);
}

// ---- kb27: the decode route at > 16 rows, one cluster of 8 CTAs per 16-token block ----
// Bit for bit route_kernel<true, 32>. CTA c of a block's cluster runs the gate chains of K quarter c % 4 for experts
// 128 (c / 4) .. + 127: warp w owns one 16-token x n8-expert chain (experts 8 w ..), the same 32 mma.sync m16n8k16
// steps over the quarter in increasing k from zero, kept in shared memory. After a cluster barrier warps 0 / 1 run the
// top-8 of rows 2 c, 2 c + 1: lane l reads its columns 8 l .. 8 l + 7 of the four quarters over DSMEM and forms
// l = (((0 + p0) + p1) + p2) + p3 as top8_row sums the global partials, then bf16 and top8_x. No global partials and
// no per-block ticket (route_kernel: partial stores, a ticket, the winner's reload); the lists as route_kernel's
// (counts atomics, one grid ticket, the last CTA's block list).
constexpr int kRcl = 8, kR48E = kE / 2, kR48Threads = 256;  // 8 warps x two n8 chains
struct R48Args {
  Args r;  // route_kernel's arguments
  // leading fill CTAs (wnf, whole clusters) warm the first wk experts' up weights (bytes [e * web, e * web + wa) and
  // [e * web + wbo, + wbl) of the INT8 expert blocks) into L2, in i8x_dec_up's order
  const unsigned char* wq;
  long long web, wa, wbo, wbl;
  int wk, wnf;
  int* wdone;  // bumped by the CTA that finishes the lists: the fill CTAs stop issuing (never outlast the route)
};
constexpr uint32_t kR48X = kBT * kKPart * 2;    // 16 KB: the block's 16 token rows over this CTA's K quarter
constexpr uint32_t kR48W = kR48E * kKPart * 2;  // 128 KB: router rows of the CTA's expert half over the quarter
constexpr uint32_t kR48P = kBT * kR48E * 4;     // 8 KB: the chains [row][expert] fp32
constexpr uint32_t kR48Smem = kR48X + kR48W + kR48P;
__device__ __forceinline__ uint32_t r48_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void r48_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float4 r48_ld_peer(uint32_t addr, uint32_t cta) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(cta));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ra)
               : "memory");
  return v;
}
// kb27: real L2 warming (discarded TMA bulk copies through a two-slot shared-memory ring; prefetch.L2 is only a hint),
// one thread per fill CTA; chunk g of the expert-major list goes to fill CTA g % wnf (expert 0 first)
constexpr int kWChunk = 65536, kWDepth = 2;
__device__ __forceinline__ void r48_warm(uint32_t ring, const R48Args& w, int f) {
  const int64_t ca = (w.wa + kWChunk - 1) / kWChunk, cb = (w.wbl + kWChunk - 1) / kWChunk, cpe = ca + cb;
  const int64_t tot = cpe * w.wk;
  const uint32_t bars = ring + kWDepth * kWChunk;
  for (int sl = 0; sl < kWDepth; ++sl) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bars + 8 * sl) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  auto issue = [&](int64_t g, int sl) {
    const int64_t e = g / cpe, j = g % cpe;
    const int64_t off = j < ca ? e * w.web + j * kWChunk : e * w.web + w.wbo + (j - ca) * kWChunk;
    const int64_t lim = j < ca ? w.wa - j * kWChunk : w.wbl - (j - ca) * kWChunk;
    const uint32_t n = static_cast<uint32_t>(lim < kWChunk ? lim : kWChunk);
    const uint32_t bar = bars + 8 * sl;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(n) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(ring + sl * kWChunk), "l"(w.wq + off), "r"(n), "r"(bar) : "memory");
  };
  const int epoch = ld_relaxed(w.wdone);
  int64_t nxt = f;
  int issued = 0;
  for (int sl = 0; sl < kWDepth && nxt < tot; ++sl, nxt += w.wnf) issue(nxt, sl), ++issued;
  // chunk i of this CTA lands in slot i % depth, phase (i / depth) & 1; wait only for chunks actually issued
  for (int done = 0; done < issued; ++done) {
    const int sl = done % kWDepth;
    const uint32_t ph = static_cast<uint32_t>(done / kWDepth) & 1u;
    const int now = ld_relaxed(w.wdone);  // in flight under the wait below
    asm volatile("{\n.reg .pred P;\nWR_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WR_%=;\n}\n"
                 ::"r"(bars + 8 * sl), "r"(ph) : "memory");
    if (now != epoch) nxt = tot;  // the route is done: drain what is in flight, issue nothing more
    if (nxt < tot) {
      issue(nxt, sl);
      nxt += w.wnf;
      ++issued;
    }
  }
}

__global__ void __launch_bounds__(kR48Threads, 1) route48_kernel(const R48Args w) {
  const Args& a = w.r;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_flag;
  __shared__ int s_scan[kR48Threads / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (static_cast<int>(blockIdx.x) < w.wnf) {  // a fill CTA (whole clusters lead the grid): static weights, no wait
    asm volatile("griddepcontrol.launch_dependents;");
    if (tid == 0) r48_warm(static_cast<uint32_t>(__cvta_generic_to_shared(smem)), w, static_cast<int>(blockIdx.x));
    return;
  }
  const int c = static_cast<int>(r48_rank()), tb = (static_cast<int>(blockIdx.x) - w.wnf) / kRcl;
  const int q = c % kSplit, e_base = (c / kSplit) * kR48E, k_base = q * kKPart;
  const uint32_t xs = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t wsm = xs + kR48X, psm = wsm + kR48W;
  // warps 0 / 1: the top-8 of rows 2 c + warp; real tokens only (padded graph rows carry position 0): before the wait
  const int rl = 2 * c + warp, t_row = tb * kBT + rl;
  const bool top_warp = warp < 2 && t_row < a.M;
  const bool row_live = top_warp && __ldg(a.pos + t_row) != 0;
  // the router rows are static: all of this CTA's go out before the wait
  for (int i = tid; i < kR48E * kChunks; i += kR48Threads) {
    const int re = i / kChunks, ch = i % kChunks;
    cp16(wsm + re * kKPart * 2 + ((ch ^ (re & 7)) << 4), a.rw + static_cast<int64_t>(e_base + re) * kH + k_base + ch * 8,
         true);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  for (int i = tid; i < kBT * kChunks; i += kR48Threads) {  // the block's token rows of the quarter (zero past M)
    const int r = i / kChunks, ch = i % kChunks, t = tb * kBT + r;
    const bool v = t < a.M;
    cp16(xs + r * kKPart * 2 + ((ch ^ (r & 7)) << 4), a.x + static_cast<int64_t>(v ? t : 0) * kH + k_base + ch * 8, v);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  {  // warp w: the chains of n8 tiles 2 w, 2 w + 1 (experts 16 w ..), sharing each step's A fragments
    float d[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, ahi = lane >> 4;
    const int bsel = lane >> 3;
#pragma unroll 4
    for (int kp = 0; kp < kKPart / 32; ++kp) {  // pairs of k16 steps, in increasing k
      const int ks = 2 * kp;
      uint32_t fa0[4], fa1[4], fb[2][4];
      ldsm_x4(fa0, xs + arow * kKPart * 2 + (((2 * ks + ahi) ^ (arow & 7)) << 4));
      ldsm_x4(fa1, xs + arow * kKPart * 2 + (((2 * ks + 2 + ahi) ^ (arow & 7)) << 4));
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int brow = (2 * warp + u) * 8 + (lane & 7);
        ldsm_x4(fb[u], wsm + brow * kKPart * 2 + (((2 * ks + bsel) ^ (brow & 7)) << 4));
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) mma16816(d[u], fa0, fb[u][0], fb[u][1]);
#pragma unroll
      for (int u = 0; u < 2; ++u) mma16816(d[u], fa1, fb[u][2], fb[u][3]);
    }
    const int g8 = lane >> 2, qd = lane & 3;
    float* p = reinterpret_cast<float*>(smem + kR48X + kR48W);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int col = (2 * warp + u) * 8 + 2 * qd;
      *reinterpret_cast<float2*>(p + g8 * kR48E + col) = make_float2(d[u][0], d[u][1]);
      *reinterpret_cast<float2*>(p + (g8 + 8) * kR48E + col) = make_float2(d[u][2], d[u][3]);
    }
  }
  r48_cluster_sync();  // every quarter's chains of the block are in its CTA's shared memory
  if (top_warp) {
    // lane l: experts 8 l .. 8 l + 7 = expert half l / 16, columns 8 (l % 16) .. of that half
    const int half = lane >> 4;
    const uint32_t off = psm + static_cast<uint32_t>((rl * kR48E + 8 * (lane & 15)) * 4);
    float l8[8];
#pragma unroll
    for (int qq = 0; qq < kSplit; ++qq) {
      const uint32_t peer = static_cast<uint32_t>(half * kSplit + qq);
      const float4 v0 = r48_ld_peer(off, peer), v1 = r48_ld_peer(off + 16, peer);
      const float pv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) l8[j] = qq == 0 ? fadd(0.f, pv[j]) : fadd(l8[j], pv[j]);
    }
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = bf_round(l8[j]);
    int id;
    float w;
    top8_x48(x, lane, id, w);
    if (lane < kTopK) {
      const int slot = t_row * kTopK + lane;
      a.ids[slot] = id;
      a.wts[slot] = w;
      if (row_live) {
        const int where = atomicAdd(a.counts + id, 1);
        a.tokens[id * kMaxT + where] = t_row;
        a.slots[id * kMaxT + where] = slot;
      }
    }
  }
  r48_cluster_sync();  // no CTA leaves while a peer may still read its chains
  if (tid == 0) s_flag = atom_add_acq_rel(a.tick + kMaxT / kBT, 1) == static_cast<int>(gridDim.x) - w.wnf - 1;
  __syncthreads();
  if (!s_flag) return;
  if (tid == 0) a.tick[kMaxT / kBT] = 0;
  // route_kernel's block lists: the shared expert's n_tb blocks first, then each expert's blocks in expert order
  const int n_sh = (static_cast<int>(gridDim.x) - w.wnf) / kRcl;
  if (tid < n_sh) {
    a.bexp[tid] = kE;
    a.bt0[tid] = tid * kBT;
  }
  const int cnt = tid < kE ? ld_relaxed(a.counts + tid) : 0;
  const int nb = (cnt + kBT - 1) / kBT;
  int incl = nb;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s_scan[warp] = incl;
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int w = 0; w < kR48Threads / 32; ++w) {
    before += w < warp ? s_scan[w] : 0;
    total += s_scan[w];
  }
  const int base = n_sh + before + incl - nb;
  for (int b = 0; b < nb; ++b) {
    a.bexp[base + b] = tid;
    a.bt0[base + b] = b * kBT;
  }
  if (tid == 0) {
    *a.nblocks = n_sh + total;
    if (w.wnf > 0) atomicAdd(w.wdone, 1);
  }
}

// ---- combine + the next layer's input norm (kcombnorm on the tri8 path) ----
// out[t] = bf16(fma(gate, shared, (((0 + c0) + c1) .. + c7))) as i8x_dec_combine (zero on padding rows), with
// gate = div.full(1, 1 + ex2.approx(-(x[t] . gate_w) * log2 e)) and the dot's association Triton's runtime layout gives
// (thread l + 32 w: fma chain over 256 w + 8 l + [1, 0, 2..7] then 1024 + 256 w + 8 l + [0..7], lane butterfly
// xor 16..1, then (w0 + w2) + (w1 + w3)); then flashinfer's CuTe-DSL Gemma fused add-RMSNorm of (out, residual) as
// qk_moe_decode.cu's kcombnorm reproduces it: h = f32(out) + f32(residual), r_next = bf16(h), lane l's sum of squares
// over 256 v1 + 8 l + v0 in order, butterfly xor 1..16, rstd = rsqrt.approx.ftz(s / 2048 + eps),
// y = bf16((h * rstd) * (w + 1)).
struct CnArgs {
  const __nv_bfloat16* cache;
  const __nv_bfloat16* x;
  const __nv_bfloat16* gate_w;
  const int64_t* pos;
  __nv_bfloat16* out;
  int* counts;
  int* nblocks;
  const __nv_bfloat16* resid;
  const __nv_bfloat16* norm_w;
  float eps;
  __nv_bfloat16* y;
  __nv_bfloat16* r_next;
  int M;
  int early;  // release the dependents at the start (the next layer's in_proj prefetches static weights until its wait)
};
// n30 (kb30nf): the > 16-row combine-norm's arguments: CnArgs plus the MoE's h rows (dropped from L2 with the cache rows)
struct CnArgsD : CnArgs {
  const __nv_bfloat16* h;
};
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(b[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
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
// One CTA per row, warp v1 owning columns 256 v1 + 8 l + [0, 8): warps 0..3 build the gate dot's four virtual-warp
// partials, then the slot sum; warp 0 then runs the norm's sum of squares over the row from shared memory. Only the
// cache rows and the lists' reset wait for the down: x, pos and resid were final once the route passed its wait
// (i8x_dec_up waits before it triggers) and load through L2 (ld.global.cg), the weights are static. The min-blocks
// bound lets ptxas keep the nine cache loads in flight together (without it they issue in dependent waves).
constexpr int kCnWarps = 8;
// n30 (kb30nf): DISC = false is the kernel above unchanged (<= 16 rows); DISC = true (> 16 rows) also drops the token's
// dead MoE rows (cache, h) from L2 once read.
template <bool DISC>
__global__ void __launch_bounds__(32 * kCnWarps, 1) combine_norm_kernel(const std::conditional_t<DISC, CnArgsD, CnArgs> a) {
  __shared__ float s_part[4];
  __shared__ float s_rstd;
  __shared__ __align__(16) float s_h[kH];
  if (a.early) asm volatile("griddepcontrol.launch_dependents;");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, t = blockIdx.x;
  const int64_t row = static_cast<int64_t>(t) * kH;
  const bool valid = __ldcg(a.pos + t) != 0;
  const int c = 256 * warp + 8 * lane;
  float r[8], wv[8];
  unpack8(__ldcg(reinterpret_cast<const uint4*>(a.resid + row + c)), r);
  unpack8(__ldg(reinterpret_cast<const uint4*>(a.norm_w + c)), wv);
  if (valid && warp < 4) {  // virtual warp `warp` of Triton's dot: columns 256 w + 8 l + [1, 0, 2..7], then 1024 + ...
    float h0[8], g0[8], h1[8], g1[8];
    unpack8(__ldcg(reinterpret_cast<const uint4*>(a.x + row + c)), h0);
    unpack8(__ldg(reinterpret_cast<const uint4*>(a.gate_w + c)), g0);
    unpack8(__ldcg(reinterpret_cast<const uint4*>(a.x + row + 1024 + c)), h1);
    unpack8(__ldg(reinterpret_cast<const uint4*>(a.gate_w + 1024 + c)), g1);
    float s = fmul(h0[1], g0[1]);
    s = ffma(h0[0], g0[0], s);
#pragma unroll
    for (int e = 2; e < 8; ++e) s = ffma(h0[e], g0[e], s);
#pragma unroll
    for (int e = 0; e < 8; ++e) s = ffma(h1[e], g1[e], s);
    s = bfly_sum(s);
    if (lane == 0) s_part[warp] = s;
  }
  __syncthreads();
  float gate;
  if (valid) {
    const float dot = fadd(fadd(s_part[0], s_part[2]), fadd(s_part[1], s_part[3]));
    float tt, ex;
    asm("sub.f32 %0, 0f00000000, %1;" : "=f"(tt) : "f"(dot));
    asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(tt) : "f"(tt));
    asm("ex2.approx.f32 %0, %1;" : "=f"(ex) : "f"(tt));
    gate = fdiv_full(1.f, fadd(ex, 1.f));
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (!a.early) asm volatile("griddepcontrol.launch_dependents;");
  if constexpr (DISC) {
    // n30: the MoE's h rows were last read by its down, which completed before the wait above: the token's nine h rows
    // (8 routed slots + the shared slot, 1 KB = 8 lines of 128 B each) are dead -- the next layer's MoE rewrites them
    // before any read. discard.global.L2 drops the lines from L2 without a write-back to HBM (no value changes).
    constexpr int kInter = 512, kHl = kInter * 2 / 128;
    if (valid && threadIdx.x < (kTopK + 1) * kHl) {
      const int r = threadIdx.x / kHl, ln = threadIdx.x % kHl;
      const int64_t slot = r < kTopK ? static_cast<int64_t>(t) * kTopK + r : static_cast<int64_t>(a.M) * kTopK + t;
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(reinterpret_cast<const char*>(a.h + slot * kInter) + ln * 128) : "memory");
    }
  }
  if (t == 0) {  // i8x_dec_combine's reset of the lists for the next call (the route relies on it)
    for (int e = threadIdx.x; e < kE; e += 32 * kCnWarps) a.counts[e] = 0;
    if (threadIdx.x == 0) *a.nblocks = 0;
  }
  float acc[8], sh[8];
  if (valid) {
    uint4 raw[kTopK + 1];
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
      raw[k] = __ldcg(reinterpret_cast<const uint4*>(a.cache + (static_cast<int64_t>(t) * kTopK + k) * kH + c));
    raw[kTopK] = __ldcg(reinterpret_cast<const uint4*>(a.cache + (static_cast<int64_t>(a.M) * kTopK + t) * kH + c));
    float v[8];
    unpack8(raw[0], v);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = fadd(0.f, v[e]);
#pragma unroll
    for (int k = 1; k < kTopK; ++k) {
      unpack8(raw[k], v);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = fadd(acc[e], v[e]);
    }
    unpack8(raw[kTopK], sh);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = ffma(gate, sh[e], acc[e]);
    if constexpr (DISC) {
      // n30: a 128-byte line of a cache row is read only by lanes 8 j .. 8 j + 7 of this warp (16 bytes each), and every
      // lane's loads are consumed above: after the warp sync the warp's 9 x 4 lines of the token's cache rows are dead
      // (the next layer's MoE rewrites them before any read) and leave L2 without a write-back.
      __syncwarp();
      for (int i = lane; i < (kTopK + 1) * 4; i += 32) {
        const int r = i >> 2, ln = 4 * warp + (i & 3);
        const int64_t slot = r < kTopK ? static_cast<int64_t>(t) * kTopK + r : static_cast<int64_t>(a.M) * kTopK + t;
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(reinterpret_cast<const char*>(a.cache + slot * kH) + ln * 128) : "memory");
      }
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  }
  __nv_bfloat162 ob[4], rb[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ob[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  *reinterpret_cast<uint4*>(a.out + row + c) = *reinterpret_cast<const uint4*>(ob);
  float o[8], h[8];
  unpack8(*reinterpret_cast<const uint4*>(ob), o);
#pragma unroll
  for (int e = 0; e < 8; ++e) h[e] = fadd(o[e], r[e]);
#pragma unroll
  for (int i = 0; i < 4; ++i) rb[i] = __floats2bfloat162_rn(h[2 * i], h[2 * i + 1]);
  *reinterpret_cast<uint4*>(a.r_next + row + c) = *reinterpret_cast<const uint4*>(rb);
  *reinterpret_cast<float4*>(s_h + c) = make_float4(h[0], h[1], h[2], h[3]);
  *reinterpret_cast<float4*>(s_h + c + 4) = make_float4(h[4], h[5], h[6], h[7]);
  __syncthreads();
  if (warp == 0) {  // the CuTe order: lane l's columns 256 v1 + 8 l + v0 in order, then xor 1 .. 16
    float ss = 0.f;
#pragma unroll
    for (int v1 = 0; v1 < 8; ++v1) {
      const float4 p = *reinterpret_cast<const float4*>(s_h + 256 * v1 + 8 * lane);
      const float4 q = *reinterpret_cast<const float4*>(s_h + 256 * v1 + 8 * lane + 4);
      ss = ffma(p.x, p.x, ss);
      ss = ffma(p.y, p.y, ss);
      ss = ffma(p.z, p.z, ss);
      ss = ffma(p.w, p.w, ss);
      ss = ffma(q.x, q.x, ss);
      ss = ffma(q.y, q.y, ss);
      ss = ffma(q.z, q.z, ss);
      ss = ffma(q.w, q.w, ss);
    }
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) ss = fadd(ss, __shfl_xor_sync(0xffffffffu, ss, off));
    if (lane == 0) {
      float rs;
      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs) : "f"(ffma(ss, 1.f / 2048.f, a.eps)));
      s_rstd = rs;
    }
  }
  __syncthreads();
  const float rstd = s_rstd;
  __nv_bfloat162 yb[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    yb[i] = __floats2bfloat162_rn(fmul(fmul(h[2 * i], rstd), fadd(wv[2 * i], 1.f)),
                                  fmul(fmul(h[2 * i + 1], rstd), fadd(wv[2 * i + 1], 1.f)));
  *reinterpret_cast<uint4*>(a.y + row + c) = *reinterpret_cast<const uint4*>(yb);
}
// ---- kb30m: the decode route at 1..16 rows (c6 verify) with predicted-expert L2 warming ----
// Bit for bit route_kernel<false, 16> (one 16-token block): the same gate chains, partial stores, per-block ticket,
// top-8 (top8_r16w: a copy of top8_row<false>) and lists, from its own argument struct and code, so route_kernel's
// source and SASS stay as they were. Its wnf leading fill CTAs read only static weights and this layer's history
// (written one decode step earlier), so they never wait: warp 0 picks the layer's wn most likely experts and issues
// L2 prefetches of their up rows and scales, in expert-id order (the fused MoE's routed task order), while the add-norm
// and the route leave HBM mostly idle; the fused MoE then finds them in L2. The history is an EMA of each expert's row
// count in this layer: the list builder writes score' = score - score / 4 + 256 * count (score loaded at its start).
// The fill CTAs write nothing.
struct RWArgs {
  Args r;                   // route_kernel's arguments
  const unsigned char* wq;  // the layer's i8x expert blocks [256, web] (nullptr: no warming)
  long long web;            // block stride
  long long wa;             // warm bytes [0, wa) of a picked expert's block ...
  long long wbo, wbl;       // ... and [wbo, wbo + wbl)
  int* hist;                // this layer's history: [0, 256) expert scores
  int wn;                   // experts to warm (the top wn scores > 0, <= 32)
  int wnf;                  // leading fill CTAs
  int wch;                  // bytes per L2 prefetch
};
constexpr int kRWMax = 32;

template <bool kFused>
__device__ __forceinline__ void top8_r16w(const float* __restrict__ part, int t, int lane, int& id_out, float& w_out) {
  float x[8];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int c0 = col_of<kFused>(lane, 4 * h);
    float4 v = __ldcg(reinterpret_cast<const float4*>(part + static_cast<int64_t>(t) * kE + c0));
    float l0 = fadd(0.f, v.x), l1 = fadd(0.f, v.y), l2 = fadd(0.f, v.z), l3 = fadd(0.f, v.w);
#pragma unroll
    for (int s = 1; s < kSplit; ++s) {
      v = __ldcg(reinterpret_cast<const float4*>(part + (static_cast<int64_t>(s) * kMaxT + t) * kE + c0));
      l0 = fadd(l0, v.x);
      l1 = fadd(l1, v.y);
      l2 = fadd(l2, v.z);
      l3 = fadd(l3, v.w);
    }
    x[4 * h] = bf_round(l0);
    x[4 * h + 1] = bf_round(l1);
    x[4 * h + 2] = bf_round(l2);
    x[4 * h + 3] = bf_round(l3);
  }
  float e[8];
  uint32_t key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) key[j] = (key16(x[j]) << 8) | static_cast<uint32_t>(255 - col_of<kFused>(lane, j));
  // sort the lane's keys descending (19-comparator network): each pick is then one redux of the heads
#define QK_CS(i, j) { const uint32_t hi = max(key[i], key[j]), lo = min(key[i], key[j]); key[i] = hi; key[j] = lo; }
  QK_CS(0, 2) QK_CS(1, 3) QK_CS(4, 6) QK_CS(5, 7)
  QK_CS(0, 4) QK_CS(1, 5) QK_CS(2, 6) QK_CS(3, 7)
  QK_CS(0, 1) QK_CS(2, 3) QK_CS(4, 5) QK_CS(6, 7)
  QK_CS(2, 4) QK_CS(3, 5)
  QK_CS(1, 4) QK_CS(3, 6)
  QK_CS(1, 2) QK_CS(3, 4) QK_CS(5, 6)
#undef QK_CS
  uint32_t picked[kTopK];
  picked[0] = redux_max(key[0]);
  const float mx = key_value(picked[0]);  // the row max (+0 / -0 give the same exponentials)
#pragma unroll
  for (int j = 0; j < 8; ++j) e[j] = exp_tri(x[j], mx);
  float sum = e[0];
#pragma unroll
  for (int j = 1; j < 8; ++j) sum = fadd(sum, e[j]);
  sum = bfly_sum(sum);
  // the winner's lane drops its head (Triton's cur = -inf at the pick: a -inf key only matters for rows with fewer
  // than eight finite logits, where it stays at the tail)
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    if (k > 0) picked[k] = redux_max(key[0]);
    if (key[0] == picked[k]) {
#pragma unroll
      for (int j = 0; j < 7; ++j) key[j] = key[j + 1];
      key[7] = (kNegInfKey << 8) | (picked[k] & 255u);
    }
  }
  // lane k: pick k's probability from its key (every lane holds every pick's key), then the weight total
  uint32_t mine = picked[0];
#pragma unroll
  for (int k = 1; k < kTopK; ++k)
    if (lane == k) mine = picked[k];
  const float wk = fdiv_full(exp_tri(key_value(mine), mx), sum);
  float w[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) w[k] = __shfl_sync(0xffffffffu, wk, k);
  float tv;
  if (kFused) {  // every lane holds all eight: in order
    tv = w[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) tv = fadd(tv, w[k]);
  } else {  // lane halves 0..3 and 4..7 in order, then xor 1
    tv = fadd(fadd(fadd(w[0], w[1]), w[2]), w[3]);
    tv = fadd(tv, fadd(fadd(fadd(w[4], w[5]), w[6]), w[7]));
  }
  w_out = fdiv_full(wk, tv > 0.f ? tv : 1.f);
  id_out = 255 - static_cast<int>(mine & 255u);
}

// A fill CTA's warp 0: lane l holds the keys of experts 8 l .. 8 l + 7 (score << 8 | 255 - id, 0 for a zero score),
// sorted descending per lane, so each of the wn picks is one redux of the lanes' heads (top8_row's trick); lane k keeps
// pick k; the picks are put in expert-id order, and the lanes issue fill CTA f's share of their pieces (piece g of the
// expert-major list -> fill CTA g % wnf) as cp.async.bulk.prefetch.L2.
__device__ __forceinline__ void rw_fill(const RWArgs& w, int* s_list, int f) {
  const int lane = threadIdx.x & 31;
  const int4 h0 = __ldcg(reinterpret_cast<const int4*>(w.hist) + 2 * lane);
  const int4 h1 = __ldcg(reinterpret_cast<const int4*>(w.hist) + 2 * lane + 1);
  const int sc[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
  uint32_t key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    key[j] = sc[j] > 0 ? (static_cast<uint32_t>(min(sc[j], (1 << 23) - 1)) << 8) | static_cast<uint32_t>(255 - (8 * lane + j)) : 0u;
#define QK_CS(i, j) { const uint32_t hi = max(key[i], key[j]), lo = min(key[i], key[j]); key[i] = hi; key[j] = lo; }
  QK_CS(0, 2) QK_CS(1, 3) QK_CS(4, 6) QK_CS(5, 7)
  QK_CS(0, 4) QK_CS(1, 5) QK_CS(2, 6) QK_CS(3, 7)
  QK_CS(0, 1) QK_CS(2, 3) QK_CS(4, 5) QK_CS(6, 7)
  QK_CS(2, 4) QK_CS(3, 5)
  QK_CS(1, 4) QK_CS(3, 6)
  QK_CS(1, 2) QK_CS(3, 4) QK_CS(5, 6)
#undef QK_CS
  uint32_t mine = 0;
  int total = 0;
  for (int k = 0; k < w.wn; ++k) {
    const uint32_t p = redux_max(key[0]);
    if (p == 0u) break;  // no positive score left
    if (lane == k) mine = p;
    if (key[0] == p) {
#pragma unroll
      for (int j = 0; j < 7; ++j) key[j] = key[j + 1];
      key[7] = 0u;
    }
    ++total;
  }
  if (total == 0) return;
  const int myid = 255 - static_cast<int>(mine & 255u);
  int pos = 0;
  for (int j = 0; j < total; ++j) pos += __shfl_sync(0xffffffffu, myid, j) < myid;
  if (lane < total) s_list[pos] = myid;
  __syncwarp();
  const int64_t ch = w.wch;
  const int64_t cpa = (w.wa + ch - 1) / ch, cpb = (w.wbl + ch - 1) / ch, cpe = cpa + cpb;
  const int64_t tot = cpe * total;
  for (int64_t g = f + static_cast<int64_t>(lane) * w.wnf; g < tot; g += static_cast<int64_t>(w.wnf) * 32) {
    const int64_t e = s_list[g / cpe], j = g % cpe;
    const int64_t off = j < cpa ? e * w.web + j * ch : e * w.web + w.wbo + (j - cpa) * ch;
    const int64_t lim = j < cpa ? w.wa - j * ch : w.wbl - (j - cpa) * ch;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(w.wq + off), "r"(static_cast<uint32_t>(lim < ch ? lim : ch))
                 : "memory");
  }
}

// grid wnf + 4 * 256 / 16: CTAs [0, wnf) fill; CTA wnf + j is route_kernel<false, 16>'s CTA (0, j)
__global__ void __launch_bounds__(kThreads) route16w_kernel(const RWArgs w) {
  constexpr bool kFused = false;
  constexpr int kEPC = 16;
  const Args& a = w.r;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_flag;
  __shared__ int s_scan[kThreads / 32];
  __shared__ int s_counts[kE];
  __shared__ int s_list[kRWMax];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (static_cast<int>(blockIdx.x) < w.wnf) {  // a fill CTA (they lead the grid): static weights, no wait
    asm volatile("griddepcontrol.launch_dependents;");
    if (warp == 0) rw_fill(w, s_list, static_cast<int>(blockIdx.x));
    return;
  }
  const int tb = 0, j = static_cast<int>(blockIdx.x) - w.wnf;
  const int n_tb = 1;
  const int e_base = (j / kSplit) * kEPC, split = j % kSplit, k_base = split * kKPart;
  const uint32_t xs = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t wsm = xs + kBT * kKPart * 2;
  constexpr int per = kChunks / kStages, pre = kPreStages * per;
  // the history score this thread's expert had (only the list builder uses it): loaded now, under everything below
  const int h_old = w.hist != nullptr && tid < kE ? __ldcg(w.hist + tid) : 0;
  // this warp's row of the top-8 below (kThreads / 32 == kBT): whether it is a real token (padded graph rows carry
  // position 0). The positions are the step's input, final before this launch: read before the wait.
  static_assert(kThreads / 32 == kBT, "one top-8 warp per row");
  const int t_row = tb * kBT + warp;
  const bool row_live = t_row < a.M && __ldg(a.pos + t_row) != 0;
  // the router rows are static: their first kPreStages column groups go out before the wait, one group older than
  // every stage group below, under the previous kernel's tail (x is that kernel's output: only after the wait)
  for (int i = tid; i < kEPC * pre; i += kThreads) {
    const int re = i / pre, c = i % pre;
    cp16(wsm + re * kKPart * 2 + ((c ^ (re & 7)) << 4), a.rw + static_cast<int64_t>(e_base + re) * kH + k_base + c * 8,
         true);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  // stage x rows tb*16.. (zero past M) and the router columns left, K quarter `split`, in kStages column groups
#pragma unroll
  for (int g = 0; g < kStages; ++g) {
    for (int i = tid; i < (kBT + (g < kPreStages ? 0 : kEPC)) * per; i += kThreads) {
      const int r = i / per, c = g * per + i % per;
      if (r < kBT) {
        const int t = tb * kBT + r;
        const bool v = t < a.M;
        cp16(xs + r * kKPart * 2 + ((c ^ (r & 7)) << 4), a.x + static_cast<int64_t>(v ? t : 0) * kH + k_base + c * 8, v);
      } else {
        const int re = r - kBT;
        cp16(wsm + re * kKPart * 2 + ((c ^ (re & 7)) << 4), a.rw + static_cast<int64_t>(e_base + re) * kH + k_base + c * 8,
             true);
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  // cp.async.wait_group covers only the calling thread's copies: every stage ends in a full barrier
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  const bool mma_warp = warp < kEPC / 8;
  const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, ahi = lane >> 4;
  const int brow = warp * 8 + (lane & 7), bsel = lane >> 3;
#pragma unroll
  for (int g = 0; g < kStages; ++g) {
    if (g == 0) asm volatile("cp.async.wait_group 3;" ::: "memory");
    if (g == 1) asm volatile("cp.async.wait_group 2;" ::: "memory");
    if (g == 2) asm volatile("cp.async.wait_group 1;" ::: "memory");
    if (g == 3) asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (mma_warp) {
#pragma unroll
      for (int kp = 0; kp < kKPart / kStages / 32; ++kp) {  // pairs of k16 steps
        const int ks = g * (kKPart / kStages / 16) + 2 * kp;
        uint32_t fa0[4], fa1[4], fb[4];
        ldsm_x4(fa0, xs + arow * kKPart * 2 + (((2 * ks + ahi) ^ (arow & 7)) << 4));
        ldsm_x4(fa1, xs + arow * kKPart * 2 + (((2 * ks + 2 + ahi) ^ (arow & 7)) << 4));
        ldsm_x4(fb, wsm + brow * kKPart * 2 + (((2 * ks + bsel) ^ (brow & 7)) << 4));
        mma16816(d, fa0, fb[0], fb[1]);
        mma16816(d, fa1, fb[2], fb[3]);
      }
    }
  }
  if (mma_warp) {
    const int g8 = lane >> 2, q = lane & 3;
    float* p = a.part + (static_cast<int64_t>(split) * kMaxT + tb * kBT + g8) * kE + e_base + warp * 8 + 2 * q;
    *reinterpret_cast<float2*>(p) = make_float2(d[0], d[1]);
    *reinterpret_cast<float2*>(p + 8 * kE) = make_float2(d[2], d[3]);
  }
  // arrival: the barrier orders the CTA's partial stores before thread 0's release (the CUTLASS semaphore pattern)
  __syncthreads();
  if (tid == 0) s_flag = atom_add_acq_rel(a.tick + tb, 1) == kSplit * kE / kEPC - 1;
  __syncthreads();
  if (!s_flag) return;
  if (tid == 0) a.tick[tb] = 0;
  // one 16-token block: the block winner is also the last, lists from shared memory, no second ticket
  if (tid < kE) s_counts[tid] = 0;
  __syncthreads();
  for (int r = warp; r < kBT; r += kThreads / 32) {
    const int t = tb * kBT + r;
    if (t < a.M) {
      int id;
      float wt;
      top8_r16w<kFused>(a.part, t, lane, id, wt);
      if (lane < kTopK) {
        const int slot = t * kTopK + lane;
        a.ids[slot] = id;
        a.wts[slot] = wt;
        if (row_live) {  // r == warp: this warp's row
          const int where = atomicAdd(s_counts + id, 1);
          a.tokens[id * kMaxT + where] = t;
          a.slots[id * kMaxT + where] = slot;
        }
      }
    }
  }
  __syncthreads();
  const int n_sh = n_tb;
  if (tid < n_sh) {
    a.bexp[tid] = kE;
    a.bt0[tid] = tid * kBT;
  }
  // one expert per thread: block counts, then a block-wide exclusive scan
  int cnt = 0;
  if (tid < kE) {
    cnt = s_counts[tid];
    a.counts[tid] = cnt;
    if (w.hist != nullptr) w.hist[tid] = h_old - (h_old >> 2) + (cnt << 8);
  }
  const int nb = (cnt + kBT - 1) / kBT;
  int incl = nb;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s_scan[warp] = incl;
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int i = 0; i < kThreads / 32; ++i) {
    before += i < warp ? s_scan[i] : 0;
    total += s_scan[i];
  }
  const int base = n_sh + before + incl - nb;
  for (int b = 0; b < nb; ++b) {
    a.bexp[base + b] = tid;
    a.bt0[base + b] = b * kBT;
  }
  if (tid == 0) *a.nblocks = n_sh + total;
}

}  // namespace

// fused = true reproduces i8x_dec_route's top-8 association (rows > 16), false i8x_dec_topk's ('an', rows <= 16)
void route(torch::Tensor x, torch::Tensor rw, torch::Tensor pos, torch::Tensor part, torch::Tensor tick, torch::Tensor ids,
           torch::Tensor wts, torch::Tensor counts, torch::Tensor tokens, torch::Tensor slots, torch::Tensor nblocks,
           torch::Tensor bexp, torch::Tensor bt0, bool fused) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == kH &&
                  M >= 1 && M <= kMaxT, "x [1..256, 2048] bf16");
  TORCH_CHECK(rw.scalar_type() == at::kBFloat16 && rw.is_contiguous() && rw.size(0) == kE && rw.size(1) == kH, "router [256, 2048] bf16");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.numel() >= M, "positions int64");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.numel() >= kSplit * kMaxT * kE, "part [4, 256, 256] fp32");
  TORCH_CHECK(tick.scalar_type() == at::kInt && tick.numel() >= kMaxT / kBT + 1, "tickets");
  const at::cuda::CUDAGuard guard(x.device());
  Args a{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(rw.data_ptr()),
         pos.data_ptr<int64_t>(), part.data_ptr<float>(), tick.data_ptr<int>(), ids.data_ptr<int>(), wts.data_ptr<float>(),
         counts.data_ptr<int>(), tokens.data_ptr<int>(), slots.data_ptr<int>(), nblocks.data_ptr<int>(), bexp.data_ptr<int>(),
         bt0.data_ptr<int>(), static_cast<int>(M)};
  const bool wide = M > kBT;
  auto kernel = fused ? (wide ? route_kernel<true, 32> : route_kernel<true, 16>)
                      : (wide ? route_kernel<false, 32> : route_kernel<false, 16>);
  const uint32_t smem = wide ? smem_bytes<32>() : smem_bytes<16>();
  static bool attr[2][2] = {{false, false}, {false, false}};
  if (!attr[fused][wide]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr[fused][wide] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((M + kBT - 1) / kBT), kSplit * kE / (wide ? 32 : 16));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a));
}

// kb27: route_kernel<true, 32>'s outputs bit for bit at 17..256 rows from one 8-CTA cluster per 16-token block
void route48(torch::Tensor x, torch::Tensor rw, torch::Tensor pos, torch::Tensor part, torch::Tensor tick,
             torch::Tensor ids, torch::Tensor wts, torch::Tensor counts, torch::Tensor tokens, torch::Tensor slots,
             torch::Tensor nblocks, torch::Tensor bexp, torch::Tensor bt0, bool fused,
             c10::optional<torch::Tensor> wq, int64_t web, int64_t wa, int64_t wbo, int64_t wbl, int64_t wk, int64_t wnf,
             c10::optional<torch::Tensor> wdone) {
  const int64_t M = x.size(0);
  TORCH_CHECK(fused && M > kBT && M <= kMaxT, "route48 serves the fused route layout at 17..256 rows");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == kH,
              "x [17..256, 2048] bf16");
  TORCH_CHECK(rw.scalar_type() == at::kBFloat16 && rw.is_contiguous() && rw.size(0) == kE && rw.size(1) == kH, "router [256, 2048] bf16");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.numel() >= M, "positions int64");
  TORCH_CHECK(tick.scalar_type() == at::kInt && tick.numel() >= kMaxT / kBT + 1, "tickets");
  const at::cuda::CUDAGuard guard(x.device());
  Args a{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(rw.data_ptr()),
         pos.data_ptr<int64_t>(), part.data_ptr<float>(), tick.data_ptr<int>(), ids.data_ptr<int>(), wts.data_ptr<float>(),
         counts.data_ptr<int>(), tokens.data_ptr<int>(), slots.data_ptr<int>(), nblocks.data_ptr<int>(), bexp.data_ptr<int>(),
         bt0.data_ptr<int>(), static_cast<int>(M)};
  R48Args w{};
  w.r = a;
  // kb27 expert warming: whole leading clusters of fill CTAs, only with a valid expert-block tensor
  if (wq.has_value() && wk > 0 && wnf > 0) {
    TORCH_CHECK(wq->is_cuda() && wq->is_contiguous() && wq->element_size() == 1 && wq->dim() == 2 && wnf % kRcl == 0 &&
                    wk <= wq->size(0) && web == wq->size(1) && wa % 16 == 0 && wbo % 16 == 0 && wbl % 16 == 0 &&
                    web % 16 == 0 && wa <= web && wbo + wbl <= web && reinterpret_cast<uintptr_t>(wq->data_ptr()) % 16 == 0,
                "route48 warm: whole clusters over 16-byte aligned [E, eb] expert blocks");
    w.wq = reinterpret_cast<const unsigned char*>(wq->data_ptr());
    w.web = web, w.wa = wa, w.wbo = wbo, w.wbl = wbl, w.wk = static_cast<int>(wk), w.wnf = static_cast<int>(wnf);
    TORCH_CHECK(wdone.has_value() && wdone->is_cuda() && wdone->scalar_type() == at::kInt && wdone->numel() >= 1,
                "route48 warm: an int32 epoch word");
    w.wdone = wdone->data_ptr<int>();
  }
  static_assert(kWDepth * kWChunk + 8 * kWDepth <= kR48Smem, "the warm ring fits the route's shared memory");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(route48_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kR48Smem));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(w.wnf + kRcl * ((M + kBT - 1) / kBT)));
  cfg.blockDim = dim3(kR48Threads);
  cfg.dynamicSmemBytes = kR48Smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = kRcl;
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, route48_kernel, w));
}

// kb30m: route_kernel<false, 16>'s outputs bit for bit at 1..16 rows; hist: this layer's int32 [256] score row; with
// wq, wnf leading fill CTAs prefetch the wn experts with the highest scores ([0, wa) and [wbo, wbo + wbl) of their
// blocks, wch bytes per prefetch) into L2
void route16w(torch::Tensor x, torch::Tensor rw, torch::Tensor pos, torch::Tensor part, torch::Tensor tick,
              torch::Tensor ids, torch::Tensor wts, torch::Tensor counts, torch::Tensor tokens, torch::Tensor slots,
              torch::Tensor nblocks, torch::Tensor bexp, torch::Tensor bt0, torch::Tensor hist,
              c10::optional<torch::Tensor> wq, int64_t web, int64_t wa, int64_t wbo, int64_t wbl, int64_t wn, int64_t wnf,
              int64_t wch) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == kH &&
                  M >= 1 && M <= kBT, "route16w: x [1..16, 2048] bf16");
  TORCH_CHECK(rw.scalar_type() == at::kBFloat16 && rw.is_contiguous() && rw.size(0) == kE && rw.size(1) == kH, "router [256, 2048] bf16");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.numel() >= M, "positions int64");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.numel() >= kSplit * kMaxT * kE, "part [4, 256, 256] fp32");
  TORCH_CHECK(tick.scalar_type() == at::kInt && tick.numel() >= kMaxT / kBT + 1, "tickets");
  TORCH_CHECK(hist.is_cuda() && hist.scalar_type() == at::kInt && hist.is_contiguous() && hist.numel() >= kE &&
                  reinterpret_cast<uintptr_t>(hist.data_ptr()) % 16 == 0, "route16w: a 16-byte aligned int32 [256] history row");
  const at::cuda::CUDAGuard guard(x.device());
  RWArgs w{};
  w.r = Args{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(rw.data_ptr()),
             pos.data_ptr<int64_t>(), part.data_ptr<float>(), tick.data_ptr<int>(), ids.data_ptr<int>(), wts.data_ptr<float>(),
             counts.data_ptr<int>(), tokens.data_ptr<int>(), slots.data_ptr<int>(), nblocks.data_ptr<int>(),
             bexp.data_ptr<int>(), bt0.data_ptr<int>(), static_cast<int>(M)};
  w.hist = hist.data_ptr<int>();
  if (wq.has_value() && wn > 0 && wnf > 0) {
    TORCH_CHECK(wq->is_cuda() && wq->is_contiguous() && wq->element_size() == 1 && wq->dim() == 2 && wq->size(0) == kE &&
                    web == wq->size(1) && wa % 16 == 0 && wbo % 16 == 0 && wbl % 16 == 0 && web % 16 == 0 && wa <= web &&
                    wbo + wbl <= web && wn <= kRWMax && wch >= 4096 && wch <= (1 << 20) && wch % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(wq->data_ptr()) % 16 == 0,
                "route16w warm: 16-byte aligned ranges of [256, eb] expert blocks, wn <= 32, 4 KB..1 MB prefetches");
    w.wq = reinterpret_cast<const unsigned char*>(wq->data_ptr());
    w.web = web, w.wa = wa, w.wbo = wbo, w.wbl = wbl;
    w.wn = static_cast<int>(wn), w.wnf = static_cast<int>(wnf), w.wch = static_cast<int>(wch);
  }
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(route16w_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<16>()));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(w.wnf + kSplit * kE / 16));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem_bytes<16>();
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, route16w_kernel, w));
}

// i8x_dec_combine's out plus layer i+1's stock input norm of (out, residual) into y / r_next, in one launch; early:
// the dependents launch at this kernel's start (only for a next kernel that loads nothing of ours before its wait)
void combine_norm(torch::Tensor cache, torch::Tensor x, torch::Tensor gate_w, torch::Tensor pos, torch::Tensor out,
                  torch::Tensor counts, torch::Tensor nblocks, torch::Tensor resid, torch::Tensor norm_w, double eps,
                  torch::Tensor y, torch::Tensor r_next, bool early, torch::Tensor h) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == kH &&
                  M >= 1 && M <= kMaxT, "x [1..256, 2048] bf16");
  for (const auto* t : {&out, &resid, &y, &r_next})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->numel() == M * kH, "rows [M, 2048] bf16");
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.is_contiguous() && cache.numel() >= M * (kTopK + 1) * kH,
              "cache [>= M * 9, 2048] bf16");
  TORCH_CHECK(gate_w.numel() == kH && norm_w.numel() == kH && gate_w.is_contiguous() && norm_w.is_contiguous() &&
                  gate_w.scalar_type() == at::kBFloat16 && norm_w.scalar_type() == at::kBFloat16, "weights [2048] bf16");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.numel() >= M, "positions int64");
  const at::cuda::CUDAGuard guard(x.device());
  CnArgs a{reinterpret_cast<const __nv_bfloat16*>(cache.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr()), pos.data_ptr<int64_t>(),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), counts.data_ptr<int>(), nblocks.data_ptr<int>(),
           reinterpret_cast<const __nv_bfloat16*>(resid.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
           static_cast<float>(eps), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
           reinterpret_cast<__nv_bfloat16*>(r_next.data_ptr()), static_cast<int>(M), early ? 1 : 0};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(M));
  cfg.blockDim = dim3(32 * kCnWarps);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  // n30 (kb30nf): > 16 rows (c48) drop the MoE's dead cache / h rows from L2 (whole 128-byte lines of the workspace)
  const bool disc = M > 16 && h.is_cuda() && h.scalar_type() == at::kBFloat16 && h.is_contiguous() &&
                    h.numel() >= M * (kTopK + 1) * 512 && reinterpret_cast<uintptr_t>(cache.data_ptr()) % 128 == 0 &&
                    reinterpret_cast<uintptr_t>(h.data_ptr()) % 128 == 0;
  if (disc) {
    CnArgsD ad;
    static_cast<CnArgs&>(ad) = a;
    ad.h = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, combine_norm_kernel<true>, ad));
  } else {
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, combine_norm_kernel<false>, a));
  }
}

// ---- the next recurrent layer's decode in_proj (kept in this unit: the combine-norm above releases it early) ----
// DENSE8's swap-AB INT8 GDN in_proj at <= 16 token rows with the whole weight share of a CTA prefetched into shared
// memory before the dependency wait. Bit for bit the king's Triton _w8_seg_kernel (dense8.py) with BLOCK_K = one
// 128-column scale group: for every weight row n and token m,
//   acc = fma(d_b, f32(s[b][n]), acc) over the 16 groups b in order from 0, d_b = one chain of eight
//   mma.sync.m16n8k16 (bf16, fp32) steps from zero with the INT8 weights (exactly bf16) as A,
// then out[m, n] = bf16(acc) into up to four column segments. Triton's dot operands are kWidth 4, which fixes the
// k each MMA slot sees: in the 32-deep chunk j, step 2j takes k = 32j + 4q + {0, 1} in slots 2q + {0, 1} and
// 32j + 16 + 4q + {0, 1} in slots 8 + 2q + {0, 1}; step 2j + 1 the same plus 2.
// One CTA per SM, 16 warps: warp w owns scale group w (k = 128 w .. + 127) of each of the CTA's 16-row weight tiles
// (at most six: 192 KB). The weights and scales are static: every warp issues all of its units (16 rows x 128 k INT8,
// 2 KB, one cp.async group each) before griddepcontrol.wait. The combine-norm before this kernel releases its
// dependents at its start (combine_norm above, `early`), so these CTAs land on the SMs the MoE down leaves idle in its tail
// and the whole weight streams in under it. After the wait each warp loads its group's token fragments once
// (registers), runs the units' MMA chains (the token tiles' chains interleaved) and parks d_b in the unit's own slot;
// one barrier, then each thread chains a 4 x 4 block of outputs (4 rows' scales loaded once for 4 tokens) over the
// 16 groups in order and stores it.
#ifndef D8P_TILES
#define D8P_TILES 6
#endif

namespace ip16 {
constexpr int kK = 2048, kGroups = kK / 128, kMaxRows = 16, kWarps = kGroups, kThreads = 32 * kWarps;
constexpr int kMaxTiles = D8P_TILES, kUnit = 16 * 128;  // a unit: 16 rows x 128 k INT8
constexpr int kDPitch = 20;                              // d_b parked as [token][row] fp32, row pitch 20 floats
constexpr int kSmemUnits = kMaxTiles * kGroups * kUnit, kSmemSc = kMaxTiles * kGroups * 16 * 4;  // fp32 scales
constexpr int kSmemBytes = kSmemUnits + kSmemSc;
static_assert(kMaxTiles >= 1 && kMaxTiles <= 7, "cp.async.wait_group immediates");
static_assert(kMaxRows * kDPitch * 4 <= kUnit, "d_b fits its unit's slot");

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
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wait_groups() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void wait_groups_dyn(int n) {  // n = groups allowed in flight, 0 .. kMaxTiles - 1
  switch (n) {
    case 0: wait_groups<0>(); break;
    case 1: wait_groups<1>(); break;
    case 2: wait_groups<2>(); break;
    case 3: wait_groups<3>(); break;
    case 4: wait_groups<4>(); break;
    case 5: wait_groups<5>(); break;
    default: wait_groups<6>(); break;
  }
}
// four 8x8 b16 matrices: lane (g, q) gets word q of row g of each; lanes 8m .. 8m + 7 give matrix m's row addresses
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t off) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(off + sbase())
               : "memory");
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
__device__ __forceinline__ uint2 ldcg64(const void* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

struct Seg {
  __nv_bfloat16* o[4];
  int lo[4];  // segment s holds columns [lo[s], lo[s + 1]); lo[0] = 0
  int ld[4];
  int n;      // total columns (a multiple of 16)
};

template <int NTT>
__global__ void __launch_bounds__(kThreads, 1) d8p_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const int8_t* __restrict__ q, const __half* __restrict__ s,
                                                         const Seg seg, int M) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
  const int n_tiles = seg.n >> 4;
  const int t0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) * n_tiles) / gridDim.x);
  const int nt = static_cast<int>((static_cast<int64_t>(blockIdx.x + 1) * n_tiles) / gridDim.x) - t0;
  const uint32_t s_sc = kSmemUnits;
  auto slot = [&](int i) { return static_cast<uint32_t>((i * kGroups + warp) * kUnit); };
  // before the wait: this warp's scales (group `warp` of each tile, 32 B) ride with unit 0, then one group per unit
  for (int i = 0; i < nt; ++i) {
    const int8_t* w = q + static_cast<int64_t>((t0 + i) * 16) * kK + warp * 128;
    const uint32_t dst = slot(i);
#pragma unroll
    for (int k = 0; k < 4; ++k) {  // chunk c of row r sits at chunk c ^ (r & 7)
      const int c = lane + 32 * k, r = c >> 3, k16 = c & 7;
      cp16(dst + r * 128 + ((k16 ^ (r & 7)) << 4), w + static_cast<int64_t>(r) * kK + k16 * 16);
    }
    commit();
  }
  // this warp's scales (group `warp` of each tile) as fp32 [tile][group][row], while the units stream
  float* scf = reinterpret_cast<float*>(smem + s_sc);
  for (int idx = lane; idx < nt * 16; idx += 32) {
    const int i = idx >> 4, n = idx & 15;
    scf[(i * kGroups + warp) * 16 + n] = __half2float(__ldg(s + static_cast<int64_t>(warp) * seg.n + (t0 + i) * 16 + n));
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  // this warp's token fragments: x[8t + g][128 warp + 32 j + 4 q .. + 3] and the same + 16 (zero past M)
  uint2 xa[NTT][4], xb[NTT][4];
#pragma unroll
  for (int t = 0; t < NTT; ++t) {
    const int m = 8 * t + g;
    const bool v = m < M;
    const __nv_bfloat16* xr = x + static_cast<int64_t>(v ? m : 0) * kK + warp * 128 + 4 * qd;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      xa[t][j] = v ? ldcg64(xr + 32 * j) : make_uint2(0u, 0u);
      xb[t][j] = v ? ldcg64(xr + 32 * j + 16) : make_uint2(0u, 0u);
    }
  }
  const uint32_t lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lsw = lane & 7, lhi = lane >> 4;
  for (int i = 0; i < nt; ++i) {
    wait_groups_dyn(nt - 1 - i);
    __syncwarp();
    const uint32_t wm = slot(i) + lrow * 128;
    uint32_t lo[4][4], hi[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t r[4];  // W[g][32j + 4q], W[g + 8][..], W[g][32j + 16 + 4q], W[g + 8][..]
      ldsm4(r, wm + (((2 * j + lhi) ^ lsw) << 4));
#pragma unroll
      for (int e = 0; e < 4; ++e) i8x4_bf16x2(r[e], lo[j][e], hi[j][e]);
    }
    float d[NTT][4];
#pragma unroll
    for (int t = 0; t < NTT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) d[t][e] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {  // the token tiles' chains interleave: each step's two MMAs are independent
#pragma unroll
      for (int t = 0; t < NTT; ++t) mma16816(d[t], lo[j][0], lo[j][1], lo[j][2], lo[j][3], xa[t][j].x, xb[t][j].x);
#pragma unroll
      for (int t = 0; t < NTT; ++t) mma16816(d[t], hi[j][0], hi[j][1], hi[j][2], hi[j][3], xa[t][j].y, xb[t][j].y);
    }
    __syncwarp();  // every lane's ldmatrix of the slot is done: park d_b over it as [token][row]
    float* dp = reinterpret_cast<float*>(smem + slot(i));
#pragma unroll
    for (int t = 0; t < NTT; ++t) {
      const int m = 8 * t + 2 * qd;
      dp[m * kDPitch + g] = d[t][0];
      dp[(m + 1) * kDPitch + g] = d[t][1];
      dp[m * kDPitch + g + 8] = d[t][2];
      dp[(m + 1) * kDPitch + g + 8] = d[t][3];
    }
  }
  __syncthreads();
  // outputs (tile i, rows 4 c .. 4 c + 3, tokens 4 mq .. 4 mq + 3): the 16 groups' d_b chained in order with their
  // scales; each scale quad is loaded once for four tokens
  const int mqs = (M + 3) >> 2;
  for (int idx = tid; idx < nt * 4 * mqs; idx += kThreads) {
    const int c = idx & 3, mq = (idx >> 2) % mqs, i = (idx >> 2) / mqs;
    float acc[4][4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[u][e] = 0.f;
#pragma unroll
    for (int b = 0; b < kGroups; ++b) {
      const float4 sv = *reinterpret_cast<const float4*>(scf + (i * kGroups + b) * 16 + 4 * c);
      const uint8_t* db = smem + (i * kGroups + b) * kUnit + (4 * mq * kDPitch + 4 * c) * 4;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float4 dv = *reinterpret_cast<const float4*>(db + u * kDPitch * 4);
        acc[u][0] = ffma(dv.x, sv.x, acc[u][0]);
        acc[u][1] = ffma(dv.y, sv.y, acc[u][1]);
        acc[u][2] = ffma(dv.z, sv.z, acc[u][2]);
        acc[u][3] = ffma(dv.w, sv.w, acc[u][3]);
      }
    }
    const int col = (t0 + i) * 16 + 4 * c;
    const int sgi = (col >= seg.lo[1]) + (col >= seg.lo[2]) + (col >= seg.lo[3]);
    __nv_bfloat16* o = sgi == 0 ? seg.o[0] : sgi == 1 ? seg.o[1] : sgi == 2 ? seg.o[2] : seg.o[3];
    const int lo_s = sgi == 0 ? 0 : sgi == 1 ? seg.lo[1] : sgi == 2 ? seg.lo[2] : seg.lo[3];
    const int ld = sgi == 0 ? seg.ld[0] : sgi == 1 ? seg.ld[1] : sgi == 2 ? seg.ld[2] : seg.ld[3];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int m = 4 * mq + u;
      if (m < M) {
        __nv_bfloat162 ob[2] = {__floats2bfloat162_rn(acc[u][0], acc[u][1]), __floats2bfloat162_rn(acc[u][2], acc[u][3])};
        *reinterpret_cast<uint2*>(o + static_cast<int64_t>(m) * ld + (col - lo_s)) = *reinterpret_cast<const uint2*>(ob);
      }
    }
  }
}

int g_sms = 0;

// outs: 1..4 bf16 tensors, segment i holding columns [lo[i], lo[i + 1]) with row stride ld[i]; one CTA per SM
void inproj16(torch::Tensor x, torch::Tensor q, torch::Tensor s, std::vector<torch::Tensor> outs, std::vector<int64_t> lo,
             std::vector<int64_t> ld) {
  const int64_t M = x.size(0), N = q.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == kK &&
                  M >= 1 && M <= kMaxRows, "x [1..16, 2048] bf16");
  TORCH_CHECK(q.scalar_type() == at::kChar && q.is_contiguous() && q.size(1) == kK && N % 16 == 0, "q [16 n, 2048] int8");
  TORCH_CHECK(s.scalar_type() == at::kHalf && s.is_contiguous() && s.size(0) == kGroups && s.size(1) == N, "s [16, N] fp16");
  TORCH_CHECK(outs.size() >= 1 && outs.size() <= 4 && lo.size() == outs.size() && ld.size() == outs.size() && lo[0] == 0,
              "segments");
  Seg seg{};
  for (int i = 0; i < 4; ++i) {
    const int j = i < static_cast<int>(outs.size()) ? i : static_cast<int>(outs.size()) - 1;
    TORCH_CHECK(outs[j].scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(outs[j].data_ptr()) % 8 == 0 &&
                    ld[j] % 4 == 0, "outputs bf16, 8-byte aligned rows");
    TORCH_CHECK(i >= static_cast<int>(outs.size()) || lo[i] % 16 == 0, "segments start on 16-row tiles");
    seg.o[i] = reinterpret_cast<__nv_bfloat16*>(outs[j].data_ptr());
    seg.lo[i] = i < static_cast<int>(outs.size()) ? static_cast<int>(lo[i]) : static_cast<int>(N);
    seg.ld[i] = static_cast<int>(ld[j]);
  }
  seg.n = static_cast<int>(N);
  const at::cuda::CUDAGuard guard(x.device());
  if (!g_sms) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, x.device().index()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(d8p_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(d8p_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  const int tiles = static_cast<int>(N / 16);
  const int grid = std::min(g_sms, tiles);
  TORCH_CHECK((tiles + grid - 1) / grid <= kMaxTiles, "too many 16-row tiles per CTA for D8P_TILES");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, M > 8 ? d8p_kernel<2> : d8p_kernel<1>,
                                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                    reinterpret_cast<const int8_t*>(q.data_ptr()),
                                    reinterpret_cast<const __half*>(s.data_ptr()), seg, static_cast<int>(M)));
}
}  // namespace ip16

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("combine_norm", &combine_norm, "decode MoE combine + the next layer's Gemma fused add-RMSNorm, bit for bit");
  m.def("inproj16", &ip16::inproj16, "DENSE8 INT8 GDN in_proj at <= 16 rows, weights prefetched before the wait; bit for bit the Triton _w8_seg_kernel (BLOCK_K 128)");
  m.attr("INPROJ16_MAX_ROWS") = ip16::kMaxRows;
  m.attr("INPROJ16_MAX_TILES") = ip16::kMaxTiles;
  m.def("route", &route, "decode MoE router gate + top-8 + expert lists in one launch, bit for bit the Triton routes");
  m.def("route48", &route48, "kb27: route's outputs at 17..256 rows from one 8-CTA cluster per 16-token block, bit for bit",
        py::arg("x"), py::arg("rw"), py::arg("pos"), py::arg("part"), py::arg("tick"), py::arg("ids"), py::arg("wts"),
        py::arg("counts"), py::arg("tokens"), py::arg("slots"), py::arg("nblocks"), py::arg("bexp"), py::arg("bt0"),
        py::arg("fused"), py::arg("wq") = py::none(), py::arg("web") = 0, py::arg("wa") = 0, py::arg("wbo") = 0,
        py::arg("wbl") = 0, py::arg("wk") = 0, py::arg("wnf") = 0, py::arg("wdone") = py::none());
  m.def("route16w", &route16w, "kb30m: route's outputs at 1..16 rows (route_kernel<false, 16> bit for bit) + predicted-expert L2 prefetch",
        py::arg("x"), py::arg("rw"), py::arg("pos"), py::arg("part"), py::arg("tick"), py::arg("ids"), py::arg("wts"),
        py::arg("counts"), py::arg("tokens"), py::arg("slots"), py::arg("nblocks"), py::arg("bexp"), py::arg("bt0"),
        py::arg("hist"), py::arg("wq") = py::none(), py::arg("web") = 0, py::arg("wa") = 0, py::arg("wbo") = 0,
        py::arg("wbl") = 0, py::arg("wn") = 0, py::arg("wnf") = 0, py::arg("wch") = 32768);
}
