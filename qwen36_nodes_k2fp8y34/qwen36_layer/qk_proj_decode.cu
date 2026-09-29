// Decode-width projections of a Qwen3.6-35B-A3B BF16 decoder layer (TP1) on sm_90a, for the
// MTP target-verify row batches (1 to 256 token rows):
//   gdn_in_proj       x [M, 2048] @ [w_qkvz; w_ba].T -> [q|k|v] [M, 8192], z [M, 32, 128], b, a [M, 32]
//   qkv_proj          x [M, 2048] @ w [9216, 2048].T -> [M, 9216]
//
// At these widths each projection is a weight stream: 37.7 to 50.6 MB of bf16 weights read once
// from HBM against at most 2 MB of activations. One CTA per SM owns a slice of weight rows. A
// producer lane streams the slice through an mbarrier
// ring with 2D TMA copies (128-byte swizzle, 64 k per stage); the weights are not the previous
// kernel's output, so the first stages are requested before the programmatic-dependent wait.
// The token rows arrive per stage as 64-row TMA boxes whose rows past M are zero-filled, and are
// the m64 A operand of wgmma; the CTA's weight rows are its n (96, 72 or 128). One math
// warpgroup per 64 token rows; fp32 accumulation; each output tile leaves through shared memory
// as whole 16-byte row chunks.
//
// Every CTA covers all of K, so the accumulators are the outputs: they are rounded to bf16 as
// cuBLAS rounds and stored into the output that owns their column.
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

constexpr int BK = 64;                        // k per stage: one 128-byte swizzle row
constexpr int XT = 64 * BK * 2;               // bytes of one 64-row token tile per stage
constexpr int MAX_T = 256;
// Ring bytes (+ 1 KB alignment slack <= 227 KB). Loaded HBM latency is 1.5-2 us, so each SM keeps
// 100+ KB of weights in flight: 48 to 100 KB rings measured 1.4 to 11 us slower at M = 16.
constexpr int STAGE_BUDGET = 224 * 1024;
// Weight stages requested before the programmatic-dependent wait. The served predecessor (the
// fused add-norm) releases this grid only as it ends, so a deeper pre-wait burst just queues the
// token rows of stage 0 behind 17 MB of weight requests (full ring: +0.8 to +1.5 us, measured).
constexpr int PRE_WAIT_STAGES = 2;
constexpr unsigned FULL = 0xffffffffu;

// ---- PTX helpers ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "W_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra W_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                       uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
template <int N>
__device__ __forceinline__ void fence_acc(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
// K-major operand in the 128-byte swizzle layout TMA writes: 8-row x 128-byte atoms stacked at
// 1024 bytes (SBO), layout type SWIZZLE_128B; a k16 step advances the start address 32 bytes.
__device__ __forceinline__ uint64_t gmma_desc(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// wgmma.mma_async m64nNk16, f32 += bf16 x bf16, A and B from shared memory, both K-major.
__device__ __forceinline__ void wgmma_n72(float (&d)[36], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %38, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35"
      "}, %36, %37, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_n96(float (&d)[48], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %50, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47"
      "}, %48, %49, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "l"(da), "l"(db), "r"(scale_d));
}

template <int N>
__device__ __forceinline__ void mma(float (&d)[N / 2], uint64_t a, uint64_t b);
template <>
__device__ __forceinline__ void mma<72>(float (&d)[36], uint64_t a, uint64_t b) { wgmma_n72(d, a, b, 1); }
template <>
__device__ __forceinline__ void mma<96>(float (&d)[48], uint64_t a, uint64_t b) { wgmma_n96(d, a, b, 1); }

// ---- configurations -------------------------------------------------------------------------
// R weight rows per CTA (the wgmma n), K, K ranges per column block, epilogue (0 store, 1 add-norm).
// CTAs [0, N_MAIN) own main-weight rows [R b, R b + R) = output columns; CTA N_MAIN owns
// the aux weight's AUX_ROWS rows at output columns [AUX_COL0, AUX_COL0 + AUX_ROWS). Output column c
// in [lo(s), lo(s + 1)) goes to dst[s][row, c - lo(s)] with row stride ld(s); every boundary is a
// multiple of 8, so the 8 columns of an accumulator group share one output. The tables are
// compile-time: a register-indexed parameter read after the K loop misses the constant cache
// and waits on L2 behind the weight stream (1.7 us of epilogue, measured).
template <int R_, int K_>
struct Cfg {
  static constexpr int R = R_, K = K_;
  static constexpr int KCH = K / BK;           // stages of work per CTA
  static constexpr int WB = R * BK * 2;        // weight bytes per stage
  static constexpr int N_MAIN = 0, AUX_ROWS = 0, AUX_COL0 = 0, NSEG = 1;
};
struct InProj : Cfg<96, 2048> {  // 128 CTAs x 96 rows of w_qkvz + 1 CTA for w_ba's 64 rows
  static constexpr int N_MAIN = 128, AUX_ROWS = 64, AUX_COL0 = 12288, NSEG = 4;  // [q|k|v], z, b, a
  __device__ static constexpr int lo(int s) { return s == 1 ? 8192 : s == 2 ? 12288 : s == 3 ? 12320 : 0; }
  __device__ static constexpr int ld(int s) { return s == 0 ? 8192 : s == 1 ? 4096 : 32; }
};
struct QkvProj : Cfg<72, 2048> {  // 128 CTAs x 72 rows
  static constexpr int N_MAIN = 128;
  __device__ static constexpr int lo(int) { return 0; }
  __device__ static constexpr int ld(int) { return 9216; }
};
static_assert(InProj::N_MAIN * InProj::R == 12288 && QkvProj::N_MAIN * QkvProj::R == 9216, "row split");

// __host__ __device__: the validator's nvcc runs without --expt-relaxed-constexpr.
template <class C, int TT>
__host__ __device__ constexpr int stages() {
  constexpr int n = STAGE_BUDGET / (C::WB + TT * XT);
  return n < C::KCH ? n : C::KCH;
}
template <class C, int TT>
__host__ __device__ constexpr int smem_bytes() {
  return stages<C, TT>() * (C::WB + TT * XT) + 1024;
}

// Copied to shared memory at kernel start and read from there (see Cfg).
struct Params {
  int M;
  bf16* dst[4];  // the output of segment s
};

template <class C, int TT>
__global__ void __launch_bounds__(TT * 128 + 32, 1)
    proj_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ CUtensorMap tm_aux,
                const __grid_constant__ CUtensorMap tm_x, const Params p) {
  constexpr int R = C::R, KCH = C::KCH, WB = C::WB, SB = WB + TT * XT, STAGES = stages<C, TT>();
  constexpr int CONSUMERS = TT * 128;
  // Stage c - 1 is released at iteration c, so one stage would deadlock.
  static_assert(STAGES >= 2, "the ring needs at least two stages");
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
  __shared__ Params sp;
  extern __shared__ unsigned char smem_raw[];
  const uint32_t base = (smem_u32(smem_raw) + 1023u) & ~1023u;  // swizzled tiles need 1 KB alignment
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  const bool aux = static_cast<int>(blockIdx.x) >= C::N_MAIN;
  const int wrow0 = aux ? 0 : static_cast<int>(blockIdx.x) * R, col0 = aux ? C::AUX_COL0 : wrow0;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], TT * 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    sp = p;  // published to the math warps by the barrier after the K loop
  }

  if (warp == TT * 4) {
    // Producer. The first PRE_WAIT_STAGES weight loads run before the programmatic-dependent wait.
    if (lane == 0) {
      const CUtensorMap* wm = aux ? &tm_aux : &tm_w;
      prefetch_tmap(wm);
      prefetch_tmap(&tm_x);
      const uint32_t tx = (aux ? C::AUX_ROWS * BK * 2 : WB) + TT * XT;
      const uint64_t pol_w = policy_evict_first(), pol_x = policy_evict_last();
      constexpr int PRE = PRE_WAIT_STAGES < STAGES ? PRE_WAIT_STAGES : STAGES;
      for (int c = 0; c < PRE; ++c) {
        mbar_expect(&full[c], tx);
        tma_2d(base + c * SB, wm, &full[c], c * BK, wrow0, pol_w);
      }
      griddep_wait();
      for (int c = 0; c < PRE; ++c)
        for (int t = 0; t < TT; ++t) tma_2d(base + c * SB + WB + t * XT, &tm_x, &full[c], c * BK, 64 * t, pol_x);
      for (int c = PRE; c < KCH; ++c) {
        const int s = c % STAGES;
        if (c >= STAGES) mbar_wait(&empty[s], ((c / STAGES) & 1) ^ 1);
        mbar_expect(&full[s], tx);
        tma_2d(base + s * SB, wm, &full[s], c * BK, wrow0, pol_w);
        for (int t = 0; t < TT; ++t) tma_2d(base + s * SB + WB + t * XT, &tm_x, &full[s], c * BK, 64 * t, pol_x);
      }
    }
    return;
  }

  // Math warpgroup wg: token rows [64 wg, 64 wg + 64) against the CTA's R weight rows.
  griddep_wait();
  const int wg = warp >> 2;
  float acc[R / 2];
#pragma unroll
  for (int i = 0; i < R / 2; ++i) acc[i] = 0.f;
#pragma unroll 1
  for (int c = 0; c < KCH; ++c) {
    const int s = c % STAGES;
    mbar_wait(&full[s], (c / STAGES) & 1);
    const uint32_t b = base + s * SB, a = b + WB + wg * XT;
    fence_acc(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < BK / 16; ++k) mma<R>(acc, gmma_desc(a + 32 * k), gmma_desc(b + 32 * k));
    wgmma_commit();
    fence_acc(acc);
    wgmma_wait<1>();
    if (c > 0 && lane == 0) mbar_arrive(&empty[(c - 1) % STAGES]);
  }
  wgmma_wait<0>();
  fence_acc(acc);
  named_sync(1, CONSUMERS);  // sp is visible
  // Dependents launch once every CTA is past its loads. Triggered at kernel start, a successor's
  // small CTAs launch while this grid holds 129 of 132 SMs, pile onto the idle ones and then run
  // there crowded: the fused add-norm after an output-projection call of this kernel at M = 124
  // took 5.5 us past its end instead of 2.6 (nsys), and the late trigger cut the served-like
  // chain by 1.5 to 2.4 us at M = 124.
  griddep_launch();

  // Accumulator layout (wgmma m64nN f32): warp wi of the warpgroup holds token rows 16 wi + lane/4
  // (acc[4j], acc[4j+1]) and + 8 (acc[4j+2], acc[4j+3]), at columns 8 j + 2 (lane % 4) + {0, 1}.
  // The tile goes out through shared memory (the drained ring) as whole 16-byte row chunks: stored
  // straight from the fragments, every warp store is 8 rows x 16 or 32 bytes, and the grid's
  // stores are then limited by the L2 request rate (1.8 us for in_proj at M = 124, measured).
  const int wi = warp & 3, M = sp.M;
  const int t0 = wg * 64 + wi * 16 + (lane >> 2), t1 = t0 + 8, cq = 2 * (lane & 3);
  unsigned char* tile = smem_raw + (base - smem_u32(smem_raw));
  constexpr int RS = R * 2 % 128 == 64 ? R * 2 + 16 : R * 2;  // row stride; bank-conflict-free fragment writes
  static_assert(TT * 64 * RS <= STAGES * SB, "the output tile fits in the ring");
#pragma unroll
  for (int j = 0; j < R / 8; ++j) {
    *reinterpret_cast<uint32_t*>(tile + t0 * RS + 16 * j + 2 * cq) = pack_bf16(acc[4 * j], acc[4 * j + 1]);
    *reinterpret_cast<uint32_t*>(tile + t1 * RS + 16 * j + 2 * cq) = pack_bf16(acc[4 * j + 2], acc[4 * j + 3]);
  }
  named_sync(1, CONSUMERS);
  // Chunk q of a row holds columns col0 + 8 q .. + 8, which share one output (see Cfg).
  auto put = [&](int row, int q) {
    const int c8 = col0 + 8 * q;
    bf16* d = sp.dst[0];
    int lo = C::lo(0), ld = C::ld(0);
#pragma unroll
    for (int s = 1; s < C::NSEG; ++s)
      if (c8 >= C::lo(s)) d = sp.dst[s], lo = C::lo(s), ld = C::ld(s);
    *reinterpret_cast<uint4*>(d + static_cast<int64_t>(row) * ld + (c8 - lo)) =
        *reinterpret_cast<const uint4*>(tile + row * RS + 16 * q);
  };
  if (aux) {
    for (int k = tid; k < M * (C::AUX_ROWS / 8); k += CONSUMERS) put(k / (C::AUX_ROWS / 8), k % (C::AUX_ROWS / 8));
  } else {
    for (int k = tid; k < M * (R / 8); k += CONSUMERS) put(k / (R / 8), k % (R / 8));
  }
}

// ---- host -----------------------------------------------------------------------------------
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeFn encode_fn() {
  static EncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &q));
    TORCH_CHECK(ptr != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled is unavailable");
    fn = reinterpret_cast<EncodeFn>(ptr);
  }
  return fn;
}

// Row-major bf16 [outer, inner] in boxes of [box_outer rows, 64 columns], 128-byte swizzle;
// rows past `outer` read as zero.
CUtensorMap make_map(const torch::Tensor& t, uint64_t inner, uint64_t outer, uint32_t box_outer) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner * 2};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(BK), box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, estr,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

void check_bf16(const torch::Tensor& t, const char* name, std::initializer_list<int64_t> shape) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bfloat16 tensor");
  TORCH_CHECK(t.sizes() == c10::IntArrayRef(shape), name, " has shape ", t.sizes(), ", expected ",
              c10::IntArrayRef(shape));
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

int64_t rows_of(const torch::Tensor& x, int64_t k) {
  TORCH_CHECK(x.dim() == 2 && x.size(1) == k, "x must be [rows, ", k, "], got ", x.sizes());
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_T, "the decode projections serve 1 to ", MAX_T, " rows, got ", M);
  return M;
}

int sm_count(int dev) {
  static int sms[64] = {};
  TORCH_CHECK(dev >= 0 && dev < 64, "device index ", dev, " out of range");
  if (sms[dev] == 0) C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev));
  return sms[dev];
}

template <class C, int TT>
void launch_tt(const CUtensorMap& w, const CUtensorMap& aux, const CUtensorMap& x, const Params& p, int grid,
               int dev) {
  auto kernel = proj_kernel<C, TT>;
  constexpr int SMEM = smem_bytes<C, TT>();
  static uint64_t ready = 0;  // devices whose dynamic shared memory limit is raised
  if (!((ready >> dev) & 1)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ready |= 1ull << dev;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(TT * 128 + 32);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, w, aux, x, p));
}

template <class C>
void launch(const CUtensorMap& w, const CUtensorMap& aux, const CUtensorMap& x, const Params& p, int grid,
            int dev) {
  TORCH_CHECK(grid <= sm_count(dev), "the projection needs ", grid, " SMs, the device has ", sm_count(dev));
  switch ((p.M + 63) / 64) {
    case 1: launch_tt<C, 1>(w, aux, x, p, grid, dev); break;
    case 2: launch_tt<C, 2>(w, aux, x, p, grid, dev); break;
    case 3: launch_tt<C, 3>(w, aux, x, p, grid, dev); break;
    default: launch_tt<C, 4>(w, aux, x, p, grid, dev); break;
  }
}

}  // namespace

// The recurrent block's two input projections: [q|k|v] = rows [0, qkv) of w_qkvz, z = the rest
// (as [M, nv, 128]), b = w_ba rows [0, nv), a = w_ba rows [nv, 2 nv).
std::vector<torch::Tensor> gdn_in_proj(torch::Tensor x, torch::Tensor w_qkvz, torch::Tensor w_ba, int64_t qkv,
                                       int64_t nv) {
  const int64_t M = rows_of(x, 2048);
  TORCH_CHECK(qkv == 8192 && nv == 32, "gdn_in_proj serves qkv=8192, nv=32, got ", qkv, ", ", nv);
  check_bf16(x, "x", {M, 2048});
  check_bf16(w_qkvz, "w_qkvz", {12288, 2048});
  check_bf16(w_ba, "w_ba", {64, 2048});
  const at::cuda::CUDAGuard guard(x.device());
  auto mixed = torch::empty({M, 8192}, x.options());
  auto z = torch::empty({M, 32, 128}, x.options());
  auto b = torch::empty({M, 32}, x.options());
  auto a = torch::empty({M, 32}, x.options());
  Params p{};
  p.M = static_cast<int>(M);
  p.dst[0] = reinterpret_cast<bf16*>(mixed.data_ptr());
  p.dst[1] = reinterpret_cast<bf16*>(z.data_ptr());
  p.dst[2] = reinterpret_cast<bf16*>(b.data_ptr());
  p.dst[3] = reinterpret_cast<bf16*>(a.data_ptr());
  const CUtensorMap mw = make_map(w_qkvz, 2048, 12288, InProj::R), ma = make_map(w_ba, 2048, 64, InProj::AUX_ROWS),
                    mx = make_map(x, 2048, M, 64);
  launch<InProj>(mw, ma, mx, p, InProj::N_MAIN + 1, x.get_device());
  return {mixed, z, b, a};
}

// The attention block's fused q/k/v projection: x [M, 2048] @ w [9216, 2048].T.
torch::Tensor qkv_proj(torch::Tensor x, torch::Tensor w) {
  const int64_t M = rows_of(x, 2048);
  check_bf16(x, "x", {M, 2048});
  check_bf16(w, "qkv weight", {9216, 2048});
  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty({M, 9216}, x.options());
  Params p{};
  p.M = static_cast<int>(M);
  p.dst[0] = reinterpret_cast<bf16*>(out.data_ptr());
  const CUtensorMap mw = make_map(w, 2048, 9216, QkvProj::R), mx = make_map(x, 2048, M, 64);
  launch<QkvProj>(mw, mw, mx, p, QkvProj::N_MAIN, x.get_device());
  return out;
}


// ---------------------------------------------------------------------------------------------------------------------
// v1 lane dense (was qk_dense8.cu; merged into this unit because the build's native-extension metadata is capped at
// 16 MiB, ~1.14 MB per unit: 15+ units are refused). Everything below sits in namespace qd8.
// DENSE8 W8A16 projections in CUDA for decode / MTP target-verify row batches of 1..32 token rows (lane "dense").
//
// A candidate-owned INT8 copy of a BF16 projection weight w [N, K] (dense8.py: q [N, K] int8, symmetric, one fp16
// scale per 128 consecutive k of a row, stored group-major s [K / 128, N]) is streamed once from HBM; the token rows
// x [M, K] bf16 are re-read per stage from L2. Every output is
//     acc = fma(d_G-1, c_G-1, ... fma(d_0, c_0, 0)),   d_g = one chain of eight m16n8k16 (bf16 x bf16 -> fp32) steps
// over the 128-deep scale group g from zero, in Triton's kWidth-4 k order (32-deep chunk j: step 2j takes
// k = 32j + 4q + {0, 1} and 32j + 16 + 4q + {0, 1}, step 2j + 1 the same plus 2), c_g = the row's fp16 group scale:
// the same k-slot products and the same fma chain as dense8.py's Triton _w8_seg_kernel (acc += dot(w, x^T) * sc) and
// qk_moe_dn16.cu. The products are computed transposed (weight rows as the m16 operand, tokens as the n8 operand), so
// 1..32 token rows waste no weight-side width.
//
// One CTA owns R = 16 W weight rows (W math warps, 16 rows each) over a K range (blockIdx.y selects the range when
// the caller splits K); a producer warp streams [R rows x 128 k] INT8 boxes plus the tokens' [8 NT rows x 128 k]
// bf16 boxes with 2D TMA (128-byte swizzle) through an mbarrier ring of as many stages as shared memory holds. The
// weights are not the previous kernel's output, so the first stages are requested before the programmatic-dependent
// wait; the token boxes follow it. The weight stream uses an evict-first L2 policy, the token rows evict-last.
// Outputs: MODE 0 stores bf16 rows into up to four column segments (in_proj's [q|k|v], z, b, a; qkv's one output);
// MODE 1 stores fp32 split-K partials [splits, M, N] for the fused add + Gemma RMSNorm that sums them (proj.py).
// Both leave through shared memory as whole 16-byte row chunks.
// ---------------------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace qd8 {

namespace {

constexpr int GROUP = 128;       // k per stage = one scale group
constexpr int MAX_TOK = 32;      // token rows served
constexpr int SMEM_BUDGET = 220 * 1024;
// fp16 operands: a prep warp turns each stage's bf16 token rows into fp16 once (exact for every bf16 value in the fp16
// normal range, saturating outside it), so the INT8 weights convert with the 5-instruction fp16 bias trick instead of
// the 8-instruction bf16 one; the products (int8 x fp16, exact in fp32) and the fp32 chains are those of the bf16
// operands, so the outputs are the bf16 path's bits whenever the token rows are fp16-exact.
constexpr int CV = 1;
constexpr int GP = 2;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "W_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra W_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t lds16(uint32_t addr) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr) : "memory");
  return v;
}
// four int8 (k order) -> bf16x2 {b0, b1}, {b2, b3}, exact (qk_moe_dn16.cu)
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
// four int8 (k order) -> f16x2 {b0, b1}, {b2, b3}, exact: 0x64XX = 1024 + XX for the biased byte XX, minus 1152
__device__ __forceinline__ void i8x4_f16x2(uint32_t v, uint32_t& lo, uint32_t& hi) {
  asm("{\n"
      ".reg .b32 u, l0, h0, m;\n"
      "xor.b32 u, %2, 0x80808080;\n"
      "mov.b32 m, 0x64806480;\n"
      "prmt.b32 l0, u, 0x64646464, 0x4140;\n"
      "prmt.b32 h0, u, 0x64646464, 0x4342;\n"
      "sub.f16x2 %0, l0, m;\n"
      "sub.f16x2 %1, h0, m;\n"
      "}"
      : "=r"(lo), "=r"(hi)
      : "r"(v));
}
__device__ __forceinline__ void mma16816h(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void sts128(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// bf16x2 -> f16x2 (round to nearest, saturating to the fp16 range)
__device__ __forceinline__ uint32_t bf2_f2(uint32_t v) {
  uint32_t r;
  asm("{\n.reg .f32 a, b;\nmov.b32 a, %1;\nmov.b32 b, %2;\ncvt.rn.satfinite.f16x2.f32 %0, b, a;\n}"
      : "=r"(r) : "r"(v << 16), "r"(v & 0xffff0000u));
  return r;
}
__device__ __forceinline__ float ffma(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

struct Params {
  const __half* s;   // [K / 128, N] group scales
  void* dst[4];      // MODE 0: bf16 outputs of the column segments; MODE 1: dst[0] = fp32 partials [splits, M, N]
  int lo[4], ld[4];  // MODE 0: segment s = columns [lo[s], lo[s + 1]) with row stride ld[s]
  int nseg, M, N, ks, pre;  // ks: k per blockIdx.y; pre: weight stages requested before the dependency wait
};

template <int W, int NT>
struct Shape {
  static constexpr int R = 16 * W;                 // weight rows per CTA
  static constexpr int WB = R * GROUP;             // weight bytes per stage
  static constexpr int XB = 8 * NT * 128;          // bytes of one 64-k token box (8 NT rows x 128 B)
  static constexpr int SB = WB + 2 * XB;           // stage bytes (multiple of 1024)
  static constexpr int SC = 16 * R * 2;            // scales: up to 16 groups x R rows fp16
  static constexpr int STAGES = (SMEM_BUDGET - SC - 1024) / SB;
  static constexpr int SMEM = STAGES * SB + SC + 1024;
  static_assert(SB % 1024 == 0, "stages stay 1 KB aligned for the 128-byte swizzle");
  static_assert(STAGES >= 2, "the ring needs two stages");
};

template <int W, int NT, int GP, int MODE, int CV>
__global__ void __launch_bounds__(32 * (W + 1 + CV), 1)
    dense8_kernel(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ Params p) {
  using S = Shape<W, NT>;
  constexpr int R = S::R, WB = S::WB, XB = S::XB, SB = S::SB, STAGES = S::STAGES;
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES], xq[STAGES];
  extern __shared__ unsigned char smem_raw[];
  const uint32_t base = (smem_u32(smem_raw) + 1023u) & ~1023u;
  const uint32_t s_sc = base + STAGES * SB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int row0 = blockIdx.x * R, split = blockIdx.y, k0 = split * p.ks, nst = p.ks / GROUP;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], W);
      mbar_init(&xq[s], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == W) {  // producer
    if (lane == 0) {
      prefetch_tmap(&tm_q);
      prefetch_tmap(&tm_x);
      const uint64_t pol_w = policy_evict_first(), pol_x = policy_evict_last();
      const int pre = p.pre < STAGES ? (p.pre < nst ? p.pre : nst) : (STAGES < nst ? STAGES : nst);
      constexpr uint32_t tx = SB;
      for (int c = 0; c < pre; ++c) {
        mbar_expect(&full[c], tx);
        tma_2d(base + c * SB, &tm_q, &full[c], k0 + c * GROUP, row0, pol_w);
      }
      griddep_wait();
      for (int c = 0; c < pre; ++c) {
        tma_2d(base + c * SB + WB, &tm_x, &full[c], k0 + c * GROUP, 0, pol_x);
        tma_2d(base + c * SB + WB + XB, &tm_x, &full[c], k0 + c * GROUP + 64, 0, pol_x);
      }
      for (int c = pre; c < nst; ++c) {
        const int s = c % STAGES;
        if (c >= STAGES) mbar_wait(&empty[s], ((c / STAGES) & 1) ^ 1);
        mbar_expect(&full[s], tx);
        tma_2d(base + s * SB, &tm_q, &full[s], k0 + c * GROUP, row0, pol_w);
        tma_2d(base + s * SB + WB, &tm_x, &full[s], k0 + c * GROUP, 0, pol_x);
        tma_2d(base + s * SB + WB + XB, &tm_x, &full[s], k0 + c * GROUP + 64, 0, pol_x);
      }
    }
    return;
  }

  if (CV && warp == W + 1) {  // fp16 path: turn each stage's bf16 token boxes into fp16 in place, once for all warps
    constexpr int CH = 2 * 8 * NT * 8;  // 16-byte chunks of the stage's two token boxes
    for (int c = 0; c < nst; ++c) {
      const int s = c % STAGES;
      mbar_wait(&full[s], (c / STAGES) & 1);
      const uint32_t xb = base + s * SB + WB;
#pragma unroll
      for (int i = lane; i < CH; i += 32) {
        const uint4 v = lds128(xb + 16 * i);
        sts128(xb + 16 * i, make_uint4(bf2_f2(v.x), bf2_f2(v.y), bf2_f2(v.z), bf2_f2(v.w)));
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&xq[s]);
    }
    return;
  }
  // scales of this CTA's rows for its groups: [g][R] fp16 (rows past N are not loaded and never stored)
  {
    const int rows = p.N - row0 < R ? p.N - row0 : R;
    const int chunks = rows / 8;  // N and row0 are multiples of 8
    const int g0 = k0 / GROUP;
    for (int i = tid; i < nst * chunks; i += 32 * W) {
      const int g = i / chunks, c = i % chunks;
      cp16(s_sc + g * R * 2 + c * 16, p.s + static_cast<int64_t>(g0 + g) * p.N + row0 + c * 8);
    }
    cp_commit_wait_all();
  }
  griddep_wait();
  named_sync(1, 32 * W);

  const int g = lane >> 2, q = lane & 3;
  // ldmatrix row address pieces: matrix m = lane / 8 -> row (lane & 7) + 8 (m & 1), 16-byte chunk (m >> 1)
  const int lrow = 16 * warp + (lane & 7) + 8 * ((lane >> 3) & 1), lchunk = lane >> 4;
  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

  // GP scale groups per iteration: their d chains are independent (only the fma into acc is ordered), so the warp
  // interleaves GP x NT mma chains instead of waiting out one chain's latency at a time.
#pragma unroll 1
  for (int c = 0; c < nst; c += GP) {
    uint32_t lo[GP][4][4], hi[GP][4][4];
    float sca[GP], scb[GP];
#pragma unroll
    for (int gp = 0; gp < GP; ++gp) {
      const int s = (c + gp) % STAGES;
      mbar_wait(&full[s], ((c + gp) / STAGES) & 1);
      if (CV) mbar_wait(&xq[s], ((c + gp) / STAGES) & 1);
      const uint32_t wb = base + s * SB;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        uint32_t r[4];  // W[g][32jj + 4q ..], W[g + 8][..], W[g][32jj + 16 + 4q ..], W[g + 8][..]
        const int ch = 2 * jj + lchunk;
        ldsm4(r, wb + lrow * 128 + ((ch ^ (lrow & 7)) << 4));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (CV) i8x4_f16x2(r[i], lo[gp][jj][i], hi[gp][jj][i]);
          else i8x4_bf16x2(r[i], lo[gp][jj][i], hi[gp][jj][i]);
        }
      }
      sca[gp] = __half2float(__ushort_as_half(static_cast<unsigned short>(lds16(s_sc + (c + gp) * R * 2 + (16 * warp + g) * 2))));
      scb[gp] = __half2float(__ushort_as_half(static_cast<unsigned short>(lds16(s_sc + (c + gp) * R * 2 + (16 * warp + g + 8) * 2))));
    }
    {
      float d[GP][NT][4];
#pragma unroll
      for (int gp = 0; gp < GP; ++gp)
#pragma unroll
        for (int t = 0; t < NT; ++t) d[gp][t][0] = d[gp][t][1] = d[gp][t][2] = d[gp][t][3] = 0.f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        // tokens' k = 32jj + 4q + {0..3} and 32jj + 16 + 4q + {0..3}: byte 64jj + 8q (+32) of the 256-byte k range, in
        // the 64-k box (64jj + 8q) / 128, 16-byte chunk ((64jj + 8q) % 128) / 16 swizzled by the token row
        const int b0 = 64 * jj + 8 * q, b1 = b0 + 32;
        uint2 x0[GP][NT], x1[GP][NT];
#pragma unroll
        for (int gp = 0; gp < GP; ++gp) {
          const uint32_t xb = base + ((c + gp) % STAGES) * SB + WB;
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            const int tok = 8 * t + g;
            x0[gp][t] = lds64(xb + (b0 >> 7) * XB + tok * 128 + ((((b0 & 127) >> 4) ^ (tok & 7)) << 4) + (b0 & 15));
            x1[gp][t] = lds64(xb + (b1 >> 7) * XB + tok * 128 + ((((b1 & 127) >> 4) ^ (tok & 7)) << 4) + (b1 & 15));
          }
        }
#pragma unroll
        for (int gp = 0; gp < GP; ++gp)
#pragma unroll
          for (int t = 0; t < NT; ++t)
            if (CV) mma16816h(d[gp][t], lo[gp][jj][0], lo[gp][jj][1], lo[gp][jj][2], lo[gp][jj][3], x0[gp][t].x, x1[gp][t].x);
            else mma16816(d[gp][t], lo[gp][jj][0], lo[gp][jj][1], lo[gp][jj][2], lo[gp][jj][3], x0[gp][t].x, x1[gp][t].x);
#pragma unroll
        for (int gp = 0; gp < GP; ++gp)
#pragma unroll
          for (int t = 0; t < NT; ++t)
            if (CV) mma16816h(d[gp][t], hi[gp][jj][0], hi[gp][jj][1], hi[gp][jj][2], hi[gp][jj][3], x0[gp][t].y, x1[gp][t].y);
            else mma16816(d[gp][t], hi[gp][jj][0], hi[gp][jj][1], hi[gp][jj][2], hi[gp][jj][3], x0[gp][t].y, x1[gp][t].y);
      }
      // D: (row g, tok 2q), (g, 2q + 1), (g + 8, 2q), (g + 8, 2q + 1); groups fold into acc in k order
#pragma unroll
      for (int gp = 0; gp < GP; ++gp)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          acc[t][0] = ffma(d[gp][t][0], sca[gp], acc[t][0]);
          acc[t][1] = ffma(d[gp][t][1], sca[gp], acc[t][1]);
          acc[t][2] = ffma(d[gp][t][2], scb[gp], acc[t][2]);
          acc[t][3] = ffma(d[gp][t][3], scb[gp], acc[t][3]);
        }
    }
    __syncwarp();
    if (lane == 0)
#pragma unroll
      for (int gp = 0; gp < GP; ++gp) mbar_arrive(&empty[(c + gp) % STAGES]);
  }
  named_sync(1, 32 * W);  // every math warp is past the ring: reuse it for the output tile
  griddep_launch();

  const int M = p.M;
  unsigned char* smem_gen = smem_raw + (base - smem_u32(smem_raw));
  if (MODE == 0) {
    constexpr int RS = R * 2 + 16;  // bf16 tile [tokens][R], padded rows
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int t0 = 8 * t + 2 * q, r0 = 16 * warp + g;
      *reinterpret_cast<__nv_bfloat16*>(smem_gen + t0 * RS + r0 * 2) = __float2bfloat16_rn(acc[t][0]);
      *reinterpret_cast<__nv_bfloat16*>(smem_gen + (t0 + 1) * RS + r0 * 2) = __float2bfloat16_rn(acc[t][1]);
      *reinterpret_cast<__nv_bfloat16*>(smem_gen + t0 * RS + (r0 + 8) * 2) = __float2bfloat16_rn(acc[t][2]);
      *reinterpret_cast<__nv_bfloat16*>(smem_gen + (t0 + 1) * RS + (r0 + 8) * 2) = __float2bfloat16_rn(acc[t][3]);
    }
    named_sync(1, 32 * W);
    const int rows = p.N - row0 < R ? p.N - row0 : R, ch = rows / 8;
    for (int k = tid; k < M * ch; k += 32 * W) {
      const int tok = k / ch, cq = k % ch, col = row0 + 8 * cq;
      int sidx = 0;
#pragma unroll
      for (int s = 1; s < 4; ++s)
        if (s < p.nseg && col >= p.lo[s]) sidx = s;
      __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(p.dst[sidx]);
      *reinterpret_cast<uint4*>(d + static_cast<int64_t>(tok) * p.ld[sidx] + (col - p.lo[sidx])) =
          *reinterpret_cast<const uint4*>(smem_gen + tok * RS + 16 * cq);
    }
  } else {
    constexpr int RS = R * 4 + 16;  // fp32 tile [tokens][R], padded rows
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int t0 = 8 * t + 2 * q, r0 = 16 * warp + g;
      *reinterpret_cast<float*>(smem_gen + t0 * RS + r0 * 4) = acc[t][0];
      *reinterpret_cast<float*>(smem_gen + (t0 + 1) * RS + r0 * 4) = acc[t][1];
      *reinterpret_cast<float*>(smem_gen + t0 * RS + (r0 + 8) * 4) = acc[t][2];
      *reinterpret_cast<float*>(smem_gen + (t0 + 1) * RS + (r0 + 8) * 4) = acc[t][3];
    }
    named_sync(1, 32 * W);
    const int rows = p.N - row0 < R ? p.N - row0 : R, ch = rows / 4;
    float* part = reinterpret_cast<float*>(p.dst[0]) + static_cast<int64_t>(split) * M * p.N;
    for (int k = tid; k < M * ch; k += 32 * W) {
      const int tok = k / ch, cq = k % ch;
      *reinterpret_cast<uint4*>(part + static_cast<int64_t>(tok) * p.N + row0 + 4 * cq) =
          *reinterpret_cast<const uint4*>(smem_gen + tok * RS + 16 * cq);
    }
  }
}

// ---- host -----------------------------------------------------------------------------------------------------------
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeFn encode_fn() {
  static EncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &q));
    TORCH_CHECK(ptr != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled is unavailable");
    fn = reinterpret_cast<EncodeFn>(ptr);
  }
  return fn;
}

// row-major [outer, inner_bytes] in boxes of [box_outer rows, 128 bytes], 128-byte swizzle, rows past outer read 0
CUtensorMap make_map(void* ptr, CUtensorMapDataType dt, uint64_t inner, uint64_t outer, uint64_t row_bytes,
                     uint32_t box_inner, uint32_t box_outer) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {row_bytes};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, dt, 2, ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

template <int W, int NT, int GP, int MODE, int CV>
void launch_one(const CUtensorMap& mq, const CUtensorMap& mx, const Params& p, int gx, int gy) {
  using S = Shape<W, NT>;
  auto kernel = dense8_kernel<W, NT, GP, MODE, CV>;
  static uint64_t ready = 0;
  const int dev = at::cuda::current_device();
  if (!((ready >> dev) & 1)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::SMEM));
    ready |= 1ull << dev;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(gx, gy);
  cfg.blockDim = dim3(32 * (W + 1 + CV));
  cfg.dynamicSmemBytes = S::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, mq, mx, p));
}

template <int W, int GP, int MODE>
void launch_w(const CUtensorMap& mq, const CUtensorMap& mx, const Params& p, int gx, int gy, int nt) {
  switch (nt) {
    case 1: launch_one<W, 1, GP, MODE, CV>(mq, mx, p, gx, gy); break;
    case 2: launch_one<W, 2, GP, MODE, CV>(mq, mx, p, gx, gy); break;
    default: launch_one<W, 4, GP, MODE, CV>(mq, mx, p, gx, gy); break;
  }
}

// The served shapes' configurations only (dense8.py's tables): in_proj 6 math warps, qkv 5 (bf16 segments), o_proj /
// out_proj 4 (fp32 partials); two scale groups per iteration.
template <int MODE>
void launch(const CUtensorMap& mq, const CUtensorMap& mx, const Params& p, int gx, int gy, int w, int nt, int gp) {
  TORCH_CHECK(gp == GP, "dense8: groups per iteration must be ", GP);
  TORCH_CHECK((p.ks / GROUP) % GP == 0, "dense8: the K range must hold whole iterations");
  if (MODE == 0 && w == 6) {
    launch_w<6, GP, MODE>(mq, mx, p, gx, gy, nt);
  } else if (MODE == 0 && w == 5) {
    launch_w<5, GP, MODE>(mq, mx, p, gx, gy, nt);
  } else if (MODE == 1 && w == 4) {
    launch_w<4, GP, MODE>(mq, mx, p, gx, gy, nt);
  } else {
    TORCH_CHECK(false, "dense8: warps ", w, " are not built for this output mode");
  }
}

int tiles_of(int M) { return M <= 8 ? 1 : M <= 16 ? 2 : 4; }

void check_common(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& s, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == K,
              "dense8: x must be contiguous bf16 [M, ", K, "]");
  TORCH_CHECK(M >= 1 && M <= MAX_TOK, "dense8: 1..", MAX_TOK, " token rows, got ", M);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kChar && q.is_contiguous() && q.dim() == 2 && q.size(0) == N &&
                  q.size(1) == K, "dense8: q must be contiguous int8 [N, K]");
  TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kHalf && s.is_contiguous() && s.dim() == 2 && s.size(0) == K / GROUP &&
                  s.size(1) == N, "dense8: s must be contiguous fp16 [K / 128, N]");
  TORCH_CHECK(K % GROUP == 0 && N % 8 == 0, "dense8: K % 128 and N % 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(s.data_ptr()) % 16 == 0, "dense8: 16-byte aligned operands");
}

}  // namespace

// out segments: x [M, K] @ dequant(q, s).T, column c in [lo[i], lo[i + 1]) -> outs[i][row, c - lo[i]] (row stride ld[i])
void seg(torch::Tensor x, torch::Tensor q, torch::Tensor s, std::vector<torch::Tensor> outs, std::vector<int64_t> lo,
         std::vector<int64_t> ld, int64_t warps, int64_t pre, int64_t gp) {
  const int64_t M = x.size(0), N = q.size(0), K = q.size(1);
  check_common(x, q, s, M, N, K);
  TORCH_CHECK(!outs.empty() && outs.size() <= 4 && lo.size() == outs.size() && ld.size() == outs.size() && lo[0] == 0,
              "dense8: 1..4 output segments");
  TORCH_CHECK(K / GROUP <= 16, "dense8: K <= 2048 in one range");
  const at::cuda::CUDAGuard guard(x.device());
  Params p{};
  p.s = reinterpret_cast<const __half*>(s.data_ptr());
  for (size_t i = 0; i < outs.size(); ++i) {
    TORCH_CHECK(outs[i].scalar_type() == at::kBFloat16 && outs[i].is_contiguous() && lo[i] % 8 == 0 && ld[i] % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(outs[i].data_ptr()) % 16 == 0, "dense8: bf16 outputs, 8-column aligned");
    p.dst[i] = outs[i].data_ptr();
    p.lo[i] = static_cast<int>(lo[i]);
    p.ld[i] = static_cast<int>(ld[i]);
  }
  p.nseg = static_cast<int>(outs.size());
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.ks = static_cast<int>(K);
  p.pre = static_cast<int>(pre);
  const int nt = tiles_of(static_cast<int>(M)), R = 16 * static_cast<int>(warps);
  const CUtensorMap mq = make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N, K, 128, R);
  const CUtensorMap mx = make_map(x.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M, K * 2, 64, 8 * nt);
  launch<0>(mq, mx, p, static_cast<int>((N + R - 1) / R), 1, static_cast<int>(warps), nt, static_cast<int>(gp));
}

// fp32 split-K partials: part[i, row, c] = x[row, K_i] @ dequant(q, s)[c, K_i], K_i the i-th of `splits` ranges
void part(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor partial, int64_t splits, int64_t warps,
          int64_t pre, int64_t gp) {
  const int64_t M = x.size(0), N = q.size(0), K = q.size(1);
  check_common(x, q, s, M, N, K);
  TORCH_CHECK(splits >= 1 && K % (splits * GROUP) == 0 && K / splits / GROUP <= 16, "dense8: split ranges of 128 k");
  TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.is_contiguous() && partial.numel() >= splits * M * N &&
                  reinterpret_cast<uintptr_t>(partial.data_ptr()) % 16 == 0, "dense8: fp32 partial workspace");
  const at::cuda::CUDAGuard guard(x.device());
  Params p{};
  p.s = reinterpret_cast<const __half*>(s.data_ptr());
  p.dst[0] = partial.data_ptr();
  p.nseg = 1;
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.ks = static_cast<int>(K / splits);
  p.pre = static_cast<int>(pre);
  const int nt = tiles_of(static_cast<int>(M)), R = 16 * static_cast<int>(warps);
  const CUtensorMap mq = make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N, K, 128, R);
  const CUtensorMap mx = make_map(x.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M, K * 2, 64, 8 * nt);
  launch<1>(mq, mx, p, static_cast<int>((N + R - 1) / R), static_cast<int>(splits), static_cast<int>(warps), nt,
            static_cast<int>(gp));
}


}  // namespace qd8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_in_proj", &gdn_in_proj, "GDN in_proj_qkvz + in_proj_ba at decode widths");
  m.def("qkv_proj", &qkv_proj, "attention qkv_proj at decode widths");
  m.attr("MAX_T") = MAX_T;
  m.def("dense8_seg", &qd8::seg, "DENSE8 W8A16 projection into bf16 column segments (1..32 token rows)", py::arg("x"), py::arg("q"),
        py::arg("s"), py::arg("outs"), py::arg("lo"), py::arg("ld"), py::arg("warps"), py::arg("pre"), py::arg("gp") = qd8::GP);
  m.def("dense8_part", &qd8::part, "DENSE8 W8A16 projection as fp32 split-K partials (1..32 token rows)", py::arg("x"), py::arg("q"),
        py::arg("s"), py::arg("partial"), py::arg("splits"), py::arg("warps"), py::arg("pre"), py::arg("gp") = qd8::GP);
}
