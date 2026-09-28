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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_in_proj", &gdn_in_proj, "GDN in_proj_qkvz + in_proj_ba at decode widths");
  m.def("qkv_proj", &qkv_proj, "attention qkv_proj at decode widths");
  m.attr("MAX_T") = MAX_T;
}
