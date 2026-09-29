// GDN prefill input projection with the recurrent block's front in its epilogue (Qwen3.6-35B-A3B, H100).
//
// [q | k | v] = x W_qkv^T and z = x W_z^T run as one persistent BF16 GEMM whose arithmetic is cuBLAS's for
// these shapes (and DeepGEMM's, which gives the same bits): one fp32 accumulator per output, wgmma k16 steps
// in increasing K, one round-to-nearest bf16 rounding. The [q | k | v] tiles never store their GEMM output:
// their epilogue runs the causal conv, silu, the q / k L2 norm and the value heads' g / beta exactly as
// qk_conv_gate.cu does (bit for bit the Triton _conv_split_gate_kernel), and writes q, k, v, g, beta.
//
// Tiles are 256 rows x 128 columns (one head). A [q | k | v] tile computes rows [253 m - 3, 253 m + 253):
// its first three rows are the previous tile's last three, recomputed with the same K order and so the same
// bits, which gives every output token its three predecessor rows inside the tile. z tiles are plain 256-row
// tiles. Two CTAs of a cluster share each A tile by TMA multicast and compute adjacent heads.
//
// Roles (384 threads, one CTA per SM): warpgroups 0-1 run the WGMMA mainloop and stage each finished tile as
// bf16 rows in shared memory; warp 8 issues the TMA loads; warps 9-11 run the tile's epilogue from shared memory
// while the math warpgroups compute the next tile (named barriers hand the single staging buffer back and
// forth). No setmaxnreg: ptxas cannot bound the epilogue warps' path to a smaller register file.
//
// The conv state is read from `snap` (the batch's old states, copied by gdn_front_prep before this kernel)
// and written to `cs` by whichever warp holds a sequence's last token, so no ordering between CTAs is needed.
//
// Portions (wgmma wrapper, GMMA descriptors, TMA loads, mbarrier pipeline, producer / math warpgroup split,
// persistent tile loop) follow qk_moe_prefill.cu, which ports them from DeepGEMM's sm90 BF16 GEMM
// (Copyright (c) 2025 DeepSeek, MIT License; sgl-deep-gemm distribution Copyright 2023-2026 SGLang Team,
// Apache License 2.0).
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace qin {

using bf16 = __nv_bfloat16;
constexpr int kK = 2048;                 // hidden size
constexpr int kD = 8192;                 // [q | k | v] channels
constexpr int kZ = 4096;                 // z channels
constexpr int kHead = 128, kNQ = 16, kNK = 16, kNV = 32;
constexpr int kBM = 256, kBN = 128, kBK = 64, kStages = 3, kHalo = 3, kQkvStride = kBM - kHalo;
constexpr int kKB = kK / kBK;
constexpr int kQkvPairs = kD / kBN / 2, kZPairs = kZ / kBN / 2;  // n-pairs per m-tile: 32, 16
constexpr uint32_t kABytes = kBM * kBK * 2;                      // 32 KB
constexpr uint32_t kBBytes = kBN * kBK * 2;                      // 16 KB
constexpr int kDRow = kBN * 2 + 16;                              // staged row stride (bytes), bank-conflict free
constexpr uint32_t kDBytes = kBM * kDRow;
constexpr int kMaxSeqs = 256;
constexpr uint32_t kSmem = kStages * (kABytes + kBBytes) + kDBytes + 2 * kStages * 8 + 4 * (kMaxSeqs + 1) * 3 + 1024 + 1024 + 16;
constexpr uint64_t kEvictNormal = 0x1000000000000000ull;

// ---------------------------------------------------------------- PTX helpers (as in qk_moe_prefill.cu)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
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
__device__ __forceinline__ void tma_load_2d_mc(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask, uint64_t hint) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "h"(mask), "r"(c0), "r"(c1), "l"(hint)
      : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ uint64_t gmma_desc(uint32_t smem_addr) {
  return static_cast<uint64_t>((smem_addr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(1024 >> 4) << 32) |
         (static_cast<uint64_t>(1) << 62);
}
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

// wgmma.mma_async m64n128k16 f32 += bf16 x bf16, both operands K-major in shared memory.
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
      "}, "
      "%64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1));
}

// ---------------------------------------------------------------- the front's arithmetic (qk_conv_gate.cu)
__device__ __forceinline__ float f_add(float a, float b) { float d; asm("add.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_sub(float a, float b) { float d; asm("sub.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_mul(float a, float b) { float d; asm("mul.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_fma(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float f_div(float a, float b) { float d; asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_ex2(float a) { float d; asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_ex2_ftz(float a) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_rcp_ftz(float a) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_sqrt_ftz(float a) { float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ uint32_t bf_fma2z(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0u));
  return d;
}
__device__ __forceinline__ float lo2f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi2f(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float v) { uint16_t d; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(v)); return d; }
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
constexpr float kLog2e = 1.44269502162933349609375f;
__device__ __forceinline__ float t_exp(float x) { return f_ex2(f_mul(x, kLog2e)); }
// acc / (1 + ex2(-acc log2 e)) as div.full computes it, for every fp32 acc (exh.cu).
__device__ __forceinline__ float silu(float acc) {
  const float r4 = f_rcp_ftz(f_fma(f_ex2_ftz(f_mul(acc, -kLog2e)), 0.25f, 0.25f));
  return f_mul(f_mul(acc, r4), 0.25f);
}
__device__ __forceinline__ float nv_logf(float v) {
  float r;
  asm("{\n"
      ".reg .pred pd, pinf, pz;\n"
      ".reg .f32 m, ea, e, f, p, t;\n"
      ".reg .b32 bi, ex, mi;\n"
      "setp.lt.f32 pd, %1, 0f00800000;\n"
      "mul.f32 t, %1, 0f4B000000;\n"
      "selp.f32 m, t, %1, pd;\n"
      "selp.f32 ea, 0fC1B80000, 0f00000000, pd;\n"
      "mov.b32 bi, m;\n"
      "add.s32 ex, bi, -1059760811;\n"
      "and.b32 ex, ex, -8388608;\n"
      "sub.s32 mi, bi, ex;\n"
      "cvt.rn.f32.s32 e, ex;\n"
      "fma.rn.ftz.f32 e, e, 0f34000000, ea;\n"
      "mov.b32 f, mi;\n"
      "add.f32 f, f, 0fBF800000;\n"
      "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n"
      "mul.f32 p, f, p;\n"
      "fma.rn.ftz.f32 p, p, f, f;\n"
      "fma.rn.ftz.f32 %0, e, 0f3F317218, p;\n"
      "setp.lt.u32 pinf, bi, 2139095040;\n"
      "@!pinf fma.rn.ftz.f32 %0, m, 0f7F800000, 0f7F800000;\n"
      "setp.eq.f32 pz, m, 0f00000000;\n"
      "@pz mov.f32 %0, 0fFF800000;\n"
      "}\n"
      : "=f"(r)
      : "f"(v));
  return r;
}
__device__ __forceinline__ float nv_expf(float x) {
  float r;
  asm("{\n"
      ".reg .f32 t, j, k, q, e;\n"
      ".reg .b32 s;\n"
      "fma.rn.ftz.f32 t, %1, 0f3BBB989D, 0f3F000000;\n"
      "cvt.ftz.sat.f32.f32 t, t;\n"
      "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n"
      "add.f32 k, j, 0fCB40007F;\n"
      "neg.f32 k, k;\n"
      "fma.rn.ftz.f32 q, %1, 0f3FB8AA3B, k;\n"
      "fma.rn.ftz.f32 q, %1, 0f32A57060, q;\n"
      "mov.b32 s, j;\n"
      "shl.b32 s, s, 23;\n"
      "ex2.approx.ftz.f32 e, q;\n"
      "mov.b32 j, s;\n"
      "mul.f32 %0, e, j;\n"
      "}\n"
      : "=f"(r)
      : "f"(x));
  return r;
}

__device__ __forceinline__ int ld_meta(const void* ptr, int i, bool is64) {
  return is64 ? static_cast<int>(static_cast<const int64_t*>(ptr)[i]) : static_cast<const int*>(ptr)[i];
}

struct Args {
  const bf16* w_conv;   // [kD, 4]
  bf16* cs;             // conv states [slots, kD, 3]
  const bf16* snap;     // [B, kD, 3]: sequence s's old state when prefix(s) > 0 and slot(s) >= 0
  const void* qsl;      // [B + 1]
  const void* idx;      // [B]
  const void* prefix;   // [B]
  int wide;             // bit 0 / 1 / 2: qsl / idx / prefix are int64
  const bf16* a;        // [T, kNV] rows ab_stride apart
  const bf16* b;
  int ab_stride;
  const float* alog;    // [kNV]
  const float* dtb;
  bf16* q;              // [T, kNQ, kHead]
  bf16* k;
  bf16* v;              // [T, kNV, kHead]
  bf16* z;              // [T, kZ]
  float* g;             // [T, kNV]
  float* beta;
  int T, B, n_qkv_m, n_z_m, n_units;
};

constexpr int GROUP = 8;  // n-pairs per group; a group sweeps every m-tile before the next group (weights stay in L2)

// Pair unit u: the cluster's m-tile and n-pair; CTA `rank` takes n-block 2 * pair + rank.
__device__ __forceinline__ void decode_unit(const Args& p, int u, uint32_t rank, bool& qkv, int& m, int& nb) {
  const int nq = p.n_qkv_m * kQkvPairs;
  int pair;
  if (u < nq) {
    qkv = true;
    {
      const int per = GROUP * p.n_qkv_m, g = u / per, r = u % per;
      m = r / GROUP, pair = g * GROUP + r % GROUP;
    }
  } else {
    u -= nq;
    qkv = false;
    {
      const int per = GROUP * p.n_z_m, g = u / per, r = u % per;
      m = r / GROUP, pair = g * GROUP + r % GROUP;
    }
  }
  nb = 2 * pair + static_cast<int>(rank);
}

// The conv front of one warp over its staged rows of a [q | k | v] tile, one token per call to step() so it
// can run between the next tile's k-blocks: the window and the taps stay in registers. Staged row r is token
// 253 m - 3 + r; lane l holds channels 4 l .. 4 l + 3 of the head.
struct Front {
  int t, t_hi, row0_tok, s, start, end, ostride, c;
  bool use, qk, active;
  bf16* dst;
  uint32_t cp[3][2];
  uint32_t wp2[4][2];
};

__device__ __forceinline__ uint2 staged_row(const uint8_t* sD, const Front& f, int t, int lane) {
  return *reinterpret_cast<const uint2*>(sD + (t - f.row0_tok) * kDRow + lane * 8);
}

__device__ __forceinline__ void front_window(const Args& p, const uint8_t* sD, Front& f, int t, int lane) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int tt = t - 3 + i;
    if (tt >= f.start) {
      const uint2 u = staged_row(sD, f, tt, lane);
      f.cp[i][0] = u.x, f.cp[i][1] = u.y;
    } else if (f.use) {
      const bf16* sb = p.snap + static_cast<int64_t>(f.s) * (kD * 3) + static_cast<int64_t>(f.c) * 3 + (tt - f.start + 3);
      const uint32_t v0 = __bfloat16_as_ushort(sb[0]), v1 = __bfloat16_as_ushort(sb[3]);
      const uint32_t v2 = __bfloat16_as_ushort(sb[6]), v3 = __bfloat16_as_ushort(sb[9]);
      f.cp[i][0] = v0 | (v1 << 16), f.cp[i][1] = v2 | (v3 << 16);
    } else {
      f.cp[i][0] = 0u, f.cp[i][1] = 0u;
    }
  }
}

__device__ __forceinline__ void front_init(const Args& p, const uint8_t* sD, const int* s_qs, const int* s_slot,
                                           const int* s_hinit, const uint2* sW, Front& f, int m, int head, int r0,
                                           int r1, int lane) {
  f.row0_tok = kQkvStride * m - kHalo;
  f.t = max(f.row0_tok + r0, s_qs[0]);
  f.t_hi = min(f.row0_tok + r1, s_qs[p.B]);
  f.active = f.t < f.t_hi;
  if (!f.active) return;
  f.c = head * kHead + 4 * lane;
  {
    const uint2* wp = sW + 4 * lane;
    uint2 u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) u[j] = wp[j];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int sh = (i & 1) * 16;
      const uint32_t w0 = ((i < 2 ? u[0].x : u[0].y) >> sh) & 0xffff, w1 = ((i < 2 ? u[1].x : u[1].y) >> sh) & 0xffff;
      const uint32_t w2 = ((i < 2 ? u[2].x : u[2].y) >> sh) & 0xffff, w3 = ((i < 2 ? u[3].x : u[3].y) >> sh) & 0xffff;
      f.wp2[i][0] = w0 | (w1 << 16);
      f.wp2[i][1] = w2 | (w3 << 16);
    }
  }
  const bool is_q = head < kNQ, is_k = !is_q && head < kNQ + kNK;
  f.qk = is_q || is_k;
  f.dst = is_q ? p.q + head * kHead : is_k ? p.k + (head - kNQ) * kHead : p.v + (head - kNQ - kNK) * kHead;
  f.ostride = (is_q ? kNQ : is_k ? kNK : kNV) * kHead;
  int s = 0;
  while (s < p.B && s_qs[s + 1] <= f.t) ++s;
  f.s = s, f.start = s_qs[s], f.end = s_qs[s + 1];
  f.use = s_hinit[s] != 0 && s_slot[s] >= 0;
  front_window(p, sD, f, f.t, lane);
}

__device__ __forceinline__ void front_step(const Args& p, const uint8_t* sD, const int* s_qs, const int* s_slot,
                                           const int* s_hinit, Front& f, int lane) {
  if (!f.active || f.t >= f.t_hi) return;
  const int t = f.t;
  if (t >= f.end) {
    // The next non-empty sequence starts at t.
    int s = f.s;
    do {
      ++s;
    } while (s_qs[s + 1] <= t);
    f.s = s, f.start = s_qs[s], f.end = s_qs[s + 1];
    f.use = s_hinit[s] != 0 && s_slot[s] >= 0;
    front_window(p, sD, f, t, lane);
  }
  const uint2 xu = staged_row(sD, f, t, lane);
  const uint32_t x2[2] = {xu.x, xu.y};
  float y[4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t q0 = bf_fma2z(f.cp[0][h], f.wp2[0][h]), q1 = bf_fma2z(f.cp[1][h], f.wp2[1][h]);
    const uint32_t q2 = bf_fma2z(f.cp[2][h], f.wp2[2][h]), q3 = bf_fma2z(x2[h], f.wp2[3][h]);
    float lo = f_add(lo2f(q0), lo2f(q1)), hi = f_add(hi2f(q0), hi2f(q1));
    lo = f_add(lo, lo2f(q2)), hi = f_add(hi, hi2f(q2));
    lo = f_add(lo, lo2f(q3)), hi = f_add(hi, hi2f(q3));
    y[2 * h] = silu(lo);
    y[2 * h + 1] = silu(hi);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) f.cp[0][h] = f.cp[1][h], f.cp[1][h] = f.cp[2][h], f.cp[2][h] = x2[h];
  uint2 o = make_uint2(f2bf2(y[0], y[1]), f2bf2(y[2], y[3]));
  if (f.qk) {
    const float z0 = lo2f(o.x), z1 = hi2f(o.x), z2 = lo2f(o.y), z3 = hi2f(o.y);
    float sq = f_mul(z1, z1);
    sq = f_fma(z0, z0, sq);
    sq = f_fma(z2, z2, sq);
    sq = f_fma(z3, z3, sq);
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) sq = f_add(sq, __shfl_xor_sync(0xffffffffu, sq, off));
    const float rr = f_rcp_ftz(f_sqrt_ftz(f_add(sq, 1e-6f)));
    o = make_uint2(f2bf2(f_mul(z0, rr), f_mul(z1, rr)), f2bf2(f_mul(z2, rr), f_mul(z3, rr)));
  }
  *reinterpret_cast<uint2*>(f.dst + static_cast<int64_t>(t) * f.ostride + 4 * lane) = o;
  if (t == f.end - 1 && s_slot[f.s] >= 0) {
    // New conv state: the raw last three input rows, or the old state shifted by the length when shorter.
    const int L = f.end - f.start;
    bf16* sbw = p.cs + static_cast<int64_t>(s_slot[f.s]) * (kD * 3) + static_cast<int64_t>(f.c) * 3;
    uint16_t nv[3][4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int tt = f.end - 3 + i;
      if (tt >= f.start) {
        const uint2 u = staged_row(sD, f, tt, lane);
        nv[i][0] = u.x & 0xffff, nv[i][1] = u.x >> 16, nv[i][2] = u.y & 0xffff, nv[i][3] = u.y >> 16;
      } else {
        const int oc = i + L;
        const bf16* sb = p.snap + static_cast<int64_t>(f.s) * (kD * 3) + static_cast<int64_t>(f.c) * 3;
#pragma unroll
        for (int j = 0; j < 4; ++j) nv[i][j] = s_hinit[f.s] != 0 && oc < 3 ? __bfloat16_as_ushort(sb[j * 3 + oc]) : 0;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 3; ++i) sbw[j * 3 + i] = __ushort_as_bfloat16(nv[i][j]);
  }
  f.t = t + 1;
}


// Four tokens of one sequence at once (none its sequence's last, all from staged rows or the carried window),
// the four chains interleaved; the same arithmetic as four front_step calls.
__device__ __forceinline__ bool front_step4(const uint8_t* sD, Front& f, int lane) {
  const int t = f.t;
  if (!(f.active && t + 4 <= f.t_hi && t >= f.start && t + 3 < f.end - 1)) return false;
  uint32_t xr[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint2 u = staged_row(sD, f, t + j, lane);
    xr[j][0] = u.x, xr[j][1] = u.y;
  }
  uint2 o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float y[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t w0 = j == 0 ? f.cp[0][h] : j == 1 ? f.cp[1][h] : j == 2 ? f.cp[2][h] : xr[0][h];
      const uint32_t w1 = j == 0 ? f.cp[1][h] : j == 1 ? f.cp[2][h] : j == 2 ? xr[0][h] : xr[1][h];
      const uint32_t w2 = j == 0 ? f.cp[2][h] : j == 1 ? xr[0][h] : j == 2 ? xr[1][h] : xr[2][h];
      const uint32_t q0 = bf_fma2z(w0, f.wp2[0][h]), q1 = bf_fma2z(w1, f.wp2[1][h]);
      const uint32_t q2 = bf_fma2z(w2, f.wp2[2][h]), q3 = bf_fma2z(xr[j][h], f.wp2[3][h]);
      float lo = f_add(lo2f(q0), lo2f(q1)), hi = f_add(hi2f(q0), hi2f(q1));
      lo = f_add(lo, lo2f(q2)), hi = f_add(hi, hi2f(q2));
      lo = f_add(lo, lo2f(q3)), hi = f_add(hi, hi2f(q3));
      y[2 * h] = silu(lo);
      y[2 * h + 1] = silu(hi);
    }
    o[j] = make_uint2(f2bf2(y[0], y[1]), f2bf2(y[2], y[3]));
  }
  if (f.qk) {
    float sq[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float z0 = lo2f(o[j].x), z1 = hi2f(o[j].x), z2 = lo2f(o[j].y), z3 = hi2f(o[j].y);
      sq[j] = f_mul(z1, z1);
      sq[j] = f_fma(z0, z0, sq[j]);
      sq[j] = f_fma(z2, z2, sq[j]);
      sq[j] = f_fma(z3, z3, sq[j]);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
#pragma unroll
      for (int j = 0; j < 4; ++j) sq[j] = f_add(sq[j], __shfl_xor_sync(0xffffffffu, sq[j], off));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float rr = f_rcp_ftz(f_sqrt_ftz(f_add(sq[j], 1e-6f)));
      const float z0 = lo2f(o[j].x), z1 = hi2f(o[j].x), z2 = lo2f(o[j].y), z3 = hi2f(o[j].y);
      o[j] = make_uint2(f2bf2(f_mul(z0, rr), f_mul(z1, rr)), f2bf2(f_mul(z2, rr), f_mul(z3, rr)));
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) *reinterpret_cast<uint2*>(f.dst + static_cast<int64_t>(t + j) * f.ostride + 4 * lane) = o[j];
#pragma unroll
  for (int h = 0; h < 2; ++h) f.cp[0][h] = xr[1][h], f.cp[1][h] = xr[2][h], f.cp[2][h] = xr[3][h];
  f.t = t + 4;
  return true;
}

// g and beta of value head `head` for the tokens of staged rows [r0, r1) (one token per lane), from the a / b
// values staged with the tile (sAB[r]: a in the low half, b in the high half).
__device__ __forceinline__ void gbeta_rows(const Args& p, const int* s_qs, const uint32_t* sAB, int m, int head, int r0,
                                           int r1, int lane) {
  const int row0_tok = kQkvStride * m - kHalo;
  const int t_lo = max(row0_tok + r0, s_qs[0]), t_hi = min(row0_tok + r1, s_qs[p.B]);
  const int hv = head - kNQ - kNK;
  for (int t = t_lo + lane; t < t_hi; t += 32) {
  const uint32_t ab = sAB[t - row0_tok];
  const float av = __uint_as_float(ab << 16), bv = __uint_as_float(ab & 0xffff0000u);
  const float xg = f_add(av, p.dtb[hv]);
  const float lg = nv_logf(f_add(t_exp(xg), 1.f));
  const float sp = xg <= 20.f ? lg : xg;
  const float gv = f_mul(f_sub(0.f, t_exp(p.alog[hv])), sp);
  p.g[static_cast<int64_t>(t) * kNV + hv] = nv_expf(gv);
  const float sg = f_div(1.f, f_add(t_exp(f_sub(0.f, bv)), 1.f));
  p.beta[static_cast<int64_t>(t) * kNV + hv] = bf2f(f2bf(sg));
  }
}

// 384 threads: warpgroups 0-1 run WGMMA (warpgroup w: tile rows [64 w, 64 w + 64) and [128 + 64 w, +64)) and
// the epilogue, warpgroup 2 issues the TMA loads (one thread).
__global__ void __launch_bounds__(384, 1)
    inproj_kernel(const Args p, const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_w) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + kStages * kABytes;
  uint8_t* sD = sB + kStages * kBBytes;
  uint64_t* full = reinterpret_cast<uint64_t*>(sD + kDBytes);
  uint64_t* empty = full + kStages;
  int* s_qs = reinterpret_cast<int*>(empty + kStages);
  int* s_slot = s_qs + (kMaxSeqs + 1);
  int* s_hinit = s_slot + (kMaxSeqs + 1);
  // pending head's conv taps [128][4] bf16 (16-byte aligned)
  uint2* sW = reinterpret_cast<uint2*>((reinterpret_cast<uintptr_t>(s_hinit + (kMaxSeqs + 1)) + 15) & ~uintptr_t{15});
  uint32_t* sAB = reinterpret_cast<uint32_t*>(sW + 128);             // pending tile's (a | b << 16) per staged row
  const uint32_t rank = cluster_rank();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 16);
    }
    fence_barrier_init();
  }
  if (threadIdx.x == 32) {
    prefetch_tmap(&tm_x);
    prefetch_tmap(&tm_w);
  }
  for (int i = threadIdx.x; i <= p.B; i += blockDim.x) {
    s_qs[i] = ld_meta(p.qsl, i, p.wide & 1);
    if (i < p.B) {
      s_slot[i] = ld_meta(p.idx, i, p.wide & 2);
      s_hinit[i] = ld_meta(p.prefix, i, p.wide & 4) > 0;
    }
  }
  __syncthreads();
  cluster_sync();
  const int unit0 = static_cast<int>(cluster_id_x()), unit_step = static_cast<int>(n_clusters_x());

  // Named barriers between the math warpgroups (256 threads) and the epilogue warps (96 threads):
  // kBarStaged: a tile is staged; kBarFree: the staging buffer (and sW / sAB) may be overwritten.
  constexpr int kBarStaged = 2, kBarFree = 3, kSync = 256 + 96;
  if (threadIdx.x >= 256) {

    const int ew = threadIdx.x / 32 - 9, lane = threadIdx.x % 32;  // epilogue warp 0..2 (warps 9-11)
    if (threadIdx.x == 256) {
      int stage = 0;
      uint32_t phase = 0;
      for (int u = unit0; u < p.n_units; u += unit_step) {
        bool qkv;
        int m, nb;
        decode_unit(p, u, rank, qkv, m, nb);
        const int a_row = (qkv ? kQkvStride * m - kHalo : kBM * m) + 128 * static_cast<int>(rank);
        const int b_row = (qkv ? 0 : kD) + nb * kBN;
        for (int kb = 0; kb < kKB; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          mbar_arrive_expect_tx(&full[stage], kABytes + kBBytes);
          tma_load_2d_mc(smem_u32(sA + stage * kABytes) + rank * (kABytes / 2), &tm_x, &full[stage], kb * kBK, a_row, 3,
                         kEvictNormal);
          tma_load_2d(smem_u32(sB + stage * kBBytes), &tm_w, &full[stage], kb * kBK, b_row, kEvictNormal);
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    } else if (ew >= 0) {
      // Epilogue warps: the staging buffer starts free; then per tile wait until staged, run the epilogue, free it.
      named_bar_arrive(kBarFree, kSync);
      for (int u = unit0; u < p.n_units; u += unit_step) {
        bool qkv;
        int m, nb;
        decode_unit(p, u, rank, qkv, m, nb);
        named_bar_sync(kBarStaged, kSync);
        if (qkv) {
          // Warp ew takes staged rows [3 + 85 ew, +85) (the last warp 83).
          const int r0 = kHalo + 85 * ew, r1 = min(r0 + 85, kBM);
          Front f;
          front_init(p, sD, s_qs, s_slot, s_hinit, sW, f, m, nb, r0, r1, lane);
          while (f.active && f.t < f.t_hi) {
            if (!front_step4(sD, f, lane)) front_step(p, sD, s_qs, s_slot, s_hinit, f, lane);
          }
          if (nb >= kNQ + kNK) gbeta_rows(p, s_qs, sAB, m, nb, r0, r1, lane);
        } else {
          // z rows: warp ew copies staged rows ew, ew + 3, ... (8 bytes per lane per row).
          for (int r = ew; r < kBM; r += 3) {
            const int t = kBM * m + r;
            if (t < p.T) {
              const uint2 val = *reinterpret_cast<const uint2*>(sD + r * kDRow + lane * 8);
              *reinterpret_cast<uint2*>(p.z + static_cast<int64_t>(t) * kZ + nb * kBN + lane * 4) = val;
            }
          }
        }
        named_bar_arrive(kBarFree, kSync);
      }
    }
  } else {

    const int wg = threadIdx.x / 128, warp = threadIdx.x / 32, lane = threadIdx.x % 32, wi = warp % 4;
    const uint32_t a_base = smem_u32(sA) + wg * 64 * 128;
    const uint32_t b_base = smem_u32(sB);
    int stage = 0;
    uint32_t phase = 0;
    for (int u = unit0; u < p.n_units; u += unit_step) {
      bool qkv;
      int m, nb;
      decode_unit(p, u, rank, qkv, m, nb);
      float acc0[64], acc1[64];
#pragma unroll
      for (int i = 0; i < 64; ++i) acc0[i] = 0.f, acc1[i] = 0.f;
      // This tile's epilogue inputs, loaded now and staged with the tile: conv taps (4 bytes per thread) and,
      // for value heads, the a / b pair of staged row threadIdx.x.
      uint32_t w_pf = 0, ab_pf = 0;
      if (qkv) {
        w_pf = reinterpret_cast<const uint32_t*>(p.w_conv + static_cast<int64_t>(nb) * kHead * 4)[threadIdx.x];
        const int t = kQkvStride * m - kHalo + static_cast<int>(threadIdx.x);
        if (nb >= kNQ + kNK && t >= 0 && t < p.T) {
          const int hv = nb - kNQ - kNK;
          ab_pf = static_cast<uint32_t>(__bfloat16_as_ushort(p.a[static_cast<int64_t>(t) * p.ab_stride + hv])) |
                  (static_cast<uint32_t>(__bfloat16_as_ushort(p.b[static_cast<int64_t>(t) * p.ab_stride + hv])) << 16);
        }
      }
      for (int kb = 0; kb < kKB; ++kb) {
        mbar_wait(&full[stage], phase);
#pragma unroll
        for (int i = 0; i < 64; ++i) fence_operand(acc0[i]), fence_operand(acc1[i]);
        wgmma_fence();
        const uint32_t a_st = a_base + stage * kABytes, b_st = b_base + stage * kBBytes;
#pragma unroll
        for (int k = 0; k < kBK / 16; ++k) wgmma_m64n128k16(acc0, gmma_desc(a_st + k * 32), gmma_desc(b_st + k * 32));
#pragma unroll
        for (int k = 0; k < kBK / 16; ++k)
          wgmma_m64n128k16(acc1, gmma_desc(a_st + 128 * 128 + k * 32), gmma_desc(b_st + k * 32));
        wgmma_commit();
#pragma unroll
        for (int i = 0; i < 64; ++i) fence_operand(acc0[i]), fence_operand(acc1[i]);
        wgmma_wait0();
        if (lane < 2) mbar_arrive_cluster(&empty[stage], lane);
        stage = stage + 1 == kStages ? 0 : stage + 1;
        phase ^= stage == 0;
      }
      // Stage the tile as bf16 rows (the GEMM's own rounding) once the epilogue warps have released the buffer.
      named_bar_sync(kBarFree, kSync);
      {
        const int r_a = 64 * wg + 16 * wi + lane / 4, col = 2 * (lane % 4);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          uint8_t* d0 = sD + r_a * kDRow + (8 * i + col) * 2;
          *reinterpret_cast<uint32_t*>(d0) = f2bf2(acc0[4 * i], acc0[4 * i + 1]);
          *reinterpret_cast<uint32_t*>(d0 + 8 * kDRow) = f2bf2(acc0[4 * i + 2], acc0[4 * i + 3]);
          *reinterpret_cast<uint32_t*>(d0 + 128 * kDRow) = f2bf2(acc1[4 * i], acc1[4 * i + 1]);
          *reinterpret_cast<uint32_t*>(d0 + 136 * kDRow) = f2bf2(acc1[4 * i + 2], acc1[4 * i + 3]);
        }
        reinterpret_cast<uint32_t*>(sW)[threadIdx.x] = w_pf;
        sAB[threadIdx.x] = ab_pf;
      }
      named_bar_arrive(kBarStaged, kSync);
    }
    // The epilogue warps' last arrival on kBarFree has no consumer: take it so the barrier ends balanced.
    named_bar_sync(kBarFree, kSync);
  }
  // No CTA leaves while its peer may still multicast into it or arrive on its barriers.
  cluster_sync();
}

// ---------------------------------------------------------------- the extend's inputs around the recurrence
// slots = cache index, or the last state slot for a sequence without one (int64); cu_seqlens as int64;
// initial_state = ssm_states[slots]; and snap[s] = conv state of sequence s before this extend (when it has
// a prefix and a slot), which the fused kernel reads while it writes the new states.
__global__ void __launch_bounds__(256) front_prep_kernel(const void* idx, int idx64, const void* qsl, int qsl64,
                                                         const void* prefix, int prefix64, int B, int64_t last_slot,
                                                         const float4* __restrict__ states, int64_t per_slot4,
                                                         const uint4* __restrict__ cs, uint4* __restrict__ snap,
                                                         int64_t* __restrict__ slots_out, int64_t* __restrict__ cu_out,
                                                         float4* __restrict__ init) {
  if (blockIdx.x == 0) {
    for (int b = threadIdx.x; b <= B; b += blockDim.x) {
      cu_out[b] = ld_meta(qsl, b, qsl64);
      if (b < B) {
        const int s = ld_meta(idx, b, idx64);
        slots_out[b] = s >= 0 ? static_cast<int64_t>(s) : last_slot;
      }
    }
  }
  const int64_t total = static_cast<int64_t>(B) * per_slot4;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(i / per_slot4);
    const int s = ld_meta(idx, b, idx64);
    const int64_t slot = s >= 0 ? static_cast<int64_t>(s) : last_slot;
    init[i] = __ldcs(states + slot * per_slot4 + (i - static_cast<int64_t>(b) * per_slot4));
  }
  constexpr int64_t kSnap16 = static_cast<int64_t>(kD) * 3 * 2 / 16;  // one state in 16-byte words
  const int64_t snap_total = static_cast<int64_t>(B) * kSnap16;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < snap_total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(i / kSnap16);
    const int s = ld_meta(idx, b, idx64);
    if (s >= 0 && ld_meta(prefix, b, prefix64) > 0) snap[i] = cs[static_cast<int64_t>(s) * kSnap16 + (i - b * kSnap16)];
  }
}

// ---------------------------------------------------------------- host
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeFn encode_fn() {
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

// Row-major bf16 [outer, inner] in boxes of [box_outer rows, 64 columns], 128B swizzle.
static CUtensorMap make_map(const void* base, uint64_t inner, uint64_t outer, uint64_t row_stride, uint32_t box_outer) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {row_stride * 2};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kBK), box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides,
                                 box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

static bool meta64(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong), name,
              " must be a contiguous CUDA int32 / int64 tensor");
  return t.scalar_type() == at::kLong;
}

// (slots int64 [B], cu_seqlens int64 [B + 1], initial_state [B, ...] = ssm_states[slots], snap [B, kD, 3]).
std::vector<torch::Tensor> front_prep(torch::Tensor idx, torch::Tensor qsl, torch::Tensor prefix, torch::Tensor ssm_states,
                                      torch::Tensor cs) {
  const bool idx64 = meta64(idx, "idx"), qsl64 = meta64(qsl, "qsl"), prefix64 = meta64(prefix, "prefix");
  TORCH_CHECK(ssm_states.is_cuda() && ssm_states.is_contiguous() && ssm_states.scalar_type() == at::kFloat, "ssm_states fp32");
  TORCH_CHECK(cs.scalar_type() == at::kBFloat16 && cs.is_contiguous() && cs.size(-1) == 3 && cs.size(-2) == kD, "conv state");
  const int64_t B = idx.numel(), per_slot = ssm_states.numel() / ssm_states.size(0);
  TORCH_CHECK(qsl.numel() == B + 1 && prefix.numel() == B && per_slot % 4 == 0 && B <= kMaxSeqs, "shapes");
  const at::cuda::CUDAGuard guard(ssm_states.device());
  auto slots = torch::empty({B}, idx.options().dtype(at::kLong));
  auto cu = torch::empty({B + 1}, idx.options().dtype(at::kLong));
  std::vector<int64_t> shape(ssm_states.sizes().begin(), ssm_states.sizes().end());
  shape[0] = B;
  auto init = torch::empty(shape, ssm_states.options());
  auto snap = torch::empty({B, kD, 3}, cs.options());
  const int64_t total4 = B * (per_slot / 4);
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>((total4 + 255) / 256, 1024)));
  front_prep_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      idx.data_ptr(), idx64, qsl.data_ptr(), qsl64, prefix.data_ptr(), prefix64, static_cast<int>(B),
      ssm_states.size(0) - 1, reinterpret_cast<const float4*>(ssm_states.data_ptr<float>()), per_slot / 4,
      reinterpret_cast<const uint4*>(cs.data_ptr()), reinterpret_cast<uint4*>(snap.data_ptr()),
      slots.data_ptr<int64_t>(), cu.data_ptr<int64_t>(), reinterpret_cast<float4*>(init.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {slots, cu, init, snap};
}

// x [T, 2048] and w_qkvz [12288, 2048] (rows [q | k | v | z]): returns (q, k, v, z, g, beta), z written into
// the caller's `z`, with the conv states of the batch's sequences updated in place. a, b: [T, 32] views of in_proj_ba's output.
std::vector<torch::Tensor> inproj_front(torch::Tensor x, torch::Tensor w_qkvz, torch::Tensor w_conv, torch::Tensor cs,
                                        torch::Tensor snap, torch::Tensor qsl, torch::Tensor idx, torch::Tensor prefix,
                                        torch::Tensor a, torch::Tensor b, torch::Tensor alog, torch::Tensor dtb,
                                        torch::Tensor z) {
  const int64_t T = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == kK, "x");
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range");
  TORCH_CHECK(w_qkvz.scalar_type() == at::kBFloat16 && w_qkvz.is_contiguous() && w_qkvz.size(0) == kD + kZ &&
                  w_qkvz.size(1) == kK, "w_qkvz [12288, 2048]");
  TORCH_CHECK(w_conv.scalar_type() == at::kBFloat16 && w_conv.is_contiguous() && w_conv.numel() == kD * 4, "conv weights");
  TORCH_CHECK(cs.scalar_type() == at::kBFloat16 && cs.is_contiguous() && cs.size(-1) == 3 && cs.size(-2) == kD, "conv state");
  const bool q64 = meta64(qsl, "qsl"), i64 = meta64(idx, "idx"), p64 = meta64(prefix, "prefix");
  const int64_t B = qsl.numel() - 1;
  TORCH_CHECK(B >= 1 && B <= kMaxSeqs && idx.numel() == B && prefix.numel() == B, "batch of 1..256 sequences");
  TORCH_CHECK(snap.scalar_type() == at::kBFloat16 && snap.is_contiguous() && snap.numel() == B * kD * 3, "snap");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && a.dim() == 2 && b.dim() == 2 &&
                  a.size(1) == kNV && b.size(1) == kNV && a.stride(1) == 1 && b.stride(1) == 1 && a.stride(0) == b.stride(0) &&
                  a.size(0) == T && b.size(0) == T, "a, b: [T, 32] rows with one stride");
  TORCH_CHECK(alog.scalar_type() == at::kFloat && dtb.scalar_type() == at::kFloat && alog.numel() == kNV && dtb.numel() == kNV,
              "A_log / dt_bias fp32 [32]");
  const at::cuda::CUDAGuard guard(x.device());
  auto q = torch::empty({1, T, kNQ, kHead}, x.options());
  auto k = torch::empty({1, T, kNK, kHead}, x.options());
  auto v = torch::empty({1, T, kNV, kHead}, x.options());
  TORCH_CHECK(z.scalar_type() == at::kBFloat16 && z.is_contiguous() && z.numel() == T * kZ && z.device() == x.device(),
              "z: contiguous bf16 [T, 4096]");
  auto g = torch::empty({1, T, kNV}, x.options().dtype(at::kFloat));
  auto beta = torch::empty({1, T, kNV}, x.options().dtype(at::kFloat));
  Args p;
  p.w_conv = reinterpret_cast<const bf16*>(w_conv.data_ptr());
  p.cs = reinterpret_cast<bf16*>(cs.data_ptr());
  p.snap = reinterpret_cast<const bf16*>(snap.data_ptr());
  p.qsl = qsl.data_ptr(), p.idx = idx.data_ptr(), p.prefix = prefix.data_ptr();
  p.wide = (q64 ? 1 : 0) | (i64 ? 2 : 0) | (p64 ? 4 : 0);
  p.a = reinterpret_cast<const bf16*>(a.data_ptr()), p.b = reinterpret_cast<const bf16*>(b.data_ptr());
  p.ab_stride = static_cast<int>(a.stride(0));
  p.alog = alog.data_ptr<float>(), p.dtb = dtb.data_ptr<float>();
  p.q = reinterpret_cast<bf16*>(q.data_ptr()), p.k = reinterpret_cast<bf16*>(k.data_ptr());
  p.v = reinterpret_cast<bf16*>(v.data_ptr()), p.z = reinterpret_cast<bf16*>(z.data_ptr());
  p.g = g.data_ptr<float>(), p.beta = beta.data_ptr<float>();
  p.T = static_cast<int>(T), p.B = static_cast<int>(B);
  p.n_qkv_m = static_cast<int>((T + kQkvStride - 1) / kQkvStride);
  p.n_z_m = static_cast<int>((T + kBM - 1) / kBM);
  p.n_units = p.n_qkv_m * kQkvPairs + p.n_z_m * kZPairs;
  const CUtensorMap m_x = make_map(x.data_ptr(), kK, T, kK, 128);
  const CUtensorMap m_w = make_map(w_qkvz.data_ptr(), kK, kD + kZ, kK, kBN);

  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  static int clusters_by_device[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index out of range");
  int& clusters = clusters_by_device[dev];
  if (clusters == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(inproj_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    cfg.gridDim = dim3(num_sms / 2 * 2);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, inproj_kernel, &cfg));
    TORCH_CHECK(clusters > 0, "the fused in_proj cannot be resident as a 2-CTA cluster");
  }
  cfg.gridDim = dim3(2 * std::min(clusters, p.n_units));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, inproj_kernel, p, m_x, m_w));
  return {q, k, v, z, g, beta};
}

}  // namespace qin

// ---------------------------------------------------------------------------------------------------------------------
// v1 lane PDENSE (was qk_inproj8.cu; merged into this unit because the build's native-extension metadata
// (extensions.json, ~1.14 MB of header dependencies per unit) is capped at 16 MiB: 15+ units are refused).
// Lane PDENSE (PRECISION CHANGE): the GDN prefill input projection of qk_inproj.cu in INT8 (Qwen3.6-35B-A3B, H100).
//
// qk_inproj.cu with the GEMM's operands in int8: the chunk's rows x / s quantized per token (amax / 127) and
// W_qkvz * s quantized per output channel (amax / 127), where s is the chunk's per-input-channel smoothing vector
// (pdense.py: amax_x^0.7 / amax_W^0.3; x W^T = (x / s)(W s)^T exactly, s moves the residual stream's outlier
// channels into the weight so both int8 grids are fine). The int8 weight is derived at run time from the bf16
// weight, which is only read. wgmma m64n128k32 s8 x s8 -> s32 over 128-byte k-blocks (the same 128B-swizzled tiles,
// 16 k-blocks, exact integer sums, one k-block group kept in flight), and each output is float(acc) * s_row * s_col
// rounded once to bf16 when the tile is staged -- what torch._int_mm followed by the two scales computes. Everything downstream of the staged bf16 rows is the
// parent kernel's: the conv front, silu, the q / k L2 norm, g / beta and the conv-state write, run from shared
// memory by dedicated warps while the next tile computes; [q | k | v] never goes to HBM.
//
// Portions follow qk_inproj.cu / qk_moe_prefill.cu (DeepGEMM-derived wgmma wrapper, GMMA descriptors, TMA loads,
// mbarrier pipeline: Copyright (c) 2025 DeepSeek, MIT License; sgl-deep-gemm distribution Copyright 2023-2026
// SGLang Team, Apache License 2.0).
// ---------------------------------------------------------------------------------------------------------------------

namespace qin8 {

using bf16 = __nv_bfloat16;
constexpr int kK = 2048;                 // hidden size
constexpr int kD = 8192;                 // [q | k | v] channels
constexpr int kZ = 4096;                 // z channels
constexpr int kHead = 128, kNQ = 16, kNK = 16, kNV = 32;
constexpr int kBM = 256, kBN = 128, kBK = 128, kStages = 3, kHalo = 3, kQkvStride = kBM - kHalo;
constexpr int kKB = kK / kBK;
constexpr int kQkvPairs = kD / kBN / 2, kZPairs = kZ / kBN / 2;  // n-pairs per m-tile: 32, 16
constexpr uint32_t kABytes = kBM * kBK;                          // 32 KB (int8)
constexpr uint32_t kBBytes = kBN * kBK;                          // 16 KB (int8)
constexpr int kDRow = kBN * 2 + 16;                              // staged row stride (bytes), bank-conflict free
constexpr uint32_t kDBytes = kBM * kDRow;
constexpr int kMaxSeqs = 256;
constexpr uint32_t kSmem = kStages * (kABytes + kBBytes) + kDBytes + 2 * kStages * 8 + 4 * (kMaxSeqs + 1) * 3 + 1024 + 1024 + 16;
constexpr uint64_t kEvictNormal = 0x1000000000000000ull;

// ---------------------------------------------------------------- PTX helpers (as in qk_moe_prefill.cu)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
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
__device__ __forceinline__ void tma_load_2d_mc(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask, uint64_t hint) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "h"(mask), "r"(c0), "r"(c1), "l"(hint)
      : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void fence_operand(int& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void wgmma_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
__device__ __forceinline__ uint64_t gmma_desc(uint32_t smem_addr) {
  return static_cast<uint64_t>((smem_addr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(1024 >> 4) << 32) |
         (static_cast<uint64_t>(1) << 62);
}
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

// wgmma.mma_async m64n128k32 s32 += s8 x s8, both operands K-major in shared memory.
__device__ __forceinline__ void wgmma_m64n128k32(int (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
      "}, "
      "%64, %65, p;\n"
      "}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
        "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
        "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
        "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
        "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
      : "l"(da), "l"(db), "r"(1));
}

// ---------------------------------------------------------------- the front's arithmetic (qk_conv_gate.cu)
__device__ __forceinline__ float f_add(float a, float b) { float d; asm("add.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_sub(float a, float b) { float d; asm("sub.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_mul(float a, float b) { float d; asm("mul.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_fma(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float f_div(float a, float b) { float d; asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float f_ex2(float a) { float d; asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_ex2_ftz(float a) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_rcp_ftz(float a) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float f_sqrt_ftz(float a) { float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ uint32_t bf_fma2z(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0u));
  return d;
}
__device__ __forceinline__ float lo2f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi2f(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float v) { uint16_t d; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(v)); return d; }
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
constexpr float kLog2e = 1.44269502162933349609375f;
__device__ __forceinline__ float t_exp(float x) { return f_ex2(f_mul(x, kLog2e)); }
// acc / (1 + ex2(-acc log2 e)) as div.full computes it, for every fp32 acc (exh.cu).
__device__ __forceinline__ float silu(float acc) {
  const float r4 = f_rcp_ftz(f_fma(f_ex2_ftz(f_mul(acc, -kLog2e)), 0.25f, 0.25f));
  return f_mul(f_mul(acc, r4), 0.25f);
}
__device__ __forceinline__ float nv_logf(float v) {
  float r;
  asm("{\n"
      ".reg .pred pd, pinf, pz;\n"
      ".reg .f32 m, ea, e, f, p, t;\n"
      ".reg .b32 bi, ex, mi;\n"
      "setp.lt.f32 pd, %1, 0f00800000;\n"
      "mul.f32 t, %1, 0f4B000000;\n"
      "selp.f32 m, t, %1, pd;\n"
      "selp.f32 ea, 0fC1B80000, 0f00000000, pd;\n"
      "mov.b32 bi, m;\n"
      "add.s32 ex, bi, -1059760811;\n"
      "and.b32 ex, ex, -8388608;\n"
      "sub.s32 mi, bi, ex;\n"
      "cvt.rn.f32.s32 e, ex;\n"
      "fma.rn.ftz.f32 e, e, 0f34000000, ea;\n"
      "mov.b32 f, mi;\n"
      "add.f32 f, f, 0fBF800000;\n"
      "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n"
      "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n"
      "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n"
      "mul.f32 p, f, p;\n"
      "fma.rn.ftz.f32 p, p, f, f;\n"
      "fma.rn.ftz.f32 %0, e, 0f3F317218, p;\n"
      "setp.lt.u32 pinf, bi, 2139095040;\n"
      "@!pinf fma.rn.ftz.f32 %0, m, 0f7F800000, 0f7F800000;\n"
      "setp.eq.f32 pz, m, 0f00000000;\n"
      "@pz mov.f32 %0, 0fFF800000;\n"
      "}\n"
      : "=f"(r)
      : "f"(v));
  return r;
}
__device__ __forceinline__ float nv_expf(float x) {
  float r;
  asm("{\n"
      ".reg .f32 t, j, k, q, e;\n"
      ".reg .b32 s;\n"
      "fma.rn.ftz.f32 t, %1, 0f3BBB989D, 0f3F000000;\n"
      "cvt.ftz.sat.f32.f32 t, t;\n"
      "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n"
      "add.f32 k, j, 0fCB40007F;\n"
      "neg.f32 k, k;\n"
      "fma.rn.ftz.f32 q, %1, 0f3FB8AA3B, k;\n"
      "fma.rn.ftz.f32 q, %1, 0f32A57060, q;\n"
      "mov.b32 s, j;\n"
      "shl.b32 s, s, 23;\n"
      "ex2.approx.ftz.f32 e, q;\n"
      "mov.b32 j, s;\n"
      "mul.f32 %0, e, j;\n"
      "}\n"
      : "=f"(r)
      : "f"(x));
  return r;
}

__device__ __forceinline__ int ld_meta(const void* ptr, int i, bool is64) {
  return is64 ? static_cast<int>(static_cast<const int64_t*>(ptr)[i]) : static_cast<const int*>(ptr)[i];
}

struct Args {
  const bf16* w_conv;   // [kD, 4]
  bf16* cs;             // conv states [slots, kD, 3]
  const bf16* snap;     // [B, kD, 3]: sequence s's old state when prefix(s) > 0 and slot(s) >= 0
  const void* qsl;      // [B + 1]
  const void* idx;      // [B]
  const void* prefix;   // [B]
  int wide;             // bit 0 / 1 / 2: qsl / idx / prefix are int64
  const bf16* a;        // [T, kNV] rows ab_stride apart
  const bf16* b;
  int ab_stride;
  const float* alog;    // [kNV]
  const float* dtb;
  bf16* q;              // [T, kNQ, kHead]
  bf16* k;
  bf16* v;              // [T, kNV, kHead]
  bf16* z;              // [T, kZ]
  float* g;             // [T, kNV]
  float* beta;
  const float* sa;      // [T] row scales of x
  const float* sw;      // [kD + kZ] output-channel scales of W
  int T, B, n_qkv_m, n_z_m, n_units;
};

constexpr int GROUP = 8;  // n-pairs per group; a group sweeps every m-tile before the next group (weights stay in L2)

// Pair unit u: the cluster's m-tile and n-pair; CTA `rank` takes n-block 2 * pair + rank.
__device__ __forceinline__ void decode_unit(const Args& p, int u, uint32_t rank, bool& qkv, int& m, int& nb) {
  const int nq = p.n_qkv_m * kQkvPairs;
  int pair;
  if (u < nq) {
    qkv = true;
    {
      const int per = GROUP * p.n_qkv_m, g = u / per, r = u % per;
      m = r / GROUP, pair = g * GROUP + r % GROUP;
    }
  } else {
    u -= nq;
    qkv = false;
    {
      const int per = GROUP * p.n_z_m, g = u / per, r = u % per;
      m = r / GROUP, pair = g * GROUP + r % GROUP;
    }
  }
  nb = 2 * pair + static_cast<int>(rank);
}

// The conv front of one warp over its staged rows of a [q | k | v] tile, one token per call to step() so it
// can run between the next tile's k-blocks: the window and the taps stay in registers. Staged row r is token
// 253 m - 3 + r; lane l holds channels 4 l .. 4 l + 3 of the head.
struct Front {
  int t, t_hi, row0_tok, s, start, end, ostride, c;
  bool use, qk, active;
  bf16* dst;
  uint32_t cp[3][2];
  uint32_t wp2[4][2];
};

__device__ __forceinline__ uint2 staged_row(const uint8_t* sD, const Front& f, int t, int lane) {
  return *reinterpret_cast<const uint2*>(sD + (t - f.row0_tok) * kDRow + lane * 8);
}

__device__ __forceinline__ void front_window(const Args& p, const uint8_t* sD, Front& f, int t, int lane) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int tt = t - 3 + i;
    if (tt >= f.start) {
      const uint2 u = staged_row(sD, f, tt, lane);
      f.cp[i][0] = u.x, f.cp[i][1] = u.y;
    } else if (f.use) {
      const bf16* sb = p.snap + static_cast<int64_t>(f.s) * (kD * 3) + static_cast<int64_t>(f.c) * 3 + (tt - f.start + 3);
      const uint32_t v0 = __bfloat16_as_ushort(sb[0]), v1 = __bfloat16_as_ushort(sb[3]);
      const uint32_t v2 = __bfloat16_as_ushort(sb[6]), v3 = __bfloat16_as_ushort(sb[9]);
      f.cp[i][0] = v0 | (v1 << 16), f.cp[i][1] = v2 | (v3 << 16);
    } else {
      f.cp[i][0] = 0u, f.cp[i][1] = 0u;
    }
  }
}

__device__ __forceinline__ void front_init(const Args& p, const uint8_t* sD, const int* s_qs, const int* s_slot,
                                           const int* s_hinit, const uint2* sW, Front& f, int m, int head, int r0,
                                           int r1, int lane) {
  f.row0_tok = kQkvStride * m - kHalo;
  f.t = max(f.row0_tok + r0, s_qs[0]);
  f.t_hi = min(f.row0_tok + r1, s_qs[p.B]);
  f.active = f.t < f.t_hi;
  if (!f.active) return;
  f.c = head * kHead + 4 * lane;
  {
    const uint2* wp = sW + 4 * lane;
    uint2 u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) u[j] = wp[j];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int sh = (i & 1) * 16;
      const uint32_t w0 = ((i < 2 ? u[0].x : u[0].y) >> sh) & 0xffff, w1 = ((i < 2 ? u[1].x : u[1].y) >> sh) & 0xffff;
      const uint32_t w2 = ((i < 2 ? u[2].x : u[2].y) >> sh) & 0xffff, w3 = ((i < 2 ? u[3].x : u[3].y) >> sh) & 0xffff;
      f.wp2[i][0] = w0 | (w1 << 16);
      f.wp2[i][1] = w2 | (w3 << 16);
    }
  }
  const bool is_q = head < kNQ, is_k = !is_q && head < kNQ + kNK;
  f.qk = is_q || is_k;
  f.dst = is_q ? p.q + head * kHead : is_k ? p.k + (head - kNQ) * kHead : p.v + (head - kNQ - kNK) * kHead;
  f.ostride = (is_q ? kNQ : is_k ? kNK : kNV) * kHead;
  int s = 0;
  while (s < p.B && s_qs[s + 1] <= f.t) ++s;
  f.s = s, f.start = s_qs[s], f.end = s_qs[s + 1];
  f.use = s_hinit[s] != 0 && s_slot[s] >= 0;
  front_window(p, sD, f, f.t, lane);
}

__device__ __forceinline__ void front_step(const Args& p, const uint8_t* sD, const int* s_qs, const int* s_slot,
                                           const int* s_hinit, Front& f, int lane) {
  if (!f.active || f.t >= f.t_hi) return;
  const int t = f.t;
  if (t >= f.end) {
    // The next non-empty sequence starts at t.
    int s = f.s;
    do {
      ++s;
    } while (s_qs[s + 1] <= t);
    f.s = s, f.start = s_qs[s], f.end = s_qs[s + 1];
    f.use = s_hinit[s] != 0 && s_slot[s] >= 0;
    front_window(p, sD, f, t, lane);
  }
  const uint2 xu = staged_row(sD, f, t, lane);
  const uint32_t x2[2] = {xu.x, xu.y};
  float y[4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t q0 = bf_fma2z(f.cp[0][h], f.wp2[0][h]), q1 = bf_fma2z(f.cp[1][h], f.wp2[1][h]);
    const uint32_t q2 = bf_fma2z(f.cp[2][h], f.wp2[2][h]), q3 = bf_fma2z(x2[h], f.wp2[3][h]);
    float lo = f_add(lo2f(q0), lo2f(q1)), hi = f_add(hi2f(q0), hi2f(q1));
    lo = f_add(lo, lo2f(q2)), hi = f_add(hi, hi2f(q2));
    lo = f_add(lo, lo2f(q3)), hi = f_add(hi, hi2f(q3));
    y[2 * h] = silu(lo);
    y[2 * h + 1] = silu(hi);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) f.cp[0][h] = f.cp[1][h], f.cp[1][h] = f.cp[2][h], f.cp[2][h] = x2[h];
  uint2 o = make_uint2(f2bf2(y[0], y[1]), f2bf2(y[2], y[3]));
  if (f.qk) {
    const float z0 = lo2f(o.x), z1 = hi2f(o.x), z2 = lo2f(o.y), z3 = hi2f(o.y);
    float sq = f_mul(z1, z1);
    sq = f_fma(z0, z0, sq);
    sq = f_fma(z2, z2, sq);
    sq = f_fma(z3, z3, sq);
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) sq = f_add(sq, __shfl_xor_sync(0xffffffffu, sq, off));
    const float rr = f_rcp_ftz(f_sqrt_ftz(f_add(sq, 1e-6f)));
    o = make_uint2(f2bf2(f_mul(z0, rr), f_mul(z1, rr)), f2bf2(f_mul(z2, rr), f_mul(z3, rr)));
  }
  *reinterpret_cast<uint2*>(f.dst + static_cast<int64_t>(t) * f.ostride + 4 * lane) = o;
  if (t == f.end - 1 && s_slot[f.s] >= 0) {
    // New conv state: the raw last three input rows, or the old state shifted by the length when shorter.
    const int L = f.end - f.start;
    bf16* sbw = p.cs + static_cast<int64_t>(s_slot[f.s]) * (kD * 3) + static_cast<int64_t>(f.c) * 3;
    uint16_t nv[3][4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int tt = f.end - 3 + i;
      if (tt >= f.start) {
        const uint2 u = staged_row(sD, f, tt, lane);
        nv[i][0] = u.x & 0xffff, nv[i][1] = u.x >> 16, nv[i][2] = u.y & 0xffff, nv[i][3] = u.y >> 16;
      } else {
        const int oc = i + L;
        const bf16* sb = p.snap + static_cast<int64_t>(f.s) * (kD * 3) + static_cast<int64_t>(f.c) * 3;
#pragma unroll
        for (int j = 0; j < 4; ++j) nv[i][j] = s_hinit[f.s] != 0 && oc < 3 ? __bfloat16_as_ushort(sb[j * 3 + oc]) : 0;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 3; ++i) sbw[j * 3 + i] = __ushort_as_bfloat16(nv[i][j]);
  }
  f.t = t + 1;
}


// Four tokens of one sequence at once (none its sequence's last, all from staged rows or the carried window),
// the four chains interleaved; the same arithmetic as four front_step calls.
__device__ __forceinline__ bool front_step4(const uint8_t* sD, Front& f, int lane) {
  const int t = f.t;
  if (!(f.active && t + 4 <= f.t_hi && t >= f.start && t + 3 < f.end - 1)) return false;
  uint32_t xr[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint2 u = staged_row(sD, f, t + j, lane);
    xr[j][0] = u.x, xr[j][1] = u.y;
  }
  uint2 o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float y[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t w0 = j == 0 ? f.cp[0][h] : j == 1 ? f.cp[1][h] : j == 2 ? f.cp[2][h] : xr[0][h];
      const uint32_t w1 = j == 0 ? f.cp[1][h] : j == 1 ? f.cp[2][h] : j == 2 ? xr[0][h] : xr[1][h];
      const uint32_t w2 = j == 0 ? f.cp[2][h] : j == 1 ? xr[0][h] : j == 2 ? xr[1][h] : xr[2][h];
      const uint32_t q0 = bf_fma2z(w0, f.wp2[0][h]), q1 = bf_fma2z(w1, f.wp2[1][h]);
      const uint32_t q2 = bf_fma2z(w2, f.wp2[2][h]), q3 = bf_fma2z(xr[j][h], f.wp2[3][h]);
      float lo = f_add(lo2f(q0), lo2f(q1)), hi = f_add(hi2f(q0), hi2f(q1));
      lo = f_add(lo, lo2f(q2)), hi = f_add(hi, hi2f(q2));
      lo = f_add(lo, lo2f(q3)), hi = f_add(hi, hi2f(q3));
      y[2 * h] = silu(lo);
      y[2 * h + 1] = silu(hi);
    }
    o[j] = make_uint2(f2bf2(y[0], y[1]), f2bf2(y[2], y[3]));
  }
  if (f.qk) {
    float sq[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float z0 = lo2f(o[j].x), z1 = hi2f(o[j].x), z2 = lo2f(o[j].y), z3 = hi2f(o[j].y);
      sq[j] = f_mul(z1, z1);
      sq[j] = f_fma(z0, z0, sq[j]);
      sq[j] = f_fma(z2, z2, sq[j]);
      sq[j] = f_fma(z3, z3, sq[j]);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
#pragma unroll
      for (int j = 0; j < 4; ++j) sq[j] = f_add(sq[j], __shfl_xor_sync(0xffffffffu, sq[j], off));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float rr = f_rcp_ftz(f_sqrt_ftz(f_add(sq[j], 1e-6f)));
      const float z0 = lo2f(o[j].x), z1 = hi2f(o[j].x), z2 = lo2f(o[j].y), z3 = hi2f(o[j].y);
      o[j] = make_uint2(f2bf2(f_mul(z0, rr), f_mul(z1, rr)), f2bf2(f_mul(z2, rr), f_mul(z3, rr)));
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) *reinterpret_cast<uint2*>(f.dst + static_cast<int64_t>(t + j) * f.ostride + 4 * lane) = o[j];
#pragma unroll
  for (int h = 0; h < 2; ++h) f.cp[0][h] = xr[1][h], f.cp[1][h] = xr[2][h], f.cp[2][h] = xr[3][h];
  f.t = t + 4;
  return true;
}

// The conv / silu arithmetic of four tokens of one range (front_step4's, from registers).
__device__ __forceinline__ void front_math4(const uint32_t (&cp)[3][2], const uint32_t (&wp2)[4][2], const uint32_t (&xr)[4][2],
                                            bool qk, uint2 (&o)[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float y[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t w0 = j == 0 ? cp[0][h] : j == 1 ? cp[1][h] : j == 2 ? cp[2][h] : xr[0][h];
      const uint32_t w1 = j == 0 ? cp[1][h] : j == 1 ? cp[2][h] : j == 2 ? xr[0][h] : xr[1][h];
      const uint32_t w2 = j == 0 ? cp[2][h] : j == 1 ? xr[0][h] : j == 2 ? xr[1][h] : xr[2][h];
      const uint32_t q0 = bf_fma2z(w0, wp2[0][h]), q1 = bf_fma2z(w1, wp2[1][h]);
      const uint32_t q2 = bf_fma2z(w2, wp2[2][h]), q3 = bf_fma2z(xr[j][h], wp2[3][h]);
      float lo = f_add(lo2f(q0), lo2f(q1)), hi = f_add(hi2f(q0), hi2f(q1));
      lo = f_add(lo, lo2f(q2)), hi = f_add(hi, hi2f(q2));
      lo = f_add(lo, lo2f(q3)), hi = f_add(hi, hi2f(q3));
      y[2 * h] = silu(lo);
      y[2 * h + 1] = silu(hi);
    }
    o[j] = make_uint2(f2bf2(y[0], y[1]), f2bf2(y[2], y[3]));
  }
}

// R ranges stepped four tokens each at once (front_step4x2 generalized): loads of every range first, then the
// 4R conv chains, then the 4R L2-norm reductions interleaved, then the stores. False unless every range can step.
template <int R>
__device__ __forceinline__ bool front_step4xR(const uint8_t* sD, Front (&f)[R], int lane) {
#pragma unroll
  for (int i = 0; i < R; ++i)
    if (!(f[i].active && f[i].t + 4 <= f[i].t_hi && f[i].t >= f[i].start && f[i].t + 3 < f[i].end - 1)) return false;
  uint32_t x[R][4][2];
#pragma unroll
  for (int i = 0; i < R; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint2 u = staged_row(sD, f[i], f[i].t + j, lane);
      x[i][j][0] = u.x, x[i][j][1] = u.y;
    }
  uint2 o[R][4];
#pragma unroll
  for (int i = 0; i < R; ++i) front_math4(f[i].cp, f[i].wp2, x[i], f[i].qk, o[i]);
  if (f[0].qk) {
    float sq[R][4];
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float z0 = lo2f(o[i][j].x), z1 = hi2f(o[i][j].x), z2 = lo2f(o[i][j].y), z3 = hi2f(o[i][j].y);
        float q = f_mul(z1, z1);
        q = f_fma(z0, z0, q);
        q = f_fma(z2, z2, q);
        sq[i][j] = f_fma(z3, z3, q);
      }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
#pragma unroll
      for (int i = 0; i < R; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) sq[i][j] = f_add(sq[i][j], __shfl_xor_sync(0xffffffffu, sq[i][j], off));
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float rr = f_rcp_ftz(f_sqrt_ftz(f_add(sq[i][j], 1e-6f)));
        const float z0 = lo2f(o[i][j].x), z1 = hi2f(o[i][j].x), z2 = lo2f(o[i][j].y), z3 = hi2f(o[i][j].y);
        o[i][j] = make_uint2(f2bf2(f_mul(z0, rr), f_mul(z1, rr)), f2bf2(f_mul(z2, rr), f_mul(z3, rr)));
      }
  }
#pragma unroll
  for (int i = 0; i < R; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      *reinterpret_cast<uint2*>(f[i].dst + static_cast<int64_t>(f[i].t + j) * f[i].ostride + 4 * lane) = o[i][j];
#pragma unroll
    for (int h = 0; h < 2; ++h) f[i].cp[0][h] = x[i][1][h], f[i].cp[1][h] = x[i][2][h], f[i].cp[2][h] = x[i][3][h];
    f[i].t += 4;
  }
  return true;
}

#ifndef PD_FRONT_RANGES
#define PD_FRONT_RANGES 2
#endif

// g and beta of value head `head` for the tokens of staged rows [r0, r1) (one token per lane), from the a / b
// values staged with the tile (sAB[r]: a in the low half, b in the high half).
__device__ __forceinline__ void gbeta_rows(const Args& p, const int* s_qs, const uint32_t* sAB, int m, int head, int r0,
                                           int r1, int lane) {
  const int row0_tok = kQkvStride * m - kHalo;
  const int t_lo = max(row0_tok + r0, s_qs[0]), t_hi = min(row0_tok + r1, s_qs[p.B]);
  const int hv = head - kNQ - kNK;
  for (int t = t_lo + lane; t < t_hi; t += 32) {
  const uint32_t ab = sAB[t - row0_tok];
  const float av = __uint_as_float(ab << 16), bv = __uint_as_float(ab & 0xffff0000u);
  const float xg = f_add(av, p.dtb[hv]);
  const float lg = nv_logf(f_add(t_exp(xg), 1.f));
  const float sp = xg <= 20.f ? lg : xg;
  const float gv = f_mul(f_sub(0.f, t_exp(p.alog[hv])), sp);
  p.g[static_cast<int64_t>(t) * kNV + hv] = nv_expf(gv);
  const float sg = f_div(1.f, f_add(t_exp(f_sub(0.f, bv)), 1.f));
  p.beta[static_cast<int64_t>(t) * kNV + hv] = bf2f(f2bf(sg));
  }
}

// 384 threads: warpgroups 0-1 run WGMMA (warpgroup w: tile rows [64 w, 64 w + 64) and [128 + 64 w, +64)) and
// the epilogue, warpgroup 2 issues the TMA loads (one thread).
__global__ void __launch_bounds__(384, 1)
    inproj8_kernel(const Args p, const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_w) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + kStages * kABytes;
  uint8_t* sD = sB + kStages * kBBytes;
  uint64_t* full = reinterpret_cast<uint64_t*>(sD + kDBytes);
  uint64_t* empty = full + kStages;
  int* s_qs = reinterpret_cast<int*>(empty + kStages);
  int* s_slot = s_qs + (kMaxSeqs + 1);
  int* s_hinit = s_slot + (kMaxSeqs + 1);
  // pending head's conv taps [128][4] bf16 (16-byte aligned)
  uint2* sW = reinterpret_cast<uint2*>((reinterpret_cast<uintptr_t>(s_hinit + (kMaxSeqs + 1)) + 15) & ~uintptr_t{15});
  uint32_t* sAB = reinterpret_cast<uint32_t*>(sW + 128);             // pending tile's (a | b << 16) per staged row
  const uint32_t rank = cluster_rank();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 16);
    }
    fence_barrier_init();
  }
  if (threadIdx.x == 32) {
    prefetch_tmap(&tm_x);
    prefetch_tmap(&tm_w);
  }
  for (int i = threadIdx.x; i <= p.B; i += blockDim.x) {
    s_qs[i] = ld_meta(p.qsl, i, p.wide & 1);
    if (i < p.B) {
      s_slot[i] = ld_meta(p.idx, i, p.wide & 2);
      s_hinit[i] = ld_meta(p.prefix, i, p.wide & 4) > 0;
    }
  }
  __syncthreads();
  cluster_sync();
  const int unit0 = static_cast<int>(cluster_id_x()), unit_step = static_cast<int>(n_clusters_x());

  // Named barriers between the math warpgroups (256 threads) and the epilogue warps (96 threads):
  // kBarStaged: a tile is staged; kBarFree: the staging buffer (and sW / sAB) may be overwritten.
  constexpr int kBarStaged = 2, kBarFree = 3, kSync = 256 + 96;
  if (threadIdx.x >= 256) {

    const int ew = threadIdx.x / 32 - 9, lane = threadIdx.x % 32;  // epilogue warp 0..2 (warps 9-11)
    if (threadIdx.x == 256) {
      int stage = 0;
      uint32_t phase = 0;
      for (int u = unit0; u < p.n_units; u += unit_step) {
        bool qkv;
        int m, nb;
        decode_unit(p, u, rank, qkv, m, nb);
        const int a_row = (qkv ? kQkvStride * m - kHalo : kBM * m) + 128 * static_cast<int>(rank);
        const int b_row = (qkv ? 0 : kD) + nb * kBN;
        for (int kb = 0; kb < kKB; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          mbar_arrive_expect_tx(&full[stage], kABytes + kBBytes);
          tma_load_2d_mc(smem_u32(sA + stage * kABytes) + rank * (kABytes / 2), &tm_x, &full[stage], kb * kBK, a_row, 3,
                         kEvictNormal);
          tma_load_2d(smem_u32(sB + stage * kBBytes), &tm_w, &full[stage], kb * kBK, b_row, kEvictNormal);
          stage = stage + 1 == kStages ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    } else if (ew >= 0) {
      // Epilogue warps: the staging buffer starts free; then per tile wait until staged, run the epilogue, free it.
      named_bar_arrive(kBarFree, kSync);
      for (int u = unit0; u < p.n_units; u += unit_step) {
        bool qkv;
        int m, nb;
        decode_unit(p, u, rank, qkv, m, nb);
        named_bar_sync(kBarStaged, kSync);
        if (qkv) {
          // Warp ew takes staged rows [3 + 85 ew, +85) (the last warp 83), as PD_FRONT_RANGES ranges stepped together.
          constexpr int kR = PD_FRONT_RANGES;
          const int r0 = kHalo + 85 * ew, r1 = min(r0 + 85, kBM), span = (r1 - r0 + kR - 1) / kR;
          Front fr[kR];
#pragma unroll
          for (int i = 0; i < kR; ++i)
            front_init(p, sD, s_qs, s_slot, s_hinit, sW, fr[i], m, nb, min(r0 + i * span, r1), min(r0 + (i + 1) * span, r1), lane);
          while (front_step4xR<kR>(sD, fr, lane)) {
          }
#pragma unroll
          for (int i = 0; i < kR; ++i) {
            while (fr[i].active && fr[i].t < fr[i].t_hi) {
              if (!front_step4(sD, fr[i], lane)) front_step(p, sD, s_qs, s_slot, s_hinit, fr[i], lane);
            }
          }
          if (nb >= kNQ + kNK) gbeta_rows(p, s_qs, sAB, m, nb, r0, r1, lane);
        } else {
          // z rows: warp ew copies staged rows ew, ew + 3, ... (8 bytes per lane per row).
          for (int r = ew; r < kBM; r += 3) {
            const int t = kBM * m + r;
            if (t < p.T) {
              const uint2 val = *reinterpret_cast<const uint2*>(sD + r * kDRow + lane * 8);
              *reinterpret_cast<uint2*>(p.z + static_cast<int64_t>(t) * kZ + nb * kBN + lane * 4) = val;
            }
          }
        }
        named_bar_arrive(kBarFree, kSync);
      }
    }
  } else {

    const int wg = threadIdx.x / 128, warp = threadIdx.x / 32, lane = threadIdx.x % 32, wi = warp % 4;
    const uint32_t a_base = smem_u32(sA) + wg * 64 * 128;
    const uint32_t b_base = smem_u32(sB);
    int stage = 0;
    uint32_t phase = 0;
    for (int u = unit0; u < p.n_units; u += unit_step) {
      bool qkv;
      int m, nb;
      decode_unit(p, u, rank, qkv, m, nb);
      int acc0[64], acc1[64];
#pragma unroll
      for (int i = 0; i < 64; ++i) acc0[i] = 0, acc1[i] = 0;
      // This tile's epilogue inputs, loaded now and staged with the tile: conv taps (4 bytes per thread) and,
      // for value heads, the a / b pair of staged row threadIdx.x.
      uint32_t w_pf = 0, ab_pf = 0;
      if (qkv) {
        w_pf = reinterpret_cast<const uint32_t*>(p.w_conv + static_cast<int64_t>(nb) * kHead * 4)[threadIdx.x];
        const int t = kQkvStride * m - kHalo + static_cast<int>(threadIdx.x);
        if (nb >= kNQ + kNK && t >= 0 && t < p.T) {
          const int hv = nb - kNQ - kNK;
          ab_pf = static_cast<uint32_t>(__bfloat16_as_ushort(p.a[static_cast<int64_t>(t) * p.ab_stride + hv])) |
                  (static_cast<uint32_t>(__bfloat16_as_ushort(p.b[static_cast<int64_t>(t) * p.ab_stride + hv])) << 16);
        }
      }
      for (int kb = 0; kb < kKB; ++kb) {
        mbar_wait(&full[stage], phase);
#pragma unroll
        for (int i = 0; i < 64; ++i) fence_operand(acc0[i]), fence_operand(acc1[i]);
        wgmma_fence();
        const uint32_t a_st = a_base + stage * kABytes, b_st = b_base + stage * kBBytes;
#pragma unroll
        for (int k = 0; k < kBK / 32; ++k) wgmma_m64n128k32(acc0, gmma_desc(a_st + k * 32), gmma_desc(b_st + k * 32));
#pragma unroll
        for (int k = 0; k < kBK / 32; ++k)
          wgmma_m64n128k32(acc1, gmma_desc(a_st + 128 * 128 + k * 32), gmma_desc(b_st + k * 32));
        wgmma_commit();
#pragma unroll
        for (int i = 0; i < 64; ++i) fence_operand(acc0[i]), fence_operand(acc1[i]);
        // One k-block group stays in flight: the previous k-block's stage is released once its group has retired.
        wgmma_wait1();
        if (kb > 0 && lane < 2) mbar_arrive_cluster(&empty[stage == 0 ? kStages - 1 : stage - 1], lane);
        stage = stage + 1 == kStages ? 0 : stage + 1;
        phase ^= stage == 0;
      }
      wgmma_wait0();
#pragma unroll
      for (int i = 0; i < 64; ++i) fence_operand(acc0[i]), fence_operand(acc1[i]);
      if (lane < 2) mbar_arrive_cluster(&empty[stage == 0 ? kStages - 1 : stage - 1], lane);
      // Stage the tile as bf16 rows (acc * s_row * s_col, one rounding) once the epilogue warps have released it.
      const int r_a = 64 * wg + 16 * wi + lane / 4, col = 2 * (lane % 4);
      float sr[4];
      {
        const int row0 = qkv ? kQkvStride * m - kHalo : kBM * m;
        const int rr[4] = {r_a, r_a + 8, 128 + r_a, 136 + r_a};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int t = row0 + rr[j];
          sr[j] = t >= 0 && t < p.T ? p.sa[t] : 0.f;
        }
      }
      const float* swp = p.sw + (qkv ? 0 : kD) + nb * kBN + col;
      named_bar_sync(kBarFree, kSync);
      {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float2 c = __ldg(reinterpret_cast<const float2*>(swp + 8 * i));
          uint8_t* d0 = sD + r_a * kDRow + (8 * i + col) * 2;
          *reinterpret_cast<uint32_t*>(d0) =
              f2bf2(static_cast<float>(acc0[4 * i]) * sr[0] * c.x, static_cast<float>(acc0[4 * i + 1]) * sr[0] * c.y);
          *reinterpret_cast<uint32_t*>(d0 + 8 * kDRow) =
              f2bf2(static_cast<float>(acc0[4 * i + 2]) * sr[1] * c.x, static_cast<float>(acc0[4 * i + 3]) * sr[1] * c.y);
          *reinterpret_cast<uint32_t*>(d0 + 128 * kDRow) =
              f2bf2(static_cast<float>(acc1[4 * i]) * sr[2] * c.x, static_cast<float>(acc1[4 * i + 1]) * sr[2] * c.y);
          *reinterpret_cast<uint32_t*>(d0 + 136 * kDRow) =
              f2bf2(static_cast<float>(acc1[4 * i + 2]) * sr[3] * c.x, static_cast<float>(acc1[4 * i + 3]) * sr[3] * c.y);
        }
        reinterpret_cast<uint32_t*>(sW)[threadIdx.x] = w_pf;
        sAB[threadIdx.x] = ab_pf;
      }
      named_bar_arrive(kBarStaged, kSync);
    }
    // The epilogue warps' last arrival on kBarFree has no consumer: take it so the barrier ends balanced.
    named_bar_sync(kBarFree, kSync);
  }
  // No CTA leaves while its peer may still multicast into it or arrive on its barriers.
  cluster_sync();
}

// ---------------------------------------------------------------- host
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeFn encode_fn() {
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

// Row-major int8 [outer, inner] in boxes of [box_outer rows, 128 columns], 128B swizzle.
static CUtensorMap make_map(const void* base, uint64_t inner, uint64_t outer, uint64_t row_stride, uint32_t box_outer) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {row_stride};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kBK), box_outer};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides,
                                 box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

static bool meta64(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong), name,
              " must be a contiguous CUDA int32 / int64 tensor");
  return t.scalar_type() == at::kLong;
}

// xq [T, 2048] int8 with row scales sa [T] fp32, wq [12288, 2048] int8 (rows [q | k | v | z]) with channel scales
// sw [12288, 1] fp32: returns (q, k, v, z, g, beta) as qk_inproj.inproj_front does for x @ W^T, z written into the
// caller's `z`, the conv states of the batch's sequences updated in place; snap from qk_inproj.front_prep.
std::vector<torch::Tensor> inproj_front8(torch::Tensor x, torch::Tensor sa, torch::Tensor w_qkvz, torch::Tensor sw,
                                         torch::Tensor w_conv, torch::Tensor cs,
                                        torch::Tensor snap, torch::Tensor qsl, torch::Tensor idx, torch::Tensor prefix,
                                        torch::Tensor a, torch::Tensor b, torch::Tensor alog, torch::Tensor dtb,
                                        torch::Tensor z) {
  const int64_t T = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kChar && x.is_contiguous() && x.dim() == 2 && x.size(1) == kK,
              "x: contiguous int8 [T, 2048]");
  TORCH_CHECK(T >= 1 && T <= (1 << 20), "token count out of range");
  TORCH_CHECK(w_qkvz.scalar_type() == at::kChar && w_qkvz.is_contiguous() && w_qkvz.dim() == 2 &&
                  w_qkvz.size(0) == kD + kZ && w_qkvz.size(1) == kK && w_qkvz.device() == x.device(), "w_qkvz: int8 [12288, 2048]");
  TORCH_CHECK(sa.scalar_type() == at::kFloat && sa.is_contiguous() && sa.numel() == T && sa.device() == x.device(),
              "sa: fp32 [T] row scales");
  TORCH_CHECK(sw.scalar_type() == at::kFloat && sw.is_contiguous() && sw.numel() == kD + kZ && sw.device() == x.device() &&
                  reinterpret_cast<uintptr_t>(sw.data_ptr()) % 8 == 0, "sw: fp32 [12288] channel scales, 8-byte aligned");
  TORCH_CHECK(w_conv.scalar_type() == at::kBFloat16 && w_conv.is_contiguous() && w_conv.numel() == kD * 4, "conv weights");
  TORCH_CHECK(cs.scalar_type() == at::kBFloat16 && cs.is_contiguous() && cs.size(-1) == 3 && cs.size(-2) == kD, "conv state");
  const bool q64 = meta64(qsl, "qsl"), i64 = meta64(idx, "idx"), p64 = meta64(prefix, "prefix");
  const int64_t B = qsl.numel() - 1;
  TORCH_CHECK(B >= 1 && B <= kMaxSeqs && idx.numel() == B && prefix.numel() == B, "batch of 1..256 sequences");
  TORCH_CHECK(snap.scalar_type() == at::kBFloat16 && snap.is_contiguous() && snap.numel() == B * kD * 3, "snap");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && a.dim() == 2 && b.dim() == 2 &&
                  a.size(1) == kNV && b.size(1) == kNV && a.stride(1) == 1 && b.stride(1) == 1 && a.stride(0) == b.stride(0) &&
                  a.size(0) == T && b.size(0) == T, "a, b: [T, 32] rows with one stride");
  TORCH_CHECK(alog.scalar_type() == at::kFloat && dtb.scalar_type() == at::kFloat && alog.numel() == kNV && dtb.numel() == kNV,
              "A_log / dt_bias fp32 [32]");
  const at::cuda::CUDAGuard guard(x.device());
  const auto bopt = x.options().dtype(at::kBFloat16);
  auto q = torch::empty({1, T, kNQ, kHead}, bopt);
  auto k = torch::empty({1, T, kNK, kHead}, bopt);
  auto v = torch::empty({1, T, kNV, kHead}, bopt);
  TORCH_CHECK(z.scalar_type() == at::kBFloat16 && z.is_contiguous() && z.numel() == T * kZ && z.device() == x.device(),
              "z: contiguous bf16 [T, 4096]");
  auto g = torch::empty({1, T, kNV}, x.options().dtype(at::kFloat));
  auto beta = torch::empty({1, T, kNV}, x.options().dtype(at::kFloat));
  TORCH_CHECK(w_conv.device() == x.device() && cs.device() == x.device() && a.device() == x.device(), "one device");
  Args p;
  p.w_conv = reinterpret_cast<const bf16*>(w_conv.data_ptr());
  p.cs = reinterpret_cast<bf16*>(cs.data_ptr());
  p.snap = reinterpret_cast<const bf16*>(snap.data_ptr());
  p.qsl = qsl.data_ptr(), p.idx = idx.data_ptr(), p.prefix = prefix.data_ptr();
  p.wide = (q64 ? 1 : 0) | (i64 ? 2 : 0) | (p64 ? 4 : 0);
  p.a = reinterpret_cast<const bf16*>(a.data_ptr()), p.b = reinterpret_cast<const bf16*>(b.data_ptr());
  p.ab_stride = static_cast<int>(a.stride(0));
  p.alog = alog.data_ptr<float>(), p.dtb = dtb.data_ptr<float>();
  p.q = reinterpret_cast<bf16*>(q.data_ptr()), p.k = reinterpret_cast<bf16*>(k.data_ptr());
  p.v = reinterpret_cast<bf16*>(v.data_ptr()), p.z = reinterpret_cast<bf16*>(z.data_ptr());
  p.g = g.data_ptr<float>(), p.beta = beta.data_ptr<float>();
  p.sa = sa.data_ptr<float>(), p.sw = sw.data_ptr<float>();
  p.T = static_cast<int>(T), p.B = static_cast<int>(B);
  p.n_qkv_m = static_cast<int>((T + kQkvStride - 1) / kQkvStride);
  p.n_z_m = static_cast<int>((T + kBM - 1) / kBM);
  p.n_units = p.n_qkv_m * kQkvPairs + p.n_z_m * kZPairs;
  const CUtensorMap m_x = make_map(x.data_ptr(), kK, T, kK, 128);
  const CUtensorMap m_w = make_map(w_qkvz.data_ptr(), kK, kD + kZ, kK, kBN);

  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  static int clusters_by_device[64] = {};
  const int dev = at::cuda::current_device();
  TORCH_CHECK(dev < 64, "device index out of range");
  int& clusters = clusters_by_device[dev];
  if (clusters == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(inproj8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    cfg.gridDim = dim3(num_sms / 2 * 2);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, inproj8_kernel, &cfg));
    TORCH_CHECK(clusters > 0, "the fused in_proj cannot be resident as a 2-CTA cluster");
  }
  cfg.gridDim = dim3(2 * std::min(clusters, p.n_units));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, inproj8_kernel, p, m_x, m_w));
  return {q, k, v, z, g, beta};
}

}  // namespace qin8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("front_prep", &qin::front_prep, "GDN extend inputs: int64 slots / cu_seqlens, initial states, old conv states");
  m.def("inproj_front", &qin::inproj_front, "GDN prefill in_proj [q|k|v|z] GEMM with the conv / gate front fused");
  m.def("inproj_front8", &qin8::inproj_front8,
        "GDN prefill in_proj [q|k|v|z] in INT8 (row x channel scales, s32 sums) with the conv / gate front fused");
}
