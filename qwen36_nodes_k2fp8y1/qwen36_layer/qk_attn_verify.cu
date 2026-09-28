// Attention verify (decode / MTP target-verify) and prefill front for Qwen3.6-35B-A3B's full-attention
// layers: the crowned fd0012a2 kernel (split-by-work verify scheduling, staged combine) with this line's
// prefill-front changes (four pipeline stages; attn_prefill_front_split reads the q / gate block
// [T, 8192] and the k / v block [T, 1024] separately, the same loads and bits as the fused front).
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#ifndef QK_STAGES
#define QK_STAGES 4
#endif
#ifndef QK_SPLIT_RS
#define QK_SPLIT_RS 1
#endif
#ifndef QK_COMBINE_STAGE
#define QK_COMBINE_STAGE 1
#endif
namespace {
constexpr int HQ = 16, HKV = 2, GQ = 8, D = 256, NT = 4, ROWS = GQ * NT, HALF_ROT = 32;
constexpr int QKV_W = 2 * HQ * D + 2 * HKV * D, K_OFF = 2 * HQ * D, V_OFF = K_OFF + HKV * D;
constexpr int OUT_W = HQ * D;
constexpr int TILE = 64, STAGES = QK_STAGES, CWARPS = 8, PWARPS = 4;
constexpr int CTHREADS = CWARPS * 32, PTHREADS = PWARPS * 32, THREADS = CTHREADS + PTHREADS;
constexpr int KROW = D + 16;
constexpr int PROW = 2 * TILE + 16;
constexpr int NEWROWS = 16;
constexpr int MAX_GROUPS = 128;
constexpr unsigned FULL = 0xffffffffu;
constexpr uint32_t SMEM_BYTES = 2 * STAGES * TILE * KROW + 2 * NEWROWS * KROW + ROWS * KROW + ROWS * PROW +
                                ROWS * 2 * D + 2 * ROWS * 4 * 4 + 2 * STAGES * 8;
#if QK_SPLIT_RS
constexpr int RS_MAX_SPLITS = 64;
static_assert(RS_MAX_SPLITS * 3 * ROWS * 4 <= STAGES * TILE * KROW, "the combine scratch holds the most splits");
#endif
#if QK_COMBINE_STAGE
constexpr int CS_MAX_SPLITS = (2 * STAGES * TILE * KROW / (4 * ROWS) - 256) / 11;
static_assert(CS_MAX_SPLITS >= 64, "the staged combine holds attn_fused.py's 64 splits");
#endif
typedef __nv_bfloat16 bf16;
struct Args {
  const bf16* qkv;
  const bf16* qw;
  const bf16* kw;
  float eps, scale_log2;
  const void* cos_sin;
  int64_t cs_stride;
  int cs_f32;
  const int64_t* pos;
  int64_t pos_axis_stride;
  const int64_t* axis_map;
  uint8_t* kcache;
  uint8_t* vcache;
  const int64_t* out_loc;
  const int32_t* page_table;
  int64_t pt_stride;
  const int32_t* seqlens;
  int splits;
#if QK_SPLIT_RS
  int batch;
#endif
  float* ws_o;
  float* ws_ml;
  int* arrive;
  int* depart;
  bf16* out;
};
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
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
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t r;
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %1;\ncvt.rn.f16x2.e4m3x2 %0, l;\n}" : "=r"(r) : "r"(w));
  return r;
}
__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t r;
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %1;\ncvt.rn.f16x2.e4m3x2 %0, h;\n}" : "=r"(r) : "r"(w));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
struct Rope {
  float c[8], s[8];
};
template <class A>
__device__ __forceinline__ Rope load_rope(const A& p, int64_t tok, int lane) {
  Rope rp;
  if (lane < 8) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int hr = 8 * (lane & 3) + e;
      const int64_t pos = p.axis_map ? p.pos[p.axis_map[hr] * p.pos_axis_stride + tok] : p.pos[tok];
      if (p.cs_f32) {
        const float* row = reinterpret_cast<const float*>(p.cos_sin) + pos * p.cs_stride;
        rp.c[e] = row[hr];
        rp.s[e] = row[HALF_ROT + hr];
      } else {
        const bf16* row = reinterpret_cast<const bf16*>(p.cos_sin) + pos * p.cs_stride;
        rp.c[e] = bf(row[hr]);
        rp.s[e] = bf(row[HALF_ROT + hr]);
      }
    }
  }
  return rp;
}
__device__ __forceinline__ uint2 norm_rope_e4m3(const uint4 raw, const uint4 wr, const Rope& rp, float eps, int lane) {
  const uint32_t rw[4] = {raw.x, raw.y, raw.z, raw.w};
  const uint32_t ww[4] = {wr.x, wr.y, wr.z, wr.w};
  float x[8], ss = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    x[2 * e] = bf_lo(rw[e]);
    x[2 * e + 1] = bf_hi(rw[e]);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float inv = rsqrtf(ss / D + eps);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    x[2 * e] = rbf(x[2 * e] * inv * (bf_lo(ww[e]) + 1.f));
    x[2 * e + 1] = rbf(x[2 * e + 1] * inv * (bf_hi(ww[e]) + 1.f));
  }
  float other[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) other[e] = __shfl_xor_sync(FULL, x[e], 4);
  if (lane < 8) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      x[e] = lane < 4 ? rbf(x[e] * rp.c[e] - other[e] * rp.s[e]) : rbf(x[e] * rp.c[e] + other[e] * rp.s[e]);
  }
  return make_uint2(e4m3x2(x[0], x[1]) | (e4m3x2(x[2], x[3]) << 16), e4m3x2(x[4], x[5]) | (e4m3x2(x[6], x[7]) << 16));
}
struct Consumer {
  uint32_t qa[16][4];
  float o[2][4][4];
  float m[4];
  float l[2];
};
constexpr float RESCALE_LOG2 = 8.f;
__device__ __forceinline__ void s_mma(const unsigned char* kt, const Consumer& cs, float (&acc)[2][2][4]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, kg = warp >> 1;
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) acc[a][nt][0] = acc[a][nt][1] = acc[a][nt][2] = acc[a][nt][3] = 0.f;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg) {
      uint32_t kb[4];
      ldsm_x4(smem_u32(kt + (16 * kg + 8 * nt + (lane & 7)) * KROW + 64 * cg + 16 * (lane >> 3)), kb);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_f16(acc[j & 1][nt], cs.qa[4 * cg + j], f16x2_lo(kb[j]), f16x2_hi(kb[j]));
    }
}
template <typename Valid>
__device__ __forceinline__ void s_scale(const float (&acc)[2][2][4], float (&s)[2][4], float* redm, float scale_log2,
                                        bool s_on, Valid valid) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int mt = warp & 1, kg = warp >> 1;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int key = 16 * kg + 8 * nt + 2 * t + (e & 1), tok = 2 * mt + (e >> 1);
      s[nt][e] = (s_on && valid(key, tok)) ? (acc[0][nt][e] + acc[1][nt][e]) * scale_log2 : -INFINITY;
    }
  float ma = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
  float mb = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
  ma = fmaxf(ma, __shfl_xor_sync(FULL, ma, 1));
  ma = fmaxf(ma, __shfl_xor_sync(FULL, ma, 2));
  mb = fmaxf(mb, __shfl_xor_sync(FULL, mb, 1));
  mb = fmaxf(mb, __shfl_xor_sync(FULL, mb, 2));
  if (t == 0) {
    redm[(8 * (2 * mt) + g) * 4 + kg] = ma;
    redm[(8 * (2 * mt + 1) + g) * 4 + kg] = mb;
  }
}
__device__ __forceinline__ void softmax_pv(const float (&s)[2][4], const unsigned char* vt, unsigned char* ps,
                                           const float* redm, Consumer& cs, bool s_on, int ksteps) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int mt = warp & 1, kg = warp >> 1;
  float mnew[4];
  bool move = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 v = *reinterpret_cast<const float4*>(redm + (8 * i + g) * 4);
    const float tm = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    mnew[i] = tm > cs.m[i] + RESCALE_LOG2 ? tm : cs.m[i];
    move |= mnew[i] != cs.m[i];
  }
  if (__any_sync(FULL, move)) {
    float alpha[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      alpha[i] = exp2f(cs.m[i] - mnew[i]);
      cs.m[i] = mnew[i];
    }
    cs.l[0] *= mt ? alpha[2] : alpha[0];
    cs.l[1] *= mt ? alpha[3] : alpha[1];
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        cs.o[m2][nt][0] *= alpha[2 * m2];
        cs.o[m2][nt][1] *= alpha[2 * m2];
        cs.o[m2][nt][2] *= alpha[2 * m2 + 1];
        cs.o[m2][nt][3] *= alpha[2 * m2 + 1];
      }
  }
  const float m_a = mt ? cs.m[2] : cs.m[0], m_b = mt ? cs.m[3] : cs.m[1];
  float sa = 0.f, sb = 0.f;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    const float p0 = exp2f(s[nt][0] - m_a), p1 = exp2f(s[nt][1] - m_a);
    const float p2 = exp2f(s[nt][2] - m_b), p3 = exp2f(s[nt][3] - m_b);
    sa += p0 + p1;
    sb += p2 + p3;
    if (s_on) {
      unsigned char* prow = ps + (16 * mt + g) * PROW + (16 * kg + 8 * nt + 2 * t) * 2;
      *reinterpret_cast<uint32_t*>(prow) = pack_f16x2(p0, p1);
      *reinterpret_cast<uint32_t*>(prow + 8 * PROW) = pack_f16x2(p2, p3);
    }
  }
  cs.l[0] += sa;
  cs.l[1] += sb;
  named_sync(1, CTHREADS);
  for (int ks = 0; ks < ksteps; ++ks) {
    uint32_t pa[2][4], vr[4], vb[4][2];
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
      ldsm_x4(smem_u32(ps + (16 * m2 + (lane & 7) + 8 * ((lane >> 3) & 1)) * PROW + (16 * ks + 8 * (lane >> 4)) * 2),
              pa[m2]);
    ldsm_x4_t(smem_u32(vt + (16 * ks + (lane & 7) + 8 * ((lane >> 3) & 1)) * KROW + 32 * warp + 16 * (lane >> 4)), vr);
#pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
      vb[2 * hb][0] = f16x2_lo(prmt(vr[2 * hb], 0x0020));
      vb[2 * hb][1] = f16x2_lo(prmt(vr[2 * hb + 1], 0x0020));
      vb[2 * hb + 1][0] = f16x2_lo(prmt(vr[2 * hb], 0x0031));
      vb[2 * hb + 1][1] = f16x2_lo(prmt(vr[2 * hb + 1], 0x0031));
    }
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma_f16(cs.o[m2][nt], pa[m2], vb[nt][0], vb[nt][1]);
  }
}
#if QK_SPLIT_RS
struct Split {
  int s, h, r, S, grp;
};
__device__ __forceinline__ int kth_set(unsigned lo, unsigned hi, int k) {
  const int nlo = __popc(lo);
  unsigned m = k < nlo ? lo : hi;
  for (int i = k < nlo ? k : k - nlo; i > 0; --i) m &= m - 1;
  return (k < nlo ? 0 : 32) + __ffs(m) - 1;
}
__device__ __forceinline__ Split rs_split(const Args& p, int lane) {
  const bool in0 = lane < p.batch, in1 = lane + 32 < p.batch;
  const int n0 = in0 ? p.seqlens[lane] - NT : 0, n1 = in1 ? p.seqlens[lane + 32] - NT : 0;
  const unsigned h0 = __ballot_sync(FULL, in0 && n0 > TILE), h1 = __ballot_sync(FULL, in1 && n1 > TILE);
  const unsigned l0 = __ballot_sync(FULL, in0) & ~h0, l1 = __ballot_sync(FULL, in1) & ~h1;
  const int heavy = HKV * (__popc(h0) + __popc(h1)), light = HKV * (__popc(l0) + __popc(l1));
  const int ctas = static_cast<int>(gridDim.x), c = static_cast<int>(blockIdx.x);
  const int sh = heavy ? max(1, min(RS_MAX_SPLITS, (ctas - light) / heavy)) : 1;
  Split m{0, 0, 0, 0, 0};
  if (c < heavy * sh) {
    const int hg = c / sh;
    m = Split{c - hg * sh, hg % HKV, kth_set(h0, h1, hg / HKV), sh, hg};
  } else if (c < heavy * sh + light) {
    const int lg = c - heavy * sh;
    m = Split{0, lg % HKV, kth_set(l0, l1, lg / HKV), 1, heavy + lg};
  }
  return m;
}
#endif
__global__ void __launch_bounds__(THREADS, 1) attn_verify_kernel(const Args p) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* ks = smem;
  unsigned char* vs = ks + STAGES * TILE * KROW;
  unsigned char* nk = vs + STAGES * TILE * KROW;
  unsigned char* nv = nk + NEWROWS * KROW;
  unsigned char* q8 = nv + NEWROWS * KROW;
  unsigned char* ps = q8 + ROWS * KROW;
  unsigned char* gs = ps + ROWS * PROW;
  float* redm = reinterpret_cast<float*>(gs + ROWS * 2 * D);
  float* redl = redm + ROWS * 4;
  uint64_t* full = reinterpret_cast<uint64_t*>(redl + ROWS * 4);
  uint64_t* empty = full + STAGES;
#if QK_SPLIT_RS
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const Split sp = rs_split(p, lane);
  if (sp.S == 0) return;
  const int s = sp.s, h = sp.h, r = sp.r, S = sp.S, grp = sp.grp, cta = grp * S + s;
#else
  const int s = blockIdx.x, h = blockIdx.y, r = blockIdx.z, S = p.splits;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int grp = r * HKV + h, cta = grp * S + s;
#endif
  const bool last = s == S - 1;
  const int n_prefix = max(p.seqlens[r] - NT, 0);
  const int tiles = (n_prefix + TILE - 1) / TILE, weighted = tiles + (S > 1 ? 2 : 0);
  const int q = weighted / S, rem = weighted % S;
  const int t0 = s * q + min(s, rem), t1 = min(t0 + q + (s < rem ? 1 : 0), tiles);
  const int lo = min(t0 * TILE, n_prefix), hi = min(t1 * TILE, n_prefix);
  const int ntiles = (hi - lo + TILE - 1) / TILE;
  const int span = ((D + S - 1) / S + 7) / 8 * 8, d0 = min(s * span, D), dn = min(span, D - d0);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(smem_u32(&full[i]), 1);
      mbar_init(smem_u32(&empty[i]), CWARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp >= CWARPS) {
    const int pw = warp - CWARPS, key = 16 * pw + (lane & 15), isv = lane >> 4;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const int32_t* pt = p.page_table + static_cast<int64_t>(r) * p.pt_stride;
    const uint8_t* base = (isv ? p.vcache : p.kcache) + static_cast<int64_t>(h) * D;
    unsigned char* ring_base = isv ? vs : ks;
    auto pt_load = [&](int it) -> int {
      const int key0 = lo + it * TILE;
      return it < ntiles ? pt[key0 + min(key, hi - key0 - 1)] : 0;
    };
    int ring[4] = {pt_load(0), pt_load(1), pt_load(2), pt_load(3)};
    for (int it0 = 0; it0 < ntiles; it0 += 4) {
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int it = it0 + u;
        if (it < ntiles) {
          const int st = it % STAGES, slot = ring[u];
          ring[u] = pt_load(it + 4);
          if (it >= STAGES) mbar_wait(smem_u32(&empty[st]), ((it / STAGES) - 1) & 1);
          const uint32_t fb = smem_u32(&full[st]);
          if (pw == 0 && lane == 0) mbar_expect(fb, 2 * TILE * D);
          bulk_g2s(smem_u32(ring_base + (st * TILE + key) * KROW), base + static_cast<int64_t>(slot) * (HKV * D), D, fb,
                   pol);
        }
      }
    }
  } else {
    const int64_t qtoken = static_cast<int64_t>(NT) * r + (warp >> 1);
    const int64_t ntoken = static_cast<int64_t>(NT) * r + (warp & 3);
    const Rope qrope = load_rope(p, qtoken, lane);
    const uint4 qw = *reinterpret_cast<const uint4*>(p.qw + 8 * lane);
    Rope krope;
    uint4 kw = make_uint4(0u, 0u, 0u, 0u);
    int64_t dst = 0;
    if (last) {
      if (warp < 4) {
        krope = load_rope(p, ntoken, lane);
        kw = *reinterpret_cast<const uint4*>(p.kw + 8 * lane);
      }
      dst = (p.out_loc[ntoken] * HKV + h) * D + 8 * lane;
    }
    griddep_wait();
    for (int e = tid; e < ROWS * (dn / 8); e += CTHREADS) {
      const int row = e / (dn / 8), c = e % (dn / 8);
      const int64_t token = static_cast<int64_t>(NT) * r + (row >> 3);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(gs + row * 2 * D + 16 * c)),
                   "l"(p.qkv + token * QKV_W + (h * GQ + (row & 7)) * 2 * D + D + d0 + 8 * c)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int row = 4 * warp + u;
      const uint4 raw = *reinterpret_cast<const uint4*>(p.qkv + qtoken * QKV_W + (h * GQ + (row & 7)) * 2 * D + 8 * lane);
      *reinterpret_cast<uint2*>(q8 + row * KROW + 8 * lane) = norm_rope_e4m3(raw, qw, qrope, p.eps, lane);
    }
    if (last) {
      const int tok = warp & 3;
      uint2 bytes;
      if (warp < 4) {
        const uint4 raw = *reinterpret_cast<const uint4*>(p.qkv + ntoken * QKV_W + K_OFF + h * D + 8 * lane);
        bytes = norm_rope_e4m3(raw, kw, krope, p.eps, lane);
        *reinterpret_cast<uint2*>(p.kcache + dst) = bytes;
        *reinterpret_cast<uint2*>(nk + tok * KROW + 8 * lane) = bytes;
      } else {
        const uint4 raw = *reinterpret_cast<const uint4*>(p.qkv + ntoken * QKV_W + V_OFF + h * D + 8 * lane);
        bytes = make_uint2(e4m3x2(bf_lo(raw.x), bf_hi(raw.x)) | (e4m3x2(bf_lo(raw.y), bf_hi(raw.y)) << 16),
                           e4m3x2(bf_lo(raw.z), bf_hi(raw.z)) | (e4m3x2(bf_lo(raw.w), bf_hi(raw.w)) << 16));
        *reinterpret_cast<uint2*>(p.vcache + dst) = bytes;
        *reinterpret_cast<uint2*>(nv + tok * KROW + 8 * lane) = bytes;
      }
      for (int e = tid; e < 2 * (NEWROWS - NT) * (D / 16); e += CTHREADS) {
        const int buf = e / ((NEWROWS - NT) * (D / 16)), rem = e % ((NEWROWS - NT) * (D / 16));
        *reinterpret_cast<uint4*>((buf ? nv : nk) + (NT + rem / (D / 16)) * KROW + 16 * (rem % (D / 16))) =
            make_uint4(0u, 0u, 0u, 0u);
      }
    }
    named_sync(1, CTHREADS);
    Consumer cs;
    {
      const int mt = warp & 1;
#pragma unroll
      for (int c2 = 0; c2 < 8; ++c2) {
        uint32_t q[4];
        ldsm_x4(smem_u32(q8 + (16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1)) * KROW + 32 * c2 + 16 * (lane >> 4)), q);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          cs.qa[2 * c2 + j][0] = f16x2_lo(q[2 * j]);
          cs.qa[2 * c2 + j][1] = f16x2_lo(q[2 * j + 1]);
          cs.qa[2 * c2 + j][2] = f16x2_hi(q[2 * j]);
          cs.qa[2 * c2 + j][3] = f16x2_hi(q[2 * j + 1]);
        }
      }
    }
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) cs.o[m2][nt][0] = cs.o[m2][nt][1] = cs.o[m2][nt][2] = cs.o[m2][nt][3] = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) cs.m[i] = -INFINITY;
    cs.l[0] = cs.l[1] = 0.f;
    float acc[2][2][4];
    if (ntiles > 0) {
      mbar_wait(smem_u32(&full[0]), 0);
      s_mma(ks, cs, acc);
    }
    for (int it = 0; it < ntiles; ++it) {
      const int st = it % STAGES, nvalid = min(TILE, hi - (lo + it * TILE));
      float sc[2][4];
      s_scale(acc, sc, redm, p.scale_log2, true, [nvalid](int key, int) { return key < nvalid; });
      if (it + 1 < ntiles) {
        const int st1 = (it + 1) % STAGES;
        mbar_wait(smem_u32(&full[st1]), ((it + 1) / STAGES) & 1);
        s_mma(ks + st1 * TILE * KROW, cs, acc);
      }
      named_sync(1, CTHREADS);
      softmax_pv(sc, vs + st * TILE * KROW, ps, redm, cs, true, TILE / 16);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&empty[st]));
    }
    if (last) {
      const bool on = warp < 2;
      if (on) s_mma(nk, cs, acc);
      float sc[2][4];
      s_scale(acc, sc, redm, p.scale_log2, on, [](int key, int tok) { return key <= tok; });
      named_sync(1, CTHREADS);
      softmax_pv(sc, nv, ps, redm, cs, on, 1);
    }
    {
      const int mt = warp & 1, kg = warp >> 1;
      float la = cs.l[0], lb = cs.l[1];
      la += __shfl_xor_sync(FULL, la, 1);
      la += __shfl_xor_sync(FULL, la, 2);
      lb += __shfl_xor_sync(FULL, lb, 1);
      lb += __shfl_xor_sync(FULL, lb, 2);
      if (t == 0) {
        redl[(8 * (2 * mt) + g) * 4 + kg] = la;
        redl[(8 * (2 * mt + 1) + g) * 4 + kg] = lb;
      }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    named_sync(1, CTHREADS);
    if (S == 1) {
#pragma unroll
      for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = 16 * m2 + 8 * hf + g;
          const float4 lv = *reinterpret_cast<const float4*>(redl + row * 4);
          const float inv = 1.f / ((lv.x + lv.y) + (lv.z + lv.w));
          const int64_t token = static_cast<int64_t>(NT) * r + (row >> 3);
          bf16* orow = p.out + token * OUT_W + (h * GQ + (row & 7)) * D;
#pragma unroll
          for (int hb = 0; hb < 2; ++hb) {
            const int dim = 32 * warp + 16 * hb + 4 * t;
            const uint2 gw = *reinterpret_cast<const uint2*>(gs + row * 2 * D + 2 * dim);
            const float a[4] = {cs.o[m2][2 * hb][2 * hf], cs.o[m2][2 * hb + 1][2 * hf], cs.o[m2][2 * hb][2 * hf + 1],
                                cs.o[m2][2 * hb + 1][2 * hf + 1]};
            const float gv[4] = {bf_lo(gw.x), bf_hi(gw.x), bf_lo(gw.y), bf_hi(gw.y)};
            uint32_t w2[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const float y0 = rbf(a[2 * k] * inv) * (1.f / (1.f + __expf(-gv[2 * k])));
              const float y1 = rbf(a[2 * k + 1] * inv) * (1.f / (1.f + __expf(-gv[2 * k + 1])));
              const __nv_bfloat162 b = __floats2bfloat162_rn(y0, y1);
              w2[k] = *reinterpret_cast<const uint32_t*>(&b);
            }
            *reinterpret_cast<uint2*>(orow + dim) = make_uint2(w2[0], w2[1]);
          }
        }
    } else {
      float* wo = p.ws_o + static_cast<int64_t>(cta) * ROWS * D;
#pragma unroll
      for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = 16 * m2 + 8 * hf + g;
#pragma unroll
          for (int hb = 0; hb < 2; ++hb)
            *reinterpret_cast<float4*>(wo + row * D + 32 * warp + 16 * hb + 4 * t) =
                make_float4(cs.o[m2][2 * hb][2 * hf], cs.o[m2][2 * hb + 1][2 * hf], cs.o[m2][2 * hb][2 * hf + 1],
                            cs.o[m2][2 * hb + 1][2 * hf + 1]);
        }
      if (warp == 0 && t == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 v = *reinterpret_cast<const float4*>(redl + (8 * i + g) * 4);
          p.ws_ml[cta * 2 * ROWS + 8 * i + g] = cs.m[i];
          p.ws_ml[cta * 2 * ROWS + ROWS + 8 * i + g] = (v.x + v.y) + (v.z + v.w);
        }
      }
    }
  }
  if (S == 1) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    atomicAdd(&p.arrive[grp], 1);
    while (ld_acquire(&p.arrive[grp]) < S) {
    }
  }
  __syncthreads();
  float* mls = reinterpret_cast<float*>(ks);
  float* wgt = mls + S * 2 * ROWS;
  const float* ml = p.ws_ml + static_cast<int64_t>(grp) * S * 2 * ROWS;
#if QK_COMBINE_STAGE
  float* ost = wgt + S * ROWS;
  const float* wsg = p.ws_o + static_cast<int64_t>(grp) * S * ROWS * D;
  for (int e = tid; e < S * ROWS * (dn / 4); e += THREADS) {
    const int j = e / (ROWS * (dn / 4)), row = (e / (dn / 4)) % ROWS, c = e % (dn / 4);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(ost + (j * ROWS + row) * dn + 4 * c)),
                 "l"(wsg + (static_cast<int64_t>(j) * ROWS + row) * D + d0 + 4 * c)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
  for (int e = tid; e < S * 2 * ROWS; e += THREADS) mls[e] = __ldcg(ml + e);
  __syncthreads();
  if (tid < ROWS) {
    float mmax = -INFINITY;
    for (int j = 0; j < S; ++j) mmax = fmaxf(mmax, mls[j * 2 * ROWS + tid]);
    float l = 0.f;
    for (int j = 0; j < S; ++j) {
      const float w = exp2f(mls[j * 2 * ROWS + tid] - mmax);
      wgt[j * ROWS + tid] = w;
      l += w * mls[j * 2 * ROWS + ROWS + tid];
    }
    const float inv = 1.f / l;
    for (int j = 0; j < S; ++j) wgt[j * ROWS + tid] *= inv;
  }
#if QK_COMBINE_STAGE
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
#else
  __syncthreads();
  const float* wsg = p.ws_o + static_cast<int64_t>(grp) * S * ROWS * D;
#endif
  for (int e = tid; e < ROWS * (dn / 4); e += THREADS) {
    const int row = e / (dn / 4), dim = d0 + 4 * (e % (dn / 4));
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
    for (int j = 0; j < S; ++j) {
      const float w = wgt[j * ROWS + row];
#if QK_COMBINE_STAGE
      const float4 v = *reinterpret_cast<const float4*>(ost + (j * ROWS + row) * dn + (dim - d0));
#else
      const float4 v = __ldcg(reinterpret_cast<const float4*>(wsg + (static_cast<int64_t>(j) * ROWS + row) * D + dim));
#endif
      a.x += w * v.x;
      a.y += w * v.y;
      a.z += w * v.z;
      a.w += w * v.w;
    }
    const int64_t token = static_cast<int64_t>(NT) * r + (row >> 3);
    const uint2 gw = *reinterpret_cast<const uint2*>(gs + row * 2 * D + 2 * (dim - d0));
    const float y0 = rbf(a.x) * (1.f / (1.f + __expf(-bf_lo(gw.x))));
    const float y1 = rbf(a.y) * (1.f / (1.f + __expf(-bf_hi(gw.x))));
    const float y2 = rbf(a.z) * (1.f / (1.f + __expf(-bf_lo(gw.y))));
    const float y3 = rbf(a.w) * (1.f / (1.f + __expf(-bf_hi(gw.y))));
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(y0, y1), b1 = __floats2bfloat162_rn(y2, y3);
    *reinterpret_cast<uint2*>(p.out + token * OUT_W + (h * GQ + (row & 7)) * D + dim) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&b0), *reinterpret_cast<const uint32_t*>(&b1));
  }
  __syncthreads();
  if (tid == 0 && atomicAdd(&p.depart[grp], 1) == S - 1) {
    p.arrive[grp] = 0;
    p.depart[grp] = 0;
    __threadfence();
  }
}
struct PArgs {
  const bf16* qkv;
  const bf16* kv;
  int64_t q_ld, kv_ld;
  const bf16* qw;
  const bf16* kw;
  float eps;
  const void* cos_sin;
  int64_t cs_stride;
  int cs_f32;
  const int64_t* pos;
  int64_t pos_axis_stride;
  const int64_t* axis_map;
  uint8_t* kcache;
  uint8_t* vcache;
  const int64_t* out_loc;
  uint8_t* q8;
  int64_t T;
};
constexpr int PROWS = HQ + 2 * HKV, PBATCH = 5;
__global__ void __launch_bounds__(256) attn_prefill_front_kernel(const PArgs p) {
  const int lane = threadIdx.x & 31;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
  if (t >= p.T) return;
  const bf16* src = p.qkv + t * p.q_ld + 8 * lane;
  const bf16* ksrc = p.kv + t * p.kv_ld + 8 * lane;
  const Rope rp = load_rope(p, t, lane);
  const uint4 qw = *reinterpret_cast<const uint4*>(p.qw + 8 * lane);
  const uint4 kw = *reinterpret_cast<const uint4*>(p.kw + 8 * lane);
  const int64_t slot = p.out_loc[t];
#pragma unroll
  for (int b = 0; b < PROWS; b += PBATCH) {
    uint4 raw[PBATCH];
#pragma unroll
    for (int i = 0; i < PBATCH; ++i) {
      const int kind = b + i;
      raw[i] = *reinterpret_cast<const uint4*>(kind < HQ ? src + kind * 2 * D : ksrc + (kind - HQ) * D);
    }
#pragma unroll
    for (int i = 0; i < PBATCH; ++i) {
      const int kind = b + i;
      uint2 o;
      uint8_t* dst;
      if (kind < HQ + HKV) {
        o = norm_rope_e4m3(raw[i], kind < HQ ? qw : kw, rp, p.eps, lane);
        dst = kind < HQ ? p.q8 + (t * HQ + kind) * D : p.kcache + (slot * HKV + (kind - HQ)) * D;
      } else {
        const uint4 r = raw[i];
        o = make_uint2(e4m3x2(bf_lo(r.x), bf_hi(r.x)) | (e4m3x2(bf_lo(r.y), bf_hi(r.y)) << 16),
                       e4m3x2(bf_lo(r.z), bf_hi(r.z)) | (e4m3x2(bf_lo(r.w), bf_hi(r.w)) << 16));
        dst = p.vcache + (slot * HKV + (kind - HQ - HKV)) * D;
      }
      *reinterpret_cast<uint2*>(dst + 8 * lane) = o;
    }
  }
}
void check(bool ok, const char* what) { TORCH_CHECK(ok, what); }
}
torch::Tensor attn_verify_fused(torch::Tensor qkv, torch::Tensor q_norm_w, torch::Tensor k_norm_w, double eps,
                                torch::Tensor cos_sin, torch::Tensor positions, c10::optional<torch::Tensor> axis_map,
                                torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out_loc,
                                torch::Tensor page_table, torch::Tensor seqlens, double scale, int64_t splits,
                                torch::Tensor ws_o, torch::Tensor ws_ml, torch::Tensor counters) {
  const int64_t T = qkv.size(0);
  check(T % NT == 0, "verify rows must be a multiple of the draft window");
  const int B = static_cast<int>(T / NT);
  check(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(1) == QKV_W &&
            qkv.is_contiguous(),
        "qkv must be contiguous bf16 [T, 9216]");
  check(q_norm_w.scalar_type() == at::kBFloat16 && q_norm_w.numel() == D && q_norm_w.is_contiguous() &&
            k_norm_w.scalar_type() == at::kBFloat16 && k_norm_w.numel() == D && k_norm_w.is_contiguous(),
        "q/k norm weights must be contiguous bf16 [256]");
  check((cos_sin.scalar_type() == at::kFloat || cos_sin.scalar_type() == at::kBFloat16) && cos_sin.dim() == 2 &&
            cos_sin.size(1) == 2 * HALF_ROT && cos_sin.stride(1) == 1,
        "cos_sin_cache must be [max_pos, 64] fp32 or bf16");
  check(positions.scalar_type() == at::kLong, "positions must be int64");
  const bool mrope = positions.dim() == 2;
  check(mrope ? (positions.size(0) == 3 && positions.size(1) >= T && positions.stride(1) == 1 && axis_map.has_value() &&
                 axis_map->scalar_type() == at::kLong && axis_map->numel() == HALF_ROT && axis_map->is_contiguous())
              : (positions.dim() == 1 && positions.size(0) >= T && positions.is_contiguous() && !axis_map.has_value()),
        "positions must be [T] or [3, T] with a 32-lane int64 axis map");
  for (auto* c : {&k_cache, &v_cache}) {
    check(c->is_cuda() && c->element_size() == 1 && c->is_contiguous() && c->numel() % (HKV * D) == 0,
          "KV cache must be contiguous one-byte [slots, 2, 256]");
  }
  check(out_loc.scalar_type() == at::kLong && out_loc.numel() >= T && out_loc.is_contiguous(),
        "out_cache_loc must hold T int64 slots");
  check(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) >= B &&
            page_table.stride(1) == 1,
        "page_table must be int32 [>= B, max_pages]");
  check(seqlens.scalar_type() == at::kInt && seqlens.numel() >= B && seqlens.is_contiguous(),
        "cache_seqlens must hold B int32 lengths");
  check(splits >= 1, "splits must be positive");
#if QK_SPLIT_RS
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  check(B * HKV <= MAX_GROUPS && B * HKV <= sms, "too many (request, head) groups for one CTA per SM");
  check(ws_o.scalar_type() == at::kFloat && ws_o.numel() >= static_cast<int64_t>(sms) * ROWS * D &&
            ws_ml.scalar_type() == at::kFloat && ws_ml.numel() >= static_cast<int64_t>(sms) * 2 * ROWS &&
            counters.scalar_type() == at::kInt && counters.numel() >= 2 * MAX_GROUPS,
        "workspace too small");
#else
  if (splits > 1) {
    check(B * HKV * splits <= at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
          "splits x requests x kv heads must fit one CTA per SM (the splits wait for each other)");
    check(splits * 3 * ROWS * 4 <= STAGES * TILE * KROW, "too many splits for the combine scratch");
#if QK_COMBINE_STAGE
    check(splits <= CS_MAX_SPLITS, "too many splits for the staged combine scratch");
#endif
    check(B * HKV <= MAX_GROUPS, "too many (request, head) groups");
    check(ws_o.scalar_type() == at::kFloat && ws_o.numel() >= static_cast<int64_t>(B) * HKV * splits * ROWS * D &&
              ws_ml.scalar_type() == at::kFloat &&
              ws_ml.numel() >= static_cast<int64_t>(B) * HKV * splits * 2 * ROWS &&
              counters.scalar_type() == at::kInt && counters.numel() >= 2 * MAX_GROUPS,
          "workspace too small");
  }
#endif
  auto out = torch::empty({T, OUT_W}, qkv.options());
  if (B == 0) return out;
  const c10::cuda::CUDAGuard guard(qkv.device());
  Args p{};
  p.qkv = reinterpret_cast<const bf16*>(qkv.data_ptr());
  p.qw = reinterpret_cast<const bf16*>(q_norm_w.data_ptr());
  p.kw = reinterpret_cast<const bf16*>(k_norm_w.data_ptr());
  p.eps = static_cast<float>(eps);
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.cos_sin = cos_sin.data_ptr();
  p.cs_stride = cos_sin.stride(0);
  p.cs_f32 = cos_sin.scalar_type() == at::kFloat;
  p.pos = positions.data_ptr<int64_t>();
  p.pos_axis_stride = mrope ? positions.stride(0) : 0;
  p.axis_map = mrope ? axis_map->data_ptr<int64_t>() : nullptr;
  p.kcache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.vcache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.out_loc = out_loc.data_ptr<int64_t>();
  p.page_table = page_table.data_ptr<int32_t>();
  p.pt_stride = page_table.stride(0);
  p.seqlens = seqlens.data_ptr<int32_t>();
  p.splits = static_cast<int>(splits);
#if QK_SPLIT_RS
  p.batch = B;
#endif
  p.ws_o = ws_o.data_ptr<float>();
  p.ws_ml = ws_ml.data_ptr<float>();
  p.arrive = counters.data_ptr<int32_t>();
  p.depart = p.arrive + MAX_GROUPS;
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(attn_verify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    configured = true;
  }
  cudaLaunchConfig_t cfg = {};
#if QK_SPLIT_RS
  cfg.gridDim = dim3(static_cast<unsigned>(sms));
#else
  cfg.gridDim = dim3(static_cast<unsigned>(splits), HKV, B);
#endif
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, attn_verify_kernel, p));
  return out;
}
static torch::Tensor prefill_front(torch::Tensor qg, int64_t q_ld, const bf16* kv, int64_t kv_ld, torch::Tensor q_norm_w,
                                   torch::Tensor k_norm_w, double eps, torch::Tensor cos_sin, torch::Tensor positions,
                                   c10::optional<torch::Tensor> axis_map, torch::Tensor k_cache, torch::Tensor v_cache,
                                   torch::Tensor out_loc) {
  const int64_t T = qg.size(0);
  torch::Tensor qkv = qg;
  check(q_norm_w.scalar_type() == at::kBFloat16 && q_norm_w.numel() == D && q_norm_w.is_contiguous() &&
            k_norm_w.scalar_type() == at::kBFloat16 && k_norm_w.numel() == D && k_norm_w.is_contiguous(),
        "q/k norm weights must be contiguous bf16 [256]");
  check((cos_sin.scalar_type() == at::kFloat || cos_sin.scalar_type() == at::kBFloat16) && cos_sin.dim() == 2 &&
            cos_sin.size(1) == 2 * HALF_ROT && cos_sin.stride(1) == 1,
        "cos_sin_cache must be [max_pos, 64] fp32 or bf16");
  check(positions.scalar_type() == at::kLong, "positions must be int64");
  const bool mrope = positions.dim() == 2;
  check(mrope ? (positions.size(0) == 3 && positions.size(1) >= T && positions.stride(1) == 1 && axis_map.has_value() &&
                 axis_map->scalar_type() == at::kLong && axis_map->numel() == HALF_ROT && axis_map->is_contiguous())
              : (positions.dim() == 1 && positions.size(0) >= T && positions.is_contiguous() && !axis_map.has_value()),
        "positions must be [T] or [3, T] with a 32-lane int64 axis map");
  for (auto* c : {&k_cache, &v_cache}) {
    check(c->is_cuda() && c->element_size() == 1 && c->is_contiguous() && c->numel() % (HKV * D) == 0,
          "KV cache must be contiguous one-byte [slots, 2, 256]");
  }
  check(out_loc.scalar_type() == at::kLong && out_loc.numel() >= T && out_loc.is_contiguous(),
        "out_cache_loc must hold T int64 slots");
  auto q8 = torch::empty({T, HQ * D}, qkv.options().dtype(at::kFloat8_e4m3fn));
  if (T == 0) return q8;
  const c10::cuda::CUDAGuard guard(qkv.device());
  PArgs p{};
  p.qkv = reinterpret_cast<const bf16*>(qkv.data_ptr());
  p.kv = kv;
  p.q_ld = q_ld;
  p.kv_ld = kv_ld;
  p.qw = reinterpret_cast<const bf16*>(q_norm_w.data_ptr());
  p.kw = reinterpret_cast<const bf16*>(k_norm_w.data_ptr());
  p.eps = static_cast<float>(eps);
  p.cos_sin = cos_sin.data_ptr();
  p.cs_stride = cos_sin.stride(0);
  p.cs_f32 = cos_sin.scalar_type() == at::kFloat;
  p.pos = positions.data_ptr<int64_t>();
  p.pos_axis_stride = mrope ? positions.stride(0) : 0;
  p.axis_map = mrope ? axis_map->data_ptr<int64_t>() : nullptr;
  p.kcache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.vcache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.out_loc = out_loc.data_ptr<int64_t>();
  p.q8 = reinterpret_cast<uint8_t*>(q8.data_ptr());
  p.T = T;
  const unsigned blocks = static_cast<unsigned>((T + 7) / 8);
  attn_prefill_front_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q8;
}
torch::Tensor attn_prefill_front(torch::Tensor qkv, torch::Tensor q_norm_w, torch::Tensor k_norm_w, double eps,
                                 torch::Tensor cos_sin, torch::Tensor positions, c10::optional<torch::Tensor> axis_map,
                                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out_loc) {
  check(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(1) == QKV_W &&
            qkv.is_contiguous(),
        "qkv must be contiguous bf16 [T, 9216]");
  return prefill_front(qkv, QKV_W, reinterpret_cast<const bf16*>(qkv.data_ptr()) + K_OFF, QKV_W, q_norm_w, k_norm_w, eps,
                       cos_sin, positions, axis_map, k_cache, v_cache, out_loc);
}
torch::Tensor attn_prefill_front_split(torch::Tensor qg, torch::Tensor kv, torch::Tensor q_norm_w, torch::Tensor k_norm_w,
                                       double eps, torch::Tensor cos_sin, torch::Tensor positions,
                                       c10::optional<torch::Tensor> axis_map, torch::Tensor k_cache, torch::Tensor v_cache,
                                       torch::Tensor out_loc) {
  check(qg.is_cuda() && qg.scalar_type() == at::kBFloat16 && qg.dim() == 2 && qg.size(1) == K_OFF && qg.is_contiguous(),
        "qg must be contiguous bf16 [T, 8192]");
  check(kv.is_cuda() && kv.scalar_type() == at::kBFloat16 && kv.dim() == 2 && kv.size(0) == qg.size(0) &&
            kv.size(1) == QKV_W - K_OFF && kv.is_contiguous(),
        "kv must be contiguous bf16 [T, 1024]");
  return prefill_front(qg, K_OFF, reinterpret_cast<const bf16*>(kv.data_ptr()), QKV_W - K_OFF, q_norm_w, k_norm_w, eps,
                       cos_sin, positions, axis_map, k_cache, v_cache, out_loc);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("attn_verify_fused", &attn_verify_fused, py::arg("qkv"), py::arg("q_norm_w"), py::arg("k_norm_w"),
        py::arg("eps"), py::arg("cos_sin"), py::arg("positions"), py::arg("axis_map"), py::arg("k_cache"),
        py::arg("v_cache"), py::arg("out_loc"), py::arg("page_table"), py::arg("seqlens"), py::arg("scale"),
        py::arg("splits"), py::arg("ws_o"), py::arg("ws_ml"), py::arg("counters"));
  m.def("attn_prefill_front", &attn_prefill_front, py::arg("qkv"), py::arg("q_norm_w"), py::arg("k_norm_w"),
        py::arg("eps"), py::arg("cos_sin"), py::arg("positions"), py::arg("axis_map"), py::arg("k_cache"),
        py::arg("v_cache"), py::arg("out_loc"));
  m.def("attn_prefill_front_split", &attn_prefill_front_split, py::arg("qg"), py::arg("kv"), py::arg("q_norm_w"),
        py::arg("k_norm_w"), py::arg("eps"), py::arg("cos_sin"), py::arg("positions"), py::arg("axis_map"),
        py::arg("k_cache"), py::arg("v_cache"), py::arg("out_loc"));
}
