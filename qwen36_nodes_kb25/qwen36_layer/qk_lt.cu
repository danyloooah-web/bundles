// Prefill projections y = x w^T (bf16, fp32 accumulate) through cuBLASLt with a zero-byte workspace.
//
// Stock F.linear lets cuBLASLt pick its top algorithm for these shapes, a persistent kernel that needs a 4-byte
// workspace, so every call is a memset node plus the GEMM: in the prefill graph the GPU idles ~3 us between the
// two (41 GEMMs per chunk here). Asking the heuristic for a zero-byte workspace gives kernels with no memset.
// A plan is only used when its algorithm does not split K (one fp32 accumulator per output, k in order, one bf16
// rounding: the same bits as the default kernel, as DeepGEMM's sequential-K kernel also reproduces them); the
// caller also compares each new shape against F.linear once before using it (lt.py). Every zero-workspace, unsplit-K
// algorithm the heuristic lists gives F.linear's bits at these shapes (measured); where one is clearly faster the
// caller names its tile ids in order of preference (lt.py), else the first listed is used.
//
// cuBLASLt is not among the libraries the build links; torch has already loaded it, so its entry points are
// looked up in the loaded library at first use. Anything missing leaves the caller on F.linear.
//
// kb24: two more parts.
// (1) Pinned plans (linear_cfg / fp8_linear_cfg): a fully specified cuBLASLt configuration instead of the heuristic's
// pick, for shapes where a sweep of every configuration (algo id x tile x stages x swizzle x custom option x cluster
// shape at split-K 1, compared bit for bit with F.linear on real weights) found a faster zero-workspace one; lt.py uses
// a pin only after it compared equal to F.linear on a real call, like the plans above.
// (2) fp8_gemm (namespace fg11): torch._scaled_mm's row-wise FP8 GEMM written out (the same per-element arithmetic:
// one wgmma e4m3 chain of four k32 steps per 128-k block from zero, fp32 adds in k order, bf16((acc * sb) * sa)) with
// 128 x 256 CTA tiles and the next tile's first k block on the tensor cores under each epilogue; fp8.py uses it where
// fp8_gemm_prefer(M, N) holds and only after it compared equal to _scaled_mm on a real call.
#include <dlfcn.h>
#include <cstdint>
#include <map>
#include <vector>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

struct Api {
  decltype(&cublasLtMatmulDescCreate) desc_create = nullptr;
  decltype(&cublasLtMatmulDescSetAttribute) desc_set = nullptr;
  decltype(&cublasLtMatrixLayoutCreate) layout_create = nullptr;
  decltype(&cublasLtMatmulPreferenceCreate) pref_create = nullptr;
  decltype(&cublasLtMatmulPreferenceSetAttribute) pref_set = nullptr;
  decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic = nullptr;
  decltype(&cublasLtMatmulAlgoConfigGetAttribute) algo_get = nullptr;
  decltype(&cublasLtMatmul) matmul = nullptr;
  decltype(&cublasLtMatmulAlgoInit) algo_init = nullptr;
  decltype(&cublasLtMatmulAlgoConfigSetAttribute) algo_set = nullptr;
  decltype(&cublasLtMatmulAlgoCheck) algo_check = nullptr;
  bool ok = false;
  bool pin_ok = false;
};

const Api& api() {
  static Api a;
  static std::once_flag once;
  std::call_once(once, [] {
    void* lib = nullptr;
    for (const char* name : {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}) {
      lib = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
      if (lib != nullptr) break;
    }
    if (lib == nullptr) return;
    a.desc_create = reinterpret_cast<decltype(a.desc_create)>(dlsym(lib, "cublasLtMatmulDescCreate"));
    a.desc_set = reinterpret_cast<decltype(a.desc_set)>(dlsym(lib, "cublasLtMatmulDescSetAttribute"));
    a.layout_create = reinterpret_cast<decltype(a.layout_create)>(dlsym(lib, "cublasLtMatrixLayoutCreate"));
    a.pref_create = reinterpret_cast<decltype(a.pref_create)>(dlsym(lib, "cublasLtMatmulPreferenceCreate"));
    a.pref_set = reinterpret_cast<decltype(a.pref_set)>(dlsym(lib, "cublasLtMatmulPreferenceSetAttribute"));
    a.heuristic = reinterpret_cast<decltype(a.heuristic)>(dlsym(lib, "cublasLtMatmulAlgoGetHeuristic"));
    a.algo_get = reinterpret_cast<decltype(a.algo_get)>(dlsym(lib, "cublasLtMatmulAlgoConfigGetAttribute"));
    a.matmul = reinterpret_cast<decltype(a.matmul)>(dlsym(lib, "cublasLtMatmul"));
    a.algo_init = reinterpret_cast<decltype(a.algo_init)>(dlsym(lib, "cublasLtMatmulAlgoInit"));
    a.algo_set = reinterpret_cast<decltype(a.algo_set)>(dlsym(lib, "cublasLtMatmulAlgoConfigSetAttribute"));
    a.algo_check = reinterpret_cast<decltype(a.algo_check)>(dlsym(lib, "cublasLtMatmulAlgoCheck"));
    a.pin_ok = a.algo_init && a.algo_set && a.algo_check;
    a.ok = a.desc_create && a.desc_set && a.layout_create && a.pref_create && a.pref_set && a.heuristic && a.algo_get &&
           a.matmul;
  });
  return a;
}

struct Plan {
  bool ok = false;
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulAlgo_t algo{};
};

std::mutex plans_mu;
std::map<std::tuple<int, int64_t, int64_t, int64_t, std::vector<int64_t>>, Plan> plans;

// Column-major view of F.linear: C [N, M] = op(A) op(B), A = w as [K, N] transposed, B = x as [K, M].
Plan make_plan(cublasLtHandle_t h, int64_t M, int64_t N, int64_t K, const std::vector<int64_t>& tiles) {
  const Api& f = api();
  Plan p;
  if (f.desc_create(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return p;
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  if (f.desc_set(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) != CUBLAS_STATUS_SUCCESS ||
      f.desc_set(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) != CUBLAS_STATUS_SUCCESS ||
      f.layout_create(&p.a, CUDA_R_16BF, K, N, K) != CUBLAS_STATUS_SUCCESS ||
      f.layout_create(&p.b, CUDA_R_16BF, K, M, K) != CUBLAS_STATUS_SUCCESS ||
      f.layout_create(&p.c, CUDA_R_16BF, N, M, N) != CUBLAS_STATUS_SUCCESS)
    return p;
  cublasLtMatmulPreference_t pref{};
  uint64_t ws = 0;
  if (f.pref_create(&pref) != CUBLAS_STATUS_SUCCESS ||
      f.pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) != CUBLAS_STATUS_SUCCESS)
    return p;
  cublasLtMatmulHeuristicResult_t res[8];
  int n = 0;
  if (f.heuristic(h, p.op, p.a, p.b, p.c, p.c, pref, 8, res, &n) != CUBLAS_STATUS_SUCCESS) return p;
  int first = -1, best = -1, best_rank = 1 << 30;
  for (int i = 0; i < n; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize != 0) continue;
    uint32_t splitk = 0, tile = 0;
    size_t written = 0;
    if (f.algo_get(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &written) !=
            CUBLAS_STATUS_SUCCESS ||
        f.algo_get(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &written) != CUBLAS_STATUS_SUCCESS)
      continue;
    if (splitk > 1) continue;  // a K split would change the accumulation order
    if (first < 0) first = i;
    for (size_t r = 0; r < tiles.size(); ++r)
      if (static_cast<int64_t>(tile) == tiles[r] && static_cast<int>(r) < best_rank) best_rank = static_cast<int>(r), best = i;
  }
  const int pick = best >= 0 ? best : first;
  if (pick >= 0) {
    p.algo = res[pick].algo;
    p.ok = true;
  }
  return p;
}

}  // namespace

// y = x w^T with a zero-workspace, unsplit-K cuBLASLt plan; an empty tensor when no such plan exists.
torch::Tensor linear(torch::Tensor x, torch::Tensor w, std::vector<int64_t> tiles) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                  x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && x.is_contiguous() && w.is_contiguous(),
              "qk_lt.linear: bf16 x [M, K] and w [N, K], contiguous");
  const at::cuda::CUDAGuard guard(x.device());
  if (!api().ok) return torch::empty({0}, x.options());
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  const Plan* p;
  {
    std::lock_guard<std::mutex> lock(plans_mu);
    auto key = std::make_tuple(static_cast<int>(x.get_device()), M, N, K, tiles);
    auto it = plans.find(key);
    if (it == plans.end()) it = plans.emplace(key, make_plan(h, M, N, K, tiles)).first;
    p = &it->second;
  }
  if (!p->ok) return torch::empty({0}, x.options());
  auto y = torch::empty({M, N}, x.options());
  const float one = 1.f, zero = 0.f;
  const cublasStatus_t s = api().matmul(h, p->op, &one, w.data_ptr(), p->a, x.data_ptr(), p->b, &zero, y.data_ptr(),
                                        p->c, y.data_ptr(), p->c, &p->algo, nullptr, 0,
                                        at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", static_cast<int>(s));
  return y;
}

// kb21: fp8.py's row-wise FP8 projections (torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t()), bf16 out) through
// the same kind of zero-workspace, unsplit-K plan: no per-call memsets. Column-major as torch lays the call out:
// C [N, M] = op(A) op(B), A = wq as [K, N] (TRANSA = T) with the outer-vector scale ws [N], B = xq as [K, M] with xs [M].
// The caller (fp8.py) compares each new shape against torch._scaled_mm once before using it.
namespace {
struct Fp8Plan {
  bool ok = false;
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulAlgo_t algo{};
};
std::map<std::tuple<int, int64_t, int64_t, int64_t>, Fp8Plan> fp8_plans;

Fp8Plan make_fp8_plan(cublasLtHandle_t h, int64_t M, int64_t N, int64_t K, const void* sa, const void* sb) {
  const Api& f = api();
  Fp8Plan p;
  if (f.desc_create(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return p;
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  if (f.desc_set(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) != CUBLAS_STATUS_SUCCESS ||
      f.desc_set(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) != CUBLAS_STATUS_SUCCESS ||
      f.desc_set(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)) != CUBLAS_STATUS_SUCCESS ||
      f.desc_set(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)) != CUBLAS_STATUS_SUCCESS ||
      // the heuristic lists nothing for outer-vector scale modes without scale pointers: the first call's (every call
      // sets its own before cublasLtMatmul)
      f.desc_set(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)) != CUBLAS_STATUS_SUCCESS ||
      f.desc_set(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)) != CUBLAS_STATUS_SUCCESS ||
      f.layout_create(&p.a, CUDA_R_8F_E4M3, K, N, K) != CUBLAS_STATUS_SUCCESS ||
      f.layout_create(&p.b, CUDA_R_8F_E4M3, K, M, K) != CUBLAS_STATUS_SUCCESS ||
      f.layout_create(&p.c, CUDA_R_16BF, N, M, N) != CUBLAS_STATUS_SUCCESS)
    return p;
  cublasLtMatmulPreference_t pref{};
  uint64_t ws = 0;
  if (f.pref_create(&pref) != CUBLAS_STATUS_SUCCESS ||
      f.pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) != CUBLAS_STATUS_SUCCESS)
    return p;
  cublasLtMatmulHeuristicResult_t res[8];
  int n = 0;
  if (f.heuristic(h, p.op, p.a, p.b, p.c, p.c, pref, 8, res, &n) != CUBLAS_STATUS_SUCCESS) return p;
  for (int i = 0; i < n; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize != 0) continue;
    uint32_t splitk = 0;
    size_t written = 0;
    if (f.algo_get(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &written) !=
            CUBLAS_STATUS_SUCCESS || splitk > 1)
      continue;  // a K split would change the accumulation order
    p.algo = res[i].algo;
    p.ok = true;
    break;
  }
  return p;
}
}  // namespace

// y = (xq wq^T) row-scaled by xs [M] and ws [N] (fp32), bf16 out; an empty tensor when no zero-workspace plan exists.
torch::Tensor fp8_linear(torch::Tensor xq, torch::Tensor xs, torch::Tensor wq, torch::Tensor ws) {
  TORCH_CHECK(xq.is_cuda() && wq.is_cuda() && xq.scalar_type() == at::kFloat8_e4m3fn &&
                  wq.scalar_type() == at::kFloat8_e4m3fn && xq.dim() == 2 && wq.dim() == 2 && xq.size(1) == wq.size(1) &&
                  xq.is_contiguous() && wq.is_contiguous(),
              "qk_lt.fp8_linear: e4m3 xq [M, K] and wq [N, K], contiguous");
  TORCH_CHECK(xs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat && xs.is_contiguous() &&
                  ws.is_contiguous() && xs.numel() == xq.size(0) && ws.numel() == wq.size(0),
              "qk_lt.fp8_linear: fp32 scales xs [M] and ws [N], contiguous");
  const at::cuda::CUDAGuard guard(xq.device());
  if (!api().ok) return torch::empty({0}, xq.options().dtype(at::kBFloat16));
  const int64_t M = xq.size(0), K = xq.size(1), N = wq.size(0);
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  const void* sa = ws.data_ptr();
  const void* sb = xs.data_ptr();
  Fp8Plan* p;
  {
    std::lock_guard<std::mutex> lock(plans_mu);
    auto key = std::make_tuple(static_cast<int>(xq.get_device()), M, N, K);
    auto it = fp8_plans.find(key);
    if (it == fp8_plans.end()) it = fp8_plans.emplace(key, make_fp8_plan(h, M, N, K, sa, sb)).first;
    p = &it->second;
  }
  if (!p->ok) return torch::empty({0}, xq.options().dtype(at::kBFloat16));
  TORCH_CHECK(api().desc_set(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)) == CUBLAS_STATUS_SUCCESS &&
                  api().desc_set(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)) == CUBLAS_STATUS_SUCCESS,
              "qk_lt.fp8_linear: scale pointers");
  auto y = torch::empty({M, N}, xq.options().dtype(at::kBFloat16));
  const float one = 1.f, zero = 0.f;
  const cublasStatus_t s = api().matmul(h, p->op, &one, wq.data_ptr(), p->a, xq.data_ptr(), p->b, &zero, y.data_ptr(),
                                        p->c, y.data_ptr(), p->c, &p->algo, nullptr, 0,
                                        at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (fp8) failed: ", static_cast<int>(s));
  return y;
}

// Pinned configurations (ltsweep): cfg = {algo id, tile id, stages id, cta swizzling, custom option, cluster shape id
// (-1: leave unset), inner shape id}; split-K 1, no reduction. A pinned plan exists only if cublasLtMatmulAlgoCheck
// accepts the configuration for the call's descriptors with a zero-byte workspace; the caller compares its first
// output against the reference before using it (lt.py / fp8.py verdicts), as for the heuristic plans above.
namespace {
bool pin_algo(cublasLtHandle_t h, cublasLtMatmulDesc_t op, cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
              cublasLtMatrixLayout_t c, cudaDataType_t ab, const std::vector<int64_t>& cfg, cublasLtMatmulAlgo_t* out) {
  const Api& f = api();
  if (!f.pin_ok || cfg.size() != 7) return false;
  cublasLtMatmulAlgo_t al;
  if (f.algo_init(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, ab, ab, CUDA_R_16BF, CUDA_R_16BF, static_cast<int>(cfg[0]), &al) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  const uint32_t tile = static_cast<uint32_t>(cfg[1]), stages = static_cast<uint32_t>(cfg[2]), swz = static_cast<uint32_t>(cfg[3]),
                 custom = static_cast<uint32_t>(cfg[4]), red = CUBLASLT_REDUCTION_SCHEME_NONE;
  const int32_t splitk = 1;
  const uint16_t cluster = static_cast<uint16_t>(cfg[5]), inner = static_cast<uint16_t>(cfg[6]);
  if (f.algo_set(&al, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile)) != CUBLAS_STATUS_SUCCESS ||
      f.algo_set(&al, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages)) != CUBLAS_STATUS_SUCCESS ||
      f.algo_set(&al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk)) != CUBLAS_STATUS_SUCCESS ||
      f.algo_set(&al, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red)) != CUBLAS_STATUS_SUCCESS ||
      f.algo_set(&al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz)) != CUBLAS_STATUS_SUCCESS ||
      f.algo_set(&al, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom)) != CUBLAS_STATUS_SUCCESS)
    return false;
  if (cfg[5] >= 0 && f.algo_set(&al, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster)) != CUBLAS_STATUS_SUCCESS)
    return false;
  if (cfg[6] > 0 && f.algo_set(&al, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner)) != CUBLAS_STATUS_SUCCESS)
    return false;
  cublasLtMatmulHeuristicResult_t r{};
  if (f.algo_check(h, op, a, b, c, c, &al, &r) != CUBLAS_STATUS_SUCCESS || r.workspaceSize != 0) return false;
  *out = al;
  return true;
}
std::map<std::tuple<int, int64_t, int64_t, int64_t, std::vector<int64_t>>, Plan> pin_plans;
std::map<std::tuple<int, int64_t, int64_t, int64_t, std::vector<int64_t>>, Fp8Plan> pin_fp8_plans;
}  // namespace

// F.linear(x, w) through a pinned configuration; an empty tensor when it is not accepted for this shape.
torch::Tensor linear_cfg(torch::Tensor x, torch::Tensor w, std::vector<int64_t> cfg) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                  x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && x.is_contiguous() && w.is_contiguous(),
              "qk_lt.linear_cfg: bf16 x [M, K] and w [N, K], contiguous");
  const at::cuda::CUDAGuard guard(x.device());
  if (!api().ok) return torch::empty({0}, x.options());
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  const Plan* p;
  {
    std::lock_guard<std::mutex> lock(plans_mu);
    auto key = std::make_tuple(static_cast<int>(x.get_device()), M, N, K, cfg);
    auto it = pin_plans.find(key);
    if (it == pin_plans.end()) {
      Plan q = make_plan(h, M, N, K, {});  // the descriptors (its heuristic algo is replaced below)
      q.ok = q.op != nullptr && q.c != nullptr && pin_algo(h, q.op, q.a, q.b, q.c, CUDA_R_16BF, cfg, &q.algo);
      it = pin_plans.emplace(key, q).first;
    }
    p = &it->second;
  }
  if (!p->ok) return torch::empty({0}, x.options());
  auto y = torch::empty({M, N}, x.options());
  const float one = 1.f, zero = 0.f;
  const cublasStatus_t s = api().matmul(h, p->op, &one, w.data_ptr(), p->a, x.data_ptr(), p->b, &zero, y.data_ptr(),
                                        p->c, y.data_ptr(), p->c, &p->algo, nullptr, 0, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (pinned) failed: ", static_cast<int>(s));
  return y;
}

// fp8_linear through a pinned configuration; an empty tensor when it is not accepted for this shape.
torch::Tensor fp8_linear_cfg(torch::Tensor xq, torch::Tensor xs, torch::Tensor wq, torch::Tensor ws, std::vector<int64_t> cfg) {
  TORCH_CHECK(xq.is_cuda() && wq.is_cuda() && xq.scalar_type() == at::kFloat8_e4m3fn &&
                  wq.scalar_type() == at::kFloat8_e4m3fn && xq.dim() == 2 && wq.dim() == 2 && xq.size(1) == wq.size(1) &&
                  xq.is_contiguous() && wq.is_contiguous() && xs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat &&
                  xs.is_contiguous() && ws.is_contiguous() && xs.numel() == xq.size(0) && ws.numel() == wq.size(0),
              "qk_lt.fp8_linear_cfg: e4m3 xq [M, K], wq [N, K], fp32 xs [M], ws [N], contiguous");
  const at::cuda::CUDAGuard guard(xq.device());
  if (!api().ok) return torch::empty({0}, xq.options().dtype(at::kBFloat16));
  const int64_t M = xq.size(0), K = xq.size(1), N = wq.size(0);
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  const void* sa = ws.data_ptr();
  const void* sb = xs.data_ptr();
  Fp8Plan* p;
  {
    std::lock_guard<std::mutex> lock(plans_mu);
    auto key = std::make_tuple(static_cast<int>(xq.get_device()), M, N, K, cfg);
    auto it = pin_fp8_plans.find(key);
    if (it == pin_fp8_plans.end()) {
      Fp8Plan q = make_fp8_plan(h, M, N, K, sa, sb);
      q.ok = q.op != nullptr && q.c != nullptr && pin_algo(h, q.op, q.a, q.b, q.c, CUDA_R_8F_E4M3, cfg, &q.algo);
      it = pin_fp8_plans.emplace(key, q).first;
    }
    p = &it->second;
  }
  if (!p->ok) return torch::empty({0}, xq.options().dtype(at::kBFloat16));
  TORCH_CHECK(api().desc_set(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)) == CUBLAS_STATUS_SUCCESS &&
                  api().desc_set(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)) == CUBLAS_STATUS_SUCCESS,
              "qk_lt.fp8_linear_cfg: scale pointers");
  auto y = torch::empty({M, N}, xq.options().dtype(at::kBFloat16));
  const float one = 1.f, zero = 0.f;
  const cublasStatus_t s = api().matmul(h, p->op, &one, wq.data_ptr(), p->a, xq.data_ptr(), p->b, &zero, y.data_ptr(),
                                        p->c, y.data_ptr(), p->c, &p->algo, nullptr, 0, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (fp8, pinned) failed: ", static_cast<int>(s));
  return y;
}

// ---- fg11: the row-wise FP8 GEMM (fp8.py) -------------------------------------------------------------------------
// out[m, n] = bf16((acc * sb[n]) * sa[m]) for e4m3 a [M, K] (row scales sa) and b [N, K] (row scales sb), bit for bit
// torch._scaled_mm (cuBLASLt's recipe: acc = fp32 sum, over 128-k blocks in increasing k, of one wgmma e4m3 chain of
// four k32 steps that starts from zero; plain fp32 adds, acc starts at 0).
// Persistent, warp-specialized, 128 x 256 CTA tiles, 4 x 48 KB 128-k TMA stages (128B swizzle). Warpgroup 2: warp 8
// lane 0 issues the TMA loads (A + B rows 0-127 on one full barrier, B rows 128-255 on a second, so the first chunk's
// chain can start early), warp 9 stages each tile's row / column scales in smem (setmaxnreg 40). Warpgroups 0 / 1
// (setmaxnreg 232) own 64 rows x 256 columns each: acc0 / acc1 (columns 0-127 / 128-255) and one partial buffer pp;
// per k block: chain(pp, cols 0-127) -> wait -> acc0 += pp, chain(pp, cols 128-255) -> wait -> acc1 += pp.
// The tensor cores would idle during a tile's epilogue, so the next tile's first k block runs under it: its cols 0-127
// chain goes into acc0's registers once epilogue round 0 has consumed acc0, its cols 128-255 chain into acc1's after
// round 1; then acc = 0 + partial (the recipe's first promotion). The last tile takes a separate instantiation without
// those chains (the branch sits before any chain is issued: a branch around in-flight accumulators makes NVPTX copy
// them and ptxas serialize every wgmma).
// Epilogue: bf16((acc * sb[n]) * sa[m]) -> stmatrix into a 128B-swizzled staging buffer (two 128-column rounds) ->
// per-warp TMA stores (L2 evict_first). prefer(M, N): where this kernel beats cuBLASLt's 128 x 128 kernel.
namespace fg11 {
// wgmma m64n128k32 e4m3 x e4m3 -> f32, both operands from shared-memory descriptors
__device__ __forceinline__ void wg_mma_n128(float (&d)[64], uint64_t da, uint64_t db, int sd) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(sd));
}
// first k step of a chain: D = A * B (scale-d 0), the old accumulator is not an input
__device__ __forceinline__ void wg_mma_n128_z(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
      : "l"(da), "l"(db));
}
constexpr int BM = 128, BN = 256, BK = 128, STAGES = 4, SLOTS = 1, STGC = 128;
constexpr uint32_t ABYTES = BM * BK, BBYTES = BN * BK, SBYTES = ABYTES + BBYTES;
constexpr uint32_t STG_WG = 64 * STGC * 2;  // one warpgroup's bf16 staging: 2 boxes of [64 rows][128 B]
constexpr uint32_t STG_OFF = STAGES * SBYTES;
constexpr uint32_t SC_OFF = STG_OFF + 2 * STG_WG;
constexpr uint32_t SC_SLOT = (BM + BN) * 4;  // sa[BM], sb[BN]
constexpr uint32_t SMEM = SC_OFF + SLOTS * SC_SLOT + 1024;
static_assert(SMEM <= 232448 - 128, "smem");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t x) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(x) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile(
      "{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(smem_u32(b)),
      "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, uint64_t* b, int c0, int c1, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(b)), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_store2d(const CUtensorMap* m, uint32_t src, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], %4;" ::"l"(reinterpret_cast<uint64_t>(m)),
               "r"(src), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void stmatrix4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// K-major operand, 128-byte rows, 128B swizzle: SBO = 1024 (8 rows), layout type 1
__device__ __forceinline__ uint64_t desc(uint32_t a) {
  return static_cast<uint64_t>((a & 0x3FFFF) >> 4) | (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_regs(float (&d)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
// one 128-k block of one 128-column chunk: four k32 steps, the first from zero (its old registers are no input)
__device__ __forceinline__ void chain(float (&p)[64], uint32_t a_addr, uint32_t b_addr) {
  wg_fence();
  wg_mma_n128_z(p, desc(a_addr), desc(b_addr));
#pragma unroll
  for (int kk = 1; kk < 4; ++kk) wg_mma_n128(p, desc(a_addr + 32 * kk), desc(b_addr + 32 * kk), 1);
  wg_commit();
}


struct Args {
  const float* sa;  // [M]
  const float* sb;  // [N]
  __nv_bfloat16* out;
  int M, N, K, tiles_m, tiles_n;
  int group, pol_a, pol_b, pol_o;  // raster group (n tiles); L2 policies: 0 evict_normal, 1 evict_first, 2 evict_last
};

// tile t -> (m tile, n tile): groups of `group` n tiles sweep every m tile
__device__ __forceinline__ void tile_mn(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_m, g = t / per, r = t % per;
  const int gn = min(group, tiles_n - g * group);
  tm = r / gn;
  tn = g * group + r % gn;
}
__device__ __forceinline__ uint64_t l2_policy(int kind) {
  uint64_t p;
  if (kind == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else if (kind == 2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__global__ void __launch_bounds__(384, 1) fg11_kernel(const __grid_constant__ CUtensorMap ta, const __grid_constant__ CUtensorMap tb,
                                                      const __grid_constant__ CUtensorMap to, const Args a) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t base = (smem_u32(smem_raw) + 1023u) & ~1023u;
  uint8_t* sbase = smem_raw + (base - smem_u32(smem_raw));
  __shared__ __align__(8) uint64_t full[STAGES], full1[STAGES], empty[STAGES], sfull[SLOTS], sempty[SLOTS];
  const int tid = threadIdx.x, wg = tid / 128, warp = tid / 32, lane = tid % 32, wi = warp % 4;
  const int KB = a.K / BK, n_tiles = a.tiles_m * a.tiles_n, t0 = blockIdx.x, tstep = gridDim.x;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&full1[s], 1);
      mbar_init(&empty[s], 8);
    }
    for (int s = 0; s < SLOTS; ++s) {
      mbar_init(&sfull[s], 32);
      mbar_init(&sempty[s], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (wg == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (tid == 256) {
      const uint64_t pol_a = l2_policy(a.pol_a), pol_b = l2_policy(a.pol_b);
      int stage = 0;
      uint32_t phase = 0;
      for (int t = t0; t < n_tiles; t += tstep) {
        int tm, tn;
        tile_mn(t, a.tiles_m, a.tiles_n, a.group, tm, tn);
        for (int kb = 0; kb < KB; ++kb) {
          mbar_wait(&empty[stage], phase ^ 1);
          const uint32_t sdst = base + stage * SBYTES;
          mbar_expect(&full[stage], ABYTES + BBYTES / 2);
          tma2d(sdst, &ta, &full[stage], kb * BK, tm * BM, pol_a);
          tma2d(sdst + ABYTES, &tb, &full[stage], kb * BK, tn * BN, pol_b);
          mbar_expect(&full1[stage], BBYTES / 2);
          tma2d(sdst + ABYTES + BBYTES / 2, &tb, &full1[stage], kb * BK, tn * BN + BN / 2, pol_b);
          stage = stage + 1 == STAGES ? 0 : stage + 1;
          phase ^= stage == 0;
        }
      }
    } else if (warp == 9) {
      // each tile's scales into slot (it % SLOTS)
      int it = 0;
      for (int t = t0; t < n_tiles; t += tstep, ++it) {
        int tm, tn;
        tile_mn(t, a.tiles_m, a.tiles_n, a.group, tm, tn);
        const int slot = it % SLOTS;
        mbar_wait(&sempty[slot], ((it / SLOTS) & 1) ^ 1);
        float* ssa = reinterpret_cast<float*>(sbase + SC_OFF + slot * SC_SLOT);
        float* ssb = ssa + BM;
#pragma unroll
        for (int e = lane; e < BM; e += 32) {
          const int r = tm * BM + e;
          ssa[e] = r < a.M ? __ldg(a.sa + r) : 0.f;
        }
#pragma unroll
        for (int e = lane; e < BN; e += 32) ssb[e] = __ldg(a.sb + tn * BN + e);
        mbar_arrive(&sfull[slot]);
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    int stage = 0, it = 0;
    uint32_t phase = 0;
    float acc0[64], acc1[64], pp[64];
    const uint64_t pol_o = l2_policy(a.pol_o);
    const uint32_t stg = base + STG_OFF + wg * STG_WG;
    const int mi = lane >> 3;  // stmatrix: this lane addresses row (lane & 7) of matrix mi (rows + 8 (mi & 1), group + (mi >> 1))
    const uint32_t st_row = stg + (wi * 16 + (mi & 1) * 8 + (lane & 7)) * 128;
    const int rr = wg * 64 + wi * 16 + lane / 4;
    auto advance = [&]() {
      stage = stage + 1 == STAGES ? 0 : stage + 1;
      phase ^= stage == 0;
    };
    auto release_stage = [&]() {  // both warpgroups' 8 warps release a stage
      if (lane == 0) mbar_arrive(&empty[stage]);
    };
    auto wait_b1 = [&]() { mbar_wait(&full1[stage], phase); };  // the stage's B rows 128-255
    // the first tile's first k block (later tiles get theirs under the previous epilogue)
    {
      mbar_wait(&full[stage], phase);
      const uint32_t sa_ = base + stage * SBYTES + wg * 64 * BK, sb_ = base + stage * SBYTES + ABYTES;
      chain(pp, sa_, sb_);
      wait_b1();
      chain(acc0, sa_, sb_ + 128 * BK);
      wg_wait<0>();
      fence_regs(pp);
      fence_regs(acc0);
#pragma unroll
      for (int i = 0; i < 64; ++i) acc1[i] = 0.f + acc0[i];
#pragma unroll
      for (int i = 0; i < 64; ++i) acc0[i] = 0.f + pp[i];
      release_stage();
      advance();
    }
    int kb_first = 1;  // the first k block the mainloop runs (the earlier ones ran under the previous epilogue)
    for (int t = t0; t < n_tiles; t += tstep, ++it) {
      int tm, tn;
      tile_mn(t, a.tiles_m, a.tiles_n, a.group, tm, tn);
      for (int kb = kb_first; kb < KB; ++kb) {
        mbar_wait(&full[stage], phase);
        const uint32_t sa_ = base + stage * SBYTES + wg * 64 * BK, sb_ = base + stage * SBYTES + ABYTES;
        chain(pp, sa_, sb_);
        wg_wait<0>();
        fence_regs(pp);
#pragma unroll
        for (int i = 0; i < 64; ++i) acc0[i] += pp[i];
        wait_b1();
        chain(pp, sa_, sb_ + 128 * BK);
        wg_wait<0>();
        fence_regs(pp);
#pragma unroll
        for (int i = 0; i < 64; ++i) acc1[i] += pp[i];
        release_stage();
        advance();
      }
      // ---- epilogue; with a next tile, that tile's first k block runs on the tensor cores under it ----
      const bool has_next = t + tstep < n_tiles;
      const int slot = it % SLOTS;
      mbar_wait(&sfull[slot], (it / SLOTS) & 1);
      const float* ssa = reinterpret_cast<const float*>(sbase + SC_OFF + slot * SC_SLOT);
      const float* ssb = ssa + BM;
      const float sa0 = ssa[rr], sa1 = ssa[rr + 8];
      // bf16 of acc[4 j + e] (row 16 wi + lane / 4 + 8 (e >> 1), column 128 rd + 8 j + 2 (lane % 4) + (e & 1)) for column
      // groups j in [j0, j0 + 8) -> stmatrix into the staging rows
      auto emit = [&](float (&acc)[64], int rd, int j0) {
        {
          uint32_t pk[8][2];
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) {
            const int j = j0 + jj;
            const float2 sbv = *reinterpret_cast<const float2*>(ssb + rd * 128 + j * 8 + 2 * (lane % 4));
            pk[jj][0] = pack_bf16(__fmul_rn(__fmul_rn(acc[4 * j], sbv.x), sa0), __fmul_rn(__fmul_rn(acc[4 * j + 1], sbv.y), sa0));
            pk[jj][1] = pack_bf16(__fmul_rn(__fmul_rn(acc[4 * j + 2], sbv.x), sa1), __fmul_rn(__fmul_rn(acc[4 * j + 3], sbv.y), sa1));
          }
#pragma unroll
          for (int jj = 0; jj < 8; jj += 2) {
            const int g = j0 + jj + (mi >> 1);
            stmatrix4(st_row + (g >> 3) * (64 * 128) + (((g & 7) ^ (lane & 7)) << 4), pk[jj][0], pk[jj][1], pk[jj + 1][0], pk[jj + 1][1]);
          }
        }
      };
      auto staging_free = [&]() {
        if (lane == 0) bulk_wait_read0();  // this warp's previous store has left its staging rows
        __syncwarp();
      };
      auto store = [&](int rd) {
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          tma_store2d(&to, stg + wi * (16 * 128), tn * BN + rd * 128, tm * BM + wg * 64 + wi * 16, pol_o);
          tma_store2d(&to, stg + 64 * 128 + wi * (16 * 128), tn * BN + rd * 128 + 64, tm * BM + wg * 64 + wi * 16, pol_o);
          bulk_commit();
        }
      };
      // With a next tile: round 0 consumes acc0, then the next tile's columns 0-127 chain into acc0's registers; round 1 consumes
      // acc1, then its columns 128-255 chain into acc1's; after both rounds: wait, acc = 0 + partial (the first
      // promotion). Two instantiations, chosen before any chain is issued: no wgmma is in flight where they merge.
      auto epilogue = [&](auto ovl_c) {
        constexpr bool OVL = decltype(ovl_c)::value;
        uint32_t nsa = 0, nsb = 0;
        if constexpr (OVL) {
          mbar_wait(&full[stage], phase);
          nsa = base + stage * SBYTES + wg * 64 * BK;
          nsb = base + stage * SBYTES + ABYTES;
        }
        staging_free();
        emit(acc0, 0, 0);
        emit(acc0, 0, 8);
        if constexpr (OVL) chain(acc0, nsa, nsb);
        store(0);
        staging_free();
        emit(acc1, 1, 0);
        emit(acc1, 1, 8);
        if constexpr (OVL) {
          wait_b1();
          chain(acc1, nsa, nsb + 128 * BK);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&sempty[slot]);  // every lane has read its scales
        store(1);
        if constexpr (OVL) {
          wg_wait<0>();
          fence_regs(acc0);
          fence_regs(acc1);
#pragma unroll
          for (int i = 0; i < 64; ++i) acc0[i] = 0.f + acc0[i];
#pragma unroll
          for (int i = 0; i < 64; ++i) acc1[i] = 0.f + acc1[i];
          release_stage();
          advance();
        }
      };
      if (has_next) epilogue(std::true_type{});
      else epilogue(std::false_type{});
      kb_first = 1;
    }
    if (lane == 0) bulk_wait0();
  }
}

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode_fn() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q));
    TORCH_CHECK(p != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled is unavailable");
    fn = reinterpret_cast<EncodeFn>(p);
  }
  return fn;
}
static CUtensorMap map2d(void* ptr, CUtensorMapDataType dt, uint32_t esize, uint64_t inner, uint64_t outer, uint32_t box_inner,
                         uint32_t box_outer) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {inner, outer}, strides[1] = {inner * esize};
  const cuuint32_t box[2] = {box_inner, box_outer}, es[2] = {1, 1};
  TORCH_CHECK(encode_fn()(&m, dt, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed");
  return m;
}

// a [M, K] e4m3, sa [M] fp32, b [N, K] e4m3, sb [N] fp32 -> bf16 [M, N] == torch._scaled_mm(a, b.t(), sa[:, None], sb[None, :])
torch::Tensor mm(torch::Tensor a, torch::Tensor sa, torch::Tensor b, torch::Tensor sb) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.element_size() == 1 && a.is_contiguous() && b.dim() == 2 && b.element_size() == 1 &&
                  b.is_contiguous() && a.size(1) == b.size(1),
              "a [M, K], b [N, K] one-byte, K-contiguous");
  const int64_t M = a.size(0), N = b.size(0), K = a.size(1);
  TORCH_CHECK(K % BK == 0 && K >= BK && N % BN == 0 && M >= 1, "K % 128 == 0, N % 256 == 0");
  TORCH_CHECK(sa.scalar_type() == at::kFloat && sa.numel() == M && sa.is_contiguous() && sb.scalar_type() == at::kFloat &&
                  sb.numel() == N && sb.is_contiguous(),
              "row / column scales fp32");
  const at::cuda::CUDAGuard guard(a.device());
  auto out = torch::empty({M, N}, a.options().dtype(at::kBFloat16));
  // per-shape raster and L2 policies (measured, H100): wide N (8192) -> n-groups of 2 with A and B evict_last; N 2048 ->
  // groups of 8, A evict_first, B evict_last; outputs evict_first
  const bool wide = N >= 4096;
  const Args args{sa.data_ptr<float>(), sb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(M),
                  static_cast<int>(N), static_cast<int>(K), static_cast<int>((M + BM - 1) / BM), static_cast<int>(N / BN),
                  wide ? 2 : 8, wide ? 2 : 1, 2, 1};
  const CUtensorMap ma = map2d(a.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, M, BK, BM);
  const CUtensorMap mb = map2d(b.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N, BK, BN / 2);
  const CUtensorMap mo = map2d(out.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, N, M, 64, 16);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fg11_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    attr = true;
  }
  // persistent CTAs: the SM count trimmed to the fewest that keep the busiest CTA's tile count (same makespan, less L2
  // contention)
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t tiles = static_cast<int64_t>(args.tiles_m) * args.tiles_n, waves = (tiles + sms - 1) / sms;
  const int grid = static_cast<int>(std::min<int64_t>((tiles + waves - 1) / waves, tiles));
  fg11_kernel<<<grid, 384, SMEM, at::cuda::getCurrentCUDAStream()>>>(ma, mb, mo, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
// Whether fg11 should take this shape (else keep cuBLASLt): it needs more 128 x 256 tiles than SMs (one tile per CTA
// leaves its epilogue exposed) and a wave efficiency within 3 % of cuBLAS's 128 x 128 tiles (2500 x 2048: 160 tiles
// = 1.2 waves against 2.4 -> +25 %; measured H100, see the report).
bool prefer(int64_t M, int64_t N) {
  if (N % BN != 0 || M < 1) return false;
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t units = (M + BM - 1) / BM * (N / BN), half = 2 * units;
  if (units <= sms) return false;
  const double eff = static_cast<double>(units) / (((units + sms - 1) / sms) * sms);
  const double eff_cb = static_cast<double>(half) / (((half + sms - 1) / sms) * sms);
  return eff >= eff_cb - 0.03;
}
}  // namespace fg11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemm", &fg11::mm, "torch._scaled_mm(a, b.t(), sa[:, None], sb[None, :]) bf16, bit for bit (fg11 FP8 GEMM)",
        py::arg("a"), py::arg("sa"), py::arg("b"), py::arg("sb"));
  m.def("fp8_gemm_prefer", &fg11::prefer, "whether fp8_gemm beats cuBLASLt's 128 x 128 kernel at this shape", py::arg("M"),
        py::arg("N"));
  m.def("linear_cfg", &linear_cfg, "F.linear through a pinned cuBLASLt configuration (empty: not accepted)",
        py::arg("x"), py::arg("w"), py::arg("cfg"));
  m.def("fp8_linear_cfg", &fp8_linear_cfg, "fp8_linear through a pinned cuBLASLt configuration (empty: not accepted)",
        py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"), py::arg("cfg"));
  m.def("linear", &linear, "prefill x @ w.T through a zero-workspace, unsplit-K cuBLASLt plan (empty: none)",
        py::arg("x"), py::arg("w"), py::arg("tiles") = std::vector<int64_t>{});
  m.def("fp8_linear", &fp8_linear,
        "fp8.py's row-wise e4m3 projection through a zero-workspace, unsplit-K cuBLASLt plan (empty: none)",
        py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
}
