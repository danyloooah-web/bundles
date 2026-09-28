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
#include <dlfcn.h>
#include <map>
#include <vector>
#include <mutex>
#include <tuple>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
  bool ok = false;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear, "prefill x @ w.T through a zero-workspace, unsplit-K cuBLASLt plan (empty: none)",
        py::arg("x"), py::arg("w"), py::arg("tiles") = std::vector<int64_t>{});
}
