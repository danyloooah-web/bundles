// The K and V rows a prefill batch attends to, re-laid as 128-row pages for FA3 (sm_90a).
//
// The served KV cache has one-token pages, so FA3 fetches every key and value row with
// cp.async gathers; on 128-row pages it streams them with TMA, 12-16% faster on this model's
// prefill shapes (16 query heads over 2 KV heads, head dim 256, E4M3), with bit-identical
// output. This kernel copies each request's rows 0 .. len - 1, in token order, into pages
// base[r] .. base[r] + ceil(len / 128) - 1 of the page buffers, and zeroes the rows past len
// in a request's last page: FA3 multiplies them by zero probabilities, and E4M3 garbage can
// be NaN.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int PAGE = 128, ROW_VEC = 2 * 256 / 16;  // a slot row: 2 KV heads x 256 E4M3 bytes, as uint4

// Grid (max pages, requests); warp w copies rows w, w + 8, ... of the page, one uint4 per lane.
__global__ void __launch_bounds__(256) kv_pages_kernel(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
                                                       const int* __restrict__ page_table, int64_t pt_stride,
                                                       const int* __restrict__ lens, const int* __restrict__ base,
                                                       uint4* __restrict__ ko, uint4* __restrict__ vo) {
  const int r = blockIdx.y, page = blockIdx.x, len = lens[r];
  if (page * PAGE >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t row0 = static_cast<int64_t>(base[r] + page) * PAGE;
  const int* slots = page_table + r * pt_stride + page * PAGE;
#pragma unroll 4
  for (int i = warp; i < PAGE; i += 8) {
    const int64_t o = (row0 + i) * ROW_VEC + lane;
    uint4 k = make_uint4(0, 0, 0, 0), v = k;
    if (page * PAGE + i < len) {
      const int64_t s = static_cast<int64_t>(__ldg(slots + i)) * ROW_VEC + lane;
      k = __ldg(kc + s);
      v = __ldg(vc + s);
    }
    ko[o] = k;
    vo[o] = v;
  }
}

}  // namespace

// Page buffers [pages, 128, 2, 256] E4M3 for K and V. `base` holds each request's first page,
// `pages` the batch's total and `max_pages` the widest request's count.
std::vector<torch::Tensor> kv_pages(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                                    torch::Tensor lens, torch::Tensor base, int64_t pages, int64_t max_pages) {
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == at::kFloat8_e4m3fn && k_cache.is_contiguous() &&
                  k_cache.numel() % (2 * 256) == 0 && k_cache.size(-1) == 256 && k_cache.size(-2) == 2,
              "k_cache must be a contiguous E4M3 [slots, 2, 256] cache");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes() && v_cache.scalar_type() == k_cache.scalar_type() &&
                  v_cache.is_contiguous(), "v_cache must match k_cache");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.stride(1) == 1,
              "page_table must be int32 [requests, len] with contiguous rows");
  const int64_t bs = page_table.size(0);
  TORCH_CHECK(lens.scalar_type() == at::kInt && lens.is_contiguous() && lens.numel() == bs, "lens must be int32 [requests]");
  TORCH_CHECK(base.scalar_type() == at::kInt && base.is_contiguous() && base.numel() == bs, "base must be int32 [requests]");
  TORCH_CHECK(pages >= 1 && max_pages >= 1 && max_pages <= pages && max_pages * PAGE <= page_table.size(1) + PAGE - 1,
              "page counts do not fit the page table");
  const at::cuda::CUDAGuard guard(k_cache.device());
  auto opts = k_cache.options();
  auto ko = torch::empty({pages, PAGE, 2, 256}, opts), vo = torch::empty({pages, PAGE, 2, 256}, opts);
  kv_pages_kernel<<<dim3(static_cast<unsigned>(max_pages), static_cast<unsigned>(bs)), 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(k_cache.data_ptr()), reinterpret_cast<const uint4*>(v_cache.data_ptr()),
      page_table.data_ptr<int>(), page_table.stride(0), lens.data_ptr<int>(), base.data_ptr<int>(),
      reinterpret_cast<uint4*>(ko.data_ptr()), reinterpret_cast<uint4*>(vo.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ko, vo};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_pages", &kv_pages, "a prefill batch's K/V rows as 128-row pages");
  m.attr("PAGE") = PAGE;
}
