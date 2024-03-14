#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> downsample_kmap_cuda(
    at::Tensor old_map,
    at::Tensor coords
);