#include <torch/extension.h>

at::Tensor downsample_kmap_cuda(
    at::Tensor old_map,
    at::Tensor coords
);