#include <torch/extension.h>

at::Tensor subm_kmap_cuda(
    at::Tensor _coords,
    at::Tensor _kernel_sizes
);