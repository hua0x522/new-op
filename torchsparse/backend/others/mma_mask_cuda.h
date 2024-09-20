#pragma once
#include <torch/torch.h>

torch::Tensor mma_mask_cuda(
    torch::Tensor bitmask,
    int kernel_volume
);