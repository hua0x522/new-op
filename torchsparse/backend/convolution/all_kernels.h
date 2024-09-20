#include <torch/extension.h>

torch::Tensor sparse_conv2_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map,
                              torch::Tensor reduced_mask, torch::Tensor mma_mask, torch::Tensor reorder_loc, 
                              int num_out_feats);

torch::Tensor mma_mask_cuda(
    torch::Tensor bitmask,
    int kernel_volume
);

torch::Tensor gray_mask_cuda(
    torch::Tensor bitmask,
    int kernel_volume
);