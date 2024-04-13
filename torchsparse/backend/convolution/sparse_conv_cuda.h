#include <torch/extension.h>

torch::Tensor sparse_conv_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map,
                              torch::Tensor reduced_mask, torch::Tensor mma_mask, torch::Tensor reorder_loc, 
                              int num_out_feats);