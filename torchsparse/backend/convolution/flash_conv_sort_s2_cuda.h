#include <torch/extension.h>

at::Tensor flash_conv_sort_s2_cuda(torch::Tensor inputs, 
                                torch::Tensor weights,
                                torch::Tensor reorder_map,
                                torch::Tensor reduced_mask,
                                torch::Tensor reorder_loc);