#include <torch/extension.h>

at::Tensor my_conv_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _reorder_map,
                        torch::Tensor _loc_map, torch::Tensor _valid_map, int n_points, int c_out);