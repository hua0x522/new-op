#include <torch/extension.h>

at::Tensor my_conv_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _out_in_map, int num_out_feats, int num_out_channels);