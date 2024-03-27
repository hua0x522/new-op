#include <torch/extension.h>

at::Tensor flash_conv_cuda(torch::Tensor inputs, 
                           torch::Tensor weights,
                           torch::Tensor out_in_map);