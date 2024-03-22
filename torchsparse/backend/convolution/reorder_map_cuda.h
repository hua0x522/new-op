#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> reorder_map_cuda(torch::Tensor _out_in_map);