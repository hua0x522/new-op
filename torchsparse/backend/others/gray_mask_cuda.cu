#include <torch/extension.h>
#include "gray_mask_cuda.h"

#define cdiv(x, y) (((x) + (y) - 1) / (y))

__global__ void gray_mask_kernel(int* gray_mask, int* bitmask, int m, int n) {
    int idx = threadIdx.x + blockIdx.x * 128;
    int bit_val = bitmask[idx];
    int bit = (bit_val >> (n - 1)) & 1;
    int gray_val = bit << (n - 1);
    int last_bit = bit;

    for (int i = n - 2; i >= 0; i--) {
        bit = (bitmask[idx] >> i) & 1;
        gray_val += (bit ^ last_bit) << i;
        last_bit = bit;
    }

    gray_mask[idx] = gray_val;
}

torch::Tensor gray_mask_cuda(torch::Tensor bitmask, int kernel_volume) {
    int* bitmask_ptr = bitmask.data_ptr<int>();
    int m = bitmask.size(1);
    int n = kernel_volume;
    auto options = torch::TensorOptions().dtype(bitmask.dtype()).device(bitmask.device());
    at::Tensor gray_mask = torch::empty({1, m}, options);
    int* gray_mask_ptr = gray_mask.data_ptr<int>();
    
    dim3 num_blocks(cdiv(m, 128));
    dim3 num_threads(128);
    gray_mask_kernel<<<num_blocks, num_threads>>>(gray_mask_ptr, bitmask_ptr, m, n);
    
    return gray_mask;
}