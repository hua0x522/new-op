#include <torch/extension.h>
#include "mma_mask_cuda.h"

#define cdiv(x, y) (((x) + (y) - 1) / (y))

__global__ void mma_mask_kernel(int* mma_mask, int* bitmask, int m, int n) {
    int idx = threadIdx.x + blockIdx.x * 128;
    if (idx < m / 8) {
        int mask = 0;
        for (int i = 0; i < 8; i++) {
            if (bitmask[idx * 8 + i] & (1 << blockIdx.y)) {
                mask += 1 << i;
            }
        }
        mma_mask[idx * n + blockIdx.y] = mask;
    }
}

torch::Tensor mma_mask_cuda(torch::Tensor bitmask, int kernel_volume) {
    int* bitmask_ptr = bitmask.data_ptr<int>();
    int m = bitmask.size(1);
    int n = kernel_volume;
    auto options = torch::TensorOptions().dtype(bitmask.dtype()).device(bitmask.device());
    at::Tensor mma_mask = torch::empty({m / 8, n}, options);
    int* mma_mask_ptr = mma_mask.data_ptr<int>();
    
    dim3 num_blocks(cdiv(m / 8, 128), n);
    dim3 num_threads(128);
    mma_mask_kernel<<<num_blocks, num_threads>>>(mma_mask_ptr, bitmask_ptr, m, n);
    
    return mma_mask;
}