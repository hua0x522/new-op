#include <torch/extension.h>
#include "mma_mask_cuda.h"

#define cdiv(x, y) (((x) + (y) - 1) / (y))

__global__ void reduced_mask_kernel(int* reduced_mask, int* bitmask, int n_points) {
    int idx = threadIdx.x + blockIdx.x * 32;
    if (idx < n_points / 16) {
        int mask = 0;
        for (int i = 0; i < 16; i++) {
            mask = mask | bitmask[idx * 16 + i];
        }
        reduced_mask[idx] = mask;
    }
}

__global__ void mma_mask_kernel(int* mma_mask, int* reduced_mask, int n_points, int kernel_volume) {
    int idx = threadIdx.x + blockIdx.x * 32;
    if (idx < n_points / 128) {
        int mask = 0;
        for (int i = 0; i < 8; i++) {
            if (reduced_mask[idx * 8 + i] & (1 << blockIdx.y)) {
                mask += 1 << i;
            }
        }
        mma_mask[idx * kernel_volume + blockIdx.y] = mask;
    }
}

torch::Tensor mma_mask_cuda(torch::Tensor bitmask, int kernel_volume) {
    int* bitmask_ptr = bitmask.data_ptr<int>();
    auto options = torch::TensorOptions().dtype(bitmask.dtype()).device(bitmask.device());
    int n_points = bitmask.size(1);
    at::Tensor reduced_mask = torch::empty({n_points / 16}, options);
    at::Tensor mma_mask = torch::empty({n_points / 128, kernel_volume}, options);
    int* reduced_mask_ptr = reduced_mask.data_ptr<int>();
    int* mma_mask_ptr = mma_mask.data_ptr<int>();

    dim3 num_blocks_0(cdiv(n_points / 16, 32));
    dim3 num_threads_0(32);
    reduced_mask_kernel<<<num_blocks_0, num_threads_0>>>(reduced_mask_ptr, bitmask_ptr, n_points);
    
    dim3 num_blocks_1(cdiv(n_points / 128, 32), kernel_volume);
    dim3 num_threads_1(32);
    mma_mask_kernel<<<num_blocks_1, num_threads_1>>>(mma_mask_ptr, reduced_mask_ptr, n_points, kernel_volume);
    
    return mma_mask;
}