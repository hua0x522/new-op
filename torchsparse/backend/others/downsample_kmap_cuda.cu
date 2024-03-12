#include <torch/extension.h>
#include "downsample_kmap_cuda.h"

#define BLK_SIZE 256

__global__ void downsample_kmap_kernel(int* old_map, int4* coords, int* out_in_map, int n_points) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 27; i++) {
        
    }
}

at::Tensor downsample_kmap_cuda(at::Tensor _old_map, at::Tensor _coords) {
    int n_points = _coords.size(0);
    int4 *coords = (int4*)_coords.data_ptr<int>();
    int* old_map = _old_map.data_ptr<int>();

    auto options = torch::TensorOptions()
                    .dtype(at::ScalarType::Int)
                    .device(_coords.device());
    
    at::Tensor _out_in_map = torch::full({n_points, 8}, -1, options);
    int* out_in_map = _out_in_map.data_ptr<int>();

    downsample_kmap_kernel<<<(n_points + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(old_map, coords, out_in_map, n_points);
}