#include <torch/extension.h>
#include "reorder_map_cuda.h"
#include <cstdio>

#define cdiv(x, y) (((x) + (y) - 1) / (y))

__global__ void reorder_map_kernel(int* out_in_map, int* reorder_map, int* loc_map, int* valid_map, int n_points) {
    __shared__ int shm_loc_map[64];
    __shared__ int shm_reorder_map[64];
    __shared__ int cnt;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    shm_loc_map[tid] = -1;
    shm_reorder_map[tid] = -1;
    if (tid == 0) {
        cnt = 0;
    }
    __syncthreads();
    
    int row = bid * 64 / n_points;
    int col = bid * 64 % n_points + tid;
    int idx = -1;
    int loc = 0;
    if (row < n_points) {
        loc = out_in_map[row * n_points + col];
        if (loc != -1) {
            idx = atomicAdd(&cnt, 1);
        }
    }
    __syncthreads();

    if (idx != -1) {
        shm_loc_map[idx] = tid;
        shm_reorder_map[idx] = loc;
    } 
    __syncthreads();

    loc_map[row * n_points + col] = shm_loc_map[tid];
    reorder_map[row * n_points + col] = shm_reorder_map[tid];
    
    if (tid == 0) {
        valid_map[row * n_points / 64 + col / 64] = cdiv(cnt, 16);
    }
}

std::vector<at::Tensor> reorder_map_cuda(torch::Tensor _out_in_map) {
    /*
        assume out_in_map is [kernel_volume, n_points]
    */
    int* out_in_map = _out_in_map.data_ptr<int>();
    int n_points = _out_in_map.size(1);
    int kernel_volume = _out_in_map.size(0);

    auto options = torch::TensorOptions().dtype(_out_in_map.dtype()).device(_out_in_map.device());

    at::Tensor _reorder_map = torch::zeros({kernel_volume, n_points}, options);
    int* reorder_map = _reorder_map.data_ptr<int>();

    at::Tensor _loc_map = torch::zeros({kernel_volume, n_points}, options);
    int* loc_map = _loc_map.data_ptr<int>();

    at::Tensor _valid_map = torch::zeros({kernel_volume, n_points / 64}, options);
    int* valid_map = _valid_map.data_ptr<int>();

    reorder_map_kernel<<<cdiv(n_points * kernel_volume, 64), 64>>>(out_in_map, reorder_map, loc_map, valid_map, n_points);
    return {_reorder_map, _loc_map, _valid_map};
}