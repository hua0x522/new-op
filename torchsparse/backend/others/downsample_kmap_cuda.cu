#include <torch/extension.h>
#include "downsample_kmap_cuda.h"

#define CDIV(X, Y) (((X) + (Y) - 1) / (Y))  
#define UP(X, Y) (CDIV(X, Y) * (Y))

__device__ int4 add(int4 a, int4 b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ bool in_box(int4 a, int4 b) {
    return (a.x == b.x && a.y / 2 == b.y / 2 && a.z / 2 == b.z / 2 && a.w / 2 == b.w / 2);
}

__device__ bool lt(int4 a, int4 b) {
    if (a.x < b.x) return true;
    else if (a.x > b.x) return false;
    if (a.y < b.y) return true;
    else if (a.y > b.y) return false;
    if (a.z < b.z) return true;
    else if (a.z > b.z) return false;
    if (a.w < b.w) return true;
    else if (a.w > b.w) return false;
    return false;
}

__global__ void downsample_kmap_kernel(int* old_map, int4* old_coords, int* out_in_map, int4* new_coords, int* cnt, int n_points) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) {
        return;
    }
    int near_points_idx[27];
    int near_points[27];
    int near_points_cnt = 0;
    int4 coord = old_coords[tid];
    bool valid = true;
    for (int i = 0; i < 27; i++) {
        if (i == 13) continue;
        int point = old_map[tid * 27 + i];
        if (point == -1) continue;

        near_points[near_points_cnt] = point;
        near_points_idx[near_points_cnt] = i;
        near_points_cnt += 1;

        int4 offset = make_int4(0, i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1);
        int4 near_coord = add(coord, offset);
        if (in_box(near_coord, coord) && lt(near_coord, coord)) {
            valid = false;
            break;
        }
    }
    if (valid) {
        int number = atomicAdd(cnt, 1);
        new_coords[number] = make_int4(coord.x, coord.y / 2, coord.z / 2, coord.w / 2);

        for (int i = 0; i < near_points_cnt; i++) {
            int idx = near_points_idx[i];
            int4 offset = make_int4(0, idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 - 1);
            int4 near_coord = add(coord, offset);
            int new_idx = (near_coord.w % 2) + (near_coord.z % 2) * 2 + (near_coord.y % 2) * 4;
            out_in_map[number * 8 + new_idx] = old_map[tid * 27 + i];
        }
    }
}

std::vector<at::Tensor> downsample_kmap_cuda(at::Tensor _old_map, at::Tensor _coords) {
    int n_points = _coords.size(0);
    int4 *coords = (int4*)_coords.data_ptr<int>();
    int* old_map = _old_map.data_ptr<int>();

    auto options = torch::TensorOptions()
                    .dtype(at::ScalarType::Int)
                    .device(_coords.device());
    
    at::Tensor _out_in_map = torch::full({n_points, 8}, -1, options);
    at::Tensor _new_coords = torch::zeros({n_points, 4}, options);
    int* out_in_map = _out_in_map.data_ptr<int>();
    int4* new_coords = (int4*)_new_coords.data_ptr<int>();

    int* cnt;
    cudaMalloc(&cnt, sizeof(int));
    cudaMemset(cnt, 0, sizeof(int));

    downsample_kmap_kernel<<<CDIV(n_points, 256), 256>>>(old_map, coords, out_in_map, new_coords, cnt, n_points);
    cudaDeviceSynchronize();
    int value;
    cudaMemcpy(&value, cnt, sizeof(int), cudaMemcpyDeviceToHost);
    value = UP(value, 128);
    _out_in_map = torch::slice(_out_in_map, 0, 0, value, 1);
    _new_coords = torch::slice(_new_coords, 0, 0, value, 1);
    return {_out_in_map, _new_coords};
}