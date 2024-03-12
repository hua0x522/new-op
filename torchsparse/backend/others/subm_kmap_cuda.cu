#include <torch/extension.h>
#include "subm_kmap_cuda.h"
#include <cuda_fp16.h>

#define NDim 4
#define MAX_KVOL 27
#define BLOCK_SIZE 256
#define CHECK_CUDA2(func)                                              \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
        }                                                              \
    }
    
#define COORD(X, Y, Z) make_int4(0, X, Y, Z)

__device__ int4 add(int4 a, int4 b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ bool great_equal(int4 a, int4 b) {
    if (a.x > b.x) return true;
    else if (a.x < b.x) return false;
    if (a.y > b.y) return true;
    else if (a.y < b.y) return false;
    if (a.z > b.z) return true;
    else if (a.z < b.z) return false;
    if (a.w > b.w) return true;
    else if (a.w < b.w) return false;
    return true;
}

__device__ bool equal(int4 a, int4 b) {
    if (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w) {
        return true;
    } else {
        return false;
    }
}

__global__ void subm_kmap_kernel(
        int4* coords, 
        int* out_in_map, 
        int n_points) {
    int4 offsets[14] = {
        COORD(0, 0, 0), COORD(0, 0, 1), COORD(0, 1, -1), COORD(0, 1, 0), COORD(0, 1, 1), COORD(1, -1, -1), COORD(1, -1, 0), 
        COORD(1, -1, 1), COORD(1, 0, -1), COORD(1, 0, 0), COORD(1, 0, 1), COORD(1, 1, -1), COORD(1, 1, 0), COORD(1, 1, 1)
    };

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_points) {
        out_in_map[tid * 27 + 27 / 2] = tid;
        int4 out_coord = coords[tid];

        int l = tid, r = n_points - 1, mid;
        for (int i = 1; i <= 27 / 2; i++) {
            int4 in_coord = add(out_coord, offsets[i]); 
            l = tid;
            r = n_points - 1;
            if (tid == 242 && i == 9) {
                printf("%d %d %d\n", in_coord.y, in_coord.z, in_coord.w);
            }
            while (l < r) {
                mid = (l + r) >> 1;
                if (tid == 242 && i == 9) {
                    printf("%d %d %d %d %d %d\n", l, r, mid, coords[mid].y, coords[mid].z, coords[mid].w);
                }   
                if (great_equal(coords[mid], in_coord)) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            r = l;
            if (equal(coords[r], in_coord)) {
                int idx = (offsets[i].w + 1) * 9 + (offsets[i].z + 1) * 3 + (offsets[i].y + 1);
                out_in_map[tid * 27 + idx] = r;
                out_in_map[r * 27 + 26 - idx] = tid;
            } 
        }
    }
}

at::Tensor subm_kmap_cuda(at::Tensor _coords, at::Tensor _kernel_sizes) {
    int n_points = _coords.size(0);
    int4 *coords = (int4*)_coords.data_ptr<int>();

    auto options = torch::TensorOptions()
                    .dtype(at::ScalarType::Int)
                    .device(_coords.device());

    at::Tensor _out_in_map = torch::full({n_points, 27}, -1, options);
    int* out_in_map = _out_in_map.data_ptr<int>();
    
    subm_kmap_kernel<<<(n_points + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        coords,
        out_in_map, 
        n_points);
    CHECK_CUDA2(cudaGetLastError());

    return _out_in_map;
}