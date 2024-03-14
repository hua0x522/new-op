#include <torch/extension.h>
#include "subm_kmap_cuda.h"
#include <cuda_fp16.h>

#include <cuda/std/tuple>
#include <cub/device/device_radix_sort.cuh>

#include <bitset>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include "cub/block/radix_rank_sort_operations.cuh"

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <ctime>
#include <sys/time.h>

struct custom_t
{
  int x, y, z, w;
};

struct decomposer_t
{
  __host__ __device__ //
  ::cuda::std::tuple<int&, int&, int&, int&> operator()(custom_t &key) const
  {
    return {key.x, key.y, key.z, key.w};
  }
};

#define CDIV(X, Y) (((X) + (Y) - 1) / (Y))  
#define UP(X, Y) (CDIV(X, Y) * (Y))
#define INT4(X, Y, Z) make_int4(0, X, Y, Z)
#define COORD(tid) (make_int4(coords[tid].x, coords[tid].y, coords[tid].z, coords[tid].w))

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
        custom_t* coords,
        int* locs, 
        int* out_in_map, 
        int n) {
    int4 offsets[14] = {
        INT4(0, 0, 0), INT4(0, 0, 1), INT4(0, 1, -1), INT4(0, 1, 0), INT4(0, 1, 1), INT4(1, -1, -1), INT4(1, -1, 0), 
        INT4(1, -1, 1), INT4(1, 0, -1), INT4(1, 0, 0), INT4(1, 0, 1), INT4(1, 1, -1), INT4(1, 1, 0), INT4(1, 1, 1)
    };

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int loc = locs[tid];
        out_in_map[loc * 27 + 27 / 2] = loc;
        int4 out_coord = COORD(tid);

        int l = tid, r = n - 1, mid;
        for (int i = 1; i <= 27 / 2; i++) {
            int4 in_coord = add(out_coord, offsets[i]); 
            l = tid;
            r = n - 1;
            while (l < r) {
                mid = (l + r) >> 1;
                if (great_equal(COORD(mid), in_coord)) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            r = l;
            if (equal(COORD(r), in_coord)) {
                int loc_r = locs[r];
                int idx = (offsets[i].w + 1) * 9 + (offsets[i].z + 1) * 3 + (offsets[i].y + 1);
                out_in_map[loc * 27 + idx] = loc_r;
                out_in_map[loc_r * 27 + 26 - idx] = loc;
            } 
        }
    }
}

__global__ void data_copy(int4* coords, custom_t* loc_coords, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int4 coord = coords[tid];
        loc_coords[tid] = {coord.x, coord.y, coord.z, coord.w};
    }
}

__global__ void init_ones(int* locs, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * tid < UP(n, 4)) {
        *((int4*)locs + tid) = make_int4(4 * tid, 4 * tid + 1, 4 * tid + 2, 4 * tid + 3);
    }
}

at::Tensor subm_kmap_cuda(at::Tensor _coords) {
    struct timeval tv;
    double start, end;
    int n = _coords.size(0);
    custom_t* coords = (custom_t*)_coords.data_ptr<int>();
    int* locs;
    custom_t* sorted_coords;
    int* sorted_locs;

    // gettimeofday(&tv, nullptr);
    // start = tv.tv_sec + tv.tv_usec / 1.0e6;

    cudaMalloc(&locs, UP(n, 4) * sizeof(int));
    cudaMalloc(&sorted_coords, n * sizeof(custom_t));
    cudaMalloc(&sorted_locs, UP(n, 4) * sizeof(int));
    init_ones<<<CDIV(n, 256), 256>>>(locs, n);
    cudaDeviceSynchronize();
    // gettimeofday(&tv, nullptr);
    // end = tv.tv_sec + tv.tv_usec / 1.0e6;
    // printf("time1: %lf\n", end - start);
    
    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    // gettimeofday(&tv, nullptr);
    // start = tv.tv_sec + tv.tv_usec / 1.0e6;

    cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                 temp_storage_bytes,
                                 coords,
                                 sorted_coords,
                                 locs,
                                 sorted_locs,
                                 n,
                                 decomposer_t{});
    
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                 temp_storage_bytes,
                                 coords,
                                 sorted_coords,
                                 locs,
                                 sorted_locs,
                                 n,
                                 decomposer_t{});
    cudaDeviceSynchronize();

    // gettimeofday(&tv, nullptr);
    // end = tv.tv_sec + tv.tv_usec / 1.0e6;
    // printf("time2: %lf\n", end - start);

    auto options = torch::TensorOptions()
                    .dtype(at::ScalarType::Int)
                    .device(_coords.device());

    at::Tensor _out_in_map = torch::full({CDIV(n, 128) * 128, 27}, -1, options);
    int* out_in_map = _out_in_map.data_ptr<int>();

    // gettimeofday(&tv, nullptr);
    // start = tv.tv_sec + tv.tv_usec / 1.0e6;
    
    // subm_kmap_kernel<<<CDIV(n, 256), 256>>>(
    //     sorted_coords,
    //     sorted_locs,
    //     out_in_map, 
    //     n);
    // cudaDeviceSynchronize();

    // gettimeofday(&tv, nullptr);
    // end = tv.tv_sec + tv.tv_usec / 1.0e6;
    // printf("time3: %lf\n", end - start);

    return _out_in_map;
}