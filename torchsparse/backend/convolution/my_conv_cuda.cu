#include <torch/extension.h>
#include "my_conv_cuda.h"
#include <cuda_fp16.h>
#include <stdexcept>
#include <cstdio>
#include <cassert>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include "ptx.h"

#define cdiv(x, y) (((x) + (y) - 1) / (y))

__device__ void load_shm_A(half *shm_A, half *A, int* reorder_map, int valid, int M, int c_in, int ko)
{
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < valid; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col;
        // int shm_col = col ^ ((shm_row & 3) << 3);
        int row_A = reorder_map[row];
        int col_A = (ko * 64 + col) % c_in;
        void* ptr = &shm_A[shm_row * 72 + shm_col];
        if (row_A == -1) {
            *(float4*)ptr = make_float4(0, 0, 0, 0);
        } else {
            *(float4*)ptr = *(float4*)&A[row_A * c_in + col_A];
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half *shm_B, half *B, int N, int K, int ko)
{
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col;
        // int shm_col = col ^ ((shm_row & 3) << 3);
        *(float4*)&shm_B[shm_row * 72 + shm_col] = *(float4*)&B[(ko * 64 + row) * N + blockIdx.y * 64 + col];
    }
    __syncthreads();
}

__device__ void store_shm_C(float *shm_C, half *C, int M, int N) 
{
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 32; i++) {
        int row = i * 2 + tid / 64;
        int col = tid % 64;
        int shm_row = row;
        int shm_col = col;
        // int shm_col = col ^ ((shm_row & 3) << 3);
        C[(blockIdx.x * 64 + row) * N + blockIdx.y * 64 + col] = __float2half(shm_C[shm_row * 72 + shm_col]);
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int mi)
{
    for (int ki = 0; ki < 4; ki++) {
        int lane_id = threadIdx.x;
        int row = mi * 16 + lane_id % 16;
        int col = ki * 16 + lane_id / 16 * 8;
        // col = col ^ ((row & 3) << 3);
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + row * 72 + col);
        LDMATRIX_X4(reg_A[ki * 4], reg_A[ki * 4 + 1], reg_A[ki * 4 + 2], reg_A[ki * 4 + 3], shm_A_lane_addr);
    }
    __syncthreads();
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki)
{
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < 2; ni++) {
        int row = ki * 16 + lane_id % 16;
        int col = threadIdx.y * 16 + ni * 8;
        // col = col ^ ((row & 3) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 72 + col);
        LDMATRIX_X2_T(reg_B[ki * 4 + ni * 2], reg_B[ki * 4 + ni * 2 + 1], shm_B_lane_addr);
    }
}

__device__ void store_reg_C(uint32_t* reg_C, float* shm_C, int* loc_map, int mi)
{
    int lane_id = threadIdx.x;

    for (int ni = 0; ni < 2; ni++) {
        int row = mi * 16 + lane_id / 4;
        int col = threadIdx.y * 16 + ni * 8 + (lane_id % 4) * 2;
        // col = col ^ ((row & 3) << 3);
        if (loc_map[row] != -1) {
            shm_C[loc_map[row] * 72 + col] += *(float*)(&reg_C[ni * 4]);
            shm_C[loc_map[row] * 72 + col + 1] += *(float*)(&reg_C[ni * 4 + 1]);
            shm_C[loc_map[row + 8] * 72 + col] += *(float*)(&reg_C[ni * 4 + 2]);
            shm_C[loc_map[row + 8] * 72 + col + 1] += *(float*)(&reg_C[ni * 4 + 3]);
        }
    }
}

__device__ void clear_shm_C(float* shm_C) {
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 36; i++) {
        shm_C[i * 128 + tid] = 0;
    }
}

__device__ void clear_reg_C(uint32_t* reg_C) {
    for (int i = 0; i < 8; i++) {
        reg_C[i] = 0;
    }
}

__global__ void my_conv_kernel(int M, int c_in, int N, int kernel_volume, 
                               int* reorder_map, int* loc_map, int* valid_map,
                               half *__restrict__ A, half *__restrict__ B, half *__restrict__ C) {
    __shared__ half shm_A[64 * 72];
    __shared__ half shm_B[64 * 72];
    __shared__ float shm_C[64 * 72];

    uint32_t reg_A[4 * 4];
    uint32_t reg_B[4 * 2 * 2];
    uint32_t reg_C[4 * 4];
    clear_shm_C(shm_C);
    
    int K = c_in * kernel_volume;

    for (int k = 0; k < K / 64; k++) {
        int offset = k * 64 / c_in * M + blockIdx.x * 64;
        int valid = valid_map[k * 64 / c_in * M / 64 + blockIdx.x];
        if (valid > 0) {
            load_shm_A(shm_A, A, reorder_map + offset, valid, M, c_in, k);
            load_shm_B(shm_B, B, N, K, k);
            __syncthreads();
    
            for (int ki = 0; ki < 4; ki++) {
                load_reg_B(reg_B, shm_B, ki);
            }
    
            for (int m = 0; m < valid; m++) {
                clear_reg_C(reg_C);
                load_reg_A(reg_A, shm_A, m);
                for (int ki = 0; ki < 4; ki++) {
                    for (int n = 0; n < 2; n++) {
                        HMMA16816(reg_C[n * 4], reg_C[n * 4 + 1], reg_C[n * 4 + 2], reg_C[n * 4 + 3],
                                  reg_A[ki * 4], reg_A[ki * 4 + 1], reg_A[ki * 4 + 2], reg_A[ki * 4 + 3],
                                  reg_B[ki * 4 + n * 2], reg_B[ki * 4 + n * 2 + 1],
                                  reg_C[n * 4], reg_C[n * 4 + 1], reg_C[n * 4 + 2], reg_C[n * 4 + 3]);
                    }
                }
                store_reg_C(reg_C, shm_C, loc_map + offset, m);
            }
        }
    }
    store_shm_C(shm_C, C, M, N);
}

at::Tensor my_conv_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _reorder_map,
                        torch::Tensor _loc_map, torch::Tensor _valid_map, int n_points, int c_out) 
{
    int num_in_feats = _in_feats.size(0);
    int c_in = _in_feats.size(1);
    int kernel_volume = _reorder_map.size(0);

    assert(kernel_volume >= 4);
    
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({n_points, c_out}, options);

    int* reorder_map = _reorder_map.data_ptr<int>();
    int* loc_map = _loc_map.data_ptr<int>();
    int* valid_map = _valid_map.data_ptr<int>();
    half* in_feats = reinterpret_cast<half *>(_in_feats.data_ptr<at::Half>());
    half* kernel = reinterpret_cast<half *>(_kernel.data_ptr<at::Half>());
    half* out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());

    // assert(c_out % 64 == 0);
    // printf("%d %d\n", c_in, c_out);
    // assert(c_in % 32 == 0);

    int M = n_points;
    int N = c_out;

    my_conv_kernel<<<dim3(cdiv(M, 64), cdiv(N, 64)), dim3(32, 4)>>>
                    (M, c_in, c_out, kernel_volume, reorder_map, loc_map, valid_map, in_feats, kernel, out_feats);
    
    return _out_feats;
}