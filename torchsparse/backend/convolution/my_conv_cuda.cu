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

__device__ void load_shm_A(half *shm_A, half *A, int* out_in_map, int num_in_channels, int kernel_volume, int ko)
{
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 8 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 3) << 3);
        int row_A = out_in_map[(blockIdx.x * 64 + row) * kernel_volume + (ko * 64 + col) / num_in_channels];
        int col_A = (ko * 64 + col) % num_in_channels;
        void* ptr = &shm_A[shm_row * 64 + shm_col];
        if (row_A == -1) {
            *(float4*)ptr = make_float4(0, 0, 0, 0);
        } else {
            *(float4*)ptr = *(float4*)&A[row_A * num_in_channels + col_A];
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half *shm_B, half *B, int N, int K, int ko)
{
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 8 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 3) << 3);
        *(float4*)&shm_B[shm_row * 64 + shm_col] = *(float4*)&B[(ko * 64 + row) * N + blockIdx.y * 64 + col];
    }
    __syncthreads();
}

__device__ void store_shm_C(float *shm_C, half *C, int M, int N) 
{
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 64; i++) {
        int row = i;
        int col = tid;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 3) << 3);
        C[(blockIdx.x * 64 + row) * N + blockIdx.y * 64 + col] = __float2half(shm_C[shm_row * 64 + shm_col]);
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int mi, int ki)
{
    int lane_id = threadIdx.x;
    int row = mi * 16 + lane_id % 16;
    int col = ki * 16 + lane_id / 16 * 8;
    col = col ^ ((row & 3) << 3);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + row * 64 + col);
    LDMATRIX_X4(reg_A[0], reg_A[1], reg_A[2], reg_A[3], shm_A_lane_addr);
    __syncthreads();
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki)
{
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < 4; ni++) {
        int row = ki * 16 + lane_id % 16;
        int col = threadIdx.y * 32 + ni * 8;
        col = col ^ ((row & 3) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col + ni * 8);
        LDMATRIX_X2_T(reg_B[ki * 8 + ni * 2], reg_B[ki * 8 + ni * 2 + 1], shm_B_lane_addr);
    }
}

__device__ void store_reg_C(uint32_t* reg_C, float* shm_C, int mi)
{
    int lane_id = threadIdx.x;

    for (int ni = 0; ni < 4; ni++) {
        int row = mi * 16 + lane_id / 4;
        int col = threadIdx.y * 32 + ni * 8 + (lane_id % 4) * 2;
        col = col ^ ((row & 3) << 3);
        shm_C[row * 64 + col] += *(float*)(&reg_C[ni * 4]);
        shm_C[row * 64 + col + 1] += *(float*)(&reg_C[ni * 4 + 1]);
        shm_C[(row + 8) * 64 + col] += *(float*)(&reg_C[ni * 4 + 2]);
        shm_C[(row + 8) * 64 + col + 1] += *(float*)(&reg_C[ni * 4 + 3]);
    }
}

__device__ void clear_shm_C(float* shm_C) {
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 64; i++) {
        shm_C[i * 64 + tid] = 0;
    }
}

__device__ void clear_reg_C(uint32_t* reg_C) {
    for (int i = 0; i < 16; i++) {
        reg_C[i] = 0;
    }
}

__global__ void my_conv_kernel(int M, int num_in_channels, int N, int kernel_volume, half *__restrict__ A, 
                    half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C) {
    __shared__ half shm_A[64 * 64];
    __shared__ half shm_B[64 * 64];
    __shared__ float shm_C[64 * 64];

    uint32_t reg_A[4];
    uint32_t reg_B[4 * 4 * 2];
    uint32_t reg_C[4 * 4];
    clear_shm_C(shm_C);
    
    int K = num_in_channels * kernel_volume;

    for (int k = 0; k < K / 64; k++) {
        load_shm_A(shm_A, A, out_in_map, num_in_channels, kernel_volume, k);
        load_shm_B(shm_B, B, N, K, k);
        __syncthreads();

        for (int ki = 0; ki < 4; ki++) {
            load_reg_B(reg_B, shm_B, ki);
        }

        for (int m = 0; m < 4; m++) {
            clear_reg_C(reg_C);
            for (int ki = 0; ki < 4; ki++) {
                load_reg_A(reg_A, shm_A, m, ki);
                __syncthreads();

                for (int n = 0; n < 4; n++) {
                    HMMA16816(reg_C[n * 4], reg_C[n * 4 + 1], reg_C[n * 4 + 2], reg_C[n * 4 + 3],
                              reg_A[0], reg_A[1], reg_A[2], reg_A[3],
                              reg_B[ki * 8 + n * 2], reg_B[ki * 8 + n * 2 + 1],
                              reg_C[n * 4], reg_C[n * 4 + 1], reg_C[n * 4 + 2], reg_C[n * 4 + 3]);
                }
            }
            __syncthreads();
            store_reg_C(reg_C, shm_C, m);
        }
    }
    store_shm_C(shm_C, C, M, N);
}

at::Tensor my_conv_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _out_in_map, 
                        int num_out_feats, int num_out_channels) 
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    int kernel_volume = _out_in_map.size(1);

    assert(kernel_volume >= 4);
    
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_out_feats, num_out_channels}, options);

    int* out_in_map = _out_in_map.data_ptr<int>();
    half* in_feats = reinterpret_cast<half *>(_in_feats.data_ptr<at::Half>());
    half* kernel = reinterpret_cast<half *>(_kernel.data_ptr<at::Half>());
    half* out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());

    // assert(num_out_channels % 64 == 0);
    // printf("%d %d\n", num_in_channels, num_out_channels);
    // assert(num_in_channels % 32 == 0);

    int M = _out_feats.size(0);
    int N = num_out_channels;

    // int smem_size = max(2 * (BLK_M + BLK_N) * BLK_K * 2, BLK_M * BLK_N * 4);
    // if (smem_size >= (48 << 10)) {
    //     cudaFuncSetAttribute(my_conv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    // }

    my_conv_kernel<<<dim3(cdiv(M, 64), cdiv(N, 64)), dim3(32, 2)>>>
                    (M, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    
    return _out_feats;
}