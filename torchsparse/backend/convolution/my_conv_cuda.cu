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

#define cdiv(x, y) (((x) + (y) - 1) / (y))

#define BLK_M 128
#define BLK_N 128
#define BLK_K 32
#define WARP_M 64
#define WARP_N 64
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define num_threads (32 * (BLK_M / WARP_M) * (BLK_N / WARP_N))
#define cdiv(x, y) (((x) + (y) - 1) / (y))

using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, nvcuda::wmma::row_major>;
using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, nvcuda::wmma::row_major>;
using Accum = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, float>;

__device__ void load_shm_A(half *shm_A, half *A, int* out_in_map, int num_in_channels, int kernel_volume, int ko)
{
    // load BLK_M * BLK_K
    // layout: [row_out, col_out, row_in, col_in] = [BLK_M / 16, BLK_K / 16, 16, 16]
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < (BLK_M * BLK_K) / (8 * num_threads); ++i) {
        int row = i * (8 * num_threads / BLK_K) + tid / (BLK_K / 8);
        int col = tid % (BLK_K / 8) * 8;
        void *ptr = (void *)(shm_A + row / 16 * ((BLK_K / 16) * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);

        int row_A = out_in_map[(blockIdx.y * BLK_M + row) * kernel_volume + (ko * BLK_K + col) / num_in_channels];
        int col_A = (ko * BLK_K + col) % num_in_channels;

        if (row_A == -1) {
            *(float4*)ptr = make_float4(0, 0, 0, 0);
        } 
        else {
            __pipeline_memcpy_async(
                ptr,
                &A[row_A * num_in_channels + col_A],
                16
            );
        }
        __syncthreads();
    }
}

__device__ void load_shm_B(half *shm_B, half *B, int N, int K, int ko)
{
    // load BLK_K * BLK_N
    // layout: [row_out, col_out, row_in, col_in] = [BLK_K / 16, BLK_N / 16, 16, 16]
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < (BLK_K * BLK_N) / (8 * num_threads); ++i) {
        int row = i * (8 * num_threads / BLK_N) + tid / (BLK_N / 8);
        int col = tid % (BLK_N / 8) * 8;
        void *ptr = (void *)(shm_B + row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);

        __pipeline_memcpy_async(
            ptr,
            &B[(ko * BLK_K + row) * N + (blockIdx.x * BLK_N + col)],
            16
        );
        __syncthreads();
    }
}

__device__ void store_shm_C(half *C, float *shm_C, int M, int N)
{
    // load BLK_M * BLK_N
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < (BLK_M * BLK_N) / num_threads; i++) {
        int row = i * num_threads / BLK_N + tid / BLK_N;
        int col = i * num_threads % BLK_N + tid % BLK_N;
        C[(blockIdx.y * BLK_M + row) * N + blockIdx.x * BLK_N + col] = 
            __float2half(shm_C[row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16]);
    }
}

__device__ void load_frag_A(FragA *frag, half *shm_A, int ki)
{
    // load WARP_M * WARP_K
    for (int i = 0; i < WARP_M / MMA_M; ++i)
    {
        int row = threadIdx.z * WARP_M + i * MMA_M;
        int col = ki * MMA_K;
        nvcuda::wmma::load_matrix_sync(frag[i], shm_A + row / 16 * ((BLK_K / 16) * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void load_frag_B(FragB *frag, half *shm_B, int ki)
{
    // load WARP_K * WARP_N
    for (int i = 0; i < WARP_N / MMA_N; ++i)
    {
        int row = ki * MMA_K; 
        int col = threadIdx.y * WARP_N + i * MMA_N;
        nvcuda::wmma::load_matrix_sync(frag[i], shm_B + row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16), 16);
    }
    __syncthreads();
}

__device__ void store_Accum(float *ptr, Accum *frag)
{
    // store 64x64
    for (int i = 0; i < WARP_M / MMA_M; ++i)
    {
        for (int j = 0; j < WARP_N / MMA_N; ++j)
        {
            int row = threadIdx.z * WARP_M + i * MMA_M;
            int col = threadIdx.y * WARP_N + j * MMA_N;
            // layout: [WARP_M / MMA_M, WARP_N / MMA_N, MMA_M, MMA_N]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16), 
                                            frag[i * (WARP_N / MMA_N) + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* A, half* B, int* out_in_map, 
                          int N, int num_in_channels, int kernel_volume, int ko) {
    shm_A += (ko % 4) * BLK_M * BLK_K;
    shm_B += (ko % 4) * BLK_N * BLK_K;
    load_shm_A(shm_A, A, out_in_map, num_in_channels, kernel_volume, ko);
    load_shm_B(shm_B, B, N, num_in_channels * kernel_volume, ko);
}

__device__ void pipe_calc(FragA* frag_A, FragB* frag_B, Accum* accum, half* shm_A, half* shm_B, int ko) {
    shm_A += (ko % 4) * BLK_M * BLK_K;
    shm_B += (ko % 4) * BLK_N * BLK_K;
    for (int ki = 0; ki < BLK_K / MMA_K; ki += 1)
    {
        // 64x64x16 mma for each warp
        load_frag_A(frag_A, shm_A, ki);
        load_frag_B(frag_B, shm_B, ki);
        for (int mii = 0; mii < WARP_M / MMA_M; mii += 1)
        {
            for (int nii = 0; nii < WARP_N / MMA_N; nii += 1)
            {
                // 16x16x16 for each wmma
                nvcuda::wmma::mma_sync(accum[mii * (WARP_N / MMA_N) + nii], frag_A[mii], frag_B[nii], accum[mii * (WARP_N / MMA_N) + nii]);
            }
        }
    }
}

__global__ void my_conv_kernel(int M, int num_in_channels, int N, int kernel_volume, half *__restrict__ A, 
                    half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C) {
    extern __shared__ uint8_t shared_storage[];
    half *shm_A = reinterpret_cast<half *>(shared_storage);
    half* shm_B = shm_A + 4 * BLK_M * BLK_K;
    float *shm_C = reinterpret_cast<float *>(shared_storage);

    FragA frag_A[WARP_M / MMA_M];
    FragB frag_B[WARP_N / MMA_N];
    Accum accum[WARP_M / MMA_M * WARP_N / MMA_N];

    for (int mii = 0; mii < WARP_M / MMA_M; mii += 1)
    {
        for (int nii = 0; nii < WARP_N / MMA_N; nii += 1)
        {
            nvcuda::wmma::fill_fragment(accum[mii * (WARP_N / MMA_N) + nii], 0.0);
        }
    }
    
    int K = num_in_channels * kernel_volume;

    pipe_load(shm_A, shm_B, A, B, out_in_map, N, num_in_channels, kernel_volume, 0);
    __pipeline_commit();

    pipe_load(shm_A, shm_B, A, B, out_in_map, N, num_in_channels, kernel_volume, 1);
    __pipeline_commit();

    pipe_load(shm_A, shm_B, A, B, out_in_map, N, num_in_channels, kernel_volume, 2);
    __pipeline_commit();

    for (int ko = 3; ko < K / BLK_K; ko++) {
        pipe_load(shm_A, shm_B, A, B, out_in_map, N, num_in_channels, kernel_volume, ko);
        __pipeline_commit();
        __pipeline_wait_prior(3);
        pipe_calc(frag_A, frag_B, accum, shm_A, shm_B, ko - 3);
        __syncthreads();
    }

    __pipeline_wait_prior(2);
    pipe_calc(frag_A, frag_B, accum, shm_A, shm_B, K / BLK_K - 3);
    __syncthreads();
    __pipeline_wait_prior(1);
    pipe_calc(frag_A, frag_B, accum, shm_A, shm_B, K / BLK_K - 2);
    __syncthreads();
    __pipeline_wait_prior(0);
    pipe_calc(frag_A, frag_B, accum, shm_A, shm_B, K / BLK_K - 1);
    __syncthreads();

    store_Accum(shm_C, accum);
    __syncthreads();
    store_shm_C(C, shm_C, M, N);
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

    int smem_size = max(4 * (BLK_M + BLK_N) * BLK_K * 2, BLK_M * BLK_N * 4);
    if (smem_size >= (48 << 10)) {
        cudaFuncSetAttribute(my_conv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    my_conv_kernel<<<dim3(cdiv(N, BLK_N), cdiv(M, BLK_M)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M), smem_size, nullptr>>>
                    (M, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    
    return _out_feats;
}