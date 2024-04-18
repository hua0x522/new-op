#include "flash_conv_sort_cuda.h"
#include <cuda_fp16.h>
#include <mma.h>
#include "ptx.h"
#include <cuda_pipeline.h>

namespace sparse_conv
{

#define cdiv(x, y) (((x) + (y) - 1) / (y))

namespace m64k32n64
{
__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // layout: [64, 32]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 2; i++) {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        int row_A = reorder_map[(blockIdx.x * 64 + row) * kernel_size + (ko * 32) / c_in];
        int col_A = (ko * 32 + col) % c_in;
        col = col ^ ((row & 6) << 2);
        if (row_A == -1) {
            *(int4*)&shm_A[row * 32 + col] = make_int4(0, 0, 0, 0);
        } 
        else {
            __pipeline_memcpy_async(
                &shm_A[row * 32 + col],
                &inputs[row_A * c_in + col_A],
                16
            );
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [32, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 2; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_B[row * 64 + shm_col],
            &B[(ko * 32 + row) * N + blockIdx.y * 64 + col],
            16
        );
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki, int m) {
    int lane_id = threadIdx.x;
    int row = m * 16 + lane_id % 16;
    int col = ki * 16 + lane_id / 16 * 8;
    col = col ^ ((row & 6) << 2);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + row * 32 + col);
    LDMATRIX_X4(reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3], shm_A_lane_addr);
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    int row = ki * 16 + lane_id % 16;
    int col = threadIdx.y * 16  + lane_id / 16 * 8;
    col = col ^ ((row & 7) << 3);
    uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col);
    LDMATRIX_X4_T(reg_B[ki * 4], reg_B[ki * 4 + 1], reg_B[ki * 4 + 2], reg_B[ki * 4 + 3], shm_B_lane_addr);
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 2; n++) {
            int shm_row = m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 16 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 64;
            int col = shm_col + blockIdx.y * 64;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * 8 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 3]);
            }
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* inputs, half* weights, int* reorder_map, 
                          int kernel_size, int c_in, int N, int ko, int loc) {
    shm_A += loc * 64 * 32;
    shm_B += loc * 32 * 64;
    load_shm_A(shm_A, inputs, reorder_map, kernel_size, c_in, ko);
    load_shm_B(shm_B, weights, kernel_size * c_in, N, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int mma_flag, int ko, int loc) {
    shm_A += loc * 64 * 32;
    shm_B += loc * 32 * 64;
    
    for (int ki = 0; ki < 2; ki++) {
        load_reg_B(reg_B, shm_B, ki);
    }

    for (int m = 0; m < 4; m++) {
        if (mma_flag & (1 << m)) {
            for (int ki = 0; ki < 2; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < 2; ki++) {
                for (int n = 0; n < 2; n++) {
                    int idx = m * 2 + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3],
                        reg_B[ki * 4 + n * 2], reg_B[ki * 4 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void sparse_conv_kernel(half* inputs, half* weights, int* reorder_map, int* reduced_mask, int* mma_mask,
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[2 * 64 * 32];
    __shared__ half shm_B[2 * 32 * 64];

    uint32_t reg_A[2 * 4 * 4];
    uint32_t reg_B[2 * 2 * 2];
    uint32_t reg_C[8 * 2 * 4] = {0};

    int ko;
    int idx0, idx1;
    int loc0, loc1;

    for (ko = 0; ko < K / 32; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 32 / c_in));
        if (flag) {
            idx1 = ko;
            loc1 = 0;
            pipe_load(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, idx1, loc1);
            __pipeline_commit();
            break;
        }
    }

    if (ko < K / 32) {
        for (ko += 1; ko < K / 32; ko++) {
            bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 32 / c_in));
            if (flag) {
                idx0 = idx1;
                loc0 = loc1;
                idx1 = ko;
                loc1 = loc0 ^ 1;
                int mma_flag = 0;
                for (int i = 0; i < 4; i++) {
                    if (mma_mask[blockIdx.x * 4 + i] & (1 << (idx0 * 32 / c_in))) {
                        mma_flag = mma_flag + (1 << i); 
                    }
                }
                pipe_load(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, idx1, loc1);
                __pipeline_commit();
                __pipeline_wait_prior(1);
                pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, mma_flag, idx0, loc0);
                __syncthreads();
            }
        }

        __pipeline_wait_prior(0);
        int mma_flag = 0;
        for (int i = 0; i < 4; i++) {
            if (mma_mask[blockIdx.x * 4 + i] & (1 << (idx1 * 32 / c_in))) {
                mma_flag = mma_flag + (1 << i); 
            }
        }
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, mma_flag, idx1, loc1);
        __syncthreads();
    }

    store_C(reg_C, outputs, reorder_loc, M, N);
}
}


namespace m128k32n64
{
__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // global layout: [128, 32]
    // shared layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        int row_A = reorder_map[(blockIdx.x * 128 + row) * kernel_size + (ko * 32) / c_in];
        int col_A = (ko * 32 + col) % c_in;
        // int shm_row = row / 2;
        // int shm_col = col + (row & 1) * 32;
        // shm_col = shm_col ^ ((shm_row & 7) << 3);
        col = col ^ ((row & 6) << 2);
        if (row_A == -1) {
            // *(int4*)&shm_A[shm_row * 64 + shm_col] = make_int4(0, 0, 0, 0);
            *(int4*)&shm_A[row * 32 + col] = make_int4(0, 0, 0, 0);
        } 
        else {
            __pipeline_memcpy_async(
                // &shm_A[shm_row * 64 + shm_col],
                &shm_A[row * 32 + col],
                &inputs[row_A * c_in + col_A],
                16
            );
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [32, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 2; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_B[row * 64 + shm_col],
            &B[(ko * 32 + row) * N + blockIdx.y * 64 + col],
            16
        );
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki, int m) {
    // for (int m = 0; m < 8; m++) {
        int lane_id = threadIdx.x;
        int row = m * 16 + lane_id % 16;
        int col = ki * 16 + lane_id / 16 * 8;
        // int shm_row = row / 2;
        // int shm_col = col + (row & 1) * 32;
        // shm_col = shm_col ^ ((shm_row & 7) << 3);
        col = col ^ ((row & 6) << 2);
        // uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 64 + shm_col);
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + row * 32 + col);
        LDMATRIX_X4(reg_A[ki * 32 + m * 4], reg_A[ki * 32 + m * 4 + 1], reg_A[ki * 32 + m * 4 + 2], reg_A[ki * 32 + m * 4 + 3], shm_A_lane_addr);
    // }
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    int row = ki * 16 + lane_id % 16;
    int col = threadIdx.y * 16  + lane_id / 16 * 8;
    col = col ^ ((row & 7) << 3);
    uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col);
    LDMATRIX_X4_T(reg_B[ki * 4], reg_B[ki * 4 + 1], reg_B[ki * 4 + 2], reg_B[ki * 4 + 3], shm_B_lane_addr);
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 8; m++) {
        for (int n = 0; n < 2; n++) {
            int shm_row = m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 16 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 64;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * 8 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 3]);
            }
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* inputs, half* weights, int* reorder_map, 
                          int kernel_size, int c_in, int N, int ko, int loc) {
    shm_A += loc * 128 * 32;
    shm_B += loc * 32 * 64;
    load_shm_A(shm_A, inputs, reorder_map, kernel_size, c_in, ko);
    load_shm_B(shm_B, weights, kernel_size * c_in, N, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int mma_flag, int ko, int loc) {
    shm_A += loc * 128 * 32;
    shm_B += loc * 32 * 64;
    
    for (int ki = 0; ki < 2; ki++) {
        load_reg_B(reg_B, shm_B, ki);
    }

    for (int m = 0; m < 8; m++) {
        if (mma_flag & (1 << m)) {
            for (int ki = 0; ki < 2; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < 2; ki++) {
                for (int n = 0; n < 2; n++) {
                    int idx = m * 2 + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * 32 + m * 4], reg_A[ki * 32 + m * 4 + 1], reg_A[ki * 32 + m * 4 + 2], reg_A[ki * 32 + m * 4 + 3],
                        reg_B[ki * 4 + n * 2], reg_B[ki * 4 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void sparse_conv_kernel(half* inputs, half* weights, int* reorder_map, int* reduced_mask, int* mma_mask,
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[2 * 128 * 32];
    __shared__ half shm_B[2 * 32 * 64];

    uint32_t reg_A[2 * 8 * 4];
    uint32_t reg_B[2 * 2 * 2];
    uint32_t reg_C[8 * 2 * 4] = {0};

    int ko;
    int idx0, idx1;
    int loc0, loc1;

    for (ko = 0; ko < K / 32; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 32 / c_in));
        if (flag) {
            idx1 = ko;
            loc1 = 0;
            pipe_load(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, idx1, loc1);
            __pipeline_commit();
            break;
        }
    }

    if (ko < K / 32) {
        for (ko += 1; ko < K / 32; ko++) {
            bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 32 / c_in));
            if (flag) {
                idx0 = idx1;
                loc0 = loc1;
                idx1 = ko;
                loc1 = loc0 ^ 1;
                int mma_flag = 0;
                for (int i = 0; i < 8; i++) {
                    if (mma_mask[blockIdx.x * 8 + i] & (1 << (idx0 * 32 / c_in))) {
                        mma_flag = mma_flag + (1 << i); 
                    }
                }
                pipe_load(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, idx1, loc1);
                __pipeline_commit();
                __pipeline_wait_prior(1);
                pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, mma_flag, idx0, loc0);
                __syncthreads();
            }
        }

        __pipeline_wait_prior(0);
        int mma_flag = 0;
        for (int i = 0; i < 8; i++) {
            if (mma_mask[blockIdx.x * 8 + i] & (1 << (idx1 * 32 / c_in))) {
                mma_flag = mma_flag + (1 << i); 
            }
        }
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, mma_flag, idx1, loc1);
        __syncthreads();
    }

    store_C(reg_C, outputs, reorder_loc, M, N);
}
}

namespace m128k64n64
{
__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // layout: [128, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int row_A = reorder_map[(blockIdx.x * 128 + row) * kernel_size + (ko * 64) / c_in];
        int col_A = (ko * 64 + col) % c_in;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 7) << 3);
        if (row_A == -1) {
            *(int4*)&shm_A[shm_row * 64 + shm_col] = make_int4(0, 0, 0, 0);
        } 
        else {
            __pipeline_memcpy_async(
                &shm_A[shm_row * 64 + shm_col],
                &inputs[row_A * c_in + col_A],
                16
            );
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_B[row * 64 + shm_col],
            &B[(ko * 64 + row) * N + blockIdx.y * 64 + col],
            16
        );
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki, int m) {
    // for (int m = 0; m < 8; m++) {
        int lane_id = threadIdx.x;
        int row = m * 16 + lane_id % 16;
        int col = ki * 16 + lane_id / 16 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 7) << 3);
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 64 + shm_col);
        LDMATRIX_X4(reg_A[ki * 32 + m * 4], reg_A[ki * 32 + m * 4 + 1], reg_A[ki * 32 + m * 4 + 2], reg_A[ki * 32 + m * 4 + 3], shm_A_lane_addr);
    // }
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    int row = ki * 16 + lane_id % 16;
    int col = threadIdx.y * 16  + lane_id / 16 * 8;
    col = col ^ ((row & 7) << 3);
    uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col);
    LDMATRIX_X4_T(reg_B[ki * 4], reg_B[ki * 4 + 1], reg_B[ki * 4 + 2], reg_B[ki * 4 + 3], shm_B_lane_addr);
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 8; m++) {
        for (int n = 0; n < 2; n++) {
            int shm_row = m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 16 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 64;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * 8 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * 8 + n * 4 + 3]);
            }
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* inputs, half* weights, int* reorder_map, 
                          int kernel_size, int c_in, int N, int ko) {
    load_shm_A(shm_A, inputs, reorder_map, kernel_size, c_in, ko);
    load_shm_B(shm_B, weights, kernel_size * c_in, N, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int mma_flag) {
    for (int ki = 0; ki < 4; ki++) {
        load_reg_B(reg_B, shm_B, ki);
    }

    for (int m = 0; m < 8; m++) {
        if (mma_flag & (1 << m)) {
            for (int ki = 0; ki < 4l; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < 4; ki++) {
                for (int n = 0; n < 2; n++) {
                    int idx = m * 2 + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * 32 + m * 4], reg_A[ki * 32 + m * 4 + 1], reg_A[ki * 32 + m * 4 + 2], reg_A[ki * 32 + m * 4 + 3],
                        reg_B[ki * 4 + n * 2], reg_B[ki * 4 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void sparse_conv_kernel(half* inputs, half* weights, int* reorder_map, int* reduced_mask, int* mma_mask,
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[128 * 64];
    __shared__ half shm_B[64 * 64];

    uint32_t reg_A[4 * 8 * 4];
    uint32_t reg_B[4 * 2 * 2];
    uint32_t reg_C[8 * 2 * 4] = {0};

    for (int ko = 0; ko < K / 64; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 64 / c_in));
        if (flag) {
            int mma_flag = 0;
            for (int i = 0; i < 8; i++) {
                if (mma_mask[blockIdx.x * 8 + i] & (1 << (ko * 64 / c_in))) {
                    mma_flag = mma_flag + (1 << i); 
                }
            }
            pipe_load(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, ko);
            __pipeline_commit();
            __pipeline_wait_prior(0);
            pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, mma_flag);
            __syncthreads();
        }
    }
    store_C(reg_C, outputs, reorder_loc, M, N);
}
}

}

torch::Tensor sparse_conv_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map, 
                               torch::Tensor reduced_mask, torch::Tensor mma_mask, torch::Tensor reorder_loc, 
                               int num_out_feats) {
    int c_in = weights.size(1);
    int c_out = weights.size(2);
    int n_points = num_out_feats;
    int kernel_size = reorder_map.size(1);

    auto options = torch::TensorOptions().dtype(inputs.dtype()).device(inputs.device());
    at::Tensor outputs = torch::empty({n_points, c_out}, options);

    int* reorder_map_ptr = reorder_map.data_ptr<int>();
    int* reduced_mask_ptr = reduced_mask.data_ptr<int>();
    int* mma_mask_ptr = mma_mask.data_ptr<int>();
    int* reorder_loc_ptr = reorder_loc.data_ptr<int>();
    half* inputs_ptr = reinterpret_cast<half*>(inputs.data_ptr<at::Half>());
    half* weights_ptr = reinterpret_cast<half*>(weights.data_ptr<at::Half>());
    half* outputs_ptr = reinterpret_cast<half*>(outputs.data_ptr<at::Half>());

    dim3 num_blocks(cdiv(n_points, 128), cdiv(c_out, 64));
    dim3 num_threads(32, 4);
    sparse_conv::m128k64n64::sparse_conv_kernel<<<num_blocks, num_threads>>>
                (inputs_ptr, weights_ptr, reorder_map_ptr, reduced_mask_ptr, mma_mask_ptr, reorder_loc_ptr,
                outputs_ptr, n_points, c_in, c_out, kernel_size);

    // dim3 num_blocks(cdiv(n_points, 64), cdiv(c_out, 64));
    // dim3 num_threads(32, 4);
    // sparse_conv::m64k32n64::sparse_conv_kernel<<<num_blocks, num_threads>>>
    //             (inputs_ptr, weights_ptr, reorder_map_ptr, reduced_mask_ptr, mma_mask_ptr, reorder_loc_ptr,
    //             outputs_ptr, n_points, c_in, c_out, kernel_size);
    return outputs;
}