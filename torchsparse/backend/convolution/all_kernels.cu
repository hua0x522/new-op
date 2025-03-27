#include "all_kernels.h"
#include <cuda_fp16.h>
#include <mma.h>
#include "ptx.h"
#include <cuda_pipeline.h>

void LastError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

namespace flash_conv
{
__device__ void load_shm_A_k32n32(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // global layout: [128, 32]
    // shared layout: [64, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        int row_A = reorder_map[(blockIdx.x * 128 + row) * kernel_size + (ko * 32) / c_in];
        int col_A = (ko * 32 + col) % c_in;
        int shm_row = row / 2;
        int shm_col = col + (row & 1) * 32;
        shm_col = shm_col ^ ((shm_row & 3) << 3);
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

__device__ void load_shm_B_k32n32(half* shm_B, half* B, int K, int N, int ko) {
    // global layout: [32, 32]
    // shared layout: [16, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    int row = tid / 4;
    int col = tid % 4 * 8;
    int shm_row = row / 2;
    int shm_col = col + (row & 1) * 32;
    shm_col = shm_col ^ ((shm_row & 7) << 3);
    __pipeline_memcpy_async(
        &shm_B[shm_row * 64 + shm_col],
        &B[(ko * 32 + row) * N + blockIdx.y * 32 + col],
        16
    );
    __syncthreads();
}

__device__ void load_reg_A_k32n32(uint32_t* reg_A, half* shm_A, int ki) {
    for (int m = 0; m < 4; m++) {
        int lane_id = threadIdx.x;
        int row = threadIdx.z * 64 + m * 16 + lane_id % 16;
        int col = ki * 16 + lane_id / 16 * 8;
        int shm_row = row / 2;
        int shm_col = col + (row & 1) * 32;
        shm_col = shm_col ^ ((shm_row & 3) << 3);
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 64 + shm_col);
        LDMATRIX_X4(reg_A[m * 4], reg_A[m * 4 + 1], reg_A[m * 4 + 2], reg_A[m * 4 + 3], shm_A_lane_addr);
    }
}

__device__ void load_reg_B_k32n32(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    int row = ki * 16 + lane_id % 16;
    int col = threadIdx.y * 16 + lane_id / 16 * 8;
    int shm_row = row / 2;
    int shm_col = col + (row & 1) * 32;
    shm_col = shm_col ^ ((shm_row & 7) << 3);
    uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + shm_row * 64 + shm_col);
    LDMATRIX_X4_T(reg_B[0], reg_B[1], reg_B[2], reg_B[3], shm_B_lane_addr);
}

__device__ void store_C_k32n32(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 2; n++) {
            int shm_row = threadIdx.z * 64 + m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 16 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 32;
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

__device__ void pipe_load_k32n32(half* shm_A, half* shm_B, half* inputs, half* weights, int* reorder_map, 
                          int kernel_size, int c_in, int N, int ko, int loc) {
    shm_A += loc * 64 * 64;
    shm_B += loc * 16 * 64;
    load_shm_A_k32n32(shm_A, inputs, reorder_map, kernel_size, c_in, ko);
    load_shm_B_k32n32(shm_B, weights, kernel_size * c_in, N, ko);
}

__device__ void pipe_calc_k32n32(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int ko, int loc) {
    shm_A += loc * 64 * 64;
    shm_B += loc * 16 * 64;
    for (int ki = 0; ki < 2; ki++) {
        load_reg_A_k32n32(reg_A, shm_A, ki);
        load_reg_B_k32n32(reg_B, shm_B, ki);

        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 2; n++) {
                int idx = m * 2 + n;
                HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                    reg_A[m * 4], reg_A[m * 4 + 1], reg_A[m * 4 + 2], reg_A[m * 4 + 3],
                    reg_B[n * 2], reg_B[n * 2 + 1],
                    reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
            }
        }
    }
}

__global__ void flash_conv_sort_k32n32(half* inputs, half* weights, int* reorder_map, int* reduced_mask, 
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[2 * 64 * 64];
    __shared__ half shm_B[2 * 16 * 64];

    uint32_t reg_A[4 * 4];
    uint32_t reg_B[2 * 2];
    uint32_t reg_C[4 * 2 * 4] = {0};

    pipe_load_k32n32(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, 0, 0);
    __pipeline_commit();
    int idx0 = 0;
    int loc0 = 0;
    int loc1;

    for (int ko = 1; ko < K / 32; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 32 / c_in));
        if (flag) {
            loc1 = loc0 ^ 1;
            pipe_load_k32n32(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, ko, loc1);
            __pipeline_commit();
            __pipeline_wait_prior(1);
            pipe_calc_k32n32(shm_A, shm_B, reg_A, reg_B, reg_C, idx0, loc0);
            __syncthreads();
            idx0 = ko;
            loc0 = loc1;
        }
    }

    __pipeline_wait_prior(0);
    pipe_calc_k32n32(shm_A, shm_B, reg_A, reg_B, reg_C, idx0, loc0);

    store_C_k32n32(reg_C, outputs, reorder_loc, M, N);
}
}

namespace sparse_conv2
{
#define cdiv(x, y) (((x) + (y) - 1) / (y))

namespace m128k64n64
{
__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // layout: [128, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int row_A = reorder_map[(blockIdx.x * 128 + row) * kernel_size + (ko * 64) / c_in];
        int col_A = (ko * 64 + col) % c_in;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 7) << 3);
        // if (row_A == -1) {
        //     *(int4*)&shm_A[shm_row * 64 + shm_col] = make_int4(0, 0, 0, 0);
        // } 
        // else {
        //     __pipeline_memcpy_async(
        //         &shm_A[shm_row * 64 + shm_col],
        //         &inputs[row_A * c_in + col_A],
        //         16
        //     );
        // }
        uint32_t smem_ptr;
        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(&shm_A[shm_row * 64 + shm_col]));

        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.eq.s32 p, %0, -1;\n"
                     "cp.async.cg.shared.global [%1], [%2], %3, p;\n" 
                     "}\n"::"r"((int)row_A),
                     "r"(smem_ptr),
                     "l"(&inputs[row_A * c_in + col_A]),
                     "n"(16));
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [64, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
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
    int lane_id = threadIdx.x;
    int row = threadIdx.z * 64 + m * 16 + lane_id % 16;
    int col = ki * 16 + lane_id / 16 * 8;
    int shm_row = row;
    int shm_col = col ^ ((shm_row & 7) << 3);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 64 + shm_col);
    LDMATRIX_X4(reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3], shm_A_lane_addr);
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < 2; ni++) {
        int row = ki * 16 + lane_id % 16;
        int col = threadIdx.y * 32 + ni * 16 + lane_id / 16 * 8;
        col = col ^ ((row & 7) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col);
        LDMATRIX_X4_T(reg_B[ki * 8 + ni * 4], reg_B[ki * 8 + ni * 4 + 1], reg_B[ki * 8 + ni * 4 + 2], reg_B[ki * 8 + ni * 4 + 3], shm_B_lane_addr);
    }
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int shm_row = threadIdx.z * 64 + m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 32 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 64;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 3]);
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

    mma_flag = mma_flag >> (4 * threadIdx.z);

    for (int m = 0; m < 4; m++) {
        if (mma_flag & (1 << m)) {
        // if (1) {
            for (int ki = 0; ki < 4; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < 4; ki++) {
                for (int n = 0; n < 4; n++) {
                    int idx = m * 4 + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3],
                        reg_B[ki * 8 + n * 2], reg_B[ki * 8 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void sparse_conv2_kernel(half* inputs, half* weights, int* reorder_map, int* reduced_mask, int* mma_mask,
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[128 * 64];
    __shared__ half shm_B[64 * 64];

    uint32_t reg_A[4 * 4 * 4];
    uint32_t reg_B[4 * 4 * 2];
    uint32_t reg_C[4 * 4 * 4] = {0};

    for (int ko = 0; ko < K / 64; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 64 / c_in));
        if (flag) {
            int mma_flag = mma_mask[blockIdx.x * kernel_size + (ko * 64 / c_in)];
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

namespace m64k64n64
{
__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // layout: [128, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
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
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
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
    int lane_id = threadIdx.x;
    int row = threadIdx.z * 64 + m * 16 + lane_id % 16;
    int col = ki * 16 + lane_id / 16 * 8;
    int shm_row = row;
    int shm_col = col ^ ((shm_row & 7) << 3);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 64 + shm_col);
    LDMATRIX_X4(reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3], shm_A_lane_addr);
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < 2; ni++) {
        int row = ki * 16 + lane_id % 16;
        int col = threadIdx.y * 32 + ni * 16 + lane_id / 16 * 8;
        col = col ^ ((row & 7) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col);
        LDMATRIX_X4_T(reg_B[ki * 8 + ni * 4], reg_B[ki * 8 + ni * 4 + 1], reg_B[ki * 8 + ni * 4 + 2], reg_B[ki * 8 + ni * 4 + 3], shm_B_lane_addr);
    }
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int shm_row = threadIdx.z * 64 + m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 32 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 64;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 3]);
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

    mma_flag = mma_flag >> (4 * threadIdx.z);

    for (int m = 0; m < 4; m++) {
        if (mma_flag & (1 << m)) {
        // if (1) {
            for (int ki = 0; ki < 4; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < 4; ki++) {
                for (int n = 0; n < 4; n++) {
                    int idx = m * 4 + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3],
                        reg_B[ki * 8 + n * 2], reg_B[ki * 8 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void sparse_conv2_kernel(half* inputs, half* weights, int* reorder_map, int* reduced_mask, int* mma_mask,
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[128 * 64];
    __shared__ half shm_B[64 * 64];

    uint32_t reg_A[4 * 4 * 4];
    uint32_t reg_B[4 * 4 * 2];
    uint32_t reg_C[4 * 4 * 4] = {0};

    for (int ko = 0; ko < K / 64; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 64 / c_in));
        if (flag) {
            int mma_flag = mma_mask[blockIdx.x * kernel_size + (ko * 64 / c_in)];
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

torch::Tensor sparse_conv2_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map,
                              torch::Tensor reduced_mask, torch::Tensor mma_mask, torch::Tensor reorder_loc, int num_out_feats) {
    int c_in = weights.size(1);
    int c_out = weights.size(2);
    int n_points = num_out_feats;
    int kernel_size = reorder_map.size(1);

    auto options = torch::TensorOptions().dtype(inputs.dtype()).device(inputs.device());
    at::Tensor outputs = torch::empty({n_points, c_out}, options);

    int* reorder_map_ptr = reorder_map.data_ptr<int>();
    int* reduced_mask_ptr = reduced_mask.data_ptr<int>();
    int* reorder_loc_ptr = reorder_loc.data_ptr<int>();
    int* mma_mask_ptr = mma_mask.data_ptr<int>();
    half* inputs_ptr = reinterpret_cast<half*>(inputs.data_ptr<at::Half>());
    half* weights_ptr = reinterpret_cast<half*>(weights.data_ptr<at::Half>());
    half* outputs_ptr = reinterpret_cast<half*>(outputs.data_ptr<at::Half>());

    if (c_in % 64 == 0 && c_out % 64 == 0) {
        dim3 num_blocks(cdiv(n_points, 128), cdiv(c_out, 64));
        dim3 num_threads(32, 2, 2);
        sparse_conv2::m128k64n64::sparse_conv2_kernel<<<num_blocks, num_threads>>>
                    (inputs_ptr, weights_ptr, reorder_map_ptr, reduced_mask_ptr, mma_mask_ptr, reorder_loc_ptr,
                outputs_ptr, n_points, c_in, c_out, kernel_size);
    }
    else if (c_in % 32 == 0 && c_out % 32 == 0) {
        dim3 num_blocks(cdiv(n_points, 128), cdiv(c_out, 32));
        dim3 num_threads(32, 2, 2);
        flash_conv::flash_conv_sort_k32n32<<<num_blocks, num_threads>>>
                            (inputs_ptr, weights_ptr, reorder_map_ptr, reduced_mask_ptr, reorder_loc_ptr,
                            outputs_ptr, n_points, c_in, c_out, kernel_size);
    }
    return outputs;
}




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

__global__ void mma_mask_kernel(int* mma_mask, int* reduced_mask, int n_points, int kernel_volume, int group_size) {
    int idx = threadIdx.x + blockIdx.x * 32;
    if (idx < n_points / group_size) {
        int mask = 0;
        int num_mma = group_size / 16;
        for (int i = 0; i < num_mma; i++) {
            if (reduced_mask[idx * num_mma + i] & (1 << blockIdx.y)) {
                mask += 1 << i;
            }
        }
        mma_mask[idx * kernel_volume + blockIdx.y] = mask;
    }
}

torch::Tensor mma_mask_cuda(torch::Tensor bitmask, int kernel_volume, int group_size) {
    int* bitmask_ptr = bitmask.data_ptr<int>();
    auto options = torch::TensorOptions().dtype(bitmask.dtype()).device(bitmask.device());
    int n_points = bitmask.size(1);
    at::Tensor reduced_mask = torch::empty({n_points / 16}, options);
    at::Tensor mma_mask = torch::empty({n_points / group_size, kernel_volume}, options);
    int* reduced_mask_ptr = reduced_mask.data_ptr<int>();
    int* mma_mask_ptr = mma_mask.data_ptr<int>();

    dim3 num_blocks_0(cdiv(n_points / 16, 32));
    dim3 num_threads_0(32);
    reduced_mask_kernel<<<num_blocks_0, num_threads_0>>>(reduced_mask_ptr, bitmask_ptr, n_points);
    
    dim3 num_blocks_1(cdiv(n_points / group_size, 32), kernel_volume);
    dim3 num_threads_1(32);
    mma_mask_kernel<<<num_blocks_1, num_threads_1>>>(mma_mask_ptr, reduced_mask_ptr, n_points, kernel_volume, group_size);
    
    return mma_mask;
}


__global__ void gray_encode_kernel(int* gray_mask, int* bitmask, int m, int n) {
    int idx = threadIdx.x + blockIdx.x * 128;
    int bit_val = bitmask[idx];

    if (bit_val == -1) {
        bit_val = 0;
    }

    int bit = (bit_val >> (n - 1)) & 1;
    int gray_val = bit << (n - 1);
    int last_bit = bit;

    for (int i = n - 2; i >= 0; i--) {
        bit = ((bitmask[idx] >> i) & 1) ^ last_bit;
        gray_val = gray_val + (bit << i);
        last_bit = bit;
    }

    gray_mask[idx] = gray_val;
}

torch::Tensor gray_encode_cuda(torch::Tensor bitmask, int kernel_volume) {
    int* bitmask_ptr = bitmask.data_ptr<int>();
    int m = bitmask.size(1);
    int n = kernel_volume;
    auto options = torch::TensorOptions().dtype(bitmask.dtype()).device(bitmask.device());
    at::Tensor gray_mask = torch::empty({1, m}, options);
    int* gray_mask_ptr = gray_mask.data_ptr<int>();
    
    dim3 num_blocks(cdiv(m, 128));
    dim3 num_threads(128);
    gray_encode_kernel<<<num_blocks, num_threads>>>(gray_mask_ptr, bitmask_ptr, m, n);
    
    return gray_mask;
}

__global__ void gray_decode_kernel(int* gray_mask, int* bitmask, int m) {
    int idx = threadIdx.x + blockIdx.x * 128;
    int bit_val = bitmask[idx];

    if (bit_val == -1) {
        bit_val = 0;
    }

    gray_mask[idx] = (bit_val >> 1) ^ (bit_val);
}

torch::Tensor gray_decode_cuda(torch::Tensor bitmask, int kernel_volume) {
    int* bitmask_ptr = bitmask.data_ptr<int>();
    int m = bitmask.size(1);
    auto options = torch::TensorOptions().dtype(bitmask.dtype()).device(bitmask.device());
    at::Tensor gray_mask = torch::empty({1, m}, options);
    int* gray_mask_ptr = gray_mask.data_ptr<int>();
    
    dim3 num_blocks(cdiv(m, 128));
    dim3 num_threads(128);
    gray_decode_kernel<<<num_blocks, num_threads>>>(gray_mask_ptr, bitmask_ptr, m);
    
    return gray_mask;
}



namespace sparse_conv_base
{
#define cdiv(x, y) (((x) + (y) - 1) / (y))

namespace m128k64n64
{
__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    // layout: [128, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int row_A = reorder_map[(blockIdx.x * 128 + row) * kernel_size + (ko * 64) / c_in];
        int col_A = (ko * 64 + col) % c_in;
        int shm_row = row;
        int shm_col = col;
        // int shm_col = col ^ ((shm_row & 7) << 3);

        if (row_A == -1) {
            *(int4*)&shm_A[shm_row * 72 + shm_col] = make_int4(0, 0, 0, 0);
        } 
        else {
            __pipeline_memcpy_async(&shm_A[shm_row * 72 + shm_col], &inputs[row_A * c_in + col_A], 16);
            // *(int4*)&shm_A[shm_row * 72 + shm_col] = *(int4*)&inputs[row_A * c_in + col_A];
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [64, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_col = col;
        // int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(&shm_B[row * 72 + shm_col], &B[(ko * 64 + row) * N + blockIdx.y * 64 + col], 16);
        // *(int4*)&shm_B[row * 72 + shm_col] = *(int4*)&B[(ko * 64 + row) * N + blockIdx.y * 64 + col];
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki, int m) {
    int lane_id = threadIdx.x;
    int row = threadIdx.z * 64 + m * 16 + lane_id % 16;
    int col = ki * 16 + lane_id / 16 * 8;
    int shm_row = row;
    int shm_col = col;
    // int shm_col = col ^ ((shm_row & 7) << 3);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 72 + shm_col);
    LDMATRIX_X4(reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3], shm_A_lane_addr);
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < 2; ni++) {
        int row = ki * 16 + lane_id % 16;
        int col = threadIdx.y * 32 + ni * 16 + lane_id / 16 * 8;
        // col = col ^ ((row & 7) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 72 + col);
        LDMATRIX_X4_T(reg_B[ki * 8 + ni * 4], reg_B[ki * 8 + ni * 4 + 1], reg_B[ki * 8 + ni * 4 + 2], reg_B[ki * 8 + ni * 4 + 3], shm_B_lane_addr);
    }
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int shm_row = threadIdx.z * 64 + m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 32 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 64;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 3]);
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

    mma_flag = mma_flag >> (4 * threadIdx.z);

    for (int m = 0; m < 4; m++) {
        if (mma_flag & (1 << m)) {
        // if (1) {
            for (int ki = 0; ki < 4; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < 4; ki++) {
                for (int n = 0; n < 4; n++) {
                    int idx = m * 4 + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * 16 + m * 4], reg_A[ki * 16 + m * 4 + 1], reg_A[ki * 16 + m * 4 + 2], reg_A[ki * 16 + m * 4 + 3],
                        reg_B[ki * 8 + n * 2], reg_B[ki * 8 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void sparse_conv_base_kernel(half* inputs, half* weights, int* reorder_map, int* reduced_mask, int* mma_mask,
                                       int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[128 * 72];
    __shared__ half shm_B[64 * 72];

    uint32_t reg_A[4 * 4 * 4];
    uint32_t reg_B[4 * 4 * 2];
    uint32_t reg_C[4 * 4 * 4] = {0};

    for (int ko = 0; ko < K / 64; ko++) {
        bool flag = reduced_mask[blockIdx.x] & (1 << (ko * 64 / c_in));
        if (flag) {
            int mma_flag = mma_mask[blockIdx.x * kernel_size + (ko * 64 / c_in)];
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

torch::Tensor sparse_conv_base_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map,
                              torch::Tensor reduced_mask, torch::Tensor mma_mask, torch::Tensor reorder_loc, int num_out_feats) {
    int c_in = weights.size(1);
    int c_out = weights.size(2);
    int n_points = num_out_feats;
    int kernel_size = reorder_map.size(1);

    auto options = torch::TensorOptions().dtype(inputs.dtype()).device(inputs.device());
    at::Tensor outputs = torch::empty({n_points, c_out}, options);

    int* reorder_map_ptr = reorder_map.data_ptr<int>();
    int* reduced_mask_ptr = reduced_mask.data_ptr<int>();
    int* reorder_loc_ptr = reorder_loc.data_ptr<int>();
    int* mma_mask_ptr = mma_mask.data_ptr<int>();
    half* inputs_ptr = reinterpret_cast<half*>(inputs.data_ptr<at::Half>());
    half* weights_ptr = reinterpret_cast<half*>(weights.data_ptr<at::Half>());
    half* outputs_ptr = reinterpret_cast<half*>(outputs.data_ptr<at::Half>());

    if (c_in % 64 == 0 && c_out % 64 == 0) {
        dim3 num_blocks(cdiv(n_points, 128), cdiv(c_out, 64));
        dim3 num_threads(32, 2, 2);
        sparse_conv_base::m128k64n64::sparse_conv_base_kernel<<<num_blocks, num_threads>>>
                    (inputs_ptr, weights_ptr, reorder_map_ptr, reduced_mask_ptr, mma_mask_ptr, reorder_loc_ptr,
                outputs_ptr, n_points, c_in, c_out, kernel_size);
    }
    return outputs;
}


namespace subm_conv
{
#define cdiv(x, y) (((x) + (y) - 1) / (y))

namespace m128k64n64
{
const int BLK_M = 128;
const int BLK_N = 64;
const int BLK_K = 64;
const int WARP_M = 64;
const int WARP_N = 32;
const int WARP_K = BLK_K;
const int MMA_M = 16;
const int MMA_N = 8;
const int MMA_K = 16;
const int NUM_WARP_M = BLK_M / WARP_M;
const int NUM_WARP_N = BLK_N / WARP_N;
const int WARP_SIZE = 32;
const int NUM_WARP = NUM_WARP_M * NUM_WARP_N;
const int NUM_THREAD = NUM_WARP * WARP_SIZE;
const int NUM_MMA_M = WARP_M / MMA_M;
const int NUM_MMA_K = WARP_K / MMA_K;
const int NUM_MMA_N = WARP_N / MMA_N;

__device__ void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
    int tid = threadIdx.z * NUM_WARP_N * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
    const int load_elements = 128 / 16;
    const int thread_per_row = BLK_K / load_elements;
    const int rows_per_load = NUM_THREAD / thread_per_row;
    const int load_per_thread = BLK_M / rows_per_load;
    
    for (int i = 0; i < load_per_thread; i++) {
        int row = i * rows_per_load + tid / thread_per_row;
        int col = tid % thread_per_row * load_elements;
        int row_A = reorder_map[(blockIdx.x * BLK_M + row) * kernel_size + (ko * BLK_K) / c_in];
        int col_A = (ko * BLK_K + col) % c_in;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 7) << 3);
        if (row_A == -1) {
            *(int4*)&shm_A[shm_row * BLK_K + shm_col] = make_int4(0, 0, 0, 0);
        } 
        else {
            __pipeline_memcpy_async(
                &shm_A[shm_row * BLK_K + shm_col],
                &inputs[row_A * c_in + col_A],
                16
            );
        }
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    int tid = threadIdx.z * NUM_WARP_N * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
    const int load_elements = 128 / 16;
    const int thread_per_row = BLK_N / load_elements;
    const int rows_per_load = NUM_THREAD / thread_per_row;
    const int load_per_thread = BLK_K / rows_per_load;

    for (int i = 0; i < load_per_thread; i++) {
        int row = i * rows_per_load + tid / thread_per_row;
        int col = tid % thread_per_row * load_elements;
        int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_B[row * BLK_N + shm_col],
            &B[(ko * BLK_K + row) * N + blockIdx.y * BLK_N + col],
            16
        );
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki, int m) {
    int lane_id = threadIdx.x;
    int row = threadIdx.z * WARP_M + m * MMA_M + lane_id % 16;
    int col = ki * MMA_K + lane_id / 16 * 8;
    int shm_row = row;
    int shm_col = col ^ ((shm_row & 7) << 3);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * BLK_K + shm_col);
    LDMATRIX_X4(reg_A[ki * NUM_MMA_M * 4 + m * 4], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 1], 
                reg_A[ki * NUM_MMA_M * 4 + m * 4 + 2], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 3], 
                shm_A_lane_addr);
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < WARP_N / (MMA_N * 2); ni++) {
        int row = ki * MMA_K + lane_id % 16;
        int col = threadIdx.y * WARP_N + ni * (MMA_N * 2) + lane_id / 16 * 8;
        col = col ^ ((row & 7) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * BLK_N + col);
        LDMATRIX_X4_T(reg_B[ki * NUM_MMA_N * 2 + ni * 4], reg_B[ki * NUM_MMA_N * 2 + ni * 4 + 1], 
            reg_B[ki * NUM_MMA_N * 2 + ni * 4 + 2], reg_B[ki * NUM_MMA_N * 2 + ni * 4 + 3], 
            shm_B_lane_addr);
    }
}

__device__ void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < NUM_MMA_M; m++) {
        for (int n = 0; n < NUM_MMA_N; n++) {
            int shm_row = threadIdx.z * WARP_M + m * MMA_M + lane_id / 4;
            int shm_col = threadIdx.y * WARP_N + n * MMA_N + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * BLK_M;
            int col = shm_col + blockIdx.y * BLK_N;
            int row_8 = reorder_loc[row + 8];
            row = reorder_loc[row];
            if (row < M) {
                C[row * N + col] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4]);
                C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4 + 1]);
            }
            if (row_8 < M) {
                C[row_8 * N + col] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4 + 2]);
                C[row_8 * N + col + 1] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4 + 3]);
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
    for (int ki = 0; ki < NUM_MMA_K; ki++) {
        load_reg_B(reg_B, shm_B, ki);
    }

    const int num_bits = NUM_MMA_M;
    mma_flag = mma_flag >> (NUM_MMA_M * threadIdx.z);

    for (int m = 0; m < NUM_MMA_M; m++) {
        if (mma_flag & (1 << m)) {
            for (int ki = 0; ki < NUM_MMA_K; ki++) {
                load_reg_A(reg_A, shm_A, ki, m);
            }
            for (int ki = 0; ki < NUM_MMA_K; ki++) {
                for (int n = 0; n < NUM_MMA_N; n++) {
                    int idx = m * NUM_MMA_N + n;
                    HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                        reg_A[ki * NUM_MMA_M * 4 + m * 4], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 1], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 2], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 3],
                        reg_B[ki * NUM_MMA_N * 2 + n * 2], reg_B[ki * NUM_MMA_N * 2 + n * 2 + 1],
                        reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
                }
            }
        }
    }
}

__global__ void subm_conv_kernel(half* inputs, half* weights, int* reorder_map, int* mma_mask, int* reorder_loc, half* outputs, 
                                       int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;
    __shared__ half shm_A[BLK_M * BLK_K];
    __shared__ half shm_B[BLK_K * BLK_N];

    uint32_t reg_A[NUM_MMA_K * NUM_MMA_M * 4];
    uint32_t reg_B[NUM_MMA_K * NUM_MMA_N * 2];
    uint32_t reg_C[NUM_MMA_M * NUM_MMA_N * 4] = {0};

    for (int ko = 0; ko < K / BLK_K; ko++) {
        int mma_flag = mma_mask[blockIdx.x * kernel_size + (ko * BLK_K / c_in)];
        if (mma_flag) {
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

torch::Tensor subm_conv_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map,
                            torch::Tensor mma_mask, torch::Tensor reorder_loc, int num_out_feats) {
    int c_in = weights.size(1);
    int c_out = weights.size(2);
    int n_points = num_out_feats;
    int kernel_size = reorder_map.size(1);

    auto options = torch::TensorOptions().dtype(inputs.dtype()).device(inputs.device());
    at::Tensor outputs = torch::empty({n_points, c_out}, options);

    int* reorder_map_ptr = reorder_map.data_ptr<int>();
    int* reorder_loc_ptr = reorder_loc.data_ptr<int>();
    int* mma_mask_ptr = mma_mask.data_ptr<int>();
    half* inputs_ptr = reinterpret_cast<half*>(inputs.data_ptr<at::Half>());
    half* weights_ptr = reinterpret_cast<half*>(weights.data_ptr<at::Half>());
    half* outputs_ptr = reinterpret_cast<half*>(outputs.data_ptr<at::Half>());

    dim3 num_blocks(cdiv(n_points, 128), cdiv(c_out, 64));
    dim3 num_threads(32, 2, 2);
    subm_conv::m128k64n64::subm_conv_kernel<<<num_blocks, num_threads>>>
            (inputs_ptr, weights_ptr, reorder_map_ptr, mma_mask_ptr, reorder_loc_ptr,
            outputs_ptr, n_points, c_in, c_out, kernel_size);
    
    return outputs;
}