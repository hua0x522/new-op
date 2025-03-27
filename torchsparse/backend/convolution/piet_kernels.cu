#include "piet_kernels.h"
#include <cuda_fp16.h>
#include <mma.h>
#include "ptx.h"
#include <cuda_pipeline.h>

#define cdiv(x, y) (((x) + (y) - 1) / (y))

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


template<int BLK_M, int BLK_K, int BLK_N, int WARP_M, int WARP_N>
struct ConvKernel 
{

static constexpr int WARP_K = BLK_K;
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;
static constexpr int NUM_WARP_M = BLK_M / WARP_M;
static constexpr int NUM_WARP_N = BLK_N / WARP_N;
static constexpr int WARP_SIZE = 32;
static constexpr int NUM_WARP = NUM_WARP_M * NUM_WARP_N;
static constexpr int NUM_THREAD = NUM_WARP * WARP_SIZE;
static constexpr int NUM_MMA_M = WARP_M / MMA_M;
static constexpr int NUM_MMA_K = WARP_K / MMA_K;
static constexpr int NUM_MMA_N = WARP_N / MMA_N;

__device__ static void load_shm_A(half* shm_A, half* inputs, int* reorder_map, int kernel_size, int c_in, int ko) {
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

__device__ static void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
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

__device__ static void load_reg_A(uint32_t* reg_A, half* shm_A, int ki, int m) {
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

__device__ static void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
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

__device__ static void store_C(uint32_t* reg_C, half* C, int* reorder_loc, int M, int N) {
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

__device__ static void pipe_load(half* shm_A, half* shm_B, half* inputs, half* weights, int* reorder_map, 
                          int kernel_size, int c_in, int N, int ko) {
    load_shm_A(shm_A, inputs, reorder_map, kernel_size, c_in, ko);
    load_shm_B(shm_B, weights, kernel_size * c_in, N, ko);
}

__device__ static void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int mma_flag) {
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

};

template<int BLK_M, int BLK_K, int BLK_N, int WARP_M, int WARP_N>
__global__ void subm_conv_kernel(half* inputs, half* weights, int* reorder_map, int* mma_mask, int* reorder_loc, half* outputs, 
    int n_points, int c_in, int c_out, int kernel_size) {
    int M = n_points;
    int N = c_out;
    int K = kernel_size * c_in;

    static constexpr int WARP_K = BLK_K;
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;
    static constexpr int NUM_MMA_M = WARP_M / MMA_M;
    static constexpr int NUM_MMA_K = WARP_K / MMA_K;
    static constexpr int NUM_MMA_N = WARP_N / MMA_N;

    __shared__ half shm_A[BLK_M * BLK_K];
    __shared__ half shm_B[BLK_K * BLK_N];

    uint32_t reg_A[NUM_MMA_K * NUM_MMA_M * 4];
    uint32_t reg_B[NUM_MMA_K * NUM_MMA_N * 2];
    uint32_t reg_C[NUM_MMA_M * NUM_MMA_N * 4] = {0};

    using kernel = ConvKernel<BLK_M, BLK_K, BLK_N, WARP_M, WARP_N>;

    for (int ko = 0; ko < K / BLK_K; ko++) {
        int mma_flag = mma_mask[blockIdx.x * kernel_size + (ko * BLK_K / c_in)];
        if (mma_flag) {
            kernel::pipe_load(shm_A, shm_B, inputs, weights, reorder_map, kernel_size, c_in, N, ko);
            __pipeline_commit();
            __pipeline_wait_prior(0);
            kernel::pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, mma_flag);
            __syncthreads();
        }
    }
    kernel::store_C(reg_C, outputs, reorder_loc, M, N);
}


torch::Tensor subm_conv_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map,
                            torch::Tensor mma_mask, torch::Tensor reorder_loc, int num_out_feats,
                            int BLK_M, int BLK_K, int BLK_N, int WARP_M, int WARP_N) {
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

    if (c_in % BLK_K != 0 || c_out % BLK_N != 0) {
        printf("tensor shape miss alignment!\n");
        return outputs;
    }

    if (BLK_M == 128 && BLK_K == 64 && BLK_N == 64 && WARP_M == 64 && WARP_N == 32) {
        subm_conv_kernel<128, 64, 64, 64, 32>
            <<<dim3(cdiv(n_points, BLK_M), cdiv(c_out, BLK_N)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M)>>>
            (inputs_ptr, weights_ptr, reorder_map_ptr, mma_mask_ptr, reorder_loc_ptr,
            outputs_ptr, n_points, c_in, c_out, kernel_size);
    }
    else if (BLK_M == 64 && BLK_K == 64 && BLK_N == 64 && WARP_M == 64 && WARP_N == 32) {
        subm_conv_kernel<64, 64, 64, 64, 32>
            <<<dim3(cdiv(n_points, BLK_M), cdiv(c_out, BLK_N)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M)>>>
            (inputs_ptr, weights_ptr, reorder_map_ptr, mma_mask_ptr, reorder_loc_ptr,
            outputs_ptr, n_points, c_in, c_out, kernel_size);
    }
    else if (BLK_M == 64 && BLK_K == 64 && BLK_N == 64 && WARP_M == 32 && WARP_N == 32) {
        subm_conv_kernel<64, 64, 64, 32, 32>
            <<<dim3(cdiv(n_points, BLK_M), cdiv(c_out, BLK_N)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M)>>>
            (inputs_ptr, weights_ptr, reorder_map_ptr, mma_mask_ptr, reorder_loc_ptr,
            outputs_ptr, n_points, c_in, c_out, kernel_size);
    }

    else if (BLK_M == 128 && BLK_K == 32 && BLK_N == 128 && WARP_M == 32 && WARP_N == 32) {
        subm_conv_kernel<128, 32, 128, 32, 32>
            <<<dim3(cdiv(n_points, BLK_M), cdiv(c_out, BLK_N)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M)>>>
            (inputs_ptr, weights_ptr, reorder_map_ptr, mma_mask_ptr, reorder_loc_ptr,
            outputs_ptr, n_points, c_in, c_out, kernel_size);
    }

    // subm_conv_kernel<BLK_M, BLK_K, BLK_N, WARP_M, WARP_N>
    //     <<<dim3(cdiv(n_points, 128), cdiv(c_out, 64)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M)>>>
    //     (inputs_ptr, weights_ptr, reorder_map_ptr, mma_mask_ptr, reorder_loc_ptr,
    //     outputs_ptr, n_points, c_in, c_out, kernel_size);
    
    return outputs;
}