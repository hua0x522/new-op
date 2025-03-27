#include <torch/extension.h>

torch::Tensor mma_mask_cuda(
    torch::Tensor bitmask,
    int kernel_volume,
    int group_size
);

torch::Tensor gray_encode_cuda(
    torch::Tensor bitmask,
    int kernel_volume
);

torch::Tensor gray_decode_cuda(
    torch::Tensor bitmask,
    int kernel_volume
);

torch::Tensor subm_conv_cuda(torch::Tensor inputs, torch::Tensor weights, torch::Tensor reorder_map, 
                            torch::Tensor mma_mask, torch::Tensor reorder_loc, int num_out_feats,
                            int BLK_M, int BLK_K, int BLK_N, int WARP_M, int WARP_N);