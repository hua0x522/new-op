#include <pybind11/pybind11.h>
// #include "convolution/all_kernels.h"
#include "convolution/piet_kernels.h"
// #include <torch/extension.h>
// #include <torch/serialize/tensor.h>

// #include "convolution/convolution_forward_implicit_gemm_sorted_cuda.h"
// #include "convolution/flash_conv_sort_cuda.h"
// #include "convolution/sparse_conv_cuda.h"
// #include "convolution/sparse_conv2_cuda.h"
// #include "others/mma_mask_cuda.h"
// #include "others/gray_mask_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("conv_forward_implicit_gemm_sorted_cuda", &conv_forward_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("num_out_feats"), py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  // m.def("flash_conv_sort_cuda", &flash_conv_sort_cuda);
  // m.def("sparse_conv_cuda", &sparse_conv_cuda);
  // m.def("sparse_conv2_cuda", &sparse_conv2_cuda);
  m.def("mma_mask_cuda", &mma_mask_cuda);
  m.def("gray_encode_cuda", &gray_encode_cuda);
  m.def("gray_decode_cuda", &gray_decode_cuda);
  // m.def("sparse_conv_base_cuda", &sparse_conv_base_cuda);
  m.def("subm_conv_cuda", &subm_conv_cuda);
}
