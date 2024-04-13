#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_forward_implicit_gemm_sorted_cuda.h"
// #include "convolution/flash_conv_sort_cuda.h"
#include "convolution/sparse_conv_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_forward_implicit_gemm_sorted_cuda", &conv_forward_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("num_out_feats"), py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  // m.def("flash_conv_sort_cuda", &flash_conv_sort_cuda);
  m.def("sparse_conv_cuda", &sparse_conv_cuda);
}
