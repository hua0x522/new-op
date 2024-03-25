#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_forward_implicit_gemm_cuda.h"
#include "convolution/my_conv_cuda.h"
#include "convolution/reorder_map_cuda.h"
#include "convolution/convolution_forward_implicit_gemm_sorted_cuda.h"
// #include "others/downsample_kmap_cuda.h"
// #include "others/subm_kmap_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_forward_implicit_gemm_cuda", &conv_forward_implicit_gemm_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("num_out_feats"),py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("my_conv_cuda", &my_conv_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_reorder_map"), py::arg("_loc_map"), py::arg("_valid_map"), py::arg("n_points"),py::arg("c_out"));
  m.def("conv_forward_implicit_gemm_sorted_cuda", &conv_forward_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("num_out_feats"), py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  // m.def("downsample_kmap_cuda", &downsample_kmap_cuda, py::arg("old_map"), py::arg("coords"));
  // m.def("subm_kmap_cuda", &subm_kmap_cuda, py::arg("coords"));
  m.def("reorder_map_cuda", &reorder_map_cuda, py::arg("_out_in_map"));
}
