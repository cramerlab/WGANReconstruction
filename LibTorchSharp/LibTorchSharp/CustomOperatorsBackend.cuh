#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/util/Exception.h>

#define CUDA_CHECK(EXPR)                                         \
  do {                                                               \
    cudaError_t __err = EXPR;                                        \
    if (__err != cudaSuccess) {                                      \
      auto error_unused C10_UNUSED = cudaGetLastError();             \
      std::cerr << "CUDA error: ", cudaGetErrorString(__err)); \
    }                                                                \
  } while (0)

#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

namespace at {
    namespace native {

        enum class GridSamplerInterpolation { Bilinear, Nearest, Bicubic };
        enum class GridSamplerPadding { Zeros, Border, Reflection };
        namespace MyOperator {
            std::tuple<Tensor, Tensor, Tensor, Tensor>
                atoms_to_grid_3d_backward_cuda(const Tensor& grad_output, const Tensor& intensities, const Tensor& positions, const Tensor& orientations, const Tensor& shift);

            Tensor atoms_to_grid_3d_cuda(const Tensor& intensities, const Tensor& positions, const Tensor& orientations, const Tensor& shift, int64_t x, int64_t y, int64_t z);

            // volume is (N, D, H, W) layout; positions are (N, W, 3), orientations are matrixes (N, 3,3) dimensional, projections will be (N, y,x) dimensional
            Tensor projectAtoms(const Tensor& intensities, const Tensor& positions, const Tensor& orientation, const Tensor& shift, int64_t x, int64_t y, int64_t z);

            std::tuple<Tensor, Tensor, Tensor, Tensor>
                projectAtoms_backward_cuda(const Tensor& grad_output, const Tensor& intensities, const Tensor& positions, const Tensor& orientation, const Tensor& shift, int64_t x, int64_t y, int64_t z);

            Tensor complex_grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                int interpolation_mode, int padding_mode,
                bool align_corners);

            Tensor complex_grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                int interpolation_mode, int padding_mode,
                bool align_corners);

            std::tuple<Tensor, Tensor>
                complex_grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int interpolation_mode,
                    int padding_mode, bool align_corners);

            std::tuple<Tensor, Tensor>
                complex_grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int interpolation_mode, int padding_mode,
                    bool align_corners);
        }
    }
}

