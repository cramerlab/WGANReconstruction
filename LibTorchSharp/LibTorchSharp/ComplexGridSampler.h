#pragma once

#include <torch/torch.h>
#include "CustomOperatorsBackend.cuh"


namespace at {
    namespace native {


        namespace ComplexGridSampler {
            
            Tensor complex_grid_sampler_cpu_2d(const Tensor& input, const Tensor& grid,
                int interpolation_mode, int padding_mode,
                bool align_corners);

            Tensor complex_grid_sampler_cpu_3d(const Tensor& input, const Tensor& grid,
                int interpolation_mode, int padding_mode,
                bool align_corners, double max_r2);

            std::tuple<Tensor, Tensor>
                complex_grid_sampler_cpu_2d_backward(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int interpolation_mode,
                    int padding_mode, bool align_corners);

            std::tuple<Tensor, Tensor>
                complex_grid_sampler_cpu_3d_backward(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int interpolation_mode, int padding_mode,
                    bool align_corners);
        }
    }
}

