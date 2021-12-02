#pragma once

#include <torch/torch.h>
using namespace c10;

torch::Tensor fftshift(const torch::Tensor& x, c10::optional<IntArrayRef> dim_opt);
torch::Tensor ifftshift(const torch::Tensor& x, c10::optional<IntArrayRef> dim_opt);
torch::Tensor fft_crop(torch::Tensor& fft_volume, int dim, int new_x, int new_y, int new_z);
