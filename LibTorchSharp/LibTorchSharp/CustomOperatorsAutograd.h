#pragma once

#include <torch/torch.h>
#include "../Stdafx.h"

using namespace torch::autograd;

torch::Tensor atoms_to_grid(const torch::Tensor& intensities, const torch::Tensor& positions, int64_t x, int64_t y, int64_t z);
torch::Tensor fft_crop(const torch::Tensor& fft_volume, int64_t x, int64_t y, int64_t z);
torch::Tensor projectAtoms(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, int64_t x, int64_t y, int64_t z);
//torch::Tensor fft_crop(const torch::Tensor& input, int x, int y, int z);
