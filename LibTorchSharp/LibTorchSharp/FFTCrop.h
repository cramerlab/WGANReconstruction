#pragma once

#include <torch/torch.h>
using namespace c10;

torch::Tensor fftshift(const torch::Tensor& x, c10::optional<IntArrayRef> dim_opt);
torch::Tensor ifftshift(const torch::Tensor& x, c10::optional<IntArrayRef> dim_opt);
