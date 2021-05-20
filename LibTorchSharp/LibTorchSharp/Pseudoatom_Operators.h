#pragma once

#include <torch/torch.h>
#include "../Stdafx.h"
using torch::Tensor;
using namespace torch::autograd;


EXPORT_API(Tensor) atoms_to_grid(const Tensor& intensities, const Tensor& positions);
EXPORT_API(Tensor) projectAtoms(const Tensor& intensities, const Tensor& positions, const Tensor& orientation);
EXPORT_API(Tensor) fft_crop(const Tensor& input, int x, int y, int z);
