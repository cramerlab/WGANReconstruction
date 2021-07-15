#include "CustomOperatorsAutograd.h"
#include "CustomModules.h"
#include "Utils.h"
#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

Tensor THSNN_FFTCrop(const Tensor fft_volume, int new_x, int new_y, int new_z)
{
	CATCH_TENSOR(fft_crop(*fft_volume, new_x, new_y, new_z));
}