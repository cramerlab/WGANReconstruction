#include "CustomOperatorsAutograd.h"
#include "CustomModules.h"
#include "Utils.h"
#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

using namespace torch::indexing;

torch::Tensor fft_crop(torch::Tensor& fft_volume, int dim, int new_x, int new_y, int new_z) {

    auto n = fft_volume.ndimension();

    auto newDims = fft_volume.sizes().vec();
    newDims[n - 1] = (int64_t)new_x / 2 + 1;
    if (dim >= 2)
        newDims[n - 2] = new_y;
    if (dim == 3)
        newDims[n - 3] = new_z;

    auto newVol = torch::empty(newDims, fft_volume.options());

    int old_x = (fft_volume.size(-1)-1)*2;
    int x = std::min(old_x, new_x);
    int old_y = fft_volume.size(-2);
    int y = std::min(old_y, new_y);
    if (dim == 2) {
        newVol.index_put_({ "...", Slice(0, y / 2 + 1), Slice(0, x / 2 + 1) },
            fft_volume.index({ "...",Slice(0, y / 2 + 1), Slice(0, x / 2 + 1) }));

        newVol.index_put_({ "...", Slice(new_y - y + (y / 2 + 1), None), Slice(0, x / 2 + 1) },
            fft_volume.index({ "...", Slice(old_y - y + (y / 2 + 1), None), Slice(0, x / 2 + 1) }));
    }

    if (dim == 3) {
        int old_z = fft_volume.size(-3);
        int z = std::min(old_z, new_z);
        newVol.index_put_({ "...", Slice(0, z / 2 + 1), Slice(0, y / 2 + 1), Slice(0, x / 2 + 1) },
            fft_volume.index({ "...", Slice(0, z / 2 + 1), Slice(0, y / 2 + 1), Slice(0, x / 2 + 1) }));

        newVol.index_put_({ "...", Slice(0, z / 2 + 1), Slice(new_y - y + (y / 2 + 1), None), Slice(0, x / 2 + 1) },
            fft_volume.index({ "...", Slice(0, z / 2 + 1), Slice(old_y - y + (y / 2 + 1), None), Slice(0, x / 2 + 1) }));

        newVol.index_put_({ "...", Slice(new_z - z + z / 2 + 1, None), Slice(0, y / 2 + 1), Slice(0, x / 2 + 1) },
            fft_volume.index({ "...", Slice(old_z - z + z / 2 + 1, None), Slice(0, y / 2 + 1), Slice(0, x / 2 + 1) }));

        newVol.index_put_({ "...", Slice(new_z - z + z / 2 + 1, None), Slice(new_y - y + (y / 2 + 1), None), Slice(0, x / 2 + 1) },
            fft_volume.index({ "...", Slice(old_z - z + z / 2 + 1, None), Slice(old_y - y + (y / 2 + 1), None), Slice(0, x / 2 + 1) }));
    }
    return newVol;
}


Tensor THSNN_FFTCrop(Tensor fft_volume, int dim, int new_x, int new_y, int new_z)
{
    CATCH_TENSOR(fft_crop(*fft_volume, dim, new_x, new_y, new_z));
}