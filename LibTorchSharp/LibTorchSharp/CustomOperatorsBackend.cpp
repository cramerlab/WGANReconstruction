#include<torch/torch.h>
#include<torch/fft.h>
#include "FFTCrop.h"
#include "Utils.h"
#include "CustomModules.h"
#include "CustomOperatorsAutograd.h"

using namespace torch::indexing;

torch::Tensor matrix_from_angles(torch::Tensor& angles)
{
    auto ca = angles.index({ Slice(0, None), 0 }).cos();
    auto cb = angles.index({ Slice(0, None), 1 }).cos();
    auto cg = angles.index({ Slice(0, None), 2 }).cos();
    auto sa = angles.index({ Slice(0, None), 0 }).sin();
    auto sb = angles.index({ Slice(0, None), 1 }).sin();
    auto sg = angles.index({ Slice(0, None), 2 }).sin();
    auto cc = cb * ca;
    auto cs = cb * sa;
    auto sc = sb * ca;
    auto ss = sb * sa;

    auto matDims = new int64_t[]{ angles.size(0), 3, 3 };
    auto matrix = torch::empty(c10::IntArrayRef(matDims, 3), angles.options());
    delete matDims;

    matrix.index_put_({ Slice(0, None), 0, 0 }, cg * cc - sg * sa);
    matrix.index_put_({ Slice(0, None), 0, 1 }, cg * cs + sg * ca);
    matrix.index_put_({ Slice(0, None), 0, 2 }, -cg * sb);
    matrix.index_put_({ Slice(0, None), 1, 0 }, -sg * cc - cg * sa);
    matrix.index_put_({ Slice(0, None), 1, 1 }, -sg * cs + cg * ca);
    matrix.index_put_({ Slice(0, None), 1, 2 }, sg * sb);
    matrix.index_put_({ Slice(0, None), 2, 0 }, sc);
    matrix.index_put_({ Slice(0, None), 2, 1 }, ss);
    matrix.index_put_({ Slice(0, None), 2, 2 }, cb);
    return matrix;
}

torch::Tensor affine_matrix_from_angles(torch::Tensor& angles, float shift) {
    torch::Tensor trans_matrices = matrix_from_angles(angles);
    torch::Tensor shifts;

    shifts = torch::ones({ angles.size(0), 3, 1 }, angles.options()) * shift;
    trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
    trans_matrices = trans_matrices.to(angles.device());
    return trans_matrices;
}

torch::Tensor rotateVolume(torch::Tensor& volume, torch::Tensor& angles, float shift) {
    torch::Tensor trans_matrices = matrix_from_angles(angles);
    torch::Tensor shifts;

    shifts = torch::ones({ angles.size(0), 3, 1 }, angles.options())*shift;
    trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
    trans_matrices = trans_matrices.to(volume.device());

    torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, volume.size(2), volume.size(2), volume.size(2) }, true);
    torch::Tensor volumeRot = torch::nn::functional::grid_sample(volume.size(0) < angles.size(0) ? volume.expand(c10::IntArrayRef(new int64_t[]{ angles.size(0), -1, -1, -1, -1 }, 5)) : volume, trans_grid, 
        torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).align_corners(true));;
    return volumeRot;
}

torch::Tensor scaleVolume(torch::Tensor& volume, int dim, int new_x, int new_y, int new_z) {
    if (dim == 2) {
        torch::Tensor fft_volume = torch::fft::rfftn(volume, c10::nullopt, c10::IntArrayRef({ (int64_t)volume.dim() - 2, (int64_t)volume.dim() - 1 }), "forward");
        torch::Tensor new_fft_volume = fft_crop(fft_volume, dim, new_x, new_y, new_z);
        torch::Tensor new_Volume = torch::fft::irfftn(new_fft_volume, c10::nullopt, c10::IntArrayRef({ (int64_t)volume.dim() - 2, (int64_t)volume.dim() - 1 }), "forward");
        return new_Volume;
    }
    if (dim == 3) {
        torch::Tensor fft_volume = torch::fft::rfftn(volume, c10::nullopt, c10::IntArrayRef({ (int64_t)volume.dim() - 3, (int64_t)volume.dim() - 2, (int64_t)volume.dim() - 1 }), "forward");
        torch::Tensor new_fft_volume = fft_crop(fft_volume, dim, new_x, new_y, new_z);
        torch::Tensor new_Volume = torch::fft::irfftn(new_fft_volume, c10::nullopt, c10::IntArrayRef({ (int64_t)volume.dim() - 3, (int64_t)volume.dim() - 2, (int64_t)volume.dim() - 1 }), "forward");
        return new_Volume;
    }
}


Tensor THSNN_ScaleVolume(Tensor volume, int dim, int new_x, int new_y, int new_z) {
    CATCH_TENSOR(scaleVolume(*volume, dim, new_x, new_y, new_z));
}

Tensor THSNN_RotateVolume(Tensor volume, Tensor angles, float shift) {
    CATCH_TENSOR(rotateVolume(*volume, *angles, shift));
}

Tensor THSNN_MatrixFromAngles(Tensor angles) {
    CATCH_TENSOR(matrix_from_angles(*angles));

}Tensor THSNN_AffineMatrixFromAngles(Tensor angles, float shift) {
    CATCH_TENSOR(affine_matrix_from_angles(*angles, shift));
}