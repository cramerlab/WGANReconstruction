#include<torch/torch.h>
#include "Utils.h"
using namespace torch::indexing;

torch::Tensor matrix_from_angles(torch::Tensor& angles)
{
    auto ca = angles.index({ Slice(0, None), 0 }).cos();
    auto cb = angles.index({ Slice(0, None), 1 }).cos();
    auto cg = angles.index({ Slice(0, None), 2 }).cos();
    auto sa = angles.index({ Slice(0, None), 0 }).sin();
    auto sb = angles.index({ Slice(0, None), 0 }).sin();
    auto sg = angles.index({ Slice(0, None), 0 }).sin();
    auto cc = cb * ca;
    auto cs = cb * sa;
    auto sc = sb * ca;
    auto ss = sb * sa;
    auto matrix = torch::empty((1, 3, 3), angles.options());
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

Tensor THSNN_MatrixFromAngles(Tensor angles) {
    CATCH_TENSOR(matrix_from_angles(*angles));
}