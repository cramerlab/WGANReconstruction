#include <cuda.h>
#include <cuda_runtime.h>
//#include <torch/torch.h>
//#include <ATen/ATen.h>


#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <C10/macros/Macros.h>
#include <C10/cuda/CUDAException.h>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "CustomOperatorsBackend.cuh"

namespace at {
    namespace native {
        using namespace at::cuda::detail;
        namespace MyOperator {
            
            template <typename scalar_t, typename index_t>
            __global__ void FFTCropKernel(TensorInfo<scalar_t, index_t> input, TensorInfo<scalar_t, index_t> output)
            {
                index_t inp_N = input.sizes[0];
                index_t inp_D = input.sizes[1];
                index_t inp_H = input.sizes[2];
                index_t inp_W = input.sizes[3];

                index_t out_N = output.sizes[0];
                index_t out_D = output.sizes[1];
                index_t out_H = output.sizes[2];
                index_t out_W = output.sizes[3];

                index_t inp_sN = input.strides[0];
                index_t inp_sD = input.strides[1];
                index_t inp_sH = input.strides[2];
                index_t inp_sW = input.strides[3];

                index_t out_sN = output.strides[0];
                index_t out_sD = output.strides[1];
                index_t out_sH = output.strides[2];
                index_t out_sW = output.strides[3];

                index_t n = blockIdx.z;

                auto p_inp_N = input.data + inp_sN * n;
                auto p_out_N = output.data + out_sN * n;


                for (int x = threadIdx.x; x < out_W; x += blockDim.x)
                {
                    int y = blockIdx.x;
                    int yy = y < out_H / 2 + 1 ? y : y - out_H + inp_H;
                    int z = blockIdx.y;
                    int zz = z < out_D / 2 + 1 ? z : z - out_D + inp_D;

                    //yy = tmax(0, tmin(yy, olddims.y - 1));
                    //zz = tmax(0, tmin(zz, olddims.z - 1));

                    *(p_out_N + z * out_sD + y * out_sH + x * out_sW) = *(p_inp_N + zz * inp_sD + yy * inp_sH + x * inp_sW);
                }
            }
            
            template <typename scalar_t, typename index_t>
            __global__ void FFTCropKernel_backwards(TensorInfo<scalar_t, index_t> grad_input, TensorInfo<scalar_t, index_t> grad_output)
            {
                index_t inp_N = grad_input.sizes[0];
                index_t inp_D = grad_input.sizes[1];
                index_t inp_H = grad_input.sizes[2];
                index_t inp_W = grad_input.sizes[3];

                index_t out_N = grad_output.sizes[0];
                index_t out_D = grad_output.sizes[1];
                index_t out_H = grad_output.sizes[2];
                index_t out_W = grad_output.sizes[3];

                index_t inp_sN = grad_input.strides[0];
                index_t inp_sD = grad_input.strides[1];
                index_t inp_sH = grad_input.strides[2];
                index_t inp_sW = grad_input.strides[3];

                index_t out_sN = grad_output.strides[0];
                index_t out_sD = grad_output.strides[1];
                index_t out_sH = grad_output.strides[2];
                index_t out_sW = grad_output.strides[3];

                index_t n = blockIdx.z;

                auto p_gInp_N = grad_input.data + inp_sN * n;
                auto p_gOut_N = grad_output.data + out_sN * n;

                for (int x = threadIdx.x; x < out_W; x += blockDim.x)
                {
                    int y = blockIdx.x;
                    int yy = y < out_H / 2 + 1 ? y : y - out_H + inp_H;
                    int z = blockIdx.y;
                    int zz = z < out_D / 2 + 1 ? z : z - out_D + inp_D;

                    *(p_gInp_N + zz * inp_sD + yy * inp_sH + x * inp_sW) = *(p_gOut_N + z * out_sD + y * out_sH + x * out_sW);
                }
            }
            
            //NewDims are in real space coordinates
            torch::Tensor fft_crop_cuda(const torch::Tensor& fft_volume, int64_t newDims_x, int64_t newDims_y, int64_t newDims_z) {
                
                //newDims is in realSpace
                
                torch::Tensor output = torch::empty({fft_volume.size(0), newDims_z, newDims_y, newDims_x / 2 + 1}, fft_volume.options());
                
                int TpB = std::min((int64_t)256, (((newDims_x / 2 + 1) + (32) - 1) / (32) * (32)));
                dim3 grid = dim3(newDims_y, newDims_z, fft_volume.size(0));
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(fft_volume.scalar_type(), "fft_crop", [&] {

                    if (canUse32BitIndexMath(fft_volume))
                    {
                        FFTCropKernel << <grid, TpB >> > (getTensorInfo<scalar_t, int>(fft_volume), getTensorInfo<scalar_t, int>(output));
                    }
                    else {
                        FFTCropKernel << <grid, TpB >> > (getTensorInfo<scalar_t, int64_t>(fft_volume), getTensorInfo<scalar_t, int64_t>(output));
                    }
                    });
                return output;
            }

            torch::Tensor fft_crop_backwards_cuda(const torch::Tensor& grad_output, int64_t oldDims_x, int64_t oldDims_y, int64_t oldDims_z) {

                // Grad output  has dimensions:
                // N - batchsize
                // H - z 
                // D - y
                // W - x 
                // oldDims is in realspace
                torch::Tensor grad_input = torch::zeros({ grad_output.size(0), oldDims_z, oldDims_y, oldDims_x }, grad_output.options());

                int TpB = std::min((int64_t)256, (((grad_output.size(3)) + (32) - 1) / (32) * (32)));
                dim3 grid = dim3(grad_output.size(2), grad_output.size(1), grad_output.size(0));
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "fft_crop", [&] {

                    if (canUse32BitIndexMath(grad_input))
                    {
                        FFTCropKernel_backwards << <grid, TpB >> > (getTensorInfo<scalar_t, int>(grad_input), getTensorInfo<scalar_t, int>(grad_output));
                    }
                    else {
                        FFTCropKernel_backwards << <grid, TpB >> > (getTensorInfo<scalar_t, int64_t>(grad_input), getTensorInfo<scalar_t, int64_t>(grad_output));
                    }
                    });
                return grad_input;
            }
        }
    }
}
