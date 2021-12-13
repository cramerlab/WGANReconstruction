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
        using at::native::GridSamplerInterpolation;
        using at::native::GridSamplerPadding;
        using namespace at::cuda::detail;

        //using at::native::detail::GridSamplerInterpolation;
        //using at::native::detail::GridSamplerPadding;

        namespace MyOperator {
            const int MY_CUDA_MAX_THREADS = 256;
            inline int MY_CUDA_GET_BLOCKS(const int64_t N) {
                AT_ASSERTM(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
                constexpr int64_t max_int = std::numeric_limits<int>::max();

                // Round up division for positive number that cannot cause integer overflow
                auto block_num = (N - 1) / MY_CUDA_MAX_THREADS + 1;
                AT_ASSERTM(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

                return static_cast<int>(block_num);
            }
            // Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
            // where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
            // if align_corners: -1 and +1 get sent to the centers of the corner pixels
            //     -1 --> 0
            //     +1 --> (size - 1)
            //     scale_factor = (size - 1) / 2
            // if not align_corners: -1 and +1 get sent to the image edges
            //     -1 --> -0.5
            //     +1 --> (size - 1) + 0.5 == size - 0.5
            //     scale_factor = size / 2
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t atoms_to_grid_unnormalize(scalar_t coord, int size, bool align_corners) {
                if (align_corners) {
                    // unnormalize coord from [-1, 1] to [0, size - 1]
                    return ((coord + 1.f) / 2) * (size - 1);
                }
                else {
                    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                    return ((coord + 1.f) * size - 1) / 2;
                }
            }

            // atoms_to_grid_unnormalize_set_grad works the same as atoms_to_grid_unnormalize
            // except that it also returns the `d output / d input` via pointer argument
            // `grad_in`.
            // This is useful in the backward pass of atoms_to_grid.
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t atoms_to_grid_unnormalize_set_grad(scalar_t coord, int size,
                    bool align_corners, scalar_t* grad_in) {
                if (align_corners) {
                    // unnormalize coord from [-1, 1] to [0, size - 1]
                    *grad_in = static_cast<scalar_t>(size - 1) / 2;
                    return ((coord + 1.f) / 2) * (size - 1);
                }
                else {
                    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                    *grad_in = static_cast<scalar_t>(size) / 2;
                    return ((coord + 1.f) * size - 1) / 2;
                }
            }

            // Clips coordinates to between 0 and clip_limit - 1
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t clip_coordinates(scalar_t in, int clip_limit) {
                return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
            }

            // clip_coordinates_set_grad works similarly to clip_coordinates except that
            // it also returns the `d output / d input` via pointer argument `grad_in`.
            // This is useful in the backward pass of atoms_to_grid.
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, scalar_t* grad_in) {
                // Note that it is important for the gradient calculation that borders
                // are considered out of bounds.
                if (in <= static_cast<scalar_t>(0)) {
                    *grad_in = static_cast<scalar_t>(0);
                    return static_cast<scalar_t>(0);
                }
                else {
                    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
                    if (in >= max) {
                        *grad_in = static_cast<scalar_t>(0);
                        return max;
                    }
                    else {
                        *grad_in = static_cast<scalar_t>(1);
                        return in;
                    }
                }
            }

            // Reflects coordinates until they fall between low and high (inclusive).
            // The bounds are passed as twice their value so that half-integer values
            // can be represented as ints.
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
                if (twice_low == twice_high) {
                    return static_cast<scalar_t>(0);
                }
                scalar_t min = static_cast<scalar_t>(twice_low) / 2;
                scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
                in = ::fabs(in - min);
                // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
                scalar_t extra = ::fmod(in, span);
                int flips = static_cast<int>(::floor(in / span));
                if (flips % 2 == 0) {
                    return extra + min;
                }
                else {
                    return span - extra + min;
                }
            }

            // reflect_coordinates_set_grad works similarly to reflect_coordinates except
            // that it also returns the `d output / d input` via pointer argument
            // `grad_in`.
            // This is useful in the backward pass of atoms_to_grid.
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t reflect_coordinates_set_grad(scalar_t in, int twice_low, int twice_high,
                    scalar_t* grad_in) {
                if (twice_low == twice_high) {
                    *grad_in = static_cast<scalar_t>(0);
                    return static_cast<scalar_t>(0);
                }
                int grad_in_mult_;
                scalar_t min = static_cast<scalar_t>(twice_low) / 2;
                scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
                in = in - min;
                if (in < static_cast<scalar_t>(0)) {
                    grad_in_mult_ = -1;
                    in = -in;
                }
                else {
                    grad_in_mult_ = 1;
                }
                // `fmod` returns same sign as `in`, which is positive after the `if` above.
                scalar_t extra = ::fmod(in, span);
                int flips = static_cast<int>(::floor(in / span));
                if (flips % 2 == 0) {
                    *grad_in = static_cast<scalar_t>(grad_in_mult_);
                    return extra + min;
                }
                else {
                    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
                    return span - extra + min;
                }
            }

            template<typename scalar_t>
            static __forceinline__ __device__
                scalar_t safe_downgrade_to_int_range(scalar_t x) {
                // -100.0 does not have special meaning. This is just to make sure
                // it's not within_bounds_2d or within_bounds_3d, and does not cause
                // undefined behavior. See #35506.
                if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
                    return static_cast<scalar_t>(-100.0);
                return x;
            }

            template<typename scalar_t>
            static __forceinline__ __device__
                scalar_t compute_coordinates(scalar_t coord, int size,
                    GridSamplerPadding padding_mode,
                    bool align_corners) {
                if (padding_mode == GridSamplerPadding::Border) {
                    // clip coordinates to image borders
                    coord = clip_coordinates(coord, size);
                }
                else if (padding_mode == GridSamplerPadding::Reflection) {
                    // reflect coordinates by image borders
                    if (align_corners) {
                        coord = reflect_coordinates(coord, 0, 2 * (size - 1));
                    }
                    else {
                        coord = reflect_coordinates(coord, -1, 2 * size - 1);
                    }
                    // clip coordinates to image borders
                    coord = clip_coordinates(coord, size);
                }

                coord = safe_downgrade_to_int_range(coord);
                return coord;
            }

            // Computes the pixel source index value for a grid coordinate
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t atoms_to_grid_compute_source_index(
                    scalar_t coord,
                    int size,
                    GridSamplerPadding padding_mode,
                    bool align_corners) {
                coord = atoms_to_grid_unnormalize(coord, size, align_corners);
                coord = compute_coordinates(coord, size, padding_mode, align_corners);
                return coord;
            }

            // atoms_to_grid_compute_source_index_set_grad works similarly to
            // atoms_to_grid_compute_source_index except that it also returns the
            // `d output / d input` via pointer argument `grad_in`.
            // This is useful in the backward pass of atoms_to_grid.
            template <typename scalar_t>
            static __forceinline__ __device__
                scalar_t atoms_to_grid_compute_source_index_set_grad(
                    scalar_t coord,
                    int size,
                    GridSamplerPadding padding_mode,
                    bool align_corners,
                    scalar_t* grad_in) {
                scalar_t grad_clip, grad_refl;
                coord = atoms_to_grid_unnormalize_set_grad(coord, size, align_corners, grad_in);
                if (padding_mode == GridSamplerPadding::Border) {
                    // clip coordinates to image borders
                    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
                    *grad_in = (*grad_in) * grad_clip;
                }
                else if (padding_mode == GridSamplerPadding::Reflection) {
                    // reflect coordinates by image borders
                    if (align_corners) {
                        coord = reflect_coordinates_set_grad(coord, 0, 2 * (size - 1), &grad_refl);
                    }
                    else {
                        coord = reflect_coordinates_set_grad(coord, -1, 2 * size - 1, &grad_refl);
                    }
                    // clip coordinates to image borders
                    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
                    *grad_in = (*grad_in) * grad_refl * grad_clip;
                }

                coord = safe_downgrade_to_int_range(coord);
                return coord;
            }

            static __forceinline__ __device__
                bool within_bounds_2d(int h, int w, int H, int W) {
                return h >= 0 && h < H&& w >= 0 && w < W;
            }

            static __forceinline__ __device__
                bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
                return d >= 0 && d < D&& h >= 0 && h < H&& w >= 0 && w < W;
            }


            template<typename scalar_t>
            static __forceinline__ __device__
                void safe_add_2d(scalar_t* data, int h, int w,
                    int sH, int sW, int H, int W,
                    scalar_t delta) {
                if (within_bounds_2d(h, w, H, W)) {
                    gpuAtomicAdd(data + h * sH + w * sW, delta);
                }
            }

            template<typename scalar_t>
            static __forceinline__ __device__
                void safe_add_3d(scalar_t* data, int d, int h, int w,
                    int sD, int sH, int sW, int D, int H, int W,
                    scalar_t delta) {
                if (within_bounds_3d(d, h, w, D, H, W)) {
                    gpuAtomicAdd(data + d * sD + h * sH + w * sW, delta);
                }
            }

            template <typename scalar_t, typename index_t>
            static __forceinline__ __device__ scalar_t unsafe_access(scalar_t* data, index_t d, index_t h, index_t w, index_t sD, index_t sH, index_t sW) {
                return *(data + d * sD + h * sH + w * sW);
            }

            template <typename scalar_t, typename index_t>
            static __forceinline__ __device__ scalar_t unsafe_access(scalar_t* data, index_t h, index_t w, index_t sH, index_t sW) {
                return *(data + h * sH + w * sW);
            }

            template <typename scalar_t, typename index_t>
            __global__ void atoms_to_grid_3d_backward_kernel(
                const index_t nthreads,
                TensorInfo<scalar_t, index_t> grad_output,
                TensorInfo<scalar_t, index_t> intensities,
                TensorInfo<scalar_t, index_t> positions,
                TensorInfo<scalar_t, index_t> orientations,
                TensorInfo<scalar_t, index_t> shift,
                TensorInfo<scalar_t, index_t> grad_intensities,  // initialized to zeros
                TensorInfo<scalar_t, index_t> grad_positions,   // initialized to empty
                TensorInfo<scalar_t, index_t> grad_orientations,
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {

                index_t int_D = intensities.sizes[1];
                index_t int_H = intensities.sizes[2];
                index_t int_W = intensities.sizes[3];
                index_t gOut_D = grad_output.sizes[1];
                index_t gOut_H = grad_output.sizes[2];
                index_t gOut_W = grad_output.sizes[3];
                index_t int_sN = intensities.strides[0];
                index_t int_sW = intensities.strides[1];
                index_t pos_sN = positions.strides[0];
                index_t pos_sW = positions.strides[1];
                index_t pos_sCoor = positions.strides[2];
                index_t orr_sN = orientations.strides[0];
                index_t orr_sR = orientations.strides[1];
                index_t orr_sC = orientations.strides[2];
                index_t shift_sN = shift.strides[0];
                index_t shift_sCoor = shift.strides[1];
                index_t gOut_sN = grad_output.strides[0];
                index_t gOut_sD = grad_output.strides[1];
                index_t gOut_sH = grad_output.strides[2];
                index_t gOut_sW = grad_output.strides[3];
                index_t gInp_sN = grad_intensities.strides[0];
                index_t gInp_sW = grad_intensities.strides[1];

                CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
                    const index_t w = index % int_W;
                    const index_t n = index / (int_W);
                    const index_t pos_offset = n * pos_sN + w * pos_sW;

                    // get the corresponding input x, y, z co-ordinates from pos
                    scalar_t ix = positions.data[pos_offset] + shift.data[n * shift_sN];
                    scalar_t iy = positions.data[pos_offset + pos_sCoor] + shift.data[n * shift_sN + shift_sCoor];
                    scalar_t iz = positions.data[pos_offset + 2 * pos_sCoor] + shift.data[n * shift_sN + 2 * shift_sCoor];

                    //rotate coordinates
                    scalar_t ox, oy, oz;
                    ox = ix;
                    oy = iy;
                    oz = iz;
                    scalar_t tIx = ix;
                    scalar_t tIy = iy;
                    scalar_t tIz = iz;
                    matMult(orientations.data + n * orr_sN, orr_sC, orr_sR, tIx, tIy, tIz, &ix, &iy, &iz);

                    scalar_t gix_mult, giy_mult, giz_mult;
                    ix = atoms_to_grid_compute_source_index_set_grad(ix, gOut_W, padding_mode, align_corners, &gix_mult);
                    iy = atoms_to_grid_compute_source_index_set_grad(iy, gOut_H, padding_mode, align_corners, &giy_mult);
                    iz = atoms_to_grid_compute_source_index_set_grad(iz, gOut_D, padding_mode, align_corners, &giz_mult);

                    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get corner pixel values from (x, y, z)
                        // for 4d, we used north-east-south-west
                        // for 5d, we add top-bottom
                        index_t ix_tnw = static_cast<index_t>((ix));
                        index_t iy_tnw = static_cast<index_t>((iy));
                        index_t iz_tnw = static_cast<index_t>((iz));

                        index_t ix_tne = ix_tnw + 1;
                        index_t iy_tne = iy_tnw;
                        index_t iz_tne = iz_tnw;

                        index_t ix_tsw = ix_tnw;
                        index_t iy_tsw = iy_tnw + 1;
                        index_t iz_tsw = iz_tnw;

                        index_t ix_tse = ix_tnw + 1;
                        index_t iy_tse = iy_tnw + 1;
                        index_t iz_tse = iz_tnw;

                        index_t ix_bnw = ix_tnw;
                        index_t iy_bnw = iy_tnw;
                        index_t iz_bnw = iz_tnw + 1;

                        index_t ix_bne = ix_tnw + 1;
                        index_t iy_bne = iy_tnw;
                        index_t iz_bne = iz_tnw + 1;

                        index_t ix_bsw = ix_tnw;
                        index_t iy_bsw = iy_tnw + 1;
                        index_t iz_bsw = iz_tnw + 1;

                        index_t ix_bse = ix_tnw + 1;
                        index_t iy_bse = iy_tnw + 1;
                        index_t iz_bse = iz_tnw + 1;

                        // get surfaces to each neighbor:
                        scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
                        scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
                        scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
                        scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
                        scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
                        scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
                        scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
                        scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

                        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0), gi = static_cast<scalar_t>(0);
                        auto int_ptr_NC = intensities.data + n * int_sN;
                        auto gInp_ptr_NCDHW = grad_intensities.data + n * gInp_sN + w * gInp_sW;
                        auto int_val_NCDHW = *(intensities.data + n * int_sN + w * int_sW);
                        auto gOut_ptr_NC = grad_output.data + n * gOut_sN;
                        // calculate grad_pos
                        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gOut_sD, gOut_sH, gOut_sW);
                            gi += tnw * gOut;
                            gix -= int_val_NCDHW * (iy_bse - iy) * (iz_bse - iz) * gOut;
                            giy -= int_val_NCDHW * (ix_bse - ix) * (iz_bse - iz) * gOut;
                            giz -= int_val_NCDHW * (ix_bse - ix) * (iy_bse - iy) * gOut;
                        }
                        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_tne, iy_tne, ix_tne, gOut_sD, gOut_sH, gOut_sW);
                            gi += tne * gOut;
                            gix += int_val_NCDHW * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
                            giy -= int_val_NCDHW * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
                            giz -= int_val_NCDHW * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
                        }
                        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gOut_sD, gOut_sH, gOut_sW);
                            gi += tsw * gOut;
                            gix -= int_val_NCDHW * (iy - iy_bne) * (iz_bne - iz) * gOut;
                            giy += int_val_NCDHW * (ix_bne - ix) * (iz_bne - iz) * gOut;
                            giz -= int_val_NCDHW * (ix_bne - ix) * (iy - iy_bne) * gOut;
                        }
                        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_tse, iy_tse, ix_tse, gOut_sD, gOut_sH, gOut_sW);
                            gi += tse * gOut;
                            gix += int_val_NCDHW * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
                            giy += int_val_NCDHW * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
                            giz -= int_val_NCDHW * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
                        }
                        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gOut_sD, gOut_sH, gOut_sW);
                            gi += bnw * gOut;
                            gix -= int_val_NCDHW * (iy_tse - iy) * (iz - iz_tse) * gOut;
                            giy -= int_val_NCDHW * (ix_tse - ix) * (iz - iz_tse) * gOut;
                            giz += int_val_NCDHW * (ix_tse - ix) * (iy_tse - iy) * gOut;
                        }
                        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_bne, iy_bne, ix_bne, gOut_sD, gOut_sH, gOut_sW);
                            gi += bne * gOut;
                            gix += int_val_NCDHW * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
                            giy -= int_val_NCDHW * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
                            giz += int_val_NCDHW * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
                        }
                        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gOut_sD, gOut_sH, gOut_sW);
                            gi += bsw * gOut;
                            gix -= int_val_NCDHW * (iy - iy_tne) * (iz - iz_tne) * gOut;
                            giy += int_val_NCDHW * (ix_tne - ix) * (iz - iz_tne) * gOut;
                            giz += int_val_NCDHW * (ix_tne - ix) * (iy - iy_tne) * gOut;
                        }
                        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, gOut_D, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iz_bse, iy_bse, ix_bse, gOut_sD, gOut_sH, gOut_sW);
                            gi += bse * gOut;
                            gix += int_val_NCDHW * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
                            giy += int_val_NCDHW * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
                            giz += int_val_NCDHW * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
                        }
                        *(gInp_ptr_NCDHW) = gi;

                        tIx = gix * gix_mult;
                        tIy = giy * giy_mult;
                        tIz = giz * giz_mult;
                        safe_matGrad(grad_orientations.data + n * orr_sN, orr_sC, orr_sR, ox, oy, oz, tIx, tIy, tIz);

                        matMultT(orientations.data + n * orr_sN, orr_sC, orr_sR, tIx, tIy, tIz, &gix, &giy, &giz);
                        grad_positions.data[pos_offset] = gix;
                        grad_positions.data[pos_offset + pos_sCoor] = giy;
                        grad_positions.data[pos_offset + 2 * pos_sCoor] = giz;
                    }
                }
            }


            template <typename scalar_t, typename index_t>
            __global__ void atoms_to_grid_3d_kernel(
                const index_t nthreads,
                TensorInfo<scalar_t, index_t> intensities,
                TensorInfo<scalar_t, index_t> positions,
                TensorInfo<scalar_t, index_t> orientations,
                TensorInfo<scalar_t, index_t> shift,
                TensorInfo<scalar_t, index_t> output, //Initialized to zeros
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {


                index_t out_D = output.sizes[1];
                index_t out_H = output.sizes[2];
                index_t out_W = output.sizes[3];
                index_t pos_N = positions.sizes[0];
                index_t pos_W = positions.sizes[1];
                index_t orr_sN = orientations.strides[0];
                index_t orr_sR = orientations.strides[1];
                index_t orr_sC = orientations.strides[2];
                index_t out_sN = output.strides[0];
                index_t out_sD = output.strides[1];
                index_t out_sH = output.strides[2];
                index_t out_sW = output.strides[3];
                index_t pos_sN = positions.strides[0];
                index_t pos_sW = positions.strides[1];
                index_t pos_sCoor = positions.strides[2];
                index_t shift_sN = shift.strides[0];
                index_t shift_sCoor = shift.strides[1];
                index_t int_sN = intensities.strides[0];
                index_t int_sW = intensities.strides[1];

                CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
                    const index_t w = index % pos_W;
                    const index_t n = index / (pos_W);
                    const auto pos_offset = n * pos_sN + w * pos_sW;

                    // get the corresponding output x, y, z co-ordinates from pos
                    // 1st the coordinates saved in pos, i.e. in [-1, 1]
                    scalar_t ix = positions.data[pos_offset] + shift.data[n * shift_sN];
                    scalar_t iy = positions.data[pos_offset + pos_sCoor] + shift.data[n * shift_sN + shift_sCoor];
                    scalar_t iz = positions.data[pos_offset + 2 * pos_sCoor] + shift.data[n * shift_sN + 2 * shift_sCoor];

                    scalar_t tIx = ix;
                    scalar_t tIy = iy;
                    scalar_t tIz = iz;
                    matMult(orientations.data + n * orr_sN, orr_sC, orr_sR, tIx, tIy, tIz, &ix, &iy, &iz);

                    // 2nd, unnormalized coordinates in [0, outsize-1]
                    ix = atoms_to_grid_compute_source_index(ix, out_W, padding_mode, align_corners);
                    iy = atoms_to_grid_compute_source_index(iy, out_H, padding_mode, align_corners);
                    iz = atoms_to_grid_compute_source_index(iz, out_D, padding_mode, align_corners);

                    // get corner pixel values from (x, y, z)
                    /*index_t ix_tnw = static_cast<index_t>((ix));
                    index_t iy_tnw = static_cast<index_t>((iy));
                    index_t iz_tnw = static_cast<index_t>((iz));*/
                    index_t ix_tnw = static_cast<index_t>((ix));
                    index_t iy_tnw = static_cast<index_t>((iy));
                    index_t iz_tnw = static_cast<index_t>((iz));
                    index_t ix_tne = ix_tnw + 1;
                    index_t iy_tne = iy_tnw;
                    index_t iz_tne = iz_tnw;

                    index_t ix_tsw = ix_tnw;
                    index_t iy_tsw = iy_tnw + 1;
                    index_t iz_tsw = iz_tnw;

                    index_t ix_tse = ix_tnw + 1;
                    index_t iy_tse = iy_tnw + 1;
                    index_t iz_tse = iz_tnw;

                    index_t ix_bnw = ix_tnw;
                    index_t iy_bnw = iy_tnw;
                    index_t iz_bnw = iz_tnw + 1;

                    index_t ix_bne = ix_tnw + 1;
                    index_t iy_bne = iy_tnw;
                    index_t iz_bne = iz_tnw + 1;

                    index_t ix_bsw = ix_tnw;
                    index_t iy_bsw = iy_tnw + 1;
                    index_t iz_bsw = iz_tnw + 1;

                    index_t ix_bse = ix_tnw + 1;
                    index_t iy_bse = iy_tnw + 1;
                    index_t iz_bse = iz_tnw + 1;

                    // get surfaces to each neighbor on cartesian grid:
                    scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
                    scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
                    scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
                    scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
                    scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
                    scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
                    scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
                    scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

                    //intensity value at current grid position
                    scalar_t int_val_NCDHW = *(intensities.data + n * int_sN + w * int_sW);
                    scalar_t* out_ptr_NC = output.data + n * out_sN;

                    // calculate bilinear weighted pixel value and set output pixel
                    safe_add_3d(out_ptr_NC, iz_tnw, iy_tnw, ix_tnw, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * tnw);
                    safe_add_3d(out_ptr_NC, iz_tne, iy_tne, ix_tne, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * tne);
                    safe_add_3d(out_ptr_NC, iz_tsw, iy_tsw, ix_tsw, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * tsw);
                    safe_add_3d(out_ptr_NC, iz_tse, iy_tse, ix_tse, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * tse);
                    safe_add_3d(out_ptr_NC, iz_bnw, iy_bnw, ix_bnw, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * bnw);
                    safe_add_3d(out_ptr_NC, iz_bne, iy_bne, ix_bne, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * bne);
                    safe_add_3d(out_ptr_NC, iz_bsw, iy_bsw, ix_bsw, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * bsw);
                    safe_add_3d(out_ptr_NC, iz_bse, iy_bse, ix_bse, out_sD, out_sH, out_sW, out_D, out_H, out_W, int_val_NCDHW * bse);
                }
            }


            template <typename scalar_t, typename index_t>
            static  __forceinline__ __device__ void matMult(scalar_t* mat, index_t sC, index_t sR, scalar_t x, scalar_t y, scalar_t z, scalar_t* rX, scalar_t* rY, scalar_t* rZ) {
                *(rX) = x * (*mat) + y * (*(mat + sC)) + z * (*(mat + 2 * sC));
                mat += sR;
                *(rY) = x * (*mat) + y * (*(mat + sC)) + z * (*(mat + 2 * sC));
                mat += sR;
                *(rZ) = x * (*mat) + y * (*(mat + sC)) + z * (*(mat + 2 * sC));
            }

            template <typename scalar_t, typename index_t>
            static  __forceinline__ __device__ void safe_matGrad(scalar_t* grad_mat, index_t sC, index_t sR, scalar_t x, scalar_t y, scalar_t z, scalar_t gx, scalar_t gy, scalar_t gz) {
                gpuAtomicAdd(grad_mat, x * gx);
                gpuAtomicAdd(grad_mat + sC, y * gx);
                gpuAtomicAdd(grad_mat + 2 * sC, z * gx);
                grad_mat += sR;
                gpuAtomicAdd(grad_mat, x * gy);
                gpuAtomicAdd(grad_mat + sC, y * gy);
                gpuAtomicAdd(grad_mat + 2 * sC, z * gy);
                grad_mat += sR;
                gpuAtomicAdd(grad_mat, x * gz);
                gpuAtomicAdd(grad_mat + sC, y * gz);
                gpuAtomicAdd(grad_mat + 2 * sC, z * gz);
            }

            template <typename scalar_t, typename index_t>
            static  __forceinline__ __device__ void matMultT(scalar_t* mat, index_t sC, index_t sR, scalar_t x, scalar_t y, scalar_t z, scalar_t* rX, scalar_t* rY, scalar_t* rZ) {
                *(rX) = x * (*mat) + y * (*(mat + sR)) + z * (*(mat + 2 * sR));
                mat += sC;
                *(rY) = x * (*mat) + y * (*(mat + sR)) + z * (*(mat + 2 * sR));
                mat += sC;
                *(rZ) = x * (*mat) + y * (*(mat + sR)) + z * (*(mat + 2 * sR));
            }



            template <typename scalar_t, typename index_t>
            __global__ void projectAtoms_backwards_kernel(
                const index_t nthreads,
                TensorInfo<scalar_t, index_t> positions,
                TensorInfo<scalar_t, index_t> intensities,
                TensorInfo<scalar_t, index_t> orientation,
                TensorInfo<scalar_t, index_t> shift,
                TensorInfo<scalar_t, index_t> grad_output,
                TensorInfo<scalar_t, index_t> grad_positions,
                TensorInfo<scalar_t, index_t> grad_intensities,
                TensorInfo<scalar_t, index_t> grad_orientations,
                int64_t x, int64_t y, int64_t z)
            {


                index_t pos_N = positions.sizes[0];
                index_t pos_W = positions.sizes[1];
                index_t gOut_H = grad_output.sizes[1];
                index_t gOut_W = grad_output.sizes[2];

                index_t ints_sN = intensities.strides[0];
                index_t ints_sW = intensities.strides[1];
                index_t orr_sN = orientation.strides[0];
                index_t orr_sR = orientation.strides[1];
                index_t orr_sC = orientation.strides[2];
                index_t pos_sN = positions.strides[0];
                index_t pos_sW = positions.strides[1];
                index_t pos_sCoor = positions.strides[2];
                index_t shift_sN = shift.strides[0];
                index_t shift_sCoor = shift.strides[1];

                index_t gOut_sN = grad_output.strides[0];
                index_t gOut_sH = grad_output.strides[1];
                index_t gOut_sW = grad_output.strides[2];
                index_t gInts_sN = grad_intensities.strides[0];
                index_t gInts_sW = grad_intensities.strides[1];
                index_t gPos_sN = grad_positions.strides[0];
                index_t gPos_sW = grad_positions.strides[1];
                index_t gPos_sCoor = grad_positions.strides[2];



                CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
                    const index_t w = index % pos_W;
                    const index_t n = index / (pos_W);
                    const auto pos_offset = n * pos_sN + w * pos_sW;
                    // get the corresponding output x, y, z co-ordinates from grid
                    // 1st the coordinates saved in grid, i.e. in [-1, 1]
                    scalar_t ix = positions.data[pos_offset] + shift.data[n * shift_sN];
                    scalar_t iy = positions.data[pos_offset + pos_sCoor] + shift.data[n * shift_sN + shift_sCoor];
                    scalar_t iz = positions.data[pos_offset + 2 * pos_sCoor] + shift.data[n * shift_sN + 2 * shift_sCoor];
                    scalar_t ox, oy, oz;
                    ox = ix;
                    oy = iy;
                    oz = iz;
                    //rotate coordinates
                    scalar_t tIx = ix;
                    scalar_t tIy = iy;
                    scalar_t tIz = iz;
                    matMult(orientation.data + n * orr_sN, orr_sC, orr_sR, tIx, tIy, tIz, &ix, &iy, &iz);

                    // 2nd, unnormalized coordinates in [0, outsize-1]
                    scalar_t gix_mult, giy_mult, giz_mult;
                    ix = atoms_to_grid_compute_source_index_set_grad(ix, x, GridSamplerPadding::Zeros, true, &gix_mult);
                    iy = atoms_to_grid_compute_source_index_set_grad(iy, y, GridSamplerPadding::Zeros, true, &giy_mult);
                    iz = atoms_to_grid_compute_source_index_set_grad(iz, z, GridSamplerPadding::Zeros, true, &giz_mult);


                    if (!within_bounds_2d(ix, iy, x, y)) {
                        continue;
                    }
                    if (true) {//interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get corner pixel values from (x, y, z)
                        index_t ix_nw = static_cast<index_t>((ix));
                        index_t iy_nw = static_cast<index_t>((iy));

                        index_t ix_ne = ix_nw + 1;
                        index_t iy_ne = iy_nw;

                        index_t ix_sw = ix_nw;
                        index_t iy_sw = iy_nw + 1;

                        index_t ix_se = ix_nw + 1;
                        index_t iy_se = iy_nw + 1;

                        // get surfaces to each neighbor on cartesian grid:
                        scalar_t nw = (ix_se - ix) * (iy_se - iy);
                        scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                        scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                        scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                        //intensity value at current grid position
                        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0), gi = static_cast<scalar_t>(0);
                        auto gInp_ptr_NCDHW = grad_intensities.data + n * gInts_sN + w * gInts_sW;
                        auto ints_val_NCDHW = *(intensities.data + n * ints_sN + w * ints_sW);
                        auto gOut_ptr_NC = grad_output.data + n * gOut_sN;
                        // calculate grad_grid
                        if (within_bounds_2d(iy_nw, ix_nw, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iy_nw, ix_nw, gOut_sH, gOut_sW);
                            gi += nw * gOut;
                            gix -= ints_val_NCDHW * (iy_se - iy) * gOut;
                            giy -= ints_val_NCDHW * (ix_se - ix) * gOut;
                        }
                        if (within_bounds_2d(iy_ne, ix_ne, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iy_ne, ix_ne, gOut_sH, gOut_sW);
                            gi += ne * gOut;
                            gix += ints_val_NCDHW * (iy_sw - iy) * gOut;
                            giy -= ints_val_NCDHW * (ix - ix_sw) * gOut;
                        }
                        if (within_bounds_2d(iy_sw, ix_sw, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iy_sw, ix_sw, gOut_sH, gOut_sW);
                            gi += sw * gOut;
                            gix -= ints_val_NCDHW * (iy - iy_ne) * gOut;
                            giy += ints_val_NCDHW * (ix_ne - ix) * gOut;
                        }
                        if (within_bounds_2d(iy_se, ix_se, gOut_H, gOut_W)) {
                            auto gOut = unsafe_access(gOut_ptr_NC, iy_se, ix_se, gOut_sH, gOut_sW);
                            gi += se * gOut;
                            gix += ints_val_NCDHW * (iy - iy_nw) * gOut;
                            giy += ints_val_NCDHW * (ix - ix_nw) * gOut;
                        }

                        *(gInp_ptr_NCDHW) = gi;
                        tIx = gix * gix_mult;
                        tIy = giy * giy_mult;
                        tIz = giz * giz_mult;
                        safe_matGrad(grad_orientations.data + n * orr_sN, orr_sC, orr_sR, ox, oy, oz, tIx, tIy, tIz);

                        matMultT(orientation.data + n * orr_sN, orr_sC, orr_sR, tIx, tIy, tIz, &gix, &giy, &giz);
                        grad_positions.data[pos_offset] = gix;
                        grad_positions.data[pos_offset + pos_sCoor] = giy;
                        grad_positions.data[pos_offset + 2 * pos_sCoor] = giz;
                    }

                }
            }

            template <typename scalar_t, typename index_t>
            __global__ void projectAtoms_kernel(
                const index_t nthreads,
                TensorInfo<scalar_t, index_t> positions,
                TensorInfo<scalar_t, index_t> intensities,
                TensorInfo<scalar_t, index_t> orientation,
                TensorInfo<scalar_t, index_t> shift,
                TensorInfo<scalar_t, index_t> output,
                int64_t x, int64_t y, int64_t z)
            {

                index_t out_H = output.sizes[1];
                index_t out_W = output.sizes[2];
                index_t pos_N = positions.sizes[0];
                index_t pos_W = positions.sizes[1];


                index_t ints_sN = intensities.strides[0];
                index_t ints_sW = intensities.strides[1];
                index_t orr_sN = orientation.strides[0];
                index_t orr_sR = orientation.strides[1];
                index_t orr_sC = orientation.strides[2];
                index_t out_sN = output.strides[0];
                index_t out_sH = output.strides[1];
                index_t out_sW = output.strides[2];
                index_t pos_sN = positions.strides[0];
                index_t pos_sW = positions.strides[1];
                index_t pos_sCoor = positions.strides[2];
                index_t shift_sN = shift.strides[0];
                index_t shift_sCoor = shift.strides[1];

                CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
                    const index_t w = index % pos_W;
                    const index_t n = index / (pos_W);
                    const auto pos_offset = n * pos_sN + w * pos_sW;

                    // get the corresponding output x, y, z co-ordinates from grid
                    // 1st the coordinates saved in grid, i.e. in [-1, 1]
                    scalar_t ix = positions.data[pos_offset] + shift.data[n * shift_sN];
                    scalar_t iy = positions.data[pos_offset + pos_sCoor] + shift.data[n * shift_sN + shift_sCoor];
                    scalar_t iz = positions.data[pos_offset + 2 * pos_sCoor] + shift.data[n * shift_sN + 2 * shift_sCoor];

                    // 2nd, unnormalized coordinates in [0, outsize-1]
                    ix = atoms_to_grid_compute_source_index(ix, x, GridSamplerPadding::Zeros, true);
                    iy = atoms_to_grid_compute_source_index(iy, y, GridSamplerPadding::Zeros, true);
                    iz = atoms_to_grid_compute_source_index(iz, z, GridSamplerPadding::Zeros, true);
                    scalar_t tIx = ix - x / 2;
                    scalar_t tIy = iy - y / 2;
                    scalar_t tIz = iz - z / 2;
                    matMult(orientation.data + n * orr_sN, orr_sC, orr_sR, tIx, tIy, tIz, &ix, &iy, &iz);
                    ix += x / 2;
                    iy += y / 2;
                    iz += z / 2;

                    // get corner pixel values from (x, y, z)
                    index_t ix_nw = static_cast<index_t>((ix));
                    index_t iy_nw = static_cast<index_t>((iy));

                    index_t ix_ne = ix_nw + 1;
                    index_t iy_ne = iy_nw;

                    index_t ix_sw = ix_nw;
                    index_t iy_sw = iy_nw + 1;

                    index_t ix_se = ix_nw + 1;
                    index_t iy_se = iy_nw + 1;

                    // get surfaces to each neighbor on cartesian grid:
                    scalar_t nw = (ix_se - ix) * (iy_se - iy);
                    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                    //intensity value at current grid position
                    scalar_t ints_val_NCHW = *(intensities.data + n * ints_sN + w * ints_sW);
                    scalar_t* out_ptr_NC = output.data + n * out_sN;

                    // calculate bilinear weighted pixel value and set output pixel
                    safe_add_2d(out_ptr_NC, iy_nw, ix_nw, out_sH, out_sW, out_H, out_W, ints_val_NCHW * nw);
                    safe_add_2d(out_ptr_NC, iy_ne, ix_ne, out_sH, out_sW, out_H, out_W, ints_val_NCHW * ne);
                    safe_add_2d(out_ptr_NC, iy_sw, ix_sw, out_sH, out_sW, out_H, out_W, ints_val_NCHW * sw);
                    safe_add_2d(out_ptr_NC, iy_se, ix_se, out_sH, out_sW, out_H, out_W, ints_val_NCHW * se);
                }
            }

            template <typename scalar_t, typename index_t>
            __global__ void grid_sampler_and_project_3d_kernel(
                const index_t nthreads,
                TensorInfo<scalar_t, index_t> input,
                TensorInfo<scalar_t, index_t> grid,
                TensorInfo<scalar_t, index_t> output,
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {

                index_t C = input.sizes[1];
                index_t inp_D = input.sizes[2];
                index_t inp_H = input.sizes[3];
                index_t inp_W = input.sizes[4];
                index_t grid_D = grid.sizes[1];
                index_t out_H = grid.sizes[2];
                index_t out_W = grid.sizes[3];
                //index_t inp_sN = input.strides[0];
                index_t inp_sC = input.strides[1];
                index_t inp_sD = input.strides[2];
                index_t inp_sH = input.strides[3];
                index_t inp_sW = input.strides[4];
                index_t grid_sN = grid.strides[0];
                index_t grid_sD = grid.strides[1];
                index_t grid_sH = grid.strides[2];
                index_t grid_sW = grid.strides[3];
                index_t grid_sCoor = grid.strides[4];
                index_t out_sN = output.strides[0];
                index_t out_sC = output.strides[1];
                index_t out_sH = output.strides[2];
                index_t out_sW = output.strides[3];

                CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
                    const index_t w = index % out_W;
                    const index_t h = (index / out_W) % out_H;
                    //const index_t d = (index / (out_H * out_W)) % out_D;

                    const index_t n = index / (out_H * out_W);

                    //This prevents atomic add operations further down
                    for (index_t d = 0; d < grid_D; d++) {
                        const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

                        // get the corresponding input x, y, z co-ordinates from grid
                        scalar_t ix = grid.data[grid_offset];
                        scalar_t iy = grid.data[grid_offset + grid_sCoor];
                        scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

                        ix = atoms_to_grid_compute_source_index(ix, inp_W, padding_mode, align_corners);
                        iy = atoms_to_grid_compute_source_index(iy, inp_H, padding_mode, align_corners);
                        iz = atoms_to_grid_compute_source_index(iz, inp_D, padding_mode, align_corners);

                        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {


                            // get corner pixel values from (x, y, z)
                            // for 4d, we used north-east-south-west
                            // for 5d, we add top-bottom
                            index_t ix_tnw = static_cast<index_t>(::floor(ix));
                            index_t iy_tnw = static_cast<index_t>(::floor(iy));
                            index_t iz_tnw = static_cast<index_t>(::floor(iz));

                            index_t ix_tne = ix_tnw + 1;
                            index_t iy_tne = iy_tnw;
                            index_t iz_tne = iz_tnw;

                            index_t ix_tsw = ix_tnw;
                            index_t iy_tsw = iy_tnw + 1;
                            index_t iz_tsw = iz_tnw;

                            index_t ix_tse = ix_tnw + 1;
                            index_t iy_tse = iy_tnw + 1;
                            index_t iz_tse = iz_tnw;

                            index_t ix_bnw = ix_tnw;
                            index_t iy_bnw = iy_tnw;
                            index_t iz_bnw = iz_tnw + 1;

                            index_t ix_bne = ix_tnw + 1;
                            index_t iy_bne = iy_tnw;
                            index_t iz_bne = iz_tnw + 1;

                            index_t ix_bsw = ix_tnw;
                            index_t iy_bsw = iy_tnw + 1;
                            index_t iz_bsw = iz_tnw + 1;

                            index_t ix_bse = ix_tnw + 1;
                            index_t iy_bse = iy_tnw + 1;
                            index_t iz_bse = iz_tnw + 1;

                            // get surfaces to each neighbor:
                            scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
                            scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
                            scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
                            scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
                            scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
                            scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
                            scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
                            scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

                            //We are reading from the same volume, no matter which batch
                            auto inp_ptr_NC = input.data;
                            auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;

                            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                                //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                                // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                                // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                                // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                                //*out_ptr_NCHW = static_cast<scalar_t>(0);
                                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
                                }
                                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
                                }
                                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
                                }
                                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
                                }
                                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
                                }
                                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
                                }
                                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
                                }
                                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;

                                }

                            }
                        }
                    }

                }
            }

            template <typename scalar_t, typename index_t>
            __global__ void grid_sampler_and_project_3d_backward_kernel(
                    const index_t nthreads,
                    TensorInfo<scalar_t, index_t> grad_output,
                    TensorInfo<scalar_t, index_t> input,
                    TensorInfo<scalar_t, index_t> grid,
                    TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros
                    TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
                    const GridSamplerInterpolation interpolation_mode,
                    const GridSamplerPadding padding_mode,
                    bool align_corners) {

                index_t C = input.sizes[1];
                index_t inp_D = input.sizes[2];
                index_t inp_H = input.sizes[3];
                index_t inp_W = input.sizes[4];
                index_t grid_D = grid.sizes[1];
                index_t out_H = grid.sizes[2];
                index_t out_W = grid.sizes[3];
                index_t inp_sN = input.strides[0];
                index_t inp_sC = input.strides[1];
                index_t inp_sD = input.strides[2];
                index_t inp_sH = input.strides[3];
                index_t inp_sW = input.strides[4];
                index_t grid_sN = grid.strides[0];
                index_t grid_sD = grid.strides[1];
                index_t grid_sH = grid.strides[2];
                index_t grid_sW = grid.strides[3];
                index_t grid_sCoor = grid.strides[4];
                index_t gOut_sN = grad_output.strides[0];
                index_t gOut_sC = grad_output.strides[1];
                //index_t gOut_sD = grad_output.strides[2];
                index_t gOut_sH = grad_output.strides[2];
                index_t gOut_sW = grad_output.strides[3];
                //index_t gInp_sN = grad_input.strides[0];
                index_t gInp_sC = grad_input.strides[1];
                index_t gInp_sD = grad_input.strides[2];
                index_t gInp_sH = grad_input.strides[3];
                index_t gInp_sW = grad_input.strides[4];
                index_t gGrid_sW = grad_grid.strides[3];

                CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
                    const index_t w = index % out_W;
                    const index_t h = (index / out_W) % out_H;
                    //const index_t d = (index / (out_H * out_W)) % grid_D;
                    const index_t n = index / (out_H * out_W);
                    for (index_t d = 0; d < grid_D; d++) {
                        const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

                        // get the corresponding input x, y, z co-ordinates from grid
                        scalar_t ix = grid.data[grid_offset];
                        scalar_t iy = grid.data[grid_offset + grid_sCoor];
                        scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

                        // multipliers for gradients on ix, iy, and iz
                        scalar_t gix_mult, giy_mult, giz_mult;
                        ix = atoms_to_grid_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
                        iy = atoms_to_grid_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
                        iz = atoms_to_grid_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

                        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                            // get corner pixel values from (x, y, z)
                            // for 4d, we used north-east-south-west
                            // for 5d, we add top-bottom
                            index_t ix_tnw = static_cast<index_t>(::floor(ix));
                            index_t iy_tnw = static_cast<index_t>(::floor(iy));
                            index_t iz_tnw = static_cast<index_t>(::floor(iz));

                            index_t ix_tne = ix_tnw + 1;
                            index_t iy_tne = iy_tnw;
                            index_t iz_tne = iz_tnw;

                            index_t ix_tsw = ix_tnw;
                            index_t iy_tsw = iy_tnw + 1;
                            index_t iz_tsw = iz_tnw;

                            index_t ix_tse = ix_tnw + 1;
                            index_t iy_tse = iy_tnw + 1;
                            index_t iz_tse = iz_tnw;

                            index_t ix_bnw = ix_tnw;
                            index_t iy_bnw = iy_tnw;
                            index_t iz_bnw = iz_tnw + 1;

                            index_t ix_bne = ix_tnw + 1;
                            index_t iy_bne = iy_tnw;
                            index_t iz_bne = iz_tnw + 1;

                            index_t ix_bsw = ix_tnw;
                            index_t iy_bsw = iy_tnw + 1;
                            index_t iz_bsw = iz_tnw + 1;

                            index_t ix_bse = ix_tnw + 1;
                            index_t iy_bse = iy_tnw + 1;
                            index_t iz_bse = iz_tnw + 1;

                            // get surfaces to each neighbor:
                            scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
                            scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
                            scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
                            scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
                            scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
                            scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
                            scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
                            scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

                            scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
                            scalar_t* gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                            // Always use the same volume to reduce memory imprint
                            scalar_t* gInp_ptr_NC = grad_input.data;
                            scalar_t* inp_ptr_NC = input.data;
                            // calculate bilinear weighted pixel value and set output pixel
                            for (index_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                                scalar_t gOut = *gOut_ptr_NCHW;

                                // calculate and set grad_input
                                safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
                                safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

                                // calculate grad_grid
                                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                                    scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
                                    gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
                                    giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
                                    giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
                                }
                                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                                    scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
                                    gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
                                    giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
                                    giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
                                }
                                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                                    scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
                                    gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
                                    giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
                                    giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
                                }
                                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                                    scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
                                    gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
                                    giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
                                    giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
                                }
                                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                                    scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
                                    gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
                                    giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
                                    giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
                                }
                                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                                    scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
                                    gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
                                    giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
                                    giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
                                }
                                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                                    scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
                                    gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
                                    giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
                                    giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
                                }
                                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                                    scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
                                    gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
                                    giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
                                    giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
                                }
                            }

                            // assuming grad_grid is contiguous
                            // thus we can
                            //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
                            //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
                            scalar_t* gGrid_ptr_NDHW = grad_grid.data + n*grid_sN + d*grid_sD + h*grid_sH + w * gGrid_sW;
                            gGrid_ptr_NDHW[0] = gix_mult * gix;
                            gGrid_ptr_NDHW[1] = giy_mult * giy;
                            gGrid_ptr_NDHW[2] = giz_mult * giz;
                        }

                    }
                }
            }
            
            // No shape checking needed here. See # NOTE [ atoms_to_grid Native Functions ].
            Tensor atoms_to_grid_3d_cuda(const Tensor& intensities, const Tensor& positions, const Tensor& orientations, const Tensor& shift, int64_t x, int64_t y, int64_t z) {
                auto N = positions.size(0);
                auto W = positions.size(1);
                auto output = at::zeros({ N, z, y, x }, intensities.options());
                int64_t count = N * W;
                GridSamplerInterpolation interpolation_mode = GridSamplerInterpolation::Bilinear;
                GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;
                bool align_corners = true;
                if (count > 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(intensities.scalar_type(), "atoms_to_grid_3d_cuda", [&] {
                        if (canUse32BitIndexMath(intensities) && canUse32BitIndexMath(positions) &&
                            canUse32BitIndexMath(output)) {
                            atoms_to_grid_3d_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(intensities),
                                    getTensorInfo<scalar_t, int>(positions),
                                    getTensorInfo<scalar_t, int>(orientations),
                                    getTensorInfo<scalar_t, int>(shift),
                                    getTensorInfo<scalar_t, int>(output),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        else {
                            atoms_to_grid_3d_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(intensities),
                                    getTensorInfo<scalar_t, int64_t>(positions),
                                    getTensorInfo<scalar_t, int64_t>(orientations),
                                    getTensorInfo<scalar_t, int64_t>(shift),
                                    getTensorInfo<scalar_t, int64_t>(output),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        });
                }
                return output;
            }

            // No shape checking needed here. See # NOTE [ atoms_to_grid Native Functions ].
            std::tuple<Tensor, Tensor, Tensor, Tensor>
                atoms_to_grid_3d_backward_cuda(const Tensor& grad_output, const Tensor& intensities,
                    const Tensor& positions, const Tensor& orientations, const Tensor& shift) {
                // See Note [Writing Nondeterministic Operations]
                // Nondeterministic because of atomicAdd usage
                globalContext().alertNotDeterministic("atoms_to_grid_3d_backward_cuda");
                auto N = positions.size(0);
                auto W = positions.size(1);
                int64_t count = N * W;
                GridSamplerInterpolation interpolation_mode = GridSamplerInterpolation::Bilinear;
                GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;
                bool align_corners = true;
                auto grad_intensities = at::zeros_like(intensities, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                auto grad_positions = at::empty_like(positions, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                auto grad_orientations = at::zeros_like(orientations, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                if (count > 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(intensities.scalar_type(), "atoms_to_grid_3d_backward_cuda", [&] {

                        if (canUse32BitIndexMath(intensities) && canUse32BitIndexMath(positions) &&
                            canUse32BitIndexMath(grad_output)) {

                            atoms_to_grid_3d_backward_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad_output),
                                    getTensorInfo<scalar_t, int>(intensities),
                                    getTensorInfo<scalar_t, int>(positions),
                                    getTensorInfo<scalar_t, int>(orientations),
                                    getTensorInfo<scalar_t, int>(shift),
                                    getTensorInfo<scalar_t, int>(grad_intensities),
                                    getTensorInfo<scalar_t, int>(grad_positions),
                                    getTensorInfo<scalar_t, int>(grad_orientations),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        else {
                            atoms_to_grid_3d_backward_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad_output),
                                    getTensorInfo<scalar_t, int64_t>(intensities),
                                    getTensorInfo<scalar_t, int64_t>(positions),
                                    getTensorInfo<scalar_t, int64_t>(orientations),
                                    getTensorInfo<scalar_t, int64_t>(shift),
                                    getTensorInfo<scalar_t, int64_t>(grad_intensities),
                                    getTensorInfo<scalar_t, int64_t>(grad_positions),
                                    getTensorInfo<scalar_t, int64_t>(grad_orientations),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        });
                }
                auto grad_shift = grad_positions.sum(1);
                return std::make_tuple(grad_intensities, grad_positions, grad_orientations, grad_shift);
            }


            Tensor projectAtoms(const Tensor& intensities, const Tensor& positions, const Tensor& orientation, const Tensor& shift, int64_t x, int64_t y, int64_t z) {
                auto N = positions.size(0);
                auto W = positions.size(1);

                auto output = at::zeros({ N, y, x}, positions.options());
                int64_t count = N*W;
                bool align_corners = true;
                if (count > 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(intensities.scalar_type(), "projectAtoms", [&] {

                        if (canUse32BitIndexMath(positions) &&
                            canUse32BitIndexMath(intensities) && canUse32BitIndexMath(orientation) &&
                            canUse32BitIndexMath(output)) {
                            projectAtoms_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(positions),
                                    getTensorInfo<scalar_t, int>(intensities),
                                    getTensorInfo<scalar_t, int>(orientation),
                                    getTensorInfo<scalar_t, int>(shift),
                                    getTensorInfo<scalar_t, int>(output), x, y, z);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        else {
                            projectAtoms_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(positions),
                                    getTensorInfo<scalar_t, int64_t>(intensities),
                                    getTensorInfo<scalar_t, int64_t>(orientation),
                                    getTensorInfo<scalar_t, int64_t>(shift),
                                    getTensorInfo<scalar_t, int64_t>(output), x, y, z);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        });
                }
                return output;// .sum(1, false);
            }
            
            std::tuple<Tensor, Tensor, Tensor, Tensor>
                projectAtoms_backward_cuda(const Tensor& grad_output, const Tensor& intensities, const Tensor& positions, const Tensor& orientation, const Tensor& shift, int64_t x, int64_t y, int64_t z) {
                auto N = positions.size(0);
                auto W = positions.size(1);
                int64_t count = N * W;
                bool align_corners = true;
                auto grad_intensities = at::zeros_like(intensities, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                auto grad_positions = at::empty_like(positions, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                auto grad_orientation = at::zeros_like(orientation, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

                if (count > 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(intensities.scalar_type(), "projectAtoms", [&] {
                        if (canUse32BitIndexMath(positions) &&
                            canUse32BitIndexMath(intensities) && canUse32BitIndexMath(orientation) &&
                            canUse32BitIndexMath(grad_output)) {
                            projectAtoms_backwards_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(positions),
                                    getTensorInfo<scalar_t, int>(intensities),
                                    getTensorInfo<scalar_t, int>(orientation),
                                    getTensorInfo<scalar_t, int>(shift),
                                    getTensorInfo<scalar_t, int>(grad_output),
                                    getTensorInfo<scalar_t, int>(grad_positions),
                                    getTensorInfo<scalar_t, int>(grad_intensities),
                                    getTensorInfo<scalar_t, int>(grad_orientation), x, y, z);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        else {

                            projectAtoms_backwards_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(positions),
                                    getTensorInfo<scalar_t, int64_t>(intensities),
                                    getTensorInfo<scalar_t, int64_t>(orientation),
                                    getTensorInfo<scalar_t, int64_t>(shift),
                                    getTensorInfo<scalar_t, int64_t>(grad_output),
                                    getTensorInfo<scalar_t, int64_t>(grad_positions),
                                    getTensorInfo<scalar_t, int64_t>(grad_intensities),
                                    getTensorInfo<scalar_t, int64_t>(grad_orientation), x, y, z);
                            AT_CUDA_CHECK(cudaGetLastError());
                        }
                        });
                }
                //Lazy Way: Shift gradient is just sum over all atoms. In the kernel, we would need atomic Add operation for this
                auto grad_shift = grad_positions.sum(1);
                return std::make_tuple(grad_intensities, grad_positions, grad_orientation, grad_shift);  
            }

            // No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
            Tensor grid_sampler_and_project_3d_cuda(const Tensor& input, const Tensor& grid,
                int64_t interpolation_mode, int64_t padding_mode,
                bool align_corners) {
                auto N = grid.size(0);
                //auto D = grid.size(1);
                auto H = grid.size(2);
                auto W = grid.size(3);
                auto output = at::zeros({ N, input.size(1), H, W }, input.options());
                int64_t count = N * H * W;

                auto inpDims = input.sizes().vec();
                auto gridDims = grid.sizes().vec();
                auto outDim = output.sizes().vec();

                if (count > 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_and_project_3d_cuda", [&] {
                        if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
                            canUse32BitIndexMath(output)) {
                            grid_sampler_and_project_3d_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<scalar_t, int>(grid),
                                    getTensorInfo<scalar_t, int>(output),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                        }
                        else {
                            grid_sampler_and_project_3d_kernel<scalar_t>
                                << <1, 1, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<scalar_t, int64_t>(grid),
                                    getTensorInfo<scalar_t, int64_t>(output),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                        }
                        });
                }
                return output;
            }

            // No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
            std::tuple<Tensor, Tensor>
                grid_sampler_and_project_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                    bool align_corners) {
                // See Note [Writing Nondeterministic Operations]
                // Nondeterministic because of atomicAdd usage
                globalContext().alertNotDeterministic("grid_sampler_3d_backward_cuda");
                auto N = input.size(0);
                auto D = grid.size(1);
                auto H = grid.size(2);
                auto W = grid.size(3);
                auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                int64_t count = N * H * W;
                if (count > 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_and_project_3d_backward_cuda", [&] {
                        if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
                            canUse32BitIndexMath(grad_output)) {
                            grid_sampler_and_project_3d_backward_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad_output),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<scalar_t, int>(grid),
                                    getTensorInfo<scalar_t, int>(grad_input),
                                    getTensorInfo<scalar_t, int>(grad_grid),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                        }
                        else {
                            grid_sampler_and_project_3d_backward_kernel<scalar_t>
                                << <MY_CUDA_GET_BLOCKS(count), MY_CUDA_MAX_THREADS, 0, at::cuda::getCurrentCUDAStream() >> > (
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad_output),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<scalar_t, int64_t>(grid),
                                    getTensorInfo<scalar_t, int64_t>(grad_input),
                                    getTensorInfo<scalar_t, int64_t>(grad_grid),
                                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                                    static_cast<GridSamplerPadding>(padding_mode),
                                    align_corners);
                        }
                        });
                }
                return std::make_tuple(grad_input, grad_grid);
            }

        }  // namespace
    }
}
    
