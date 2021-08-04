#include "CustomModules.h"
#include "CustomOperatorsAutograd.h"


//#include <torch/torch.h>
//#include <ATen/ATen.h>
#include "CustomOperatorsBackend.cuh"


#include <ATen/cuda/detail/IndexUtils.cuh>
#include <C10/macros/Macros.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include "CustomOperatorsBackend.cuh"
#include "ComplexGridSampler.h"



namespace at {
    namespace native {

        using namespace at::cuda::detail;
        namespace ComplexGridSampler {

           
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
            static __forceinline__
                scalar_t complex_grid_sampler_cpu_unnormalize(scalar_t coord, int size, bool align_corners) {
                if (align_corners) {
                    // unnormalize coord from [-1, 1] to [0, size - 1]
                    return ((coord + 1.f) / 2) * (size - 1);
                }
                else {
                    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                    return ((coord + 1.f) * size - 1) / 2;
                }
            }

            // complex_grid_sampler_cpu_unnormalize_set_grad works the same as complex_grid_sampler_cpu_unnormalize
            // except that it also returns the `d output / d input` via pointer argument
            // `grad_in`.
            // This is useful in the backward pass of complex_grid_sampler_cpu_.
            template <typename scalar_t>
            static __forceinline__ 
                scalar_t complex_grid_sampler_cpu_unnormalize_set_grad(scalar_t coord, int size,
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
            static __forceinline__ 
                scalar_t clip_coordinates(scalar_t in, int clip_limit) {
                return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
            }

            // clip_coordinates_set_grad works similarly to clip_coordinates except that
            // it also returns the `d output / d input` via pointer argument `grad_in`.
            // This is useful in the backward pass of complex_grid_sampler_cpu_.
            template <typename scalar_t>
            static __forceinline__ 
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
            static __forceinline__ 
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
            // This is useful in the backward pass of complex_grid_sampler_cpu_.
            template <typename scalar_t>
            static __forceinline__ 
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
            static __forceinline__ 
                scalar_t safe_downgrade_to_int_range(scalar_t x) {
                // -100.0 does not have special meaning. This is just to make sure
                // it's not within_bounds_2d or within_bounds_3d, and does not cause
                // undefined behavior. See #35506.
                if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
                    return static_cast<scalar_t>(-100.0);
                return x;
            }

            // Computes the pixel source index value for a grid coordinate
            template <typename scalar_t>
            static __forceinline__ 
                scalar_t complex_grid_sampler_cpu_compute_source_index(
                    scalar_t coord,
                    int size,
                    GridSamplerPadding padding_mode,
                    bool align_corners) {
                coord = complex_grid_sampler_cpu_unnormalize(coord, size, align_corners);
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

            // complex_grid_sampler_cpu_compute_source_index_set_grad works similarly to
            // complex_grid_sampler_cpu_compute_source_index except that it also returns the
            // `d output / d input` via pointer argument `grad_in`.
            // This is useful in the backward pass of complex_grid_sampler_cpu_.
            template <typename scalar_t>
            static __forceinline__ 
                scalar_t complex_grid_sampler_cpu_compute_source_index_set_grad(
                    scalar_t coord,
                    int size,
                    GridSamplerPadding padding_mode,
                    bool align_corners,
                    scalar_t* grad_in) {
                scalar_t grad_clip, grad_refl;
                coord = complex_grid_sampler_cpu_unnormalize_set_grad(coord, size, align_corners, grad_in);
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

            static __forceinline__ 
                bool within_bounds_2d(int h, int w, int H, int W) {
                return h >= 0 && h < H&& w >= 0 && w < W;
            }

            static __forceinline__ 
                bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
                return d >= 0 && d < D&& h >= 0 && h < H&& w >= 0 && w < W;
            }

            template<typename scalar_t>
            static __forceinline__
                void cpuAtomicAdd(scalar_t* data, scalar_t delta) {
                *(data) += delta;
            }

            template<typename scalar_t>
            static __forceinline__ 
                void safe_add_2d(scalar_t* data, int h, int w,
                    int sH, int sW, int H, int W,
                    scalar_t delta) {
                if (within_bounds_2d(h, w, H, W)) {
                    cpuAtomicAdd(data + h * sH + w * sW, delta);
                }
            }

            template<typename scalar_t>
            static __forceinline__ 
                void safe_add_3d(scalar_t* data, int d, int h, int w,
                    int sD, int sH, int sW, int D, int H, int W,
                    scalar_t delta) {
                if (within_bounds_3d(d, h, w, D, H, W)) {
                    cpuAtomicAdd(data + d * sD + h * sH + w * sW, delta);
                }
            }

            template <typename scalar_t>
             void complex_grid_sampler_cpu_2d_kernel(
                const int nthreads,
                TensorInfo<c10::complex<float>, int> input,
                TensorInfo<scalar_t, int> grid,
                TensorInfo<c10::complex<float>, int> output,
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {
                int C = input.sizes[1];
                int inp_H = input.sizes[2];
                int inp_W = input.sizes[3];
                int out_H = grid.sizes[1];
                int out_W = grid.sizes[2];
                int inp_sN = input.strides[0];
                int inp_sC = input.strides[1];
                int inp_sH = input.strides[2];
                int inp_sW = input.strides[3];
                int grid_sN = grid.strides[0];
                int grid_sH = grid.strides[1];
                int grid_sW = grid.strides[2];
                int grid_sCoor = grid.strides[3];
                int out_sN = output.strides[0];
                int out_sC = output.strides[1];
                int out_sH = output.strides[2];
                int out_sW = output.strides[3];

                for (size_t index = 0; index < nthreads; index++)
                {

                    const int w = index % out_W;
                    const int h = (index / out_W) % out_H;
                    const int n = index / (out_H * out_W);
                    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

                    // get the corresponding input x, y co-ordinates from grid
                    scalar_t ix = grid.data[grid_offset];
                    scalar_t iy = grid.data[grid_offset + grid_sCoor];

                    ix = complex_grid_sampler_cpu_compute_source_index(ix, inp_W, padding_mode, align_corners);
                    iy = complex_grid_sampler_cpu_compute_source_index(iy, inp_H, padding_mode, align_corners);



                    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get NE, NW, SE, SW pixel values from (x, y)
                        int ix_nw = static_cast<int>(::floor(ix));
                        int iy_nw = static_cast<int>(::floor(iy));
                        int ix_ne = ix_nw + 1;
                        int iy_ne = iy_nw;
                        int ix_sw = ix_nw;
                        int iy_sw = iy_nw + 1;
                        int ix_se = ix_nw + 1;
                        int iy_se = iy_nw + 1;

                        // get surfaces to each neighbor:
                        float nw = (ix_se - ix) * (iy_se - iy);
                        float ne = (ix - ix_sw) * (iy_sw - iy);
                        float sw = (ix_ne - ix) * (iy - iy_ne);
                        float se = (ix - ix_nw) * (iy - iy_nw);

                        // calculate bilinear weighted pixel value and set output pixel
                        auto inp_ptr_NC = input.data + n * inp_sN;
                        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                            *out_ptr_NCHW = static_cast<c10::complex<float>>(0);
                            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                                *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                            }
                            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                                *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                            }
                            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                                *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                            }
                            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                                *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                            }
                        }
                    }
                    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                        int ix_nearest = static_cast<int>(::round(ix));
                        int iy_nearest = static_cast<int>(::round(iy));

                        // assign nearest neighor pixel value to output pixel
                        auto inp_ptr_NC = input.data + n * inp_sN;
                        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                            if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                                *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                            }
                            else {
                                *out_ptr_NCHW = static_cast<c10::complex<float>>(0);
                            }
                        }
                    }
                }
            }

            template <typename scalar_t>
             void complex_grid_sampler_cpu_3d_kernel(
                const int nthreads,
                TensorInfo<c10::complex<float>, int> input,
                TensorInfo<float, int> grid,
                TensorInfo<c10::complex<float>, int> output,
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {

                int C = input.sizes[1];
                int inp_D = input.sizes[2];
                int inp_H = input.sizes[3];
                int inp_W = input.sizes[4];
                int out_D = grid.sizes[1];
                int out_H = grid.sizes[2];
                int out_W = grid.sizes[3];
                int inp_sN = input.strides[0];
                int inp_sC = input.strides[1];
                int inp_sD = input.strides[2];
                int inp_sH = input.strides[3];
                int inp_sW = input.strides[4];
                int grid_sN = grid.strides[0];
                int grid_sD = grid.strides[1];
                int grid_sH = grid.strides[2];
                int grid_sW = grid.strides[3];
                int grid_sCoor = grid.strides[4];
                int out_sN = output.strides[0];
                int out_sC = output.strides[1];
                int out_sD = output.strides[2];
                int out_sH = output.strides[3];
                int out_sW = output.strides[4];

                for (size_t index = 0; index < nthreads; index++) {
                    const int w = index % out_W;
                    const int h = (index / out_W) % out_H;
                    const int d = (index / (out_H * out_W)) % out_D;
                    const int n = index / (out_D * out_H * out_W);
                    const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

                    // get the corresponding input x, y, z co-ordinates from grid
                    float ix = grid.data[grid_offset];
                    float iy = grid.data[grid_offset + grid_sCoor];
                    float iz = grid.data[grid_offset + 2 * grid_sCoor];

                    ix = complex_grid_sampler_cpu_compute_source_index(ix, inp_W, padding_mode, align_corners);
                    iy = complex_grid_sampler_cpu_compute_source_index(iy, inp_H, padding_mode, align_corners);
                    iz = complex_grid_sampler_cpu_compute_source_index(iz, inp_D, padding_mode, align_corners);

                    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get corner pixel values from (x, y, z)
                        // for 4d, we used north-east-south-west
                        // for 5d, we add top-bottom
                        int ix_tnw = static_cast<int>(::floor(ix));
                        int iy_tnw = static_cast<int>(::floor(iy));
                        int iz_tnw = static_cast<int>(::floor(iz));

                        int ix_tne = ix_tnw + 1;
                        int iy_tne = iy_tnw;
                        int iz_tne = iz_tnw;

                        int ix_tsw = ix_tnw;
                        int iy_tsw = iy_tnw + 1;
                        int iz_tsw = iz_tnw;

                        int ix_tse = ix_tnw + 1;
                        int iy_tse = iy_tnw + 1;
                        int iz_tse = iz_tnw;

                        int ix_bnw = ix_tnw;
                        int iy_bnw = iy_tnw;
                        int iz_bnw = iz_tnw + 1;

                        int ix_bne = ix_tnw + 1;
                        int iy_bne = iy_tnw;
                        int iz_bne = iz_tnw + 1;

                        int ix_bsw = ix_tnw;
                        int iy_bsw = iy_tnw + 1;
                        int iz_bsw = iz_tnw + 1;

                        int ix_bse = ix_tnw + 1;
                        int iy_bse = iy_tnw + 1;
                        int iz_bse = iz_tnw + 1;

                        // get surfaces to each neighbor:
                        scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
                        scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
                        scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
                        scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
                        scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
                        scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
                        scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
                        scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

                        auto inp_ptr_NC = input.data + n * inp_sN;
                        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
                            //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                            // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                            // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                            // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                            *out_ptr_NCDHW = static_cast<c10::complex<float>>(0);
                            if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
                            }
                            if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
                            }
                            if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
                            }
                            if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
                            }
                            if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
                            }
                            if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
                            }
                            if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
                            }
                            if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
                            }
                        }
                    }
                    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                        int ix_nearest = static_cast<int>(::round(ix));
                        int iy_nearest = static_cast<int>(::round(iy));
                        int iz_nearest = static_cast<int>(::round(iz));

                        // assign nearest neighor pixel value to output pixel
                        auto inp_ptr_NC = input.data + n * inp_sN;
                        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
                            if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
                                *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
                            }
                            else {
                                *out_ptr_NCDHW = static_cast<c10::complex<float>>(0);
                            }
                        }
                    }
                }
            }

            template <typename scalar_t>
            void complex_grid_sampler_cpu_2d_backward_kernel(
                const int nthreads,
                TensorInfo<c10::complex<float>, int> grad_output,
                TensorInfo<c10::complex<float>, int> input,
                TensorInfo<scalar_t, int> grid,
                TensorInfo<c10::complex<float>, int> grad_input,  // initialized to zeros
                TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {

                int C = input.sizes[1];
                int inp_H = input.sizes[2];
                int inp_W = input.sizes[3];
                int out_H = grid.sizes[1];
                int out_W = grid.sizes[2];
                int inp_sN = input.strides[0];
                int inp_sC = input.strides[1];
                int inp_sH = input.strides[2];
                int inp_sW = input.strides[3];
                int grid_sN = grid.strides[0];
                int grid_sH = grid.strides[1];
                int grid_sW = grid.strides[2];
                int grid_sCoor = grid.strides[3];
                int gOut_sN = grad_output.strides[0];
                int gOut_sC = grad_output.strides[1];
                int gOut_sH = grad_output.strides[2];
                int gOut_sW = grad_output.strides[3];
                int gInp_sN = grad_input.strides[0];
                int gInp_sC = grad_input.strides[1];
                int gInp_sH = grad_input.strides[2];
                int gInp_sW = grad_input.strides[3];
                int gGrid_sW = grad_grid.strides[2];

                for (size_t index = 0; index < nthreads; index++) {
                    const int w = index % out_W;
                    const int h = (index / out_W) % out_H;
                    const int n = index / (out_H * out_W);
                    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

                    // get the corresponding input x, y co-ordinates from grid
                    scalar_t ix = grid.data[grid_offset];
                    scalar_t iy = grid.data[grid_offset + grid_sCoor];

                    // multipliers for gradients on ix and iy
                    scalar_t gix_mult, giy_mult;
                    ix = complex_grid_sampler_cpu_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
                    iy = complex_grid_sampler_cpu_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);

                    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get NE, NW, SE, SW pixel values from (x, y)
                        int ix_nw = static_cast<int>(::floor(ix));
                        int iy_nw = static_cast<int>(::floor(iy));
                        int ix_ne = ix_nw + 1;
                        int iy_ne = iy_nw;
                        int ix_sw = ix_nw;
                        int iy_sw = iy_nw + 1;
                        int ix_se = ix_nw + 1;
                        int iy_se = iy_nw + 1;

                        // get surfaces to each neighbor:
                        scalar_t nw = (ix_se - ix) * (iy_se - iy);
                        scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                        scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                        scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
                        c10::complex<float>* gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                        c10::complex<float>* gInp_ptr_NC = grad_input.data + n * gInp_sN;
                        c10::complex<float>* inp_ptr_NC = input.data + n * inp_sN;
                        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
                            c10::complex<float> gOut = *gOut_ptr_NCHW;

                            // calculate and set grad_input
                            safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
                            safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
                            safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
                            safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

                            // calculate grad_grid
                            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                                c10::complex<float> nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
                                gix -= (nw_val * (iy_se - iy) * gOut).real_;
                                giy -= (nw_val * (ix_se - ix) * gOut).real_;
                            }
                            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                                c10::complex<float> ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
                                gix += (ne_val * (iy_sw - iy) * gOut).real_;
                                giy -= (ne_val * (ix - ix_sw) * gOut).real_;
                            }
                            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                                c10::complex<float> sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
                                gix -= (sw_val * (iy - iy_ne) * gOut).real_;
                                giy += (sw_val * (ix_ne - ix) * gOut).real_;
                            }
                            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                                c10::complex<float> se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
                                gix += (se_val * (iy - iy_nw) * gOut).real_;
                                giy += (se_val * (ix - ix_nw) * gOut).real_;
                            }
                        }

                        // assuming grad_grid is contiguous
                        // thus we can
                        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
                        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
                        scalar_t* gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
                        gGrid_ptr_NHW[0] = gix_mult * gix;
                        gGrid_ptr_NHW[1] = giy_mult * giy;
                    }
                    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                        int ix_nearest = static_cast<int>(::round(ix));
                        int iy_nearest = static_cast<int>(::round(iy));

                        // assign nearest neighor pixel value to output pixel
                        c10::complex<float>* gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                        c10::complex<float>* gInp_ptr_NC = grad_input.data + n * gInp_sN;
                        for (int c = 0; c < C; ++c, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
                            // calculate and set grad_input
                            safe_add_2d(gInp_ptr_NC, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW);
                        }

                        // assuming grad_grid is contiguous
                        // thus we can
                        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
                        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
                        scalar_t* gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
                        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
                        gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
                    }
                }
            }

            template <typename scalar_t>
            void complex_grid_sampler_cpu_3d_backward_kernel(
                const int nthreads,
                TensorInfo<c10::complex<float>, int> grad_output,
                TensorInfo<c10::complex<float>, int> input,
                TensorInfo<scalar_t, int> grid,
                TensorInfo<c10::complex<float>, int> grad_input,  // initialized to zeros
                TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
                const GridSamplerInterpolation interpolation_mode,
                const GridSamplerPadding padding_mode,
                bool align_corners) {

                int C = input.sizes[1];
                int inp_D = input.sizes[2];
                int inp_H = input.sizes[3];
                int inp_W = input.sizes[4];
                int out_D = grid.sizes[1];
                int out_H = grid.sizes[2];
                int out_W = grid.sizes[3];
                int inp_sN = input.strides[0];
                int inp_sC = input.strides[1];
                int inp_sD = input.strides[2];
                int inp_sH = input.strides[3];
                int inp_sW = input.strides[4];
                int grid_sN = grid.strides[0];
                int grid_sD = grid.strides[1];
                int grid_sH = grid.strides[2];
                int grid_sW = grid.strides[3];
                int grid_sCoor = grid.strides[4];
                int gOut_sN = grad_output.strides[0];
                int gOut_sC = grad_output.strides[1];
                int gOut_sD = grad_output.strides[2];
                int gOut_sH = grad_output.strides[3];
                int gOut_sW = grad_output.strides[4];
                int gInp_sN = grad_input.strides[0];
                int gInp_sC = grad_input.strides[1];
                int gInp_sD = grad_input.strides[2];
                int gInp_sH = grad_input.strides[3];
                int gInp_sW = grad_input.strides[4];
                int gGrid_sW = grad_grid.strides[3];

                for (size_t index = 0; index < nthreads; index++) {
                    const int w = index % out_W;
                    const int h = (index / out_W) % out_H;
                    const int d = (index / (out_H * out_W)) % out_D;
                    const int n = index / (out_D * out_H * out_W);
                    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

                    // get the corresponding input x, y, z co-ordinates from grid
                    scalar_t ix = grid.data[grid_offset];
                    scalar_t iy = grid.data[grid_offset + grid_sCoor];
                    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

                    // multipliers for gradients on ix, iy, and iz
                    scalar_t gix_mult, giy_mult, giz_mult;
                    ix = complex_grid_sampler_cpu_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
                    iy = complex_grid_sampler_cpu_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
                    iz = complex_grid_sampler_cpu_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

                    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get corner pixel values from (x, y, z)
                        // for 4d, we used north-east-south-west
                        // for 5d, we add top-bottom
                        int ix_tnw = static_cast<int>(::floor(ix));
                        int iy_tnw = static_cast<int>(::floor(iy));
                        int iz_tnw = static_cast<int>(::floor(iz));

                        int ix_tne = ix_tnw + 1;
                        int iy_tne = iy_tnw;
                        int iz_tne = iz_tnw;

                        int ix_tsw = ix_tnw;
                        int iy_tsw = iy_tnw + 1;
                        int iz_tsw = iz_tnw;

                        int ix_tse = ix_tnw + 1;
                        int iy_tse = iy_tnw + 1;
                        int iz_tse = iz_tnw;

                        int ix_bnw = ix_tnw;
                        int iy_bnw = iy_tnw;
                        int iz_bnw = iz_tnw + 1;

                        int ix_bne = ix_tnw + 1;
                        int iy_bne = iy_tnw;
                        int iz_bne = iz_tnw + 1;

                        int ix_bsw = ix_tnw;
                        int iy_bsw = iy_tnw + 1;
                        int iz_bsw = iz_tnw + 1;

                        int ix_bse = ix_tnw + 1;
                        int iy_bse = iy_tnw + 1;
                        int iz_bse = iz_tnw + 1;

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
                        c10::complex<float>* gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                        c10::complex<float>* gInp_ptr_NC = grad_input.data + n * gInp_sN;
                        c10::complex<float>* inp_ptr_NC = input.data + n * inp_sN;
                        // calculate bilinear weighted pixel value and set output pixel
                        for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                            c10::complex<float> gOut = *gOut_ptr_NCDHW;

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
                                c10::complex<float> tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
                                gix -= (tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut).real_;
                                giy -= (tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut).real_;
                                giz -= (tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                                c10::complex<float> tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
                                gix += (tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut).real_;
                                giy -= (tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut).real_;
                                giz -= (tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                                c10::complex<float> tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
                                gix -= (tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut).real_;
                                giy += (tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut).real_;
                                giz -= (tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                                c10::complex<float> tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
                                gix += (tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut).real_;
                                giy += (tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut).real_;
                                giz -= (tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                                c10::complex<float> bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
                                gix -= (bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut).real_;
                                giy -= (bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut).real_;
                                giz += (bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                                c10::complex<float> bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
                                gix += (bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut).real_;
                                giy -= (bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut).real_;
                                giz += (bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                                c10::complex<float> bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
                                gix -= (bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut).real_;
                                giy += (bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut).real_;
                                giz += (bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut).real_;
                            }
                            if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                                c10::complex<float> bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
                                gix += (bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut).real_;
                                giy += (bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut).real_;
                                giz += (bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut).real_;
                            }
                        }

                        // assuming grad_grid is contiguous
                        // thus we can
                        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
                        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
                        scalar_t* gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
                        gGrid_ptr_NDHW[0] = gix_mult * gix;
                        gGrid_ptr_NDHW[1] = giy_mult * giy;
                        gGrid_ptr_NDHW[2] = giz_mult * giz;
                    }
                    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                        auto ix_nearest = static_cast<int>(::round(ix));
                        auto iy_nearest = static_cast<int>(::round(iy));
                        auto iz_nearest = static_cast<int>(::round(iz));

                        // assign nearest neighor pixel value to output pixel
                        c10::complex<float>* gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                        c10::complex<float>* gInp_ptr_NC = grad_input.data + n * gInp_sN;
                        for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
                            // calculate and set grad_input
                            safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
                                gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
                        }

                        // assuming grad_grid is contiguous
                        // thus we can
                        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
                        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
                        scalar_t* gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
                        gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
                        gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
                        gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
                    }
                }
            }

            // No shape checking needed here. See # NOTE [ complex_grid_sampler_cpu_ Native Functions ].
            Tensor complex_grid_sampler_cpu_2d(const Tensor& input, const Tensor& grid,
                int interpolation_mode, int padding_mode,
                bool align_corners) {
                auto N = input.size(0);
                auto C = input.size(1);
                auto H = grid.size(1);
                auto W = grid.size(2);
                auto output = at::empty({ N, C, H, W }, input.options());
                int count = N * H * W;
                if (count > 0) {
                    complex_grid_sampler_cpu_2d_kernel<float> (
                            static_cast<int>(count),
                            getTensorInfo<c10::complex<float>, int>(input),
                            getTensorInfo<float, int>(grid),
                            getTensorInfo<c10::complex<float>, int>(output),
                            static_cast<GridSamplerInterpolation>(interpolation_mode),
                            static_cast<GridSamplerPadding>(padding_mode),
                            align_corners);

                }
                return output;
            }

            // No shape checking needed here. See # NOTE [ complex_grid_sampler_cpu_ Native Functions ].
            Tensor complex_grid_sampler_cpu_3d(const Tensor& input, const Tensor& grid,
                int interpolation_mode, int padding_mode,
                bool align_corners, double max_r2) {
                auto N = input.size(0);
                auto D = grid.size(1);
                auto H = grid.size(2);
                auto W = grid.size(3);
                auto output = at::empty({ N, input.size(1), D, H, W }, input.options());
                int count = N * D * H * W;
                std::cout << "input" << std::endl;
                auto foo_a = input.accessor<c10::complex<float>, 5>();
                for (size_t yy = 0; yy < 33; yy++)
                {
                    for (size_t xx = 0; xx < 33; xx++)
                    {
                        std::cout << foo_a[0][0][16][yy][xx] << "\t";
                    }
                    std::cout << std::endl;
                }

                auto cp_input = input.clone();
                foo_a = cp_input.accessor<c10::complex<float>, 5>();
                /*for (size_t zz = 0; zz < input.size(2); zz++)
                {
                    for (size_t yy = 0; yy < input.size(3); yy++)
                    {
                        for (size_t xx = 0; xx < input.size(4); xx++)
                        {
                            if ((xx - input.size(4) / 2) * (xx - input.size(4) / 2) + (yy - input.size(3) / 2) * (yy - input.size(3) / 2) + (zz - input.size(2) / 2) * (zz - input.size(2) / 2) > max_r2)
                                foo_a[0][0][zz][yy][xx] = 0;
                            //else
                                //foo_a[0][0][zz][yy][xx] *= 34;
                        }

                    }
                }*/
                if (count > 0) {
                    complex_grid_sampler_cpu_3d_kernel<float>(
                            static_cast<int>(count),
                            getTensorInfo<c10::complex<float>, int>(cp_input),
                            getTensorInfo<float, int>(grid),
                            getTensorInfo<c10::complex<float>, int>(output),
                            static_cast<GridSamplerInterpolation>(interpolation_mode),
                            static_cast<GridSamplerPadding>(padding_mode),
                            align_corners);


                }
                std::cout << "output" << std::endl;
                auto foo_b = output.accessor<c10::complex<float>, 5>();
                for (size_t yy = 0; yy < 33; yy++)
                {
                    for (size_t xx = 0; xx < 33; xx++)
                    {
                        std::cout << foo_b[0][0][0][yy][xx] << "\t";
                    }
                    std::cout << std::endl;
                }
                return output;
            }

            // No shape checking needed here. See # NOTE [ complex_grid_sampler_cpu_ Native Functions ].
            std::tuple<Tensor, Tensor>
                complex_grid_sampler_cpu_2d_backward(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int interpolation_mode,
                    int padding_mode, bool align_corners) {
                // See Note [Writing Nondeterministic Operations]
                // Nondeterministic because of atomicAdd usage
                globalContext().alertNotDeterministic("grid_sampler_2d_backward_cuda");
                auto N = input.size(0);
                auto H = grid.size(1);
                auto W = grid.size(2);
                auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                int count = N * H * W;
                if (count > 0) {
                    complex_grid_sampler_cpu_2d_backward_kernel<float>(
                            static_cast<int>(count),
                            getTensorInfo<c10::complex<float>, int>(grad_output),
                            getTensorInfo<c10::complex<float>, int>(input),
                            getTensorInfo<float, int>(grid),
                            getTensorInfo<c10::complex<float>, int>(grad_input),
                            getTensorInfo<float, int>(grad_grid),
                            static_cast<GridSamplerInterpolation>(interpolation_mode),
                            static_cast<GridSamplerPadding>(padding_mode),
                            align_corners);
                }
                return std::make_tuple(grad_input, grad_grid);
            }

            // No shape checking needed here. See # NOTE [ complex_grid_sampler_cpu_ Native Functions ].
            std::tuple<Tensor, Tensor>
                complex_grid_sampler_cpu_3d_backward(const Tensor& grad_output, const Tensor& input,
                    const Tensor& grid, int interpolation_mode, int padding_mode,
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
                int count = N * D * H * W;
                if (count > 0) {

                    complex_grid_sampler_cpu_3d_backward_kernel<float>(
                            static_cast<int>(count),
                            getTensorInfo<c10::complex<float>, int>(grad_output),
                            getTensorInfo<c10::complex<float>, int>(input),
                            getTensorInfo<float, int>(grid),
                            getTensorInfo<c10::complex<float>, int>(grad_input),
                            getTensorInfo<float, int>(grad_grid),
                            static_cast<GridSamplerInterpolation>(interpolation_mode),
                            static_cast<GridSamplerPadding>(padding_mode),
                            align_corners);

                }
                return std::make_tuple(grad_input, grad_grid);
            }


        }  // namespace
    }
}  // namespace at::native



Tensor THSNN_ComplexGridSampling(const Tensor input, const Tensor grid, double max_r2) {
	CATCH_TENSOR(complexGridSample(*input, *grid, max_r2))
}

