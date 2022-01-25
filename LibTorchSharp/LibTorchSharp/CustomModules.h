#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(NNModule) THSNN_ResNet_ctor(const int64_t size_input, const int64_t blocks1, const int64_t blocks2, const int64_t blocks3, const int64_t blocks4, const int64_t num_classes, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ResNet_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_UNet3D_ctor(const int64_t depth_block, const int64_t width_block, const int64_t input_channels, const int64_t final_channels, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_UNet3D_forward(const NNModule module, const Tensor input);

EXPORT_API(NNModule) THSNN_UNet2D_ctor(const int64_t depth_block, const int64_t width_block, const int64_t input_channels, const int64_t final_channels, const int64_t final_kernel, const bool dochannelattn, const bool dospatialattn, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_UNet2D_forward(const NNModule module, const Tensor input);

EXPORT_API(NNModule) THSNN_ParticleWGANGenerator_ctor(int64_t boxsize, int64_t codelength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ParticleWGANGenerator_forward_particle(const NNModule module, const Tensor code, const bool transform, const double sigmashift);
EXPORT_API(Tensor)   THSNN_ParticleWGANGenerator_forward_noise(const NNModule module, const Tensor crapcode, const Tensor fakeimages, const Tensor ctf);

EXPORT_API(NNModule) THSNN_ParticleWGANDiscriminator_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ParticleWGANDiscriminator_forward(const NNModule module, const Tensor input);
EXPORT_API(void)     THSNN_ParticleWGANDiscriminator_clipweights(const NNModule module, const double clip);
EXPORT_API(Tensor)   THSNN_ParticleWGANDiscriminator_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda);

EXPORT_API(NNModule) THSNN_ReconstructionWGANGenerator_ctor(Tensor volume, int64_t boxsize, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANGenerator_forward(const NNModule module, const Tensor angles, const bool do_shift);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANGenerator_forward_new(const NNModule module, const Tensor angles, const bool do_shift);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANGenerator_project(const NNModule module, const Tensor angles, const double sigmashift);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANGenerator_forward_normalized(const NNModule module, const Tensor angles, const Tensor factor);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANGenerator_apply_noise(const NNModule module, const Tensor fakeimages, const Tensor ctf);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANGenerator_get_volume(const NNModule module);
EXPORT_API(void)   THSNN_ReconstructionWGANGenerator_apply_volume_masks(const NNModule module, Tensor binaryMask, Tensor maxMask);
EXPORT_API(double)   THSNN_ReconstructionWGANGenerator_clip_gradient(const NNModule module, const double clip_Value);

EXPORT_API(NNModule) THSNN_ReconstructionWGANDiscriminator_ctor(NNAnyModule* outAsAnyModule, int64_t boxsize);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANDiscriminator_forward(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_ReconstructionWGANDiscriminator_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda);
EXPORT_API(double)   THSNN_ReconstructionWGANDiscriminator_clip_gradient(const NNModule module, const double clip_Value);

EXPORT_API(NNModule) THSNN_AtomProjector_ctor(const Tensor intensities, int sizeX, int sizeY, int sizeZ, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AtomProjector_ProjectToPlane(const NNModule module, const Tensor positions, const Tensor orientations, const Tensor shift);
EXPORT_API(Tensor)   THSNN_AtomProjector_RasterToCartesian(const NNModule module, const Tensor positions, const Tensor orientations, const Tensor shift);
EXPORT_API(Tensor)   THSNN_ProjectAtomsToPlane(const Tensor intensities, const Tensor positions, const Tensor orientations, const Tensor shift, const int64_t sizeX, const int64_t sizeY, const int64_t sizeZ);
EXPORT_API(Tensor)   THSNN_RasterAtomsToCartesian(const Tensor intensities, const Tensor positions, const Tensor orientations, const Tensor shift, const int64_t sizeX, const int64_t sizeY, const int64_t sizeZ);

EXPORT_API(NNModule) THSNN_DistanceNet_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_DistanceNet_forward(const NNModule module, const Tensor reference, const Tensor data, void* d_reference, void* d_data);

EXPORT_API(NNModule) THSNN_C2DNetEncoder_ctor(const int64_t boxsize, const int64_t codelength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_C2DNetEncoder_forward(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_C2DNetEncoder_forward_pose(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_C2DNetEncoder_apply_pose(const NNModule module, const Tensor input, const Tensor pose);
EXPORT_API(NNModule) THSNN_C2DNetDecoder_ctor(const int64_t boxsize, const int64_t codelength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_C2DNetDecoder_forward(const NNModule module, const Tensor input, const bool usekl);
EXPORT_API(Tensor)   THSNN_C2DNetDecoder_kld(const NNModule module, const Tensor input, const double weight);
EXPORT_API(Tensor)   THSNN_C2DNetDecoder_minmse(const Tensor decoded, const Tensor data);

EXPORT_API(NNModule) THSNN_MLP_ctor(int64_t* block_widths, int nblocks, bool residual, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MLP_forward(const NNModule module, const Tensor input);

EXPORT_API(Tensor) THSNN_FFTCrop(const Tensor fft_volume, int dim, int new_x, int new_y, int new_z);
EXPORT_API(Tensor) THSNN_ComplexGridSampling(const Tensor input, const Tensor grid);

EXPORT_API(NNModule) THSNN_Projector_ctor(const Tensor volume, int oversampling, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor) THSNN_Projector_Project(const NNModule module, const Tensor angles);
EXPORT_API(Tensor) THSNN_Projector_GetData(const NNModule module);
EXPORT_API(Tensor) THSNN_Projector_GetCorrectedVolume(const NNModule module);

EXPORT_API(Tensor) THSNN_MatrixFromAngles(Tensor angles);
EXPORT_API(Tensor) THSNN_AffineMatrixFromAngles(Tensor angles, float shift);
EXPORT_API(Tensor) THSNN_RotateVolume(Tensor volume, Tensor angles, float shift);
EXPORT_API(Tensor) THSNN_ScaleVolume(Tensor volume, int dim, int new_x, int new_y, int new_z);