#pragma once

#include "CustomModules.h"

#include <torch/nn/init.h>
#include <torch/nn/pimpl.h>
#include <torch/fft.h>
#include <torch/nn/parallel/data_parallel.h>

#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

#include "MultiGPUModule.h"
#include "SpectralNormalization.h"
#include "CustomOperatorsAutograd.h"

extern torch::Tensor matrix_from_angles(torch::Tensor& angles);

struct ReconstructionWGANResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };

    ReconstructionWGANResidualBlock(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual(x.clone());

        x = conv1->forward(x);
        if (_donorm)
            x = bn1->forward(x);
        x = torch::leaky_relu(x, 0.2);

        x = conv2->forward(x);
        if (_donorm)
            x = bn2->forward(x);

        x += residual;
        x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct ReconstructionWGANResidualBlockSPNormInstanceNorm : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    SpNormConv2d conv1{ nullptr };
    torch::nn::InstanceNorm2d bn1{ nullptr };
    SpNormConv2d conv2{ nullptr };
    torch::nn::InstanceNorm2d bn2{ nullptr };

    ReconstructionWGANResidualBlockSPNormInstanceNorm(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", SpNormConv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(_channels).affine(false)));

        conv2 = register_module("conv2", SpNormConv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn2 = register_module("bn2", torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(_channels).affine(false)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual(x.clone());

        x = conv1->forward(x);
        if (_donorm)
            x = bn1->forward(x);
        x = torch::leaky_relu(x, 0.2);

        x = conv2->forward(x);
        if (_donorm)
            x = bn2->forward(x);

        x += residual;
        x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct ReconstructionWGANResidualBlockInstanceNorm : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::InstanceNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::InstanceNorm2d bn2{ nullptr };

    ReconstructionWGANResidualBlockInstanceNorm(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(_channels).affine(false)));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn2 = register_module("bn2", torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(_channels).affine(false)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual(x.clone());

        x = conv1->forward(x);
        if (_donorm)
            x = bn1->forward(x);
        x = torch::leaky_relu(x, 0.2);

        x = conv2->forward(x);
        if (_donorm)
            x = bn2->forward(x);

        x += residual;
        x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct ReconstructionWGANGeneratorImpl : MultiGPUModule
{
    torch::nn::Sequential ProcessorAdd;
    torch::nn::Sequential ProcessorMul;

    torch::nn::Sequential ProcessorAngles;

    int64_t _boxsize;

    bool alignCorners = true;
    torch::Tensor _volume;
    torch::Tensor _sigmashift;
    float _oversamplimg;
    int64_t _oversampledBoxsize;
    ReconstructionWGANGeneratorImpl(torch::Tensor & volume, int64_t boxsize)
    {
        _boxsize = boxsize;
        _oversamplimg = 4;
        _oversampledBoxsize = boxsize;
        
        int currentchannels = 4;
        int currentsize = _boxsize;
        while (currentsize / 2 >= 4)
        {
            currentchannels *= 2;
            currentsize /= 2;
        }
        torch::Tensor oversampledVol = _oversampledBoxsize != _boxsize? scaleVolume(volume, 3, _oversampledBoxsize, _oversampledBoxsize, _oversampledBoxsize): volume;
        _volume = register_parameter("volume", oversampledVol);
        _sigmashift = register_parameter("sigmaShift", torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat)));
        // Additive noise
        {
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_add", ProcessorAdd);
        }

        // Multiplicative noise
        {
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_mul", ProcessorMul);
        }

        {
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(3, 9)));
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(18, 36)));
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(36, 18)));
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(18, 9)));

            register_module("processor_angles", ProcessorAngles);
        }
    }

    void apply_volume_masks(torch::Tensor binaryMask, torch::Tensor maxMask, double multiplicator) {

        torch::NoGradGuard guard;
        auto dimsBinMask = binaryMask.sizes().vec();
        auto dimsMaxMask = maxMask.sizes().vec();
        auto dims_volume = _volume.sizes().vec();
        //torch::Tensor maskedVol = _volume.mul(binaryMask);
        //maskedVol = maskedVol.maximum(-0.08*maskedVol.max() * maxMask);
        //  formula 65 from supp k/K*R/100 * max{x^k}
        torch::Tensor volMin = _volume.max() * multiplicator;
        // formula 64 in supp: g(k) = V_min *(1-r/rcut)

        torch::Tensor gk = volMin * maxMask;
        torch::Tensor res = (torch::nn::ReLU()(_volume - gk) + gk).mul(binaryMask);
        auto dims_res = res.sizes().vec();
        auto dev_res = res.get_device();
        auto dev_volume = _volume.get_device();
        _volume.set_data(res);
    }

    torch::Tensor forward(torch::Tensor angles, bool do_shift)
    {

        torch::Tensor trans_matrices = matrix_from_angles(angles);

        torch::Tensor shifts = torch::zeros({ angles.size(0), 3, 1 }, angles.options());;
        if (do_shift) {
            shifts = torch::randn({ angles.size(0), 3, 1 }, angles.options()) * _sigmashift;
            //shifts = shifts.minimum(torch::ones_like(shifts) * _sigmashift * 3);
        }
        trans_matrices = trans_matrices.transpose(1, 2);
        trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
        trans_matrices = trans_matrices.to(_volume.device());
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, _oversampledBoxsize, _oversampledBoxsize, _oversampledBoxsize }, alignCorners);
        torch::Tensor volumeRot = torch::nn::functional::grid_sample(_volume.size(0) < angles.size(0) ? _volume.expand(c10::IntArrayRef(new int64_t[]{ angles.size(0), -1, -1, -1, -1 }, 5)) : _volume,
            trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).align_corners(alignCorners));

        auto proj = volumeRot.sum(2);
        if (_oversampledBoxsize != _boxsize)
            proj = scaleVolume(proj, 2, _boxsize, _boxsize, -1);
        return proj;
    }

    torch::Tensor forward_new(torch::Tensor angles, bool do_shift)
    {

        torch::Tensor trans_matrices = matrix_from_angles(angles);

        torch::Tensor shifts = torch::zeros({ angles.size(0), 3, 1 }, angles.options());;
        if (do_shift) {
            shifts = torch::randn({ angles.size(0), 3, 1 }, angles.options()) * _sigmashift;
            //shifts = shifts.minimum(torch::ones_like(shifts) * _sigmashift * 3);
        }
        trans_matrices = trans_matrices.transpose(1, 2);
        trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
        trans_matrices = trans_matrices.to(_volume.device());
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, _oversampledBoxsize, _oversampledBoxsize, _oversampledBoxsize }, alignCorners);

        //auto deviceVol = _volume.device();
        //auto dimsVol = _volume.sizes().vec();
        //auto deviceGrid = trans_grid.device();

        //auto dimsGrid = trans_grid.sizes().vec();
        auto proj = gridSampleAndProject(_volume, trans_grid);
        //auto projDev = proj.device();
        //auto dimsAngles = angles.sizes().vec();
        //auto size = angles.size(0) * _oversampledBoxsize * _oversampledBoxsize;
        //float * projFlat = new float[angles.size(0) * _oversampledBoxsize * _oversampledBoxsize];
        //auto projFlatDims = proj.sizes().vec();
        //auto err = cudaPeekAtLastError();
        //cudaDeviceSynchronize();
        //cudaMemcpy(projFlat, proj.data_ptr(), angles.size(0) * _oversampledBoxsize * _oversampledBoxsize * sizeof(float), cudaMemcpyDeviceToHost);
        //auto err2 = cudaPeekAtLastError();
        //cudaDeviceSynchronize();
        if (_oversampledBoxsize != _boxsize)
            proj = scaleVolume(proj, 2, _boxsize, _boxsize, -1);
        //auto err3 = cudaPeekAtLastError();
        return proj;
    }

    torch::Tensor project(torch::Tensor angles, double sigmashift)
    {

        torch::Tensor trans_matrices = matrix_from_angles(angles);
        torch::Tensor shifts;
        if (sigmashift > 0.0) {
            shifts = torch::randn({ angles.size(0), 3, 1 }, angles.options()) * sigmashift;
            shifts = shifts.minimum(torch::ones_like(shifts) * sigmashift * 3);
        }
        else
            shifts = torch::zeros({ angles.size(0), 3, 1 }, angles.options());
        
        trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
        trans_matrices = trans_matrices.to(_volume.device());
        
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, _boxsize, _boxsize, _boxsize }, alignCorners);
        torch::Tensor volumeRot = torch::nn::functional::grid_sample(_volume.size(0) < angles.size(0) ? _volume.expand(c10::IntArrayRef(new int64_t[]{angles.size(0), -1, -1, -1, -1}, 5)) : _volume,
            trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).align_corners(alignCorners));

        auto proj = volumeRot.sum(2);

        return proj;
    }

    torch::Tensor forward_normalized(torch::Tensor angles, torch::Tensor factor)
    {
        torch::Tensor trans_matrices = matrix_from_angles(angles);
        torch::Tensor shifts;

        shifts = torch::zeros({ angles.size(0), 3, 1 }, angles.options());

        trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
        trans_matrices = trans_matrices.to(_volume.device());

        torch::Tensor Volume = _volume * factor;
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, _boxsize, _boxsize, _boxsize }, alignCorners);
        torch::Tensor volumeRot = torch::nn::functional::grid_sample(Volume.size(0) < angles.size(0) ? Volume.expand(c10::IntArrayRef(new int64_t[]{ angles.size(0), -1, -1, -1, -1 }, 5)) : Volume,
            trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).align_corners(alignCorners));

        auto proj = volumeRot.sum(2);

        return proj;
    }


    torch::Tensor apply_noise(torch::Tensor fakeimages, torch::Tensor ctf)
    {
        
        torch::Tensor noise_add = torch::randn(fakeimages.sizes(), fakeimages.options());
        noise_add = ProcessorAdd->forward(noise_add);
        noise_add = torch::fft::rfftn(noise_add, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");
        noise_add = noise_add.mul(ctf);
        noise_add = torch::fft::irfftn(noise_add, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");

        torch::Tensor noise_mul = torch::randn(fakeimages.sizes(), fakeimages.options());
        noise_mul = ProcessorMul->forward(noise_mul);

        torch::Tensor allnoise = noise_add + noise_mul;
        torch::Tensor noisestd = torch::std(allnoise.flatten(1, 3), 1, true, true).unsqueeze(2).unsqueeze(3);
        torch::Tensor noisemean = torch::mean(allnoise.flatten(1, 3), 1, true).unsqueeze(2).unsqueeze(3);
        allnoise = (allnoise - noisemean) / (noisestd + 1e-4f);

        fakeimages = fakeimages.add(at::roll(allnoise, { (long long)((std::rand() / RAND_MAX)-0.5)* fakeimages.size(2), (long long)((std::rand() / RAND_MAX) - 0.5) * fakeimages.size(3) }, {2,3}));
        
        //torch::Tensor noise = torch::randn(fakeimages.sizes(), fakeimages.options());
        //fakeimages = fakeimages.add(noise);
        return fakeimages;
    }

    torch::Tensor get_Volume() {
        torch::Tensor smallerVolume = _boxsize!=_oversampledBoxsize? scaleVolume(_volume, 3, _boxsize, _boxsize, _boxsize): _volume;
        return smallerVolume;
    }

    torch::Tensor get_OversampledVolume() {
        return _volume;
    }

};

TORCH_MODULE(ReconstructionWGANGenerator);

struct ReconstructionWGANGeneratorSymImpl : MultiGPUModule
{
    torch::nn::Sequential ProcessorAdd;
    torch::nn::Sequential ProcessorMul;

    torch::nn::Sequential ProcessorAngles;

    int64_t _boxsize;

    bool alignCorners = true;
    torch::Tensor _volume;
    torch::Tensor _sigmashift;

    ReconstructionWGANGeneratorSymImpl(int64_t boxsize)
    {
        _boxsize = boxsize;


        int currentchannels = 4;
        int currentsize = _boxsize;
        while (currentsize / 2 >= 4)
        {
            currentchannels *= 2;
            currentsize /= 2;
        }

        torch::Tensor quarterVolume = torch::zeros({ 1, boxsize / 4, boxsize / 4, boxsize / 4 });
        _volume = register_parameter("volume", quarterVolume);
        _sigmashift = register_parameter("sigmaShift", torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat)));
        // Additive noise
        {
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_add", ProcessorAdd);
        }

        // Multiplicative noise
        {
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_mul", ProcessorMul);
        }

        {
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(3, 9)));
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(18, 36)));
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(36, 18)));
            ProcessorAngles->push_back(torch::nn::Linear(torch::nn::LinearOptions(18, 9)));

            register_module("processor_angles", ProcessorAngles);
        }
    }

    void apply_volume_masks(torch::Tensor binaryMask, torch::Tensor maxMask, double multiplicator) {

        torch::NoGradGuard guard;
        auto dimsBinMask = binaryMask.sizes().vec();
        auto dimsMaxMask = maxMask.sizes().vec();
        auto dims_volume = _volume.sizes().vec();
        //  formula 65 from supp k/K*R/100 * max{x^k}
        torch::Tensor volMin = _volume.max() * multiplicator;
        // formula 64 in supp: g(k) = V_min *(1-r/rcut)

        torch::Tensor gk = volMin * maxMask;
        torch::Tensor res = (torch::nn::ReLU()(_volume - gk) + gk).mul(binaryMask);
        auto dims_res = res.sizes().vec();
        auto dev_res = res.get_device();
        auto dev_volume = _volume.get_device();
        _volume.set_data(res);
    }

    torch::Tensor forward(torch::Tensor angles, bool do_shift)
    {

        torch::Tensor trans_matrices = matrix_from_angles(angles);

        torch::Tensor shifts = torch::zeros({ angles.size(0), 3, 1 }, angles.options());;
        if (do_shift) {
            shifts = torch::randn({ angles.size(0), 3, 1 }, angles.options()) * _sigmashift;
            //shifts = shifts.minimum(torch::ones_like(shifts) * _sigmashift * 3);
        }
        
        /*
            xd2=torch.flip(xd1, [2,3])
            im1=torch.cat([xd1, xd2],2)
            im2=torch.flip(im1,[1,2])
            im=torch.cat([im1,im2], 1)
        */
        torch::Tensor xd2 = torch::flip(_volume, { 3,4 });
        torch::Tensor im1 = torch::cat({ _volume, xd2 }, 3);
        torch::Tensor im2 = torch::flip(im1, { 2, 3 });
        torch::Tensor thisVolume = torch::cat({ im1, im2 }, 2);

        trans_matrices = trans_matrices.transpose(1, 2);
        trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
        trans_matrices = trans_matrices.to(_volume.device());
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, _boxsize, _boxsize, _boxsize }, alignCorners);
        torch::Tensor volumeRot = torch::nn::functional::grid_sample(thisVolume.size(0) < angles.size(0) ? thisVolume.expand(c10::IntArrayRef(new int64_t[]{ angles.size(0), -1, -1, -1, -1 }, 5)) : _volume,
            trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).align_corners(alignCorners));

        auto proj = volumeRot.sum(2);
        return proj;
    }



    torch::Tensor apply_noise(torch::Tensor fakeimages, torch::Tensor ctf)
    {

        torch::Tensor noise_add = torch::randn(fakeimages.sizes(), fakeimages.options());
        noise_add = ProcessorAdd->forward(noise_add);
        noise_add = torch::fft::rfftn(noise_add, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");
        noise_add = noise_add.mul(ctf);
        noise_add = torch::fft::irfftn(noise_add, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");

        torch::Tensor noise_mul = torch::randn(fakeimages.sizes(), fakeimages.options());
        noise_mul = ProcessorMul->forward(noise_mul);

        torch::Tensor allnoise = noise_add + noise_mul;
        torch::Tensor noisestd = torch::std(allnoise.flatten(1, 3), 1, true, true).unsqueeze(2).unsqueeze(3);
        torch::Tensor noisemean = torch::mean(allnoise.flatten(1, 3), 1, true).unsqueeze(2).unsqueeze(3);
        allnoise = (allnoise - noisemean) / (noisestd + 1e-4f);

        fakeimages = fakeimages.add(at::roll(allnoise, { (long long)((std::rand() / RAND_MAX) - 0.5) * fakeimages.size(2), (long long)((std::rand() / RAND_MAX) - 0.5) * fakeimages.size(3) }, { 2,3 }));

        //torch::Tensor noise = torch::randn(fakeimages.sizes(), fakeimages.options());
        //fakeimages = fakeimages.add(noise);
        return fakeimages;
    }

    torch::Tensor get_quarter_Volume() {
        return _volume;
    }

    torch::Tensor get_Volume() {
        torch::NoGradGuard g;
        torch::Tensor xd2 = torch::flip(_volume, { 3,4 });
        torch::Tensor im1 = torch::cat({ _volume, xd2 }, 3);
        torch::Tensor im2 = torch::flip(im1, { 2, 3 });
        torch::Tensor thisVolume = torch::cat({ im1, im2 }, 2);
        return thisVolume;
    }
};

TORCH_MODULE(ReconstructionWGANGeneratorSym);

struct ReconstructionWGANDiscriminatorImpl : MultiGPUModule
{
    torch::nn::Sequential Discriminator;


    ReconstructionWGANDiscriminatorImpl(int64_t boxsize, bool normalize)
    {
        const bool sn = false;

        {
            if (normalize) {
                Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(true).momentum(0.0).affine(false).eps(1e-5).track_running_stats(false)));
            }
            if (sn) {
                
                /*
                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(1, 96, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(96, 192, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(384, 768, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(768, 1536, 3).stride(1).padding(1)));
                if (boxsize >= 32) 
                    Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                
                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(1536, 3072, 3).stride(1).padding(1)));
                if (boxsize >= 64)
                    Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                int size = 0;
                {
                    torch::NoGradGuard no_grad;
                    auto output = Discriminator->forward(torch::ones({ 1,1,boxsize ,boxsize }));
                    auto dims = output.sizes().vec();
                    size = dims[1];
                }
                Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(size, 10)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(10, 1)));*/

                /* Own tries */
                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(1, 96, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(ReconstructionWGANResidualBlockSPNormInstanceNorm(96));
                //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(96).affine(false)));
                //Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 96, 2).stride(2)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(96, 192, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(192));
                //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(192).affine(false)));
                //Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(192, 192, 2).stride(2)));

                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(384));
                //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(384).affine(false)));
                //Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(384, 384, 2).stride(2)));

                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(384, 768, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(768, 1536, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(1536, 3072, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //if (boxsize >= 64)
                //  Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                //Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(768, 768, 2).stride(2)));

                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                int size = 0;
                /* {
                    torch::NoGradGuard no_grad;
                    auto output = Discriminator->forward(torch::ones({ 1,1,boxsize ,boxsize }));
                    auto dims = output.sizes().vec();
                    size = dims[1];
                }*/

                Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(3072*16, 50*4)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(50*4, 1)));
            }
            else {

                /*
                * CryoGAN
                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 96, 3).stride(1).padding(1)));
                //Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 192, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 768, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(768, 1536, 3).stride(1).padding(1)));
                if(boxsize>=32)
                    Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1536, 3072, 3).stride(1).padding(1)));
                if(boxsize>=64)
                    Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                
                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                int size = 0;
                {
                    torch::NoGradGuard no_grad;
                    auto output = Discriminator->forward(torch::ones({ 1,1,boxsize ,boxsize }));
                    auto dims = output.sizes().vec();
                    size = dims[1];
                }
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(size, 10)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(10, 1)));
                */
                
                /* Own tries */
                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 96, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));


                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 192, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));


                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 768, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));


                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(768, 1536, 3).stride(1).padding(1)));
                if (boxsize >= 32)
                    Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1536, 3072, 3).stride(1).padding(1)));
                if (boxsize >= 64)
                    Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                int size = 0;
                {
                    torch::NoGradGuard no_grad;
                    auto output = Discriminator->forward(torch::ones({ 1,1,boxsize ,boxsize }));
                    auto dims = output.sizes().vec();
                    size = dims[1];
                }
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(size, 10)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(10, 1)));

                /* MultiCryoGAN */
                /*
                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 96, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(96));
                //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(96).affine(false)));
                //Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 96, 2).stride(2)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                
                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 192, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(192));
                //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(192).affine(false)));
                //Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 192, 2).stride(2)));
                
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                //Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(384));
                //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(384).affine(false)));
                //Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 2).stride(2)));
                
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

                Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 768, 3).stride(1).padding(1)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                //Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(768, 768, 2).stride(2)));
                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                int size = 0;
                {
                    torch::NoGradGuard no_grad;
                    auto output = Discriminator->forward(torch::ones({ 1,1,boxsize ,boxsize }));
                    auto dims = output.sizes().vec();
                    size = dims[1];
                }
                
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(size, 50)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(50, 1)));
                */

            }

            register_module("discriminator", Discriminator);
        }

    }

    torch::Tensor forward(torch::Tensor image)
    {
        auto out = Discriminator->forward(image);
        auto dims = out.sizes().vec();
        return out;
    }

    torch::Tensor PenalizeGradient(torch::Tensor real, torch::Tensor fake, float lambda)
    {


        torch::Tensor eta = torch::rand({ real.size(0), 1, 1, 1 }, real.options());
        eta = eta.expand(real.sizes());


        torch::Tensor interp = eta * real + ((1.0f - eta) * fake);
        interp = interp.detach();
        interp.set_requires_grad(true);

        torch::Tensor pred_interp = forward(interp);

        torch::Tensor gradients = torch::autograd::grad({ pred_interp }, { interp }, { torch::ones(pred_interp.sizes(), real.options()) }, true, true)[0];
        gradients = gradients.view({ gradients.size(0), -1 });

        torch::Tensor grad_penalty = (gradients.norm(2, 1) - 1).square().mean() * lambda;

        return grad_penalty;
    }
};

TORCH_MODULE(ReconstructionWGANDiscriminator);

struct ReconstructionWGANSimpleCriticImpl : MultiGPUModule
{
    torch::nn::Sequential Critic;
    torch::Tensor Image;

    ReconstructionWGANSimpleCriticImpl(int64_t boxsize, torch::Tensor image)
    {
        Image = image;
        Critic->push_back(torch::nn::Flatten());
        Critic->push_back(torch::nn::Linear(torch::nn::LinearOptions(boxsize * boxsize, 10)));
        Critic->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        Critic->push_back(torch::nn::Linear(torch::nn::LinearOptions(10, 1)));
        register_module("critic", Critic);
    }

    torch::Tensor forward(torch::Tensor image)
    {
        torch::Tensor imageExpanded = this->Image.expand_as(image);
        auto dimIM = image.sizes().vec();
        auto devIM = image.get_device();
        auto TIM = imageExpanded.sizes().vec();
        auto devTIM = imageExpanded.get_device();

        torch::Tensor multiplied = image.mul(imageExpanded);
        return Critic->forward(multiplied);
    }

    torch::Tensor PenalizeGradient(torch::Tensor real, torch::Tensor fake, float lambda)
    {
        torch::Tensor eta = torch::rand({ real.size(0), 1, 1, 1 }, real.options());
        eta = eta.expand(real.sizes());


        torch::Tensor interp = eta * real + ((1.0f - eta) * fake);
        interp = interp.detach();
        interp.set_requires_grad(true);

        torch::Tensor pred_interp = forward(interp);

        torch::Tensor gradients = torch::autograd::grad({ pred_interp }, { interp }, { torch::ones(pred_interp.sizes(), real.options()) }, true, true)[0];
        gradients = gradients.view({ gradients.size(0), -1 });

        torch::Tensor grad_penalty = (gradients.norm(2, 1) - 1).square().mean() * lambda;

        return grad_penalty;
    }
};

TORCH_MODULE(ReconstructionWGANSimpleCritic);

NNModule THSNN_ReconstructionWGANGeneratorSym_ctor(int64_t boxsize, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ReconstructionWGANGeneratorSymImpl Net(boxsize);
    auto mod = std::make_shared<ReconstructionWGANGeneratorSymImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ReconstructionWGANGeneratorSymImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

double THSNN_ReconstructionWGANGeneratorSym_clip_gradient(const NNModule module, const double clip_Value) {
    return torch::nn::utils::clip_grad_norm_({ (*module)->as<ReconstructionWGANGeneratorSymImpl>()->get_quarter_Volume() }, clip_Value, std::numeric_limits<double>::infinity());
}

Tensor THSNN_ReconstructionWGANGeneratorSym_forward(const NNModule module, const Tensor angles, const bool do_shift)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->forward(*angles, do_shift));
}

Tensor THSNN_ReconstructionWGANGeneratorSym_get_volume(const NNModule module)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorSymImpl>()->get_Volume());
}

NNModule THSNN_ReconstructionWGANGenerator_ctor(Tensor volume, int64_t boxsize, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ReconstructionWGANGeneratorImpl Net(*volume, boxsize);
    auto mod = std::make_shared<ReconstructionWGANGeneratorImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ReconstructionWGANGeneratorImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

double THSNN_ReconstructionWGANGenerator_clip_gradient(const NNModule module, const double clip_Value) {
    return torch::nn::utils::clip_grad_norm_({ (*module)->as<ReconstructionWGANGeneratorImpl>()->get_OversampledVolume() }, clip_Value, std::numeric_limits<double>::infinity());
}

Tensor THSNN_ReconstructionWGANGenerator_project(const NNModule module, const Tensor angles, const double sigmashift)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->project(*angles, sigmashift));
}

Tensor THSNN_ReconstructionWGANGenerator_forward(const NNModule module, const Tensor angles, const bool do_shift)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->forward(*angles, do_shift));
}

Tensor THSNN_ReconstructionWGANGenerator_forward_new(const NNModule module, const Tensor angles, const bool do_shift)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->forward_new(*angles, do_shift));
}

Tensor THSNN_ReconstructionWGANGenerator_forward_normalized(const NNModule module, const Tensor angles, const Tensor factor)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->forward_normalized(*angles, *factor));
}

Tensor THSNN_ReconstructionWGANGenerator_apply_noise(const NNModule module, const Tensor fakeimages, const Tensor ctf)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->apply_noise(*fakeimages, *ctf));
}

Tensor THSNN_ReconstructionWGANGenerator_get_volume(const NNModule module)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->get_Volume());
}

void THSNN_ReconstructionWGANGenerator_apply_volume_masks(const NNModule module, Tensor binaryMask, Tensor maxMask, double multiplicator)
{
    (*module)->as<ReconstructionWGANGeneratorImpl>()->apply_volume_masks(*binaryMask, *maxMask, multiplicator);
}

NNModule THSNN_ReconstructionWGANDiscriminator_ctor(NNAnyModule* outAsAnyModule, int64_t boxsize, bool normalizeInput)
{
    //CATCH_RETURN_NNModule
    //(
    ReconstructionWGANDiscriminatorImpl Net(boxsize, normalizeInput);
    auto mod = std::make_shared<ReconstructionWGANDiscriminatorImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ReconstructionWGANDiscriminatorImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

double THSNN_ReconstructionWGANDiscriminator_clip_gradient(const NNModule module, const double clip_Value) 
{
    return torch::nn::utils::clip_grad_norm_((*module)->as<ReconstructionWGANDiscriminatorImpl>()->parameters(), clip_Value);
}

Tensor THSNN_ReconstructionWGANDiscriminator_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANDiscriminatorImpl>()->forward(*input));
}

Tensor THSNN_ReconstructionWGANDiscriminator_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANDiscriminatorImpl>()->PenalizeGradient(*real, *fake, lambda));
}


NNModule THSNN_ReconstructionWGANSimpleCritic_ctor(NNAnyModule* outAsAnyModule, int64_t boxsize, Tensor image)
{
    //CATCH_RETURN_NNModule
    //(
    ReconstructionWGANSimpleCriticImpl Net(boxsize, *image);
    auto mod = std::make_shared<ReconstructionWGANSimpleCriticImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ReconstructionWGANSimpleCriticImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

double THSNN_ReconstructionWGANSimpleCritic_clip_gradient(const NNModule module, const double clip_Value)
{
    return torch::nn::utils::clip_grad_norm_((*module)->as<ReconstructionWGANSimpleCriticImpl>()->parameters(), clip_Value);
}

Tensor THSNN_ReconstructionWGANSimpleCritic_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANSimpleCriticImpl>()->forward(*input));
}

Tensor THSNN_ReconstructionWGANSimpleCritic_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANSimpleCriticImpl>()->PenalizeGradient(*real, *fake, lambda));
}