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
    torch::nn::Sequential ProcessorImage;

    torch::nn::Sequential ParticleDecoder;

    torch::nn::Sequential CrapDecoder0;
    torch::nn::Sequential CrapDecoder;

    torch::nn::Sequential ProcessorAdd;
    torch::nn::Sequential ProcessorMul;

    torch::nn::Sequential ProcessorAngles;

    int64_t _codelength;

    int64_t _boxsize;
    int64_t _smallestsize;
    int64_t _smallestchannels;

    torch::Tensor _volume;

    ReconstructionWGANGeneratorImpl(torch::Tensor & volume, int64_t boxsize, int64_t codelength)
    {
        _boxsize = boxsize;
        _codelength = codelength;

        int currentchannels = 4;
        int currentsize = _boxsize;
        while (currentsize / 2 >= 4)
        {
            currentchannels *= 2;
            currentsize /= 2;
        }

        _volume = register_parameter("volume", volume);

        _smallestchannels = currentchannels;
        _smallestsize = currentsize;

        // Particle decoder
        {
            ParticleDecoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength, _codelength * 2)));
            ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            ParticleDecoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength * 2, _codelength * 4)));
            ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            ParticleDecoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength * 4, _boxsize * _boxsize).bias(true)));

            //ParticleDecoder->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(codelength, currentchannels, currentsize).stride(1)));
            //ParticleDecoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            //ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //ParticleDecoder->push_back(ReconstructionWGANResidualBlock(currentchannels, true));

            //while (currentsize * 2 <= boxsize)
            //{
            //    currentchannels /= 2;
            //    currentsize *= 2;

            //    //decoder_conv->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));

            //    //decoder_conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels * 2, currentchannels, 3).padding(1)));
            //    ParticleDecoder->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(currentchannels * 2, currentchannels, 5).stride(2).padding(2).output_padding(1)));
            //    ParticleDecoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            //    ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //    //decoder_conv->push_back(C2DNetResidualBlock(currentchannels, true, false));
            //}

            //ParticleDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels, 1, 1).padding(0)));

            //register_module("particle_decoder", ParticleDecoder);
        }

        // Crapcoder
        {
            //CrapDecoder0->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength, 64).bias(true)));
            //CrapDecoder0->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //CrapDecoder0->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 128).bias(true)));
            //CrapDecoder0->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //CrapDecoder0->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 64 * 32 * 32).bias(true)));

            //CrapDecoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            //CrapDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));                        // 32

            //CrapDecoder->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
            //CrapDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 3).padding(1)));
            //CrapDecoder->push_back(ReconstructionWGANResidualBlock(32, true));                                                            // 64

            //CrapDecoder->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
            //CrapDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).padding(1)));
            //CrapDecoder->push_back(ReconstructionWGANResidualBlock(16, true));                                                            // 128

            //CrapDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 1, 5).padding(2)));

            //register_module("crap_decoder0", CrapDecoder0);
            //register_module("crap_decoder", CrapDecoder);
        }

        // Image
        /*{
            ProcessorImage->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 1).bias(true)));

            register_module("processor_image", ProcessorImage);
        }*/

        // Additive noise
        {
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            //ProcessorAdd->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            //ProcessorAdd->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_add", ProcessorAdd);
        }

        // Multiplicative noise
        {
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
            //ProcessorMul->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            //ProcessorMul->push_back(ReconstructionWGANResidualBlock(32, true));
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

    torch::Tensor forward_particle(torch::Tensor code, torch::Tensor angles, bool transform, double sigmashift)
    {
        //torch::Tensor result = ParticleDecoder->forward(code.reshape({ -1, _codelength, 1, 1 }));
        //torch::Tensor result = ParticleDecoder->forward(code).reshape({ -1, 1, _boxsize, _boxsize });

        torch::Tensor trans_matrices = matrix_from_angles(angles);
        torch::Tensor shifts;
        if (sigmashift > 0.0) {
            shifts = torch::randn({ angles.size(0), 3, 1 }, angles.options()) * sigmashift;
            shifts = shifts.minimum(torch::ones_like(shifts) * sigmashift * 3);
        }
        else
            shifts = torch::zeros({ angles.size(0), 3, 1 }, angles.options());
        //auto dimShift = shifts.sizes().vec();
        //auto dimTransMat = trans_matrices.sizes().vec();

        trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
        trans_matrices = trans_matrices.to(_volume.device());
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { angles.size(0), 1, _boxsize, _boxsize, _boxsize }, true);
        torch::Tensor volumeRot = torch::nn::functional::grid_sample(_volume.size(0) < angles.size(0) ? _volume.expand(c10::IntArrayRef(new int64_t[]{angles.size(0), -1, -1, -1, -1}, 5)) : _volume,
            trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).align_corners(true));

        auto proj = volumeRot.sum(2);
        //torch::Tensor projstd = torch::std(proj.flatten(1, 3), 1, true, true).unsqueeze(2).unsqueeze(3);
        //torch::Tensor projmean = torch::mean(proj.flatten(1, 3), 1, true).unsqueeze(2).unsqueeze(3);

        return proj;
    }

    torch::Tensor forward(torch::Tensor crapcode, torch::Tensor fakeimages, torch::Tensor ctf)
    {
        /*torch::Tensor crap = CrapDecoder0->forward(crapcode);
        crap = crap.view({ -1, 64, 32, 32 });
        crap = CrapDecoder->forward(crap);*/
        
        torch::Tensor noise_add = torch::randn(fakeimages.sizes(), fakeimages.options());
        //noise_add = noise_add.add(crap);
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
       
        //fakeimages = ProcessorImage->forward(fakeimages);
        fakeimages = fakeimages.add(at::roll(allnoise, { (long long)((std::rand() / RAND_MAX)-0.5)* fakeimages.size(2), (long long)((std::rand() / RAND_MAX) - 0.5) * fakeimages.size(3) }, {2,3}));
         
        return fakeimages;
    }

    torch::Tensor get_Volume() {
        return _volume;
    }

};

TORCH_MODULE(ReconstructionWGANGenerator);

struct ReconstructionWGANDiscriminatorImpl : MultiGPUModule
{
    torch::nn::Sequential Discriminator;
    //torch::nn::Sequential DiscriminatorSpectral;

    //torch::nn::Sequential DiscriminatorPooled;

    ReconstructionWGANDiscriminatorImpl()
    {
        const bool sn = false;

        // Spatial
        {
            //if(sn)
            //    Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(1, 64, 7).stride(1).padding(3)));              // 256
            //else
            //    Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 7).stride(1).padding(3)));              // 256
            //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(64).affine(false)));
            //Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            ////Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            //if (sn)
            //    Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1))); // 128
            //else
            //    Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));            // 128
            //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(128).affine(false)));
            //
            ////if(sn)
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockSPNormInstanceNorm(128, true));
            ////else
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(128, true));

            ////Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            //if (sn)
            //    Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));           // 64
            //else
            //    Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));           // 64
            //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(256).affine(false)));
            ////if (sn)
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockSPNormInstanceNorm(256, true));
            ////else
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(256, true));

            ////Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            //if(sn)
            //    Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)));          // 32
            //else
            //    Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)));          // 32
            //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(512).affine(false)));
            ////if (sn)
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockSPNormInstanceNorm(512, true));
            ////else
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(512, true));

            ////Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            //
            //if(sn)
            //    Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1)));          // 16
            //else
            //    Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1)));          // 16
            ////Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(1024).affine(false)));
            ////if (sn)
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockSPNormInstanceNorm(1024, true));
            ////else
            ////    Discriminator->push_back(ReconstructionWGANResidualBlockInstanceNorm(1024, false));

            //if(sn)
            //    Discriminator->push_back(SpNormConv2d(torch::nn::Conv2dOptions(1024, 2048, 4).stride(1).padding(0)));          // 16
            //else
            //    Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 2048, 4).stride(1).padding(0)));          // 16
            ////Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(1024).affine(false)));

            ////Discriminator->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

            //Discriminator->push_back(torch::nn::Flatten());
            //
            //if(sn)
            //    Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(2048 * 1 * 1, 256)));
            //else
            //    Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(2048 * 1 * 1, 256)));
            //Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            //if(sn)
            //    Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(256, 64)));
            //else
            //    Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 64)));
            //Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            ////Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 1)));
            ////Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));

            //if(sn)
            //    DiscriminatorPooled->push_back(SpNormLinear(torch::nn::LinearOptions(64 * 1, 64)));
            //else
            //    DiscriminatorPooled->push_back(torch::nn::Linear(torch::nn::LinearOptions(64 * 1, 64)));
            //DiscriminatorPooled->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            //if(sn)
            //    DiscriminatorPooled->push_back(SpNormLinear(torch::nn::LinearOptions(64, 64)));
            //else
            //    DiscriminatorPooled->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 64)));
            //DiscriminatorPooled->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            //if(sn)
            //    DiscriminatorPooled->push_back(SpNormLinear(torch::nn::LinearOptions(64, 1)));
            //else
            //    DiscriminatorPooled->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 1)));
            // 
            if (sn) {
                

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
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                

                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(1536*9, 10)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(SpNormLinear(torch::nn::LinearOptions(10, 1)));
            }
            else {
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
                Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));


                Discriminator->push_back(torch::nn::Flatten(torch::nn::FlattenOptions()));
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(1536*9, 10)));
                Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(10, 1)));
            }

            //DiscriminatorPooled->push_back(torch::nn::Sigmoid());
            register_module("discriminator", Discriminator);
            //register_module("discriminator_pooled", DiscriminatorPooled);
        }

    }

    torch::Tensor forward(torch::Tensor image)
    {

        auto out = Discriminator->forward(image);
        //auto outDims = out.sizes().vec();
        return out;
    }

    void ClipWeights(double min, double max)
    {
        auto this_params = this->parameters(true);

        for (int64_t i = 0; i < this_params.size(); i++)
            this_params[i].data().clamp_(min, max);
    }

    torch::Tensor PenalizeGradient(torch::Tensor real, torch::Tensor fake, float lambda)
    {
        //fake = fake.to(real.device());

        torch::Tensor eta = torch::rand({ real.size(0), 1, 1, 1 }, real.options());
        eta = eta.expand(real.sizes());
        //eta.print();

        torch::Tensor interp = eta * real + ((1.0f - eta) * fake);
        interp = interp.detach();
        interp.set_requires_grad(true);

        torch::Tensor pred_interp = forward(interp);
        //pred_interp.print();

        torch::Tensor gradients = torch::autograd::grad({ pred_interp }, { interp }, { torch::ones(pred_interp.sizes(), real.options()) }, true, true)[0];
        //gradients.print();
        gradients = gradients.view({ gradients.size(0), -1 });
        //gradients.print();

        //torch::Tensor norm = gradients.norm(2, 1);
        //norm.print();
        torch::Tensor grad_penalty = (gradients.norm(2, 1) - 1).square().mean() * lambda;
        //grad_penalty.print();

        return grad_penalty;
    }
};

TORCH_MODULE(ReconstructionWGANDiscriminator);


NNModule THSNN_ReconstructionWGANGenerator_ctor(Tensor volume, int64_t boxsize, int64_t codelength, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ReconstructionWGANGeneratorImpl Net(*volume, boxsize, codelength);
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
    return torch::nn::utils::clip_grad_norm_({ (*module)->as<ReconstructionWGANGeneratorImpl>()->get_Volume() }, clip_Value, std::numeric_limits<double>::infinity());
}

Tensor THSNN_ReconstructionWGANGenerator_forward_particle(const NNModule module, const Tensor code, const Tensor angles, const bool transform, const double sigmashift)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->forward_particle(*code, *angles, transform, sigmashift));
}

Tensor THSNN_ReconstructionWGANGenerator_forward_noise(const NNModule module, const Tensor crapcode, const Tensor fakeimages, const Tensor ctf)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANGeneratorImpl>()->forward(*crapcode, *fakeimages, *ctf));
}


NNModule THSNN_ReconstructionWGANDiscriminator_ctor(NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ReconstructionWGANDiscriminatorImpl Net;
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

void THSNN_ReconstructionWGANDiscriminator_clipweights(const NNModule module, const double clip)
{
    (*module)->as<ReconstructionWGANDiscriminatorImpl>()->ClipWeights(-clip, clip);
}

Tensor THSNN_ReconstructionWGANDiscriminator_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda)
{
    CATCH_TENSOR((*module)->as<ReconstructionWGANDiscriminatorImpl>()->PenalizeGradient(*real, *fake, lambda));
}