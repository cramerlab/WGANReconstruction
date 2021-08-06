#include "CustomModules.h"
#include "Utils.h"
#include "CustomOperatorsAutograd.h"
#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/fft.h>
#include <math.h>
#include "MultiGPUModule.h"
#include "THSTensor.h"
#include <torch/nn/functional/padding.h>
#include "FFTCrop.h"

struct FourierProjectorImpl : MultiGPUModule
{
	int _sizeX;
	int _sizeY;
	int _sizeZ;
	torch::Tensor _volume;
	torch::Tensor _data;
	torch::Tensor _coordinates;
	torch::Tensor _correctedVolume;
	FourierProjectorImpl(const torch::Tensor& volume, int oversampling)
	{
	TORCH_CHECK(volume.sizes().size() == 5 && volume.size(0) == 1);
		_volume = register_parameter("volume", volume);
		std::vector<int64_t> dims_ori = _volume.sizes().vec();
		dims_ori.erase(dims_ori.begin());
		dims_ori.erase(dims_ori.begin());
		
		

	
		auto dims_oversampled = dims_ori;
		for (size_t i = 0; i < dims_oversampled.size(); i++)
		{
			dims_oversampled[i] *= oversampling;
		}
		torch::Tensor coordinates = torch::zeros(c10::IntArrayRef(new int64_t[]{ 1, 1, dims_ori[1], dims_ori[2]/2+1, 3 }, 5), torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
		auto a_coordinates = coordinates.accessor<float, 5>();
		for (int y = 0; y < dims_ori[1]; y++)
		{
			for (int x = 0; x < dims_ori[2] / 2 + 1; x++)
			{
				float xx = x;
				float yy = y < dims_ori[1] / 2 ? y + (dims_ori[1] / 2) : y - dims_ori[1] / 2;

				a_coordinates[0][0][y][x][0] = ((float)xx) / ((float)(((int)dims_oversampled[1] / 2) + 1) - 1) * 2 - 1;
				a_coordinates[0][0][y][x][1] = ((float)yy) / ((float)dims_oversampled[1] - 1) * 2 - 1;
				a_coordinates[0][0][y][x][2] = 0;
			}
		}
		_coordinates = coordinates.to(volume.device());

		float max_r2 = std::min(pow(dims_oversampled[0] / 2, 2), pow(dims_ori[0] / 2, 2));

		//correct for gridding
		torch::Tensor realspace_correct = torch::zeros(volume.sizes(), torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
		auto volDims = volume.sizes();
		auto corrDims = realspace_correct.sizes();
		auto a_realspace_correct = realspace_correct.accessor<float, 5>();
		for (size_t z = 0; z < dims_ori[0]; z++)
		{
			int zp = z - dims_ori[0]/2;
			float zp2 = zp * zp;
			for (size_t y = 0; y < dims_ori[1]; y++)
			{
				int yp = y - dims_ori[1] / 2;
				float yp2 = yp * yp;
				for (size_t x = 0; x < dims_ori[2]; x++)
				{
					int xp = x - dims_ori[2]/2;
					float r = sqrt(xp * xp + yp2 + zp2);
					if (r > 0)
					{
						float rval = r / (dims_ori[2] * oversampling);
						float sinc = (float)sin(M_PI * rval) / (float)(M_PI * rval);

						a_realspace_correct[0][0][z][y][x] = 1 / (sinc * sinc);
					}
					else {
						a_realspace_correct[0][0][z][y][x] = 1;
					}
				}
			}
		}
		torch::Tensor correctedVolume = _volume * realspace_correct.to(_volume.device());
		_correctedVolume = correctedVolume;
		torch::Tensor paddedVolume = torch::nn::functional::pad(correctedVolume, torch::nn::functional::PadFuncOptions({
			dims_ori[0] / 2 * (oversampling - 1), dims_ori[0] / 2 * (oversampling - 1),
			dims_ori[0] / 2 * (oversampling - 1), dims_ori[0] / 2 * (oversampling - 1),
			dims_ori[0] / 2 * (oversampling - 1), dims_ori[0] / 2 * (oversampling - 1) }).mode(torch::kConstant));
		torch::Tensor shiftedVolume = at::roll(correctedVolume, {dims_oversampled[0] / 2, dims_oversampled[1] / 2, dims_oversampled[2] / 2 }, { 2,3,4 });
		torch::Tensor fft = torch::fft::rfftn(shiftedVolume, c10::nullopt, c10::IntArrayRef(new int64_t[]{ 2,3,4 }, 3), "backward");
		torch::Tensor fft_shifted = fftshift(fft, IntArrayRef(new int64_t[] { 2,3 }, 2));
		torch::Tensor fft_mask = torch::zeros(fft.sizes(), torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCPU));
		auto dims = fft_shifted.sizes();
		auto a_fft_mask = fft_mask.accessor<c10::complex<float>, 5>();
		for (size_t n = 0; n < fft_mask.size(0); n++)
		{
			for (size_t z = 0; z < fft_mask.size(2); z++)
			{
				int zp = z < fft_mask.size(2) / 2 ? z : z - fft_mask.size(2);
				float zp2 = zp * zp;
				for (size_t y = 0; y < fft_mask.size(3); y++)
				{
					int yp = y < fft_mask.size(3) / 2 ? y : y - fft_mask.size(3);
					float yp2 = yp * yp;
					for (size_t x = 0; x < fft_mask.size(4); x++)
					{
						int xp = x;
						float r2 = xp * xp + yp2 + zp2;
						if (r2 <= max_r2) {
							a_fft_mask[n][0][z][y][x] = dims_ori[0];
						}
					}
				}
			}
		}
		_data = fft_shifted* fft_mask;
		
	}
	
	torch::Tensor project(torch::Tensor &angles)
	{

		torch::Tensor projected = complexGridSample(_data.to(torch::kCPU), _coordinates.to(torch::kCPU)).to(torch::kCUDA);
		return projected;
	}

	torch::Tensor projectRealspace(torch::Tensor& angles)
	{
		torch::Tensor projected = complexGridSample(_data.to(torch::kCPU), _coordinates.to(torch::kCPU)).to(torch::kCUDA);
		auto projectedIFFT = torch::fft::ifftn(projected, c10::nullopt, IntArrayRef(new int64_t[]{ 3,4 }, 2), "backward");

		return at::roll(projectedIFFT, IntArrayRef(new int64_t[]{ projectedIFFT.size(3)/2,projectedIFFT.size(4) / 2 }, 2), IntArrayRef(new int64_t[]{ 3,4 }, 2));
	}

	torch::Tensor getData()
	{
		return _data;
	}
	torch::Tensor getCorrectedVolume()
	{
		return _correctedVolume;
	}
	torch::Tensor forward()
	{
		return torch::Tensor();
	}

};


NNModule THSNN_FourierProjector_ctor(const Tensor volume, int oversampling, NNAnyModule* outAsAnyModule)
{
	at::globalContext().setBenchmarkCuDNN(true);

	FourierProjectorImpl Projector(*volume, oversampling);
	auto mod = std::make_shared<FourierProjectorImpl>(Projector);

	// Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
	// a Module can only be boxed to AnyModule at the point its static type is known).
	if (outAsAnyModule != NULL)
	{
		auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<FourierProjectorImpl>(*mod));
		*outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
	}

	auto res = new std::shared_ptr<torch::nn::Module>(mod);
	return res;
}

Tensor THSNN_FourierProjector_Project(const NNModule module, const Tensor angles)
{
	CATCH_TENSOR((*module)->as<FourierProjectorImpl>()->project(*angles));
}

Tensor THSNN_FourierProjector_ProjectRealspace(const NNModule module, const Tensor angles)
{
	CATCH_TENSOR((*module)->as<FourierProjectorImpl>()->projectRealspace(*angles));
}

Tensor THSNN_FourierProjector_GetData(const NNModule module)
{
	CATCH_TENSOR((*module)->as<FourierProjectorImpl>()->getData());
}

Tensor THSNN_FourierProjector_GetCorrectedVolume(const NNModule module)
{
	CATCH_TENSOR((*module)->as<FourierProjectorImpl>()->getCorrectedVolume());
}

