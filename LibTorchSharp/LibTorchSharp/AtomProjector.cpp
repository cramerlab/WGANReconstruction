#pragma once
#include "CustomModules.h"
#include "Utils.h"
#include "CustomOperatorsAutograd.h"
#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

#include "MultiGPUModule.h"

struct AtomProjectorImpl : MultiGPUModule 
{
	int _sizeX;
	int _sizeY;
	int _sizeZ;
	torch::Tensor _intensities;

	AtomProjectorImpl(const torch::Tensor& intensities, int sizeX, int sizeY, int sizeZ ): _sizeX(sizeX), _sizeY(sizeY), _sizeZ(sizeZ) 
	{
		_intensities = register_parameter("intensities", intensities);
	}

	torch::Tensor ProjectToPlane(const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift)
	{
		return projectAtoms(_intensities.expand({ orientations.size(0), -1 }), positions, orientations, shift, _sizeX, _sizeY, _sizeZ);
	}

	torch::Tensor RasterToCartesian(const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift)
	{
		return atoms_to_grid(_intensities.expand({ positions.size(0), -1 }), positions, orientations, shift, _sizeX, _sizeY, _sizeZ);
	}

	torch::Tensor forward() 
	{
		return torch::Tensor();
	}

};

//TORCH_MODULE(AtomProjector);

NNModule THSNN_AtomProjector_ctor(const Tensor intensities, int sizeX, int sizeY, int sizeZ, NNAnyModule* outAsAnyModule) 
{
	at::globalContext().setBenchmarkCuDNN(true);

	AtomProjectorImpl Projector(*intensities, sizeX, sizeY, sizeZ);
	auto mod = std::make_shared<AtomProjectorImpl>(Projector);

	// Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
	// a Module can only be boxed to AnyModule at the point its static type is known).
	if (outAsAnyModule != NULL)
	{
		auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<AtomProjectorImpl>(*mod));
		*outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
	}

	auto res = new std::shared_ptr<torch::nn::Module>(mod);
	return res;
}

Tensor THSNN_AtomProjector_ProjectToPlane(const NNModule module, const Tensor positions, const Tensor orientations, const Tensor shift)
{
	CATCH_TENSOR((*module)->as<AtomProjectorImpl>()->ProjectToPlane(*positions, *orientations, *shift));
}

Tensor THSNN_AtomProjector_RasterToCartesian(const NNModule module, const Tensor positions, const Tensor orientations, const Tensor shift)
{
	CATCH_TENSOR((*module)->as<AtomProjectorImpl>()->RasterToCartesian(*positions, *orientations, *shift));
}

Tensor THSNN_ProjectAtomsToPlane(const Tensor intensities, const Tensor positions, const Tensor orientations, const Tensor shift, const int64_t sizeX, const int64_t sizeY, const int64_t sizeZ)
{
	CATCH_TENSOR(projectAtoms(intensities->expand({ orientations->size(0), -1 }), *positions, *orientations, *shift, sizeX, sizeY, sizeZ));
}

Tensor THSNN_RasterAtomsToCartesian(const Tensor intensities, const Tensor positions, const Tensor orientations, const Tensor shift, const int64_t sizeX, const int64_t sizeY, const int64_t sizeZ)
{
	CATCH_TENSOR(atoms_to_grid(intensities->expand({ positions->size(0), -1 }), *positions, *orientations, *shift, sizeX, sizeY, sizeZ));
}