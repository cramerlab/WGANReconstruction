#include "SpectralNormalization.h"

torch::Tensor SpNormConv2dImpl::forward(torch::Tensor input)
{
		auto params = this->named_parameters();
		auto oldWeight = params["weight_orig"];
		auto oldWeightMat = oldWeight.reshape(c10::IntArrayRef(new int64_t[]{ oldWeight.size(0), -1 }, 2));
		{
			torch::NoGradGuard no_grad;
			for (size_t i = 0; i < power_iterations; i++)
			{
				v = torch::nn::functional::normalize(torch::mv(oldWeightMat.transpose(0, 1), u), torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
				u = torch::nn::functional::normalize(torch::mv(oldWeightMat, v), torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
			}
		}
		
		auto sigma = torch::dot(u, torch::mv(oldWeightMat, v));
		auto newWeight = oldWeight / sigma;

		params["convolution.weight"].set_data(newWeight);

		auto inpDev = input.device();
		auto oldWeightDev = oldWeight.device();
		auto uDev = u.device();
		auto vDev = v.device();
		auto newWeightDev = newWeight.device();
		auto output = conv->forward(input);

		return output;
}


torch::Tensor SpNormLinearImpl::forward(torch::Tensor input)
{
	auto params = this->named_parameters();
	auto oldWeight = params["weight_orig"];
	auto oldWeightMat = oldWeight.reshape(c10::IntArrayRef(new int64_t[]{ oldWeight.size(0), -1 }, 2));
	{
		torch::NoGradGuard no_grad;
		for (size_t i = 0; i < power_iterations; i++)
		{
			v = torch::nn::functional::normalize(torch::mv(oldWeightMat.transpose(0, 1), u), torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
			u = torch::nn::functional::normalize(torch::mv(oldWeightMat, v), torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
		}
	}
	auto sigma = torch::dot(u, torch::mv(oldWeightMat, v));
	auto newWeight = oldWeight / sigma;

	params["linear.weight"].set_data(newWeight);

	return linear->forward(input);
}
