#pragma once
#include <torch/torch.h>
struct SpNormConv2dImpl : torch::nn::Module {

	torch::nn::Conv2d conv;
	int power_iterations;
	float eps;
	torch::Tensor u;
	torch::Tensor v;

	SpNormConv2dImpl(torch::nn::Conv2dOptions options, int power_iterations = 1, float eps = 1e-12f) : conv(register_module("convolution", torch::nn::Conv2d(options))), power_iterations(power_iterations), eps(eps) {
		auto params = this->named_parameters();
		auto oldWeight = params["convolution.weight"];
		auto oldWeightMat = oldWeight.reshape(c10::IntArrayRef(new int64_t[]{ oldWeight.size(0), -1 }, 2));
		auto h = oldWeightMat.size(0);
		auto w = oldWeightMat.size(1);
		u = register_buffer("u", torch::nn::functional::normalize(torch::empty(c10::IntArrayRef(new int64_t[]{ h }, 1), oldWeightMat.options()).normal_(0, 1), torch::nn::functional::NormalizeFuncOptions().dim(0).eps(eps)));
		v = register_buffer("v", torch::nn::functional::normalize(torch::empty(c10::IntArrayRef(new int64_t[]{ w }, 1), oldWeightMat.options()).normal_(0, 1), torch::nn::functional::NormalizeFuncOptions().dim(0).eps(eps)));
	}

	torch::Tensor forward(torch::Tensor input);
};
TORCH_MODULE(SpNormConv2d);