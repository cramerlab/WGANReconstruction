#include "CustomOperatorsAutograd.h"
#include "CustomOperatorsBackend.cuh"
#include "ComplexGridSampler.h"


using namespace torch::autograd;

// Atoms To Grid part

torch::Tensor atoms_to_grid(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::atoms_to_grid", "")
		.typed<decltype(atoms_to_grid)>();
	// Make sure that we have expected dimensionality for the lower level function calls
	TORCH_CHECK(intensities.dim() == 2 && positions.dim() == 3 && orientations.dim() == 3 && shift.dim() == 2);
	TORCH_CHECK(intensities.size(0) == positions.size(0) && orientations.size(0) == positions.size(0) && shift.size(0) == positions.size(0));
	return op.call(intensities, positions, orientations, shift, x, y, z);
}

tensor_list atoms_to_grid_backwards(const torch::Tensor& grad_output, const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::atoms_to_grid_backwards", "")
		.typed<decltype(atoms_to_grid_backwards)>();
	//No need to check in backwards pass
	return op.call(grad_output, intensities, positions, orientations, shift);
}

class Atoms_to_grid : public Function<Atoms_to_grid>
{
public:
	static torch::Tensor forward(
		AutogradContext* ctx, torch::Tensor intensities, torch::Tensor positions, torch::Tensor orientations, torch::Tensor shift, int64_t x, int64_t y, int64_t z) {
		ctx->save_for_backward({ intensities, positions, orientations, shift });
		ctx->saved_data["x"] = x;
		ctx->saved_data["y"] = y;
		ctx->saved_data["z"] = z;
		at::AutoNonVariableTypeMode g;
		return atoms_to_grid(intensities, positions, orientations, shift, x, y, z);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = atoms_to_grid_backwards(grad_outputs[0], saved[0], saved[1], saved[2], saved[3]);
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		return ret;
	}
};

torch::Tensor atoms_to_grid_autograd(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	return Atoms_to_grid::apply(intensities, positions, orientations, shift, x, y, z);
}

torch::Tensor atoms_to_grid_cuda(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	return at::native::MyOperator::atoms_to_grid_3d_cuda(intensities, positions, orientations, shift, x, y, z);
}
tensor_list atoms_to_grid_backwards_cuda(const torch::Tensor& grad_output, const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift) {
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ret = at::native::MyOperator::atoms_to_grid_3d_backward_cuda(grad_output, intensities, positions, orientations, shift);
	return tensor_list({ std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret) });
}

//Project Atoms part

torch::Tensor projectAtoms(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::projectAtoms", "")
		.typed<decltype(projectAtoms)>();
	// Make sure that we have expected dimensionality for the lower level function calls
	TORCH_CHECK(intensities.dim() == 2 && positions.dim() == 3 && orientation.dim() == 3 && shift.dim() == 2);
	TORCH_CHECK(intensities.size(0) == positions.size(0) && orientation.size(0) == positions.size(0) && shift.size(0) == positions.size(0));
	return op.call(intensities, positions, orientation, shift, x, y, z);
}

tensor_list projectAtoms_backwards(const torch::Tensor& grad_output, const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::projectAtoms_backwards", "")
		.typed<decltype(projectAtoms_backwards)>();
	return op.call(grad_output, intensities, positions, orientation, shift, x, y, z);
}

class ProjectAtoms : public Function<ProjectAtoms>
{
public:
	static torch::Tensor forward(
		AutogradContext* ctx, torch::Tensor intensities, torch::Tensor positions, torch::Tensor orientation, torch::Tensor shift, int64_t x, int64_t y, int64_t z) {
		ctx->save_for_backward({ intensities, positions, orientation, shift});
		ctx->saved_data["x"] = x;
		ctx->saved_data["y"] = y;
		ctx->saved_data["z"] = z;
		at::AutoNonVariableTypeMode g;
		return projectAtoms(intensities, positions, orientation, shift, x, y, z);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = projectAtoms_backwards(grad_outputs[0], saved[0], saved[1], saved[2], saved[3], ctx->saved_data["x"].toInt(), ctx->saved_data["y"].toInt(), ctx->saved_data["z"].toInt());
		// Gradients for x,y,z: "Gradients of non-tensor arguments to forward must be `torch::Tensor()`."  https://pytorch.org/tutorials/advanced/cpp_autograd.html
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		return ret;
	}
};

torch::Tensor projectAtoms_cuda(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {

	return at::native::MyOperator::projectAtoms(intensities, positions, orientation, shift, z, y, x);
}

tensor_list projectAtoms_backwards_cuda(const torch::Tensor& grad_output, const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ret = at::native::MyOperator::projectAtoms_backward_cuda(grad_output, intensities, positions, orientation, shift, x, y, z);
	return  { std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret) };
}

torch::Tensor projectAtoms_autograd(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, const torch::Tensor& shift, int64_t x, int64_t y, int64_t z) {
	return ProjectAtoms::apply(intensities, positions, orientations, shift, x, y, z);
}

//Complex grid Sample part

torch::Tensor complexGridSample(const torch::Tensor& input, const torch::Tensor& grid) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::complexGridSample", "")
		.typed<decltype(complexGridSample)>();
	// Make sure that we have expected dimensionality for the lower level function calls
	//TORCH_CHECK(intensities.dim() == 2 && positions.dim() == 3 && orientation.dim() == 3 && shift.dim() == 2);
	//TORCH_CHECK(intensities.size(0) == positions.size(0) && orientation.size(0) == positions.size(0) && shift.size(0) == positions.size(0));
	return op.call(input, grid);
}

tensor_list complexGridSample_backwards(const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& grid) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::complexGridSample_backwards", "")
		.typed<decltype(complexGridSample_backwards)>();
	return op.call(grad_output, input, grid);
}

class ComplexGridSample : public Function<ComplexGridSample>
{
public:
	static torch::Tensor forward(
		AutogradContext* ctx, torch::Tensor input, torch::Tensor grid) {
		ctx->save_for_backward({ input, grid });

		at::AutoNonVariableTypeMode g;
		return complexGridSample(input, grid);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = complexGridSample_backwards(grad_outputs[0], saved[0], saved[1]);
		return ret;
	}
};

torch::Tensor complexGridSample_cuda(const torch::Tensor& input, const torch::Tensor& grid) {
	
	TORCH_CHECK(input.dim() == 4 || input.dim() == 5)
	if(input.dim() == 4)
		return at::native::MyOperator::complex_grid_sampler_2d_cuda(input, grid, static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
	else 
		return at::native::MyOperator::complex_grid_sampler_3d_cuda(input, grid, static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
}

tensor_list complexGridSample_backwards_cuda(const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& grid) {
	TORCH_CHECK(input.dim() == 4 || input.dim() == 5)
		if (input.dim() == 4) {
			std::tuple<torch::Tensor, torch::Tensor> ret = at::native::MyOperator::complex_grid_sampler_2d_backward_cuda(grad_output, input, grid,
				static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
			return  { std::get<0>(ret), std::get<1>(ret) };
		}
		else {
			std::tuple<torch::Tensor, torch::Tensor> ret = at::native::MyOperator::complex_grid_sampler_3d_backward_cuda(grad_output, input, grid,
				static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
			return  { std::get<0>(ret), std::get<1>(ret) };
		}
}


torch::Tensor complexGridSample_cpu(const torch::Tensor& input, const torch::Tensor& grid) {
	
	TORCH_CHECK(input.dim() == 4 || input.dim() == 5)
		if (input.dim() == 4)
			return at::native::ComplexGridSampler::complex_grid_sampler_cpu_2d(input, grid, static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
		else
			return at::native::ComplexGridSampler::complex_grid_sampler_cpu_3d(input, grid, static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
}

tensor_list complexGridSample_backwards_cpu(const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& grid) {
	TORCH_CHECK(input.dim() == 4 || input.dim() == 5)
		if (input.dim() == 4) {
			std::tuple<torch::Tensor, torch::Tensor> ret = at::native::ComplexGridSampler::complex_grid_sampler_cpu_2d_backward(grad_output, input, grid,
				static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
			return  { std::get<0>(ret), std::get<1>(ret) };
		}
		else {
			std::tuple<torch::Tensor, torch::Tensor> ret = at::native::ComplexGridSampler::complex_grid_sampler_cpu_3d_backward(grad_output, input, grid,
				static_cast<int>(at::native::GridSamplerInterpolation::Bilinear), static_cast<int>(at::native::GridSamplerPadding::Zeros), true);
			return  { std::get<0>(ret), std::get<1>(ret) };
		}
}


torch::Tensor complexGridSample_autograd(const torch::Tensor& input, const torch::Tensor& grid) {
	return ComplexGridSample::apply(input, grid);
}



// Dispatcher definitions

TORCH_LIBRARY(myops, m) {
	m.def("atoms_to_grid", atoms_to_grid);
	m.def("atoms_to_grid_backwards", atoms_to_grid_backwards);
	m.def("projectAtoms", projectAtoms);
	m.def("projectAtoms_backwards", projectAtoms_backwards);
	m.def("complexGridSample", complexGridSample);
	m.def("complexGridSample_backwards", complexGridSample_backwards);
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
	m.impl("atoms_to_grid", atoms_to_grid_cuda);
	m.impl("atoms_to_grid_backwards", atoms_to_grid_backwards_cuda);
	m.impl("projectAtoms", projectAtoms_cuda);
	m.impl("projectAtoms_backwards", projectAtoms_backwards_cuda);
	m.impl("complexGridSample", complexGridSample_cuda);
	m.impl("complexGridSample_backwards", complexGridSample_backwards_cuda);
}

TORCH_LIBRARY_IMPL(myops, CPU, m) {
	m.impl("complexGridSample", complexGridSample_cpu);
	m.impl("complexGridSample_backwards", complexGridSample_backwards_cpu);
}

TORCH_LIBRARY_IMPL(myops, Autograd, m) {
	m.impl("atoms_to_grid", atoms_to_grid_autograd);
	m.impl("projectAtoms", projectAtoms_autograd);
	m.impl("complexGridSample", complexGridSample_autograd);
}