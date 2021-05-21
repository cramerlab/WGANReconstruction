#include "Pseudoatom_Operators.h"
#include "Pseudoatom_Operators_backend.cuh"


using namespace torch::autograd;

TORCH_LIBRARY(myops, m) {
	m.def("atoms_to_grid(Tensor intensities, torch::Tensor positions, int64_t x, int64_t y, int64_t z) -> torch::Tensor");
	m.def("atoms_to_grid_backwards(Tensor grad_output, torch::Tensor input, torch::Tensor grid) -> torch::Tensor[]");
	m.def("projectAtoms(Tensor intensities, torch::Tensor positions, torch::Tensor orientations, int64_t x, int64_t y, int64_t z) -> torch::Tensor");
	m.def("projectAtoms_backwards(Tensor grad_output, torch::Tensor intensities, torch::Tensor positions, torch::Tensor orientations, int64_t x, int64_t y, int64_t z) -> torch::Tensor[]");
}

torch::Tensor atoms_to_grid(const torch::Tensor& intensities, const torch::Tensor& positions, int64_t x, int64_t y, int64_t z) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::atoms_to_grid", "")
		.typed<decltype(atoms_to_grid)>();
	return op.call(intensities, positions, x, y, z);
}

tensor_list atoms_to_grid_backwards(const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& grid) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::atoms_to_grid_backwards", "")
		.typed<decltype(atoms_to_grid_backwards)>();
	return op.call(grad_output, input, grid);
}

int64_t dim = 8;

class Atoms_to_grid : public Function<Atoms_to_grid>
{
public:
	static torch::Tensor forward(
		AutogradContext* ctx, torch::Tensor intensities, torch::Tensor positions, int64_t x, int64_t y, int64_t z) {
		ctx->save_for_backward({ intensities, positions });
		ctx->saved_data["x"] = x;
		ctx->saved_data["y"] = y;
		ctx->saved_data["z"] = z;
		at::AutoNonVariableTypeMode g;
		return atoms_to_grid(intensities, positions, x, y, z);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = atoms_to_grid_backwards(grad_outputs[0], saved[0], saved[1]);
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		return ret;
	}
};

torch::Tensor atoms_to_grid_autograd(const torch::Tensor& intensities, const torch::Tensor& positions, int64_t x, int64_t y, int64_t z) {
	return Atoms_to_grid::apply(intensities, positions, x, y, z);
}

torch::Tensor atoms_to_grid_cuda(const torch::Tensor& intensities, const torch::Tensor& positions, int64_t x, int64_t y, int64_t z) {
	return at::native::MyOperator::atoms_to_grid_3d_cuda(intensities, positions, z, y, x);
}
tensor_list atoms_to_grid_backwards_cuda(const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& grid) {
	std::tuple<torch::Tensor, torch::Tensor> ret = at::native::MyOperator::atoms_to_grid_3d_backward_cuda(grad_output, input, grid);
	return tensor_list({ std::get<0>(ret), std::get<1>(ret)});
}



torch::Tensor projectAtoms(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, int64_t x, int64_t y, int64_t z) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::projectAtoms", "")
		.typed<decltype(projectAtoms)>();
	return op.call(intensities, positions, orientation, x, y, z);
}

tensor_list projectAtoms_backwards(const torch::Tensor& grad_output, const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, int64_t x, int64_t y, int64_t z) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::projectAtoms_backwards", "")
		.typed<decltype(projectAtoms_backwards)>();
	return op.call(grad_output, intensities, positions, orientation, x, y, z);
}

class ProjectAtoms : public Function<ProjectAtoms>
{
public:
	static torch::Tensor forward(
		AutogradContext* ctx, torch::Tensor intensities, torch::Tensor positions, torch::Tensor orientation, int64_t x, int64_t y, int64_t z) {
		ctx->save_for_backward({ intensities, positions, orientation});
		ctx->saved_data["x"] = x;
		ctx->saved_data["y"] = y;
		ctx->saved_data["z"] = z;
		at::AutoNonVariableTypeMode g;
		return projectAtoms(intensities, positions, orientation, x, y, z);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = projectAtoms_backwards(grad_outputs[0], saved[0], saved[1], saved[2], ctx->saved_data["x"].toInt(), ctx->saved_data["y"].toInt(), ctx->saved_data["z"].toInt());
		// Gradients for x,y,z: "Gradients of non-tensor arguments to forward must be `torch::Tensor()`."  https://pytorch.org/tutorials/advanced/cpp_autograd.html
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		ret.push_back(torch::Tensor());
		return ret;
	}
};

torch::Tensor projectAtoms_cuda(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, int64_t x, int64_t y, int64_t z) {

	return at::native::MyOperator::projectAtoms(intensities, positions, orientation, z, y, x);
}

tensor_list projectAtoms_backwards_cuda(const torch::Tensor& grad_output, const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientation, int64_t x, int64_t y, int64_t z) {
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ret = at::native::MyOperator::projectAtoms_backward_cuda(grad_output, intensities, positions, orientation, x, y, z);
	return  { std::get<0>(ret), std::get<1>(ret), std::get<2>(ret) };
}

torch::Tensor projectAtoms_autograd(const torch::Tensor& intensities, const torch::Tensor& positions, const torch::Tensor& orientations, int64_t x, int64_t y, int64_t z) {
	return ProjectAtoms::apply(intensities, positions, orientations, x, y, z);
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
	m.impl("atoms_to_grid", atoms_to_grid_cuda);
	m.impl("atoms_to_grid_backwards", atoms_to_grid_backwards_cuda);
	m.impl("projectAtoms", projectAtoms_cuda);
	m.impl("projectAtoms_backwards", projectAtoms_backwards_cuda);
}
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
	m.impl("atoms_to_grid", atoms_to_grid_autograd);
	m.impl("projectAtoms", projectAtoms_autograd);
}


torch::Tensor fft_crop(const torch::Tensor& input, int x, int y, int z) {
	
	if (input.dim() < 4) {
		//Assuming we only have HDW or DW format, we need to add dimensions until we have N, H, D, W format by adding size 1 axes
		auto reshaped = input.unsqueeze(0);
		while (reshaped.dim() != 4) {
			reshaped = reshaped.unsqueeze(0);
		}
		return at::native::MyOperator::fft_crop(reshaped, { x,y,z });
	}
	if (input.dim() > 4)
	{
		std::cerr << "Image Crop at " << __FILE__ << ": " << __LINE__ << " Don't know how to process an image with dimension " << input.dim() << std::endl;
		exit(1);
	}
	return at::native::MyOperator::fft_crop(input, { x,y,z });
}
