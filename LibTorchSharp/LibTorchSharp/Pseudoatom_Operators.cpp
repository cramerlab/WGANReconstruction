#include "Pseudoatom_Operators.h"
#include "Pseudoatom_Operators_backend.cuh"

using torch::Tensor;
using namespace torch::autograd;

TORCH_LIBRARY(myops, m) {
	m.def("atoms_to_grid(Tensor intensities, Tensor positions) -> Tensor");
	m.def("atoms_to_grid_backwards(Tensor grad_output, Tensor input, Tensor grid) -> Tensor[]");
	m.def("projectAtoms(Tensor intensities, Tensor positions, Tensor orientations) -> Tensor");
	m.def("projectAtoms_backwards(Tensor grad_output, Tensor intensities, Tensor positions, Tensor orientations) -> Tensor[]");
}

Tensor atoms_to_grid(const Tensor& intensities, const Tensor& positions) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::atoms_to_grid", "")
		.typed<decltype(atoms_to_grid)>();
	return op.call(intensities, positions);
}

tensor_list atoms_to_grid_backwards(const Tensor& grad_output, const Tensor& input, const Tensor& grid) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::atoms_to_grid_backwards", "")
		.typed<decltype(atoms_to_grid_backwards)>();
	return op.call(grad_output, input, grid);
}

int64_t dim = 8;

class Atoms_to_grid : public Function<Atoms_to_grid>
{
public:
	static Tensor forward(
		AutogradContext* ctx, Tensor intensities, Tensor positions) {
		ctx->save_for_backward({ intensities, positions });
		at::AutoNonVariableTypeMode g;
		return atoms_to_grid(intensities, positions);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = atoms_to_grid_backwards(grad_outputs[0], saved[0], saved[1]);
		return ret;
	}
};

Tensor atoms_to_grid_autograd(const Tensor& intensities, const Tensor& positions) {
	return Atoms_to_grid::apply(intensities, positions);
}

Tensor atoms_to_grid_cuda(const Tensor& intensities, const Tensor& positions) {
	int64_t x = dim;
	int64_t y = dim;
	int64_t z = dim;
	return at::native::MyOperator::atoms_to_grid_3d_cuda(intensities, positions, z, y, x);
}
tensor_list atoms_to_grid_backwards_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& grid) {
	std::tuple<Tensor, Tensor> ret = at::native::MyOperator::atoms_to_grid_3d_backward_cuda(grad_output, input, grid);
	return tensor_list({ std::get<0>(ret), std::get<1>(ret)});
}



Tensor projectAtoms(const Tensor& intensities, const Tensor& positions, const Tensor& orientation) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::projectAtoms", "")
		.typed<decltype(projectAtoms)>();
	return op.call(intensities, positions, orientation);
}

tensor_list projectAtoms_backwards(const Tensor& grad_output, const Tensor& intensities, const Tensor& positions, const Tensor& orientation) {
	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow("myops::projectAtoms_backwards", "")
		.typed<decltype(projectAtoms_backwards)>();
	return op.call(grad_output, intensities, positions, orientation);
}

class ProjectAtoms : public Function<ProjectAtoms>
{
public:
	static Tensor forward(
		AutogradContext* ctx, Tensor intensities, Tensor positions, Tensor orientation) {
		ctx->save_for_backward({ intensities, positions, orientation });
		at::AutoNonVariableTypeMode g;
		return projectAtoms(intensities, positions, orientation);
	}

	static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto ret = projectAtoms_backwards(grad_outputs[0], saved[0], saved[1], saved[2]);
		return ret;
	}
};

Tensor projectAtoms_cuda(const Tensor& intensities, const Tensor& positions, const Tensor& orientation) {
	int64_t x = dim;
	int64_t y = dim;
	int64_t z = dim;
	return at::native::MyOperator::projectAtoms(intensities, positions, orientation, z, y, x);
}

tensor_list projectAtoms_backwards_cuda(const Tensor& grad_output, const Tensor& intensities, const Tensor& positions, const Tensor& orientation) {
	int64_t x = dim;
	int64_t y = dim;
	int64_t z = dim;
	std::tuple<Tensor, Tensor, Tensor> ret = at::native::MyOperator::projectAtoms_backward_cuda(grad_output, intensities, positions, orientation, x,y,z);
	return  { std::get<0>(ret), std::get<1>(ret), std::get<2>(ret) };
}

Tensor projectAtoms_autograd(const Tensor& intensities, const Tensor& positions, const Tensor& orientations) {
	return ProjectAtoms::apply(intensities, positions, orientations);
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


Tensor fft_crop(const Tensor& input, int x, int y, int z) {
	
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
