// C++/Pybind11 wrapper for CUDA differentiable dynamics.

#include <torch/extension.h>

// CUDA launchers
torch::Tensor dynamics_forward_cuda(
    const torch::Tensor& state,
    const torch::Tensor& action,
    const torch::Tensor& A,
    const torch::Tensor& B);

std::vector<torch::Tensor> dynamics_backward_cuda(
    const torch::Tensor& grad_out,
    const torch::Tensor& A,
    const torch::Tensor& B);

// Forward binding
torch::Tensor dynamics_forward(
    const torch::Tensor& state,
    const torch::Tensor& action,
    const torch::Tensor& A,
    const torch::Tensor& B) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(action.is_cuda(), "action must be CUDA");
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(state.dim() == 2, "state must be [batch, state_dim]");
    TORCH_CHECK(action.dim() == 2, "action must be [batch, action_dim]");
    TORCH_CHECK(A.dim() == 2, "A must be [state_dim, state_dim]");
    TORCH_CHECK(B.dim() == 2, "B must be [state_dim, action_dim]");
    return dynamics_forward_cuda(state, action, A, B);
}

// Backward binding
std::vector<torch::Tensor> dynamics_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& A,
    const torch::Tensor& B) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be [batch, state_dim]");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B dims invalid");
    return dynamics_backward_cuda(grad_out, A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dynamics_forward, "Dynamics forward (CUDA)");
    m.def("backward", &dynamics_backward, "Dynamics backward (CUDA)");
}


