// CUDA kernels and launchers for differentiable dynamics

#include <torch/extension.h>

template <typename scalar_t>
__global__ void dynamics_forward_kernel(
    const scalar_t* __restrict__ state,   // [batch, state_dim]
    const scalar_t* __restrict__ action,  // [batch, action_dim]
    const scalar_t* __restrict__ A,       // [state_dim, state_dim]
    const scalar_t* __restrict__ B,       // [state_dim, action_dim]
    scalar_t* __restrict__ out,           // [batch, state_dim]
    int batch,
    int state_dim,
    int action_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; // batch index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // state component index
    if (b >= batch || i >= state_dim) return;
    scalar_t acc = static_cast<scalar_t>(0);
    for (int j = 0; j < state_dim; ++j) {
        acc += state[b * state_dim + j] * A[i * state_dim + j];
    }
    for (int k = 0; k < action_dim; ++k) {
        acc += action[b * action_dim + k] * B[i * action_dim + k];
    }
    out[b * state_dim + i] = acc;
}

torch::Tensor dynamics_forward_cuda(
    const torch::Tensor& state,
    const torch::Tensor& action,
    const torch::Tensor& A,
    const torch::Tensor& B) {
    const int batch = state.size(0);
    const int state_dim = state.size(1);
    const int action_dim = action.size(1);
    auto out = torch::empty({batch, state_dim}, state.options());
    const dim3 threads(16, 16);
    const dim3 blocks((batch + threads.x - 1) / threads.x,
                      (state_dim + threads.y - 1) / threads.y);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(state.scalar_type(), "dynamics_forward_cuda", ([&] {
        dynamics_forward_kernel<scalar_t><<<blocks, threads>>>(
            state.data_ptr<scalar_t>(),
            action.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            state_dim,
            action_dim);
    }));
    return out;
}

// Backward: grad_state = grad_out @ A, grad_action = grad_out @ B

template <typename scalar_t>
__global__ void dynamics_backward_state_kernel(
    const scalar_t* __restrict__ grad_out, // [batch, state_dim]
    const scalar_t* __restrict__ A,        // [state_dim, state_dim]
    scalar_t* __restrict__ grad_state,     // [batch, state_dim]
    int batch,
    int state_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= batch || j >= state_dim) return;
    scalar_t acc = static_cast<scalar_t>(0);
    for (int i = 0; i < state_dim; ++i) {
        acc += grad_out[b * state_dim + i] * A[i * state_dim + j];
    }
    grad_state[b * state_dim + j] = acc;
}

template <typename scalar_t>
__global__ void dynamics_backward_action_kernel(
    const scalar_t* __restrict__ grad_out, // [batch, state_dim]
    const scalar_t* __restrict__ B,        // [state_dim, action_dim]
    scalar_t* __restrict__ grad_action,    // [batch, action_dim]
    int batch,
    int state_dim,
    int action_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= batch || k >= action_dim) return;
    scalar_t acc = static_cast<scalar_t>(0);
    for (int i = 0; i < state_dim; ++i) {
        acc += grad_out[b * state_dim + i] * B[i * action_dim + k];
    }
    grad_action[b * action_dim + k] = acc;
}

std::vector<torch::Tensor> dynamics_backward_cuda(
    const torch::Tensor& grad_out,
    const torch::Tensor& A,
    const torch::Tensor& B) {
    const int batch = grad_out.size(0);
    const int state_dim = grad_out.size(1);
    const int action_dim = B.size(1);
    auto opts = grad_out.options();
    auto grad_state = torch::empty({batch, state_dim}, opts);
    auto grad_action = torch::empty({batch, action_dim}, opts);
    const dim3 threads(16, 16);
    const dim3 blocks_state((batch + threads.x - 1) / threads.x,
                            (state_dim + threads.y - 1) / threads.y);
    const dim3 blocks_action((batch + threads.x - 1) / threads.x,
                             (action_dim + threads.y - 1) / threads.y);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "dynamics_backward_cuda", ([&] {
        dynamics_backward_state_kernel<scalar_t><<<blocks_state, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            grad_state.data_ptr<scalar_t>(),
            batch,
            state_dim);
        dynamics_backward_action_kernel<scalar_t><<<blocks_action, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            grad_action.data_ptr<scalar_t>(),
            batch,
            state_dim,
            action_dim);
    }));
    return {grad_state, grad_action};
}


