#include <metal_stdlib>
using namespace metal;

struct SoftmaxExpParams {
    int n;
    float max_value;
};

// Compute exp(x - max) per element
kernel void softmax_exp_f32(
    device const float* x [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant SoftmaxExpParams& params [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(params.n)) {
        result[idx] = exp(x[idx] - params.max_value);
    }
}
