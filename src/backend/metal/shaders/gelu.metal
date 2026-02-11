#include <metal_stdlib>
using namespace metal;

kernel void gelu_f32(
    device const float* x [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        float val = x[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float c = 0.7978845608f; // sqrt(2/pi)
        float inner = c * (val + 0.044715f * val * val * val);
        inner = clamp(inner, -10.0f, 10.0f);
        result[idx] = 0.5f * val * (1.0f + tanh(inner));
    }
}
