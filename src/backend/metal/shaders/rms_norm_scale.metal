#include <metal_stdlib>
using namespace metal;

struct RmsNormScaleParams {
    int n;
    float rms_inv;
};

// Normalize and scale by weight
kernel void rms_norm_scale_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant RmsNormScaleParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(params.n)) {
        result[idx] = x[idx] * params.rms_inv * w[idx];
    }
}
