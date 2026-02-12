#include <metal_stdlib>
using namespace metal;

kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        float val = x[idx];
        result[idx] = val / (1.0f + exp(-val));
    }
}
