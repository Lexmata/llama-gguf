#include <metal_stdlib>
using namespace metal;

struct ScaleParams {
    int n;
    float scalar;
};

kernel void scale_f32(
    device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant ScaleParams& params [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(params.n)) {
        result[idx] = a[idx] * params.scalar;
    }
}
