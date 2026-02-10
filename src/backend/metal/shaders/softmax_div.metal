#include <metal_stdlib>
using namespace metal;

struct SoftmaxDivParams {
    int n;
    float inv_sum;
};

// Normalize by dividing by sum
kernel void softmax_div_f32(
    device float* data [[buffer(0)]],
    constant SoftmaxDivParams& params [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(params.n)) {
        data[idx] *= params.inv_sum;
    }
}
