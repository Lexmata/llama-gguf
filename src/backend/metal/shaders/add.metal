#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        result[idx] = a[idx] + b[idx];
    }
}
