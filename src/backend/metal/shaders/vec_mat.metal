#include <metal_stdlib>
using namespace metal;

struct VecMatParams {
    int k;  // inner dimension
    int n;  // output dimension
};

// Vector-matrix multiply: out[j] = sum_i(a[i] * W[i,j])
// a: [k], b: [k, n] (GGUF column-major: W[i,j] at index i + j*k), out: [n]
kernel void vec_mat_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant VecMatParams& params [[buffer(3)]],
    uint j [[thread_position_in_grid]]
) {
    if (j >= uint(params.n)) return;

    float acc = 0.0f;
    for (int i = 0; i < params.k; i++) {
        acc += a[i] * b[i + j * params.k];
    }
    result[j] = acc;
}
