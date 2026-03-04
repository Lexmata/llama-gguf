#include <metal_stdlib>
using namespace metal;

struct MatvecParams {
    int m;  // rows of A / output dimension
    int n;  // cols of A / input dimension
};

// Matrix-vector multiply: out[i] = sum_j A[i,j] * x[j]
// A: [m, n] row-major, x: [n], out: [m]
kernel void matvec_f32(
    device const float* a [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant MatvecParams& params [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(params.m)) return;

    float acc = 0.0f;
    uint row_base = i * uint(params.n);
    for (int j = 0; j < params.n; j++) {
        acc += a[row_base + uint(j)] * x[j];
    }
    result[i] = acc;
}
