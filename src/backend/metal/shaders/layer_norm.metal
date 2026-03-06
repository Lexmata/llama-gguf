#include <metal_stdlib>
using namespace metal;

// LayerNorm: out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
// Single-workgroup three-pass approach using threadgroup memory.
// Handles hidden sizes up to ~16K elements with 256 threads.

struct LayerNormParams {
    int n;
    float eps;
};

kernel void layer_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* result [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float sdata[256];
    int n = params.n;

    // Pass 1: compute sum for mean (parallel reduction)
    float local_sum = 0.0f;
    for (int i = int(tid); i < n; i += 256) {
        local_sum += x[i];
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: compute variance (parallel reduction)
    float local_var = 0.0f;
    for (int i = int(tid); i < n; i += 256) {
        float d = x[i] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(sdata[0] / float(n) + params.eps);

    // Pass 3: normalize, scale by weight, add bias (element-wise)
    for (int i = int(tid); i < n; i += 256) {
        float normed = (x[i] - mean) * inv_std;
        result[i] = normed * w[i] + b[i];
    }
}
