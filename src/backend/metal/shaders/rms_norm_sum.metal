#include <metal_stdlib>
using namespace metal;

// Compute partial sum of squares per threadgroup
kernel void rms_norm_sum_f32(
    device const float* x [[buffer(0)]],
    device float* partial_sum [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float sdata[256];

    float val = (idx < uint(n)) ? x[idx] : 0.0f;
    sdata[tid] = val * val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sum[tg_id] = sdata[0];
    }
}
