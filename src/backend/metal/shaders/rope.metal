#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    int num_q_heads;
    int num_k_heads;
    int head_dim;
    int position;
    float freq_base;
    float freq_scale;
    int use_neox; // 0 = normal, 1 = NeoX style
};

// Rotary Position Embedding for single-position inference
// q/k: [num_heads, head_dim] (interleaved)
kernel void rope_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant RopeParams& params [[buffer(2)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint pair_idx [[thread_index_in_threadgroup]]
) {
    if (pair_idx >= uint(params.head_dim / 2)) return;

    // Compute frequency for this dimension pair
    float freq = params.freq_scale / pow(params.freq_base, float(2 * pair_idx) / float(params.head_dim));
    float angle = float(params.position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Apply RoPE to query (if within q head count)
    if (head_idx < uint(params.num_q_heads)) {
        uint base = head_idx * uint(params.head_dim);
        uint i0, i1;
        if (params.use_neox != 0) {
            // NeoX: first half + second half
            i0 = base + pair_idx;
            i1 = base + pair_idx + uint(params.head_dim / 2);
        } else {
            // Normal: consecutive pairs
            i0 = base + 2 * pair_idx;
            i1 = base + 2 * pair_idx + 1;
        }

        float q0 = q[i0];
        float q1 = q[i1];
        q[i0] = q0 * cos_val - q1 * sin_val;
        q[i1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply RoPE to key (if within k head count)
    if (head_idx < uint(params.num_k_heads)) {
        uint base = head_idx * uint(params.head_dim);
        uint i0, i1;
        if (params.use_neox != 0) {
            i0 = base + pair_idx;
            i1 = base + pair_idx + uint(params.head_dim / 2);
        } else {
            i0 = base + 2 * pair_idx;
            i1 = base + 2 * pair_idx + 1;
        }

        float k0 = k[i0];
        float k1 = k[i1];
        k[i0] = k0 * cos_val - k1 * sin_val;
        k[i1] = k0 * sin_val + k1 * cos_val;
    }
}
