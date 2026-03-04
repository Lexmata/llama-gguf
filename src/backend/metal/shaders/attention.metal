#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    int num_heads;
    int num_kv_heads;
    int seq_len;
    int kv_len;
    int head_dim;
    float scale;
};

// Online-softmax multi-head attention with causal masking and GQA
// Q: [num_heads, seq_len, head_dim]  K: [num_kv_heads, kv_len, head_dim]
// V: [num_kv_heads, kv_len, head_dim]  Out: [num_heads, seq_len, head_dim]
// Dispatch: (num_heads, seq_len, 1) threadgroups of (256, 1, 1)
kernel void attention_f32(
    device const float* q_data [[buffer(0)]],
    device const float* k_data [[buffer(1)]],
    device const float* v_data [[buffer(2)]],
    device float* out_data [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint nt = 256;
    threadgroup float accum[256];
    threadgroup float reduction[256];
    threadgroup float s_max_score;
    threadgroup float s_sum_exp;
    threadgroup float s_weight;
    threadgroup float s_correction;

    uint head = gid.x;
    uint s = gid.y;
    uint kv_head = head / (uint(p.num_heads) / uint(p.num_kv_heads));
    uint q_abs_pos = uint(p.kv_len) - uint(p.seq_len) + s;

    for (uint d = tid; d < uint(p.head_dim); d += nt) {
        accum[d] = 0.0f;
    }
    if (tid == 0) {
        s_max_score = -3.402823466e+38f;
        s_sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint q_base = head * uint(p.seq_len) * uint(p.head_dim) + s * uint(p.head_dim);

    for (uint kv_pos = 0; kv_pos <= q_abs_pos && kv_pos < uint(p.kv_len); kv_pos++) {
        float local_dot = 0.0f;
        uint k_base = kv_head * uint(p.kv_len) * uint(p.head_dim) + kv_pos * uint(p.head_dim);
        for (uint d = tid; d < uint(p.head_dim); d += nt) {
            local_dot += q_data[q_base + d] * k_data[k_base + d];
        }

        reduction[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nt / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float score = reduction[0] * p.scale;

        if (tid == 0) {
            float old_max = s_max_score;
            if (score > old_max) {
                s_correction = exp(old_max - score);
                s_sum_exp *= s_correction;
                s_max_score = score;
            } else {
                s_correction = 1.0f;
            }
            s_weight = exp(score - s_max_score);
            s_sum_exp += s_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w = s_weight;
        float c = s_correction;

        uint v_base = kv_head * uint(p.kv_len) * uint(p.head_dim) + kv_pos * uint(p.head_dim);
        for (uint d = tid; d < uint(p.head_dim); d += nt) {
            accum[d] = accum[d] * c + w * v_data[v_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        s_weight = (s_sum_exp > 0.0f) ? 1.0f / s_sum_exp : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = s_weight;
    uint out_base = head * uint(p.seq_len) * uint(p.head_dim) + s * uint(p.head_dim);
    for (uint d = tid; d < uint(p.head_dim); d += nt) {
        out_data[out_base + d] = accum[d] * inv_sum;
    }
}
