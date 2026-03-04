// Online-softmax cached attention for single-token generation with GQA
// Q: [num_heads, 1, head_dim]  K_cache: [num_kv_heads, max_seq_len, head_dim]
// V_cache: [num_kv_heads, max_seq_len, head_dim]  Out: [num_heads, 1, head_dim]
// kv_len = valid positions, max_seq_len = stride between heads
// Dispatch: (num_heads, 1, 1) groups of (256, 1, 1)

cbuffer Params : register(b0) {
    int num_heads;
    int num_kv_heads;
    int kv_len;
    int max_seq_len;
    int head_dim;
    float scale;
};

RWStructuredBuffer<float> q_data : register(u0);
RWStructuredBuffer<float> k_data : register(u1);
RWStructuredBuffer<float> v_data : register(u2);
RWStructuredBuffer<float> out_data : register(u3);

groupshared float accum[256];
groupshared float reduction[256];
groupshared float s_max_score;
groupshared float s_sum_exp;
groupshared float s_weight;
groupshared float s_correction;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint head = gid.x;
    uint tid = gtid.x;
    const uint nt = 256;

    uint kv_head = head / ((uint)num_heads / (uint)num_kv_heads);

    for (uint d = tid; d < (uint)head_dim; d += nt) {
        accum[d] = 0.0;
    }
    if (tid == 0) {
        s_max_score = -3.402823466e+38;
        s_sum_exp = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint q_base = head * (uint)head_dim;

    for (uint kv_pos = 0; kv_pos < (uint)kv_len; kv_pos++) {
        float local_dot = 0.0;
        uint k_base = kv_head * (uint)max_seq_len * (uint)head_dim + kv_pos * (uint)head_dim;
        for (uint d = tid; d < (uint)head_dim; d += nt) {
            local_dot += q_data[q_base + d] * k_data[k_base + d];
        }

        reduction[tid] = local_dot;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = nt / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        float score = reduction[0] * scale;

        if (tid == 0) {
            float old_max = s_max_score;
            if (score > old_max) {
                s_correction = exp(old_max - score);
                s_sum_exp *= s_correction;
                s_max_score = score;
            } else {
                s_correction = 1.0;
            }
            s_weight = exp(score - s_max_score);
            s_sum_exp += s_weight;
        }
        GroupMemoryBarrierWithGroupSync();

        float w = s_weight;
        float c = s_correction;

        uint v_base = kv_head * (uint)max_seq_len * (uint)head_dim + kv_pos * (uint)head_dim;
        for (uint d = tid; d < (uint)head_dim; d += nt) {
            accum[d] = accum[d] * c + w * v_data[v_base + d];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        s_weight = (s_sum_exp > 0.0) ? 1.0 / s_sum_exp : 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    float inv_sum = s_weight;
    uint out_base = head * (uint)head_dim;
    for (uint d = tid; d < (uint)head_dim; d += nt) {
        out_data[out_base + d] = accum[d] * inv_sum;
    }
}
