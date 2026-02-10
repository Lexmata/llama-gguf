// Rotary Position Embedding for single-position inference
// q/k: [num_heads, head_dim] (interleaved)

cbuffer Params : register(b0) {
    int num_q_heads;
    int num_k_heads;
    int head_dim;
    int position;
    float freq_base;
    float freq_scale;
    int use_neox; // 0 = normal, 1 = NeoX style
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> kk : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint head_idx = gid.x;
    uint pair_idx = gtid.x;

    if (pair_idx >= (uint)(head_dim / 2)) return;

    // Compute frequency for this dimension pair
    float freq = freq_scale / pow(freq_base, float(2 * pair_idx) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Apply RoPE to query (if within q head count)
    if (head_idx < (uint)num_q_heads) {
        uint base_off = head_idx * head_dim;
        uint i0, i1;
        if (use_neox != 0) {
            // NeoX: first half + second half
            i0 = base_off + pair_idx;
            i1 = base_off + pair_idx + head_dim / 2;
        } else {
            // Normal: consecutive pairs
            i0 = base_off + 2 * pair_idx;
            i1 = base_off + 2 * pair_idx + 1;
        }

        float q0 = q[i0];
        float q1 = q[i1];
        q[i0] = q0 * cos_val - q1 * sin_val;
        q[i1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply RoPE to key (if within k head count)
    if (head_idx < (uint)num_k_heads) {
        uint base_off = head_idx * head_dim;
        uint i0, i1;
        if (use_neox != 0) {
            i0 = base_off + pair_idx;
            i1 = base_off + pair_idx + head_dim / 2;
        } else {
            i0 = base_off + 2 * pair_idx;
            i1 = base_off + 2 * pair_idx + 1;
        }

        float k0 = kk[i0];
        float k1 = kk[i1];
        kk[i0] = k0 * cos_val - k1 * sin_val;
        kk[i1] = k0 * sin_val + k1 * cos_val;
    }
}
