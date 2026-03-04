#include <metal_stdlib>
using namespace metal;

// Dequantize Q4_K: 144-byte blocks, 256 elements each
// Block: [f16 d, f16 dmin, u8 scales[12], u8 qs[128]]
kernel void dequant_q4_k(
    device const uchar* raw [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint block_idx [[threadgroup_position_in_grid]],
    uint elem_idx [[thread_index_in_threadgroup]]
) {
    if (block_idx >= uint(num_blocks)) return;

    device const uchar* block = raw + block_idx * 144u;

    float d = float(*reinterpret_cast<device const half*>(block));
    float dmin = float(*reinterpret_cast<device const half*>(block + 2));
    device const uchar* sc = block + 4;
    device const uchar* qs = block + 16;

    float scl[8];
    float mn[8];
    for (int j = 0; j < 4; j++) {
        scl[j] = float(sc[j] & 0x3F);
        mn[j] = float(sc[j + 4] & 0x3F);
    }
    for (int j = 4; j < 8; j++) {
        scl[j] = float((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4));
        mn[j] = float(((sc[j + 4] >> 4) & 0x0F) | ((sc[j] >> 6) << 4));
    }

    uint group = elem_idx / 64u;
    uint within_group = elem_idx % 64u;
    uint is = group * 2u;
    device const uchar* qs_ptr = qs + group * 32u;

    float sc_val, mn_val;
    uint q;

    if (within_group < 32u) {
        sc_val = d * scl[is];
        mn_val = dmin * mn[is];
        q = qs_ptr[within_group] & 0x0Fu;
    } else {
        sc_val = d * scl[is + 1u];
        mn_val = dmin * mn[is + 1u];
        q = (qs_ptr[within_group - 32u] >> 4) & 0x0Fu;
    }

    result[block_idx * 256u + elem_idx] = sc_val * float(q) - mn_val;
}
