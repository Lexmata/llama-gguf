#include <metal_stdlib>
using namespace metal;

// Dequantize Q6_K: 210-byte blocks, 256 elements each
// Block: [u8 ql[128], u8 qh[64], i8 scales[16], f16 d]
kernel void dequant_q6_k(
    device const uchar* raw [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint block_idx [[threadgroup_position_in_grid]],
    uint elem_idx [[thread_index_in_threadgroup]]
) {
    if (block_idx >= uint(num_blocks)) return;

    device const uchar* block = raw + block_idx * 210u;
    device const uchar* ql = block;
    device const uchar* qh = block + 128;
    device const char* scales = reinterpret_cast<device const char*>(block + 192);
    float d = float(*reinterpret_cast<device const half*>(block + 208));

    uint half_idx = elem_idx / 128u;
    uint within_half = elem_idx % 128u;
    uint l = within_half % 32u;
    uint quarter = within_half / 32u;

    uint ql_off = half_idx * 64u + l;
    uint qh_off = half_idx * 32u + l;
    uint sc_off = half_idx * 8u + (l / 16u);

    uint ql_val, qh_val;
    int scale;

    if (quarter == 0u) {
        ql_val = ql[ql_off] & 0x0Fu;
        qh_val = qh[qh_off] & 0x03u;
        scale = int(scales[sc_off]);
    } else if (quarter == 1u) {
        ql_val = ql[ql_off + 32u] & 0x0Fu;
        qh_val = (qh[qh_off] >> 2) & 0x03u;
        scale = int(scales[sc_off + 2u]);
    } else if (quarter == 2u) {
        ql_val = (ql[ql_off] >> 4) & 0x0Fu;
        qh_val = (qh[qh_off] >> 4) & 0x03u;
        scale = int(scales[sc_off + 4u]);
    } else {
        ql_val = (ql[ql_off + 32u] >> 4) & 0x0Fu;
        qh_val = (qh[qh_off] >> 6) & 0x03u;
        scale = int(scales[sc_off + 6u]);
    }

    int q = int(ql_val | (qh_val << 4)) - 32;
    result[block_idx * 256u + elem_idx] = d * float(scale) * float(q);
}
