#include <metal_stdlib>
using namespace metal;

// Dequantize Q8_0: 34-byte blocks, 32 elements each
// Block: [f16 d (2 bytes), i8 qs[32] (32 bytes)]
kernel void dequant_q8_0(
    device const uchar* raw [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint block_idx [[threadgroup_position_in_grid]],
    uint elem_idx [[thread_index_in_threadgroup]]
) {
    if (block_idx >= uint(num_blocks)) return;

    uint byte_base = block_idx * 34u;

    half d = *reinterpret_cast<device const half*>(raw + byte_base);

    char qs_val = *reinterpret_cast<device const char*>(raw + byte_base + 2u + elem_idx);

    result[block_idx * 32u + elem_idx] = float(d) * float(qs_val);
}
