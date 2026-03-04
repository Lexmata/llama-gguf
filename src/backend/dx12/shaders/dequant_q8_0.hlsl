// Dequantize Q8_0: 34-byte blocks, 32 elements each
// Block: [f16 d (2 bytes), i8 qs[32] (32 bytes)]

cbuffer Params : register(b0) {
    int num_blocks;
};

ByteAddressBuffer raw : register(t0);
RWStructuredBuffer<float> result : register(u0);

float half_to_float_manual(uint h) {
    uint sign = (h & 0x8000) << 16;
    uint expo = (h >> 10) & 0x1F;
    uint mant = h & 0x3FF;
    if (expo == 0) {
        if (mant == 0) return asfloat(sign);
        while ((mant & 0x400) == 0) { mant <<= 1; expo--; }
        expo++; mant &= 0x3FF;
    } else if (expo == 31) {
        return asfloat(sign | 0x7F800000 | (mant << 13));
    }
    return asfloat(sign | ((expo + 112) << 23) | (mant << 13));
}

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint block_idx = gid.x;
    uint elem_idx = gtid.x;
    if (block_idx >= (uint)num_blocks) return;

    uint byte_base = block_idx * 34;

    uint d_raw = raw.Load(byte_base & ~3);
    uint d_shift = (byte_base & 3) * 8;
    uint d_bits = (d_raw >> d_shift) & 0xFFFF;
    float d = half_to_float_manual(d_bits);

    uint qs_byte = byte_base + 2 + elem_idx;
    uint qs_raw = raw.Load(qs_byte & ~3);
    uint qs_shift = (qs_byte & 3) * 8;
    int qs_val = (int)((qs_raw >> qs_shift) & 0xFF);
    if (qs_val >= 128) qs_val -= 256;

    result[block_idx * 32 + elem_idx] = d * (float)qs_val;
}
