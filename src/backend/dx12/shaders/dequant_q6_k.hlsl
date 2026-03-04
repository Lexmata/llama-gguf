// Dequantize Q6_K: 210-byte blocks, 256 elements each
// Block: [u8 ql[128], u8 qh[64], i8 scales[16], f16 d]

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

uint read_byte(uint byte_offset) {
    uint word = raw.Load(byte_offset & ~3);
    return (word >> ((byte_offset & 3) * 8)) & 0xFF;
}

int read_i8(uint byte_offset) {
    int v = (int)read_byte(byte_offset);
    if (v >= 128) v -= 256;
    return v;
}

uint read_u16(uint byte_offset) {
    return read_byte(byte_offset) | (read_byte(byte_offset + 1) << 8);
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint block_idx = gid.x;
    uint elem_idx = gtid.x;
    if (block_idx >= (uint)num_blocks) return;

    uint base = block_idx * 210;
    uint ql_base = base;
    uint qh_base = base + 128;
    uint sc_base = base + 192;
    float d = half_to_float_manual(read_u16(base + 208));

    uint half_idx = elem_idx / 128;
    uint within_half = elem_idx % 128;
    uint l = within_half % 32;
    uint quarter = within_half / 32;

    uint ql_off = half_idx * 64 + l;
    uint qh_off = half_idx * 32 + l;
    uint sc_off = half_idx * 8 + (l / 16);

    uint ql_val, qh_val;
    int scale;

    if (quarter == 0) {
        ql_val = read_byte(ql_base + ql_off) & 0x0F;
        qh_val = read_byte(qh_base + qh_off) & 0x03;
        scale = read_i8(sc_base + sc_off);
    } else if (quarter == 1) {
        ql_val = read_byte(ql_base + ql_off + 32) & 0x0F;
        qh_val = (read_byte(qh_base + qh_off) >> 2) & 0x03;
        scale = read_i8(sc_base + sc_off + 2);
    } else if (quarter == 2) {
        ql_val = (read_byte(ql_base + ql_off) >> 4) & 0x0F;
        qh_val = (read_byte(qh_base + qh_off) >> 4) & 0x03;
        scale = read_i8(sc_base + sc_off + 4);
    } else {
        ql_val = (read_byte(ql_base + ql_off + 32) >> 4) & 0x0F;
        qh_val = (read_byte(qh_base + qh_off) >> 6) & 0x03;
        scale = read_i8(sc_base + sc_off + 6);
    }

    int q = (int)(ql_val | (qh_val << 4)) - 32;
    result[block_idx * 256 + elem_idx] = d * (float)scale * (float)q;
}
