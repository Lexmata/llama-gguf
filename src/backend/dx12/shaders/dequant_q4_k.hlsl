// Dequantize Q4_K: 144-byte blocks, 256 elements each
// Block: [f16 d, f16 dmin, u8 scales[12], u8 qs[128]]

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

uint read_u16(uint byte_offset) {
    return read_byte(byte_offset) | (read_byte(byte_offset + 1) << 8);
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint block_idx = gid.x;
    uint elem_idx = gtid.x;
    if (block_idx >= (uint)num_blocks) return;

    uint base = block_idx * 144;

    float d = half_to_float_manual(read_u16(base));
    float dmin = half_to_float_manual(read_u16(base + 2));
    uint sc_base = base + 4;
    uint qs_base = base + 16;

    uint sc[12];
    for (int i = 0; i < 12; i++) {
        sc[i] = read_byte(sc_base + (uint)i);
    }

    float scl[8];
    float mn[8];
    for (int j = 0; j < 4; j++) {
        scl[j] = (float)(sc[j] & 0x3F);
        mn[j] = (float)(sc[j + 4] & 0x3F);
    }
    for (int j = 4; j < 8; j++) {
        scl[j] = (float)((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4));
        mn[j] = (float)(((sc[j + 4] >> 4) & 0x0F) | ((sc[j] >> 6) << 4));
    }

    uint grp = elem_idx / 64;
    uint within_group = elem_idx % 64;
    uint is = grp * 2;
    uint qs_ptr = qs_base + grp * 32;

    float sc_val, mn_val;
    uint q;

    if (within_group < 32) {
        sc_val = d * scl[is];
        mn_val = dmin * mn[is];
        q = read_byte(qs_ptr + within_group) & 0x0F;
    } else {
        sc_val = d * scl[is + 1];
        mn_val = dmin * mn[is + 1];
        q = (read_byte(qs_ptr + within_group - 32) >> 4) & 0x0F;
    }

    result[block_idx * 256 + elem_idx] = sc_val * (float)q - mn_val;
}
