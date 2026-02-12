// Softmax pass 1: Find max value per workgroup via parallel reduction

cbuffer Params : register(b0) {
    int n;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> partial_max : register(u1);

groupshared float sdata[256];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tid = gtid.x;
    uint idx = dtid.x;

    sdata[tid] = (idx < (uint)n) ? x[idx] : -3.402823466e+38;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        partial_max[gid.x] = sdata[0];
    }
}
