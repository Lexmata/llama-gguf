// RMS norm pass 1: Compute partial sum of squares per workgroup

cbuffer Params : register(b0) {
    int n;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> partial_sum : register(u1);

groupshared float sdata[256];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tid = gtid.x;
    uint idx = dtid.x;

    float val = (idx < (uint)n) ? x[idx] : 0.0;
    sdata[tid] = val * val;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        partial_sum[gid.x] = sdata[0];
    }
}
