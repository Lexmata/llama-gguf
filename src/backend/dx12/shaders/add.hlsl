// Element-wise addition: result[i] = a[i] + b[i]

cbuffer Params : register(b0) {
    int n;
};

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> result : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < (uint)n) {
        result[idx] = a[idx] + b[idx];
    }
}
