// Scale: result[i] = a[i] * scalar

cbuffer Params : register(b0) {
    int n;
    float scalar;
};

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> result : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < (uint)n) {
        result[idx] = a[idx] * scalar;
    }
}
