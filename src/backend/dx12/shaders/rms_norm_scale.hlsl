// RMS norm pass 2: Normalize and scale by weight

cbuffer Params : register(b0) {
    int n;
    float rms_inv;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> w : register(u1);
RWStructuredBuffer<float> result : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < (uint)n) {
        result[idx] = x[idx] * rms_inv * w[idx];
    }
}
