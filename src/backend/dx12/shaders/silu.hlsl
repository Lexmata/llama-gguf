// SiLU activation: result[i] = x[i] / (1.0 + exp(-x[i]))

cbuffer Params : register(b0) {
    int n;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> result : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < (uint)n) {
        float val = x[idx];
        result[idx] = val / (1.0 + exp(-val));
    }
}
