// Softmax pass 2: Compute exp(x - max) per element

cbuffer Params : register(b0) {
    int n;
    float max_value;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> result : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < (uint)n) {
        result[idx] = exp(x[idx] - max_value);
    }
}
