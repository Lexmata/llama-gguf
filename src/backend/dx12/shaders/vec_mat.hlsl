// Vector-matrix multiply: out[j] = sum_i(a[i] * W[i,j])
// a: [k], b: [k, n] (GGUF column-major: W[i,j] at index i + j*k), out: [n]

cbuffer Params : register(b0) {
    int k; // inner dimension
    int n; // output dimension
};

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> result : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint j = dtid.x;
    if (j >= (uint)n) return;

    float acc = 0.0;
    for (int i = 0; i < k; i++) {
        acc += a[i] * b[i + j * k];
    }
    result[j] = acc;
}
