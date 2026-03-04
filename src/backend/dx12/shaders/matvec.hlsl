// Matrix-vector multiply: out[i] = sum_j A[i,j] * x[j]
// A: [m, n] row-major, x: [n], out: [m]

cbuffer Params : register(b0) {
    int mm;  // rows of A / output dimension
    int n;   // cols of A / input dimension
};

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> x : register(u1);
RWStructuredBuffer<float> result : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= (uint)mm) return;

    float acc = 0.0;
    uint row_base = i * (uint)n;
    for (int j = 0; j < n; j++) {
        acc += a[row_base + (uint)j] * x[j];
    }
    result[i] = acc;
}
