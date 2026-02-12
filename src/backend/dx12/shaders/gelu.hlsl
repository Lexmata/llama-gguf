// GELU activation (tanh approximation):
// result[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

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
        float c = 0.7978845608; // sqrt(2/pi)
        float inner = c * (val + 0.044715 * val * val * val);
        // Use exp-based form: 0.5*x*(1+tanh(z)) = x/(1+exp(-2z))
        // Avoids tanh() which is unreliable on some GPU drivers (WARP)
        result[idx] = val / (1.0 + exp(-2.0 * inner));
    }
}
