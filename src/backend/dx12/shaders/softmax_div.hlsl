// Softmax pass 3: Normalize by dividing by sum

cbuffer Params : register(b0) {
    int n;
    float inv_sum;
};

RWStructuredBuffer<float> data : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < (uint)n) {
        data[idx] *= inv_sum;
    }
}
