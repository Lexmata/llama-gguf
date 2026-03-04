// Tiled matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]
// A: [m, K] row-major, B: [K, n] row-major, C: [m, n] row-major

cbuffer Params : register(b0) {
    int m;  // rows of A / rows of C
    int K;  // cols of A / rows of B
    int n;  // cols of B / cols of C
};

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> result : register(u2);

groupshared float tileA[16][16];
groupshared float tileB[16][16];

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    uint row = dtid.y;
    uint col = dtid.x;
    uint localRow = gtid.y;
    uint localCol = gtid.x;

    float acc = 0.0;
    uint numTiles = ((uint)K + 15) / 16;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * 16 + localCol;
        uint bRow = t * 16 + localRow;

        tileA[localRow][localCol] = (row < (uint)m && aCol < (uint)K)
            ? a[row * (uint)K + aCol] : 0.0;
        tileB[localRow][localCol] = (bRow < (uint)K && col < (uint)n)
            ? b[bRow * (uint)n + col] : 0.0;

        GroupMemoryBarrierWithGroupSync();

        for (uint i = 0; i < 16; i++) {
            acc += tileA[localRow][i] * tileB[i][localCol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < (uint)m && col < (uint)n) {
        result[row * (uint)n + col] = acc;
    }
}
