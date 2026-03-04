#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    int m;  // rows of A / rows of C
    int K;  // cols of A / rows of B
    int n;  // cols of B / cols of C
};

// Tiled matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]
// A: [m, K] row-major, B: [K, n] row-major, C: [m, n] row-major
kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    uint row = gid.y;
    uint col = gid.x;
    uint localRow = tid.y;
    uint localCol = tid.x;

    float acc = 0.0f;
    uint numTiles = (uint(params.K) + TILE - 1) / TILE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE + localCol;
        uint bRow = t * TILE + localRow;

        tileA[localRow][localCol] = (row < uint(params.m) && aCol < uint(params.K))
            ? a[row * uint(params.K) + aCol] : 0.0f;
        tileB[localRow][localCol] = (bRow < uint(params.K) && col < uint(params.n))
            ? b[bRow * uint(params.n) + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += tileA[localRow][i] * tileB[i][localCol];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < uint(params.m) && col < uint(params.n)) {
        result[row * uint(params.n) + col] = acc;
    }
}
