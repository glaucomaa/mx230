// v3, shared-memory tiling: a 32x32 block loads tiles of A and B into shared
// memory, cutting global reads per element by a factor of TILE.
// Requires M, N, K divisible by 32.
#define TILE 32

extern "C" __global__ void gemm_tiled(const float *A, const float *B, float *C,
                                      int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = A[row * K + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}
