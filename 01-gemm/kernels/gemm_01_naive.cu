// v1, naive: one thread per C element; threadIdx.x walks rows, so adjacent
// threads write C with stride N and read B down the same column — global
// memory accesses are not coalesced.
// C[M,N] = A[M,K] * B[K,N], all row-major.
extern "C" __global__ void gemm_naive(const float *A, const float *B, float *C,
                                      int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
