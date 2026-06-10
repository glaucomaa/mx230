// v2, coalesced: same as v1, but threadIdx.x walks columns — adjacent threads
// of a warp read adjacent B elements and write adjacent C elements, so global
// memory transactions coalesce.
extern "C" __global__ void gemm_coalesced(const float *A, const float *B, float *C,
                                          int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
