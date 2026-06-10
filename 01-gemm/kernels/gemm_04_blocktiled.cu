// v4, register blocking (2D block tiling): a block of 256 threads computes a
// BM x BN tile of C; each thread keeps its own TM x TN subtile in registers.
// FMAs per shared-memory access grow by ~TM*TN.
// Requires M, N, K divisible by 128.
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void gemm_blocktiled(const float *A, const float *B, float *C,
                                           int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int numThreads = (BM * BN) / (TM * TN); // 256

    const int threadCol = threadIdx.x % (BN / TN); // 0..15
    const int threadRow = threadIdx.x / (BN / TN); // 0..15

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA = numThreads / BK; // 32 rows per pass
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB = numThreads / BN; // 2 rows per pass

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        for (int off = 0; off < BM; off += strideA) {
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
        }
        for (int off = 0; off < BK; off += strideB) {
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + k];
            }
#pragma unroll
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[k * BN + threadCol * TN + i];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m * TN + n] += regM[m] * regN[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            C[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
        }
    }
}
