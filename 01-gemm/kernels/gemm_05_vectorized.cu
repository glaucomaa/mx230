// v5, vectorized: like v4, but global loads are float4 (128-bit transactions)
// and the A tile is stored transposed in shared memory so inner-loop reads of
// As are contiguous too. C is written with float4 as well.
// Requires M, N, K divisible by 128.
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void gemm_vectorized(const float *A, const float *B, float *C,
                                           int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK * BM]; // transposed: As[k][m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // 256 threads, float4: A tile BM*BK = 1024 elems = 256 float4 — exactly one each
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    // B tile BK*BN = 1024 elems = 256 float4
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[k * BM + threadRow * TM + i];
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
        for (int n = 0; n < TN; n += 4) {
            float4 out;
            out.x = acc[m * TN + n + 0];
            out.y = acc[m * TN + n + 1];
            out.z = acc[m * TN + n + 2];
            out.w = acc[m * TN + n + 3];
            reinterpret_cast<float4 *>(&C[(threadRow * TM + m) * N + threadCol * TN + n])[0] = out;
        }
    }
}
