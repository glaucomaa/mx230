// v6, double buffering: like v5, but shared memory holds two tile buffers.
// While the current tile is being consumed from one buffer, the next tile's
// global loads are already in flight into registers, then stored into the
// other buffer — one __syncthreads per iteration instead of two, and global
// memory latency overlaps with FMAs.
// Requires M, N, K divisible by 128.
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void gemm_dbuf(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[2][BK * BM]; // transposed: As[buf][k][m]
    __shared__ float Bs[2][BK * BN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // preload tile 0 into buffer 0
    float4 a4 = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    float4 b4 = reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    As[0][(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
    As[0][(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
    As[0][(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
    As[0][(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
    reinterpret_cast<float4 *>(&Bs[0][innerRowB * BN + innerColB * 4])[0] = b4;
    __syncthreads();

    int buf = 0;
    for (int bk = 0; bk < K; bk += BK) {
        const bool has_next = bk + BK < K;
        if (has_next) {
            // issue next tile's global loads early; they overlap with compute below
            a4 = reinterpret_cast<const float4 *>(&A[innerRowA * K + (bk + BK) + innerColA * 4])[0];
            b4 = reinterpret_cast<const float4 *>(&B[(bk + BK + innerRowB) * N + innerColB * 4])[0];
        }

        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[buf][k * BM + threadRow * TM + i];
            }
#pragma unroll
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[buf][k * BN + threadCol * TN + i];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m * TN + n] += regM[m] * regN[n];
                }
            }
        }

        if (has_next) {
            As[buf ^ 1][(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
            As[buf ^ 1][(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
            As[buf ^ 1][(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
            As[buf ^ 1][(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
            reinterpret_cast<float4 *>(&Bs[buf ^ 1][innerRowB * BN + innerColB * 4])[0] = b4;
        }
        __syncthreads();
        buf ^= 1;
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
