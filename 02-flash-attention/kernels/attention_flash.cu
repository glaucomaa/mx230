// Flash Attention (forward, fp32), after Dao et al. 2022: never materializes
// the N x N score matrix. One thread owns one query row; its Q row, running
// max/sum and output accumulator live in registers, while K/V tiles are
// staged through shared memory. Online softmax: every new score rescales the
// accumulator by exp(m_old - m_new).
//
// Extra global memory: zero (vs N*N*4 bytes for the naive version).
#include <math_constants.h>

#define D 64  // head dim, fixed
#define BR 64 // query rows per block = threads per block
#define BC 32 // key/value columns per shared-memory tile

extern "C" __global__ void attn_flash(const float *Q, const float *K, const float *V,
                                      float *O, int N, int causal) {
    __shared__ float Ks[BC][D]; // 8 KB
    __shared__ float Vs[BC][D]; // 8 KB

    const int row = blockIdx.x * BR + (int)threadIdx.x;

    float q[D];
    if (row < N) {
#pragma unroll
        for (int x = 0; x < D; ++x) {
            q[x] = Q[row * D + x];
        }
    }

    float acc[D] = {0.0f};
    float m = -CUDART_INF_F;
    float l = 0.0f;
    const float scale = rsqrtf((float)D);
    // last row of this block — no tile beyond it contributes when causal
    const int block_last_row = blockIdx.x * BR + BR - 1;

    for (int j0 = 0; j0 < N; j0 += BC) {
        if (causal && j0 > block_last_row) {
            break;
        }
        // cooperative tile load: BR threads, BC*D elements each for K and V
        for (int t = threadIdx.x; t < BC * D; t += BR) {
            int j = t / D, x = t % D;
            if (j0 + j < N) {
                Ks[j][x] = K[(j0 + j) * D + x];
                Vs[j][x] = V[(j0 + j) * D + x];
            }
        }
        __syncthreads();

        if (row < N) {
            for (int j = 0; j < BC && j0 + j < N; ++j) {
                if (causal && j0 + j > row) {
                    break;
                }
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < D; ++x) {
                    s += q[x] * Ks[j][x];
                }
                s *= scale;

                float m_new = fmaxf(m, s);
                float corr = __expf(m - m_new);
                float p = __expf(s - m_new);
                l = l * corr + p;
#pragma unroll
                for (int x = 0; x < D; ++x) {
                    acc[x] = acc[x] * corr + p * Vs[j][x];
                }
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (row < N) {
        float inv = 1.0f / l;
#pragma unroll
        for (int x = 0; x < D; ++x) {
            O[row * D + x] = acc[x] * inv;
        }
    }
}
