// Flash Attention (backward, fp32). Closes the training half of stage 2.
//
// Given Q, K, V, the forward output O, the upstream gradient dO and the per-row
// log-sum-exp L from the forward, compute dQ, dK, dV. The N x N probability
// matrix is never stored: each score s_ij = scale * q_i . k_j is recomputed and
// p_ij = exp(s_ij - L_i) follows from the single saved scalar L_i.
//
// Recipe (Dao et al. 2022, eq. for the softmax-attention VJP):
//   Delta_i = sum_x dO[i,x] * O[i,x]           (= sum_j p_ij * dP_ij)
//   dP_ij   = dO_i . v_j
//   dS_ij   = p_ij * (dP_ij - Delta_i) * scale (grad wrt q_i . k_j)
//   dV_j   += p_ij * dO_i
//   dK_j   += dS_ij * q_i
//   dQ_i   += dS_ij * k_j
//
// One thread owns one query row i: it accumulates dQ_i privately in registers,
// but dK_j / dV_j are touched by every query row that attends key j, so those
// scatter with atomicAdd (native fp32 on sm_61). dQ/dK/dV must be zeroed by the
// host before launch.
#include <math_constants.h>

#define D 64

// Delta_i = sum_x dO[i,x] * O[i,x]: the "row correction" term, one per query
// row. One thread per row.
extern "C" __global__ void attn_bwd_preprocess(const float *dO, const float *O,
                                               float *Delta, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    float d = 0.0f;
#pragma unroll
    for (int x = 0; x < D; ++x) {
        d += dO[i * D + x] * O[i * D + x];
    }
    Delta[i] = d;
}

extern "C" __global__ void attn_bwd(const float *Q, const float *K, const float *V,
                                    const float *dO, const float *L, const float *Delta,
                                    float *dQ, float *dK, float *dV, int N, int causal) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    const float scale = rsqrtf((float)D);

    float q[D], doi[D], dq[D];
#pragma unroll
    for (int x = 0; x < D; ++x) {
        q[x] = Q[i * D + x];
        doi[x] = dO[i * D + x];
        dq[x] = 0.0f;
    }
    const float Li = L[i];
    const float Di = Delta[i];

    // causal: query i attends only keys j <= i (and always at least j = i).
    const int jmax = causal ? i : (N - 1);
    for (int j = 0; j <= jmax; ++j) {
        const float *kj = K + j * D;
        const float *vj = V + j * D;
        float s = 0.0f, dp = 0.0f;
#pragma unroll
        for (int x = 0; x < D; ++x) {
            s += q[x] * kj[x];
            dp += doi[x] * vj[x];
        }
        float p = __expf(s * scale - Li);
        float ds = p * (dp - Di) * scale;
#pragma unroll
        for (int x = 0; x < D; ++x) {
            atomicAdd(&dV[j * D + x], p * doi[x]);
            atomicAdd(&dK[j * D + x], ds * q[x]);
            dq[x] += ds * kj[x];
        }
    }
#pragma unroll
    for (int x = 0; x < D; ++x) {
        dQ[i * D + x] = dq[x];
    }
}
