// Flash Attention (backward, fp32). Closes the training half of stage 2.
//
// Given Q, K, V, the forward output O, the upstream gradient dO and the per-row
// log-sum-exp L from the forward, compute dQ, dK, dV. The N x N probability
// matrix is never stored: each score s_ij = scale * q_i . k_j is recomputed and
// p_ij = exp(s_ij - L_i) follows from the single saved scalar L_i.
//
// Softmax-attention VJP (Dao et al. 2022):
//   Delta_i = sum_x dO[i,x] * O[i,x]           (= sum_j p_ij * dP_ij)
//   dP_ij   = dO_i . v_j
//   dS_ij   = p_ij * (dP_ij - Delta_i) * scale (grad wrt q_i . k_j)
//   dV_j   += p_ij * dO_i
//   dK_j   += dS_ij * q_i
//   dQ_i   += dS_ij * k_j
//
// Two layouts are provided:
//   * attn_bwd_naive  — one thread per query row, K/V read from global on every
//     key, dK/dV scattered with atomicAdd. Simple, slow; kept as the baseline.
//   * attn_bwd_dq / attn_bwd_dkv — the FlashAttention-2 structure: dQ is owned
//     by query-row threads (K/V tiled in shared memory), dK/dV by key-row
//     threads (Q/dO tiled in shared memory). Each output element is written by
//     exactly one thread, so there are NO atomics, and the opposite operand is
//     reused across the whole block instead of re-read from global per element.
#include <math_constants.h>

#define D 64
#define BR 64 // rows owned per block = threads per block (queries for dq, keys for dkv)
#define BC 32 // tile width over the streamed axis, staged in shared memory

// Delta_i = sum_x dO[i,x] * O[i,x]: the "row correction" term, one per query
// row. One thread per row. Shared by both backward layouts.
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

// ---- naive baseline: one thread per query row, dK/dV via atomicAdd ----
// dQ/dK/dV must be zeroed by the host (dK/dV accumulate across query rows).
extern "C" __global__ void attn_bwd_naive(const float *Q, const float *K, const float *V,
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

// ---- dQ: one thread per query row, K/V tiled in shared memory, no atomics ----
// Mirrors the forward exactly (same tiling, same causal break); the only
// difference is the per-key body computes dQ_i += dS_ij * k_j instead of the
// output accumulation. dQ_i lives in registers and is written once.
extern "C" __global__ void attn_bwd_dq(const float *Q, const float *K, const float *V,
                                       const float *dO, const float *L, const float *Delta,
                                       float *dQ, int N, int causal) {
    __shared__ float Ks[BC][D]; // 8 KB
    __shared__ float Vs[BC][D]; // 8 KB
    const int row = blockIdx.x * BR + (int)threadIdx.x;
    const float scale = rsqrtf((float)D);

    float q[D], doi[D], dq[D];
    if (row < N) {
#pragma unroll
        for (int x = 0; x < D; ++x) {
            q[x] = Q[row * D + x];
            doi[x] = dO[row * D + x];
            dq[x] = 0.0f;
        }
    }
    const float Li = (row < N) ? L[row] : 0.0f;
    const float Di = (row < N) ? Delta[row] : 0.0f;
    const int block_last_row = blockIdx.x * BR + BR - 1;

    for (int j0 = 0; j0 < N; j0 += BC) {
        if (causal && j0 > block_last_row) {
            break;
        }
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
                float s = 0.0f, dp = 0.0f;
#pragma unroll
                for (int x = 0; x < D; ++x) {
                    s += q[x] * Ks[j][x];
                    dp += doi[x] * Vs[j][x];
                }
                float p = __expf(s * scale - Li);
                float ds = p * (dp - Di) * scale;
#pragma unroll
                for (int x = 0; x < D; ++x) {
                    dq[x] += ds * Ks[j][x];
                }
            }
        }
        __syncthreads();
    }

    if (row < N) {
#pragma unroll
        for (int x = 0; x < D; ++x) {
            dQ[row * D + x] = dq[x];
        }
    }
}

// ---- dK, dV: one thread per key row, Q/dO tiled in shared memory, no atomics --
// The transpose of the dQ pass. Each thread owns key j and accumulates dK_j,
// dV_j in registers; the thread's own V row is parked in shared memory (rather
// than registers) so the register footprint stays at ~192 floats like dQ
// instead of spilling. Q/dO tiles are reused across all BR keys of the block.
extern "C" __global__ void attn_bwd_dkv(const float *Q, const float *K, const float *V,
                                        const float *dO, const float *L, const float *Delta,
                                        float *dK, float *dV, int N, int causal) {
    __shared__ float Qs[BC][D];  // 8 KB
    __shared__ float dOs[BC][D]; // 8 KB
    // this block's key V rows, accessed only by the owning thread. The +1 pad
    // makes the per-thread row stride 65 (coprime with 32), so the 64 threads
    // read 64 distinct banks for a given x instead of colliding 64-way.
    __shared__ float Vrow[BR][D + 1];
    __shared__ float Ls[BC];
    __shared__ float Ds[BC];
    const int tid = (int)threadIdx.x;
    const int row = blockIdx.x * BR + tid; // this thread owns key j = row
    const float scale = rsqrtf((float)D);

    float k[D], dk[D], dv[D];
    if (row < N) {
#pragma unroll
        for (int x = 0; x < D; ++x) {
            k[x] = K[row * D + x];
            Vrow[tid][x] = V[row * D + x];
            dk[x] = 0.0f;
            dv[x] = 0.0f;
        }
    }
    // causal: key j attends only queries i >= j, so query tiles entirely below
    // the block's smallest key contribute nothing to any thread in the block.
    const int block_min_key = blockIdx.x * BR;

    for (int i0 = 0; i0 < N; i0 += BC) {
        if (causal && i0 + BC - 1 < block_min_key) {
            continue;
        }
        for (int t = tid; t < BC * D; t += BR) {
            int i = t / D, x = t % D;
            if (i0 + i < N) {
                Qs[i][x] = Q[(i0 + i) * D + x];
                dOs[i][x] = dO[(i0 + i) * D + x];
            }
        }
        for (int t = tid; t < BC; t += BR) {
            if (i0 + t < N) {
                Ls[t] = L[i0 + t];
                Ds[t] = Delta[i0 + t];
            }
        }
        __syncthreads();

        if (row < N) {
            for (int i = 0; i < BC && i0 + i < N; ++i) {
                if (causal && i0 + i < row) {
                    continue; // query before this key contributes nothing
                }
                float s = 0.0f, dp = 0.0f;
#pragma unroll
                for (int x = 0; x < D; ++x) {
                    s += k[x] * Qs[i][x];
                    dp += Vrow[tid][x] * dOs[i][x];
                }
                float p = __expf(s * scale - Ls[i]);
                float ds = p * (dp - Ds[i]) * scale;
#pragma unroll
                for (int x = 0; x < D; ++x) {
                    dv[x] += p * dOs[i][x];
                    dk[x] += ds * Qs[i][x];
                }
            }
        }
        __syncthreads();
    }

    if (row < N) {
#pragma unroll
        for (int x = 0; x < D; ++x) {
            dK[row * D + x] = dk[x];
            dV[row * D + x] = dv[x];
        }
    }
}
