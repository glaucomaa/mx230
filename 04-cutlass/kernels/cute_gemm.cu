// Stage 4: a fused GEMM + bias + GELU written with CuTe (CUTLASS 3.x), on the
// SIMT path — sm_61 has no tensor cores, so the inner `gemm()` lowers to plain
// FMAs (UniversalFMA), not an MMA atom. Structure after the CuTe sgemm_1
// tutorial: 128x128x8 CTA tile, 256 threads, each thread owns an 8x8 output
// block. NT layout (the well-coalesced tutorial layout): A (M,K) m-major,
// B (N,K) n-major, C (M,N) m-major (all column-major in BLAS terms).
//
// The point of the stage is fusion: the GPT-2 ffn-up is `GELU(x·W_fc + b)`, and
// folding the bias + GELU into the GEMM epilogue avoids a full M·N round-trip to
// DRAM — the whole lesson of this lab restated in CuTe.
#include <cute/tensor.hpp>
using namespace cute;

// GPT-2's tanh-approximation GELU.
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Shared body. FUSE selects the epilogue: GELU(alpha*acc + bias) vs alpha*acc.
template <bool FUSE>
__device__ __forceinline__ void ffn_gemm(const float *A, const float *B, const float *bias,
                                         float *C, int M, int N, int K, float alpha) {
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto prob = make_shape(M, N, K);

    // NT (column-major) strides: A (M,K) m-major, B (N,K) n-major, C (M,N) m-major
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(prob), make_stride(Int<1>{}, M));
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(prob), make_stride(Int<1>{}, N));
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(prob), make_stride(Int<1>{}, M));

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (bM,bK,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (bN,bK,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (bM,bN)

    auto sA_layout = make_layout(make_shape(bM, bK));
    auto sB_layout = make_layout(make_shape(bN, bK));
    __shared__ float smemA[cosize_v<decltype(sA_layout)>];
    __shared__ float smemB[cosize_v<decltype(sB_layout)>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // 256 threads, 8x8 each

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    for (int k = 0; k < K_TILE_MAX; ++k) {
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);
        __syncthreads();
        gemm(tCsA, tCsB, tCrC); // SIMT FMA: tCrC += tCsA * tCsB
        __syncthreads();
    }

    if constexpr (FUSE) {
        // bias broadcast over M: (M,N) tensor with a 0 stride on the M mode
        Tensor mBias = make_tensor(make_gmem_ptr(bias), select<0, 1>(prob),
                                   make_stride(Int<0>{}, Int<1>{}));
        Tensor gBias = local_tile(mBias, cta_tiler, cta_coord, Step<_1, _1, X>{});
        Tensor tCgBias = local_partition(gBias, tC, threadIdx.x, Step<_1, _1>{});
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            tCgC(i) = gelu_tanh(alpha * tCrC(i) + tCgBias(i));
        }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            tCgC(i) = alpha * tCrC(i);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void cute_gemm(const float *A, const float *B,
                                                            float *C, int M, int N, int K,
                                                            float alpha) {
    ffn_gemm<false>(A, B, nullptr, C, M, N, K, alpha);
}

extern "C" __global__ __launch_bounds__(256) void cute_gemm_bias_gelu(const float *A,
                                                                      const float *B,
                                                                      const float *bias, float *C,
                                                                      int M, int N, int K,
                                                                      float alpha) {
    ffn_gemm<true>(A, B, bias, C, M, N, K, alpha);
}

// Unfused-baseline epilogue: GELU(C + bias) elementwise over column-major
// C (M x N), where element (m,n) lives at idx = m + n*M, so n = idx / M.
extern "C" __global__ void bias_gelu(float *C, const float *bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        C[idx] = gelu_tanh(C[idx] + bias[idx / M]);
    }
}
