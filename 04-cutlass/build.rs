fn main() {
    // CuTe (CUTLASS 3.x) is header-only; the .cu in kernels/ need its include
    // path and C++17. The flags are harmless for the plain gemm_06 baseline, so
    // both kernels compile through the same call. CUTLASS is vendored as the
    // third_party/cutlass submodule (pinned commit recorded in the superproject).
    kernel_build::compile_kernels_with_args(&[
        "-std=c++17",
        "-I",
        "../third_party/cutlass/include",
    ]);
}
