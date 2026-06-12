fn main() {
    kernel_build::compile_kernels();
    // Second engine-kernel variant with 8-wide activation-scale groups:
    // the RoPE models tolerate them (no GPT-2-style activation outliers)
    // and the dp4a GEMMs pay half the scale-FMAs.
    kernel_build::compile_kernel_variant("kernels/llm.cu", "llm_ag8", &["-DAG=8"]);
}
