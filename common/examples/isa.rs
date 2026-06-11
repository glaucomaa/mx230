//! ISA throughput audit for the MX230 (GP108, sm_61): what does one
//! instruction stream actually deliver — fp32 FMA, dp4a (4x int8 MAC),
//! half2 FMA — plus streaming-read bandwidth as the memory roof.
//! Run with: `cargo run -rp common --example isa`

use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

const ISA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/isa.ptx"));

// must match isa.cu
const ITERS: usize = 4096;
const CHAINS: usize = 8;
const BLOCKS: usize = 32;
const THREADS: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let sm_count = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as f64;
    let clock_ghz = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)? as f64 / 1e6;
    let cores = sm_count * 128.0; // Pascal: 128 fp32 cores per SM
    let issue_rate = cores * clock_ghz; // G instructions/s if 1/clock/core
    println!(
        "device: {} | {} SMs x 128 cores @ {clock_ghz:.2} GHz",
        ctx.name()?,
        sm_count as u32
    );
    println!("fp32 FMA peak: {:.0} GFLOPS (2 FLOP/instr)\n", issue_rate * 2.0);

    let module = common::load_ptx(&ctx, "isa", ISA_PTX)?;
    let stream = ctx.default_stream();
    let cfg = LaunchConfig {
        grid_dim: (BLOCKS as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let instrs = (BLOCKS * THREADS * ITERS * CHAINS) as f64;

    let mut out_f = stream.alloc_zeros::<f32>(BLOCKS * THREADS)?;
    let mut out_i = stream.alloc_zeros::<i32>(BLOCKS * THREADS)?;

    println!("| probe | instr | per instr | measured | vs fp32 peak |");
    println!("|-------|-------|-----------|----------|--------------|");

    // (kernel, ops-per-instruction, unit, int output buffer?)
    for (name, ops, unit, int_out) in [
        ("fma_f32", 2.0, "GFLOPS", false),
        ("dp4a_s32", 8.0, "GOPS (int8 MAC+add)", true),
        ("fma_h2", 4.0, "GFLOPS (fp16)", false),
    ] {
        let f = module.load_function(name)?;
        let (ai, bi) = (0x01010101i32, 0x02020202i32);
        let (af, bf) = (1.0001f32, 0.0001f32);
        let ms = common::time_median_ms(&stream, 3, 20, || {
            let mut lb = stream.launch_builder(&f);
            if int_out {
                lb.arg(&mut out_i).arg(&ai).arg(&bi);
            } else {
                lb.arg(&mut out_f).arg(&af).arg(&bf);
            }
            unsafe { lb.launch(cfg) }.map(|_| ())
        })?;
        let gops = instrs * ops / (ms as f64 * 1e6);
        println!(
            "| {name} | {} | {ops} ops | {gops:.0} {unit} | {:.2}x |",
            instrs as u64,
            gops / (issue_rate * 2.0),
        );
    }

    // memory roof: stream 256 MB of float4 reads
    let n = 64 << 20; // floats
    let x = stream.alloc_zeros::<f32>(n)?;
    let f = module.load_function("stream_f4")?;
    let n4 = (n / 4) as i32;
    let mem_cfg = LaunchConfig {
        grid_dim: (256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let ms = common::time_median_ms(&stream, 3, 20, || {
        let mut lb = stream.launch_builder(&f);
        lb.arg(&mut out_f).arg(&x).arg(&n4);
        unsafe { lb.launch(mem_cfg) }.map(|_| ())
    })?;
    let gbs = (n * 4) as f64 / (ms as f64 * 1e6);
    println!("| stream_f4 | — | — | {gbs:.1} GB/s | memory roof |");
    Ok(())
}
