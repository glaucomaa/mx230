//! Stage 0 smoke test: can cudarc see the GPU, and does the
//! .cu → nvcc → PTX → kernel-launch chain work end to end?
//! Run with: `cargo run -p common --example smoke`

use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

const VECTOR_ADD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vector_add.ptx"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let cc_major = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let cc_minor = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let sm_count = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    println!("device      : {}", ctx.name()?);
    println!("compute cap : sm_{cc_major}{cc_minor}");
    println!("SMs         : {sm_count}");

    let module = common::load_ptx(&ctx, "vector_add", VECTOR_ADD_PTX)?;
    let f = module.load_function("vector_add")?;
    let stream = ctx.default_stream();

    let n = 1 << 20;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let a = stream.clone_htod(&a_host)?;
    let b = stream.clone_htod(&b_host)?;
    let mut c = stream.alloc_zeros::<f32>(n)?;

    let mut launch = stream.launch_builder(&f);
    let n_i32 = n as i32;
    launch.arg(&a).arg(&b).arg(&mut c).arg(&n_i32);
    unsafe { launch.launch(LaunchConfig::for_num_elems(n as u32)) }?;

    let c_host = stream.clone_dtoh(&c)?;
    assert!(c_host.iter().all(|&x| x == n as f32), "vector_add produced wrong results");
    println!("vector_add  : OK ({n} elements, all = {n})");
    Ok(())
}
