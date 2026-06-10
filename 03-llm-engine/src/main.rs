//! Stage 3: GPT-2 124M inference engine in plain CUDA.
//!
//!   cargo run -rp llm-engine -- export                 # download + convert weights
//!   cargo run -rp llm-engine -- generate "prompt" [-n 64] [--int8]
//!   cargo run -rp llm-engine -- verify                 # GPU logits vs CPU reference
//!   cargo run -rp llm-engine -- bench [-n 64]          # tokens/sec, fp32 vs int8

mod cpu;
mod export;
mod gpu;
mod model;
mod tokenizer;

use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::CudaContext;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn load_model() -> model::Model {
    let path = models_dir().join("gpt2.bin");
    assert!(path.exists(), "{} not found — run `export` first", path.display());
    model::Model::load(&path).unwrap()
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn opt_n(args: &[String], default: usize) -> usize {
    args.iter()
        .position(|a| a == "-n")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().expect("-n expects a number"))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("export") => {
            let dir = models_dir();
            export::download(&dir);
            println!("converting safetensors -> gpt2.bin ...");
            let model = export::convert(&dir);
            model.save(&dir.join("gpt2.bin")).unwrap();
            println!("wrote {}", dir.join("gpt2.bin").display());
        }
        Some("generate") => {
            let prompt = args.get(1).filter(|p| !p.starts_with('-')).expect("usage: generate \"prompt\"");
            let n_new = opt_n(&args, 64);
            let int8 = flag(&args, "--int8");

            let tok = tokenizer::Tokenizer::load(&models_dir());
            let model = load_model();
            let ctx = CudaContext::new(0).unwrap();
            let mut engine = gpu::Engine::new(&ctx, &model, int8);

            let ids = tok.encode(prompt);
            print!("{prompt}");
            use std::io::Write;
            let t0 = Instant::now();
            engine.generate(&ids, n_new, |id| {
                print!("{}", tok.decode(&[id]));
                std::io::stdout().flush().unwrap();
            });
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "\n\n[{} prompt + {} new tokens in {:.2}s = {:.1} tok/s, {}]",
                ids.len(),
                n_new,
                dt,
                (ids.len() + n_new) as f64 / dt,
                if int8 { "int8" } else { "fp32" }
            );
        }
        Some("verify") => {
            let tok = tokenizer::Tokenizer::load(&models_dir());
            let model = load_model();
            let prompt = "Alan Turing was a British mathematician";
            let ids = tok.encode(prompt);
            println!("prompt: {prompt:?} -> {ids:?}");

            println!("CPU reference forward ({} tokens)...", ids.len());
            let want = cpu::forward(&model, &ids);

            let ctx = CudaContext::new(0).unwrap();
            for int8 in [false, true] {
                let mut engine = gpu::Engine::new(&ctx, &model, int8);
                let mut got = Vec::new();
                for (pos, &t) in ids.iter().enumerate() {
                    got = engine.forward(t, pos);
                }
                let name = if int8 { "int8" } else { "fp32" };
                let err = common::allclose_err(&got, &want, 1e-2, 5e-2);
                let (cw, gw) = (gpu::argmax(&want), gpu::argmax(&got));
                println!(
                    "GPU {name}: allclose_err = {err:.2e}, argmax cpu={cw} ({:?}) gpu={gw} ({:?})",
                    tok.decode(&[cw]),
                    tok.decode(&[gw]),
                );
                if int8 {
                    // int8 only has to agree on ranking, not bit-exact logits
                    assert_eq!(cw, gw, "int8 argmax mismatch");
                } else {
                    assert!(err < 1.0, "fp32 logits mismatch: {err}");
                    assert_eq!(cw, gw, "fp32 argmax mismatch");
                }
                println!("  OK");
            }
        }
        Some("bench") => {
            let tok = tokenizer::Tokenizer::load(&models_dir());
            let model = load_model();
            let n_new = opt_n(&args, 64);
            let ids = tok.encode("The history of computing began");
            let ctx = CudaContext::new(0).unwrap();

            println!("| mode | weights | tokens/sec |");
            println!("|------|---------|------------|");
            for int8 in [false, true] {
                let mut engine = gpu::Engine::new(&ctx, &model, int8);
                // warmup + prefill
                let mut logits = Vec::new();
                for (pos, &t) in ids.iter().enumerate() {
                    logits = engine.forward(t, pos);
                }
                let mut pos = ids.len();
                let t0 = Instant::now();
                for _ in 0..n_new {
                    let next = gpu::argmax(&logits);
                    logits = engine.forward(next, pos);
                    pos += 1;
                }
                let dt = t0.elapsed().as_secs_f64();
                let mb = if int8 { 124.0 } else { 498.0 };
                println!("| {} | ~{mb:.0} MB | {:.1} |", if int8 { "int8" } else { "fp32" }, n_new as f64 / dt);
            }
        }
        _ => {
            eprintln!("usage: llm-engine <export|generate|verify|bench> [args]");
            std::process::exit(1);
        }
    }
}
