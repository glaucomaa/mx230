//! Stage 3: GPT-2 124M inference engine in plain CUDA.
//!
//!   cargo run -rp llm-engine -- export                 # download + convert weights
//!   cargo run -rp llm-engine -- generate "prompt" [-n 64] [--fp16|--int8] [--kv8]
//!   cargo run -rp llm-engine -- verify                 # GPU logits vs CPU reference
//!   cargo run -rp llm-engine -- bench [-n 64] [--graphs] [--kv8]
//!   cargo run -rp llm-engine -- ppl --data path [-n tokens]

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
    assert!(
        path.exists(),
        "{} not found — run `export` first",
        path.display()
    );
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

fn opt_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn logprob(logits: &[f32], target: u32) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp: f64 = logits.iter().map(|&x| ((x as f64) - m).exp()).sum();
    logits[target as usize] as f64 - m - sum_exp.ln()
}

fn perplexity(engine: &mut gpu::Engine, tokens: &[u32]) -> (f64, usize) {
    let ctx = engine.config.n_ctx;
    let mut nll = 0.0f64;
    let mut count = 0usize;
    let mut start = 0;
    while start + 1 < tokens.len() {
        let end = (start + ctx).min(tokens.len());
        let chunk = &tokens[start..end];
        let mut logits = Vec::new();
        for (pos, &tok) in chunk.iter().enumerate() {
            if pos > 0 {
                nll -= logprob(&logits, tok);
                count += 1;
            }
            logits = engine.forward(tok, pos);
        }
        start = end - 1;
    }
    ((nll / count as f64).exp(), count)
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
        Some("ppl-data") => {
            export::download_wikitext2(&models_dir());
        }
        Some("generate") => {
            let prompt = args
                .get(1)
                .filter(|p| !p.starts_with('-'))
                .expect("usage: generate \"prompt\"");
            let n_new = opt_n(&args, 64);
            let mode = gpu::WeightMode::parse(&args);
            let kv8 = flag(&args, "--kv8");

            let tok = tokenizer::Tokenizer::load(&models_dir());
            let model = load_model();
            let ctx = CudaContext::new(0).unwrap();
            let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);

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
                if kv8 {
                    format!("{mode} + kv8")
                } else {
                    mode.to_string()
                }
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
            for (mode, kv8) in [
                (gpu::WeightMode::Fp32, false),
                (gpu::WeightMode::Fp16, false),
                (gpu::WeightMode::Int8, false),
                (gpu::WeightMode::Fp32, true),
                (gpu::WeightMode::Int8, true),
            ] {
                let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                let mut got = Vec::new();
                for (pos, &t) in ids.iter().enumerate() {
                    got = engine.forward(t, pos);
                }
                let err = common::allclose_err(&got, &want, 1e-2, 5e-2);
                let (cw, gw) = (gpu::argmax(&want), gpu::argmax(&got));
                let kv = if kv8 { "/kv8" } else { "" };
                println!(
                    "GPU {mode}{kv}: allclose_err = {err:.2e}, argmax cpu={cw} ({:?}) gpu={gw} ({:?})",
                    tok.decode(&[cw]),
                    tok.decode(&[gw]),
                );
                if mode == gpu::WeightMode::Fp32 && !kv8 {
                    assert!(err < 1.0, "fp32 logits mismatch: {err}");
                }
                assert_eq!(cw, gw, "{mode}{kv} argmax mismatch");
                println!("  OK");
            }

            // graph decode must produce the same greedy continuation as the
            // host loop; any divergence propagates, so comparing the token
            // after n steps checks the whole path
            for kv8 in [false, true] {
                let n_steps = 16;
                let mut engine = gpu::Engine::new(&ctx, &model, gpu::WeightMode::Fp32, kv8);
                let mut logits = Vec::new();
                for (pos, &t) in ids.iter().enumerate() {
                    logits = engine.forward(t, pos);
                }
                let first = gpu::argmax(&logits);
                let mut pos = ids.len();
                for _ in 0..n_steps {
                    let next = gpu::argmax(&logits);
                    logits = engine.forward(next, pos);
                    pos += 1;
                }
                let host_tok = gpu::argmax(&logits);

                let mut engine = gpu::Engine::new(&ctx, &model, gpu::WeightMode::Fp32, kv8);
                for (pos, &t) in ids.iter().enumerate() {
                    engine.forward(t, pos);
                }
                let graph_tok = engine.graph_decode(first, ids.len(), n_steps);
                assert_eq!(
                    graph_tok, host_tok,
                    "graph decode (kv8={kv8}) diverged from host decode after {n_steps} steps"
                );
                println!(
                    "graph decode kv8={kv8}: token after {n_steps} steps matches host loop ({host_tok})  OK"
                );
            }
        }
        Some("bench") => {
            let tok = tokenizer::Tokenizer::load(&models_dir());
            let model = load_model();
            let n_new = opt_n(&args, 64);
            let graphs = flag(&args, "--graphs");
            let kv8 = flag(&args, "--kv8");
            let ids = tok.encode("The history of computing began");
            let ctx = CudaContext::new(0).unwrap();

            println!("| mode | weights | kv | graph | tokens/sec |");
            println!("|------|---------|----|-------|------------|");
            for mode in [
                gpu::WeightMode::Fp32,
                gpu::WeightMode::Fp16,
                gpu::WeightMode::Int8,
            ] {
                let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                // warmup + prefill
                let mut logits = Vec::new();
                for (pos, &t) in ids.iter().enumerate() {
                    logits = engine.forward(t, pos);
                }
                if graphs {
                    engine.prepare_decode_graph();
                }
                let t0 = Instant::now();
                if graphs {
                    let first = gpu::argmax(&logits);
                    engine.graph_decode(first, ids.len(), n_new);
                } else {
                    let mut pos = ids.len();
                    for _ in 0..n_new {
                        let next = gpu::argmax(&logits);
                        logits = engine.forward(next, pos);
                        pos += 1;
                    }
                }
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "| {mode} | ~{:.0} MB | {} | {} | {:.1} |",
                    mode.weight_mb(),
                    if kv8 { "int8" } else { "fp32" },
                    if graphs { "yes" } else { "no" },
                    n_new as f64 / dt
                );
            }
        }
        Some("ppl") => {
            let default_data = models_dir().join("wikitext-2-raw/wiki.test.raw");
            let data_path = opt_value(&args, "--data")
                .map(PathBuf::from)
                .unwrap_or(default_data);
            let max_tokens = opt_n(&args, 2048);
            assert!(
                data_path.exists(),
                "{} not found; run `cargo run -rp llm-engine -- ppl-data` or pass --data",
                data_path.display()
            );
            let text = std::fs::read_to_string(&data_path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", data_path.display()));
            let tok = tokenizer::Tokenizer::load(&models_dir());
            let model = load_model();
            let mut ids = tok.encode(&text);
            ids.truncate(max_tokens.min(ids.len()));
            assert!(ids.len() > 1, "need at least two tokens for perplexity");
            let ctx = CudaContext::new(0).unwrap();

            println!("dataset: {} ({} tokens)", data_path.display(), ids.len());
            println!("| mode | weights | kv | tokens | perplexity |");
            println!("|------|---------|----|--------|------------|");
            for kv8 in [false, true] {
                for mode in [
                    gpu::WeightMode::Fp32,
                    gpu::WeightMode::Fp16,
                    gpu::WeightMode::Int8,
                ] {
                    let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                    let (ppl, n) = perplexity(&mut engine, &ids);
                    println!(
                        "| {mode} | ~{:.0} MB | {} | {n} | {ppl:.3} |",
                        mode.weight_mb(),
                        if kv8 { "int8" } else { "fp32" },
                    );
                }
            }
        }
        _ => {
            eprintln!("usage: llm-engine <export|ppl-data|generate|verify|bench|ppl> [args]");
            std::process::exit(1);
        }
    }
}
