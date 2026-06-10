//! Stage 3: GPT-2 124M / Qwen2.5-0.5B / TinyLlama-1.1B inference engine in
//! plain CUDA.
//!
//! Every subcommand takes `--model gpt2|qwen|tinyllama` (default gpt2):
//!   cargo run -rp llm-engine -- export                 # download + convert weights
//!   cargo run -rp llm-engine -- generate "prompt" [-n 64] [--fp16|--int8|--int4] [--kv8] [--spec]
//!   cargo run -rp llm-engine -- verify                 # GPU logits vs CPU reference
//!   cargo run -rp llm-engine -- bench [-n 64] [--graphs] [--kv8]
//!   cargo run -rp llm-engine -- prefill-bench [-n 512] [--kv8]
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

/// Model directory, weight file and arch selected by `--model gpt2|qwen`.
struct ModelChoice {
    dir: PathBuf,
    bin: PathBuf,
    arch: model::Arch,
}

fn model_choice(args: &[String]) -> ModelChoice {
    match opt_value(args, "--model").unwrap_or("gpt2") {
        "gpt2" => {
            let dir = models_dir();
            ModelChoice {
                bin: dir.join("gpt2.bin"),
                dir,
                arch: model::Arch::Gpt2,
            }
        }
        "qwen" => {
            let dir = models_dir().join("qwen");
            ModelChoice {
                bin: dir.join("qwen2.5-0.5b.bin"),
                dir,
                arch: model::Arch::Qwen2,
            }
        }
        "tinyllama" => {
            let dir = models_dir().join("tinyllama");
            ModelChoice {
                bin: dir.join("tinyllama-1.1b.bin"),
                dir,
                arch: model::Arch::Llama,
            }
        }
        m => panic!("unknown --model {m} (expected gpt2, qwen or tinyllama)"),
    }
}

fn load_model(choice: &ModelChoice) -> model::Model {
    assert!(
        choice.bin.exists(),
        "{} not found — run `export` first",
        choice.bin.display()
    );
    model::Model::load(&choice.bin).unwrap()
}

/// Weight-storage modes that fit in 2 GB VRAM for this model.
/// Qwen2.5-0.5B in fp32 is ~1.9 GB of weights — more than the whole card;
/// TinyLlama-1.1B even in fp16 is 2.2 GB, so only int4 (~620 MB) and
/// int8 (~1.1 GB, just barely) run at all.
fn modes_for(arch: model::Arch) -> &'static [gpu::WeightMode] {
    match arch {
        model::Arch::Gpt2 => &[
            gpu::WeightMode::Fp32,
            gpu::WeightMode::Fp16,
            gpu::WeightMode::Int8,
            gpu::WeightMode::Int4,
        ],
        model::Arch::Qwen2 => &[
            gpu::WeightMode::Fp16,
            gpu::WeightMode::Int8,
            gpu::WeightMode::Int4,
        ],
        model::Arch::Llama => &[gpu::WeightMode::Int4, gpu::WeightMode::Int8],
    }
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

fn opt_usize(args: &[String], name: &str, default: usize) -> usize {
    opt_value(args, name)
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} expects a number"))
        })
        .unwrap_or(default)
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
            let choice = model_choice(&args);
            let model = match choice.arch {
                model::Arch::Gpt2 => {
                    export::download(&choice.dir);
                    println!("converting safetensors -> {} ...", choice.bin.display());
                    export::convert(&choice.dir)
                }
                model::Arch::Qwen2 => {
                    export::download_qwen(&choice.dir);
                    println!("converting safetensors -> {} ...", choice.bin.display());
                    export::convert_qwen(&choice.dir)
                }
                model::Arch::Llama => {
                    export::download_tinyllama(&choice.dir);
                    println!("converting safetensors -> {} ...", choice.bin.display());
                    export::convert_tinyllama(&choice.dir)
                }
            };
            model.save(&choice.bin).unwrap();
            println!("wrote {}", choice.bin.display());
        }
        Some("encode") => {
            // debug helper: print token ids for a string (compare against HF)
            let text = args.get(1).expect("usage: encode \"text\"");
            let choice = model_choice(&args);
            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let ids = tok.encode(text);
            println!("{ids:?}");
            println!("decoded: {:?}", tok.decode(&ids));
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
            let spec = flag(&args, "--spec");
            let spec_k = opt_usize(&args, "--spec-k", 8);
            let choice = model_choice(&args);

            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let model = load_model(&choice);
            let ctx = CudaContext::new(0).unwrap();
            let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);

            let ids = tok.encode(prompt);
            print!("{prompt}");
            use std::io::Write;
            let t0 = Instant::now();
            if spec {
                engine.generate_speculative(&ids, n_new, spec_k, |id| {
                    print!("{}", tok.decode(&[id]));
                    std::io::stdout().flush().unwrap();
                });
            } else {
                engine.generate(&ids, n_new, |id| {
                    print!("{}", tok.decode(&[id]));
                    std::io::stdout().flush().unwrap();
                });
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "\n\n[{} prompt + {} new tokens in {:.2}s = {:.1} tok/s, {}, {}]",
                ids.len(),
                n_new,
                dt,
                (ids.len() + n_new) as f64 / dt,
                if kv8 {
                    format!("{mode} + kv8")
                } else {
                    mode.to_string()
                },
                if spec { "prompt-lookup spec" } else { "greedy" }
            );
        }
        Some("verify") => {
            let choice = model_choice(&args);
            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let model = load_model(&choice);
            let prompt = "Alan Turing was a British mathematician";
            let ids = tok.encode(prompt);
            println!("prompt: {prompt:?} -> {ids:?}");

            println!("CPU reference forward ({} tokens)...", ids.len());
            let want = cpu::forward(&model, &ids);

            let ctx = CudaContext::new(0).unwrap();
            let mut combos: Vec<(gpu::WeightMode, bool)> =
                modes_for(choice.arch).iter().map(|&m| (m, false)).collect();
            combos.push((modes_for(choice.arch)[0], true));
            combos.push((gpu::WeightMode::Int8, true));
            for (mode, kv8) in combos {
                let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                let mut got = Vec::new();
                for (pos, &t) in ids.iter().enumerate() {
                    got = engine.forward(t, pos);
                }
                let batch = engine.prefill(&ids, 0);
                let err = common::allclose_err(&got, &want, 1e-2, 5e-2);
                let batch_err = common::allclose_err(&batch, &got, 1e-2, 5e-2);
                let (cw, gw) = (gpu::argmax(&want), gpu::argmax(&got));
                let bw = gpu::argmax(&batch);
                let kv = if kv8 { "/kv8" } else { "" };
                println!(
                    "GPU {mode}{kv}: allclose_err = {err:.2e}, batch_err = {batch_err:.2e}, argmax cpu={cw} ({:?}) gpu={gw} ({:?}) batch={bw} ({:?})",
                    tok.decode(&[cw]),
                    tok.decode(&[gw]),
                    tok.decode(&[bw]),
                );
                if mode == gpu::WeightMode::Fp32 && !kv8 {
                    assert!(err < 1.0, "fp32 logits mismatch: {err}");
                    assert!(batch_err < 1.0, "fp32 batch prefill mismatch: {batch_err}");
                }
                // int4 may legitimately change the argmax (on GPT-2 the
                // quantization damage is real — see the perplexity table),
                // so the CPU comparison is informational there; internal
                // consistency between decode and batch prefill always holds
                if mode == gpu::WeightMode::Int4 {
                    if cw != gw {
                        println!("  note: int4 argmax differs from fp32 CPU (quantization)");
                    }
                } else {
                    assert_eq!(cw, gw, "{mode}{kv} argmax mismatch");
                }
                assert_eq!(gw, bw, "{mode}{kv} batch prefill argmax mismatch");
                println!("  OK");
            }

            // a prompt with repeated n-grams guarantees prompt_lookup finds
            // drafts, so the verify/accept path is actually exercised (the
            // fallback path still runs on the non-repeating stretches)
            let spec_ids = tok.encode(
                "The quick brown fox jumps over the lazy dog. \
                 The quick brown fox jumps over the lazy dog. The quick brown fox",
            );
            for kv8 in [false, true] {
                let mode = modes_for(choice.arch)[0];
                let n_steps = 32;
                let mut greedy_engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                let greedy = greedy_engine.generate(&spec_ids, n_steps, |_| {});
                drop(greedy_engine);
                let mut spec_engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                let spec = spec_engine.generate_speculative(&spec_ids, n_steps, 8, |_| {});
                assert_eq!(
                    spec, greedy,
                    "prompt-lookup speculative decode diverged from greedy (kv8={kv8})"
                );
                println!("prompt-lookup speculative kv8={kv8}: {n_steps} tokens match greedy  OK");
            }

            // graph decode must produce the same greedy continuation as the
            // host loop; any divergence propagates, so comparing the token
            // after n steps checks the whole path
            for kv8 in [false, true] {
                let n_steps = 16;
                let graph_mode = modes_for(choice.arch)[0];
                let mut engine = gpu::Engine::new(&ctx, &model, graph_mode, kv8);
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
                // both engines don't fit in VRAM at once for the larger model
                drop(engine);

                let mut engine = gpu::Engine::new(&ctx, &model, graph_mode, kv8);
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
            let choice = model_choice(&args);
            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let model = load_model(&choice);
            let n_new = opt_n(&args, 64);
            let graphs = flag(&args, "--graphs");
            let kv8 = flag(&args, "--kv8");
            let spec = flag(&args, "--spec");
            let spec_k = opt_usize(&args, "--spec-k", 8);
            let ids = tok.encode("The history of computing began");
            let ctx = CudaContext::new(0).unwrap();

            println!("| mode | weights | kv | graph | spec | tokens/sec |");
            println!("|------|---------|----|-------|------|------------|");
            for &mode in modes_for(choice.arch) {
                let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                // warmup + prefill
                let mut logits = engine.prefill(&ids, 0);
                if graphs {
                    engine.prepare_decode_graph();
                }
                let t0 = Instant::now();
                if spec {
                    // prefill happened above, outside the timed region — same
                    // as the non-speculative branches
                    engine.speculative_loop(&ids, gpu::argmax(&logits), n_new, spec_k, |_| {});
                } else if graphs {
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
                    "| {mode} | ~{:.0} MB | {} | {} | {} | {:.1} |",
                    gpu::weight_mb(&model.config, mode),
                    if kv8 { "int8" } else { "fp32" },
                    if graphs { "yes" } else { "no" },
                    if spec { "yes" } else { "no" },
                    n_new as f64 / dt
                );
            }
        }
        Some("prefill-bench") => {
            let choice = model_choice(&args);
            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let model = load_model(&choice);
            let max_prompt = opt_n(&args, 512).min(model.config.n_ctx);
            let kv8 = flag(&args, "--kv8");
            let seed = "The history of computing began with machines for arithmetic. ";
            let mut text = String::new();
            let mut ids = Vec::new();
            while ids.len() < max_prompt {
                text.push_str(seed);
                ids = tok.encode(&text);
            }
            ids.truncate(max_prompt);
            let ctx = CudaContext::new(0).unwrap();

            println!("prompt tokens: {}", ids.len());
            println!("| mode | kv | token-loop TTFT | batch TTFT | speedup |");
            println!("|------|----|-----------------|------------|---------|");
            for &mode in modes_for(choice.arch) {
                let mut loop_engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                let t0 = Instant::now();
                for (pos, &t) in ids.iter().enumerate() {
                    loop_engine.forward(t, pos);
                }
                let loop_dt = t0.elapsed().as_secs_f64();
                drop(loop_engine);

                let mut batch_engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                let t0 = Instant::now();
                batch_engine.prefill(&ids, 0);
                let batch_dt = t0.elapsed().as_secs_f64();
                println!(
                    "| {mode} | {} | {:.3}s | {:.3}s | {:.2}x |",
                    if kv8 { "int8" } else { "fp32" },
                    loop_dt,
                    batch_dt,
                    loop_dt / batch_dt
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
            let choice = model_choice(&args);
            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let model = load_model(&choice);
            let mut ids = tok.encode(&text);
            ids.truncate(max_tokens.min(ids.len()));
            assert!(ids.len() > 1, "need at least two tokens for perplexity");
            let ctx = CudaContext::new(0).unwrap();

            println!("dataset: {} ({} tokens)", data_path.display(), ids.len());
            println!("| mode | weights | kv | tokens | perplexity |");
            println!("|------|---------|----|--------|------------|");
            for kv8 in [false, true] {
                for &mode in modes_for(choice.arch) {
                    let mut engine = gpu::Engine::new(&ctx, &model, mode, kv8);
                    let (ppl, n) = perplexity(&mut engine, &ids);
                    println!(
                        "| {mode} | ~{:.0} MB | {} | {n} | {ppl:.3} |",
                        gpu::weight_mb(&model.config, mode),
                        if kv8 { "int8" } else { "fp32" },
                    );
                }
            }
        }
        _ => {
            eprintln!(
                "usage: llm-engine <export|ppl-data|generate|verify|bench|prefill-bench|ppl> [args]"
            );
            std::process::exit(1);
        }
    }
}
