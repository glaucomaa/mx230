//! Stage 3: GPT-2 124M / Qwen2.5-0.5B / TinyLlama-1.1B inference engine in
//! plain CUDA.
//!
//! Every subcommand takes `--model gpt2|qwen|tinyllama` (default gpt2):
//!   cargo run -rp llm-engine -- export                 # download + convert weights
//!   cargo run -rp llm-engine -- generate "prompt" [-n 64] [--fp16|--int8|--int4] [--kv8] [--spec]
//!       [--temp 0.8 --top-k 40 --top-p 0.95 --seed 1]   # default greedy; --spec stays greedy
//!       [--smooth [--smooth-alpha 0.5] [--calib-tokens 512]]  # SmoothQuant fold
//!   cargo run -rp llm-engine -- verify                 # GPU logits vs CPU reference
//!   cargo run -rp llm-engine -- bench [-n 64] [--graphs] [--kv8]
//!   cargo run -rp llm-engine -- prefill-bench [-n 512] [--kv8]
//!   cargo run -rp llm-engine -- ppl --data path [-n tokens] [--smooth] [--int4 --gptq]
//!   cargo run -rp llm-engine -- calib-data              # WikiText-2 validation (calibration)
//!   cargo run -rp llm-engine -- gptq [--calib-tokens 1024] [--gptq-damp 0.01]  # build sidecar

mod calib;
mod cpu;
mod export;
mod gpu;
mod gptq;
mod model;
mod sample;
mod smooth;
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

/// If `--smooth` is set, calibrate per-channel activation magnitudes on the
/// validation split and return the SmoothQuant-folded model (exact in fp32,
/// only the quantized-weight error downstream changes). Otherwise the model
/// passes through untouched.
fn maybe_smooth(
    model: model::Model,
    args: &[String],
    tok: &tokenizer::Tokenizer,
) -> model::Model {
    if !flag(args, "--smooth") {
        return model;
    }
    let alpha = opt_f32(args, "--smooth-alpha", 0.5);
    let max_tokens = opt_usize(args, "--calib-tokens", 512);
    let calib_path = models_dir().join("wikitext-2-raw/wiki.calib.raw");
    assert!(
        calib_path.exists(),
        "{} not found; run `cargo run -rp llm-engine -- calib-data`",
        calib_path.display()
    );
    let text = std::fs::read_to_string(&calib_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", calib_path.display()));
    let ids = tok.encode(&text);
    let n = max_tokens.min(ids.len());
    eprintln!("SmoothQuant: calibrating on {n} tokens (alpha={alpha})...");
    let t0 = Instant::now();
    let stats = calib::collect(&model, &ids, 128, max_tokens);
    let smoothed = smooth::smooth(&model, &stats, alpha);
    eprintln!("SmoothQuant: done in {:.1}s", t0.elapsed().as_secs_f64());
    smoothed
}

/// `<model>.bin` -> `<model>.bin.gptq4.bin`, the GPTQ sidecar next to the model.
fn sidecar_path(choice: &ModelChoice) -> PathBuf {
    let mut s = choice.bin.clone().into_os_string();
    s.push(".gptq4.bin");
    PathBuf::from(s)
}

/// Loads the GPTQ sidecar when `--gptq` is set (mutually exclusive with
/// `--smooth`, since the sidecar was quantized from the un-smoothed weights).
fn load_gptq(args: &[String], choice: &ModelChoice) -> Option<gptq::Sidecar> {
    if !flag(args, "--gptq") {
        return None;
    }
    assert!(
        !flag(args, "--smooth"),
        "--gptq and --smooth are mutually exclusive"
    );
    let path = sidecar_path(choice);
    assert!(
        path.exists(),
        "{} not found; build it with `cargo run -rp llm-engine -- gptq [--model ...]`",
        path.display()
    );
    Some(gptq::Sidecar::load(&path).unwrap())
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
            gpu::WeightMode::Int4K,
            gpu::WeightMode::Int3,
            gpu::WeightMode::Int2,
        ],
        model::Arch::Qwen2 => &[
            gpu::WeightMode::Fp16,
            gpu::WeightMode::Int8,
            gpu::WeightMode::Int4,
            gpu::WeightMode::Int4K,
            gpu::WeightMode::Int3,
            gpu::WeightMode::Int2,
        ],
        model::Arch::Llama => &[
            gpu::WeightMode::Int4,
            gpu::WeightMode::Int4K,
            gpu::WeightMode::Int8,
            gpu::WeightMode::Int3,
            gpu::WeightMode::Int2,
        ],
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

fn opt_f32(args: &[String], name: &str, default: f32) -> f32 {
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

/// A mode flag restricts a sweep to that one mode (fast iteration: the
/// k-quants load-time fit is minutes on the 1.1B model).
fn mode_filter(args: &[String]) -> Option<gpu::WeightMode> {
    args.iter()
        .any(|a| {
            matches!(
                a.as_str(),
                "--fp32" | "--fp16" | "--int8" | "--int4" | "--int4k" | "--int3" | "--int2"
            )
        })
        .then(|| gpu::WeightMode::parse(args))
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
        Some("calib-data") => {
            export::download_wikitext2_calib(&models_dir());
        }
        Some("gptq") => {
            // build the GPTQ sidecar: load model -> collect Hessians on the
            // calibration split -> Hessian-guided Q4_0 -> write <model>.gptq4.bin
            let choice = model_choice(&args);
            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let model = load_model(&choice);
            let max_tokens = opt_usize(&args, "--calib-tokens", 1024);
            let damp = opt_f32(&args, "--gptq-damp", 0.01) as f64;
            let calib_path = models_dir().join("wikitext-2-raw/wiki.calib.raw");
            assert!(
                calib_path.exists(),
                "{} not found; run `cargo run -rp llm-engine -- calib-data`",
                calib_path.display()
            );
            let text = std::fs::read_to_string(&calib_path).unwrap();
            let ids = tok.encode(&text);
            let n = max_tokens.min(ids.len());
            eprintln!("GPTQ: collecting input Hessians on {n} tokens...");
            let t0 = Instant::now();
            let hess = calib::collect_hessians(&model, &ids, 128, max_tokens);
            eprintln!(
                "GPTQ: Hessians ({} positions) in {:.1}s; quantizing (damp={damp})...",
                hess.count,
                t0.elapsed().as_secs_f64()
            );
            let t1 = Instant::now();
            let act_order = !flag(&args, "--no-act-order");
            eprintln!("GPTQ: act-order {}", if act_order { "on" } else { "off" });
            let sidecar = gptq::build(&model, &hess, damp, act_order);
            let path = sidecar_path(&choice);
            sidecar.save(&path).unwrap();
            eprintln!(
                "GPTQ: quantized in {:.1}s, wrote {}",
                t1.elapsed().as_secs_f64(),
                path.display()
            );
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
            // sampling knobs: temp 0 = greedy (default), top-k 0 = off,
            // top-p 1.0 = off. Speculative decode is lossless-vs-greedy by
            // construction, so it ignores these and always decodes greedily.
            let temp = opt_f32(&args, "--temp", 0.0);
            let top_k = opt_usize(&args, "--top-k", 0);
            let top_p = opt_f32(&args, "--top-p", 1.0);
            let seed = opt_usize(&args, "--seed", 0) as u64;
            let mut sampler = if temp > 0.0 {
                sample::Sampler::new(temp, top_k, top_p, seed)
            } else {
                sample::Sampler::greedy()
            };
            let choice = model_choice(&args);

            let tok = tokenizer::Tokenizer::load(&choice.dir, choice.arch);
            let sidecar = load_gptq(&args, &choice);
            let model = maybe_smooth(load_model(&choice), &args, &tok);
            let ctx = CudaContext::new(0).unwrap();
            // GPTQ's Q4_0 sidecar only substitutes in int4 mode
            let sc = (mode == gpu::WeightMode::Int4).then_some(()).and(sidecar.as_ref());
            if sidecar.is_some() && mode != gpu::WeightMode::Int4 {
                eprintln!("note: --gptq only applies in --int4 mode; ignoring the sidecar");
            }
            let mut engine = gpu::Engine::new_quant(&ctx, &model, mode, kv8, sc);

            if spec && !sampler.is_greedy() {
                eprintln!("note: --spec is greedy by construction; ignoring sampling flags");
            }

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
                engine.generate(&ids, n_new, &mut sampler, |id| {
                    print!("{}", tok.decode(&[id]));
                    std::io::stdout().flush().unwrap();
                });
            }
            let dt = t0.elapsed().as_secs_f64();
            let decode_label = if spec {
                "prompt-lookup spec".to_string()
            } else if sampler.is_greedy() {
                "greedy".to_string()
            } else {
                format!("sample(temp={temp}, top_k={top_k}, top_p={top_p}, seed={seed})")
            };
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
                decode_label
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
                if matches!(
                    mode,
                    gpu::WeightMode::Int4
                        | gpu::WeightMode::Int4K
                        | gpu::WeightMode::Int3
                        | gpu::WeightMode::Int2
                ) {
                    if cw != gw {
                        println!(
                            "  note: {} argmax differs from fp32 CPU (quantization)",
                            mode.label()
                        );
                    }
                } else {
                    assert_eq!(cw, gw, "{mode}{kv} argmax mismatch");
                }
                assert_eq!(gw, bw, "{mode}{kv} batch prefill argmax mismatch");
                println!("  OK");
            }

            // the short prompt above stays under 64 tokens, so its batch
            // prefill only exercises the small GEMM tiers. A >64-token prompt
            // drives the wide tier (tier 3: gemm_*_wide); batch-vs-decode
            // argmax must still agree for every weight mode.
            let mut long_ids = Vec::new();
            while long_ids.len() <= 80 {
                long_ids.extend(tok.encode(
                    "Alan Turing was a British mathematician and pioneer of computer science. ",
                ));
            }
            long_ids.truncate(96);
            println!("wide-tier batch prefill (M={}):", long_ids.len());
            for &mode in modes_for(choice.arch) {
                let mut engine = gpu::Engine::new(&ctx, &model, mode, false);
                let mut logits = Vec::new();
                for (pos, &t) in long_ids.iter().enumerate() {
                    logits = engine.forward(t, pos);
                }
                let d = gpu::argmax(&logits);
                drop(engine);
                let mut engine = gpu::Engine::new(&ctx, &model, mode, false);
                let b = gpu::argmax(&engine.prefill(&long_ids, 0));
                assert_eq!(d, b, "{mode} wide-tier batch prefill argmax mismatch (M={})", long_ids.len());
                println!("  {mode}: decode={d} batch={b}  OK");
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
                let greedy =
                    greedy_engine.generate(&spec_ids, n_steps, &mut sample::Sampler::greedy(), |_| {});
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

            let only = mode_filter(&args);
            println!("| mode | weights | kv | graph | spec | tokens/sec |");
            println!("|------|---------|----|-------|------|------------|");
            for &mode in modes_for(choice.arch) {
                if only.is_some_and(|m| m != mode) {
                    continue;
                }
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

            let only = mode_filter(&args);
            println!("prompt tokens: {}", ids.len());
            println!("| mode | kv | token-loop TTFT | batch TTFT | speedup |");
            println!("|------|----|-----------------|------------|---------|");
            for &mode in modes_for(choice.arch) {
                if only.is_some_and(|m| m != mode) {
                    continue;
                }
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
            let sidecar = load_gptq(&args, &choice);
            let model = maybe_smooth(load_model(&choice), &args, &tok);
            let mut ids = tok.encode(&text);
            ids.truncate(max_tokens.min(ids.len()));
            assert!(ids.len() > 1, "need at least two tokens for perplexity");
            let ctx = CudaContext::new(0).unwrap();

            let only = mode_filter(&args);
            println!("dataset: {} ({} tokens)", data_path.display(), ids.len());
            println!("| mode | weights | kv | tokens | perplexity |");
            println!("|------|---------|----|--------|------------|");
            for kv8 in [false, true] {
                for &mode in modes_for(choice.arch) {
                    if only.is_some_and(|m| m != mode) {
                        continue;
                    }
                    // GPTQ's Q4_0 sidecar only substitutes in int4 mode
                    let sc = (mode == gpu::WeightMode::Int4)
                        .then_some(())
                        .and(sidecar.as_ref());
                    let mut engine = gpu::Engine::new_quant(&ctx, &model, mode, kv8, sc);
                    let (ppl, n) = perplexity(&mut engine, &ids);
                    println!(
                        "| {mode}{} | ~{:.0} MB | {} | {n} | {ppl:.3} |",
                        if sc.is_some() { "+gptq" } else { "" },
                        gpu::weight_mb(&model.config, mode),
                        if kv8 { "int8" } else { "fp32" },
                    );
                }
            }
        }
        _ => {
            eprintln!(
                "usage: llm-engine <export|ppl-data|calib-data|gptq|generate|verify|bench|prefill-bench|ppl> [args]"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    /// fp32 GPU forward (decode loop) and batch prefill match the CPU reference
    /// on GPT-2 — the cheap fp32 path of the `verify` subcommand, as a test.
    /// Skips gracefully (green) without a CUDA device, without compiled kernels,
    /// or before `export` has produced gpt2.bin (so CI without the model passes).
    #[test]
    fn verify_fp32_matches_cpu() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip llm verify test: no CUDA device ({e:?})");
                return;
            }
        };
        if !gpu::ptx_available() {
            eprintln!("skip llm verify test: kernels not compiled (nvcc missing at build time)");
            return;
        }
        let dir = models_dir();
        let bin = dir.join("gpt2.bin");
        if !bin.exists() {
            eprintln!(
                "skip llm verify test: {} missing (run `export` first)",
                bin.display()
            );
            return;
        }

        let tok = tokenizer::Tokenizer::load(&dir, model::Arch::Gpt2);
        let model = model::Model::load(&bin).unwrap();
        let ids = tok.encode("Alan Turing was a British mathematician");
        let want = cpu::forward(&model, &ids);

        let mut engine = gpu::Engine::new(&ctx, &model, gpu::WeightMode::Fp32, false);
        let mut got = Vec::new();
        for (pos, &t) in ids.iter().enumerate() {
            got = engine.forward(t, pos);
        }
        let batch = engine.prefill(&ids, 0);

        let err = common::allclose_err(&got, &want, 1e-2, 5e-2);
        let batch_err = common::allclose_err(&batch, &got, 1e-2, 5e-2);
        assert!(err < 1.0, "fp32 logits mismatch vs CPU: {err}");
        assert!(batch_err < 1.0, "fp32 batch prefill mismatch vs decode: {batch_err}");
        assert_eq!(gpu::argmax(&want), gpu::argmax(&got), "fp32 argmax mismatch vs CPU");
        assert_eq!(gpu::argmax(&got), gpu::argmax(&batch), "fp32 batch argmax mismatch");
    }

    /// SmoothQuant's fold is exactly weight-preserving in fp32: the smoothed
    /// model's CPU logits must match the original's (up to fp rounding). If the
    /// gamma/beta-vs-weight channel indexing is wrong, this diverges. CPU-only,
    /// so it runs without a GPU; skips green if gpt2.bin is missing.
    #[test]
    fn smooth_preserves_cpu_logits() {
        let dir = models_dir();
        let bin = dir.join("gpt2.bin");
        if !bin.exists() {
            eprintln!("skip smooth test: {} missing (run `export` first)", bin.display());
            return;
        }
        let tok = tokenizer::Tokenizer::load(&dir, model::Arch::Gpt2);
        let model = model::Model::load(&bin).unwrap();
        let ids = tok.encode("Alan Turing was a British mathematician");
        let want = cpu::forward(&model, &ids);

        // calibrate on the same prompt — enough to exercise the fold path
        let stats = calib::collect(&model, &ids, ids.len(), ids.len());
        let smoothed = smooth::smooth(&model, &stats, 0.5);
        let got = cpu::forward(&smoothed, &ids);

        let err = common::allclose_err(&got, &want, 1e-2, 5e-2);
        assert!(err < 1.0, "smoothed CPU logits diverge from original: {err}");
        assert_eq!(
            gpu::argmax(&want),
            gpu::argmax(&got),
            "smoothed argmax changed (fold should be exact in fp32)"
        );
    }
}
