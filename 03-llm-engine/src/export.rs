//! Downloads GPT-2 124M from Hugging Face (via curl) and converts the
//! safetensors checkpoint into our model.bin format.

use std::path::Path;
use std::process::Command;

use safetensors::SafeTensors;
use serde_json::Value;

use crate::model::{Config, Layer, Model};

const HF_BASE: &str = "https://huggingface.co/openai-community/gpt2/resolve/main";
const HF_QWEN_BASE: &str = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main";
const HF_TINYLLAMA_BASE: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/resolve/main";
const FILES: &[&str] = &["model.safetensors", "vocab.json", "merges.txt"];
// SentencePiece-family checkpoint: vocab+merges live inside tokenizer.json
const FILES_SP: &[&str] = &["model.safetensors", "tokenizer.json"];

fn fetch(base: &str, dir: &Path, files: &[&str]) {
    std::fs::create_dir_all(dir).unwrap();
    for f in files {
        let dst = dir.join(f);
        if dst.exists() {
            println!("{} already present", dst.display());
            continue;
        }
        println!("downloading {f}...");
        let status = Command::new("curl")
            .args(["-L", "--fail", "--progress-bar", "-o"])
            .arg(&dst)
            .arg(format!("{base}/{f}"))
            .status()
            .expect("failed to run curl");
        assert!(status.success(), "download of {f} failed");
    }
}

pub fn download(dir: &Path) {
    fetch(HF_BASE, dir, FILES);
}

pub fn download_qwen(dir: &Path) {
    fetch(HF_QWEN_BASE, dir, FILES);
}

pub fn download_tinyllama(dir: &Path) {
    fetch(HF_TINYLLAMA_BASE, dir, FILES_SP);
}

/// The WikiText-2 `test` split — the perplexity eval corpus.
pub fn download_wikitext2(dir: &Path) {
    download_wikitext2_split(dir, "test", "wiki.test.raw");
}

/// The WikiText-2 `validation` split — a *separate* corpus for SmoothQuant /
/// GPTQ calibration (never the test split used for ppl, to avoid measuring on
/// data the quantizer was tuned to).
pub fn download_wikitext2_calib(dir: &Path) {
    download_wikitext2_split(dir, "validation", "wiki.calib.raw");
}

fn download_wikitext2_split(dir: &Path, split: &str, filename: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let out_dir = dir.join("wikitext-2-raw");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out_file = out_dir.join(filename);
    if out_file.exists() {
        println!("{} already present", out_file.display());
        return;
    }

    let base = format!(
        "https://datasets-server.huggingface.co/rows\
         ?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split={split}"
    );
    println!("downloading WikiText-2 raw {split} split...");
    let mut out = String::new();
    let mut offset = 0usize;
    let page = 100usize;
    loop {
        let json_path = out_dir.join(format!("page-{split}-{offset}.json"));
        let status = Command::new("curl")
            .args(["-L", "--fail", "--silent", "--show-error", "-o"])
            .arg(&json_path)
            .arg(format!("{base}&offset={offset}&length={page}"))
            .status()
            .expect("failed to run curl");
        assert!(
            status.success(),
            "download of WikiText-2 {split} page {offset} failed"
        );

        let bytes = std::fs::read(&json_path).unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let rows = v["rows"].as_array().expect("rows must be an array");
        if rows.is_empty() {
            break;
        }
        for row in rows {
            let text = row["row"]["text"]
                .as_str()
                .expect("row.text must be a string");
            out.push_str(text);
            if !text.ends_with('\n') {
                out.push('\n');
            }
        }
        let total = v["num_rows_total"]
            .as_u64()
            .expect("num_rows_total missing") as usize;
        let _ = std::fs::remove_file(&json_path);
        offset += rows.len();
        if offset >= total {
            break;
        }
    }

    std::fs::write(&out_file, out).unwrap();
    println!("wrote {}", out_file.display());
}

fn tensor(st: &SafeTensors, name: &str) -> Vec<f32> {
    // checkpoint tensor names come either bare or with a "transformer." prefix
    let view = st
        .tensor(name)
        .or_else(|_| st.tensor(&format!("transformer.{name}")))
        .unwrap_or_else(|_| panic!("tensor {name} not found"));
    match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|c| {
                let hi = u16::from_le_bytes(c.try_into().unwrap());
                f32::from_bits((hi as u32) << 16)
            })
            .collect(),
        d => panic!("{name}: unsupported dtype {d:?}"),
    }
}

/// HF Linear stores [n_out, n_in]; our GEMV wants [n_in, n_out].
fn transpose(w: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; w.len()];
    for o in 0..n_out {
        for i in 0..n_in {
            t[i * n_out + o] = w[o * n_in + i];
        }
    }
    t
}

pub fn convert(dir: &Path) -> Model {
    let buf = std::fs::read(dir.join("model.safetensors")).expect("run `export` download first");
    let st = SafeTensors::deserialize(&buf).unwrap();
    let config = Config::gpt2_small();

    let layers = (0..config.n_layer)
        .map(|l| Layer {
            ln1_g: tensor(&st, &format!("h.{l}.ln_1.weight")),
            ln1_b: tensor(&st, &format!("h.{l}.ln_1.bias")),
            qkv_w: tensor(&st, &format!("h.{l}.attn.c_attn.weight")),
            qkv_b: tensor(&st, &format!("h.{l}.attn.c_attn.bias")),
            proj_w: tensor(&st, &format!("h.{l}.attn.c_proj.weight")),
            proj_b: tensor(&st, &format!("h.{l}.attn.c_proj.bias")),
            ln2_g: tensor(&st, &format!("h.{l}.ln_2.weight")),
            ln2_b: tensor(&st, &format!("h.{l}.ln_2.bias")),
            fc_w: tensor(&st, &format!("h.{l}.mlp.c_fc.weight")),
            fc_b: tensor(&st, &format!("h.{l}.mlp.c_fc.bias")),
            up_w: Vec::new(),
            fc2_w: tensor(&st, &format!("h.{l}.mlp.c_proj.weight")),
            fc2_b: tensor(&st, &format!("h.{l}.mlp.c_proj.bias")),
        })
        .collect();

    Model {
        config,
        wte: tensor(&st, "wte.weight"),
        wpe: tensor(&st, "wpe.weight"),
        layers,
        lnf_g: tensor(&st, "ln_f.weight"),
        lnf_b: tensor(&st, "ln_f.bias"),
        lm_head: Vec::new(),
    }
}

pub fn convert_qwen(dir: &Path) -> Model {
    let buf = std::fs::read(dir.join("model.safetensors")).expect("run qwen download first");
    let st = SafeTensors::deserialize(&buf).unwrap();
    let config = Config::qwen25_05b();
    let (e, inter) = (config.n_embd, config.n_inter);
    let (qd, kvd, qkvd) = (config.q_dim(), config.kv_dim(), config.qkv_dim());

    let layers = (0..config.n_layer)
        .map(|l| {
            let p = format!("model.layers.{l}");
            // q/k/v are separate Linears; concatenate into one [e, q+k+v] GEMV
            let q = transpose(&tensor(&st, &format!("{p}.self_attn.q_proj.weight")), qd, e);
            let k = transpose(
                &tensor(&st, &format!("{p}.self_attn.k_proj.weight")),
                kvd,
                e,
            );
            let v = transpose(
                &tensor(&st, &format!("{p}.self_attn.v_proj.weight")),
                kvd,
                e,
            );
            let mut qkv_w = vec![0.0f32; e * qkvd];
            for i in 0..e {
                qkv_w[i * qkvd..i * qkvd + qd].copy_from_slice(&q[i * qd..(i + 1) * qd]);
                qkv_w[i * qkvd + qd..i * qkvd + qd + kvd]
                    .copy_from_slice(&k[i * kvd..(i + 1) * kvd]);
                qkv_w[i * qkvd + qd + kvd..(i + 1) * qkvd]
                    .copy_from_slice(&v[i * kvd..(i + 1) * kvd]);
            }
            let mut qkv_b = tensor(&st, &format!("{p}.self_attn.q_proj.bias"));
            qkv_b.extend(tensor(&st, &format!("{p}.self_attn.k_proj.bias")));
            qkv_b.extend(tensor(&st, &format!("{p}.self_attn.v_proj.bias")));

            Layer {
                ln1_g: tensor(&st, &format!("{p}.input_layernorm.weight")),
                ln1_b: Vec::new(),
                qkv_w,
                qkv_b,
                proj_w: transpose(&tensor(&st, &format!("{p}.self_attn.o_proj.weight")), e, qd),
                proj_b: vec![0.0; e],
                ln2_g: tensor(&st, &format!("{p}.post_attention_layernorm.weight")),
                ln2_b: Vec::new(),
                fc_w: transpose(&tensor(&st, &format!("{p}.mlp.gate_proj.weight")), inter, e),
                fc_b: Vec::new(),
                up_w: transpose(&tensor(&st, &format!("{p}.mlp.up_proj.weight")), inter, e),
                fc2_w: transpose(&tensor(&st, &format!("{p}.mlp.down_proj.weight")), e, inter),
                fc2_b: Vec::new(),
            }
        })
        .collect();

    Model {
        config,
        wte: tensor(&st, "model.embed_tokens.weight"), // tied lm_head
        wpe: Vec::new(),                               // RoPE, no learned positions
        layers,
        lnf_g: tensor(&st, "model.norm.weight"),
        lnf_b: Vec::new(),
        lm_head: Vec::new(),
    }
}

/// TinyLlama-1.1B: same layer layout as Qwen2 (the q/k/v concatenation and
/// HF Linear transposes are identical) but bias-free and with an untied
/// lm_head tensor.
pub fn convert_tinyllama(dir: &Path) -> Model {
    // the fp32 checkpoint is 4.4 GB — mmap it instead of reading into RAM
    let file =
        std::fs::File::open(dir.join("model.safetensors")).expect("run tinyllama download first");
    let buf = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let st = SafeTensors::deserialize(&buf).unwrap();
    let config = Config::tinyllama_11b();
    let (e, inter) = (config.n_embd, config.n_inter);
    let (qd, kvd, qkvd) = (config.q_dim(), config.kv_dim(), config.qkv_dim());

    let layers = (0..config.n_layer)
        .map(|l| {
            let p = format!("model.layers.{l}");
            let q = transpose(&tensor(&st, &format!("{p}.self_attn.q_proj.weight")), qd, e);
            let k = transpose(
                &tensor(&st, &format!("{p}.self_attn.k_proj.weight")),
                kvd,
                e,
            );
            let v = transpose(
                &tensor(&st, &format!("{p}.self_attn.v_proj.weight")),
                kvd,
                e,
            );
            let mut qkv_w = vec![0.0f32; e * qkvd];
            for i in 0..e {
                qkv_w[i * qkvd..i * qkvd + qd].copy_from_slice(&q[i * qd..(i + 1) * qd]);
                qkv_w[i * qkvd + qd..i * qkvd + qd + kvd]
                    .copy_from_slice(&k[i * kvd..(i + 1) * kvd]);
                qkv_w[i * qkvd + qd + kvd..(i + 1) * qkvd]
                    .copy_from_slice(&v[i * kvd..(i + 1) * kvd]);
            }

            Layer {
                ln1_g: tensor(&st, &format!("{p}.input_layernorm.weight")),
                ln1_b: Vec::new(),
                qkv_w,
                qkv_b: vec![0.0; qkvd],
                proj_w: transpose(&tensor(&st, &format!("{p}.self_attn.o_proj.weight")), e, qd),
                proj_b: vec![0.0; e],
                ln2_g: tensor(&st, &format!("{p}.post_attention_layernorm.weight")),
                ln2_b: Vec::new(),
                fc_w: transpose(&tensor(&st, &format!("{p}.mlp.gate_proj.weight")), inter, e),
                fc_b: Vec::new(),
                up_w: transpose(&tensor(&st, &format!("{p}.mlp.up_proj.weight")), inter, e),
                fc2_w: transpose(&tensor(&st, &format!("{p}.mlp.down_proj.weight")), e, inter),
                fc2_b: Vec::new(),
            }
        })
        .collect();

    Model {
        config,
        wte: tensor(&st, "model.embed_tokens.weight"),
        wpe: Vec::new(),
        layers,
        lnf_g: tensor(&st, "model.norm.weight"),
        lnf_b: Vec::new(),
        lm_head: tensor(&st, "lm_head.weight"),
    }
}
