//! Downloads GPT-2 124M from Hugging Face (via curl) and converts the
//! safetensors checkpoint into our model.bin format.

use std::path::Path;
use std::process::Command;

use safetensors::SafeTensors;
use serde_json::Value;

use crate::model::{Config, Layer, Model};

const HF_BASE: &str = "https://huggingface.co/openai-community/gpt2/resolve/main";
const FILES: &[&str] = &["model.safetensors", "vocab.json", "merges.txt"];
const WIKITEXT2_ROWS: &str = "https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split=test";

pub fn download(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    for f in FILES {
        let dst = dir.join(f);
        if dst.exists() {
            println!("{} already present", dst.display());
            continue;
        }
        println!("downloading {f}...");
        let status = Command::new("curl")
            .args(["-L", "--fail", "--progress-bar", "-o"])
            .arg(&dst)
            .arg(format!("{HF_BASE}/{f}"))
            .status()
            .expect("failed to run curl");
        assert!(status.success(), "download of {f} failed");
    }
}

pub fn download_wikitext2(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let out_dir = dir.join("wikitext-2-raw");
    std::fs::create_dir_all(&out_dir).unwrap();
    let test = out_dir.join("wiki.test.raw");
    if test.exists() {
        println!("{} already present", test.display());
        return;
    }

    println!("downloading WikiText-2 raw test split...");
    let mut out = String::new();
    let mut offset = 0usize;
    let page = 100usize;
    loop {
        let json_path = out_dir.join(format!("page-{offset}.json"));
        let status = Command::new("curl")
            .args(["-L", "--fail", "--silent", "--show-error", "-o"])
            .arg(&json_path)
            .arg(format!("{WIKITEXT2_ROWS}&offset={offset}&length={page}"))
            .status()
            .expect("failed to run curl");
        assert!(
            status.success(),
            "download of WikiText-2 page {offset} failed"
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

    std::fs::write(&test, out).unwrap();
    println!("wrote {}", test.display());
}

fn tensor(st: &SafeTensors, name: &str) -> Vec<f32> {
    // checkpoint tensor names come either bare or with a "transformer." prefix
    let view = st
        .tensor(name)
        .or_else(|_| st.tensor(&format!("transformer.{name}")))
        .unwrap_or_else(|_| panic!("tensor {name} not found"));
    assert_eq!(
        view.dtype(),
        safetensors::Dtype::F32,
        "{name}: expected f32"
    );
    view.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
    }
}
