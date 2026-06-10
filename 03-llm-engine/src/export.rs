//! Downloads GPT-2 124M from Hugging Face (via curl) and converts the
//! safetensors checkpoint into our model.bin format.

use std::path::Path;
use std::process::Command;

use safetensors::SafeTensors;

use crate::model::{Config, Layer, Model};

const HF_BASE: &str = "https://huggingface.co/openai-community/gpt2/resolve/main";
const FILES: &[&str] = &["model.safetensors", "vocab.json", "merges.txt"];

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

fn tensor(st: &SafeTensors, name: &str) -> Vec<f32> {
    // checkpoint tensor names come either bare or with a "transformer." prefix
    let view = st
        .tensor(name)
        .or_else(|_| st.tensor(&format!("transformer.{name}")))
        .unwrap_or_else(|_| panic!("tensor {name} not found"));
    assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}: expected f32");
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
