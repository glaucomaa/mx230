//! Tokenizers, no external tokenizer crates.
//!
//! GPT-2/Qwen2: byte-level BPE (vocab.json + merges.txt). Pre-tokenization
//! reproduces each model's split regex with a hand-rolled scanner, since the
//! `regex` crate lacks lookahead. GPT-2:
//! ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
//! Qwen2 differs: case-insensitive contractions, a word may absorb one
//! leading non-letter, digits split one at a time, newlines separated.
//!
//! TinyLlama: SentencePiece BPE (vocab + merges parsed out of tokenizer.json).
//! No split regex at all — spaces become the U+2581 marker, a marker is
//! prepended, merges apply greedily by rank, and characters that never merged
//! into a vocab entry fall back to <0xXX> byte tokens. BOS (id 1) is
//! prepended on encode, the way the model was trained.

use std::collections::HashMap;
use std::path::Path;

use crate::model::Arch;

const SP_SPACE: char = '\u{2581}'; // '▁'

pub enum Tokenizer {
    Bpe(Box<ByteBpe>),
    Sp(SpBpe),
}

pub struct ByteBpe {
    arch: Arch,
    vocab: HashMap<String, u32>,
    inv_vocab: HashMap<u32, String>,
    merges: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

pub struct SpBpe {
    vocab: HashMap<String, u32>,
    inv_vocab: HashMap<u32, String>,
    merges: HashMap<(String, String), usize>,
    bos: u32,
}

/// GPT-2's reversible byte <-> unicode mapping: printable bytes map to
/// themselves, the rest are shifted into 256+.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(256 + extra - 1).unwrap()
        };
    }
    table
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_digit(c: char) -> bool {
    c.is_numeric()
}

/// Splits text into pre-tokens following the GPT-2 regex.
fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // contractions: 's 't 're 've 'm 'll 'd
        if c == '\'' && i + 1 < chars.len() {
            let rest: String = chars[i + 1..chars.len().min(i + 3)].iter().collect();
            let suf = ["s", "t", "re", "ve", "m", "ll", "d"]
                .iter()
                .filter(|s| rest.starts_with(*s))
                .max_by_key(|s| s.len());
            if let Some(s) = suf {
                out.push(format!("'{s}"));
                i += 1 + s.len();
                continue;
            }
        }
        // ' ?\p{L}+' / ' ?\p{N}+' / ' ?[^\s\p{L}\p{N}]+' with optional leading space
        let (start, first) = if c == ' ' && i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
            (i, chars[i + 1])
        } else {
            (i, c)
        };
        if !first.is_whitespace() {
            let mut j = if chars[start] == ' ' {
                start + 1
            } else {
                start
            };
            let class = if is_letter(first) {
                0
            } else if is_digit(first) {
                1
            } else {
                2
            };
            // a contraction apostrophe ends a punctuation run
            while j < chars.len() {
                let cj = chars[j];
                let ok = match class {
                    0 => is_letter(cj),
                    1 => is_digit(cj),
                    _ => !cj.is_whitespace() && !is_letter(cj) && !is_digit(cj),
                };
                if !ok {
                    break;
                }
                if class == 2
                    && cj == '\''
                    && j > (if chars[start] == ' ' {
                        start + 1
                    } else {
                        start
                    })
                {
                    break;
                }
                j += 1;
            }
            out.push(chars[start..j].iter().collect());
            i = j;
            continue;
        }
        // whitespace run: '\s+(?!\S)' takes all but the last ws char when a
        // non-space follows; the leftover char is either a ' ' that attaches
        // to the next token (handled by the ' ?' branch above) or stands alone
        let mut j = i;
        while j < chars.len() && chars[j].is_whitespace() {
            j += 1;
        }
        let end = if j < chars.len() && j - i > 1 {
            j - 1
        } else {
            j
        };
        out.push(chars[i..end].iter().collect());
        i = end;
    }
    out
}

/// Splits text following the Qwen2 regex:
/// (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
/// | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
fn pretokenize_qwen(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // case-insensitive contractions
        if c == '\'' && i + 1 < chars.len() {
            let rest: String = chars[i + 1..chars.len().min(i + 3)]
                .iter()
                .collect::<String>()
                .to_lowercase();
            let suf = ["s", "t", "re", "ve", "m", "ll", "d"]
                .iter()
                .filter(|s| rest.starts_with(*s))
                .max_by_key(|s| s.len());
            if let Some(s) = suf {
                out.push(chars[i..i + 1 + s.len()].iter().collect());
                i += 1 + s.len();
                continue;
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+ — a word with one optional leading symbol
        let prefix_ok = !is_letter(c) && !is_digit(c) && c != '\r' && c != '\n';
        if is_letter(c) || (prefix_ok && i + 1 < chars.len() && is_letter(chars[i + 1])) {
            let mut j = if is_letter(c) { i } else { i + 1 };
            while j < chars.len() && is_letter(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // \p{N} — one digit at a time
        if is_digit(c) {
            out.push(c.to_string());
            i += 1;
            continue;
        }
        // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        let punct_at = |k: usize| {
            k < chars.len()
                && !chars[k].is_whitespace()
                && !is_letter(chars[k])
                && !is_digit(chars[k])
        };
        if punct_at(i) || (c == ' ' && punct_at(i + 1)) {
            let mut j = if c == ' ' { i + 1 } else { i };
            while punct_at(j) {
                j += 1;
            }
            while j < chars.len() && (chars[j] == '\r' || chars[j] == '\n') {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // \s*[\r\n]+ — whitespace run ending in its last newline
        let mut j = i;
        while j < chars.len() && chars[j].is_whitespace() {
            j += 1;
        }
        let last_nl = chars[i..j]
            .iter()
            .rposition(|&w| w == '\r' || w == '\n')
            .map(|p| i + p + 1);
        if let Some(end) = last_nl {
            out.push(chars[i..end].iter().collect());
            i = end;
            continue;
        }
        // '\s+(?!\S)' / '\s+' — same trailing-space rule as GPT-2
        let end = if j < chars.len() && j - i > 1 {
            j - 1
        } else {
            j
        };
        out.push(chars[i..end].iter().collect());
        i = end;
    }
    out
}

impl Tokenizer {
    pub fn load(dir: &Path, arch: Arch) -> Self {
        match arch {
            Arch::Llama => Tokenizer::Sp(SpBpe::load(dir)),
            _ => Tokenizer::Bpe(Box::new(ByteBpe::load(dir, arch))),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Tokenizer::Bpe(t) => t.encode(text),
            Tokenizer::Sp(t) => t.encode(text),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tokenizer::Bpe(t) => t.decode(ids),
            Tokenizer::Sp(t) => t.decode(ids),
        }
    }
}

impl ByteBpe {
    fn load(dir: &Path, arch: Arch) -> Self {
        let vocab_json = std::fs::read_to_string(dir.join("vocab.json")).expect("vocab.json");
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_json).unwrap();
        let inv_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        let merges_txt = std::fs::read_to_string(dir.join("merges.txt")).expect("merges.txt");
        let merges = merges_txt
            .lines()
            .skip(1) // header line
            .filter(|l| !l.is_empty())
            .enumerate()
            .map(|(rank, l)| {
                let (a, b) = l.split_once(' ').unwrap();
                ((a.to_string(), b.to_string()), rank)
            })
            .collect();

        let byte_to_char = bytes_to_unicode();
        let char_to_byte = (0..256).map(|b| (byte_to_char[b], b as u8)).collect();
        ByteBpe {
            arch,
            vocab,
            inv_vocab,
            merges,
            byte_to_char,
            char_to_byte,
        }
    }

    fn bpe(&self, token: &str) -> Vec<u32> {
        let mut parts: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0].clone(), w[1].clone()))
                        .map(|r| (*r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
        parts
            .iter()
            .map(|p| {
                *self
                    .vocab
                    .get(p)
                    .unwrap_or_else(|| panic!("token piece {p:?} not in vocab"))
            })
            .collect()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let pre_tokens = match self.arch {
            Arch::Gpt2 => pretokenize(text),
            _ => pretokenize_qwen(text),
        };
        let mut ids = Vec::new();
        for pre in pre_tokens {
            let mapped: String = pre.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            ids.extend(self.bpe(&mapped));
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let chars: String = ids
            .iter()
            .map(|id| self.inv_vocab.get(id).map(String::as_str).unwrap_or(""))
            .collect();
        let bytes: Vec<u8> = chars
            .chars()
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl SpBpe {
    fn load(dir: &Path) -> Self {
        let json = std::fs::read_to_string(dir.join("tokenizer.json")).expect("tokenizer.json");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let model = &v["model"];
        let vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .expect("model.vocab")
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32))
            .collect();
        let inv_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        // merges come as "a b" strings in older files, ["a","b"] pairs in newer
        let merges = model["merges"]
            .as_array()
            .expect("model.merges")
            .iter()
            .enumerate()
            .map(|(rank, m)| {
                let (a, b) = match m {
                    serde_json::Value::String(s) => {
                        let (a, b) = s.split_once(' ').unwrap();
                        (a.to_string(), b.to_string())
                    }
                    serde_json::Value::Array(p) => (
                        p[0].as_str().unwrap().to_string(),
                        p[1].as_str().unwrap().to_string(),
                    ),
                    _ => panic!("unexpected merge entry"),
                };
                ((a, b), rank)
            })
            .collect();
        SpBpe {
            vocab,
            inv_vocab,
            merges,
            bos: 1,
        }
    }

    /// Greedy lowest-rank-first BPE over one piece, GPT-2-style.
    fn bpe(&self, parts: &mut Vec<String>) {
        while parts.len() > 1 {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0].clone(), w[1].clone()))
                        .map(|r| (*r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // SentencePiece normalization: prepend a space, then space -> '▁'
        let marked: String = format!(" {text}")
            .chars()
            .map(|c| if c == ' ' { SP_SPACE } else { c })
            .collect();

        // No vocab entry carries '▁' anywhere but at its start, so a merge
        // can never cross a (non-marker, marker) boundary — splitting there
        // makes per-piece BPE equal to global BPE and keeps pieces word-sized.
        let chars: Vec<char> = marked.chars().collect();
        let mut ids = vec![self.bos];
        let mut start = 0;
        for i in 0..=chars.len() {
            let boundary =
                i == chars.len() || (i > 0 && chars[i] == SP_SPACE && chars[i - 1] != SP_SPACE);
            if !boundary || i == start {
                continue;
            }
            let mut parts: Vec<String> = chars[start..i].iter().map(|c| c.to_string()).collect();
            self.bpe(&mut parts);
            for p in &parts {
                match self.vocab.get(p) {
                    Some(&id) => ids.push(id),
                    // unmerged character missing from the vocab: byte fallback
                    None => {
                        for b in p.as_bytes() {
                            let tok = format!("<0x{b:02X}>");
                            ids.push(*self.vocab.get(&tok).unwrap_or_else(|| {
                                panic!("no byte-fallback token {tok} in vocab")
                            }));
                        }
                    }
                }
            }
            start = i;
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            let Some(tok) = self.inv_vocab.get(id) else {
                continue;
            };
            // <0xXX> byte-fallback tokens, and skip <s>/</s>/<unk>
            if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                bytes.push(u8::from_str_radix(&tok[3..5], 16).unwrap());
            } else if tok.starts_with('<') && tok.ends_with('>') {
                continue;
            } else {
                for c in tok.chars() {
                    if c == SP_SPACE {
                        bytes.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
