//! GPT-2 byte-level BPE tokenizer (vocab.json + merges.txt), no external
//! tokenizer crates. Pre-tokenization reproduces the GPT-2 split regex
//! ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
//! with a hand-rolled scanner, since the `regex` crate lacks lookahead.

use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    inv_vocab: HashMap<u32, String>,
    merges: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
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

impl Tokenizer {
    pub fn load(dir: &Path) -> Self {
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
        Tokenizer {
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

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for pre in pretokenize(text) {
            let mapped: String = pre.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            ids.extend(self.bpe(&mapped));
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
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
