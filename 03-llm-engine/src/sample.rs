//! Host-side token sampler.
//!
//! Greedy is the default everywhere: `verify`, `bench`, `ppl` and speculative
//! decode all stay bit-identical to `gpu::argmax` (speculative decode is lossless
//! *by construction* against greedy, so it never samples). Only the `generate`
//! subcommand opts into temperature / top-k / top-p (nucleus) sampling so the
//! demo text stops looping. Logits already come back to the host in `generate`,
//! so sampling adds no device work — one sort of the vocab per token (a few ms,
//! negligible next to a forward pass).

use crate::gpu;

pub struct Sampler {
    temp: f32,    // <= 0.0 means greedy (argmax)
    top_k: usize, // 0 = disabled
    top_p: f32,   // >= 1.0 = disabled
    rng: u64,
}

impl Sampler {
    /// Greedy decode: reproduces `gpu::argmax` exactly (first-index tie-break),
    /// so paths that must match greedy can pass this and stay identical.
    pub fn greedy() -> Self {
        Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            rng: 0,
        }
    }

    pub fn new(temp: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Sampler {
            temp,
            top_k,
            top_p,
            // xorshift64* is stuck at zero, so never seed it there
            rng: seed.max(1),
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0
    }

    /// xorshift64* — same generator as `common::pseudo_rand`.
    fn next_u64(&mut self) -> u64 {
        let mut s = self.rng;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.rng = s;
        s.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pick the next token id from raw logits.
    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        if self.is_greedy() {
            return gpu::argmax(logits);
        }

        // sort candidates by logit, descending (so cand[0] is the max)
        let mut cand: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // top-k truncation
        if self.top_k > 0 && self.top_k < cand.len() {
            cand.truncate(self.top_k);
        }

        // temperature softmax over the surviving candidates (max already at [0])
        let maxl = cand[0].1;
        let mut probs: Vec<f32> = cand
            .iter()
            .map(|&(_, l)| ((l - maxl) / self.temp).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // top-p (nucleus): keep the shortest descending prefix whose cumulative
        // probability first reaches top_p
        let mut cutoff = probs.len();
        if self.top_p < 1.0 {
            let mut cum = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
        }

        // sample from the renormalized nucleus
        let nucleus_sum: f32 = probs[..cutoff].iter().sum();
        let r = self.next_f32() * nucleus_sum;
        let mut acc = 0.0f32;
        for i in 0..cutoff {
            acc += probs[i];
            if r < acc {
                return cand[i].0 as u32;
            }
        }
        cand[cutoff - 1].0 as u32
    }
}
