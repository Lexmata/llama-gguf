//! Token sampling strategies for text generation
//!
//! This module provides various sampling strategies for selecting the next token
//! during text generation, including temperature, top-k, top-p (nucleus), and more.

pub mod grammar;

use rand::prelude::*;
use rand::rngs::StdRng;

use crate::tensor::Tensor;

pub use grammar::{Grammar, GrammarSampler, GbnfGrammar, JsonGrammar, RegexGrammar};

/// Mirostat sampling configuration
#[derive(Debug, Clone)]
pub struct MirostatConfig {
    /// Target surprise value (tau) - higher = more random
    pub tau: f32,
    /// Learning rate (eta)
    pub eta: f32,
    /// Mirostat version (1 or 2)
    pub version: u8,
}

impl Default for MirostatConfig {
    fn default() -> Self {
        Self {
            tau: 5.0,
            eta: 0.1,
            version: 2,
        }
    }
}

/// Sampler configuration
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for softmax scaling (higher = more random)
    pub temperature: f32,
    /// Top-K: only consider the K most likely tokens
    pub top_k: usize,
    /// Top-P (nucleus): only consider tokens with cumulative probability <= p
    pub top_p: f32,
    /// Min-P: only consider tokens with probability >= min_p * max_prob
    pub min_p: f32,
    /// Typical-P sampling
    pub typical_p: f32,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Window size for repetition penalty
    pub repeat_window: usize,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Random seed (None for random)
    pub seed: Option<u64>,
    /// Mirostat sampling (overrides other sampling methods if set)
    pub mirostat: Option<MirostatConfig>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            typical_p: 1.0,
            repeat_penalty: 1.1,
            repeat_window: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            mirostat: None,
        }
    }
}

impl SamplerConfig {
    /// Create a greedy sampling config (always picks most likely token)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            mirostat: None,
        }
    }

    /// Create a creative sampling config
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0, // Disabled
            top_p: 0.9,
            min_p: 0.05,
            typical_p: 1.0,
            repeat_penalty: 1.2,
            repeat_window: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            mirostat: None,
        }
    }

    /// Create a Mirostat v2 sampling config
    pub fn mirostat_v2(tau: f32, eta: f32) -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            mirostat: Some(MirostatConfig {
                tau,
                eta,
                version: 2,
            }),
        }
    }
}

/// Token sampler for text generation
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
    /// Token frequency counts for repetition penalty
    token_counts: Vec<u32>,
    /// Mirostat mu (adaptive parameter)
    mirostat_mu: f32,
}

impl Sampler {
    /// Create a new sampler
    pub fn new(config: SamplerConfig, vocab_size: usize) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Initialize mirostat mu based on tau
        let mirostat_mu = config
            .mirostat
            .as_ref()
            .map(|m| m.tau * 2.0)
            .unwrap_or(10.0);

        Self {
            config,
            rng,
            token_counts: vec![0; vocab_size],
            mirostat_mu,
        }
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.token_counts.fill(0);
        // Reset mirostat mu
        if let Some(ref mirostat) = self.config.mirostat {
            self.mirostat_mu = mirostat.tau * 2.0;
        }
    }

    /// Sample next token from logits
    ///
    /// # Arguments
    /// * `logits` - Logits tensor [vocab_size]
    /// * `recent_tokens` - Recently generated tokens for repetition penalty
    ///
    /// # Returns
    /// Selected token ID
    pub fn sample(&mut self, logits: &Tensor, recent_tokens: &[u32]) -> u32 {
        let logits_data = logits.as_f32().expect("Logits must be F32");
        let vocab_size = logits_data.len();

        // Debug: show top 5 tokens before sampling
        // let mut top5: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
        // top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // eprintln!("Top 5 logits: {:?}", &top5[..5.min(top5.len())]);

        // Copy logits to work with
        let mut probs: Vec<f32> = logits_data.to_vec();

        // Apply repetition penalty
        if self.config.repeat_penalty != 1.0 {
            self.apply_repetition_penalty(&mut probs, recent_tokens);
        }

        // Apply frequency and presence penalties
        if self.config.frequency_penalty != 0.0 || self.config.presence_penalty != 0.0 {
            self.apply_frequency_presence_penalty(&mut probs);
        }

        // Check if Mirostat is enabled
        if let Some(ref mirostat) = self.config.mirostat {
            return self.sample_mirostat(&mut probs, mirostat.clone());
        }

        // Apply temperature
        if self.config.temperature > 0.0 && self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for p in &mut probs {
                *p *= inv_temp;
            }
        }

        // Convert to probabilities with softmax
        let max_logit = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for p in &mut probs {
            *p = (*p - max_logit).exp();
            sum += *p;
        }
        for p in &mut probs {
            *p /= sum;
        }

        // Greedy decoding
        if self.config.temperature == 0.0 || self.config.top_k == 1 {
            return probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        // Create sorted indices by probability
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        // Apply min-p filtering
        if self.config.min_p > 0.0 {
            let threshold = probs[indices[0]] * self.config.min_p;
            let cutoff = indices
                .iter()
                .position(|&i| probs[i] < threshold)
                .unwrap_or(vocab_size);
            if cutoff > 0 {
                indices.truncate(cutoff);
            }
        }

        // Apply top-k filtering
        if self.config.top_k > 0 && self.config.top_k < indices.len() {
            indices.truncate(self.config.top_k);
        }

        // Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let cutoff = indices
                .iter()
                .position(|&i| {
                    cumsum += probs[i];
                    cumsum > self.config.top_p
                })
                .unwrap_or(indices.len());
            if cutoff > 0 {
                indices.truncate(cutoff + 1); // Include the token that crossed threshold
            }
        }

        // Renormalize probabilities over filtered tokens
        let filtered_sum: f32 = indices.iter().map(|&i| probs[i]).sum();
        for &i in &indices {
            probs[i] /= filtered_sum;
        }

        // Sample from filtered distribution
        let r: f32 = self.rng.r#gen();
        let mut cumsum = 0.0f32;
        for &i in &indices {
            cumsum += probs[i];
            if r < cumsum {
                let token_id = i as u32;
                self.token_counts[i] += 1;
                return token_id;
            }
        }

        // Fallback to last token in filtered set
        let token_id = *indices.last().unwrap() as u32;
        self.token_counts[token_id as usize] += 1;
        token_id
    }

    /// Mirostat sampling algorithm
    ///
    /// Mirostat dynamically adjusts the sampling to target a specific "surprise" level.
    fn sample_mirostat(&mut self, logits: &mut [f32], config: MirostatConfig) -> u32 {
        let vocab_size = logits.len();

        // Convert logits to probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for p in logits.iter_mut() {
            *p = (*p - max_logit).exp();
            sum += *p;
        }
        for p in logits.iter_mut() {
            *p /= sum;
        }

        // Sort tokens by probability (descending)
        let mut sorted_indices: Vec<usize> = (0..vocab_size).collect();
        sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

        let token_id = if config.version == 1 {
            // Mirostat v1: uses a fixed number of candidates based on mu
            let n = ((2.0f32.powf(self.mirostat_mu) * vocab_size as f32) as usize)
                .max(1)
                .min(vocab_size);

            // Truncate to top n candidates
            let candidates = &sorted_indices[..n];

            // Renormalize and sample
            let filtered_sum: f32 = candidates.iter().map(|&i| logits[i]).sum();
            let r: f32 = self.rng.r#gen::<f32>() * filtered_sum;
            let mut cumsum = 0.0f32;
            let mut selected = candidates[0];
            for &i in candidates {
                cumsum += logits[i];
                if cumsum > r {
                    selected = i;
                    break;
                }
            }
            selected
        } else {
            // Mirostat v2: uses mu to truncate based on surprise
            // Find the truncation point where -log2(p) > mu
            let mu = self.mirostat_mu;

            let mut truncation_idx = vocab_size;
            for (rank, &i) in sorted_indices.iter().enumerate() {
                let surprise = -logits[i].log2();
                if surprise > mu {
                    truncation_idx = rank.max(1);
                    break;
                }
            }

            // Sample from truncated distribution
            let candidates = &sorted_indices[..truncation_idx];
            let filtered_sum: f32 = candidates.iter().map(|&i| logits[i]).sum();
            let r: f32 = self.rng.r#gen::<f32>() * filtered_sum;
            let mut cumsum = 0.0f32;
            let mut selected = candidates[0];
            for &i in candidates {
                cumsum += logits[i];
                if cumsum > r {
                    selected = i;
                    break;
                }
            }
            selected
        };

        // Update mu based on the surprise of the selected token
        let selected_prob = logits[token_id];
        let surprise = -selected_prob.log2();
        self.mirostat_mu -= config.eta * (surprise - config.tau);

        // Clamp mu to reasonable bounds
        self.mirostat_mu = self.mirostat_mu.clamp(0.0, 20.0);

        self.token_counts[token_id] += 1;
        token_id as u32
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &mut [f32], recent_tokens: &[u32]) {
        let window = if self.config.repeat_window > 0 {
            recent_tokens.len().min(self.config.repeat_window)
        } else {
            recent_tokens.len()
        };

        let start = recent_tokens.len().saturating_sub(window);
        for &token_id in &recent_tokens[start..] {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.config.repeat_penalty;
                } else {
                    logits[idx] *= self.config.repeat_penalty;
                }
            }
        }
    }

    /// Apply frequency and presence penalties
    fn apply_frequency_presence_penalty(&self, logits: &mut [f32]) {
        for (i, &count) in self.token_counts.iter().enumerate() {
            if count > 0 {
                // Frequency penalty: scales with count
                logits[i] -= self.config.frequency_penalty * count as f32;
                // Presence penalty: constant if present
                logits[i] -= self.config.presence_penalty;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SamplerConfig::default();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 40);
        assert!((config.top_p - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_greedy_config() {
        let config = SamplerConfig::greedy();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
    }

    #[test]
    fn test_greedy_sampling() {
        let config = SamplerConfig::greedy();
        let mut sampler = Sampler::new(config, 10);

        // Create logits where token 5 has highest probability
        let logits_data = vec![0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.2, 0.1, 0.0, -0.1];
        let logits = Tensor::from_f32(&logits_data, vec![10]).unwrap();

        let token = sampler.sample(&logits, &[]);
        assert_eq!(token, 5);
    }

    #[test]
    fn test_sampler_reset() {
        let config = SamplerConfig::default();
        let mut sampler = Sampler::new(config, 10);

        sampler.token_counts[5] = 10;
        sampler.reset();

        assert_eq!(sampler.token_counts[5], 0);
    }
}
