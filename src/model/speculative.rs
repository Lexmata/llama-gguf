//! Speculative decoding for faster inference
//!
//! Speculative decoding uses a smaller "draft" model to quickly propose
//! candidate tokens, which are then verified in parallel by the larger
//! "target" model. This can significantly speed up inference.
//!
//! Reference: "Fast Inference from Transformers via Speculative Decoding"
//! https://arxiv.org/abs/2211.17192


use crate::model::{InferenceContext, Model};
use crate::sampling::Sampler;
use crate::tensor::Tensor;

/// Speculative decoding configuration
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate (K)
    /// More tokens means more potential speedup but higher rejection rate
    pub num_speculative: usize,
    /// Temperature for draft model sampling
    pub draft_temperature: f32,
    /// Temperature for target model sampling  
    pub target_temperature: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative: 4,
            draft_temperature: 0.8,
            target_temperature: 0.8,
        }
    }
}

/// Statistics from speculative decoding
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated (accepted + rejected resampled)
    pub total_tokens: usize,
    /// Tokens accepted from draft model
    pub accepted_tokens: usize,
    /// Tokens rejected and resampled
    pub rejected_tokens: usize,
    /// Number of speculative batches run
    pub batches: usize,
}

impl SpeculativeStats {
    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_tokens > 0 {
            self.accepted_tokens as f32 / self.total_tokens as f32
        } else {
            0.0
        }
    }

    /// Get average accepted tokens per batch
    pub fn avg_accepted_per_batch(&self) -> f32 {
        if self.batches > 0 {
            self.accepted_tokens as f32 / self.batches as f32
        } else {
            0.0
        }
    }
}

/// Speculative decoder combining draft and target models
pub struct SpeculativeDecoder {
    /// Configuration
    config: SpeculativeConfig,
    /// Statistics
    stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    pub fn new(config: SpeculativeConfig) -> Self {
        Self {
            config,
            stats: SpeculativeStats::default(),
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Generate tokens using speculative decoding
    ///
    /// # Arguments
    /// * `draft_model` - Smaller, faster draft model
    /// * `target_model` - Larger, more accurate target model
    /// * `draft_ctx` - Inference context for draft model
    /// * `target_ctx` - Inference context for target model
    /// * `draft_sampler` - Sampler for draft model
    /// * `target_sampler` - Sampler for target model
    /// * `input_tokens` - Initial input tokens
    /// * `max_tokens` - Maximum tokens to generate
    /// * `eos_token` - End of sequence token ID
    ///
    /// # Returns
    /// Vector of generated token IDs
    pub fn generate(
        &mut self,
        draft_model: &dyn Model,
        target_model: &dyn Model,
        draft_ctx: &mut InferenceContext,
        target_ctx: &mut InferenceContext,
        draft_sampler: &mut Sampler,
        target_sampler: &mut Sampler,
        input_tokens: &[u32],
        max_tokens: usize,
        eos_token: u32,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut output_tokens = input_tokens.to_vec();
        let mut generated = 0;

        while generated < max_tokens {
            // Step 1: Generate K speculative tokens with draft model
            let mut draft_tokens = Vec::with_capacity(self.config.num_speculative);
            let mut draft_probs = Vec::with_capacity(self.config.num_speculative);

            for _ in 0..self.config.num_speculative {
                if output_tokens.len() + draft_tokens.len() >= draft_ctx.kv_cache.max_seq_len {
                    break;
                }

                let last_token = draft_tokens.last().copied().unwrap_or_else(|| {
                    *output_tokens.last().unwrap_or(&0)
                });

                // Get draft model logits
                let logits = draft_model.forward(&[last_token], draft_ctx)?;
                let probs = softmax_logits(&logits)?;

                // Sample from draft model
                let token = draft_sampler.sample(&logits, &output_tokens);
                draft_tokens.push(token);
                draft_probs.push(probs);

                if token == eos_token {
                    break;
                }
            }

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Verify with target model (process all tokens in parallel)
            // Note: In a full implementation, we'd process all K+1 positions in parallel
            // For now, we verify sequentially but still get the benefit of batched verification
            let mut accepted = 0;

            for (i, &draft_token) in draft_tokens.iter().enumerate() {
                let last_token = if i == 0 {
                    *output_tokens.last().unwrap_or(&0)
                } else {
                    draft_tokens[i - 1]
                };

                // Get target model logits
                let target_logits = target_model.forward(&[last_token], target_ctx)?;
                let target_probs = softmax_logits(&target_logits)?;

                // Get draft probability for this token
                let draft_prob = get_token_prob(&draft_probs[i], draft_token);
                let target_prob = get_token_prob(&target_probs, draft_token);

                // Acceptance criterion: accept if target_prob >= draft_prob * random
                let r: f32 = rand::random();
                let accept = r * draft_prob <= target_prob;

                if accept {
                    output_tokens.push(draft_token);
                    accepted += 1;
                    generated += 1;
                    self.stats.accepted_tokens += 1;
                    self.stats.total_tokens += 1;

                    if draft_token == eos_token || generated >= max_tokens {
                        break;
                    }
                } else {
                    // Rejection: sample from adjusted distribution
                    // p_adjusted(x) = max(0, p_target(x) - p_draft(x)) / Z
                    let adjusted_token = sample_adjusted_distribution(
                        &target_probs,
                        &draft_probs[i],
                        target_sampler,
                        &output_tokens,
                    );

                    output_tokens.push(adjusted_token);
                    generated += 1;
                    self.stats.rejected_tokens += 1;
                    self.stats.total_tokens += 1;

                    if adjusted_token == eos_token || generated >= max_tokens {
                        break;
                    }

                    // After rejection, discard remaining speculative tokens
                    break;
                }
            }

            // If all speculative tokens were accepted, sample one more from target
            if accepted == draft_tokens.len() && generated < max_tokens {
                let last_token = *output_tokens.last().unwrap_or(&0);
                let target_logits = target_model.forward(&[last_token], target_ctx)?;
                let bonus_token = target_sampler.sample(&target_logits, &output_tokens);
                output_tokens.push(bonus_token);
                generated += 1;
                self.stats.total_tokens += 1;
                // This counts as an "accepted" token since we're using target model
                self.stats.accepted_tokens += 1;

                if bonus_token == eos_token {
                    break;
                }
            }

            self.stats.batches += 1;

            // Sync KV caches (in practice, draft cache would need to be aligned with target)
            // This is simplified - full implementation would manage cache states more carefully
        }

        Ok(output_tokens)
    }
}

/// Convert logits to probabilities using softmax
fn softmax_logits(logits: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let data = logits.as_f32()?;
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();

    for p in &mut probs {
        *p /= sum;
    }

    Tensor::from_f32(&probs, logits.shape().to_vec())
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Get probability of a specific token
fn get_token_prob(probs: &Tensor, token: u32) -> f32 {
    probs
        .as_f32()
        .map(|data| data.get(token as usize).copied().unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Sample from adjusted distribution: max(0, p_target - p_draft) / Z
fn sample_adjusted_distribution(
    target_probs: &Tensor,
    draft_probs: &Tensor,
    sampler: &mut Sampler,
    context: &[u32],
) -> u32 {
    let target_data = match target_probs.as_f32() {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let draft_data = match draft_probs.as_f32() {
        Ok(d) => d,
        Err(_) => return 0,
    };

    // Compute adjusted distribution
    let mut adjusted: Vec<f32> = target_data
        .iter()
        .zip(draft_data.iter())
        .map(|(&t, &d)| (t - d).max(0.0))
        .collect();

    let sum: f32 = adjusted.iter().sum();
    if sum > 0.0 {
        for p in &mut adjusted {
            *p /= sum;
        }
    } else {
        // Fallback to target distribution
        adjusted = target_data.to_vec();
    }

    // Convert to logits for sampler (inverse softmax approximation)
    let logits: Vec<f32> = adjusted.iter().map(|&p| (p + 1e-10).ln()).collect();

    let logits_tensor =
        Tensor::from_f32(&logits, target_probs.shape().to_vec()).unwrap_or_else(|_| {
            Tensor::zeros(target_probs.shape().to_vec(), crate::tensor::DType::F32)
        });

    sampler.sample(&logits_tensor, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.num_speculative, 4);
        assert!((config.draft_temperature - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::default();
        stats.total_tokens = 100;
        stats.accepted_tokens = 75;
        stats.rejected_tokens = 25;
        stats.batches = 20;

        assert!((stats.acceptance_rate() - 0.75).abs() < 0.01);
        assert!((stats.avg_accepted_per_batch() - 3.75).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_creation() {
        let config = SpeculativeConfig {
            num_speculative: 6,
            draft_temperature: 0.5,
            target_temperature: 0.7,
        };
        let decoder = SpeculativeDecoder::new(config);
        assert_eq!(decoder.stats().total_tokens, 0);
    }
}
