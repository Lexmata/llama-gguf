//! LLaMA model architecture implementation
//!
//! This module implements the LLaMA transformer architecture, supporting:
//! - LLaMA 1, 2, 3 variants
//! - Grouped Query Attention (GQA)
//! - RoPE position embeddings
//! - Quantized weights

use crate::backend::Backend;
use crate::tensor::{DType, Tensor};

use super::config::ModelConfig;
use super::error::{ModelError, ModelResult};
use super::layers::{Linear, RMSNorm, TransformerLayer};
use super::{Architecture, InferenceContext, Model};

/// LLaMA model implementation
pub struct LlamaModel {
    /// Model configuration
    config: ModelConfig,
    /// Token embedding matrix [vocab_size, hidden_size]
    token_embedding: Tensor,
    /// Transformer layers
    layers: Vec<TransformerLayer>,
    /// Final RMS normalization
    norm: RMSNorm,
    /// Output projection (may share weights with embedding)
    output: Linear,
    /// Model architecture variant
    architecture: Architecture,
}

impl LlamaModel {
    /// Create a new LLaMA model from loaded weights
    pub fn new(
        config: ModelConfig,
        token_embedding: Tensor,
        layers: Vec<TransformerLayer>,
        norm: RMSNorm,
        output: Linear,
        architecture: Architecture,
    ) -> ModelResult<Self> {
        // Validate configuration
        if layers.len() != config.num_layers {
            return Err(ModelError::ConfigError(format!(
                "Expected {} layers, got {}",
                config.num_layers,
                layers.len()
            )));
        }

        Ok(Self {
            config,
            token_embedding,
            layers,
            norm,
            output,
            architecture,
        })
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get transformer layers
    pub fn layers(&self) -> &[TransformerLayer] {
        &self.layers
    }

    /// Get final normalization layer
    pub fn norm(&self) -> &RMSNorm {
        &self.norm
    }

    /// Get output projection layer  
    pub fn output(&self) -> &Linear {
        &self.output
    }

    /// Get token embedding tensor
    pub fn token_embedding(&self) -> &Tensor {
        &self.token_embedding
    }

    /// Get token embedding for given token IDs (public for testing)
    pub fn embed_tokens(&self, tokens: &[u32], backend: &dyn Backend) -> ModelResult<Tensor> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let seq_len = tokens.len();

        // Handle both quantized and non-quantized embeddings
        let embedding_data: Vec<f32> = if self.token_embedding.dtype() == DType::F32 {
            self.token_embedding.as_f32()?.to_vec()
        } else {
            // Dequantize the embedding tensor
            let numel = self.token_embedding.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            backend.dequantize(&self.token_embedding, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        let mut output = vec![0.0f32; seq_len * hidden_size];

        // GGUF stores embeddings with shape listed as [hidden_size, vocab_size]
        // but in GGML convention, this means the data is laid out as [vocab_size][hidden_size]
        // i.e., each row is a token's embedding vector
        // So embedding for token t starts at t * hidden_size
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = token as usize;
            if token_idx >= vocab_size {
                return Err(ModelError::InvalidMetadata {
                    key: "token".into(),
                    message: format!("Token ID {} exceeds vocab size {}", token, vocab_size),
                });
            }

            let src_start = token_idx * hidden_size;
            let src_end = src_start + hidden_size;

            if src_end > embedding_data.len() {
                return Err(ModelError::InvalidMetadata {
                    key: "embedding".into(),
                    message: format!(
                        "Embedding index out of bounds: token_idx={}, src_end={}, embedding_len={}",
                        token_idx,
                        src_end,
                        embedding_data.len()
                    ),
                });
            }

            let dst_start = i * hidden_size;
            output[dst_start..dst_start + hidden_size]
                .copy_from_slice(&embedding_data[src_start..src_end]);
        }

        if seq_len == 1 {
            Tensor::from_f32(&output, vec![hidden_size])
        } else {
            Tensor::from_f32(&output, vec![seq_len, hidden_size])
        }
        .map_err(|e| e.into())
    }

    /// Compute logits from hidden state
    fn compute_logits(&self, hidden: &Tensor, backend: &dyn Backend) -> ModelResult<Tensor> {
        // Apply final normalization
        let mut normed = Tensor::zeros(hidden.shape().to_vec(), DType::F32);
        self.norm.forward(hidden, &mut normed, backend)?;

        // Project to vocabulary
        let mut logits = Tensor::zeros(vec![self.config.vocab_size], DType::F32);
        self.output.forward(&normed, &mut logits, backend)?;

        Ok(logits)
    }
}

impl Model for LlamaModel {
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor> {
        let backend = ctx.backend.as_ref();

        // Check context length
        let new_pos = ctx.position + tokens.len();
        if new_pos > self.config.max_seq_len {
            return Err(ModelError::ContextLengthExceeded {
                current: new_pos,
                max: self.config.max_seq_len,
            });
        }

        // Process each token sequentially to properly build KV cache
        let mut final_hidden = None;

        for (token_offset, &token) in tokens.iter().enumerate() {
            let current_pos = ctx.position + token_offset;

            // Get embedding for single token
            let mut hidden = self.embed_tokens(&[token], backend)?;

            // Run through transformer layers
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(
                    &hidden,
                    &mut ctx.kv_cache.k_cache[layer_idx],
                    &mut ctx.kv_cache.v_cache[layer_idx],
                    current_pos,
                    self.config.rope_config.freq_base,
                    self.config.rope_config.freq_scale,
                    backend,
                )?;
            }

            final_hidden = Some(hidden);
        }

        // Update position
        ctx.position = new_pos;
        ctx.kv_cache.seq_len = new_pos;

        // Compute logits from final hidden state
        let hidden =
            final_hidden.ok_or_else(|| ModelError::ConfigError("No tokens to process".into()))?;
        self.compute_logits(&hidden, backend)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn architecture(&self) -> Architecture {
        self.architecture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_config() {
        let config = ModelConfig::llama_7b();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
    }
}
