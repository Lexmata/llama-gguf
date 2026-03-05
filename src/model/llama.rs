//! LLaMA model architecture implementation
//!
//! This module implements the LLaMA transformer architecture, supporting:
//! - LLaMA 1, 2, 3 variants
//! - Gemma2 (final logit softcapping)
//! - Grouped Query Attention (GQA)
//! - RoPE position embeddings
//! - Quantized weights

use std::sync::Arc;

use crate::backend::Backend;
use crate::tensor::{DType, Tensor};

use super::config::ModelConfig;
use super::deltanet::RecurrentConfig;
use super::error::{ModelError, ModelResult};
use super::layers::{Linear, RMSNorm, TransformerLayer};
use super::deltanet::DeltaNetConfig;
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
    /// Per-layer recurrent flag (true = delta-net or Mamba, false = full attention)
    recurrent_mask: Vec<bool>,
    /// Recurrent config (None if no recurrent layers)
    recurrent_config: Option<RecurrentConfig>,
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
        if layers.len() != config.num_layers {
            return Err(ModelError::ConfigError(format!(
                "Expected {} layers, got {}",
                config.num_layers,
                layers.len()
            )));
        }

        let recurrent_mask: Vec<bool> = layers.iter().map(|l| l.is_recurrent()).collect();
        let has_recurrent = recurrent_mask.iter().any(|&r| r);

        let recurrent_config = if has_recurrent && config.has_ssm() {
            let is_mamba =
                matches!(architecture, Architecture::Mamba | Architecture::Mamba2);
            Some(if is_mamba {
                RecurrentConfig::Mamba(super::mamba::MambaConfig {
                    d_inner: config.ssm_d_inner,
                    d_state: config.ssm_d_state,
                    dt_rank: config.ssm_dt_rank,
                    conv_kernel: config.ssm_conv_kernel.max(1),
                })
            } else {
                let d_inner = config.ssm_d_inner;
                let d_state = config.ssm_d_state;
                let num_v_heads = config.ssm_dt_rank;
                let num_k_heads = config.ssm_n_group.max(1);
                let head_v_dim = d_inner / num_v_heads.max(1);
                let head_k_dim = d_state;
                let conv_kernel = config.ssm_conv_kernel;
                let q_dim = num_k_heads * head_k_dim;
                let k_dim = num_k_heads * head_k_dim;
                let qkv_dim = q_dim + k_dim + d_inner;
                RecurrentConfig::DeltaNet(DeltaNetConfig {
                    d_inner,
                    d_state,
                    num_v_heads,
                    num_k_heads,
                    head_v_dim,
                    head_k_dim,
                    conv_kernel,
                    qkv_dim,
                })
            })
        } else {
            None
        };

        Ok(Self {
            config,
            token_embedding,
            layers,
            norm,
            output,
            architecture,
            recurrent_mask,
            recurrent_config,
        })
    }

    /// Create an InferenceContext appropriate for this model (with recurrent state if needed).
    pub fn create_context(&self, backend: Arc<dyn Backend>) -> InferenceContext {
        if let Some(ref rc) = self.recurrent_config {
            InferenceContext::new_with_recurrent(
                &self.config,
                backend,
                &self.recurrent_mask,
                rc,
            )
        } else {
            InferenceContext::new(&self.config, backend)
        }
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get transformer layers
    pub fn layers(&self) -> &[TransformerLayer] {
        &self.layers
    }

    /// Decompose the model into its parts for GPU hybrid inference.
    /// The layers are moved out to avoid duplicating the weights.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        ModelConfig,
        Tensor,
        Vec<TransformerLayer>,
        RMSNorm,
        Linear,
        Architecture,
        Vec<bool>,
        Option<RecurrentConfig>,
    ) {
        (
            self.config,
            self.token_embedding,
            self.layers,
            self.norm,
            self.output,
            self.architecture,
            self.recurrent_mask,
            self.recurrent_config,
        )
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

    /// Dequantize the embedding table once and return it as a `Cow`.
    ///
    /// For F32 embeddings this borrows the existing data (zero-copy).
    /// For quantized embeddings this dequantizes once into an owned `Vec`.
    fn dequantize_embeddings<'a>(
        &'a self,
        backend: &dyn Backend,
    ) -> ModelResult<std::borrow::Cow<'a, [f32]>> {
        if self.token_embedding.dtype() == DType::F32 {
            Ok(std::borrow::Cow::Borrowed(self.token_embedding.as_f32()?))
        } else {
            let numel = self.token_embedding.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            backend.dequantize(&self.token_embedding, &mut dequant)?;
            Ok(std::borrow::Cow::Owned(dequant.as_f32()?.to_vec()))
        }
    }

    /// Get token embedding for given token IDs (public for testing)
    pub fn embed_tokens(&self, tokens: &[u32], backend: &dyn Backend) -> ModelResult<Tensor> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let seq_len = tokens.len();

        let embedding_data = self.dequantize_embeddings(backend)?;

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

        // Final logit softcapping (Gemma2): logits = cap * tanh(logits / cap)
        if self.config.final_logit_softcap > 0.0 {
            let cap = self.config.final_logit_softcap;
            let data = logits.as_f32_mut()?;
            for v in data.iter_mut() {
                *v = cap * (*v / cap).tanh();
            }
        }

        Ok(logits)
    }
}

impl Model for LlamaModel {
    fn create_context(&self, backend: Arc<dyn Backend>) -> InferenceContext {
        self.create_context(backend)
    }

    /// Forward pass. Supports LLaMA 1/2/3, Gemma2 (final logit softcapping when `final_logit_softcap` > 0).
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor> {
        let backend = ctx.backend.as_ref();
        let num_tokens = tokens.len();

        // Check context length
        let new_pos = ctx.position + num_tokens;
        if new_pos > self.config.max_seq_len {
            return Err(ModelError::ContextLengthExceeded {
                current: new_pos,
                max: self.config.max_seq_len,
            });
        }

        let embedding_data = self.dequantize_embeddings(backend)?;
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // Build per-token hidden states from embeddings
        let mut hiddens: Vec<Tensor> = Vec::with_capacity(num_tokens);
        for &token in tokens {
            let token_idx = token as usize;
            if token_idx >= vocab_size {
                return Err(ModelError::InvalidMetadata {
                    key: "token".into(),
                    message: format!("Token ID {} exceeds vocab size {}", token, vocab_size),
                });
            }
            let src = token_idx * hidden_size;
            hiddens.push(Tensor::from_f32(
                &embedding_data[src..src + hidden_size],
                vec![hidden_size],
            )?);
        }

        // Gemma scales token embeddings by sqrt(hidden_size)
        if self.architecture.is_gemma() {
            let scale = (hidden_size as f32).sqrt();
            for hidden in &mut hiddens {
                let data = hidden.as_f32_mut()?;
                for v in data.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Layer-first ordering: process ALL tokens through each layer before
        // moving to the next. This keeps each layer's weight matrices hot in
        // CPU cache across all tokens, dramatically reducing memory bandwidth
        // during prefill (each ~32MB weight set is read once from RAM instead
        // of once per token).
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (token_offset, hidden) in hiddens.iter_mut().enumerate() {
                let current_pos = ctx.position + token_offset;
                let recurrent_state = ctx
                    .recurrent_state
                    .as_mut()
                    .and_then(|rs| rs.states[layer_idx].as_mut());

                *hidden = layer.forward(
                    hidden,
                    &mut ctx.kv_cache.k_cache[layer_idx],
                    &mut ctx.kv_cache.v_cache[layer_idx],
                    current_pos,
                    self.config.rope_config.freq_base,
                    self.config.rope_config.freq_scale,
                    backend,
                    recurrent_state,
                )?;
            }
        }

        ctx.position = new_pos;
        ctx.kv_cache.seq_len = new_pos;

        // Compute logits from the last token's hidden state
        self.compute_logits(hiddens.last().unwrap(), backend)
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
