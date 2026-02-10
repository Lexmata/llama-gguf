//! Model architectures and inference
//!
//! This module provides:
//! - Model configuration types
//! - Architecture definitions
//! - The `Model` trait for inference
//! - LLaMA and other model implementations
//! - Model loading from GGUF files
//! - Speculative decoding

mod architecture;
pub mod cache;
mod config;
pub mod embeddings;
mod error;
pub mod layers;
mod llama;
mod loader;
pub mod lora;
pub mod moe;
pub mod speculative;

pub use architecture::Architecture;
pub use cache::{
    CachedPrefix, PrefixId, PrefixSharing, PromptCache, PromptCacheConfig, PromptCacheStats,
};
pub use config::{ActivationType, ModelConfig, RopeConfig, RopeScalingType, RopeType};
pub use embeddings::{
    EmbeddingConfig, EmbeddingError, EmbeddingExtractor, PoolingStrategy, TruncationStrategy,
    cosine_similarity, dot_product, euclidean_distance, find_nearest,
};
pub use error::{ModelError, ModelResult};
pub use layers::TransformerLayer;
pub use llama::LlamaModel;
pub use loader::{load_llama_model, ModelLoader};
pub use lora::{LoraAdapter, LoraAdapters, LoraConfig};
pub use moe::{MoeConfig, MoeExpert, MoeLayer, MoeRouter, MoeStats};
pub use speculative::{SpeculativeConfig, SpeculativeDecoder, SpeculativeStats};

use std::sync::Arc;

use crate::backend::Backend;
use crate::tensor::Tensor;

/// KV cache for efficient autoregressive generation
#[derive(Debug)]
pub struct KVCache {
    /// Key cache for each layer: [num_kv_heads, max_seq_len, head_dim]
    pub k_cache: Vec<Tensor>,
    /// Value cache for each layer
    pub v_cache: Vec<Tensor>,
    /// Current sequence length in cache
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        use crate::tensor::DType;

        let k_cache: Vec<Tensor> = (0..num_layers)
            .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
            .collect();

        let v_cache: Vec<Tensor> = (0..num_layers)
            .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
            .collect();

        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            num_layers,
        }
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self) {
        self.seq_len = 0;
        // Optionally zero out the cache data
        for k in &mut self.k_cache {
            if let Ok(data) = k.as_f32_mut() {
                data.fill(0.0);
            }
        }
        for v in &mut self.v_cache {
            if let Ok(data) = v.as_f32_mut() {
                data.fill(0.0);
            }
        }
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.seq_len)
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }

    /// Truncate cache to a specific length (for context shifting)
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.seq_len {
            self.seq_len = new_len;
        }
    }

    /// Shift cache left by `amount` positions (for sliding window)
    /// Keeps the last (seq_len - amount) positions
    pub fn shift_left(&mut self, amount: usize) {
        if amount == 0 || amount >= self.seq_len {
            self.reset();
            return;
        }

        let new_len = self.seq_len - amount;

        for layer_idx in 0..self.num_layers {
            // Shift K cache
            if let Ok(k_data) = self.k_cache[layer_idx].as_f32_mut() {
                for head in 0..self.num_kv_heads {
                    for pos in 0..new_len {
                        let src_offset = head * self.max_seq_len * self.head_dim
                            + (pos + amount) * self.head_dim;
                        let dst_offset =
                            head * self.max_seq_len * self.head_dim + pos * self.head_dim;

                        // Copy head_dim elements
                        for d in 0..self.head_dim {
                            k_data[dst_offset + d] = k_data[src_offset + d];
                        }
                    }
                }
            }

            // Shift V cache
            if let Ok(v_data) = self.v_cache[layer_idx].as_f32_mut() {
                for head in 0..self.num_kv_heads {
                    for pos in 0..new_len {
                        let src_offset = head * self.max_seq_len * self.head_dim
                            + (pos + amount) * self.head_dim;
                        let dst_offset =
                            head * self.max_seq_len * self.head_dim + pos * self.head_dim;

                        for d in 0..self.head_dim {
                            v_data[dst_offset + d] = v_data[src_offset + d];
                        }
                    }
                }
            }
        }

        self.seq_len = new_len;
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let tensor_size = self.num_kv_heads * self.max_seq_len * self.head_dim * 4; // f32 = 4 bytes
        tensor_size * 2 * self.num_layers // K and V for each layer
    }
}

/// Context for model inference
pub struct InferenceContext {
    /// KV cache for attention
    pub kv_cache: KVCache,
    /// Backend to use for computation
    pub backend: Arc<dyn Backend>,
    /// Current position in sequence
    pub position: usize,
}

impl InferenceContext {
    /// Create a new inference context
    pub fn new(config: &ModelConfig, backend: Arc<dyn Backend>) -> Self {
        Self {
            kv_cache: KVCache::new(
                config.num_layers,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            ),
            backend,
            position: 0,
        }
    }

    /// Reset context for a new sequence
    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.position = 0;
    }
}

/// Trait for language models
pub trait Model: Send + Sync {
    /// Run forward pass and return logits
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs
    /// * `ctx` - Inference context with KV cache
    ///
    /// # Returns
    /// Logits tensor of shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor>;

    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Get model architecture
    fn architecture(&self) -> Architecture;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    /// Get maximum sequence length
    fn max_seq_len(&self) -> usize {
        self.config().max_seq_len
    }
}
