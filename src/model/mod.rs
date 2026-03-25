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
mod kv_quantized;
pub mod kv_turboquant;
pub mod deltanet;
pub mod mamba;
pub mod embeddings;
mod error;
pub mod layers;
mod llama;
pub mod bert;
mod loader;
pub mod lora;
pub mod moe;
pub mod paged;
pub mod speculative;
pub mod turboquant;

pub use architecture::Architecture;
pub use kv_quantized::{KVCacheFormat, QuantizedKVCache};
pub use kv_turboquant::TurboQuantKVCache;
pub use turboquant::TurboQuantConfig;
pub use cache::{
    CachedPrefix, PrefixId, PrefixSharing, PromptCache, PromptCacheConfig, PromptCacheStats,
};
pub use config::{ActivationType, ModelConfig, RopeConfig, RopeScalingType, RopeType};
pub use embeddings::{
    EmbeddingConfig, EmbeddingError, EmbeddingExtractor, PoolingStrategy, TruncationStrategy,
    cosine_similarity, dot_product, euclidean_distance, find_nearest,
};
pub use error::{ModelError, ModelResult};
pub use deltanet::{
    DeltaNetConfig, DeltaNetLayer, DeltaNetState, RecurrentConfig, RecurrentLayerState,
    RecurrentState,
};
pub use mamba::{MambaConfig, MambaState, MambaLayer};
pub use bert::{BertLayer, BertModel};
pub use layers::{AttentionLayer, FfnLayer, TransformerLayer};
pub use llama::LlamaModel;
pub use loader::{ModelLoader, load_llama_model};
pub use lora::{LoraAdapter, LoraAdapters, LoraConfig};
pub use moe::{MoeConfig, MoeExpert, MoeLayer, MoeRouter, MoeStats};
pub use paged::{BlockId, BlockTable, PageAllocator, PagedKVPool, PagedSequence, DEFAULT_BLOCK_SIZE};
pub use speculative::{SpeculativeConfig, SpeculativeDecoder, SpeculativeMode, SpeculativeStats};

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

    /// Reset the cache for a new sequence.
    ///
    /// Only resets the position counter. Cache data is not zeroed because
    /// `attention_cached` only reads positions `0..seq_len`, so stale data
    /// beyond `seq_len` is never accessed.
    pub fn reset(&mut self) {
        self.seq_len = 0;
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

    /// Shift cache left by `amount` positions (for sliding window).
    /// Keeps the last `(seq_len - amount)` positions.
    ///
    /// Uses `copy_within` for each head's contiguous run, which compiles to
    /// a single `memmove` — dramatically faster than the element-wise loop
    /// it replaces (especially for long sequences).
    pub fn shift_left(&mut self, amount: usize) {
        if amount == 0 || amount >= self.seq_len {
            self.seq_len = 0;
            return;
        }

        let new_len = self.seq_len - amount;
        let row_stride = self.max_seq_len * self.head_dim;
        let copy_elems = new_len * self.head_dim;

        for layer_idx in 0..self.num_layers {
            if let Ok(k_data) = self.k_cache[layer_idx].as_f32_mut() {
                for head in 0..self.num_kv_heads {
                    let base = head * row_stride;
                    let src_start = base + amount * self.head_dim;
                    k_data.copy_within(src_start..src_start + copy_elems, base);
                }
            }

            if let Ok(v_data) = self.v_cache[layer_idx].as_f32_mut() {
                for head in 0..self.num_kv_heads {
                    let base = head * row_stride;
                    let src_start = base + amount * self.head_dim;
                    v_data.copy_within(src_start..src_start + copy_elems, base);
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

/// Which KV cache implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KVCacheType {
    /// Standard f32 KV cache (default).
    F32,
    /// TurboQuant MSE: Hadamard rotation + scalar quantization (biased, lower overhead).
    TurboQuantMSE { bits: u8 },
    /// TurboQuant prod: MSE + QJL correction (unbiased, higher accuracy).
    TurboQuantProd { bits: u8 },
}

impl Default for KVCacheType {
    fn default() -> Self {
        Self::F32
    }
}

impl KVCacheType {
    /// Convert to a `TurboQuantConfig` if this is a TurboQuant variant.
    pub fn to_tq_config(&self, dim: usize) -> Option<TurboQuantConfig> {
        match *self {
            Self::F32 => None,
            Self::TurboQuantMSE { bits } => Some(TurboQuantConfig {
                bits,
                use_qjl: false,
                dim,
            }),
            Self::TurboQuantProd { bits } => Some(TurboQuantConfig {
                bits,
                use_qjl: true,
                dim,
            }),
        }
    }

    /// Whether this is any TurboQuant variant.
    pub fn is_turboquant(&self) -> bool {
        !matches!(self, Self::F32)
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
    /// Recurrent state for delta-net layers (None if model has no SSM layers)
    pub recurrent_state: Option<RecurrentState>,
    /// Optional TurboQuant-compressed KV cache (replaces f32 cache for attention)
    pub tq_cache: Option<TurboQuantKVCache>,
}

impl InferenceContext {
    /// Create a new inference context
    pub fn new(config: &ModelConfig, backend: Arc<dyn Backend>) -> Self {
        Self {
            kv_cache: KVCache::new(
                config.num_layers,
                config.num_kv_heads,
                config.max_seq_len,
                config.key_length,
            ),
            backend,
            position: 0,
            recurrent_state: None,
            tq_cache: None,
        }
    }

    /// Create inference context with a specific KV cache type.
    pub fn new_with_cache_type(
        config: &ModelConfig,
        backend: Arc<dyn Backend>,
        cache_type: KVCacheType,
    ) -> Self {
        let tq_cache = cache_type
            .to_tq_config(config.key_length)
            .map(|tq_config| {
                TurboQuantKVCache::new(
                    config.num_layers,
                    config.num_kv_heads,
                    config.max_seq_len,
                    config.key_length,
                    tq_config,
                )
            });

        Self {
            kv_cache: KVCache::new(
                config.num_layers,
                config.num_kv_heads,
                config.max_seq_len,
                config.key_length,
            ),
            backend,
            position: 0,
            recurrent_state: None,
            tq_cache,
        }
    }

    /// Create inference context with recurrent state for SSM layers.
    /// `is_recurrent[i]` marks which layers are recurrent (DeltaNet or Mamba).
    pub fn new_with_recurrent(
        config: &ModelConfig,
        backend: Arc<dyn Backend>,
        is_recurrent: &[bool],
        rc: &RecurrentConfig,
    ) -> Self {
        Self {
            kv_cache: KVCache::new(
                config.num_layers,
                config.num_kv_heads,
                config.max_seq_len,
                config.key_length,
            ),
            backend,
            position: 0,
            recurrent_state: Some(RecurrentState::new(
                config.num_layers,
                is_recurrent,
                rc,
            )),
            tq_cache: None,
        }
    }

    /// Reset context for a new sequence
    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.position = 0;
        if let Some(ref mut rs) = self.recurrent_state {
            rs.reset();
        }
        if let Some(ref mut tq) = self.tq_cache {
            tq.reset();
        }
    }

    /// Whether TurboQuant KV cache is active.
    pub fn has_turboquant(&self) -> bool {
        self.tq_cache.is_some()
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

    /// Create an InferenceContext with the right state for this model.
    fn create_context(&self, backend: Arc<dyn Backend>) -> InferenceContext {
        InferenceContext::new(self.config(), backend)
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    /// Get maximum sequence length
    fn max_seq_len(&self) -> usize {
        self.config().max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_type_default() {
        assert_eq!(KVCacheType::default(), KVCacheType::F32);
    }

    #[test]
    fn test_kv_cache_type_is_turboquant() {
        assert!(!KVCacheType::F32.is_turboquant());
        assert!(KVCacheType::TurboQuantMSE { bits: 2 }.is_turboquant());
        assert!(KVCacheType::TurboQuantProd { bits: 3 }.is_turboquant());
    }

    #[test]
    fn test_kv_cache_type_to_tq_config() {
        assert!(KVCacheType::F32.to_tq_config(64).is_none());

        let cfg = KVCacheType::TurboQuantMSE { bits: 2 }
            .to_tq_config(128)
            .unwrap();
        assert_eq!(cfg.bits, 2);
        assert_eq!(cfg.dim, 128);
        assert!(!cfg.use_qjl);

        let cfg = KVCacheType::TurboQuantProd { bits: 3 }
            .to_tq_config(64)
            .unwrap();
        assert_eq!(cfg.bits, 3);
        assert_eq!(cfg.dim, 64);
        assert!(cfg.use_qjl);
    }

    #[test]
    fn test_kv_cache_type_serde_roundtrip() {
        let types = [
            KVCacheType::F32,
            KVCacheType::TurboQuantMSE { bits: 2 },
            KVCacheType::TurboQuantProd { bits: 3 },
        ];
        for ty in &types {
            let json = serde_json::to_string(ty).unwrap();
            let parsed: KVCacheType = serde_json::from_str(&json).unwrap();
            assert_eq!(*ty, parsed);
        }
    }
}
