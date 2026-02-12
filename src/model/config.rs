//! Model configuration types

use serde::{Deserialize, Serialize};

/// RoPE implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RopeType {
    /// Normal/LLaMA style: consecutive pairs (x[2i], x[2i+1])
    #[default]
    Normal,
    /// NeoX/Qwen2 style: first half paired with second half (x[i], x[i+d/2])
    NeoX,
}

/// Configuration for Rotary Position Embeddings (RoPE)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    /// Base frequency for RoPE (typically 10000.0)
    pub freq_base: f32,
    /// Frequency scale factor
    pub freq_scale: f32,
    /// Number of dimensions to apply RoPE to (usually head_dim)
    pub n_dims: usize,
    /// RoPE scaling type
    pub scaling_type: RopeScalingType,
    /// Original context length (for scaled RoPE)
    pub original_max_position_embeddings: usize,
    /// RoPE implementation type (Normal vs NeoX)
    pub rope_type: RopeType,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_dims: 0, // Will be set from head_dim
            scaling_type: RopeScalingType::None,
            original_max_position_embeddings: 2048,
            rope_type: RopeType::Normal,
        }
    }
}

/// RoPE scaling types for extended context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RopeScalingType {
    /// No scaling
    #[default]
    None,
    /// Linear scaling (divide positions by factor)
    Linear,
    /// YaRN (Yet another RoPE extension)
    Yarn,
    /// Dynamic NTK-aware scaling
    DynamicNtk,
}

/// Full model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension (embedding size)
    pub hidden_size: usize,
    /// Intermediate size (FFN dimension, typically 4 * hidden_size or computed)
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMS normalization epsilon
    pub norm_eps: f32,
    /// RoPE configuration
    pub rope_config: RopeConfig,
    /// Whether to use parallel attention (compute QKV in parallel)
    pub use_parallel_residual: bool,
    /// Activation function type
    pub hidden_act: ActivationType,
    /// Whether there's a bias in attention projections
    pub attention_bias: bool,
    /// Whether there's a bias in MLP layers
    pub mlp_bias: bool,
    /// Tie word embeddings with output projection
    pub tie_word_embeddings: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            rope_config: RopeConfig::default(),
            use_parallel_residual: false,
            hidden_act: ActivationType::SiLU,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
        }
    }
}

impl ModelConfig {
    /// Create config for LLaMA 7B
    pub fn llama_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            rope_config: RopeConfig {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_dims: 128,
                scaling_type: RopeScalingType::None,
                original_max_position_embeddings: 2048,
                rope_type: RopeType::Normal,
            },
            use_parallel_residual: false,
            hidden_act: ActivationType::SiLU,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
        }
    }

    /// Create config for LLaMA 2 7B
    pub fn llama2_7b() -> Self {
        let mut config = Self::llama_7b();
        config.max_seq_len = 4096;
        config.rope_config.original_max_position_embeddings = 4096;
        config
    }

    /// Create config for LLaMA 3 8B
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA
            head_dim: 128,
            max_seq_len: 8192,
            norm_eps: 1e-5,
            rope_config: RopeConfig {
                freq_base: 500000.0,
                freq_scale: 1.0,
                n_dims: 128,
                scaling_type: RopeScalingType::None,
                original_max_position_embeddings: 8192,
                rope_type: RopeType::Normal,
            },
            use_parallel_residual: false,
            hidden_act: ActivationType::SiLU,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
        }
    }

    /// Check if this model uses Grouped Query Attention
    pub fn uses_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Get the number of query heads per KV head
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ActivationType {
    /// Gaussian Error Linear Unit
    GELU,
    /// GELU approximation (tanh-based)
    GELUApprox,
    /// Sigmoid Linear Unit (Swish)
    #[default]
    SiLU,
    /// Rectified Linear Unit
    ReLU,
    /// Squared ReLU
    ReLUSquared,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_llama3_gqa() {
        let config = ModelConfig::llama3_8b();
        assert!(config.uses_gqa());
        assert_eq!(config.num_queries_per_kv(), 4);
    }

    #[test]
    fn test_llama_no_gqa() {
        let config = ModelConfig::llama_7b();
        assert!(!config.uses_gqa());
        assert_eq!(config.num_queries_per_kv(), 1);
    }
}
