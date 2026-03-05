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
    /// Number of MoE experts (0 = dense model)
    pub num_experts: usize,
    /// Number of experts activated per token
    pub num_experts_per_token: usize,
    /// Expert FFN intermediate dimension (may differ from dense intermediate_size)
    pub expert_intermediate_size: usize,
    /// Per-head key dimension (defaults to head_dim if not specified)
    pub key_length: usize,
    /// Per-head value dimension (defaults to head_dim if not specified)
    pub value_length: usize,
    /// SSM/DeltaNet inner dimension (0 = no SSM layers)
    pub ssm_d_inner: usize,
    /// SSM state dimension (per-head key dim for delta-net)
    pub ssm_d_state: usize,
    /// SSM group count (number of key heads in delta-net)
    pub ssm_n_group: usize,
    /// SSM time step rank (number of value heads in delta-net)
    pub ssm_dt_rank: usize,
    /// SSM convolution kernel size
    pub ssm_conv_kernel: usize,
    /// Attention logit soft-capping value (Gemma2: 50.0, 0.0 = disabled)
    pub attn_logit_softcap: f32,
    /// Final logit soft-capping value (Gemma2: 30.0, 0.0 = disabled)
    pub final_logit_softcap: f32,
    /// Sliding window attention size (0 = disabled)
    pub sliding_window: usize,
    /// Whether this architecture uses combined QKV tensor
    pub has_combined_qkv: bool,
    /// Whether this architecture uses LayerNorm instead of RMSNorm
    pub uses_layer_norm: bool,
    /// Whether this architecture uses GELU activation
    pub uses_gelu: bool,
    /// Whether this architecture has a gate projection in FFN
    pub has_ffn_gate: bool,
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
            num_experts: 0,
            num_experts_per_token: 0,
            expert_intermediate_size: 0,
            key_length: 128,
            value_length: 128,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            sliding_window: 0,
            has_combined_qkv: false,
            uses_layer_norm: false,
            uses_gelu: false,
            has_ffn_gate: true,
        }
    }
}

impl ModelConfig {
    /// Whether this model has SSM/delta-net recurrent layers
    pub fn has_ssm(&self) -> bool {
        self.ssm_d_inner > 0
    }

    /// Check if this is an MoE model
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

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
            num_experts: 0,
            num_experts_per_token: 0,
            expert_intermediate_size: 0,
            key_length: 128,
            value_length: 128,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            sliding_window: 0,
            has_combined_qkv: false,
            uses_layer_norm: false,
            uses_gelu: false,
            has_ffn_gate: true,
        }
    }

    /// Create config for LLaMA 2 7B
    pub fn llama2_7b() -> Self {
        let mut config = Self::llama_7b();
        config.max_seq_len = 4096;
        config.rope_config.original_max_position_embeddings = 4096;
        config.attn_logit_softcap = 0.0;
        config.final_logit_softcap = 0.0;
        config.sliding_window = 0;
        config.has_combined_qkv = false;
        config.uses_layer_norm = false;
        config.uses_gelu = false;
        config.has_ffn_gate = true;
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
            num_experts: 0,
            num_experts_per_token: 0,
            expert_intermediate_size: 0,
            key_length: 128,
            value_length: 128,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            sliding_window: 0,
            has_combined_qkv: false,
            uses_layer_norm: false,
            uses_gelu: false,
            has_ffn_gate: true,
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
