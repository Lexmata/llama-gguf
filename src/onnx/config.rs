//! HuggingFace config.json parser
//!
//! Parses the `config.json` file that accompanies HuggingFace Optimum ONNX exports
//! into the internal `ModelConfig` type.

use std::path::Path;

use crate::model::{
    ActivationType, Architecture, ModelConfig, RopeConfig, RopeScalingType, RopeType,
};

use super::error::{OnnxError, OnnxResult};

/// Parsed HuggingFace config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HfConfig {
    /// Model type identifier (e.g., "llama", "mistral", "qwen2")
    pub model_type: Option<String>,

    /// Vocabulary size
    pub vocab_size: Option<usize>,

    /// Hidden size / embedding dimension
    pub hidden_size: Option<usize>,

    /// Intermediate (FFN) size
    pub intermediate_size: Option<usize>,

    /// Number of hidden layers
    pub num_hidden_layers: Option<usize>,

    /// Number of attention heads
    pub num_attention_heads: Option<usize>,

    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: Option<usize>,

    /// Maximum position embeddings
    pub max_position_embeddings: Option<usize>,

    /// RMS normalization epsilon
    pub rms_norm_eps: Option<f32>,

    /// RoPE theta (frequency base)
    pub rope_theta: Option<f32>,

    /// RoPE scaling configuration
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Whether to tie word embeddings (input and output)
    pub tie_word_embeddings: Option<bool>,

    /// Hidden activation function
    pub hidden_act: Option<String>,

    /// Whether attention uses bias
    #[serde(default)]
    pub attention_bias: bool,

    /// Whether MLP uses bias
    #[serde(default)]
    pub mlp_bias: bool,

    /// Head dimension (explicit override, rare)
    pub head_dim: Option<usize>,

    /// BOS token ID
    pub bos_token_id: Option<u32>,

    /// EOS token ID
    pub eos_token_id: Option<serde_json::Value>,
}

/// RoPE scaling configuration from config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type (e.g., "linear", "dynamic")
    #[serde(rename = "type")]
    pub scaling_type: Option<String>,

    /// Scaling factor
    pub factor: Option<f32>,
}

impl HfConfig {
    /// Load config from a JSON file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> OnnxResult<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path).map_err(|e| {
            OnnxError::MissingConfig(format!("{}: {}", path.display(), e))
        })?;
        Self::from_json(&data)
    }

    /// Parse config from a JSON string
    pub fn from_json(json: &str) -> OnnxResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| OnnxError::ConfigParse(format!("Failed to parse config.json: {}", e)))
    }

    /// Detect the model architecture from model_type
    pub fn architecture(&self) -> Architecture {
        match self.model_type.as_deref() {
            Some("llama") => Architecture::Llama,
            Some("mistral") => Architecture::Mistral,
            Some("qwen2") => Architecture::Qwen2,
            Some("codellama") => Architecture::CodeLlama,
            Some("yi") => Architecture::Yi,
            Some("deepseek") | Some("deepseek_v2") => Architecture::DeepSeek,
            Some("mixtral") => Architecture::Mixtral,
            _ => Architecture::Unknown,
        }
    }

    /// Convert to internal ModelConfig
    pub fn to_model_config(&self) -> OnnxResult<ModelConfig> {
        let hidden_size = self.hidden_size.ok_or_else(|| {
            OnnxError::ConfigParse("missing hidden_size in config.json".into())
        })?;

        let num_heads = self.num_attention_heads.ok_or_else(|| {
            OnnxError::ConfigParse("missing num_attention_heads in config.json".into())
        })?;

        let num_layers = self.num_hidden_layers.ok_or_else(|| {
            OnnxError::ConfigParse("missing num_hidden_layers in config.json".into())
        })?;

        let num_kv_heads = self.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = self.head_dim.unwrap_or(hidden_size / num_heads);
        let intermediate_size = self.intermediate_size.unwrap_or(hidden_size * 4 * 2 / 3);
        let max_seq_len = self.max_position_embeddings.unwrap_or(2048);
        let norm_eps = self.rms_norm_eps.unwrap_or(1e-5);
        let vocab_size = self.vocab_size.unwrap_or(32000);

        // Determine RoPE type from architecture
        let architecture = self.architecture();
        let rope_type = match architecture {
            Architecture::Qwen2 => RopeType::NeoX,
            _ => RopeType::Normal,
        };

        let freq_base = self.rope_theta.unwrap_or(10000.0);
        let freq_scale = self
            .rope_scaling
            .as_ref()
            .and_then(|s| s.factor)
            .unwrap_or(1.0);

        let rope_config = RopeConfig {
            freq_base,
            freq_scale,
            n_dims: head_dim,
            scaling_type: RopeScalingType::None,
            original_max_position_embeddings: max_seq_len,
            rope_type,
        };

        let hidden_act = match self.hidden_act.as_deref() {
            Some("gelu") | Some("gelu_new") => ActivationType::GELU,
            _ => ActivationType::SiLU,
        };

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            norm_eps,
            rope_config,
            use_parallel_residual: false,
            hidden_act,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        })
    }
}
