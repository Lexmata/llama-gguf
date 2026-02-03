//! Model loader for GGUF files
//!
//! This module provides functionality to load model weights from GGUF files
//! and construct model instances.

use std::path::Path;

use crate::gguf::{GgufFile, MetadataValue};
use crate::tensor::{DType, Tensor};

use super::config::{ActivationType, ModelConfig, RopeConfig, RopeScalingType, RopeType};
use super::error::{ModelError, ModelResult};
use super::layers::{Attention, FeedForward, Linear, RMSNorm, TransformerLayer};
use super::llama::LlamaModel;
use super::Architecture;

/// Model loader for GGUF files
pub struct ModelLoader {
    /// Loaded GGUF file
    gguf: GgufFile,
    /// Detected architecture
    architecture: Architecture,
    /// Parsed model configuration
    config: ModelConfig,
}

impl ModelLoader {
    /// Load a model from a GGUF file path
    pub fn load<P: AsRef<Path>>(path: P) -> ModelResult<Self> {
        let gguf = GgufFile::open(path)?;

        // Detect architecture
        let arch_str = gguf
            .data
            .get_string("general.architecture")
            .ok_or_else(|| ModelError::MissingMetadata("general.architecture".into()))?;

        let architecture = Architecture::from_gguf_str(arch_str);

        if matches!(architecture, Architecture::Unknown) {
            return Err(ModelError::UnsupportedArchitecture(arch_str.to_string()));
        }

        // Parse configuration from metadata
        let config = Self::parse_config(&gguf, &architecture)?;

        Ok(Self {
            gguf,
            architecture,
            config,
        })
    }

    /// Parse model configuration from GGUF metadata
    fn parse_config(gguf: &GgufFile, architecture: &Architecture) -> ModelResult<ModelConfig> {
        let arch = architecture.as_str();

        // Helper to get u32 metadata
        let get_u32 = |key: &str| -> ModelResult<u32> {
            gguf.data
                .get_u32(key)
                .ok_or_else(|| ModelError::MissingMetadata(key.into()))
        };

        // Helper to get f32 metadata with default
        let get_f32_or = |key: &str, default: f32| -> f32 {
            gguf.data.get_f32(key).unwrap_or(default)
        };

        // Get core configuration
        // Try multiple methods to determine vocab size
        let vocab_size = get_u32(&format!("{}.vocab_size", arch))
            .or_else(|_| get_u32("tokenizer.ggml.vocab_size"))
            .map(|v| v as usize)
            .unwrap_or_else(|_| {
                // Fallback: get vocab size from tokenizer tokens array length
                if let Some(tokens) = gguf.data.metadata.get("tokenizer.ggml.tokens") {
                    if let MetadataValue::Array(arr) = tokens {
                        return arr.values.len();
                    }
                }
                // Last resort: infer from embedding tensor shape
                if let Some(emb_info) = gguf.data.get_tensor("token_embd.weight") {
                    // Shape is [hidden_size, vocab_size] in llama.cpp convention
                    if emb_info.dims.len() == 2 {
                        return emb_info.dims[1] as usize;
                    }
                }
                // Default
                32000
            });

        let hidden_size = get_u32(&format!("{}.embedding_length", arch))? as usize;

        let num_layers = get_u32(&format!("{}.block_count", arch))? as usize;

        let num_heads = get_u32(&format!("{}.attention.head_count", arch))? as usize;

        let num_kv_heads = get_u32(&format!("{}.attention.head_count_kv", arch))
            .unwrap_or(num_heads as u32) as usize;

        let head_dim = hidden_size / num_heads;

        let intermediate_size = get_u32(&format!("{}.feed_forward_length", arch))
            .unwrap_or((hidden_size * 4 * 2 / 3) as u32) as usize;

        let max_seq_len = get_u32(&format!("{}.context_length", arch))
            .unwrap_or(2048) as usize;

        let norm_eps = get_f32_or(&format!("{}.attention.layer_norm_rms_epsilon", arch), 1e-5);

        // Parse RoPE configuration
        let freq_base = get_f32_or(&format!("{}.rope.freq_base", arch), 10000.0);
        let freq_scale = get_f32_or(&format!("{}.rope.scale_linear", arch), 1.0);
        
        // Determine RoPE type based on architecture
        // Qwen2 uses NeoX style (type 2), most others use Normal style (type 0)
        let rope_type = match architecture {
            Architecture::Qwen2 => RopeType::NeoX,
            _ => RopeType::Normal,
        };

        let rope_config = RopeConfig {
            freq_base,
            freq_scale,
            n_dims: head_dim,
            scaling_type: RopeScalingType::None,
            original_max_position_embeddings: max_seq_len,
            rope_type,
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
            hidden_act: ActivationType::SiLU,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: gguf.data.get_string("general.tie_word_embeddings")
                .map(|s| s == "true")
                .unwrap_or(false),
        })
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the detected architecture
    pub fn architecture(&self) -> Architecture {
        self.architecture
    }

    /// Build the model from loaded weights
    pub fn build_model(self) -> ModelResult<LlamaModel> {
        // Load token embeddings
        let token_embedding = self.load_tensor("token_embd.weight")?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(self.config.num_layers);
        for i in 0..self.config.num_layers {
            let layer = self.load_transformer_layer(i)?;
            layers.push(layer);
        }

        // Load final normalization
        let norm_weight = self.load_tensor("output_norm.weight")?;
        let norm = RMSNorm::new(norm_weight, self.config.norm_eps)?;

        // Load output projection (may be tied to embeddings)
        // Check if output.weight exists - if not, use weight tying
        let output = if self.config.tie_word_embeddings || self.try_load_tensor("output.weight").is_none() {
            // Use embedding weights as output (weight tying)
            Linear::new(token_embedding.clone(), None)?
        } else {
            let output_weight = self.load_tensor("output.weight")?;
            Linear::new(output_weight, None)?
        };

        LlamaModel::new(
            self.config,
            token_embedding,
            layers,
            norm,
            output,
            self.architecture,
        )
    }

    /// Load a single transformer layer
    fn load_transformer_layer(&self, layer_idx: usize) -> ModelResult<TransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention normalization
        let attn_norm_weight = self.load_tensor(&format!("{}.attn_norm.weight", prefix))?;
        let attn_norm = RMSNorm::new(attn_norm_weight, self.config.norm_eps)?;

        // Attention projections (with optional biases)
        let wq_bias = self.try_load_tensor(&format!("{}.attn_q.bias", prefix));
        let wq = Linear::new(
            self.load_tensor(&format!("{}.attn_q.weight", prefix))?,
            wq_bias,
        )?;
        let wk_bias = self.try_load_tensor(&format!("{}.attn_k.bias", prefix));
        let wk = Linear::new(
            self.load_tensor(&format!("{}.attn_k.weight", prefix))?,
            wk_bias,
        )?;
        let wv_bias = self.try_load_tensor(&format!("{}.attn_v.bias", prefix));
        let wv = Linear::new(
            self.load_tensor(&format!("{}.attn_v.weight", prefix))?,
            wv_bias,
        )?;
        let wo = Linear::new(
            self.load_tensor(&format!("{}.attn_output.weight", prefix))?,
            None,
        )?;

        let use_neox_rope = matches!(self.config.rope_config.rope_type, RopeType::NeoX);
        let attention = Attention::with_rope_type(
            wq,
            wk,
            wv,
            wo,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            use_neox_rope,
        );

        // FFN normalization
        let ffn_norm_weight = self.load_tensor(&format!("{}.ffn_norm.weight", prefix))?;
        let ffn_norm = RMSNorm::new(ffn_norm_weight, self.config.norm_eps)?;

        // FFN projections
        let w_gate = Linear::new(
            self.load_tensor(&format!("{}.ffn_gate.weight", prefix))?,
            None,
        )?;
        let w_up = Linear::new(
            self.load_tensor(&format!("{}.ffn_up.weight", prefix))?,
            None,
        )?;
        let w_down = Linear::new(
            self.load_tensor(&format!("{}.ffn_down.weight", prefix))?,
            None,
        )?;

        let ffn = FeedForward::new(w_gate, w_up, w_down);

        Ok(TransformerLayer {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            layer_idx,
        })
    }

    /// Try to load a tensor from the GGUF file, returning None if not found
    fn try_load_tensor(&self, name: &str) -> Option<Tensor> {
        let tensor_info = self.gguf.data.get_tensor(name)?;
        let tensor_data = self.gguf.tensor_data(name)?;

        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        let dtype = DType::from(tensor_info.dtype);

        Tensor::new(tensor_data.to_vec(), shape, dtype).ok()
    }

    /// Load a tensor from the GGUF file
    fn load_tensor(&self, name: &str) -> ModelResult<Tensor> {
        let tensor_info = self
            .gguf
            .data
            .get_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;

        let tensor_data = self
            .gguf
            .tensor_data(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;

        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        let dtype = DType::from(tensor_info.dtype);

        // Copy the tensor data to owned storage
        // This is necessary because the GGUF file is dropped after build_model() returns
        // and the memory-mapped data would become invalid
        Tensor::new(tensor_data.to_vec(), shape, dtype)
            .map_err(|e| e.into())
    }
}

/// Convenience function to load a LLaMA model from a GGUF file
pub fn load_llama_model<P: AsRef<Path>>(path: P) -> ModelResult<LlamaModel> {
    let loader = ModelLoader::load(path)?;

    if !loader.architecture().is_llama_like() {
        return Err(ModelError::UnsupportedArchitecture(
            loader.architecture().to_string(),
        ));
    }

    loader.build_model()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection() {
        assert!(Architecture::Llama.is_llama_like());
        assert!(Architecture::Mistral.is_llama_like());
        assert!(!Architecture::GPT2.is_llama_like());
    }
}
