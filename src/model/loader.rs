//! Model loader for GGUF files
//!
//! This module provides functionality to load model weights from GGUF files
//! and construct model instances.

use std::path::Path;

use crate::gguf::{GgufFile, MetadataValue};
use crate::tensor::{DType, Tensor};

use super::Architecture;
use super::config::{ActivationType, ModelConfig, RopeConfig, RopeScalingType, RopeType};
use super::deltanet::{BetaAlphaProjection, DeltaNetConfig, DeltaNetLayer};
use super::error::{ModelError, ModelResult};
use super::layers::{
    Attention, AttentionLayer, FeedForward, FfnLayer, Linear, RMSNorm, TransformerLayer,
};
use super::llama::LlamaModel;
use super::moe::{MoeConfig, MoeExpert, MoeLayer, MoeRouter};

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
        let get_f32_or =
            |key: &str, default: f32| -> f32 { gguf.data.get_f32(key).unwrap_or(default) };

        // Get core configuration
        // Try multiple methods to determine vocab size
        let vocab_size = get_u32(&format!("{}.vocab_size", arch))
            .or_else(|_| get_u32("tokenizer.ggml.vocab_size"))
            .map(|v| v as usize)
            .unwrap_or_else(|_| {
                // Fallback: get vocab size from tokenizer tokens array length
                if let Some(tokens) = gguf.data.metadata.get("tokenizer.ggml.tokens")
                    && let MetadataValue::Array(arr) = tokens
                {
                    return arr.values.len();
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

        let max_seq_len = get_u32(&format!("{}.context_length", arch)).unwrap_or(2048) as usize;

        let norm_eps = get_f32_or(&format!("{}.attention.layer_norm_rms_epsilon", arch), 1e-5);

        // Parse RoPE configuration
        let freq_base = get_f32_or(&format!("{}.rope.freq_base", arch), 10000.0);
        let freq_scale = get_f32_or(&format!("{}.rope.scale_linear", arch), 1.0);

        // Determine RoPE type based on architecture
        // Qwen2/3 use NeoX style (type 2), most others use Normal style (type 0)
        let rope_type = match architecture {
            Architecture::Qwen2
            | Architecture::Qwen2Moe
            | Architecture::Qwen3
            | Architecture::Qwen35
            | Architecture::Qwen3Moe
            | Architecture::Qwen3Next => RopeType::NeoX,
            _ => RopeType::Normal,
        };

        // MoE configuration
        let num_experts = get_u32(&format!("{}.expert_count", arch)).unwrap_or(0) as usize;
        let num_experts_per_token =
            get_u32(&format!("{}.expert_used_count", arch)).unwrap_or(0) as usize;
        let expert_intermediate_size =
            get_u32(&format!("{}.expert_feed_forward_length", arch)).unwrap_or(0) as usize;

        // Attention head dimensions (may differ from hidden_size / num_heads)
        let key_length =
            get_u32(&format!("{}.attention.key_length", arch)).unwrap_or(head_dim as u32) as usize;
        let value_length = get_u32(&format!("{}.attention.value_length", arch))
            .unwrap_or(head_dim as u32) as usize;

        let rope_n_dims = get_u32(&format!("{}.rope.dimension_count", arch))
            .unwrap_or(head_dim as u32) as usize;

        let rope_config = RopeConfig {
            freq_base,
            freq_scale,
            n_dims: rope_n_dims,
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
            tie_word_embeddings: gguf
                .data
                .get_string("general.tie_word_embeddings")
                .map(|s| s == "true")
                .unwrap_or(false),
            num_experts,
            num_experts_per_token,
            expert_intermediate_size,
            key_length,
            value_length,
            ssm_d_inner: get_u32(&format!("{}.ssm.inner_size", arch)).unwrap_or(0) as usize,
            ssm_d_state: get_u32(&format!("{}.ssm.state_size", arch)).unwrap_or(0) as usize,
            ssm_n_group: get_u32(&format!("{}.ssm.group_count", arch)).unwrap_or(0) as usize,
            ssm_dt_rank: get_u32(&format!("{}.ssm.time_step_rank", arch)).unwrap_or(0) as usize,
            ssm_conv_kernel: get_u32(&format!("{}.ssm.conv_kernel", arch)).unwrap_or(0) as usize,
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

        // Log recurrent layer summary
        let recurrent_count = layers.iter().filter(|l| l.is_recurrent()).count();
        if recurrent_count > 0 {
            tracing::info!(
                "Model has {}/{} DeltaNet recurrent layers",
                recurrent_count,
                layers.len()
            );
        }

        // Load final normalization
        let norm_weight = self.load_tensor("output_norm.weight")?;
        let norm = RMSNorm::new(norm_weight, self.config.norm_eps)?;

        // Load output projection (may be tied to embeddings)
        let output =
            if self.config.tie_word_embeddings || self.try_load_tensor("output.weight").is_none() {
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

    /// Get the DeltaNet config for creating recurrent state.
    /// Returns None if the model has no SSM layers.
    pub fn deltanet_config(&self) -> Option<DeltaNetConfig> {
        if !self.config.has_ssm() {
            return None;
        }
        let d_inner = self.config.ssm_d_inner;
        let d_state = self.config.ssm_d_state;
        let num_v_heads = self.config.ssm_dt_rank;
        let num_k_heads = self.config.ssm_n_group;
        let head_v_dim = d_inner / num_v_heads;
        let head_k_dim = d_state;
        let conv_kernel = self.config.ssm_conv_kernel;
        let q_dim = num_k_heads * head_k_dim;
        let k_dim = num_k_heads * head_k_dim;
        let qkv_dim = q_dim + k_dim + d_inner;

        Some(DeltaNetConfig {
            d_inner,
            d_state,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            conv_kernel,
            qkv_dim,
        })
    }

    /// Load a single transformer layer
    fn load_transformer_layer(&self, layer_idx: usize) -> ModelResult<TransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention normalization
        let attn_norm_weight = self.load_tensor(&format!("{}.attn_norm.weight", prefix))?;
        let attn_norm = RMSNorm::new(attn_norm_weight, self.config.norm_eps)?;

        // Load attention based on available tensors
        let attn_layer = self.load_attention_layer(layer_idx)?;

        // Post-attention normalization (Qwen3Next has this separate from ffn_norm)
        let post_attn_norm =
            if let Some(w) = self.try_load_tensor(&format!("{}.post_attention_norm.weight", prefix))
            {
                Some(RMSNorm::new(w, self.config.norm_eps)?)
            } else {
                None
            };

        // FFN normalization
        let ffn_norm_weight = self
            .try_load_tensor(&format!("{}.ffn_norm.weight", prefix))
            .ok_or_else(|| {
                // Qwen3Next uses post_attention_norm for FFN norm role
                if post_attn_norm.is_some() {
                    // Create a dummy norm — layer forward won't use it in practice
                    // because post_attn_norm already handled normalization
                }
                ModelError::MissingTensor(format!("{}.ffn_norm.weight", prefix))
            });

        let ffn_norm = match ffn_norm_weight {
            Ok(w) => RMSNorm::new(w, self.config.norm_eps)?,
            Err(_) => {
                if post_attn_norm.is_some() {
                    // Models with post_attention_norm reuse it for FFN normalization
                    // Create a unit-weight identity norm as placeholder
                    let hidden = self.config.hidden_size;
                    RMSNorm::new(
                        Tensor::from_f32(&vec![1.0f32; hidden], vec![hidden])?,
                        self.config.norm_eps,
                    )?
                } else {
                    return Err(ModelError::MissingTensor(format!(
                        "{}.ffn_norm.weight",
                        prefix
                    )));
                }
            }
        };

        // Load FFN: MoE or dense
        let ffn_layer = if self.config.is_moe() {
            self.load_moe_layer(layer_idx)?
        } else {
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
            FfnLayer::Dense(FeedForward::new(w_gate, w_up, w_down))
        };

        Ok(TransformerLayer {
            attn_norm,
            attn_layer,
            post_attn_norm,
            ffn_norm,
            ffn_layer,
            layer_idx,
        })
    }

    /// Load attention for a layer: either full softmax or delta-net recurrent.
    fn load_attention_layer(&self, layer_idx: usize) -> ModelResult<AttentionLayer> {
        let prefix = format!("blk.{}", layer_idx);

        if let Some(wq_weight) = self.try_load_tensor(&format!("{}.attn_q.weight", prefix)) {
            // Full softmax attention layer
            let attn = self.load_full_attention(layer_idx, wq_weight)?;
            Ok(AttentionLayer::FullAttention(attn))
        } else if self.config.has_ssm()
            && self
                .try_load_tensor(&format!("{}.attn_qkv.weight", prefix))
                .is_some()
        {
            // DeltaNet recurrent layer
            let dn = self.load_deltanet_layer(layer_idx)?;
            Ok(AttentionLayer::DeltaNet(Box::new(dn)))
        } else {
            Err(ModelError::MissingTensor(format!(
                "{}.attn_q.weight or {}.attn_qkv.weight",
                prefix, prefix
            )))
        }
    }

    /// Load a full softmax attention layer from separate Q/K/V/O tensors.
    fn load_full_attention(
        &self,
        layer_idx: usize,
        wq_weight: Tensor,
    ) -> ModelResult<Attention> {
        let prefix = format!("blk.{}", layer_idx);
        let use_neox_rope = matches!(self.config.rope_config.rope_type, RopeType::NeoX);
        let kl = self.config.key_length;
        let vl = self.config.value_length;
        let rope_dims = self.config.rope_config.n_dims;

        let wq_bias = self.try_load_tensor(&format!("{}.attn_q.bias", prefix));
        let actual_q_out = wq_weight.shape()[1];
        let has_attention_gate = actual_q_out == self.config.num_heads * (kl + vl);

        let wq = Linear::new(wq_weight, wq_bias)?;

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
        let wo_bias = self.try_load_tensor(&format!("{}.attn_output.bias", prefix));
        let wo = Linear::new(
            self.load_tensor(&format!("{}.attn_output.weight", prefix))?,
            wo_bias,
        )?;

        let mut attention = Attention::with_kv_dims(
            wq, wk, wv, wo,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            kl, vl, rope_dims,
            use_neox_rope,
            has_attention_gate,
        );

        if self.architecture.uses_qk_norm()
            && let (Some(q_norm_w), Some(k_norm_w)) = (
                self.try_load_tensor(&format!("{}.attn_q_norm.weight", prefix)),
                self.try_load_tensor(&format!("{}.attn_k_norm.weight", prefix)),
            )
        {
            let q_norm = RMSNorm::new(q_norm_w, self.config.norm_eps)?;
            let k_norm = RMSNorm::new(k_norm_w, self.config.norm_eps)?;
            attention.set_qk_norms(q_norm, k_norm);
        }

        Ok(attention)
    }

    /// Load a DeltaNet (recurrent) layer from SSM tensors.
    fn load_deltanet_layer(&self, layer_idx: usize) -> ModelResult<DeltaNetLayer> {
        let prefix = format!("blk.{}", layer_idx);
        let cfg = &self.config;

        let d_inner = cfg.ssm_d_inner;
        let d_state = cfg.ssm_d_state;
        let num_v_heads = cfg.ssm_dt_rank;
        let num_k_heads = cfg.ssm_n_group;
        let head_v_dim = d_inner / num_v_heads;
        let head_k_dim = d_state;
        let conv_kernel = cfg.ssm_conv_kernel;
        let q_dim = num_k_heads * head_k_dim;
        let k_dim = num_k_heads * head_k_dim;
        let qkv_dim = q_dim + k_dim + d_inner;

        let dn_config = DeltaNetConfig {
            d_inner,
            d_state,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            conv_kernel,
            qkv_dim,
        };

        let attn_qkv = Linear::new(
            self.load_tensor(&format!("{}.attn_qkv.weight", prefix))?,
            None,
        )?;

        let attn_gate = Linear::new(
            self.load_tensor(&format!("{}.attn_gate.weight", prefix))?,
            None,
        )?;

        let ssm_ba = if let Some(ba_weight) =
            self.try_load_tensor(&format!("{}.ssm_ba.weight", prefix))
        {
            BetaAlphaProjection::Combined(Linear::new(ba_weight, None)?)
        } else {
            let beta_w = self.load_tensor(&format!("{}.ssm_beta.weight", prefix))?;
            let alpha_w = self.load_tensor(&format!("{}.ssm_alpha.weight", prefix))?;
            BetaAlphaProjection::Separate {
                beta: Linear::new(beta_w, None)?,
                alpha: Linear::new(alpha_w, None)?,
            }
        };

        let ssm_conv1d_weight = self.load_tensor(&format!("{}.ssm_conv1d.weight", prefix))?;
        let ssm_a = self.load_tensor(&format!("{}.ssm_a", prefix))?;
        let ssm_dt_bias = self.load_tensor(&format!("{}.ssm_dt.bias", prefix))?;

        let ssm_norm_weight = self.load_tensor(&format!("{}.ssm_norm.weight", prefix))?;
        let ssm_norm = RMSNorm::new(ssm_norm_weight, cfg.norm_eps)?;

        let ssm_out = Linear::new(
            self.load_tensor(&format!("{}.ssm_out.weight", prefix))?,
            None,
        )?;

        tracing::info!("Layer {}: loaded DeltaNet (d_inner={}, d_state={}, v_heads={}, k_heads={}, conv={})",
            layer_idx, d_inner, d_state, num_v_heads, num_k_heads, conv_kernel);

        Ok(DeltaNetLayer {
            config: dn_config,
            attn_qkv,
            attn_gate,
            ssm_ba,
            ssm_conv1d_weight,
            ssm_a,
            ssm_dt_bias,
            ssm_norm,
            ssm_out,
        })
    }

    /// Load MoE layer tensors for a given layer index
    fn load_moe_layer(&self, layer_idx: usize) -> ModelResult<FfnLayer> {
        let prefix = format!("blk.{}", layer_idx);
        let num_experts = self.config.num_experts;
        let hidden_dim = self.config.hidden_size;

        // Expert FFN dimension: use expert_intermediate_size if set,
        // otherwise fall back to intermediate_size / num_experts_per_token
        let expert_ffn_dim = if self.config.expert_intermediate_size > 0 {
            self.config.expert_intermediate_size
        } else {
            self.config.intermediate_size / self.config.num_experts_per_token
        };

        // Router/gate weights: [hidden_dim, num_experts]
        let router_weight = self.load_tensor(&format!("{}.ffn_gate_inp.weight", prefix))?;
        let router = MoeRouter::from_weight(
            router_weight,
            self.config.num_experts_per_token,
            false, // Qwen3 MoE uses softmax, not log-softmax normalization
        );

        // Load batched expert weights and split into individual experts
        // GGUF stores these as 3D tensors: [n_expert, ffn_dim, hidden_dim] or similar
        let gate_exps = self.load_tensor(&format!("{}.ffn_gate_exps.weight", prefix))?;
        let up_exps = self.load_tensor(&format!("{}.ffn_up_exps.weight", prefix))?;
        let down_exps = self.load_tensor(&format!("{}.ffn_down_exps.weight", prefix))?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let mut gate_proj = self.extract_expert_tensor(&gate_exps, e)?;
            let mut up_proj = self.extract_expert_tensor(&up_exps, e)?;
            let mut down_proj = self.extract_expert_tensor(&down_exps, e)?;

            gate_proj.set_name(format!("blk.{}.ffn_gate.{}.weight", layer_idx, e));
            up_proj.set_name(format!("blk.{}.ffn_up.{}.weight", layer_idx, e));
            down_proj.set_name(format!("blk.{}.ffn_down.{}.weight", layer_idx, e));

            experts.push(MoeExpert {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        // Load shared experts if present (Qwen3Next)
        let mut shared_experts = Vec::new();
        if let (Some(mut gate_shexp), Some(mut up_shexp), Some(mut down_shexp)) = (
            self.try_load_tensor(&format!("{}.ffn_gate_shexp.weight", prefix)),
            self.try_load_tensor(&format!("{}.ffn_up_shexp.weight", prefix)),
            self.try_load_tensor(&format!("{}.ffn_down_shexp.weight", prefix)),
        ) {
            gate_shexp.set_name(format!("blk.{}.ffn_gate_shexp.0.weight", layer_idx));
            up_shexp.set_name(format!("blk.{}.ffn_up_shexp.0.weight", layer_idx));
            down_shexp.set_name(format!("blk.{}.ffn_down_shexp.0.weight", layer_idx));
            shared_experts.push(MoeExpert {
                gate_proj: gate_shexp,
                up_proj: up_shexp,
                down_proj: down_shexp,
            });
        }

        // Load shared expert gate weight if present (Qwen3Next).
        // This tensor may be BF16 — convert to F32 for inference.
        let shared_expert_gate =
            self.try_load_tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix))
                .map(|t| {
                    if t.dtype() == DType::F32 {
                        t
                    } else {
                        let raw = t.data();
                        let f32_vals: Vec<f32> = match t.dtype() {
                            DType::BF16 => {
                                raw.chunks_exact(2)
                                    .map(|c| {
                                        let bits = u16::from_le_bytes([c[0], c[1]]);
                                        f32::from_bits((bits as u32) << 16)
                                    })
                                    .collect()
                            }
                            _ => {
                                tracing::warn!("Unsupported dtype for shared expert gate, zeroing");
                                vec![0.0f32; t.numel()]
                            }
                        };
                        let shape = t.shape().to_vec();
                        Tensor::from_f32(&f32_vals, shape).unwrap()
                    }
                });
        if shared_expert_gate.is_some() {
            tracing::debug!("Layer {}: loaded shared expert gate", layer_idx);
        }

        let num_shared = shared_experts.len();
        let moe_config = MoeConfig {
            num_experts,
            num_experts_per_token: self.config.num_experts_per_token,
            expert_hidden_dim: expert_ffn_dim,
            num_shared_experts: num_shared,
            aux_loss_coef: 0.0,
            normalize_router_logits: false,
        };

        let mut moe_layer = MoeLayer::new(hidden_dim, moe_config);
        moe_layer.router = router;
        moe_layer.experts = experts;
        moe_layer.shared_experts = shared_experts;
        moe_layer.shared_expert_gate = shared_expert_gate;

        Ok(FfnLayer::Moe(moe_layer))
    }

    /// Extract a single expert's weight tensor from a batched 3D expert tensor.
    ///
    /// GGUF stores batched expert weights as `[ne0, ne1, n_expert]` where the
    /// expert dimension is outermost (slowest). Each expert's 2D weight has
    /// shape `[ne0, ne1]` preserving GGML column-major convention.
    fn extract_expert_tensor(
        &self,
        batched: &Tensor,
        expert_idx: usize,
    ) -> ModelResult<Tensor> {
        let shape = batched.shape();
        if shape.len() != 3 {
            return Err(ModelError::ConfigError(format!(
                "Expected 3D batched expert tensor, got shape {:?}",
                shape
            )));
        }
        let ne0 = shape[0];
        let ne1 = shape[1];
        let num_experts = shape[2];
        let expert_numel = ne0 * ne1;

        if expert_idx >= num_experts {
            return Err(ModelError::ConfigError(format!(
                "Expert index {} out of bounds ({})",
                expert_idx, num_experts
            )));
        }

        let per_expert_shape = vec![ne0, ne1];

        if batched.dtype().is_quantized() {
            let block_size = batched.dtype().block_size();
            let block_bytes = batched.dtype().block_bytes();

            if !expert_numel.is_multiple_of(block_size) {
                return Err(ModelError::ConfigError(format!(
                    "Expert tensor elements ({}) not aligned to block size ({})",
                    expert_numel, block_size
                )));
            }

            let blocks_per_expert = expert_numel / block_size;
            let bytes_per_expert = blocks_per_expert * block_bytes;
            let byte_offset = expert_idx * bytes_per_expert;

            let raw_data = batched.data();
            let expert_bytes = &raw_data[byte_offset..byte_offset + bytes_per_expert];

            let mut tensor =
                Tensor::new(expert_bytes.to_vec(), per_expert_shape, batched.dtype())?;
            tensor.set_name(format!("expert.{}", expert_idx));
            Ok(tensor)
        } else {
            let f32_data = batched.as_f32()?;
            let offset = expert_idx * expert_numel;
            let expert_slice = &f32_data[offset..offset + expert_numel];

            let mut tensor = Tensor::from_f32(expert_slice, per_expert_shape)?;
            tensor.set_name(format!("expert.{}", expert_idx));
            Ok(tensor)
        }
    }

    /// Try to load a tensor from the GGUF file, returning None if not found
    fn try_load_tensor(&self, name: &str) -> Option<Tensor> {
        let tensor_info = self.gguf.data.get_tensor(name)?;
        let tensor_data = self.gguf.tensor_data(name)?;

        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        let dtype = DType::from(tensor_info.dtype);

        Tensor::new(tensor_data.to_vec(), shape, dtype)
            .ok()
            .map(|mut t| {
                t.set_name(name);
                t
            })
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
        let mut tensor = Tensor::new(tensor_data.to_vec(), shape, dtype)?;

        // Store the GGUF tensor name for GPU weight lookup
        tensor.set_name(name);

        Ok(tensor)
    }
}

/// Convenience function to load a LLaMA-like model from a GGUF file
///
/// Supports all LLaMA-compatible architectures including Qwen3 MoE.
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
