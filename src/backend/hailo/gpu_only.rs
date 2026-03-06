//! Hailo GPU-only inference engine with hybrid CPU+Hailo forward pass.

use bytemuck;
use tracing;

use crate::backend::{Backend, BackendError, BackendResult, GpuInference};
use crate::model::layers::{AttentionLayer, FfnLayer, NormLayer};
use crate::model::{Architecture, ModelConfig};
use crate::tensor::{DType, Tensor};

use super::config::HailoConfig;
use super::context::{HailoContext, HefHandle};

fn resolve_hef_path(config: &HailoConfig, layer_idx: usize, kind: &str) -> std::path::PathBuf {
    let dir = config.hef_dir.as_ref().unwrap_or(&config.cache_dir);
    dir.join(format!("layer_{:02}_{}.hef", layer_idx, kind))
}

pub struct HailoGpuInference {
    ctx: HailoContext,
    config: ModelConfig,
    architecture: Architecture,

    layer_attn_hefs: Vec<HefHandle>,
    layer_ffn_hefs: Vec<HefHandle>,

    embeddings: Vec<f32>,
    output_weight: Tensor,
    output_bias: Option<Tensor>,
    output_norm: NormLayer,
    o_proj_weights: Vec<Tensor>,
    o_proj_biases: Vec<Option<Tensor>>,

    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,

    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    position: usize,

    rope_freq_base: f32,
    rope_freq_scale: f32,
    use_neox_rope: bool,
    attn_scale: f32,

    hidden: Vec<f32>,
    qkv_buf: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_out: Vec<f32>,

    cpu_backend: crate::backend::cpu::CpuBackend,
}

impl HailoGpuInference {
    pub fn from_model(
        model: crate::model::LlamaModel,
        max_seq_len: usize,
        hailo_config: HailoConfig,
    ) -> BackendResult<Self> {
        let (
            config,
            token_embedding,
            layers,
            norm,
            output,
            architecture,
            recurrent_mask,
            recurrent_config,
        ) = model.into_parts();

        if recurrent_config.is_some() || recurrent_mask.iter().any(|&r| r) {
            return Err(BackendError::Unsupported(
                "Hailo backend does not support DeltaNet or Mamba layers".into(),
            ));
        }

        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;

        let ctx = HailoContext::new()?;

        let embeddings = if token_embedding.dtype() == DType::F32 {
            token_embedding.as_f32()?.to_vec()
        } else {
            let mut dequant = Tensor::zeros(vec![token_embedding.numel()], DType::F32);
            crate::backend::cpu::ops::dequantize(&token_embedding, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        let mut o_proj_weights = Vec::with_capacity(num_layers);
        let mut o_proj_biases = Vec::with_capacity(num_layers);
        let mut layer_attn_hefs = Vec::with_capacity(num_layers);
        let mut layer_ffn_hefs = Vec::with_capacity(num_layers);

        let mut attn_scale = 1.0 / (head_dim as f32).sqrt();

        for (layer_idx, layer) in layers.into_iter().enumerate() {
            let attn = match &layer.attn_layer {
                AttentionLayer::FullAttention(a) => a,
                AttentionLayer::DeltaNet(_) | AttentionLayer::Mamba(_) => {
                    return Err(BackendError::Unsupported(
                        "Hailo backend requires FullAttention layers".into(),
                    ))
                }
            };
            attn_scale = attn.scale;

            match &layer.ffn_layer {
                FfnLayer::Dense(_) | FfnLayer::NoGate(_) => {}
                FfnLayer::Moe(_) | FfnLayer::Identity => {
                    return Err(BackendError::Unsupported(
                        "Hailo backend does not support MoE or Identity FFN layers".into(),
                    ))
                }
            }

            o_proj_weights.push(attn.wo.weight.clone());
            o_proj_biases.push(attn.wo.bias.clone());

            let attn_path = resolve_hef_path(&hailo_config, layer_idx, "attn");
            let ffn_path = resolve_hef_path(&hailo_config, layer_idx, "ffn");

            if !attn_path.exists() {
                return Err(BackendError::OperationFailed(format!(
                    "HEF not found: {}. Run compiler to generate HEF files.",
                    attn_path.display()
                )));
            }
            if !ffn_path.exists() {
                return Err(BackendError::OperationFailed(format!(
                    "HEF not found: {}. Run compiler to generate HEF files.",
                    ffn_path.display()
                )));
            }

            tracing::info!("Loading HEF layer {} attn from {:?}", layer_idx, attn_path);
            let attn_hef = ctx.load_hef(attn_path.as_path())?;
            tracing::info!("Loading HEF layer {} ffn from {:?}", layer_idx, ffn_path);
            let ffn_hef = ctx.load_hef(ffn_path.as_path())?;

            layer_attn_hefs.push(attn_hef);
            layer_ffn_hefs.push(ffn_hef);
        }

        let k_cache: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0; num_kv_heads * max_seq_len * head_dim])
            .collect();
        let v_cache: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0; num_kv_heads * max_seq_len * head_dim])
            .collect();

        let rope_freq_base = config.rope_config.freq_base;
        let rope_freq_scale = config.rope_config.freq_scale;
        let use_neox_rope = matches!(
            config.rope_config.rope_type,
            crate::model::RopeType::NeoX
        );

        let qkv_size = (num_heads + 2 * num_kv_heads) * head_dim;

        Ok(Self {
            ctx,
            config: config.clone(),
            architecture,
            layer_attn_hefs,
            layer_ffn_hefs,
            embeddings,
            output_weight: output.weight,
            output_bias: output.bias,
            output_norm: norm,
            o_proj_weights,
            o_proj_biases,
            k_cache,
            v_cache,
            num_layers,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            position: 0,
            rope_freq_base,
            rope_freq_scale,
            use_neox_rope,
            attn_scale,
            hidden: vec![0.0; hidden_size],
            qkv_buf: vec![0.0; qkv_size],
            attn_out: vec![0.0; hidden_size],
            ffn_out: vec![0.0; hidden_size],
            cpu_backend: crate::backend::cpu::CpuBackend::new(),
        })
    }

    fn embed_token(&mut self, token_id: u32) -> BackendResult<()> {
        let token_idx = token_id as usize;
        if token_idx >= self.config.vocab_size {
            return Err(BackendError::InvalidArgument(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, self.config.vocab_size
            )));
        }
        let src_start = token_idx * self.hidden_size;
        self.hidden
            .copy_from_slice(&self.embeddings[src_start..src_start + self.hidden_size]);

        if self.architecture.is_gemma() {
            let scale = (self.hidden_size as f32).sqrt();
            for v in self.hidden.iter_mut() {
                *v *= scale;
            }
        }
        Ok(())
    }

    fn run_layers(&mut self) -> BackendResult<()> {
        let pos = self.position;
        let num_queries_per_kv = self.num_heads / self.num_kv_heads;
        let k_head_stride = self.max_seq_len * self.head_dim;
        let v_head_stride = self.max_seq_len * self.head_dim;

        for layer_idx in 0..self.num_layers {
            let attn_hef = &self.layer_attn_hefs[layer_idx];
            let ffn_hef = &self.layer_ffn_hefs[layer_idx];

            let bytes = bytemuck::cast_slice::<f32, u8>(&self.hidden);
            attn_hef.write_input(bytes)?;

            let out_bytes = bytemuck::cast_slice_mut::<f32, u8>(&mut self.qkv_buf);
            attn_hef.read_output(out_bytes)?;

            let q_size = self.num_heads * self.head_dim;
            let k_size = self.num_kv_heads * self.head_dim;
            let _v_size = self.num_kv_heads * self.head_dim;

            let (q_part, rest) = self.qkv_buf.split_at_mut(q_size);
            let (k_part, v_part) = rest.split_at_mut(k_size);

            let mut q_t = Tensor::from_f32(q_part, vec![self.num_heads, 1, self.head_dim])?;
            let mut k_t = Tensor::from_f32(k_part, vec![self.num_kv_heads, 1, self.head_dim])?;

            self.cpu_backend.rope(
                &mut q_t,
                &mut k_t,
                pos,
                self.rope_freq_base,
                self.rope_freq_scale,
                self.use_neox_rope,
            )?;

            let k_data = k_t.as_f32()?;
            let v_data = v_part;
            let k_cache = &mut self.k_cache[layer_idx];
            let v_cache = &mut self.v_cache[layer_idx];

            for h in 0..self.num_kv_heads {
                let cache_off = h * k_head_stride + pos * self.head_dim;
                k_cache[cache_off..cache_off + self.head_dim]
                    .copy_from_slice(&k_data[h * self.head_dim..(h + 1) * self.head_dim]);
            }
            for h in 0..self.num_kv_heads {
                let cache_off = h * v_head_stride + pos * self.head_dim;
                v_cache[cache_off..cache_off + self.head_dim]
                    .copy_from_slice(&v_data[h * self.head_dim..(h + 1) * self.head_dim]);
            }

            let q_data = q_t.as_f32()?;
            let kv_len = pos + 1;

            self.attn_out.fill(0.0);
            for head in 0..self.num_heads {
                let kv_head = head / num_queries_per_kv;
                let q_offset = head * self.head_dim;
                let q_vec = &q_data[q_offset..q_offset + self.head_dim];

                let mut scores = vec![0.0f32; kv_len];
                let k_base = kv_head * k_head_stride;
                let v_base = kv_head * v_head_stride;

                for (kv_pos, score) in scores.iter_mut().enumerate() {
                    if kv_pos > pos {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_offset = k_base + kv_pos * self.head_dim;
                    let k_vec = &k_cache[k_offset..k_offset + self.head_dim];
                    *score = crate::backend::cpu::simd::dot_f32(q_vec, k_vec) * self.attn_scale;
                }

                crate::backend::cpu::simd::softmax_inplace(&mut scores);

                let out_offset = head * self.head_dim;
                let out_vec = &mut self.attn_out[out_offset..out_offset + self.head_dim];
                for (kv_pos, &score_val) in scores.iter().enumerate() {
                    if score_val > 1e-8 {
                        let v_offset = v_base + kv_pos * self.head_dim;
                        let v_vec = &v_cache[v_offset..v_offset + self.head_dim];
                        crate::backend::cpu::simd::axpy_f32(score_val, v_vec, out_vec);
                    }
                }
            }

            let attn_t = Tensor::from_f32(&self.attn_out, vec![self.hidden_size])?;
            let o_weight = &self.o_proj_weights[layer_idx];
            let mut o_out = Tensor::zeros(vec![self.hidden_size], DType::F32);

            if o_weight.dtype().is_quantized() {
                self.cpu_backend
                    .vec_mat_q(&attn_t, o_weight, &mut o_out)?;
            } else {
                self.cpu_backend.vec_mat(&attn_t, o_weight, &mut o_out)?;
            }

            if let Some(ref bias) = self.o_proj_biases[layer_idx] {
                let out_data = o_out.as_f32_mut()?;
                let bias_data = bias.as_f32()?;
                for (o, &b) in out_data.iter_mut().zip(bias_data.iter()) {
                    *o += b;
                }
            }

            let o_data = o_out.as_f32()?;
            for i in 0..self.hidden_size {
                self.hidden[i] += o_data[i];
            }

            let bytes = bytemuck::cast_slice::<f32, u8>(&self.hidden);
            ffn_hef.write_input(bytes)?;

            let out_bytes = bytemuck::cast_slice_mut::<f32, u8>(&mut self.ffn_out);
            ffn_hef.read_output(out_bytes)?;

            for i in 0..self.hidden_size {
                self.hidden[i] += self.ffn_out[i];
            }
        }

        Ok(())
    }

    fn final_norm_and_logits(&mut self) -> BackendResult<Vec<f32>> {
        let hidden_t = Tensor::from_f32(&self.hidden, vec![self.hidden_size])?;
        let mut normed = Tensor::zeros(vec![self.hidden_size], DType::F32);
        self.output_norm
            .forward(&hidden_t, &mut normed, &self.cpu_backend)?;

        let mut logits = Tensor::zeros(vec![self.config.vocab_size], DType::F32);
        if self.output_weight.dtype().is_quantized() {
            self.cpu_backend
                .vec_mat_q(&normed, &self.output_weight, &mut logits)?;
        } else {
            self.cpu_backend
                .vec_mat(&normed, &self.output_weight, &mut logits)?;
        }

        if let Some(ref bias) = self.output_bias {
            let out_data = logits.as_f32_mut()?;
            let bias_data = bias.as_f32()?;
            for (o, &b) in out_data.iter_mut().zip(bias_data.iter()) {
                *o += b;
            }
        }

        let mut logits_data = logits.as_f32()?.to_vec();

        if self.config.final_logit_softcap > 0.0 {
            let cap = self.config.final_logit_softcap;
            for v in logits_data.iter_mut() {
                *v = cap * (*v / cap).tanh();
            }
        }

        Ok(logits_data)
    }
}

impl GpuInference for HailoGpuInference {
    fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> {
        self.embed_token(token_id)?;
        self.run_layers()?;
        let logits = self.final_norm_and_logits()?;
        self.position += 1;
        Ok(logits)
    }

    fn prefill_token(&mut self, token_id: u32) -> BackendResult<()> {
        self.embed_token(token_id)?;
        self.run_layers()?;
        self.position += 1;
        Ok(())
    }

    fn reset(&mut self) {
        for k in &mut self.k_cache {
            k.fill(0.0);
        }
        for v in &mut self.v_cache {
            v.fill(0.0);
        }
        self.position = 0;
    }

    fn position(&self) -> usize {
        self.position
    }
}

unsafe impl Send for HailoGpuInference {}
