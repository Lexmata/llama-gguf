//! GPU-resident inference engine for Metal (Apple Silicon / macOS).
//!
//! All model weights are uploaded to GPU at initialization and all layer
//! computations run through Metal compute shaders. Only the final logits
//! are read back to CPU. Apple Silicon's unified memory architecture makes
//! this especially efficient -- there are no physical copies between CPU
//! and GPU memory.
//!
//! Supported layer types:
//! - Dense attention (standard multi-head, GQA)
//! - Dense FFN (SwiGLU)
//! - MoE (router on GPU, expert weights streamed, SwiGLU on GPU)

use std::collections::HashMap;

use metal::{Buffer, MTLSize};

use crate::backend::cpu::simd;
use crate::backend::{BackendError, BackendResult};
use crate::model::layers::{AttentionLayer, FfnLayer, TransformerLayer};
use crate::model::LlamaModel;
use crate::tensor::{DType, Tensor};

use super::context::MetalContext;

// ---------------------------------------------------------------------------
// Weight store
// ---------------------------------------------------------------------------

struct WeightEntry {
    buffer: Buffer,
    shape: Vec<usize>,
}

struct QuantEntry {
    buffer: Buffer,
    dtype: DType,
    shape: Vec<usize>,
}

struct MetalWeightStore {
    f32_weights: HashMap<String, WeightEntry>,
    quant_weights: HashMap<String, QuantEntry>,
    total_bytes: usize,
}

impl MetalWeightStore {
    fn new() -> Self {
        Self {
            f32_weights: HashMap::new(),
            quant_weights: HashMap::new(),
            total_bytes: 0,
        }
    }

    fn upload_f32(
        &mut self,
        ctx: &MetalContext,
        name: &str,
        tensor: &Tensor,
    ) -> BackendResult<()> {
        let numel = tensor.numel();
        let shape = tensor.shape().to_vec();
        let f32_data: Vec<f32> = if tensor.dtype() == DType::F32 {
            tensor.as_f32()?.to_vec()
        } else {
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            crate::backend::cpu::ops::dequantize(tensor, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        let buffer = ctx.create_buffer_with_data(&f32_data)?;
        self.total_bytes += numel * 4;
        self.f32_weights
            .insert(name.to_string(), WeightEntry { buffer, shape });
        Ok(())
    }

    fn upload_quantized(
        &mut self,
        ctx: &MetalContext,
        name: &str,
        tensor: &Tensor,
    ) -> BackendResult<()> {
        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();
        let raw_bytes = tensor.data();
        let buffer = ctx.create_buffer_with_raw_bytes(raw_bytes)?;
        self.total_bytes += raw_bytes.len();
        self.quant_weights
            .insert(name.to_string(), QuantEntry { buffer, dtype, shape });
        Ok(())
    }

    fn upload_auto(
        &mut self,
        ctx: &MetalContext,
        name: &str,
        tensor: &Tensor,
    ) -> BackendResult<()> {
        let dt = tensor.dtype();
        let is_2d = tensor.shape().len() >= 2;
        if is_2d && dt.is_quantized() && matches!(dt, DType::Q8_0 | DType::Q4K | DType::Q6K) {
            self.upload_quantized(ctx, name, tensor)
        } else {
            self.upload_f32(ctx, name, tensor)
        }
    }

    fn get_f32(&self, name: &str) -> Option<&WeightEntry> {
        self.f32_weights.get(name)
    }

    fn get_quant(&self, name: &str) -> Option<&QuantEntry> {
        self.quant_weights.get(name)
    }

    fn contains(&self, name: &str) -> bool {
        self.f32_weights.contains_key(name) || self.quant_weights.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct InferenceConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    vocab_size: usize,
    norm_eps: f32,
    freq_base: f32,
    freq_scale: f32,
    expert_intermediate: usize,
}

// ---------------------------------------------------------------------------
// Main engine
// ---------------------------------------------------------------------------

pub struct MetalGpuInference {
    ctx: MetalContext,
    weights: MetalWeightStore,
    config: InferenceConfig,
    pos: usize,
    // Persistent scratch buffers
    hidden: Buffer,
    hidden_norm: Buffer,
    residual: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    ffn_down: Buffer,
    logits: Buffer,
    // Dequantization scratch
    dequant_scratch: Buffer,
    dequant_scratch_floats: usize,
    // Attention scratch
    attn_q: Buffer,
    attn_k: Buffer,
    attn_v: Buffer,
    attn_out: Buffer,
    // KV caches [num_kv_heads * max_seq_len * head_dim]
    k_caches: Vec<Option<Buffer>>,
    v_caches: Vec<Option<Buffer>>,
    max_seq_len: usize,
    // CPU embeddings table
    cpu_embeddings: Vec<f32>,
    // Owned layers
    layers: Vec<TransformerLayer>,
    // MoE scratch
    moe_hidden: Buffer,
    moe_expert_gate: Buffer,
    moe_expert_up: Buffer,
    moe_expert_down: Buffer,
    moe_expert_out: Buffer,
    moe_temp: Buffer,
    // Per-layer flags
    has_attention: Vec<bool>,
    // RMS norm partial sums
    rms_partial_buf: Buffer,
    rms_max_workgroups: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytes_of_f32(n: usize) -> u64 {
    (n * std::mem::size_of::<f32>()) as u64
}

impl MetalGpuInference {
    pub fn from_model(model: LlamaModel, max_seq_len: usize) -> BackendResult<Self> {
        let (
            model_config,
            token_embedding,
            layers,
            norm,
            output,
            _architecture,
            _recurrent_mask,
            _deltanet_config,
        ) = model.into_parts();

        eprintln!("Initializing Metal GPU-only inference...");

        let ctx = MetalContext::new(0)?;
        let mut weights = MetalWeightStore::new();

        for (i, layer) in layers.iter().enumerate() {
            if i % 4 == 0 {
                eprintln!("  Uploading layer {}/{}", i + 1, layers.len());
            }
            upload_layer_weights(&ctx, &mut weights, i, layer)?;
        }

        weights.upload_f32(&ctx, "output_norm.weight", &norm.weight)?;
        weights.upload_auto(&ctx, "output.weight", &output.weight)?;
        if let Some(ref bias) = output.bias {
            weights.upload_f32(&ctx, "output.bias", bias)?;
        }

        let has_attention: Vec<bool> = layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                l.attention().is_some()
                    && weights.contains(&format!("blk.{}.attn_q.weight", i))
            })
            .collect();

        let expert_intermediate = layers
            .iter()
            .find_map(|l| l.moe())
            .and_then(|m| m.experts.first().map(|e| e.gate_proj.shape()[1]))
            .unwrap_or(model_config.intermediate_size);

        let config = InferenceConfig {
            hidden_size: model_config.hidden_size,
            intermediate_size: model_config.intermediate_size,
            num_heads: model_config.num_heads,
            num_kv_heads: model_config.num_kv_heads,
            num_layers: model_config.num_layers,
            vocab_size: model_config.vocab_size,
            norm_eps: model_config.norm_eps,
            freq_base: model_config.rope_config.freq_base,
            freq_scale: model_config.rope_config.freq_scale,
            expert_intermediate,
        };

        let cpu_embeddings = if token_embedding.dtype() == DType::F32 {
            token_embedding.as_f32()?.to_vec()
        } else {
            let numel = token_embedding.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            crate::backend::cpu::ops::dequantize(&token_embedding, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        let mut max_quant_elements = 0usize;
        for w in weights.quant_weights.values() {
            let numel: usize = w.shape.iter().product();
            max_quant_elements = max_quant_elements.max(numel);
        }
        let dequant_scratch_floats = max_quant_elements.max(1);

        // Allocate persistent buffers
        let hidden = ctx.create_output_buffer(config.hidden_size)?;
        let hidden_norm = ctx.create_output_buffer(config.hidden_size)?;
        let residual = ctx.create_output_buffer(config.hidden_size)?;
        let ffn_gate = ctx.create_output_buffer(config.intermediate_size)?;
        let ffn_up = ctx.create_output_buffer(config.intermediate_size)?;
        let ffn_down = ctx.create_output_buffer(config.hidden_size)?;
        let logits = ctx.create_output_buffer(config.vocab_size)?;
        let dequant_scratch = ctx.create_output_buffer(dequant_scratch_floats)?;

        let max_q_out = layers.iter().filter_map(|l| l.attention())
            .map(|a| a.wq.out_features).max().unwrap_or(config.hidden_size);
        let max_kv_flat = layers.iter().filter_map(|l| l.attention())
            .map(|a| config.num_kv_heads * a.key_length.max(a.value_length))
            .max().unwrap_or(config.hidden_size);
        let max_attn_out = layers.iter().filter_map(|l| l.attention())
            .map(|a| config.num_heads * a.value_length)
            .max().unwrap_or(config.hidden_size);

        let attn_q = ctx.create_output_buffer(max_q_out.max(1))?;
        let attn_k = ctx.create_output_buffer(max_kv_flat.max(1))?;
        let attn_v = ctx.create_output_buffer(max_kv_flat.max(1))?;
        let attn_out = ctx.create_output_buffer(max_attn_out.max(1))?;

        // KV caches
        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            if has_attention[i] {
                if let Some(attn) = layers[i].attention() {
                    let kl = attn.key_length;
                    let vl = attn.value_length;
                    k_caches.push(Some(ctx.create_output_buffer(config.num_kv_heads * max_seq_len * kl)?));
                    v_caches.push(Some(ctx.create_output_buffer(config.num_kv_heads * max_seq_len * vl)?));
                } else {
                    k_caches.push(None);
                    v_caches.push(None);
                }
            } else {
                k_caches.push(None);
                v_caches.push(None);
            }
        }

        // MoE scratch
        let moe_hidden = ctx.create_output_buffer(config.hidden_size)?;
        let moe_expert_gate = ctx.create_output_buffer(expert_intermediate)?;
        let moe_expert_up = ctx.create_output_buffer(expert_intermediate)?;
        let moe_expert_down = ctx.create_output_buffer(config.hidden_size)?;
        let moe_expert_out = ctx.create_output_buffer(config.hidden_size)?;
        let moe_temp = ctx.create_output_buffer(config.hidden_size)?;

        // RMS norm partial sums buffer
        let rms_max_workgroups = ((config.hidden_size + 255) / 256).max(1);
        let rms_partial_buf = ctx.create_output_buffer(rms_max_workgroups)?;

        let attn_count = has_attention.iter().filter(|&&x| x).count();
        let moe_count = layers.iter().filter(|l| l.moe().is_some()).count();
        let vram_mb = weights.total_bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "Metal GPU-only inference ready: {:.1} MB, {} attn + {} MoE layers, {:.1} MB dequant scratch",
            vram_mb, attn_count, moe_count,
            dequant_scratch_floats as f64 * 4.0 / (1024.0 * 1024.0),
        );

        Ok(Self {
            ctx,
            weights,
            config,
            pos: 0,
            hidden,
            hidden_norm,
            residual,
            ffn_gate,
            ffn_up,
            ffn_down,
            logits,
            dequant_scratch,
            dequant_scratch_floats,
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            k_caches,
            v_caches,
            max_seq_len,
            cpu_embeddings,
            layers,
            moe_hidden,
            moe_expert_gate,
            moe_expert_up,
            moe_expert_down,
            moe_expert_out,
            moe_temp,
            has_attention,
            rms_partial_buf,
            rms_max_workgroups,
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    pub fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> {
        self.embed_token(token_id)?;
        self.ctx.copy_buffer(&self.hidden, &self.residual, bytes_of_f32(self.config.hidden_size))?;

        for layer_idx in 0..self.config.num_layers {
            self.process_layer(layer_idx)?;
        }

        self.rms_norm_gpu("output_norm.weight", BufRef::Hidden, BufRef::HiddenNorm)?;
        self.linear_gpu("output.weight", None, BufRef::HiddenNorm, BufRef::Logits)?;

        let logits = self.ctx.read_buffer(&self.logits, self.config.vocab_size);
        self.pos += 1;
        Ok(logits)
    }

    pub fn forward_batch(&mut self, token_ids: &[u32]) -> BackendResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(BackendError::InvalidArgument("Empty token batch".to_string()));
        }
        for &tid in &token_ids[..token_ids.len() - 1] {
            self.prefill_token(tid)?;
        }
        self.forward(*token_ids.last().unwrap())
    }

    pub fn prefill_token(&mut self, token_id: u32) -> BackendResult<()> {
        self.embed_token(token_id)?;
        self.ctx.copy_buffer(&self.hidden, &self.residual, bytes_of_f32(self.config.hidden_size))?;
        for layer_idx in 0..self.config.num_layers {
            self.process_layer(layer_idx)?;
        }
        self.pos += 1;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        let zeros_h = vec![0.0f32; self.config.hidden_size];
        let _ = self.ctx.write_to_buffer(&self.hidden, &zeros_h);
        let _ = self.ctx.write_to_buffer(&self.residual, &zeros_h);
        for kc in self.k_caches.iter() {
            if let Some(buf) = kc {
                let n = buf.length() as usize / 4;
                let zeros = vec![0.0f32; n];
                let _ = self.ctx.write_to_buffer(buf, &zeros);
            }
        }
        for vc in self.v_caches.iter() {
            if let Some(buf) = vc {
                let n = buf.length() as usize / 4;
                let zeros = vec![0.0f32; n];
                let _ = self.ctx.write_to_buffer(buf, &zeros);
            }
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    fn embed_token(&mut self, token_id: u32) -> BackendResult<()> {
        let hs = self.config.hidden_size;
        let offset = token_id as usize * hs;
        self.ctx.write_to_buffer(&self.hidden, &self.cpu_embeddings[offset..offset + hs])
    }

    // -----------------------------------------------------------------------
    // Layer processing
    // -----------------------------------------------------------------------

    fn process_layer(&mut self, layer_idx: usize) -> BackendResult<()> {
        let hs = self.config.hidden_size;

        self.rms_norm_gpu(
            &format!("blk.{}.attn_norm.weight", layer_idx),
            BufRef::Hidden,
            BufRef::HiddenNorm,
        )?;

        if self.has_attention[layer_idx] {
            self.attention_forward(layer_idx)?;
        } else {
            return Err(BackendError::OperationFailed(format!(
                "Layer {} has no GPU attention path (DeltaNet not supported in Metal gpu_only)",
                layer_idx
            )));
        }

        self.dispatch_add(&self.residual, &self.hidden_norm, &self.hidden)?;
        self.ctx.copy_buffer(&self.hidden, &self.residual, bytes_of_f32(hs))?;

        let ffn_norm_name = if self.layers[layer_idx].post_attn_norm.is_some() {
            format!("blk.{}.post_attention_norm.weight", layer_idx)
        } else {
            format!("blk.{}.ffn_norm.weight", layer_idx)
        };
        self.rms_norm_gpu(&ffn_norm_name, BufRef::Hidden, BufRef::HiddenNorm)?;

        match &self.layers[layer_idx].ffn_layer {
            FfnLayer::Dense(_) => self.dense_ffn_forward(layer_idx)?,
            FfnLayer::Moe(_) => self.moe_forward(layer_idx)?,
        }

        self.dispatch_add(&self.residual, &self.hidden_norm, &self.hidden)?;
        self.ctx.copy_buffer(&self.hidden, &self.residual, bytes_of_f32(hs))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Attention
    // -----------------------------------------------------------------------

    fn attention_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let attn = self.layers[layer_idx]
            .attention()
            .ok_or_else(|| BackendError::OperationFailed("Expected attention layer".into()))?;

        let num_heads = attn.num_heads;
        let num_kv_heads = attn.num_kv_heads;
        let kl = attn.key_length;
        let vl = attn.value_length;
        let scale = attn.scale;
        let use_neox = attn.use_neox_rope;

        self.linear_gpu(
            &format!("blk.{}.attn_q.weight", layer_idx),
            Some(&format!("blk.{}.attn_q.bias", layer_idx)),
            BufRef::HiddenNorm,
            BufRef::AttnQ,
        )?;
        self.linear_gpu(
            &format!("blk.{}.attn_k.weight", layer_idx),
            Some(&format!("blk.{}.attn_k.bias", layer_idx)),
            BufRef::HiddenNorm,
            BufRef::AttnK,
        )?;
        self.linear_gpu(
            &format!("blk.{}.attn_v.weight", layer_idx),
            Some(&format!("blk.{}.attn_v.bias", layer_idx)),
            BufRef::HiddenNorm,
            BufRef::AttnV,
        )?;

        self.dispatch_rope(num_heads, num_kv_heads, kl, self.pos, use_neox)?;

        // Update KV cache
        let k_new = self.ctx.read_buffer_floats(&self.attn_k, num_kv_heads * kl);
        let v_new = self.ctx.read_buffer_floats(&self.attn_v, num_kv_heads * vl);

        if let Some(k_cache) = &self.k_caches[layer_idx] {
            for h in 0..num_kv_heads {
                let byte_offset = (h * self.max_seq_len * kl + self.pos * kl) * 4;
                self.ctx.write_to_buffer_offset(
                    k_cache,
                    &k_new[h * kl..(h + 1) * kl],
                    byte_offset,
                )?;
            }
        }
        if let Some(v_cache) = &self.v_caches[layer_idx] {
            for h in 0..num_kv_heads {
                let byte_offset = (h * self.max_seq_len * vl + self.pos * vl) * 4;
                self.ctx.write_to_buffer_offset(
                    v_cache,
                    &v_new[h * vl..(h + 1) * vl],
                    byte_offset,
                )?;
            }
        }

        let kv_len = self.pos + 1;
        self.dispatch_attention_cached(
            layer_idx, num_heads, num_kv_heads, kl, vl, kv_len, scale,
        )?;

        self.linear_gpu(
            &format!("blk.{}.attn_output.weight", layer_idx),
            None,
            BufRef::AttnOut,
            BufRef::HiddenNorm,
        )?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Dense FFN (SwiGLU)
    // -----------------------------------------------------------------------

    fn dense_ffn_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let prefix = format!("blk.{}", layer_idx);
        let inter = self.config.intermediate_size;

        self.linear_gpu(
            &format!("{}.ffn_gate.weight", prefix),
            None,
            BufRef::HiddenNorm,
            BufRef::FfnGate,
        )?;

        self.linear_gpu(
            &format!("{}.ffn_up.weight", prefix),
            None,
            BufRef::HiddenNorm,
            BufRef::FfnUp,
        )?;

        self.dispatch_silu(&self.ffn_gate, inter)?;
        self.dispatch_mul(&self.ffn_gate, &self.ffn_up, &self.ffn_down, inter)?;

        self.linear_gpu(
            &format!("{}.ffn_down.weight", prefix),
            None,
            BufRef::FfnDown,
            BufRef::HiddenNorm,
        )?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // MoE
    // -----------------------------------------------------------------------

    fn moe_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let prefix = format!("blk.{}", layer_idx);
        let hs = self.config.hidden_size;
        let expert_inter = self.config.expert_intermediate;

        let moe = self.layers[layer_idx]
            .moe()
            .ok_or_else(|| BackendError::OperationFailed("Expected MoE layer".into()))?;

        let num_experts = moe.num_experts();
        let top_k = moe.num_experts_per_token();

        // Router logits on GPU
        let router_buf = self.ctx.create_output_buffer(num_experts)?;
        self.linear_gpu_buf(
            &format!("{}.ffn_gate_inp.weight", prefix),
            None,
            &self.hidden_norm,
            &router_buf,
        )?;

        let logits_cpu = self.ctx.read_buffer(&router_buf, num_experts);

        // Top-k on CPU
        let max_logit = logits_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = logits_cpu.iter().map(|&l| l - max_logit).collect();
        let mut indexed: Vec<(usize, f32)> = shifted.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_indices: Vec<usize> = indexed[..top_k].iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = indexed[..top_k].iter().map(|(_, l)| *l).collect();

        let max_top = top_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = top_logits.iter().map(|&l| (l - max_top).exp()).sum();
        let routing_weights: Vec<f32> = top_logits
            .iter()
            .map(|&l| (l - max_top).exp() / exp_sum)
            .collect();

        self.ctx.write_to_buffer(&self.moe_hidden, &vec![0.0f32; hs])?;

        for (sel_idx, &expert_idx) in top_indices.iter().enumerate() {
            let weight = routing_weights[sel_idx];
            let expert = &moe.experts[expert_idx];

            let gate_name = format!("{}.ffn_experts.{}.gate.weight", prefix, expert_idx);
            let up_name = format!("{}.ffn_experts.{}.up.weight", prefix, expert_idx);
            let down_name = format!("{}.ffn_experts.{}.down.weight", prefix, expert_idx);

            self.weights.upload_auto(&self.ctx, &gate_name, &expert.gate_proj)?;
            self.weights.upload_auto(&self.ctx, &up_name, &expert.up_proj)?;
            self.weights.upload_auto(&self.ctx, &down_name, &expert.down_proj)?;

            self.linear_gpu(&gate_name, None, BufRef::HiddenNorm, BufRef::MoeGate)?;
            self.dispatch_silu(&self.moe_expert_gate, expert_inter)?;

            self.linear_gpu(&up_name, None, BufRef::HiddenNorm, BufRef::MoeUp)?;

            self.dispatch_mul(
                &self.moe_expert_gate,
                &self.moe_expert_up,
                &self.moe_expert_down,
                expert_inter,
            )?;

            self.linear_gpu(&down_name, None, BufRef::MoeDown, BufRef::MoeOut)?;

            self.dispatch_scale(&self.moe_expert_out, weight, &self.moe_temp, hs)?;
            self.dispatch_add(&self.moe_hidden, &self.moe_temp, &self.moe_expert_out)?;
            self.ctx.copy_buffer(&self.moe_expert_out, &self.moe_hidden, bytes_of_f32(hs))?;
        }

        // Shared experts
        if !moe.shared_experts.is_empty() {
            let gate_scale = if self.weights.contains(&format!("{}.ffn_gate_shexp_gate", prefix)) {
                let h = self.ctx.read_buffer_floats(&self.hidden_norm, hs);
                let gw = moe
                    .shared_expert_gate
                    .as_ref()
                    .and_then(|t| t.as_f32().ok())
                    .map(|d| d.to_vec())
                    .unwrap_or_default();
                let len = hs.min(gw.len());
                let dot = simd::dot_f32(&h[..len], &gw[..len]);
                1.0 / (1.0 + (-dot).exp())
            } else {
                1.0
            };

            for se_idx in 0..moe.shared_experts.len() {
                let gn = format!("{}.ffn_gate_shexp.{}.weight", prefix, se_idx);
                let un = format!("{}.ffn_up_shexp.{}.weight", prefix, se_idx);
                let dn = format!("{}.ffn_down_shexp.{}.weight", prefix, se_idx);

                self.linear_gpu(&gn, None, BufRef::HiddenNorm, BufRef::MoeGate)?;
                self.dispatch_silu(&self.moe_expert_gate, expert_inter)?;
                self.linear_gpu(&un, None, BufRef::HiddenNorm, BufRef::MoeUp)?;
                self.dispatch_mul(
                    &self.moe_expert_gate,
                    &self.moe_expert_up,
                    &self.moe_expert_down,
                    expert_inter,
                )?;
                self.linear_gpu(&dn, None, BufRef::MoeDown, BufRef::MoeOut)?;

                self.dispatch_scale(&self.moe_expert_out, gate_scale, &self.moe_temp, hs)?;
                self.dispatch_add(&self.moe_hidden, &self.moe_temp, &self.moe_expert_out)?;
                self.ctx.copy_buffer(&self.moe_expert_out, &self.moe_hidden, bytes_of_f32(hs))?;
            }
        }

        self.ctx.copy_buffer(&self.moe_hidden, &self.hidden_norm, bytes_of_f32(hs))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Low-level dispatch helpers
    // -----------------------------------------------------------------------

    fn rms_norm_gpu(
        &self,
        weight_name: &str,
        input: BufRef,
        output: BufRef,
    ) -> BackendResult<()> {
        let n = self.config.hidden_size;
        let w = self.weights.get_f32(weight_name)
            .ok_or_else(|| BackendError::OperationFailed(format!("Missing {}", weight_name)))?;

        let x_buf = self.resolve_buf(input);
        let out_buf = self.resolve_buf(output);

        // Pass 1: partial sum of squares
        let mut push_n = Vec::with_capacity(4);
        push_n.extend_from_slice(&(n as u32).to_le_bytes());
        let tg_count = MetalContext::threadgroup_count_1d(n, 256);
        let tg_size = MTLSize::new(256, 1, 1);
        self.ctx.dispatch_threadgroups(
            "rms_norm_sum",
            &[x_buf, &self.rms_partial_buf],
            Some((&push_n, 2)),
            tg_count,
            tg_size,
        )?;

        let partial_sums = self.ctx.read_buffer(&self.rms_partial_buf, self.rms_max_workgroups);
        let sum_sq: f32 = partial_sums.iter().sum();
        let rms_inv = 1.0 / (sum_sq / n as f32 + self.config.norm_eps).sqrt();

        // Pass 2: normalize and scale
        let mut push2 = Vec::with_capacity(8);
        push2.extend_from_slice(&(n as u32).to_le_bytes());
        push2.extend_from_slice(&rms_inv.to_le_bytes());
        let (grid, tg) = MetalContext::sizes_1d(n);
        self.ctx.dispatch(
            "rms_norm_scale",
            &[x_buf, &w.buffer, out_buf],
            Some((&push2, 3)),
            grid,
            tg,
        )?;

        Ok(())
    }

    fn linear_gpu(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        input: BufRef,
        output: BufRef,
    ) -> BackendResult<()> {
        let x_buf = self.resolve_buf(input);
        let out_buf = self.resolve_buf(output);
        self.linear_gpu_buf(weight_name, bias_name, x_buf, out_buf)
    }

    fn linear_gpu_buf(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        x_buf: &Buffer,
        out_buf: &Buffer,
    ) -> BackendResult<()> {
        // Quantized path
        if let Some(qw) = self.weights.get_quant(weight_name) {
            let k = qw.shape[0];
            let n = if qw.shape.len() >= 2 { qw.shape[1] } else { 1 };
            let num_elements = k * n;

            let (shader, block_size) = match qw.dtype {
                DType::Q8_0 => ("dequant_q8_0", 32usize),
                DType::Q4K => ("dequant_q4_k", 256usize),
                DType::Q6K => ("dequant_q6_k", 256usize),
                _ => return Err(BackendError::OperationFailed(format!(
                    "Unsupported quant type {:?} for {}", qw.dtype, weight_name
                ))),
            };

            let num_blocks = num_elements / block_size;
            let mut push = Vec::with_capacity(4);
            push.extend_from_slice(&(num_blocks as u32).to_le_bytes());
            let grid = MTLSize::new(num_blocks as u64, 1, 1);
            let tg = MTLSize::new(256, 1, 1);
            self.ctx.dispatch(
                shader,
                &[&qw.buffer, &self.dequant_scratch],
                Some((&push, 2)),
                grid,
                tg,
            )?;

            let mut push2 = Vec::with_capacity(8);
            push2.extend_from_slice(&(k as u32).to_le_bytes());
            push2.extend_from_slice(&(n as u32).to_le_bytes());
            let (grid2, tg2) = MetalContext::sizes_1d(n);
            self.ctx.dispatch(
                "vec_mat",
                &[x_buf, &self.dequant_scratch, out_buf],
                Some((&push2, 3)),
                grid2,
                tg2,
            )?;

            self.apply_bias(bias_name, out_buf, n)?;
            return Ok(());
        }

        // F32 path
        if let Some(w) = self.weights.get_f32(weight_name) {
            let k = w.shape[0];
            let n = if w.shape.len() >= 2 { w.shape[1] } else { 1 };

            let mut push = Vec::with_capacity(8);
            push.extend_from_slice(&(k as u32).to_le_bytes());
            push.extend_from_slice(&(n as u32).to_le_bytes());
            let (grid, tg) = MetalContext::sizes_1d(n);
            self.ctx.dispatch(
                "vec_mat",
                &[x_buf, &w.buffer, out_buf],
                Some((&push, 3)),
                grid,
                tg,
            )?;

            self.apply_bias(bias_name, out_buf, n)?;
            return Ok(());
        }

        Err(BackendError::OperationFailed(format!("Missing weight {}", weight_name)))
    }

    fn apply_bias(&self, bias_name: Option<&str>, out_buf: &Buffer, n: usize) -> BackendResult<()> {
        if let Some(bn) = bias_name {
            if let Some(bias) = self.weights.get_f32(bn) {
                let mut push = Vec::with_capacity(4);
                push.extend_from_slice(&(n as u32).to_le_bytes());
                let (grid, tg) = MetalContext::sizes_1d(n);
                self.ctx.dispatch("add", &[out_buf, &bias.buffer, &self.moe_temp], Some((&push, 3)), grid, tg)?;
                self.ctx.copy_buffer(&self.moe_temp, out_buf, bytes_of_f32(n))?;
            }
        }
        Ok(())
    }

    fn dispatch_add(&self, a: &Buffer, b: &Buffer, out: &Buffer) -> BackendResult<()> {
        let n = self.config.hidden_size;
        let mut push = Vec::with_capacity(4);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        let (grid, tg) = MetalContext::sizes_1d(n);
        self.ctx.dispatch("add", &[a, b, out], Some((&push, 3)), grid, tg)
    }

    fn dispatch_mul(&self, a: &Buffer, b: &Buffer, out: &Buffer, n: usize) -> BackendResult<()> {
        let mut push = Vec::with_capacity(4);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        let (grid, tg) = MetalContext::sizes_1d(n);
        self.ctx.dispatch("mul", &[a, b, out], Some((&push, 3)), grid, tg)
    }

    fn dispatch_silu(&self, buf: &Buffer, n: usize) -> BackendResult<()> {
        let mut push = Vec::with_capacity(4);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        let (grid, tg) = MetalContext::sizes_1d(n);
        self.ctx.dispatch("silu", &[buf, &self.moe_temp], Some((&push, 2)), grid, tg)?;
        self.ctx.copy_buffer(&self.moe_temp, buf, bytes_of_f32(n))
    }

    fn dispatch_scale(&self, input: &Buffer, scalar: f32, output: &Buffer, n: usize) -> BackendResult<()> {
        let mut push = Vec::with_capacity(8);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&scalar.to_le_bytes());
        let (grid, tg) = MetalContext::sizes_1d(n);
        self.ctx.dispatch("scale", &[input, output], Some((&push, 2)), grid, tg)
    }

    fn dispatch_rope(
        &self,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        pos: usize,
        use_neox: bool,
    ) -> BackendResult<()> {
        let mut push = Vec::with_capacity(28);
        push.extend_from_slice(&(num_q_heads as u32).to_le_bytes());
        push.extend_from_slice(&(num_k_heads as u32).to_le_bytes());
        push.extend_from_slice(&(head_dim as u32).to_le_bytes());
        push.extend_from_slice(&(pos as u32).to_le_bytes());
        push.extend_from_slice(&self.config.freq_base.to_le_bytes());
        push.extend_from_slice(&self.config.freq_scale.to_le_bytes());
        push.extend_from_slice(&(if use_neox { 1u32 } else { 0u32 }).to_le_bytes());

        let max_heads = num_q_heads.max(num_k_heads);
        let grid = MTLSize::new(max_heads as u64, 1, 1);
        let tg = MTLSize::new(max_heads.min(256) as u64, 1, 1);
        self.ctx.dispatch("rope", &[&self.attn_q, &self.attn_k], Some((&push, 2)), grid, tg)
    }

    fn dispatch_attention_cached(
        &self,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        kl: usize,
        _vl: usize,
        kv_len: usize,
        scale: f32,
    ) -> BackendResult<()> {
        let k_cache = self.k_caches[layer_idx].as_ref()
            .ok_or_else(|| BackendError::OperationFailed("Missing K cache".into()))?;
        let v_cache = self.v_caches[layer_idx].as_ref()
            .ok_or_else(|| BackendError::OperationFailed("Missing V cache".into()))?;

        let mut push = Vec::with_capacity(24);
        push.extend_from_slice(&(num_heads as u32).to_le_bytes());
        push.extend_from_slice(&(num_kv_heads as u32).to_le_bytes());
        push.extend_from_slice(&(kv_len as u32).to_le_bytes());
        push.extend_from_slice(&(self.max_seq_len as u32).to_le_bytes());
        push.extend_from_slice(&(kl as u32).to_le_bytes());
        push.extend_from_slice(&scale.to_le_bytes());

        let grid = MTLSize::new(num_heads as u64, 1, 1);
        let tg = MTLSize::new(num_heads.min(256) as u64, 1, 1);
        self.ctx.dispatch(
            "attention_cached",
            &[&self.attn_q, k_cache, v_cache, &self.attn_out],
            Some((&push, 4)),
            grid,
            tg,
        )
    }

    // -----------------------------------------------------------------------
    // Buffer reference resolution
    // -----------------------------------------------------------------------

    fn resolve_buf(&self, r: BufRef) -> &Buffer {
        match r {
            BufRef::Hidden => &self.hidden,
            BufRef::HiddenNorm => &self.hidden_norm,
            BufRef::Residual => &self.residual,
            BufRef::FfnGate => &self.ffn_gate,
            BufRef::FfnUp => &self.ffn_up,
            BufRef::FfnDown => &self.ffn_down,
            BufRef::Logits => &self.logits,
            BufRef::AttnQ => &self.attn_q,
            BufRef::AttnK => &self.attn_k,
            BufRef::AttnV => &self.attn_v,
            BufRef::AttnOut => &self.attn_out,
            BufRef::MoeGate => &self.moe_expert_gate,
            BufRef::MoeUp => &self.moe_expert_up,
            BufRef::MoeDown => &self.moe_expert_down,
            BufRef::MoeOut => &self.moe_expert_out,
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum BufRef {
    Hidden,
    HiddenNorm,
    Residual,
    FfnGate,
    FfnUp,
    FfnDown,
    Logits,
    AttnQ,
    AttnK,
    AttnV,
    AttnOut,
    MoeGate,
    MoeUp,
    MoeDown,
    MoeOut,
}

// ---------------------------------------------------------------------------
// Weight upload
// ---------------------------------------------------------------------------

fn upload_layer_weights(
    ctx: &MetalContext,
    store: &mut MetalWeightStore,
    i: usize,
    layer: &TransformerLayer,
) -> BackendResult<()> {
    store.upload_f32(ctx, &format!("blk.{}.attn_norm.weight", i), &layer.attn_norm.weight)?;

    if let AttentionLayer::FullAttention(attn) = &layer.attn_layer {
        store.upload_auto(ctx, &format!("blk.{}.attn_q.weight", i), &attn.wq.weight)?;
        store.upload_auto(ctx, &format!("blk.{}.attn_k.weight", i), &attn.wk.weight)?;
        store.upload_auto(ctx, &format!("blk.{}.attn_v.weight", i), &attn.wv.weight)?;
        store.upload_auto(ctx, &format!("blk.{}.attn_output.weight", i), &attn.wo.weight)?;

        if let Some(ref bias) = attn.wq.bias {
            store.upload_f32(ctx, &format!("blk.{}.attn_q.bias", i), bias)?;
        }
        if let Some(ref bias) = attn.wk.bias {
            store.upload_f32(ctx, &format!("blk.{}.attn_k.bias", i), bias)?;
        }
        if let Some(ref bias) = attn.wv.bias {
            store.upload_f32(ctx, &format!("blk.{}.attn_v.bias", i), bias)?;
        }
    }

    if let Some(ref norm) = layer.post_attn_norm {
        store.upload_f32(ctx, &format!("blk.{}.post_attention_norm.weight", i), &norm.weight)?;
    }

    store.upload_f32(ctx, &format!("blk.{}.ffn_norm.weight", i), &layer.ffn_norm.weight)?;

    match &layer.ffn_layer {
        FfnLayer::Dense(ffn) => {
            store.upload_auto(ctx, &format!("blk.{}.ffn_gate.weight", i), &ffn.w_gate.weight)?;
            store.upload_auto(ctx, &format!("blk.{}.ffn_up.weight", i), &ffn.w_up.weight)?;
            store.upload_auto(ctx, &format!("blk.{}.ffn_down.weight", i), &ffn.w_down.weight)?;
        }
        FfnLayer::Moe(moe) => {
            store.upload_auto(ctx, &format!("blk.{}.ffn_gate_inp.weight", i), &moe.router.weight)?;
            for (se_idx, se) in moe.shared_experts.iter().enumerate() {
                store.upload_auto(ctx, &format!("blk.{}.ffn_gate_shexp.{}.weight", i, se_idx), &se.gate_proj)?;
                store.upload_auto(ctx, &format!("blk.{}.ffn_up_shexp.{}.weight", i, se_idx), &se.up_proj)?;
                store.upload_auto(ctx, &format!("blk.{}.ffn_down_shexp.{}.weight", i, se_idx), &se.down_proj)?;
            }
            if let Some(ref gate) = moe.shared_expert_gate {
                store.upload_f32(ctx, &format!("blk.{}.ffn_gate_shexp_gate", i), gate)?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// GpuModelWrapper: implements Model trait
// ---------------------------------------------------------------------------

use crate::model::{Architecture, InferenceContext, Model, ModelConfig, ModelError, ModelResult};
use crate::tensor::Tensor as TensorType;
use std::sync::Mutex;

pub struct MetalGpuModelWrapper {
    gpu: Mutex<MetalGpuInference>,
    config: ModelConfig,
    architecture: Architecture,
}

impl MetalGpuModelWrapper {
    pub fn new(
        gpu: MetalGpuInference,
        config: ModelConfig,
        architecture: Architecture,
    ) -> Self {
        Self {
            gpu: Mutex::new(gpu),
            config,
            architecture,
        }
    }
}

impl Model for MetalGpuModelWrapper {
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<TensorType> {
        let mut gpu = self.gpu.lock().map_err(|e| {
            ModelError::ConfigError(format!("Metal GPU inference lock poisoned: {}", e))
        })?;

        if ctx.position == 0 && gpu.position() > 0 {
            gpu.reset();
        }

        if tokens.is_empty() {
            return Err(ModelError::ConfigError("No tokens to process".into()));
        }

        let last_idx = tokens.len() - 1;
        for &token in &tokens[..last_idx] {
            gpu.prefill_token(token)?;
        }

        let logits_vec = gpu.forward(tokens[last_idx])?;
        ctx.position += tokens.len();
        ctx.kv_cache.seq_len = ctx.position;

        TensorType::from_f32(&logits_vec, vec![logits_vec.len()]).map_err(|e| e.into())
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn architecture(&self) -> Architecture {
        self.architecture
    }
}
