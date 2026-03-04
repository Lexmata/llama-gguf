//! GPU-resident inference engine for Vulkan.
//!
//! All model weights are uploaded to GPU at initialization and all layer
//! computations run through Vulkan compute shaders. Only the final logits
//! are read back to CPU. This eliminates per-operation data transfers and
//! provides significantly better throughput than the per-op Backend path.
//!
//! Supported layer types:
//! - Dense attention (standard multi-head, GQA)
//! - Dense FFN (SwiGLU)
//! - MoE (router on GPU, expert weights streamed, SwiGLU on GPU)
//!
//! Not supported (requires additional shaders):
//! - DeltaNet recurrent layers
//! - QK normalization / attention gating (Qwen3Next-specific)
//! - Partial RoPE with head dimensions != rope_dims

use std::collections::HashMap;

use crate::backend::cpu::simd;
use crate::backend::{BackendError, BackendResult};
use crate::model::layers::{AttentionLayer, FfnLayer, TransformerLayer};
use crate::model::LlamaModel;
use crate::tensor::{DType, Tensor};

use super::context::{GpuBuffer, VulkanContext};

// ---------------------------------------------------------------------------
// Weight store
// ---------------------------------------------------------------------------

struct WeightEntry {
    buffer: GpuBuffer,
    shape: Vec<usize>,
}

struct QuantEntry {
    buffer: GpuBuffer,
    dtype: DType,
    shape: Vec<usize>,
}

struct VulkanWeightStore {
    f32_weights: HashMap<String, WeightEntry>,
    quant_weights: HashMap<String, QuantEntry>,
    total_bytes: usize,
}

impl VulkanWeightStore {
    fn new() -> Self {
        Self {
            f32_weights: HashMap::new(),
            quant_weights: HashMap::new(),
            total_bytes: 0,
        }
    }

    fn upload_f32(
        &mut self,
        ctx: &VulkanContext,
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
        self.f32_weights.insert(
            name.to_string(),
            WeightEntry { buffer, shape },
        );
        Ok(())
    }

    fn upload_quantized(
        &mut self,
        ctx: &VulkanContext,
        name: &str,
        tensor: &Tensor,
    ) -> BackendResult<()> {
        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();
        let raw_bytes = tensor.data();
        let buffer = ctx.create_buffer_with_bytes(raw_bytes, raw_bytes.len() as u64)?;
        self.total_bytes += raw_bytes.len();
        self.quant_weights.insert(
            name.to_string(),
            QuantEntry { buffer, dtype, shape },
        );
        Ok(())
    }

    fn upload_auto(
        &mut self,
        ctx: &VulkanContext,
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

    fn free_all(mut self, ctx: &VulkanContext) {
        for (_, w) in self.f32_weights.drain() {
            ctx.free_buffer(w.buffer);
        }
        for (_, w) in self.quant_weights.drain() {
            ctx.free_buffer(w.buffer);
        }
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

pub struct VulkanGpuInference {
    ctx: VulkanContext,
    weights: VulkanWeightStore,
    config: InferenceConfig,
    pos: usize,
    // Persistent scratch buffers
    hidden: GpuBuffer,
    hidden_norm: GpuBuffer,
    residual: GpuBuffer,
    ffn_gate: GpuBuffer,
    ffn_up: GpuBuffer,
    ffn_down: GpuBuffer,
    logits: GpuBuffer,
    // Dequantization scratch (sized for largest weight matrix)
    dequant_scratch: GpuBuffer,
    dequant_scratch_floats: usize,
    // Attention scratch
    attn_q: GpuBuffer,
    attn_k: GpuBuffer,
    attn_v: GpuBuffer,
    attn_out: GpuBuffer,
    // KV caches [num_kv_heads * max_seq_len * head_dim]
    k_caches: Vec<Option<GpuBuffer>>,
    v_caches: Vec<Option<GpuBuffer>>,
    max_seq_len: usize,
    // CPU embeddings table
    cpu_embeddings: Vec<f32>,
    // Owned layers (for reading config at inference time)
    layers: Vec<TransformerLayer>,
    // MoE scratch
    moe_hidden: GpuBuffer,
    moe_expert_gate: GpuBuffer,
    moe_expert_up: GpuBuffer,
    moe_expert_down: GpuBuffer,
    moe_expert_out: GpuBuffer,
    moe_temp: GpuBuffer,
    // Per-layer flags
    has_attention: Vec<bool>,
    // Temp buffers for RMS norm partial sums
    rms_partial_buf: GpuBuffer,
    rms_max_workgroups: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytes_of_f32(n: usize) -> u64 {
    (n * std::mem::size_of::<f32>()) as u64
}

impl VulkanGpuInference {
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

        eprintln!("Initializing Vulkan GPU-only inference...");

        let ctx = VulkanContext::new(0, false)?;
        let mut weights = VulkanWeightStore::new();

        // Upload all layer weights
        for (i, layer) in layers.iter().enumerate() {
            if i % 4 == 0 {
                eprintln!("  Uploading layer {}/{}", i + 1, layers.len());
            }
            upload_layer_weights(&ctx, &mut weights, i, layer)?;
        }

        // Upload norm + output weights
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

        // CPU embeddings
        let cpu_embeddings = if token_embedding.dtype() == DType::F32 {
            token_embedding.as_f32()?.to_vec()
        } else {
            let numel = token_embedding.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            crate::backend::cpu::ops::dequantize(&token_embedding, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        // Determine dequant scratch size (largest quantized weight)
        let mut max_quant_elements = 0usize;
        for w in weights.quant_weights.values() {
            let numel: usize = w.shape.iter().product();
            max_quant_elements = max_quant_elements.max(numel);
        }
        let dequant_scratch_floats = max_quant_elements.max(1);

        // Allocate persistent buffers
        let hidden = ctx.create_persistent_buffer(config.hidden_size)?;
        let hidden_norm = ctx.create_persistent_buffer(config.hidden_size)?;
        let residual = ctx.create_persistent_buffer(config.hidden_size)?;
        let ffn_gate = ctx.create_persistent_buffer(config.intermediate_size)?;
        let ffn_up = ctx.create_persistent_buffer(config.intermediate_size)?;
        let ffn_down = ctx.create_persistent_buffer(config.hidden_size)?;
        let logits = ctx.create_output_buffer(config.vocab_size)?;
        let dequant_scratch = ctx.create_persistent_buffer(dequant_scratch_floats)?;

        // Attention scratch
        let max_q_out = layers.iter().filter_map(|l| l.attention()).map(|a| a.wq.out_features).max().unwrap_or(config.hidden_size);
        let max_kv_flat = layers.iter().filter_map(|l| l.attention())
            .map(|a| config.num_kv_heads * a.key_length.max(a.value_length))
            .max().unwrap_or(config.hidden_size);
        let max_attn_out = layers.iter().filter_map(|l| l.attention())
            .map(|a| config.num_heads * a.value_length)
            .max().unwrap_or(config.hidden_size);

        let attn_q = ctx.create_persistent_buffer(max_q_out.max(1))?;
        let attn_k = ctx.create_persistent_buffer(max_kv_flat.max(1))?;
        let attn_v = ctx.create_persistent_buffer(max_kv_flat.max(1))?;
        let attn_out = ctx.create_persistent_buffer(max_attn_out.max(1))?;

        // KV caches
        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            if has_attention[i] {
                if let Some(attn) = layers[i].attention() {
                    let kl = attn.key_length;
                    let vl = attn.value_length;
                    k_caches.push(Some(ctx.create_persistent_buffer(config.num_kv_heads * max_seq_len * kl)?));
                    v_caches.push(Some(ctx.create_persistent_buffer(config.num_kv_heads * max_seq_len * vl)?));
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
        let moe_hidden = ctx.create_persistent_buffer(config.hidden_size)?;
        let moe_expert_gate = ctx.create_persistent_buffer(expert_intermediate)?;
        let moe_expert_up = ctx.create_persistent_buffer(expert_intermediate)?;
        let moe_expert_down = ctx.create_persistent_buffer(config.hidden_size)?;
        let moe_expert_out = ctx.create_persistent_buffer(config.hidden_size)?;
        let moe_temp = ctx.create_persistent_buffer(config.hidden_size)?;

        // RMS norm partial sums buffer
        let rms_max_workgroups = ((config.hidden_size + 255) / 256).max(1);
        let rms_partial_buf = ctx.create_output_buffer(rms_max_workgroups)?;

        let attn_count = has_attention.iter().filter(|&&x| x).count();
        let moe_count = layers.iter().filter(|l| l.moe().is_some()).count();
        let vram_mb = weights.total_bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "Vulkan GPU-only inference ready: {:.1} MB VRAM, {} attn + {} MoE layers, {:.1} MB dequant scratch",
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

        // Final norm
        self.rms_norm_gpu("output_norm.weight", &BufferRef::Hidden, &BufferRef::HiddenNorm)?;

        // Output projection
        self.linear_gpu("output.weight", None, &BufferRef::HiddenNorm, &BufferRef::Logits)?;

        let logits = self.ctx.read_buffer(&self.logits)?;
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
                let n = buf.size as usize / 4;
                let zeros = vec![0.0f32; n];
                let _ = self.ctx.write_to_buffer(buf, &zeros);
            }
        }
        for vc in self.v_caches.iter() {
            if let Some(buf) = vc {
                let n = buf.size as usize / 4;
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

        // Attention norm
        self.rms_norm_gpu(
            &format!("blk.{}.attn_norm.weight", layer_idx),
            &BufferRef::Hidden,
            &BufferRef::HiddenNorm,
        )?;

        // Attention
        if self.has_attention[layer_idx] {
            self.attention_forward(layer_idx)?;
        } else {
            return Err(BackendError::OperationFailed(format!(
                "Layer {} has no GPU attention path (DeltaNet not supported in Vulkan gpu_only)",
                layer_idx
            )));
        }

        // Residual add: hidden = residual + hidden_norm (attn output)
        self.dispatch_add(&self.residual, &self.hidden_norm, &self.hidden)?;
        self.ctx.copy_buffer(&self.hidden, &self.residual, bytes_of_f32(hs))?;

        // FFN norm
        let ffn_norm_name = if self.layers[layer_idx].post_attn_norm.is_some() {
            format!("blk.{}.post_attention_norm.weight", layer_idx)
        } else {
            format!("blk.{}.ffn_norm.weight", layer_idx)
        };
        self.rms_norm_gpu(&ffn_norm_name, &BufferRef::Hidden, &BufferRef::HiddenNorm)?;

        // FFN
        match &self.layers[layer_idx].ffn_layer {
            FfnLayer::Dense(_) => self.dense_ffn_forward(layer_idx)?,
            FfnLayer::Moe(_) => self.moe_forward(layer_idx)?,
        }

        // FFN residual add
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

        // Q/K/V projections
        self.linear_gpu(
            &format!("blk.{}.attn_q.weight", layer_idx),
            Some(&format!("blk.{}.attn_q.bias", layer_idx)),
            &BufferRef::HiddenNorm,
            &BufferRef::AttnQ,
        )?;
        self.linear_gpu(
            &format!("blk.{}.attn_k.weight", layer_idx),
            Some(&format!("blk.{}.attn_k.bias", layer_idx)),
            &BufferRef::HiddenNorm,
            &BufferRef::AttnK,
        )?;
        self.linear_gpu(
            &format!("blk.{}.attn_v.weight", layer_idx),
            Some(&format!("blk.{}.attn_v.bias", layer_idx)),
            &BufferRef::HiddenNorm,
            &BufferRef::AttnV,
        )?;

        // RoPE on Q and K
        let num_q_heads = num_heads;
        let num_k_heads = num_kv_heads;
        self.dispatch_rope(&self.attn_q, &self.attn_k, num_q_heads, num_k_heads, kl, self.pos, use_neox)?;

        // Update KV cache from CPU: read Q/K tensors, write to cache at position
        let k_new = self.ctx.read_buffer_floats(&self.attn_k, num_kv_heads * kl)?;
        let v_new = self.ctx.read_buffer_floats(&self.attn_v, num_kv_heads * vl)?;

        if let Some(ref k_cache) = self.k_caches[layer_idx] {
            for h in 0..num_kv_heads {
                let byte_offset = (h * self.max_seq_len * kl + self.pos * kl) * 4;
                self.ctx.write_to_buffer_offset(
                    k_cache,
                    &k_new[h * kl..(h + 1) * kl],
                    byte_offset,
                )?;
            }
        }
        if let Some(ref v_cache) = self.v_caches[layer_idx] {
            for h in 0..num_kv_heads {
                let byte_offset = (h * self.max_seq_len * vl + self.pos * vl) * 4;
                self.ctx.write_to_buffer_offset(
                    v_cache,
                    &v_new[h * vl..(h + 1) * vl],
                    byte_offset,
                )?;
            }
        }

        // Attention cached
        let kv_len = self.pos + 1;
        self.dispatch_attention_cached(
            layer_idx, num_heads, num_kv_heads, kl, vl, kv_len, scale,
        )?;

        // Output projection: attn_out → hidden_norm
        self.linear_gpu(
            &format!("blk.{}.attn_output.weight", layer_idx),
            None,
            &BufferRef::AttnOut,
            &BufferRef::HiddenNorm,
        )?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Dense FFN (SwiGLU)
    // -----------------------------------------------------------------------

    fn dense_ffn_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let prefix = format!("blk.{}", layer_idx);
        let inter = self.config.intermediate_size;

        // gate = hidden_norm @ W_gate
        self.linear_gpu(
            &format!("{}.ffn_gate.weight", prefix),
            None,
            &BufferRef::HiddenNorm,
            &BufferRef::FfnGate,
        )?;

        // up = hidden_norm @ W_up
        self.linear_gpu(
            &format!("{}.ffn_up.weight", prefix),
            None,
            &BufferRef::HiddenNorm,
            &BufferRef::FfnUp,
        )?;

        // SiLU(gate)
        self.dispatch_silu(&self.ffn_gate, inter)?;

        // gate * up → ffn_down (reuse as temp)
        self.dispatch_mul(&self.ffn_gate, &self.ffn_up, &self.ffn_down, inter)?;

        // down projection: ffn_down @ W_down → hidden_norm
        self.linear_gpu(
            &format!("{}.ffn_down.weight", prefix),
            None,
            &BufferRef::FfnDown,
            &BufferRef::HiddenNorm,
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
        let mut router_buf = self.ctx.create_output_buffer(num_experts)?;
        self.linear_gpu(
            &format!("{}.ffn_gate_inp.weight", prefix),
            None,
            &BufferRef::HiddenNorm,
            &BufferRef::Custom(&mut router_buf),
        )?;

        let logits_cpu = self.ctx.read_buffer(&router_buf)?;
        self.ctx.free_buffer(router_buf);

        // Top-k on CPU
        let max_logit = logits_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = logits_cpu.iter().map(|&l| l - max_logit).collect();
        let mut indexed: Vec<(usize, f32)> = shifted.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_indices: Vec<usize> = indexed[..top_k].iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = indexed[..top_k].iter().map(|(_, l)| *l).collect();

        let max_top = top_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = top_logits.iter().map(|&l| (l - max_top).exp()).sum();
        let routing_weights: Vec<f32> = top_logits.iter().map(|&l| (l - max_top).exp() / exp_sum).collect();

        // Zero accumulator
        self.ctx.write_to_buffer(&self.moe_hidden, &vec![0.0f32; hs])?;

        // Compute selected experts
        for (sel_idx, &expert_idx) in top_indices.iter().enumerate() {
            let weight = routing_weights[sel_idx];
            let expert = &moe.experts[expert_idx];

            // Upload expert weights to scratch store entries
            let gate_name = format!("{}.moe_scratch.gate", prefix);
            let up_name = format!("{}.moe_scratch.up", prefix);
            let down_name = format!("{}.moe_scratch.down", prefix);

            self.weights.upload_auto(&self.ctx, &gate_name, &expert.gate_proj)?;
            self.weights.upload_auto(&self.ctx, &up_name, &expert.up_proj)?;
            self.weights.upload_auto(&self.ctx, &down_name, &expert.down_proj)?;

            // gate projection
            self.linear_gpu(&gate_name, None, &BufferRef::HiddenNorm, &BufferRef::MoeGate)?;
            self.dispatch_silu(&self.moe_expert_gate, expert_inter)?;

            // up projection
            self.linear_gpu(&up_name, None, &BufferRef::HiddenNorm, &BufferRef::MoeUp)?;

            // gate * up
            self.dispatch_mul(&self.moe_expert_gate, &self.moe_expert_up, &self.moe_expert_down, expert_inter)?;

            // down projection → moe_expert_out
            self.linear_gpu(&down_name, None, &BufferRef::MoeDown, &BufferRef::MoeOut)?;

            // Weighted accumulate: moe_hidden += weight * expert_out
            self.dispatch_scale(&self.moe_expert_out, weight, &self.moe_temp, hs)?;
            self.dispatch_add(&self.moe_hidden, &self.moe_temp, &self.moe_expert_out)?;
            self.ctx.copy_buffer(&self.moe_expert_out, &self.moe_hidden, bytes_of_f32(hs))?;
        }

        // Shared experts
        if !moe.shared_experts.is_empty() {
            let gate_scale = if self.weights.contains(&format!("{}.ffn_gate_shexp_gate", prefix)) {
                let h = self.ctx.read_buffer_floats(&self.hidden_norm, hs)?;
                let gw = moe.shared_expert_gate.as_ref()
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

                self.linear_gpu(&gn, None, &BufferRef::HiddenNorm, &BufferRef::MoeGate)?;
                self.dispatch_silu(&self.moe_expert_gate, expert_inter)?;
                self.linear_gpu(&un, None, &BufferRef::HiddenNorm, &BufferRef::MoeUp)?;
                self.dispatch_mul(&self.moe_expert_gate, &self.moe_expert_up, &self.moe_expert_down, expert_inter)?;
                self.linear_gpu(&dn, None, &BufferRef::MoeDown, &BufferRef::MoeOut)?;

                self.dispatch_scale(&self.moe_expert_out, gate_scale, &self.moe_temp, hs)?;
                self.dispatch_add(&self.moe_hidden, &self.moe_temp, &self.moe_expert_out)?;
                self.ctx.copy_buffer(&self.moe_expert_out, &self.moe_hidden, bytes_of_f32(hs))?;
            }
        }

        // Copy MoE result → hidden_norm
        self.ctx.copy_buffer(&self.moe_hidden, &self.hidden_norm, bytes_of_f32(hs))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Low-level dispatch helpers
    // -----------------------------------------------------------------------

    fn rms_norm_gpu(
        &self,
        weight_name: &str,
        input: &BufferRef,
        output: &BufferRef,
    ) -> BackendResult<()> {
        let n = self.config.hidden_size;
        let w = self.weights.get_f32(weight_name)
            .ok_or_else(|| BackendError::OperationFailed(format!("Missing {}", weight_name)))?;

        let x_buf = self.resolve_buf(input);
        let out_buf = self.resolve_buf(output);

        // Pass 1: partial sum of squares
        let push_n = (n as i32).to_le_bytes();
        let wg = VulkanContext::workgroup_count_1d(n, 256);
        self.ctx.dispatch("rms_norm_sum", &[x_buf, &self.rms_partial_buf], &push_n, wg)?;

        let partial_sums = self.ctx.read_buffer(&self.rms_partial_buf)?;
        let sum_sq: f32 = partial_sums[..self.rms_max_workgroups].iter().sum();
        let rms_inv = 1.0 / (sum_sq / n as f32 + self.config.norm_eps).sqrt();

        // Pass 2: normalize and scale
        let mut push2 = Vec::with_capacity(8);
        push2.extend_from_slice(&(n as i32).to_le_bytes());
        push2.extend_from_slice(&rms_inv.to_le_bytes());
        self.ctx.dispatch("rms_norm_scale", &[x_buf, &w.buffer, out_buf], &push2, wg)?;

        Ok(())
    }

    fn linear_gpu(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        input: &BufferRef,
        output: &BufferRef,
    ) -> BackendResult<()> {
        let x_buf = self.resolve_buf(input);
        let out_buf = self.resolve_buf(output);

        // Try quantized path first
        if let Some(qw) = self.weights.get_quant(weight_name) {
            let k = qw.shape[0];
            let n = if qw.shape.len() >= 2 { qw.shape[1] } else { 1 };
            let num_elements = k * n;

            let (shader, block_size, _type_size) = match qw.dtype {
                DType::Q8_0 => ("dequant_q8_0", 32usize, 34usize),
                DType::Q4K => ("dequant_q4_k", 256usize, 144usize),
                DType::Q6K => ("dequant_q6_k", 256usize, 210usize),
                _ => return Err(BackendError::OperationFailed(format!(
                    "Unsupported quant type {:?} for {}",
                    qw.dtype, weight_name
                ))),
            };

            // Dequant to scratch
            let num_blocks = num_elements / block_size;
            let push = (num_blocks as i32).to_le_bytes();
            let wg = (num_blocks as u32, 1, 1);
            self.ctx.dispatch(shader, &[&qw.buffer, &self.dequant_scratch], &push, wg)?;

            // vec_mat: x @ dequantized_weight
            let mut push2 = Vec::with_capacity(8);
            push2.extend_from_slice(&(k as i32).to_le_bytes());
            push2.extend_from_slice(&(n as i32).to_le_bytes());
            let wg2 = VulkanContext::workgroup_count_1d(n, 256);
            self.ctx.dispatch("vec_mat", &[x_buf, &self.dequant_scratch, out_buf], &push2, wg2)?;

            self.apply_bias(bias_name, out_buf, n)?;
            return Ok(());
        }

        // F32 path
        if let Some(w) = self.weights.get_f32(weight_name) {
            let k = w.shape[0];
            let n = if w.shape.len() >= 2 { w.shape[1] } else { 1 };

            let mut push = Vec::with_capacity(8);
            push.extend_from_slice(&(k as i32).to_le_bytes());
            push.extend_from_slice(&(n as i32).to_le_bytes());
            let wg = VulkanContext::workgroup_count_1d(n, 256);
            self.ctx.dispatch("vec_mat", &[x_buf, &w.buffer, out_buf], &push, wg)?;

            self.apply_bias(bias_name, out_buf, n)?;
            return Ok(());
        }

        Err(BackendError::OperationFailed(format!("Missing weight {}", weight_name)))
    }

    fn apply_bias(&self, bias_name: Option<&str>, out_buf: &GpuBuffer, n: usize) -> BackendResult<()> {
        if let Some(bn) = bias_name {
            if let Some(bias) = self.weights.get_f32(bn) {
                let push = (n as i32).to_le_bytes();
                let wg = VulkanContext::workgroup_count_1d(n, 256);
                // add: out = out + bias → need temp
                // We use moe_temp as scratch for bias add
                self.ctx.dispatch("add", &[out_buf, &bias.buffer, &self.moe_temp], &push, wg)?;
                self.ctx.copy_buffer(&self.moe_temp, out_buf, bytes_of_f32(n))?;
            }
        }
        Ok(())
    }

    fn dispatch_add(&self, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer) -> BackendResult<()> {
        let n = self.config.hidden_size;
        let push = (n as i32).to_le_bytes();
        let wg = VulkanContext::workgroup_count_1d(n, 256);
        self.ctx.dispatch("add", &[a, b, out], &push, wg)
    }

    fn dispatch_mul(&self, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer, n: usize) -> BackendResult<()> {
        let push = (n as i32).to_le_bytes();
        let wg = VulkanContext::workgroup_count_1d(n, 256);
        self.ctx.dispatch("mul", &[a, b, out], &push, wg)
    }

    fn dispatch_silu(&self, buf: &GpuBuffer, n: usize) -> BackendResult<()> {
        // silu writes to output buffer; use moe_temp as temp then copy back
        let push = (n as i32).to_le_bytes();
        let wg = VulkanContext::workgroup_count_1d(n, 256);
        self.ctx.dispatch("silu", &[buf, &self.moe_temp], &push, wg)?;
        self.ctx.copy_buffer(&self.moe_temp, buf, bytes_of_f32(n))
    }

    fn dispatch_scale(&self, input: &GpuBuffer, scalar: f32, output: &GpuBuffer, n: usize) -> BackendResult<()> {
        let mut push = Vec::with_capacity(8);
        push.extend_from_slice(&(n as i32).to_le_bytes());
        push.extend_from_slice(&scalar.to_le_bytes());
        let wg = VulkanContext::workgroup_count_1d(n, 256);
        self.ctx.dispatch("scale", &[input, output], &push, wg)
    }

    fn dispatch_rope(
        &self,
        q_buf: &GpuBuffer,
        k_buf: &GpuBuffer,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        pos: usize,
        use_neox: bool,
    ) -> BackendResult<()> {
        let mut push = Vec::with_capacity(28);
        push.extend_from_slice(&(num_q_heads as i32).to_le_bytes());
        push.extend_from_slice(&(num_k_heads as i32).to_le_bytes());
        push.extend_from_slice(&(head_dim as i32).to_le_bytes());
        push.extend_from_slice(&(pos as i32).to_le_bytes());
        push.extend_from_slice(&self.config.freq_base.to_le_bytes());
        push.extend_from_slice(&self.config.freq_scale.to_le_bytes());
        push.extend_from_slice(&(if use_neox { 1i32 } else { 0i32 }).to_le_bytes());

        let max_heads = num_q_heads.max(num_k_heads);
        let wg = (max_heads as u32, 1, 1);
        self.ctx.dispatch("rope", &[q_buf, k_buf], &push, wg)
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
        push.extend_from_slice(&(num_heads as i32).to_le_bytes());
        push.extend_from_slice(&(num_kv_heads as i32).to_le_bytes());
        push.extend_from_slice(&(kv_len as i32).to_le_bytes());
        push.extend_from_slice(&(self.max_seq_len as i32).to_le_bytes());
        push.extend_from_slice(&(kl as i32).to_le_bytes());
        push.extend_from_slice(&scale.to_le_bytes());

        let wg = (num_heads as u32, 1, 1);
        self.ctx.dispatch(
            "attention_cached",
            &[&self.attn_q, k_cache, v_cache, &self.attn_out],
            &push,
            wg,
        )
    }

    // -----------------------------------------------------------------------
    // Buffer reference resolution
    // -----------------------------------------------------------------------

    fn resolve_buf<'a>(&'a self, r: &'a BufferRef<'a>) -> &'a GpuBuffer {
        match r {
            BufferRef::Hidden => &self.hidden,
            BufferRef::HiddenNorm => &self.hidden_norm,
            BufferRef::Residual => &self.residual,
            BufferRef::FfnGate => &self.ffn_gate,
            BufferRef::FfnUp => &self.ffn_up,
            BufferRef::FfnDown => &self.ffn_down,
            BufferRef::Logits => &self.logits,
            BufferRef::AttnQ => &self.attn_q,
            BufferRef::AttnK => &self.attn_k,
            BufferRef::AttnV => &self.attn_v,
            BufferRef::AttnOut => &self.attn_out,
            BufferRef::MoeGate => &self.moe_expert_gate,
            BufferRef::MoeUp => &self.moe_expert_up,
            BufferRef::MoeDown => &self.moe_expert_down,
            BufferRef::MoeOut => &self.moe_expert_out,
            BufferRef::Custom(buf) => buf,
        }
    }
}

#[allow(dead_code)]
enum BufferRef<'a> {
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
    Custom(&'a GpuBuffer),
}

// ---------------------------------------------------------------------------
// Weight upload
// ---------------------------------------------------------------------------

fn upload_layer_weights(
    ctx: &VulkanContext,
    store: &mut VulkanWeightStore,
    i: usize,
    layer: &TransformerLayer,
) -> BackendResult<()> {
    // Attention norm
    store.upload_f32(ctx, &format!("blk.{}.attn_norm.weight", i), &layer.attn_norm.weight)?;

    // Attention weights
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

    // Post-attention norm
    if let Some(ref norm) = layer.post_attn_norm {
        store.upload_f32(ctx, &format!("blk.{}.post_attention_norm.weight", i), &norm.weight)?;
    }

    // FFN norm
    store.upload_f32(ctx, &format!("blk.{}.ffn_norm.weight", i), &layer.ffn_norm.weight)?;

    // FFN weights
    match &layer.ffn_layer {
        FfnLayer::Dense(ffn) => {
            store.upload_auto(ctx, &format!("blk.{}.ffn_gate.weight", i), &ffn.w_gate.weight)?;
            store.upload_auto(ctx, &format!("blk.{}.ffn_up.weight", i), &ffn.w_up.weight)?;
            store.upload_auto(ctx, &format!("blk.{}.ffn_down.weight", i), &ffn.w_down.weight)?;
        }
        FfnLayer::Moe(moe) => {
            store.upload_auto(ctx, &format!("blk.{}.ffn_gate_inp.weight", i), &moe.router.weight)?;
            // Shared expert weights
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
use std::sync::Mutex;

pub struct VulkanGpuModelWrapper {
    gpu: Mutex<VulkanGpuInference>,
    config: ModelConfig,
    architecture: Architecture,
}

impl VulkanGpuModelWrapper {
    pub fn new(
        gpu: VulkanGpuInference,
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

impl Model for VulkanGpuModelWrapper {
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor> {
        let mut gpu = self.gpu.lock().map_err(|e| {
            ModelError::ConfigError(format!("Vulkan GPU inference lock poisoned: {}", e))
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

        Tensor::from_f32(&logits_vec, vec![logits_vec.len()]).map_err(|e| e.into())
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn architecture(&self) -> Architecture {
        self.architecture
    }
}
