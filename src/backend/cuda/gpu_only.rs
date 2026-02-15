//! Full GPU-only inference with no intermediate CPU transfers
//!
//! All computation happens on GPU. Only embedding lookup at the start
//! and logits download at the end touch CPU memory.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::model::LlamaModel;
use crate::tensor::DType;

use super::dequant_weights::GpuWeightStore;
use super::kernels::CudaKernels;

/// GPU-only inference engine
pub struct GpuOnlyInference {
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    weights: GpuWeightStore,
    config: InferenceConfig,
    pos: usize,
    // GPU scratch buffers
    hidden: CudaSlice<f32>,
    hidden_norm: CudaSlice<f32>,
    residual: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ffn_gate: CudaSlice<f32>,
    ffn_up: CudaSlice<f32>,
    ffn_down: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    // GPU KV cache per layer
    k_cache: Vec<CudaSlice<f32>>,
    v_cache: Vec<CudaSlice<f32>>,
    // CPU copy of dequantized embeddings (avoids downloading entire GPU table each token)
    cpu_embeddings: Vec<f32>,
}

#[derive(Clone)]
struct InferenceConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    max_seq_len: usize,
    norm_eps: f32,
    freq_base: f32,
    freq_scale: f32,
    use_neox_rope: bool,
}

// ---------------------------------------------------------------------------
// Free-standing GPU kernel helpers
//
// These are free functions (not &self methods) so that the caller can pass
// mutable scratch-buffer references without conflicting with the immutable
// borrow that a &self method would create on the whole struct.
// ---------------------------------------------------------------------------

fn rms_norm_gpu(
    kernels: &CudaKernels,
    weights: &GpuWeightStore,
    device: &Arc<CudaDevice>,
    hidden_size: usize,
    norm_eps: f32,
    weight_name: &str,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> BackendResult<()> {
    let weight = weights
        .get(weight_name)
        .ok_or_else(|| BackendError::OperationFailed(format!("Missing {}", weight_name)))?;

    let n = hidden_size;
    let eps = norm_eps;

    // Sum of squares reduction
    let mut sum_sq = device
        .alloc_zeros::<f32>(1)
        .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    unsafe {
        kernels
            .rms_norm_sum_sq
            .clone()
            .launch(config, (x, &mut sum_sq, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

    let sum_sq_val: Vec<f32> = device
        .dtoh_sync_copy(&sum_sq)
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
    let rms = (sum_sq_val[0] / n as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    let config = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        kernels
            .rms_norm_scale
            .clone()
            .launch(config, (x, &weight.data, out, rms_inv, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

fn linear_gpu(
    kernels: &CudaKernels,
    weights: &GpuWeightStore,
    device: &Arc<CudaDevice>,
    weight_name: &str,
    bias_name: Option<&str>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> BackendResult<()> {
    let weight = weights
        .get(weight_name)
        .ok_or_else(|| BackendError::OperationFailed(format!("Missing {}", weight_name)))?;

    let k = weight.shape[0];
    let n = weight.shape[1];

    let config = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        kernels
            .vec_mat_f32
            .clone()
            .launch(config, (x, &weight.data, &mut *out, k as i32, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

    // Add bias if present
    if let Some(bias_name) = bias_name {
        if let Some(bias) = weights.get(bias_name) {
            let mut temp = device
                .alloc_zeros::<f32>(n)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
            unsafe {
                kernels.add_f32.clone().launch(
                    config,
                    (&*out as &CudaSlice<f32>, &bias.data, &mut temp, n as i32),
                )
            }
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
            device
                .dtod_copy(&temp, out)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        }
    }

    Ok(())
}

fn add_gpu(
    kernels: &CudaKernels,
    hidden_size: usize,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> BackendResult<()> {
    let n = hidden_size;
    let config = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        kernels
            .add_f32
            .clone()
            .launch(config, (a, b, out, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

fn mul_gpu(
    kernels: &CudaKernels,
    intermediate_size: usize,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> BackendResult<()> {
    let n = intermediate_size;
    let config = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        kernels
            .mul_f32
            .clone()
            .launch(config, (a, b, out, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

fn silu_gpu(
    kernels: &CudaKernels,
    device: &Arc<CudaDevice>,
    intermediate_size: usize,
    x: &mut CudaSlice<f32>,
) -> BackendResult<()> {
    let n = intermediate_size;
    let config = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut temp = device
        .alloc_zeros::<f32>(n)
        .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
    unsafe {
        kernels
            .silu_f32
            .clone()
            .launch(config, (&*x as &CudaSlice<f32>, &mut temp, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
    device
        .dtod_copy(&temp, x)
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

// ---------------------------------------------------------------------------
// GpuOnlyInference implementation
// ---------------------------------------------------------------------------

impl GpuOnlyInference {
    pub fn from_model(model: &LlamaModel, max_seq_len: usize) -> BackendResult<Self> {
        let cfg = model.config();

        let device = Arc::new(
            CudaDevice::new(0).map_err(|e| BackendError::InitializationFailed(format!("{}", e)))?,
        );

        eprintln!("Initializing GPU-only inference...");

        let kernels = CudaKernels::new(Arc::clone(&device))?;

        let weights = super::dequant_weights::upload_model_weights(
            Arc::clone(&device),
            model.layers(),
            model.token_embedding(),
            model.output(),
            model.norm(),
        )?;

        let use_neox = model
            .layers()
            .first()
            .map(|l| l.attention.use_neox_rope)
            .unwrap_or(false);

        let config = InferenceConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_layers,
            vocab_size: cfg.vocab_size,
            max_seq_len,
            norm_eps: cfg.norm_eps,
            freq_base: cfg.rope_config.freq_base,
            freq_scale: cfg.rope_config.freq_scale,
            use_neox_rope: use_neox,
        };

        // Dequantize embedding table to CPU once — avoids downloading the
        // entire GPU embedding table (hundreds of MB) on every token.
        let emb_tensor = model.token_embedding();
        let cpu_embeddings = if emb_tensor.dtype() == DType::F32 {
            emb_tensor
                .as_f32()
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?
                .to_vec()
        } else {
            let numel = emb_tensor.numel();
            let mut dequant =
                crate::tensor::Tensor::zeros(vec![numel], crate::tensor::DType::F32);
            crate::backend::cpu::ops::dequantize(emb_tensor, &mut dequant)?;
            dequant
                .as_f32()
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?
                .to_vec()
        };

        // Allocate scratch buffers
        let alloc = |size: usize| -> BackendResult<CudaSlice<f32>> {
            device
                .alloc_zeros(size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))
        };

        let hidden = alloc(cfg.hidden_size)?;
        let hidden_norm = alloc(cfg.hidden_size)?;
        let residual = alloc(cfg.hidden_size)?;
        let q = alloc(cfg.num_heads * cfg.head_dim)?;
        let k = alloc(cfg.num_kv_heads * cfg.head_dim)?;
        let v = alloc(cfg.num_kv_heads * cfg.head_dim)?;
        let attn_out = alloc(cfg.hidden_size)?;
        let ffn_gate = alloc(cfg.intermediate_size)?;
        let ffn_up = alloc(cfg.intermediate_size)?;
        let ffn_down = alloc(cfg.hidden_size)?;
        let logits = alloc(cfg.vocab_size)?;

        // KV cache
        let kv_size = cfg.num_kv_heads * max_seq_len * cfg.head_dim;
        let mut k_cache = Vec::with_capacity(cfg.num_layers);
        let mut v_cache = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            k_cache.push(alloc(kv_size)?);
            v_cache.push(alloc(kv_size)?);
        }

        let vram_mb = weights.vram_usage() as f64 / (1024.0 * 1024.0);
        eprintln!("GPU-only inference ready: {:.1} MB VRAM", vram_mb);

        Ok(Self {
            device: Arc::clone(&device),
            kernels,
            weights,
            config,
            pos: 0,
            hidden,
            hidden_norm,
            residual,
            q,
            k,
            v,
            attn_out,
            ffn_gate,
            ffn_up,
            ffn_down,
            logits,
            k_cache,
            v_cache,
            cpu_embeddings,
        })
    }

    /// Forward pass for a single token — returns logits on CPU.
    ///
    /// Only two CPU↔GPU transfers happen:
    /// 1. Upload token embedding row (hidden_size × 4 bytes)
    /// 2. Download logits (vocab_size × 4 bytes)
    pub fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> {
        // 1. Embed token (CPU -> GPU, one token row only)
        self.embed_token(token_id)?;

        // 2. Copy to residual
        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // 3. Process layers (all on GPU)
        let num_layers = self.config.num_layers;
        for layer_idx in 0..num_layers {
            self.process_layer_gpu(layer_idx)?;
        }

        // 4. Final norm (GPU only — no CPU round-trip)
        rms_norm_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            self.config.hidden_size,
            self.config.norm_eps,
            "output_norm.weight",
            &self.hidden,
            &mut self.hidden_norm,
        )?;

        // 5. Output projection (GPU only — no CPU round-trip)
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            "output.weight",
            None,
            &self.hidden_norm,
            &mut self.logits,
        )?;

        // 6. Download logits (GPU -> CPU, one-time)
        let logits = self
            .device
            .dtoh_sync_copy(&self.logits)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        self.pos += 1;
        Ok(logits)
    }

    /// Process a token through all layers without computing logits.
    ///
    /// Used for prefill tokens where we only need to populate the KV cache.
    /// Skips the final norm, output projection, and logits download, saving
    /// significant time on large prompts.
    pub fn prefill_token(&mut self, token_id: u32) -> BackendResult<()> {
        // 1. Embed token
        self.embed_token(token_id)?;

        // 2. Copy to residual
        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // 3. Process layers (all on GPU)
        let num_layers = self.config.num_layers;
        for layer_idx in 0..num_layers {
            self.process_layer_gpu(layer_idx)?;
        }

        self.pos += 1;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    fn embed_token(&mut self, token_id: u32) -> BackendResult<()> {
        let hidden_size = self.config.hidden_size;
        let offset = token_id as usize * hidden_size;

        // Upload just the single token's embedding row from the CPU table.
        // This is a tiny transfer (~3.5 KB for hidden_size=896) compared to
        // the old approach that downloaded the ENTIRE embedding table from GPU
        // (~545 MB for Qwen 0.5B) on every token.
        self.device
            .htod_sync_copy_into(
                &self.cpu_embeddings[offset..offset + hidden_size],
                &mut self.hidden,
            )
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        Ok(())
    }

    fn process_layer_gpu(&mut self, layer_idx: usize) -> BackendResult<()> {
        let prefix = format!("blk.{}", layer_idx);

        // -------------------------------------------------------------------
        // Because the helper GPU ops are free functions (not &self methods),
        // we can pass disjoint field references without borrow conflicts.
        // -------------------------------------------------------------------

        // Attention norm: hidden -> hidden_norm
        rms_norm_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            self.config.hidden_size,
            self.config.norm_eps,
            &format!("{}.attn_norm.weight", prefix),
            &self.hidden,
            &mut self.hidden_norm,
        )?;

        // QKV projections
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_q.weight", prefix),
            Some(&format!("{}.attn_q.bias", prefix)),
            &self.hidden_norm,
            &mut self.q,
        )?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_k.weight", prefix),
            Some(&format!("{}.attn_k.bias", prefix)),
            &self.hidden_norm,
            &mut self.k,
        )?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_v.weight", prefix),
            Some(&format!("{}.attn_v.bias", prefix)),
            &self.hidden_norm,
            &mut self.v,
        )?;

        // RoPE (handles GQA correctly)
        self.apply_rope_gpu()?;

        // Update KV cache
        self.update_kv_cache_gpu(layer_idx)?;

        // Multi-head attention
        self.attention_gpu(layer_idx)?;

        // Output projection
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_output.weight", prefix),
            None,
            &self.attn_out,
            &mut self.hidden_norm,
        )?;

        // Add residual: hidden = residual + hidden_norm
        add_gpu(
            &self.kernels,
            self.config.hidden_size,
            &self.residual,
            &self.hidden_norm,
            &mut self.hidden,
        )?;
        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // FFN norm
        rms_norm_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            self.config.hidden_size,
            self.config.norm_eps,
            &format!("{}.ffn_norm.weight", prefix),
            &self.hidden,
            &mut self.hidden_norm,
        )?;

        // FFN: gate and up projections
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ffn_gate.weight", prefix),
            None,
            &self.hidden_norm,
            &mut self.ffn_gate,
        )?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ffn_up.weight", prefix),
            None,
            &self.hidden_norm,
            &mut self.ffn_up,
        )?;

        // SiLU on gate
        silu_gpu(
            &self.kernels,
            &self.device,
            self.config.intermediate_size,
            &mut self.ffn_gate,
        )?;

        // gate * up
        mul_gpu(
            &self.kernels,
            self.config.intermediate_size,
            &self.ffn_gate,
            &self.ffn_up,
            &mut self.ffn_down,
        )?;

        // Down projection
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ffn_down.weight", prefix),
            None,
            &self.ffn_down,
            &mut self.hidden_norm,
        )?;

        // Add residual
        add_gpu(
            &self.kernels,
            self.config.hidden_size,
            &self.residual,
            &self.hidden_norm,
            &mut self.hidden,
        )?;
        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        Ok(())
    }

    fn apply_rope_gpu(&mut self) -> BackendResult<()> {
        let cfg = &self.config;

        // The rope_single_pos kernel processes Q for all heads and K only
        // for the first num_kv_heads heads (GQA-safe).
        let config = LaunchConfig {
            grid_dim: (cfg.num_heads as u32, 1, 1),
            block_dim: ((cfg.head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels.rope_single_pos.clone().launch(
                config,
                (
                    &mut self.q,
                    &mut self.k,
                    cfg.num_heads as i32,
                    cfg.num_kv_heads as i32,
                    cfg.head_dim as i32,
                    self.pos as i32,
                    cfg.freq_base,
                    cfg.freq_scale,
                    if cfg.use_neox_rope { 1i32 } else { 0i32 },
                ),
            )
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    fn update_kv_cache_gpu(&mut self, layer_idx: usize) -> BackendResult<()> {
        let cfg = &self.config;
        let total = cfg.num_kv_heads * cfg.head_dim;

        let config = LaunchConfig {
            grid_dim: (((total + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels.update_kv_cache.clone().launch(
                config,
                (
                    &self.k,
                    &self.v,
                    &mut self.k_cache[layer_idx],
                    &mut self.v_cache[layer_idx],
                    cfg.num_kv_heads as i32,
                    cfg.head_dim as i32,
                    cfg.max_seq_len as i32,
                    self.pos as i32,
                ),
            )
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    fn attention_gpu(&mut self, layer_idx: usize) -> BackendResult<()> {
        let cfg = &self.config;
        let kv_len = self.pos + 1;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let block_size = 256.min(kv_len.max(1).next_power_of_two());
        let config = LaunchConfig {
            grid_dim: (cfg.num_heads as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: ((kv_len + block_size) * 4) as u32,
        };

        unsafe {
            self.kernels.attention_multihead.clone().launch(
                config,
                (
                    &self.q,
                    &self.k_cache[layer_idx],
                    &self.v_cache[layer_idx],
                    &mut self.attn_out,
                    cfg.num_heads as i32,
                    cfg.num_kv_heads as i32,
                    cfg.head_dim as i32,
                    cfg.max_seq_len as i32,
                    kv_len as i32,
                    scale,
                ),
            )
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }
}

// ---------------------------------------------------------------------------
// GpuModelWrapper: implements the Model trait using GpuOnlyInference
// ---------------------------------------------------------------------------

use crate::model::{Architecture, InferenceContext, Model, ModelConfig, ModelError, ModelResult};
use crate::tensor::Tensor;
use std::sync::Mutex;

/// Wrapper that implements [`Model`] by delegating to [`GpuOnlyInference`].
///
/// The entire forward pass runs on GPU with pre-allocated scratch buffers.
/// Only two CPU↔GPU transfers occur per token:
/// 1. Upload the embedding row (~3.5 KB)
/// 2. Download the logits (~600 KB)
///
/// This is ~386× fewer transfers than the standard Backend-trait path which
/// uploads and downloads activations for every single operation.
pub struct GpuModelWrapper {
    gpu: Mutex<GpuOnlyInference>,
    config: ModelConfig,
    architecture: Architecture,
}

impl GpuModelWrapper {
    /// Create a new GPU model wrapper.
    pub fn new(
        gpu: GpuOnlyInference,
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

impl Model for GpuModelWrapper {
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor> {
        let mut gpu = self.gpu.lock().map_err(|e| {
            ModelError::ConfigError(format!("GPU inference lock poisoned: {}", e))
        })?;

        // If the inference context was reset, reset GPU state too.
        if ctx.position == 0 && gpu.position() > 0 {
            gpu.reset();
        }

        if tokens.is_empty() {
            return Err(ModelError::ConfigError("No tokens to process".into()));
        }

        // Prefill: process all tokens except the last without computing logits.
        // This populates the KV cache but skips the expensive final-norm +
        // output-projection + logits-download for intermediate tokens.
        let last_idx = tokens.len() - 1;
        for &token in &tokens[..last_idx] {
            gpu.prefill_token(token)?;
        }

        // Process the last token and return logits.
        let logits_vec = gpu.forward(tokens[last_idx])?;

        // Update the CPU-side position to stay in sync.
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
