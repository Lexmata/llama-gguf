//! Full GPU-only inference with no intermediate CPU transfers
//!
//! All computation happens on GPU. Only embedding lookup at the start
//! and logits download at the end touch CPU memory.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::model::LlamaModel;

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
        })
    }

    pub fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> {
        // 1. Embed token (CPU->GPU, one-time per token)
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

        // 4. Final norm -- round-trip through a temporary to avoid &self + &mut self conflict
        {
            let hidden_clone: Vec<f32> = self
                .device
                .dtoh_sync_copy(&self.hidden)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
            let hidden_gpu = self
                .device
                .htod_sync_copy(&hidden_clone)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
            rms_norm_gpu(
                &self.kernels,
                &self.weights,
                &self.device,
                self.config.hidden_size,
                self.config.norm_eps,
                "output_norm.weight",
                &hidden_gpu,
                &mut self.hidden_norm,
            )?;
        }

        // 5. Output projection
        {
            let hidden_norm_clone: Vec<f32> = self
                .device
                .dtoh_sync_copy(&self.hidden_norm)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
            let hidden_norm_gpu = self
                .device
                .htod_sync_copy(&hidden_norm_clone)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
            linear_gpu(
                &self.kernels,
                &self.weights,
                &self.device,
                "output.weight",
                None,
                &hidden_norm_gpu,
                &mut self.logits,
            )?;
        }

        // 6. Download logits (GPU->CPU, one-time)
        let logits = self
            .device
            .dtoh_sync_copy(&self.logits)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        self.pos += 1;
        Ok(logits)
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

        let emb = self
            .weights
            .get("token_embd.weight")
            .ok_or_else(|| BackendError::OperationFailed("Missing token embedding".into()))?;

        // Download embedding, then upload the slice we need
        // TODO: Use a kernel to copy directly on GPU
        let emb_data: Vec<f32> = self
            .device
            .dtoh_sync_copy(&emb.data)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        self.device
            .htod_sync_copy_into(&emb_data[offset..offset + hidden_size], &mut self.hidden)
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

        // RoPE
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
        let config = LaunchConfig {
            grid_dim: (cfg.num_heads as u32, 1, 1),
            block_dim: ((cfg.head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Note: This kernel expects same num_heads for Q and K
        // For GQA, we need separate launches
        unsafe {
            self.kernels.rope_single_pos.clone().launch(
                config,
                (
                    &mut self.q,
                    &mut self.k,
                    cfg.num_heads as i32,
                    cfg.head_dim as i32,
                    self.pos as i32,
                    cfg.freq_base,
                    cfg.freq_scale,
                    0i32,
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

        let config = LaunchConfig {
            grid_dim: (cfg.num_heads as u32, 1, 1),
            block_dim: (256.min(kv_len) as u32, 1, 1),
            shared_mem_bytes: (kv_len * 4) as u32,
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
