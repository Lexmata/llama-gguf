//! GPU-accelerated inference engine for Qwen3Next and similar architectures.
//!
//! - Attention layers: fully on GPU (QKV projections, QK norm, partial/full
//!   RoPE, GPU KV cache, attention, optional gating, output projection)
//! - DeltaNet layers: fully on GPU (matmul projections + fused CUDA kernels)
//! - MoE layers: router on GPU, active expert weights streamed from CPU,
//!   SwiGLU computed on GPU, weighted accumulation on GPU
//! - Dense FFN: fully on GPU

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::cpu::simd;
use crate::backend::cpu::CpuBackend;
use crate::backend::{BackendError, BackendResult};
use crate::model::deltanet::DeltaNetConfig;
use crate::model::layers::{AttentionLayer, FfnLayer, TransformerLayer};
use crate::model::LlamaModel;
use crate::tensor::{DType, Tensor};

use super::dequant_weights::GpuWeightStore;
use super::kernels::CudaKernels;

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
    ffn_gate: CudaSlice<f32>,
    ffn_up: CudaSlice<f32>,
    ffn_down: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    // CPU copy of dequantized embeddings
    cpu_embeddings: Vec<f32>,
    // Owned model layers (for accessing MoE expert weights and CPU attention at inference time)
    layers: Vec<TransformerLayer>,
    // CPU backend for running attention forward passes
    cpu_backend: CpuBackend,
    // CPU KV caches (kept as fallback for unsupported attention configs)
    cpu_k_caches: Vec<Option<Tensor>>,
    cpu_v_caches: Vec<Option<Tensor>>,
    // GPU KV caches [num_kv_heads * max_seq_len * key_length]
    gpu_k_caches: Vec<Option<CudaSlice<f32>>>,
    gpu_v_caches: Vec<Option<CudaSlice<f32>>>,
    // GPU attention scratch buffers
    attn_q_raw: CudaSlice<f32>,
    attn_k: CudaSlice<f32>,
    attn_v: CudaSlice<f32>,
    attn_q_proper: CudaSlice<f32>,
    attn_gate: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    // DeltaNet config
    deltanet_config: Option<DeltaNetConfig>,
    // DeltaNet GPU state buffers (conv_state + ssm_state per DeltaNet layer)
    dn_conv_states: Vec<Option<CudaSlice<f32>>>,
    dn_ssm_states: Vec<Option<CudaSlice<f32>>>,
    // DeltaNet scratch buffers
    dn_qkv: Option<CudaSlice<f32>>,
    dn_gate_z: Option<CudaSlice<f32>>,
    dn_ba: Option<CudaSlice<f32>>,
    dn_conv_out: Option<CudaSlice<f32>>,
    dn_recurrent_out: Option<CudaSlice<f32>>,
    // DeltaNet config buffer on GPU: [num_v_heads, num_k_heads, head_v_dim, head_k_dim, kv_ratio, d_inner, qkv_dim]
    dn_config_gpu: Option<CudaSlice<i32>>,
    // Whether DeltaNet uses separate beta/alpha projections (Qwen3.5) vs combined
    dn_ba_separate: bool,
    dn_beta_tmp: Option<CudaSlice<f32>>,
    dn_alpha_tmp: Option<CudaSlice<f32>>,
    // MoE scratch buffers
    moe_hidden: CudaSlice<f32>,
    moe_expert_out: CudaSlice<f32>,
    moe_expert_gate: CudaSlice<f32>,
    moe_expert_up: CudaSlice<f32>,
    moe_expert_down: CudaSlice<f32>,
    // Per-layer flag: true if this layer has full attention
    has_gpu_attention: Vec<bool>,
    // Per-layer flag: true if this layer is DeltaNet
    is_deltanet: Vec<bool>,
}

#[derive(Clone)]
struct InferenceConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    norm_eps: f32,
    freq_base: f32,
    freq_scale: f32,
    expert_intermediate: usize,
}

// ---------------------------------------------------------------------------
// Free-standing GPU kernel helpers
// ---------------------------------------------------------------------------

fn rms_norm_gpu(
    kernels: &CudaKernels,
    weights: &GpuWeightStore,
    _device: &Arc<CudaDevice>,
    hidden_size: usize,
    norm_eps: f32,
    weight_name: &str,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> BackendResult<()> {
    let weight = weights
        .get(weight_name)
        .ok_or_else(|| BackendError::OperationFailed(format!("Missing {}", weight_name)))?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    unsafe {
        kernels
            .rms_norm_fused
            .clone()
            .launch(
                config,
                (x, &weight.data, out, norm_eps, hidden_size as i32),
            )
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
    if let Some(qw) = weights.get_quantized(weight_name) {
        let k = qw.shape[0];
        let n = if qw.shape.len() >= 2 { qw.shape[1] } else { 1 };

        let config = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        match qw.dtype {
            DType::Q4K => unsafe {
                kernels
                    .vec_mat_q4k
                    .clone()
                    .launch(config, (&qw.data, x, &mut *out, k as i32, n as i32))
            },
            DType::Q6K => unsafe {
                kernels
                    .vec_mat_q6k
                    .clone()
                    .launch(config, (&qw.data, x, &mut *out, k as i32, n as i32))
            },
            DType::Q5K => unsafe {
                kernels
                    .vec_mat_q5k
                    .clone()
                    .launch(config, (&qw.data, x, &mut *out, k as i32, n as i32))
            },
            DType::Q4_0 => unsafe {
                kernels
                    .vec_mat_q4_0
                    .clone()
                    .launch(config, (&qw.data, x, &mut *out, k as i32, n as i32))
            },
            DType::Q8_0 => unsafe {
                kernels
                    .vec_mat_q8_0
                    .clone()
                    .launch(config, (&qw.data, x, &mut *out, k as i32, n as i32))
            },
            _ => {
                return Err(BackendError::OperationFailed(format!(
                    "No GPU kernel for {:?} ({})",
                    qw.dtype, weight_name
                )));
            }
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

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

        return Ok(());
    }

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
    n: usize,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> BackendResult<()> {
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
    n: usize,
    x: &mut CudaSlice<f32>,
) -> BackendResult<()> {
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

fn scaled_add_gpu(
    kernels: &CudaKernels,
    n: usize,
    out: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    scale: f32,
) -> BackendResult<()> {
    let config = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        kernels
            .scaled_add_f32
            .clone()
            .launch(config, (&mut *out, x, scale, n as i32))
    }
    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

// ---------------------------------------------------------------------------
// GpuOnlyInference implementation
// ---------------------------------------------------------------------------

impl GpuOnlyInference {
    pub fn from_model(model: LlamaModel, max_seq_len: usize) -> BackendResult<Self> {
        let (
            model_config,
            token_embedding,
            layers,
            norm,
            output,
            _architecture,
            _recurrent_mask,
            deltanet_config,
        ) = model.into_parts();

        let device = Arc::new(
            CudaDevice::new(0).map_err(|e| BackendError::InitializationFailed(format!("{}", e)))?,
        );

        eprintln!("Initializing full GPU inference...");

        let kernels = CudaKernels::new(Arc::clone(&device))?;

        let weights = super::dequant_weights::upload_model_weights(
            Arc::clone(&device),
            &layers,
            &token_embedding,
            &output,
            &norm,
        )?;

        let has_gpu_attention: Vec<bool> = layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                layer.attention().is_some()
                    && weights.contains(&format!("blk.{}.attn_q.weight", i))
            })
            .collect();

        let is_deltanet: Vec<bool> = layers
            .iter()
            .map(|l| matches!(&l.attn_layer, AttentionLayer::DeltaNet(_)))
            .collect();

        // Determine expert intermediate size from first MoE layer
        let expert_intermediate = layers
            .iter()
            .find_map(|l| l.moe())
            .map(|m| m.experts.first().map(|e| e.gate_proj.shape()[1]).unwrap_or(512))
            .unwrap_or(model_config.intermediate_size);

        let config = InferenceConfig {
            hidden_size: model_config.hidden_size,
            intermediate_size: model_config.intermediate_size,
            num_layers: model_config.num_layers,
            norm_eps: model_config.norm_eps,
            freq_base: model_config.rope_config.freq_base,
            freq_scale: model_config.rope_config.freq_scale,
            expert_intermediate,
        };

        // CPU embeddings table for token lookup
        let cpu_embeddings = if token_embedding.dtype() == DType::F32 {
            token_embedding
                .as_f32()
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?
                .to_vec()
        } else {
            let numel = token_embedding.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            crate::backend::cpu::ops::dequantize(&token_embedding, &mut dequant)?;
            dequant
                .as_f32()
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?
                .to_vec()
        };

        // Allocate GPU scratch buffers
        let alloc = |size: usize| -> BackendResult<CudaSlice<f32>> {
            device
                .alloc_zeros(size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))
        };

        let hidden = alloc(model_config.hidden_size)?;
        let hidden_norm = alloc(model_config.hidden_size)?;
        let residual = alloc(model_config.hidden_size)?;
        let ffn_gate = alloc(model_config.intermediate_size)?;
        let ffn_up = alloc(model_config.intermediate_size)?;
        let ffn_down = alloc(model_config.hidden_size)?;
        let logits = alloc(model_config.vocab_size)?;

        // MoE scratch buffers
        let moe_hidden = alloc(model_config.hidden_size)?;
        let moe_expert_out = alloc(model_config.hidden_size)?;
        let moe_expert_gate = alloc(expert_intermediate)?;
        let moe_expert_up = alloc(expert_intermediate)?;
        let moe_expert_down = alloc(model_config.hidden_size)?;

        // KV caches — GPU primary, CPU fallback
        let cpu_backend = CpuBackend::new();
        let mut cpu_k_caches: Vec<Option<Tensor>> = Vec::with_capacity(model_config.num_layers);
        let mut cpu_v_caches: Vec<Option<Tensor>> = Vec::with_capacity(model_config.num_layers);
        let mut gpu_k_caches: Vec<Option<CudaSlice<f32>>> =
            Vec::with_capacity(model_config.num_layers);
        let mut gpu_v_caches: Vec<Option<CudaSlice<f32>>> =
            Vec::with_capacity(model_config.num_layers);

        let mut max_q_out = 0usize;
        let mut max_kv_flat = 0usize;
        let mut max_attn_flat = 0usize;

        for i in 0..model_config.num_layers {
            if has_gpu_attention[i] {
                if let Some(attn) = layers[i].attention() {
                    let kl = attn.key_length;
                    let vl = attn.value_length;
                    let kv_size = model_config.num_kv_heads * max_seq_len * kl;
                    let kv_v_size = model_config.num_kv_heads * max_seq_len * vl;

                    cpu_k_caches.push(Some(Tensor::zeros(
                        vec![model_config.num_kv_heads, max_seq_len, kl],
                        DType::F32,
                    )));
                    cpu_v_caches.push(Some(Tensor::zeros(
                        vec![model_config.num_kv_heads, max_seq_len, vl],
                        DType::F32,
                    )));
                    gpu_k_caches.push(Some(alloc(kv_size)?));
                    gpu_v_caches.push(Some(alloc(kv_v_size)?));

                    max_q_out = max_q_out.max(attn.wq.out_features);
                    max_kv_flat = max_kv_flat.max(model_config.num_kv_heads * kl);
                    max_kv_flat = max_kv_flat.max(model_config.num_kv_heads * vl);
                    max_attn_flat =
                        max_attn_flat.max(model_config.num_heads * vl.max(kl));
                } else {
                    cpu_k_caches.push(None);
                    cpu_v_caches.push(None);
                    gpu_k_caches.push(None);
                    gpu_v_caches.push(None);
                }
            } else {
                cpu_k_caches.push(None);
                cpu_v_caches.push(None);
                gpu_k_caches.push(None);
                gpu_v_caches.push(None);
            }
        }

        let attn_q_raw = alloc(max_q_out.max(1))?;
        let attn_k = alloc(max_kv_flat.max(1))?;
        let attn_v = alloc(max_kv_flat.max(1))?;
        let attn_q_proper = alloc(max_attn_flat.max(1))?;
        let attn_gate = alloc(max_attn_flat.max(1))?;
        let attn_out = alloc(max_attn_flat.max(1))?;

        // DeltaNet GPU state buffers
        let mut dn_conv_states = Vec::with_capacity(model_config.num_layers);
        let mut dn_ssm_states = Vec::with_capacity(model_config.num_layers);

        let mut dn_qkv_buf = None;
        let mut dn_gate_z_buf = None;
        let mut dn_ba_buf = None;
        let mut dn_conv_out_buf = None;
        let mut dn_recurrent_out_buf = None;
        let mut dn_config_gpu_buf = None;
        let mut dn_ba_separate = false;
        let mut dn_beta_tmp_buf = None;
        let mut dn_alpha_tmp_buf = None;

        if let Some(ref dn_cfg) = deltanet_config {
            for i in 0..model_config.num_layers {
                if is_deltanet[i] {
                    let conv_len = (dn_cfg.conv_kernel - 1) * dn_cfg.qkv_dim;
                    let ssm_len = dn_cfg.num_v_heads * dn_cfg.head_v_dim * dn_cfg.head_k_dim;
                    dn_conv_states.push(Some(alloc(conv_len)?));
                    dn_ssm_states.push(Some(alloc(ssm_len)?));
                } else {
                    dn_conv_states.push(None);
                    dn_ssm_states.push(None);
                }
            }
            dn_qkv_buf = Some(alloc(dn_cfg.qkv_dim)?);
            dn_gate_z_buf = Some(alloc(dn_cfg.d_inner)?);
            let ba_size = dn_cfg.num_k_heads * 2 * (dn_cfg.num_v_heads / dn_cfg.num_k_heads.max(1));
            dn_ba_buf = Some(alloc(ba_size)?);

            dn_ba_separate = weights.contains("blk.0.ssm_beta.weight");
            if dn_ba_separate {
                dn_beta_tmp_buf = Some(alloc(dn_cfg.num_v_heads)?);
                dn_alpha_tmp_buf = Some(alloc(dn_cfg.num_v_heads)?);
            }

            dn_conv_out_buf = Some(alloc(dn_cfg.qkv_dim)?);
            dn_recurrent_out_buf = Some(alloc(dn_cfg.d_inner)?);

            // Upload DeltaNet config as GPU i32 buffer
            let kv_ratio = dn_cfg.num_v_heads / dn_cfg.num_k_heads.max(1);
            let cfg_data = vec![
                dn_cfg.num_v_heads as i32,
                dn_cfg.num_k_heads as i32,
                dn_cfg.head_v_dim as i32,
                dn_cfg.head_k_dim as i32,
                kv_ratio as i32,
                dn_cfg.d_inner as i32,
                dn_cfg.qkv_dim as i32,
            ];
            dn_config_gpu_buf = Some(
                device
                    .htod_sync_copy(&cfg_data)
                    .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            );
        } else {
            for _ in 0..model_config.num_layers {
                dn_conv_states.push(None);
                dn_ssm_states.push(None);
            }
        }

        let attn_count = has_gpu_attention.iter().filter(|&&x| x).count();
        let dn_count = is_deltanet.iter().filter(|&&x| x).count();
        let moe_count = layers.iter().filter(|l| l.moe().is_some()).count();
        let vram_mb = weights.vram_usage() as f64 / (1024.0 * 1024.0);
        eprintln!(
            "Full GPU inference ready: {:.1} MB VRAM, {} attn + {} DeltaNet + {} MoE layers",
            vram_mb, attn_count, dn_count, moe_count,
        );

        Ok(Self {
            device: Arc::clone(&device),
            kernels,
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
            cpu_embeddings,
            layers,
            cpu_backend,
            cpu_k_caches,
            cpu_v_caches,
            gpu_k_caches,
            gpu_v_caches,
            attn_q_raw,
            attn_k,
            attn_v,
            attn_q_proper,
            attn_gate,
            attn_out,
            deltanet_config,
            dn_conv_states,
            dn_ssm_states,
            dn_qkv: dn_qkv_buf,
            dn_gate_z: dn_gate_z_buf,
            dn_ba: dn_ba_buf,
            dn_conv_out: dn_conv_out_buf,
            dn_recurrent_out: dn_recurrent_out_buf,
            dn_config_gpu: dn_config_gpu_buf,
            dn_ba_separate,
            dn_beta_tmp: dn_beta_tmp_buf,
            dn_alpha_tmp: dn_alpha_tmp_buf,
            moe_hidden,
            moe_expert_out,
            moe_expert_gate,
            moe_expert_up,
            moe_expert_down,
            has_gpu_attention,
            is_deltanet,
        })
    }

    pub fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> {
        self.embed_token(token_id)?;

        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let num_layers = self.config.num_layers;
        for layer_idx in 0..num_layers {
            self.process_layer(layer_idx)?;
        }

        // Final norm
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

        // Output projection
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            "output.weight",
            None,
            &self.hidden_norm,
            &mut self.logits,
        )?;

        let logits = self
            .device
            .dtoh_sync_copy(&self.logits)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        self.pos += 1;
        Ok(logits)
    }

    /// Process a batch of tokens during prefill. Returns logits for the last token.
    /// Each token is processed sequentially through all layers but the batch
    /// enables future optimization with batched attention during prefill.
    pub fn forward_batch(&mut self, token_ids: &[u32]) -> BackendResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(BackendError::InvalidArgument(
                "Empty token batch".to_string(),
            ));
        }

        // Process all tokens except the last as prefill
        for &tid in &token_ids[..token_ids.len() - 1] {
            self.prefill_token(tid)?;
        }

        // Process the last token and return logits
        self.forward(*token_ids.last().unwrap())
    }

    pub fn prefill_token(&mut self, token_id: u32) -> BackendResult<()> {
        self.embed_token(token_id)?;

        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let num_layers = self.config.num_layers;
        for layer_idx in 0..num_layers {
            self.process_layer(layer_idx)?;
        }

        self.pos += 1;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        // Zero DeltaNet states
        for cs in self.dn_conv_states.iter_mut().flatten() {
            let len = cs.len();
            let _ = self.device.htod_sync_copy_into(&vec![0.0f32; len], cs);
        }
        for ss in self.dn_ssm_states.iter_mut().flatten() {
            let len = ss.len();
            let _ = self.device.htod_sync_copy_into(&vec![0.0f32; len], ss);
        }
        // Zero CPU KV caches
        for kc in self.cpu_k_caches.iter_mut().flatten() {
            if let Ok(data) = kc.as_f32_mut() {
                data.fill(0.0);
            }
        }
        for vc in self.cpu_v_caches.iter_mut().flatten() {
            if let Ok(data) = vc.as_f32_mut() {
                data.fill(0.0);
            }
        }
        // Zero GPU KV caches
        for kc in self.gpu_k_caches.iter_mut().flatten() {
            let len = kc.len();
            let _ = self.device.htod_sync_copy_into(&vec![0.0f32; len], kc);
        }
        for vc in self.gpu_v_caches.iter_mut().flatten() {
            let len = vc.len();
            let _ = self.device.htod_sync_copy_into(&vec![0.0f32; len], vc);
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    fn embed_token(&mut self, token_id: u32) -> BackendResult<()> {
        let hidden_size = self.config.hidden_size;
        let offset = token_id as usize * hidden_size;
        self.device
            .htod_sync_copy_into(
                &self.cpu_embeddings[offset..offset + hidden_size],
                &mut self.hidden,
            )
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    fn process_layer(&mut self, layer_idx: usize) -> BackendResult<()> {
        let prefix = format!("blk.{}", layer_idx);
        let hidden_size = self.config.hidden_size;

        // ---- Attention normalization (GPU) ----
        rms_norm_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            hidden_size,
            self.config.norm_eps,
            &format!("{}.attn_norm.weight", prefix),
            &self.hidden,
            &mut self.hidden_norm,
        )?;

        // ---- Attention (GPU) / DeltaNet (GPU) ----
        if self.has_gpu_attention[layer_idx] {
            self.attention_gpu_forward(layer_idx)?;
        } else if self.is_deltanet[layer_idx] {
            self.deltanet_gpu_forward(layer_idx, &prefix)?;
        } else {
            return Err(BackendError::OperationFailed(format!(
                "Layer {} has no GPU attention or DeltaNet path",
                layer_idx
            )));
        }

        // Add attention residual: hidden = residual + attn_output
        add_gpu(
            &self.kernels,
            hidden_size,
            &self.residual,
            &self.hidden_norm,
            &mut self.hidden,
        )?;
        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // ---- FFN normalization ----
        let ffn_norm_name = if self.layers[layer_idx].post_attn_norm.is_some() {
            format!("{}.post_attention_norm.weight", prefix)
        } else {
            format!("{}.ffn_norm.weight", prefix)
        };
        rms_norm_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            hidden_size,
            self.config.norm_eps,
            &ffn_norm_name,
            &self.hidden,
            &mut self.hidden_norm,
        )?;

        // ---- FFN (Dense GPU / MoE GPU) ----
        match &self.layers[layer_idx].ffn_layer {
            FfnLayer::Dense(_) => {
                self.dense_ffn_gpu_forward(&prefix)?;
            }
            FfnLayer::Moe(_) => {
                self.moe_gpu_forward(layer_idx)?;
            }
        }

        // Add FFN residual
        add_gpu(
            &self.kernels,
            hidden_size,
            &self.residual,
            &self.hidden_norm,
            &mut self.hidden,
        )?;
        self.device
            .dtod_copy(&self.hidden, &mut self.residual)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Attention (fully on GPU: QKV projections, QK norm, partial/full RoPE,
    // KV cache update, attention, optional gating, output projection)
    // -----------------------------------------------------------------------

    fn attention_gpu_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let attn = self.layers[layer_idx]
            .attention()
            .ok_or_else(|| BackendError::OperationFailed("Expected attention layer".into()))?;

        let prefix = format!("blk.{}", layer_idx);
        let num_heads = attn.num_heads;
        let num_kv_heads = attn.num_kv_heads;
        let kl = attn.key_length;
        let vl = attn.value_length;
        let rope_dims = attn.rope_dims;
        let scale = attn.scale;
        let use_neox = attn.use_neox_rope;
        let has_gate = attn.has_attention_gate;
        let has_q_norm = attn.q_norm.is_some();
        let has_k_norm = attn.k_norm.is_some();
        let norm_eps = self.config.norm_eps;
        let freq_base = self.config.freq_base;
        let freq_scale = self.config.freq_scale;
        let pos = self.pos;

        // 1. Q/K/V projections on GPU
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_q.weight", prefix),
            None,
            &self.hidden_norm,
            &mut self.attn_q_raw,
        )?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_k.weight", prefix),
            None,
            &self.hidden_norm,
            &mut self.attn_k,
        )?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_v.weight", prefix),
            None,
            &self.hidden_norm,
            &mut self.attn_v,
        )?;

        // 2. Split Q into q_proper and gate if needed
        if has_gate {
            let config = LaunchConfig {
                grid_dim: (num_heads as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels.split_q_gate.clone().launch(
                    config,
                    (
                        &self.attn_q_raw,
                        &mut self.attn_q_proper,
                        &mut self.attn_gate,
                        num_heads as i32,
                        kl as i32,
                        vl as i32,
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("split_q_gate failed: {}", e))
            })?;
        } else {
            self.device
                .dtod_copy(&self.attn_q_raw, &mut self.attn_q_proper)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        }

        // 3. QK normalization (per-head RMS norm)
        if has_q_norm {
            let q_norm = self.layers[layer_idx]
                .attention()
                .and_then(|a| a.q_norm.as_ref())
                .ok_or_else(|| BackendError::OperationFailed("Missing q_norm".into()))?;
            let norm_w = q_norm.weight.as_f32().map_err(|e| {
                BackendError::OperationFailed(format!("{}", e))
            })?;
            let norm_dim = norm_w.len();
            let norm_gpu = self
                .device
                .htod_sync_copy(norm_w)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

            let block_size = 256.min(norm_dim.next_power_of_two());
            let config = LaunchConfig {
                grid_dim: (num_heads as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: (block_size * 4) as u32,
            };
            unsafe {
                self.kernels.qk_norm_per_head.clone().launch(
                    config,
                    (
                        &mut self.attn_q_proper,
                        &norm_gpu,
                        norm_eps,
                        num_heads as i32,
                        kl as i32,
                        norm_dim as i32,
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("qk_norm Q failed: {}", e))
            })?;
        }
        if has_k_norm {
            let k_norm = self.layers[layer_idx]
                .attention()
                .and_then(|a| a.k_norm.as_ref())
                .ok_or_else(|| BackendError::OperationFailed("Missing k_norm".into()))?;
            let norm_w = k_norm.weight.as_f32().map_err(|e| {
                BackendError::OperationFailed(format!("{}", e))
            })?;
            let norm_dim = norm_w.len();
            let norm_gpu = self
                .device
                .htod_sync_copy(norm_w)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

            let block_size = 256.min(norm_dim.next_power_of_two());
            let config = LaunchConfig {
                grid_dim: (num_kv_heads as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: (block_size * 4) as u32,
            };
            unsafe {
                self.kernels.qk_norm_per_head.clone().launch(
                    config,
                    (
                        &mut self.attn_k,
                        &norm_gpu,
                        norm_eps,
                        num_kv_heads as i32,
                        kl as i32,
                        norm_dim as i32,
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("qk_norm K failed: {}", e))
            })?;
        }

        // 4. RoPE (full or partial)
        if rope_dims > 0 && rope_dims < kl {
            let total_blocks = (num_heads + num_kv_heads) as u32;
            let half_rope = (rope_dims / 2) as u32;
            let config = LaunchConfig {
                grid_dim: (total_blocks, 1, 1),
                block_dim: (half_rope.min(256), 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels.partial_rope.clone().launch(
                    config,
                    (
                        &mut self.attn_q_proper,
                        &mut self.attn_k,
                        num_heads as i32,
                        num_kv_heads as i32,
                        kl as i32,
                        rope_dims as i32,
                        pos as i32,
                        freq_base,
                        freq_scale,
                        if use_neox { 1i32 } else { 0i32 },
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("partial_rope failed: {}", e))
            })?;
        } else {
            let max_heads = num_heads.max(num_kv_heads) as u32;
            let half_dim = (kl / 2) as u32;
            let config = LaunchConfig {
                grid_dim: (max_heads, 1, 1),
                block_dim: (half_dim.min(256), 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels.rope_single_pos.clone().launch(
                    config,
                    (
                        &mut self.attn_q_proper,
                        &mut self.attn_k,
                        num_heads as i32,
                        num_kv_heads as i32,
                        kl as i32,
                        pos as i32,
                        freq_base,
                        freq_scale,
                        if use_neox { 1i32 } else { 0i32 },
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("rope_single_pos failed: {}", e))
            })?;
        }

        // 5. Update GPU KV cache
        let gpu_k_cache = self.gpu_k_caches[layer_idx]
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("Missing GPU K cache".into()))?;
        let gpu_v_cache = self.gpu_v_caches[layer_idx]
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("Missing GPU V cache".into()))?;

        let max_seq_len = gpu_k_cache.len() / (num_kv_heads * kl);
        let total_kv = num_kv_heads * kl;
        let block_size = 256.min(total_kv.next_power_of_two());
        let grid = ((total_kv + block_size - 1) / block_size) as u32;
        let config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.kernels.update_kv_cache.clone().launch(
                config,
                (
                    &self.attn_k,
                    &self.attn_v,
                    gpu_k_cache as &mut CudaSlice<f32>,
                    gpu_v_cache as &mut CudaSlice<f32>,
                    num_kv_heads as i32,
                    kl as i32,
                    max_seq_len as i32,
                    pos as i32,
                ),
            )
        }
        .map_err(|e| {
            BackendError::OperationFailed(format!("update_kv_cache failed: {}", e))
        })?;

        // When vl != kl, we need a separate V cache update since the K update
        // above used kl as head_dim. Re-update V with the correct vl stride.
        if vl != kl {
            let total_kv_v = num_kv_heads * vl;
            let bs2 = 256.min(total_kv_v.next_power_of_two());
            let grid2 = ((total_kv_v + bs2 - 1) / bs2) as u32;
            let config2 = LaunchConfig {
                grid_dim: (grid2, 1, 1),
                block_dim: (bs2 as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let max_v_seq = self.gpu_v_caches[layer_idx]
                .as_ref()
                .map(|vc| vc.len() / (num_kv_heads * vl))
                .unwrap_or(0);
            let gpu_k_ref = self.gpu_k_caches[layer_idx].as_mut().unwrap();
            let gpu_v_ref = self.gpu_v_caches[layer_idx].as_mut().unwrap();
            unsafe {
                self.kernels.update_kv_cache.clone().launch(
                    config2,
                    (
                        &self.attn_v,
                        &self.attn_v,
                        gpu_k_ref as &mut CudaSlice<f32>,
                        gpu_v_ref as &mut CudaSlice<f32>,
                        num_kv_heads as i32,
                        vl as i32,
                        max_v_seq as i32,
                        pos as i32,
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("update_kv_cache V failed: {}", e))
            })?;
        }

        // 6. Flash Attention — O(head_dim) shared memory, supports any kv_len
        let kv_len = pos + 1;
        let block_attn = 256u32;
        let shared_bytes = (kl + 256 + 4) * 4;
        let config_attn = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: (block_attn, 1, 1),
            shared_mem_bytes: shared_bytes as u32,
        };
        unsafe {
            self.kernels.flash_attention_cached.clone().launch(
                config_attn,
                (
                    &self.attn_q_proper,
                    &*self.gpu_k_caches[layer_idx].as_ref().unwrap(),
                    &*self.gpu_v_caches[layer_idx].as_ref().unwrap(),
                    &mut self.attn_out,
                    num_heads as i32,
                    num_kv_heads as i32,
                    kl as i32,
                    max_seq_len as i32,
                    kv_len as i32,
                    scale,
                ),
            )
        }
        .map_err(|e| {
            BackendError::OperationFailed(format!("flash_attention_cached failed: {}", e))
        })?;

        // 7. Apply attention gate: out = sigmoid(gate) * attn_out
        if has_gate {
            let total = num_heads * vl;
            let bs = 256;
            let grid_gate = ((total + bs - 1) / bs) as u32;
            let config_gate = LaunchConfig {
                grid_dim: (grid_gate, 1, 1),
                block_dim: (bs as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels.attention_gate_sigmoid.clone().launch(
                    config_gate,
                    (
                        &self.attn_gate,
                        &self.attn_out,
                        &mut self.attn_q_raw,
                        total as i32,
                    ),
                )
            }
            .map_err(|e| {
                BackendError::OperationFailed(format!("attention_gate failed: {}", e))
            })?;
            let src = self.attn_q_raw.try_slice(..total).ok_or_else(|| {
                BackendError::OperationFailed("gate slice out of bounds".into())
            })?;
            let mut dst = self.attn_out.try_slice_mut(..total).ok_or_else(|| {
                BackendError::OperationFailed("gate slice_mut out of bounds".into())
            })?;
            self.device
                .dtod_copy(&src, &mut dst)
                .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        }

        // 8. Output projection: hidden_norm = wo @ attn_out
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_output.weight", prefix),
            None,
            &self.attn_out,
            &mut self.hidden_norm,
        )?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // DeltaNet (full GPU: projections + fused kernels)
    // -----------------------------------------------------------------------

    fn deltanet_gpu_forward(&mut self, layer_idx: usize, prefix: &str) -> BackendResult<()> {
        let dn_cfg = self
            .deltanet_config
            .as_ref()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet config".into()))?
            .clone();

        // 1. QKV projection on GPU
        let dn_qkv = self
            .dn_qkv
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet QKV buffer".into()))?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_qkv.weight", prefix),
            None,
            &self.hidden_norm,
            dn_qkv,
        )?;

        // 2. Gate projection on GPU
        let dn_gate_z = self
            .dn_gate_z
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet gate buffer".into()))?;
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.attn_gate.weight", prefix),
            None,
            &self.hidden_norm,
            dn_gate_z,
        )?;

        // 3. Beta/Alpha projection on GPU
        let dn_ba = self
            .dn_ba
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet ba buffer".into()))?;

        if self.dn_ba_separate {
            let dn_beta = self.dn_beta_tmp.as_mut().ok_or_else(|| {
                BackendError::OperationFailed("No DeltaNet beta buffer".into())
            })?;
            let dn_alpha = self.dn_alpha_tmp.as_mut().ok_or_else(|| {
                BackendError::OperationFailed("No DeltaNet alpha buffer".into())
            })?;
            linear_gpu(
                &self.kernels, &self.weights, &self.device,
                &format!("{}.ssm_beta.weight", prefix), None,
                &self.hidden_norm, dn_beta,
            )?;
            linear_gpu(
                &self.kernels, &self.weights, &self.device,
                &format!("{}.ssm_alpha.weight", prefix), None,
                &self.hidden_norm, dn_alpha,
            )?;

            let beta_cpu = self.device.dtoh_sync_copy(dn_beta)
                .map_err(|e| BackendError::OperationFailed(format!("beta download: {}", e)))?;
            let alpha_cpu = self.device.dtoh_sync_copy(dn_alpha)
                .map_err(|e| BackendError::OperationFailed(format!("alpha download: {}", e)))?;

            let kv_ratio = dn_cfg.num_v_heads / dn_cfg.num_k_heads.max(1);
            let mut ba_combined = vec![0.0f32; dn_cfg.num_k_heads * 2 * kv_ratio];
            for kh in 0..dn_cfg.num_k_heads {
                for r in 0..kv_ratio {
                    let vh = kh * kv_ratio + r;
                    let group_offset = kh * 2 * kv_ratio;
                    ba_combined[group_offset + r] = beta_cpu[vh];
                    ba_combined[group_offset + kv_ratio + r] = alpha_cpu[vh];
                }
            }

            self.device.htod_sync_copy_into(&ba_combined, dn_ba)
                .map_err(|e| BackendError::OperationFailed(format!("ba upload: {}", e)))?;
        } else {
            linear_gpu(
                &self.kernels, &self.weights, &self.device,
                &format!("{}.ssm_ba.weight", prefix), None,
                &self.hidden_norm, dn_ba,
            )?;
        }

        // 4. Conv1d + SiLU on GPU
        let conv_state = self.dn_conv_states[layer_idx]
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet conv state".into()))?;

        let conv_w = self
            .weights
            .get(&format!("{}.ssm_conv1d.weight", prefix))
            .ok_or_else(|| {
                BackendError::OperationFailed("Missing ssm_conv1d.weight".into())
            })?;

        let dn_conv_out = self
            .dn_conv_out
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet conv_out buffer".into()))?;

        {
            let channels = dn_cfg.qkv_dim;
            let config = LaunchConfig {
                grid_dim: (((channels + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels.deltanet_conv1d_silu.clone().launch(
                    config,
                    (
                        &mut *conv_state,
                        dn_qkv as &CudaSlice<f32>,
                        &conv_w.data,
                        &mut *dn_conv_out,
                        channels as i32,
                        dn_cfg.conv_kernel as i32,
                    ),
                )
            }
            .map_err(|e| BackendError::OperationFailed(format!("conv1d_silu: {}", e)))?;
        }

        // 5. Recurrent state update on GPU
        let ssm_state = self.dn_ssm_states[layer_idx]
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet ssm state".into()))?;

        let ssm_a = self
            .weights
            .get(&format!("{}.ssm_a", prefix))
            .ok_or_else(|| BackendError::OperationFailed("Missing ssm_a".into()))?;

        let dt_bias = self
            .weights
            .get(&format!("{}.ssm_dt.bias", prefix))
            .ok_or_else(|| BackendError::OperationFailed("Missing ssm_dt.bias".into()))?;

        let norm_w = self
            .weights
            .get(&format!("{}.ssm_norm.weight", prefix))
            .ok_or_else(|| BackendError::OperationFailed("Missing ssm_norm.weight".into()))?;

        let dn_recurrent_out = self
            .dn_recurrent_out
            .as_mut()
            .ok_or_else(|| {
                BackendError::OperationFailed("No DeltaNet recurrent_out buffer".into())
            })?;

        let dn_config_gpu = self
            .dn_config_gpu
            .as_ref()
            .ok_or_else(|| BackendError::OperationFailed("No DeltaNet config buffer".into()))?;

        {
            let config = LaunchConfig {
                grid_dim: (dn_cfg.num_v_heads as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels.deltanet_recurrent.clone().launch(
                    config,
                    (
                        &mut *ssm_state,
                        dn_conv_out as &CudaSlice<f32>,
                        dn_gate_z as &CudaSlice<f32>,
                        dn_ba as &CudaSlice<f32>,
                        &ssm_a.data,
                        &dt_bias.data,
                        &norm_w.data,
                        &mut *dn_recurrent_out,
                        dn_config_gpu,
                        self.config.norm_eps,
                    ),
                )
            }
            .map_err(|e| BackendError::OperationFailed(format!("deltanet_recurrent: {}", e)))?;
        }

        // 6. Output projection on GPU → result goes into hidden_norm (reused as scratch)
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ssm_out.weight", prefix),
            None,
            dn_recurrent_out as &CudaSlice<f32>,
            &mut self.hidden_norm,
        )?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Dense FFN (existing GPU path)
    // -----------------------------------------------------------------------

    fn dense_ffn_gpu_forward(&mut self, prefix: &str) -> BackendResult<()> {
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

        silu_gpu(
            &self.kernels,
            &self.device,
            self.config.intermediate_size,
            &mut self.ffn_gate,
        )?;

        mul_gpu(
            &self.kernels,
            self.config.intermediate_size,
            &self.ffn_gate,
            &self.ffn_up,
            &mut self.ffn_down,
        )?;

        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ffn_down.weight", prefix),
            None,
            &self.ffn_down,
            &mut self.hidden_norm,
        )?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // MoE (full GPU: router + stream active expert weights + GPU compute)
    // -----------------------------------------------------------------------

    fn moe_gpu_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let prefix = format!("blk.{}", layer_idx);
        let hidden_size = self.config.hidden_size;
        let expert_intermediate = self.config.expert_intermediate;

        // 1. Router projection on GPU (weights already uploaded)
        let moe = self.layers[layer_idx]
            .moe()
            .ok_or_else(|| BackendError::OperationFailed("Expected MoE layer".into()))?;

        let num_experts = moe.num_experts();
        let top_k = moe.num_experts_per_token();

        // Router: compute logits on GPU, then download for top-k selection
        let mut router_logits_gpu = self
            .device
            .alloc_zeros::<f32>(num_experts)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ffn_gate_inp.weight", prefix),
            None,
            &self.hidden_norm,
            &mut router_logits_gpu,
        )?;

        // Download router logits to CPU for top-k selection (~2 KB)
        let mut logits_cpu = self
            .device
            .dtoh_sync_copy(&router_logits_gpu)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // Normalize logits (subtract max for numerical stability)
        let max_logit = logits_cpu
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        for l in &mut logits_cpu {
            *l -= max_logit;
        }

        // Top-k selection on CPU
        let mut indexed: Vec<(usize, f32)> = logits_cpu.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_indices: Vec<usize> = indexed[..top_k].iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = indexed[..top_k].iter().map(|(_, l)| *l).collect();

        // Softmax over top-k to get routing weights
        let max_val = top_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = top_logits.iter().map(|&l| (l - max_val).exp()).sum();
        let routing_weights: Vec<f32> = top_logits
            .iter()
            .map(|&l| (l - max_val).exp() / exp_sum)
            .collect();

        // 2. Zero accumulator
        let zeros = vec![0.0f32; hidden_size];
        self.device
            .htod_sync_copy_into(&zeros, &mut self.moe_hidden)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // 3. For each selected expert: upload weights, compute on GPU, accumulate
        for (sel_idx, &expert_idx) in top_indices.iter().enumerate() {
            let weight = routing_weights[sel_idx];
            let expert = &moe.experts[expert_idx];

            // Upload expert weights to GPU scratch store (transposed for kernels)
            let gate_name = format!("{}.moe_scratch.gate", prefix);
            let up_name = format!("{}.moe_scratch.up", prefix);
            let down_name = format!("{}.moe_scratch.down", prefix);

            super::dequant_weights::upload_expert_weight_to_store(
                &mut self.weights,
                &gate_name,
                &expert.gate_proj,
            )?;
            super::dequant_weights::upload_expert_weight_to_store(
                &mut self.weights,
                &up_name,
                &expert.up_proj,
            )?;
            super::dequant_weights::upload_expert_weight_to_store(
                &mut self.weights,
                &down_name,
                &expert.down_proj,
            )?;

            // Gate projection: hidden_norm → moe_expert_gate
            linear_gpu(
                &self.kernels,
                &self.weights,
                &self.device,
                &gate_name,
                None,
                &self.hidden_norm,
                &mut self.moe_expert_gate,
            )?;

            // SiLU(gate)
            silu_gpu(
                &self.kernels,
                &self.device,
                expert_intermediate,
                &mut self.moe_expert_gate,
            )?;

            // Up projection: hidden_norm → moe_expert_up
            linear_gpu(
                &self.kernels,
                &self.weights,
                &self.device,
                &up_name,
                None,
                &self.hidden_norm,
                &mut self.moe_expert_up,
            )?;

            // gate * up → moe_expert_down (reuse as intermediate)
            mul_gpu(
                &self.kernels,
                expert_intermediate,
                &self.moe_expert_gate,
                &self.moe_expert_up,
                &mut self.moe_expert_down,
            )?;

            // Down projection: intermediate → moe_expert_out
            linear_gpu(
                &self.kernels,
                &self.weights,
                &self.device,
                &down_name,
                None,
                &self.moe_expert_down,
                &mut self.moe_expert_out,
            )?;

            // Weighted accumulation: moe_hidden += weight * expert_out
            scaled_add_gpu(
                &self.kernels,
                hidden_size,
                &mut self.moe_hidden,
                &self.moe_expert_out,
                weight,
            )?;
        }

        // 4. Shared experts (always active, on GPU)
        let has_shared = !moe.shared_experts.is_empty();
        if has_shared {
            // Shared expert gate (sigmoid) if present
            let gate_scale = if self
                .weights
                .contains(&format!("{}.ffn_gate_shexp_gate", prefix))
            {
                // Download hidden_norm to CPU to compute dot product for sigmoid gate
                let h = self
                    .device
                    .dtoh_sync_copy(&self.hidden_norm)
                    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
                let gw = moe
                    .shared_expert_gate
                    .as_ref()
                    .and_then(|t| t.as_f32().ok())
                    .map(|d| d.to_vec())
                    .unwrap_or_default();
                let len = hidden_size.min(gw.len());
                let dot = simd::dot_f32(&h[..len], &gw[..len]);
                1.0 / (1.0 + (-dot).exp())
            } else {
                1.0
            };

            for se_idx in 0..moe.shared_experts.len() {
                let gate_name = format!("{}.ffn_gate_shexp.{}.weight", prefix, se_idx);
                let up_name = format!("{}.ffn_up_shexp.{}.weight", prefix, se_idx);
                let down_name = format!("{}.ffn_down_shexp.{}.weight", prefix, se_idx);

                linear_gpu(
                    &self.kernels,
                    &self.weights,
                    &self.device,
                    &gate_name,
                    None,
                    &self.hidden_norm,
                    &mut self.moe_expert_gate,
                )?;

                silu_gpu(
                    &self.kernels,
                    &self.device,
                    expert_intermediate,
                    &mut self.moe_expert_gate,
                )?;

                linear_gpu(
                    &self.kernels,
                    &self.weights,
                    &self.device,
                    &up_name,
                    None,
                    &self.hidden_norm,
                    &mut self.moe_expert_up,
                )?;

                mul_gpu(
                    &self.kernels,
                    expert_intermediate,
                    &self.moe_expert_gate,
                    &self.moe_expert_up,
                    &mut self.moe_expert_down,
                )?;

                linear_gpu(
                    &self.kernels,
                    &self.weights,
                    &self.device,
                    &down_name,
                    None,
                    &self.moe_expert_down,
                    &mut self.moe_expert_out,
                )?;

                scaled_add_gpu(
                    &self.kernels,
                    hidden_size,
                    &mut self.moe_hidden,
                    &self.moe_expert_out,
                    gate_scale,
                )?;
            }
        }

        // 5. Copy MoE result to hidden_norm (used by residual add)
        self.device
            .dtod_copy(&self.moe_hidden, &mut self.hidden_norm)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        Ok(())
    }

    // (Attention is handled via CPU roundtrip in attention_cpu_forward)
}

// ---------------------------------------------------------------------------
// GpuModelWrapper: implements the Model trait using GpuOnlyInference
// ---------------------------------------------------------------------------

use crate::model::{Architecture, InferenceContext, Model, ModelConfig, ModelError, ModelResult};
use std::sync::Mutex;

pub struct GpuModelWrapper {
    gpu: Mutex<GpuOnlyInference>,
    config: ModelConfig,
    architecture: Architecture,
}

impl GpuModelWrapper {
    pub fn new(gpu: GpuOnlyInference, config: ModelConfig, architecture: Architecture) -> Self {
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
