//! GPU-accelerated inference engine for Qwen3Next and similar architectures.
//!
//! - Attention layers: CPU roundtrip for correct Qwen3Next handling
//!   (QK norm, partial RoPE, attention gating, different kl/vl dims).
//!   Transfer is ~16 KB per attention layer per token.
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
    // CPU KV caches for attention layers [num_kv_heads, max_seq_len, key_length/value_length]
    cpu_k_caches: Vec<Option<Tensor>>,
    cpu_v_caches: Vec<Option<Tensor>>,
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

        // CPU KV caches for attention layers (using correct per-model key_length/value_length)
        let cpu_backend = CpuBackend::new();
        let mut cpu_k_caches: Vec<Option<Tensor>> = Vec::with_capacity(model_config.num_layers);
        let mut cpu_v_caches: Vec<Option<Tensor>> = Vec::with_capacity(model_config.num_layers);
        for i in 0..model_config.num_layers {
            if has_gpu_attention[i] {
                if let Some(attn) = layers[i].attention() {
                    let kl = attn.key_length;
                    let vl = attn.value_length;
                    cpu_k_caches.push(Some(Tensor::zeros(
                        vec![model_config.num_kv_heads, max_seq_len, kl],
                        DType::F32,
                    )));
                    cpu_v_caches.push(Some(Tensor::zeros(
                        vec![model_config.num_kv_heads, max_seq_len, vl],
                        DType::F32,
                    )));
                } else {
                    cpu_k_caches.push(None);
                    cpu_v_caches.push(None);
                }
            } else {
                cpu_k_caches.push(None);
                cpu_v_caches.push(None);
            }
        }

        // DeltaNet GPU state buffers
        let mut dn_conv_states = Vec::with_capacity(model_config.num_layers);
        let mut dn_ssm_states = Vec::with_capacity(model_config.num_layers);

        let mut dn_qkv_buf = None;
        let mut dn_gate_z_buf = None;
        let mut dn_ba_buf = None;
        let mut dn_conv_out_buf = None;
        let mut dn_recurrent_out_buf = None;
        let mut dn_config_gpu_buf = None;

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
            deltanet_config,
            dn_conv_states,
            dn_ssm_states,
            dn_qkv: dn_qkv_buf,
            dn_gate_z: dn_gate_z_buf,
            dn_ba: dn_ba_buf,
            dn_conv_out: dn_conv_out_buf,
            dn_recurrent_out: dn_recurrent_out_buf,
            dn_config_gpu: dn_config_gpu_buf,
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

        // ---- Attention (CPU roundtrip) / DeltaNet (GPU) ----
        if self.has_gpu_attention[layer_idx] {
            self.attention_cpu_forward(layer_idx)?;
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
    // Attention (CPU roundtrip for Qwen3Next-style: QK norm, partial RoPE,
    // attention gating, different key_length/value_length)
    // -----------------------------------------------------------------------

    fn attention_cpu_forward(&mut self, layer_idx: usize) -> BackendResult<()> {
        let hidden_size = self.config.hidden_size;

        // Download normalized hidden state from GPU → CPU (~8 KB)
        let hidden_cpu = self
            .device
            .dtoh_sync_copy(&self.hidden_norm)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        let x_tensor = Tensor::from_f32(&hidden_cpu, vec![hidden_size])
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // Run the full Attention::forward on CPU (handles QK norm, partial RoPE,
        // attention gating, different kl/vl — all correctly)
        let attn = self.layers[layer_idx]
            .attention()
            .ok_or_else(|| BackendError::OperationFailed("Expected attention layer".into()))?;

        let k_cache = self.cpu_k_caches[layer_idx]
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("Missing CPU K cache".into()))?;
        let v_cache = self.cpu_v_caches[layer_idx]
            .as_mut()
            .ok_or_else(|| BackendError::OperationFailed("Missing CPU V cache".into()))?;

        let attn_out = attn
            .forward(
                &x_tensor,
                k_cache,
                v_cache,
                self.pos,
                self.config.freq_base,
                self.config.freq_scale,
                &self.cpu_backend,
            )
            .map_err(|e| BackendError::OperationFailed(format!("CPU attention: {}", e)))?;

        // Upload attention output back to GPU → hidden_norm (~8 KB)
        let out_data = attn_out
            .as_f32()
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        self.device
            .htod_sync_copy_into(out_data, &mut self.hidden_norm)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

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
        linear_gpu(
            &self.kernels,
            &self.weights,
            &self.device,
            &format!("{}.ssm_ba.weight", prefix),
            None,
            &self.hidden_norm,
            dn_ba,
        )?;

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
