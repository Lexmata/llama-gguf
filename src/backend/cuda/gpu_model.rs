//! GPU-resident model for full GPU acceleration
//!
//! This module provides infrastructure for keeping model weights on the GPU
//! to avoid repeated CPU<->GPU transfers during inference.

use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// A weight tensor stored on GPU in dequantized F32 format
pub struct GpuWeight {
    /// GPU memory containing dequantized F32 weights
    pub data: CudaSlice<f32>,
    /// Original shape [in_features, out_features] for linear layers
    pub shape: Vec<usize>,
    /// Number of elements
    pub numel: usize,
}

impl GpuWeight {
    /// Create GPU weight from CPU tensor (dequantizes if needed)
    pub fn from_tensor(device: &Arc<CudaDevice>, tensor: &Tensor) -> BackendResult<Self> {
        let shape = tensor.shape().to_vec();
        let numel: usize = shape.iter().product();

        // Get F32 data, dequantizing if needed
        let f32_data: Vec<f32> = if tensor.dtype() == DType::F32 {
            tensor.as_f32()?.to_vec()
        } else {
            // Dequantize to F32
            let mut dequant = Tensor::zeros(shape.clone(), DType::F32);
            crate::backend::cpu::ops::dequantize(tensor, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        // Upload to GPU
        let data = device
            .htod_sync_copy(&f32_data)
            .map_err(|e| BackendError::AllocationFailed(format!("GPU upload failed: {}", e)))?;

        Ok(Self { data, shape, numel })
    }
}

/// GPU-resident linear layer
pub struct GpuLinear {
    /// Weight on GPU [in_features, out_features]
    pub weight: GpuWeight,
    /// Bias on GPU [out_features] (optional)
    pub bias: Option<CudaSlice<f32>>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl GpuLinear {
    /// Create from CPU linear layer
    pub fn from_linear(
        device: &Arc<CudaDevice>,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> BackendResult<Self> {
        let gpu_weight = GpuWeight::from_tensor(device, weight)?;
        let in_features = gpu_weight.shape[0];
        let out_features = gpu_weight.shape[1];

        let gpu_bias = if let Some(b) = bias {
            let bias_data = b.as_f32()?;
            Some(
                device
                    .htod_sync_copy(bias_data)
                    .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            )
        } else {
            None
        };

        Ok(Self {
            weight: gpu_weight,
            bias: gpu_bias,
            in_features,
            out_features,
        })
    }
}

/// GPU-resident RMS normalization layer
pub struct GpuRMSNorm {
    /// Weight on GPU [hidden_size]
    pub weight: CudaSlice<f32>,
    /// Epsilon
    pub eps: f32,
    /// Hidden size
    pub hidden_size: usize,
}

impl GpuRMSNorm {
    /// Create from CPU RMSNorm
    pub fn from_rms_norm(
        device: &Arc<CudaDevice>,
        weight: &Tensor,
        eps: f32,
    ) -> BackendResult<Self> {
        let weight_data = weight.as_f32()?;
        let hidden_size = weight_data.len();

        let gpu_weight = device
            .htod_sync_copy(weight_data)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        Ok(Self {
            weight: gpu_weight,
            eps,
            hidden_size,
        })
    }
}

/// GPU-resident attention layer
pub struct GpuAttention {
    pub wq: GpuLinear,
    pub wk: GpuLinear,
    pub wv: GpuLinear,
    pub wo: GpuLinear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub use_neox_rope: bool,
}

/// GPU-resident FFN layer
pub struct GpuFFN {
    pub w1: GpuLinear, // gate
    pub w2: GpuLinear, // down
    pub w3: GpuLinear, // up
}

/// GPU-resident transformer layer
pub struct GpuTransformerLayer {
    pub attention_norm: GpuRMSNorm,
    pub attention: GpuAttention,
    pub ffn_norm: GpuRMSNorm,
    pub ffn: GpuFFN,
}

/// GPU KV cache for a single layer
pub struct GpuKVCache {
    /// Key cache [num_kv_heads, max_seq_len, head_dim]
    pub k_cache: CudaSlice<f32>,
    /// Value cache [num_kv_heads, max_seq_len, head_dim]
    pub v_cache: CudaSlice<f32>,
    /// Current position in cache
    pub pos: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl GpuKVCache {
    /// Create new KV cache
    pub fn new(
        device: &Arc<CudaDevice>,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> BackendResult<Self> {
        let cache_size = num_kv_heads * max_seq_len * head_dim;

        let k_cache = device
            .alloc_zeros::<f32>(cache_size)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        let v_cache = device
            .alloc_zeros::<f32>(cache_size)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        Ok(Self {
            k_cache,
            v_cache,
            pos: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
        })
    }

    /// Reset cache
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Full GPU-resident model
pub struct GpuModel {
    pub device: Arc<CudaDevice>,
    /// Token embedding [vocab_size, hidden_size] on GPU
    pub token_embedding: GpuWeight,
    /// Transformer layers
    pub layers: Vec<GpuTransformerLayer>,
    /// Final normalization
    pub norm: GpuRMSNorm,
    /// Output projection
    pub output: GpuLinear,
    /// KV cache for each layer
    pub kv_caches: Vec<GpuKVCache>,
    /// Model config
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    /// Scratch buffers for intermediate results
    pub scratch: GpuScratchBuffers,
}

/// Scratch buffers for GPU computation
pub struct GpuScratchBuffers {
    /// Hidden state [hidden_size]
    pub hidden: CudaSlice<f32>,
    /// Residual connection [hidden_size]
    pub residual: CudaSlice<f32>,
    /// Attention output [hidden_size]
    pub attn_out: CudaSlice<f32>,
    /// FFN intermediate [intermediate_size]
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    pub ffn_out: CudaSlice<f32>,
    /// Query/Key/Value projections
    pub q: CudaSlice<f32>, // [num_heads * head_dim]
    pub k: CudaSlice<f32>, // [num_kv_heads * head_dim]
    pub v: CudaSlice<f32>, // [num_kv_heads * head_dim]
    /// Attention scores [num_heads, kv_len]
    pub attn_scores: CudaSlice<f32>,
    /// Output logits [vocab_size]
    pub logits: CudaSlice<f32>,
}

impl GpuScratchBuffers {
    pub fn new(
        device: &Arc<CudaDevice>,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        vocab_size: usize,
    ) -> BackendResult<Self> {
        Ok(Self {
            hidden: device
                .alloc_zeros::<f32>(hidden_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            residual: device
                .alloc_zeros::<f32>(hidden_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            attn_out: device
                .alloc_zeros::<f32>(hidden_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            ffn_gate: device
                .alloc_zeros::<f32>(intermediate_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            ffn_up: device
                .alloc_zeros::<f32>(intermediate_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            ffn_out: device
                .alloc_zeros::<f32>(hidden_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            q: device
                .alloc_zeros::<f32>(num_heads * head_dim)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            k: device
                .alloc_zeros::<f32>(num_kv_heads * head_dim)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            v: device
                .alloc_zeros::<f32>(num_kv_heads * head_dim)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            attn_scores: device
                .alloc_zeros::<f32>(num_heads * max_seq_len)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
            logits: device
                .alloc_zeros::<f32>(vocab_size)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?,
        })
    }
}
