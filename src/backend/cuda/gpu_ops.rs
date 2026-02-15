//! GPU-only operations for full GPU inference
//!
//! These operations work entirely on GPU memory without CPU transfers.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::gpu_model::*;
use super::kernels::CudaKernels;
use crate::backend::{BackendError, BackendResult};

/// GPU operations context
pub struct GpuOps {
    pub device: Arc<CudaDevice>,
    pub kernels: CudaKernels,
}

impl GpuOps {
    /// Create new GPU operations context
    pub fn new(device: Arc<CudaDevice>) -> BackendResult<Self> {
        let kernels = CudaKernels::new(Arc::clone(&device))?;
        Ok(Self { device, kernels })
    }

    /// Helper to create launch config
    fn launch_config(&self, n: usize, block_size: usize) -> LaunchConfig {
        let grid_size = (n + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn launch_config_shared(&self, n: usize, block_size: usize, shared: usize) -> LaunchConfig {
        let grid_size = (n + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        }
    }

    // =========================================================================
    // Basic operations (GPU buffer to GPU buffer)
    // =========================================================================

    /// Element-wise add: out = a + b (all on GPU)
    pub fn add_gpu(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> BackendResult<()> {
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .add_f32
                .clone()
                .launch(config, (a, b, out, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    /// Element-wise multiply: out = a * b
    pub fn mul_gpu(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> BackendResult<()> {
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .mul_f32
                .clone()
                .launch(config, (a, b, out, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    /// Scale: out = a * scalar
    pub fn scale_gpu(
        &self,
        a: &CudaSlice<f32>,
        scalar: f32,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> BackendResult<()> {
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .scale_f32
                .clone()
                .launch(config, (a, scalar, out, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    /// Copy GPU buffer
    pub fn copy_gpu(&self, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>) -> BackendResult<()> {
        self.device
            .dtod_copy(src, dst)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    // =========================================================================
    // Activations
    // =========================================================================

    /// SiLU activation: out = x * sigmoid(x)
    pub fn silu_gpu(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> BackendResult<()> {
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .silu_f32
                .clone()
                .launch(config, (x, out, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    /// GELU activation
    pub fn gelu_gpu(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> BackendResult<()> {
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .gelu_f32
                .clone()
                .launch(config, (x, out, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    // =========================================================================
    // Normalization
    // =========================================================================

    /// RMS normalization: out = x / rms(x) * weight
    pub fn rms_norm_gpu(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> BackendResult<()> {
        // Allocate temp for sum of squares
        let mut sum_sq = self
            .device
            .alloc_zeros::<f32>(1)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        // Pass 1: Compute sum of squares
        let config = self.launch_config_shared(n, 256, 256 * 4);
        unsafe {
            self.kernels
                .rms_norm_sum_sq
                .clone()
                .launch(config, (x, &mut sum_sq, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // Read sum and compute RMS inverse
        let sum_sq_val: Vec<f32> = self
            .device
            .dtoh_sync_copy(&sum_sq)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        let rms = (sum_sq_val[0] / n as f32 + eps).sqrt();
        let rms_inv = 1.0 / rms;

        // Pass 2: Normalize and scale
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .rms_norm_scale
                .clone()
                .launch(config, (x, weight, out, rms_inv, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    /// Softmax (copies x to out, then normalizes out)
    pub fn softmax_gpu(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> BackendResult<()> {
        let mut max_val = self
            .device
            .alloc_zeros::<f32>(1)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        let mut sum_val = self
            .device
            .alloc_zeros::<f32>(1)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        // Initialize max to neg infinity
        self.device
            .htod_sync_copy_into(&[f32::NEG_INFINITY], &mut max_val)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let config = self.launch_config_shared(n, 256, 256 * 4);

        // Find max
        unsafe {
            self.kernels
                .softmax_max
                .clone()
                .launch(config.clone(), (x, &mut max_val, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let max_v: Vec<f32> = self
            .device
            .dtoh_sync_copy(&max_val)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        // Exp and sum (writes to out)
        unsafe {
            self.kernels
                .softmax_exp_sum
                .clone()
                .launch(config, (x, &mut *out, &mut sum_val, max_v[0], n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let sum_v: Vec<f32> = self
            .device
            .dtoh_sync_copy(&sum_val)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        let sum_inv = 1.0 / sum_v[0];

        // Normalize (in place on out)
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .softmax_div
                .clone()
                .launch(config, (&mut *out, sum_inv, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    // =========================================================================
    // Matrix operations
    // =========================================================================

    /// Vector-matrix multiply: out = vec @ mat
    /// vec: [k], mat: [k, n], out: [n]
    pub fn vec_mat_gpu(
        &self,
        vec: &CudaSlice<f32>,
        mat: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        k: usize,
        n: usize,
    ) -> BackendResult<()> {
        let config = self.launch_config(n, 256);
        unsafe {
            self.kernels
                .vec_mat_f32
                .clone()
                .launch(config, (vec, mat, out, k as i32, n as i32))
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    /// Linear layer forward pass: out = x @ W + b
    pub fn linear_gpu(
        &self,
        x: &CudaSlice<f32>,
        layer: &GpuLinear,
        out: &mut CudaSlice<f32>,
    ) -> BackendResult<()> {
        // x @ W
        self.vec_mat_gpu(
            x,
            &layer.weight.data,
            out,
            layer.in_features,
            layer.out_features,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.bias {
            let mut temp = self
                .device
                .alloc_zeros::<f32>(layer.out_features)
                .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
            self.add_gpu(out, bias, &mut temp, layer.out_features)?;
            self.copy_gpu(&temp, out)?;
        }

        Ok(())
    }

    // =========================================================================
    // RoPE
    // =========================================================================

    /// Apply RoPE to Q and K tensors on GPU
    pub fn rope_gpu(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
        use_neox: bool,
    ) -> BackendResult<()> {
        let config = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels.rope_single_pos.clone().launch(
                config,
                (
                    q,
                    k,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    pos as i32,
                    freq_base,
                    freq_scale,
                    if use_neox { 1i32 } else { 0i32 },
                ),
            )
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    // =========================================================================
    // Embedding lookup
    // =========================================================================

    /// Look up token embedding from GPU embedding table
    pub fn embed_token(
        &self,
        token_id: u32,
        embedding: &GpuWeight,
        out: &mut CudaSlice<f32>,
        hidden_size: usize,
    ) -> BackendResult<()> {
        // Copy the embedding row for this token
        let offset = token_id as usize * hidden_size;

        // Create a view into the embedding table
        // cudarc doesn't have slicing, so we need to copy through CPU for now
        // TODO: Use a CUDA kernel for this
        let emb_data: Vec<f32> = self
            .device
            .dtoh_sync_copy(&embedding.data)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let token_emb = &emb_data[offset..offset + hidden_size];
        self.device
            .htod_sync_copy_into(token_emb, out)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }

    // =========================================================================
    // Attention
    // =========================================================================

    /// Compute attention scores and output (single head)
    /// q: [head_dim]
    /// k_cache: [kv_len, head_dim]
    /// v_cache: [kv_len, head_dim]
    /// out: [head_dim]
    pub fn attention_head_gpu(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        _scores: &mut CudaSlice<f32>,
        head_dim: usize,
        kv_len: usize,
        q_pos: usize,
        scale: f32,
    ) -> BackendResult<()> {
        // For small kv_len, use simple implementation
        // For larger, would want to use flash attention

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (kv_len.min(1024) as u32, 1, 1),
            shared_mem_bytes: (kv_len * 4) as u32,
        };

        unsafe {
            self.kernels.attention_single_head.clone().launch(
                config,
                (
                    q,
                    k_cache,
                    v_cache,
                    out,
                    head_dim as i32,
                    kv_len as i32,
                    q_pos as i32,
                    scale,
                ),
            )
        }
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
    }
}
