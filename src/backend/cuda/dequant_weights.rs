//! Weight storage for GPU acceleration
//!
//! Supports two storage modes:
//! - **F32 (dequantized)**: Small weights (norms, biases) stored as f32
//! - **Quantized**: Large weight matrices kept in their compressed format
//!   and dequantized on-the-fly during matmul via CUDA kernels
//!
//! The quantized path uses ~5-8× less VRAM than f32, enabling larger models
//! on memory-constrained GPUs.

use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// Storage for weights on GPU (both f32 and quantized)
pub struct GpuWeightStore {
    device: Arc<CudaDevice>,
    /// F32 weights stored by GGUF tensor name (norms, biases, small tensors)
    weights: HashMap<String, GpuWeight>,
    /// Quantized weights stored by GGUF tensor name (large weight matrices)
    quantized_weights: HashMap<String, QuantizedGpuWeight>,
    /// Total bytes allocated on GPU
    total_bytes: usize,
}

/// A single f32 weight stored on GPU
pub struct GpuWeight {
    /// GPU memory containing dequantized F32 weights
    pub data: CudaSlice<f32>,
    /// Shape of the weight
    pub shape: Vec<usize>,
    /// Number of elements
    pub numel: usize,
}

/// A quantized weight stored on GPU in its compressed format
pub struct QuantizedGpuWeight {
    /// Raw quantized bytes on GPU, transposed to kernel-expected layout:
    /// `[num_blocks_per_col, n_cols, block_bytes]`
    pub data: CudaSlice<u8>,
    /// Original tensor shape (e.g. [k, n] for a weight matrix)
    pub shape: Vec<usize>,
    /// Quantization type
    pub dtype: DType,
    /// Number of logical elements
    pub numel: usize,
}

impl GpuWeightStore {
    /// Create a new empty weight store
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            weights: HashMap::new(),
            quantized_weights: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Upload a tensor to GPU as f32 (dequantizing if needed).
    /// Used for small weights like norms and biases.
    pub fn upload(&mut self, name: &str, tensor: &Tensor) -> BackendResult<()> {
        let numel = tensor.numel();
        let shape = tensor.shape().to_vec();

        let key = tensor.name().unwrap_or(name).to_string();

        let f32_data: Vec<f32> = if tensor.dtype() == DType::F32 {
            tensor.as_f32()?.to_vec()
        } else {
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            crate::backend::cpu::ops::dequantize(tensor, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        let gpu_data = self
            .device
            .htod_sync_copy(&f32_data)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        self.total_bytes += numel * 4;
        self.weights.insert(
            key,
            GpuWeight {
                data: gpu_data,
                shape,
                numel,
            },
        );

        Ok(())
    }

    /// Upload a quantized tensor to GPU in its compressed format.
    ///
    /// The block data is transposed from GGUF layout `[n_cols, blocks_per_col, block_bytes]`
    /// to the kernel-expected layout `[blocks_per_col, n_cols, block_bytes]` for coalesced
    /// memory access in the matmul kernels.
    pub fn upload_quantized(&mut self, name: &str, tensor: &Tensor) -> BackendResult<()> {
        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();
        let numel = tensor.numel();
        let key = tensor.name().unwrap_or(name).to_string();

        if !dtype.is_quantized() {
            return Err(BackendError::OperationFailed(format!(
                "upload_quantized called on non-quantized tensor {} ({:?})",
                key, dtype
            )));
        }

        let raw_bytes = tensor.data();
        let block_size = dtype.block_size();
        let block_bytes = dtype.block_bytes();

        // For a 2D weight tensor with shape [k, n] (GGUF: first dim varies fastest):
        // - k = shape[0] (input dimension, innermost)
        // - n = shape[1] (output dimension, outermost)
        // - Each column has k/block_size blocks
        // GGUF stores: all blocks for col 0, then col 1, etc.
        // Kernel expects: block[row_block][col] i.e. all cols for row_block 0, etc.
        let k = shape[0];
        let n = if shape.len() >= 2 { shape[1] } else { 1 };
        let blocks_per_col = k / block_size;

        let total_blocks = blocks_per_col * n;
        let total_raw_bytes = total_blocks * block_bytes;

        if raw_bytes.len() < total_raw_bytes {
            return Err(BackendError::OperationFailed(format!(
                "Tensor {} has {} bytes but expected {} ({} blocks × {} bytes)",
                key,
                raw_bytes.len(),
                total_raw_bytes,
                total_blocks,
                block_bytes
            )));
        }

        // Transpose blocks from [n, blocks_per_col, block_bytes]
        // to [blocks_per_col, n, block_bytes]
        let mut transposed = vec![0u8; total_raw_bytes];
        for col in 0..n {
            for j in 0..blocks_per_col {
                let src_offset = (col * blocks_per_col + j) * block_bytes;
                let dst_offset = (j * n + col) * block_bytes;
                transposed[dst_offset..dst_offset + block_bytes]
                    .copy_from_slice(&raw_bytes[src_offset..src_offset + block_bytes]);
            }
        }

        let gpu_data = self
            .device
            .htod_sync_copy(&transposed)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        self.total_bytes += total_raw_bytes;
        self.quantized_weights.insert(
            key,
            QuantizedGpuWeight {
                data: gpu_data,
                shape,
                dtype,
                numel,
            },
        );

        Ok(())
    }

    /// Get an f32 weight by name
    pub fn get(&self, name: &str) -> Option<&GpuWeight> {
        self.weights.get(name)
    }

    /// Get a quantized weight by name
    pub fn get_quantized(&self, name: &str) -> Option<&QuantizedGpuWeight> {
        self.quantized_weights.get(name)
    }

    /// Check if a weight exists (either f32 or quantized)
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name) || self.quantized_weights.contains_key(name)
    }

    /// Get total VRAM usage in bytes
    pub fn vram_usage(&self) -> usize {
        self.total_bytes
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get number of weights stored (both f32 and quantized)
    pub fn len(&self) -> usize {
        self.weights.len() + self.quantized_weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty() && self.quantized_weights.is_empty()
    }
}

/// Supported quantized types that have GPU matmul kernels
fn has_quantized_kernel(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Q4K | DType::Q6K | DType::Q5K | DType::Q4_0 | DType::Q8_0
    )
}

/// Upload a weight tensor to GPU, choosing quantized or f32 path automatically.
///
/// Large 2D weight matrices with supported quantization are kept compressed;
/// everything else (norms, biases, unsupported quant types) is dequantized to f32.
fn upload_weight(store: &mut GpuWeightStore, name: &str, tensor: &Tensor) -> BackendResult<()> {
    let dtype = tensor.dtype();
    let is_2d = tensor.shape().len() >= 2;

    if is_2d && dtype.is_quantized() && has_quantized_kernel(dtype) {
        store.upload_quantized(name, tensor)
    } else {
        store.upload(name, tensor)
    }
}

/// Upload all model weights to GPU.
///
/// Large weight matrices are kept in quantized format when a GPU matmul kernel
/// exists for their dtype. Small weights (norms, biases) are always stored as f32.
pub fn upload_model_weights(
    device: Arc<CudaDevice>,
    layers: &[crate::model::TransformerLayer],
    embedding: &Tensor,
    output: &crate::model::layers::Linear,
    norm: &crate::model::layers::RMSNorm,
) -> BackendResult<GpuWeightStore> {
    let mut store = GpuWeightStore::new(device);

    // Embedding: always dequantize to f32 (looked up by index on CPU)
    store.upload("token_embd.weight", embedding)?;

    for (i, layer) in layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("  Layer {}/{}", i + 1, layers.len());
        }

        // Attention weights — quantized path when possible
        upload_weight(
            &mut store,
            &format!("blk.{}.attn_q.weight", i),
            &layer.attention.wq.weight,
        )?;
        upload_weight(
            &mut store,
            &format!("blk.{}.attn_k.weight", i),
            &layer.attention.wk.weight,
        )?;
        upload_weight(
            &mut store,
            &format!("blk.{}.attn_v.weight", i),
            &layer.attention.wv.weight,
        )?;
        upload_weight(
            &mut store,
            &format!("blk.{}.attn_output.weight", i),
            &layer.attention.wo.weight,
        )?;

        // Biases are always f32
        if let Some(ref bias) = layer.attention.wq.bias {
            store.upload(&format!("blk.{}.attn_q.bias", i), bias)?;
        }
        if let Some(ref bias) = layer.attention.wk.bias {
            store.upload(&format!("blk.{}.attn_k.bias", i), bias)?;
        }
        if let Some(ref bias) = layer.attention.wv.bias {
            store.upload(&format!("blk.{}.attn_v.bias", i), bias)?;
        }

        // Norms are always f32 (small 1D tensors)
        store.upload(
            &format!("blk.{}.attn_norm.weight", i),
            &layer.attn_norm.weight,
        )?;

        // FFN weights — quantized path when possible
        upload_weight(
            &mut store,
            &format!("blk.{}.ffn_gate.weight", i),
            &layer.ffn.w_gate.weight,
        )?;
        upload_weight(
            &mut store,
            &format!("blk.{}.ffn_up.weight", i),
            &layer.ffn.w_up.weight,
        )?;
        upload_weight(
            &mut store,
            &format!("blk.{}.ffn_down.weight", i),
            &layer.ffn.w_down.weight,
        )?;

        // FFN norm
        store.upload(
            &format!("blk.{}.ffn_norm.weight", i),
            &layer.ffn_norm.weight,
        )?;
    }

    // Final norm (f32)
    store.upload("output_norm.weight", &norm.weight)?;

    // Output projection — quantized path when possible
    upload_weight(&mut store, "output.weight", &output.weight)?;
    if let Some(ref bias) = output.bias {
        store.upload("output.bias", bias)?;
    }

    let vram_mb = store.vram_usage() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "Upload complete: {} weights, {:.1} MB VRAM",
        store.len(),
        vram_mb
    );

    Ok(store)
}
