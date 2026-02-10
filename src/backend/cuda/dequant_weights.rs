//! Dequantized weight storage for GPU acceleration
//!
//! This module provides functionality to dequantize model weights once
//! at load time and store them as F32 for fast GPU inference.

use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// Storage for dequantized weights on GPU
pub struct GpuWeightStore {
    device: Arc<CudaDevice>,
    /// Weights stored by GGUF tensor name
    weights: HashMap<String, GpuWeight>,
    /// Total bytes allocated
    total_bytes: usize,
}

/// A single weight stored on GPU
pub struct GpuWeight {
    /// GPU memory containing dequantized F32 weights
    pub data: CudaSlice<f32>,
    /// Shape of the weight
    pub shape: Vec<usize>,
    /// Number of elements
    pub numel: usize,
}

impl GpuWeightStore {
    /// Create a new empty weight store
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            weights: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Upload a tensor to GPU, dequantizing if needed
    /// Uses the tensor's name if set, otherwise uses provided name
    pub fn upload(&mut self, name: &str, tensor: &Tensor) -> BackendResult<()> {
        let numel = tensor.numel();
        let shape = tensor.shape().to_vec();

        // Use tensor's name if available, otherwise use provided name
        let key = tensor.name().unwrap_or(name).to_string();

        // Get F32 data
        let f32_data: Vec<f32> = if tensor.dtype() == DType::F32 {
            tensor.as_f32()?.to_vec()
        } else {
            // Dequantize
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            crate::backend::cpu::ops::dequantize(tensor, &mut dequant)?;
            dequant.as_f32()?.to_vec()
        };

        // Upload to GPU
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

    /// Get a weight by name
    pub fn get(&self, name: &str) -> Option<&GpuWeight> {
        self.weights.get(name)
    }

    /// Check if a weight exists
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Get total VRAM usage in bytes
    pub fn vram_usage(&self) -> usize {
        self.total_bytes
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get number of weights stored
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Upload all model weights to GPU
/// Weights are stored by their GGUF tensor names for easy lookup during inference
pub fn upload_model_weights(
    device: Arc<CudaDevice>,
    layers: &[crate::model::TransformerLayer],
    embedding: &Tensor,
    output: &crate::model::layers::Linear,
    norm: &crate::model::layers::RMSNorm,
) -> BackendResult<GpuWeightStore> {
    let mut store = GpuWeightStore::new(device);

    // Upload embedding (name is "token_embd.weight")
    store.upload("token_embd.weight", embedding)?;

    // Upload each layer
    for (i, layer) in layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("  Layer {}/{}", i + 1, layers.len());
        }

        // Attention weights - use tensor's stored name
        store.upload(
            &format!("blk.{}.attn_q.weight", i),
            &layer.attention.wq.weight,
        )?;
        store.upload(
            &format!("blk.{}.attn_k.weight", i),
            &layer.attention.wk.weight,
        )?;
        store.upload(
            &format!("blk.{}.attn_v.weight", i),
            &layer.attention.wv.weight,
        )?;
        store.upload(
            &format!("blk.{}.attn_output.weight", i),
            &layer.attention.wo.weight,
        )?;

        // Attention biases (if present)
        if let Some(ref bias) = layer.attention.wq.bias {
            store.upload(&format!("blk.{}.attn_q.bias", i), bias)?;
        }
        if let Some(ref bias) = layer.attention.wk.bias {
            store.upload(&format!("blk.{}.attn_k.bias", i), bias)?;
        }
        if let Some(ref bias) = layer.attention.wv.bias {
            store.upload(&format!("blk.{}.attn_v.bias", i), bias)?;
        }

        // Attention norm
        store.upload(
            &format!("blk.{}.attn_norm.weight", i),
            &layer.attn_norm.weight,
        )?;

        // FFN
        store.upload(
            &format!("blk.{}.ffn_gate.weight", i),
            &layer.ffn.w_gate.weight,
        )?;
        store.upload(&format!("blk.{}.ffn_up.weight", i), &layer.ffn.w_up.weight)?;
        store.upload(
            &format!("blk.{}.ffn_down.weight", i),
            &layer.ffn.w_down.weight,
        )?;

        // FFN norm
        store.upload(
            &format!("blk.{}.ffn_norm.weight", i),
            &layer.ffn_norm.weight,
        )?;
    }

    // Upload final norm
    store.upload("output_norm.weight", &norm.weight)?;

    // Upload output projection
    store.upload("output.weight", &output.weight)?;
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
