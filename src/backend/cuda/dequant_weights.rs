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

        let key = name.to_string();

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
        let key = name.to_string();

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

/// Upload ALL model weights to GPU for full GPU inference.
///
/// Permanently uploads:
/// - Attention layer weights (Q/K/V/O projections + biases)
/// - DeltaNet layer weights (QKV, gate, ssm_ba, ssm_out, conv1d, ssm_a, dt_bias, ssm_norm)
/// - MoE router weights + shared expert weights
/// - All normalization weights, embeddings, output projection
///
/// MoE expert weights (512 per layer) are NOT pre-uploaded — they are streamed
/// on-the-fly (only the 10 active experts per token per layer) via
/// `upload_expert_weight`.
pub fn upload_model_weights(
    device: Arc<CudaDevice>,
    layers: &[crate::model::TransformerLayer],
    embedding: &Tensor,
    output: &crate::model::layers::Linear,
    norm: &crate::model::layers::NormLayer,
) -> BackendResult<GpuWeightStore> {
    use crate::model::layers::AttentionLayer;

    let mut store = GpuWeightStore::new(device);

    store.upload("token_embd.weight", embedding)?;

    for (i, layer) in layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("  Layer {}/{}", i + 1, layers.len());
        }

        match &layer.attn_layer {
            AttentionLayer::FullAttention(attn) => {
                upload_weight(
                    &mut store,
                    &format!("blk.{}.attn_q.weight", i),
                    &attn.wq.weight,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.attn_k.weight", i),
                    &attn.wk.weight,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.attn_v.weight", i),
                    &attn.wv.weight,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.attn_output.weight", i),
                    &attn.wo.weight,
                )?;

                if let Some(ref bias) = attn.wq.bias {
                    store.upload(&format!("blk.{}.attn_q.bias", i), bias)?;
                }
                if let Some(ref bias) = attn.wk.bias {
                    store.upload(&format!("blk.{}.attn_k.bias", i), bias)?;
                }
                if let Some(ref bias) = attn.wv.bias {
                    store.upload(&format!("blk.{}.attn_v.bias", i), bias)?;
                }
            }
            AttentionLayer::Mamba(mb) => {
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ssm_in.weight", i),
                    &mb.ssm_in.weight,
                )?;
                if let Some(ref bias) = mb.ssm_in.bias {
                    store.upload(&format!("blk.{}.ssm_in.bias", i), bias)?;
                }
                store.upload(
                    &format!("blk.{}.ssm_conv1d.weight", i),
                    &mb.ssm_conv1d_weight,
                )?;
                if let Some(ref bias) = mb.ssm_conv1d_bias {
                    store.upload(&format!("blk.{}.ssm_conv1d.bias", i), bias)?;
                }
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ssm_x.weight", i),
                    &mb.ssm_x.weight,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ssm_dt.weight", i),
                    &mb.ssm_dt.weight,
                )?;
                store.upload(&format!("blk.{}.ssm_dt.bias", i), &mb.ssm_dt_bias)?;
                store.upload(&format!("blk.{}.ssm_a", i), &mb.ssm_a)?;
                if let Some(ref d) = mb.ssm_d {
                    store.upload(&format!("blk.{}.ssm_d", i), d)?;
                }
                if let Some(ref norm) = mb.ssm_norm {
                    store.upload(&format!("blk.{}.ssm_norm.weight", i), &norm.weight)?;
                }
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ssm_out.weight", i),
                    &mb.ssm_out.weight,
                )?;
                if let Some(ref bias) = mb.ssm_out.bias {
                    store.upload(&format!("blk.{}.ssm_out.bias", i), bias)?;
                }
            }
            AttentionLayer::DeltaNet(dn) => {
                upload_weight(
                    &mut store,
                    &format!("blk.{}.attn_qkv.weight", i),
                    &dn.attn_qkv.weight,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.attn_gate.weight", i),
                    &dn.attn_gate.weight,
                )?;
                match &dn.ssm_ba {
                    crate::model::deltanet::BetaAlphaProjection::Combined(linear) => {
                        upload_weight(
                            &mut store,
                            &format!("blk.{}.ssm_ba.weight", i),
                            &linear.weight,
                        )?;
                    }
                    crate::model::deltanet::BetaAlphaProjection::Separate { beta, alpha } => {
                        upload_weight(
                            &mut store,
                            &format!("blk.{}.ssm_beta.weight", i),
                            &beta.weight,
                        )?;
                        upload_weight(
                            &mut store,
                            &format!("blk.{}.ssm_alpha.weight", i),
                            &alpha.weight,
                        )?;
                    }
                }
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ssm_out.weight", i),
                    &dn.ssm_out.weight,
                )?;
                store.upload(
                    &format!("blk.{}.ssm_conv1d.weight", i),
                    &dn.ssm_conv1d_weight,
                )?;
                store.upload(&format!("blk.{}.ssm_a", i), &dn.ssm_a)?;
                store.upload(&format!("blk.{}.ssm_dt.bias", i), &dn.ssm_dt_bias)?;
                store.upload(
                    &format!("blk.{}.ssm_norm.weight", i),
                    &dn.ssm_norm.weight,
                )?;
            }
        }

        store.upload(
            &format!("blk.{}.attn_norm.weight", i),
            layer.attn_norm.weight(),
        )?;
        if let Some(bias) = layer.attn_norm.bias() {
            store.upload(&format!("blk.{}.attn_norm.bias", i), bias)?;
        }

        if let Some(ref pan) = layer.post_attn_norm {
            store.upload(
                &format!("blk.{}.post_attention_norm.weight", i),
                pan.weight(),
            )?;
            if let Some(bias) = pan.bias() {
                store.upload(
                    &format!("blk.{}.post_attention_norm.bias", i),
                    bias,
                )?;
            }
        }

        // Dense FFN weights
        if let Some(ffn) = layer.ffn() {
            upload_weight(
                &mut store,
                &format!("blk.{}.ffn_gate.weight", i),
                &ffn.w_gate.weight,
            )?;
            upload_weight(
                &mut store,
                &format!("blk.{}.ffn_up.weight", i),
                &ffn.w_up.weight,
            )?;
            upload_weight(
                &mut store,
                &format!("blk.{}.ffn_down.weight", i),
                &ffn.w_down.weight,
            )?;
        }

        // NoGate FFN weights (ffn_up, ffn_down + biases)
        if let Some(ffn) = layer.no_gate_ffn() {
            upload_weight(
                &mut store,
                &format!("blk.{}.ffn_up.weight", i),
                &ffn.w_up.weight,
            )?;
            if let Some(ref bias) = ffn.w_up.bias {
                store.upload(&format!("blk.{}.ffn_up.bias", i), bias)?;
            }
            upload_weight(
                &mut store,
                &format!("blk.{}.ffn_down.weight", i),
                &ffn.w_down.weight,
            )?;
            if let Some(ref bias) = ffn.w_down.bias {
                store.upload(&format!("blk.{}.ffn_down.bias", i), bias)?;
            }
        }

        // MoE router + shared expert weights
        if let Some(moe) = layer.moe() {
            store.upload(
                &format!("blk.{}.ffn_gate_inp.weight", i),
                &moe.router.weight,
            )?;
            for (se_idx, se) in moe.shared_experts.iter().enumerate() {
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ffn_gate_shexp.{}.weight", i, se_idx),
                    &se.gate_proj,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ffn_up_shexp.{}.weight", i, se_idx),
                    &se.up_proj,
                )?;
                upload_weight(
                    &mut store,
                    &format!("blk.{}.ffn_down_shexp.{}.weight", i, se_idx),
                    &se.down_proj,
                )?;
            }
            if let Some(ref gate_w) = moe.shared_expert_gate {
                store.upload(&format!("blk.{}.ffn_gate_shexp_gate", i), gate_w)?;
            }
        }

        if !matches!(layer.ffn_layer, crate::model::layers::FfnLayer::Identity) {
            store.upload(
                &format!("blk.{}.ffn_norm.weight", i),
                layer.ffn_norm.weight(),
            )?;
            if let Some(bias) = layer.ffn_norm.bias() {
                store.upload(&format!("blk.{}.ffn_norm.bias", i), bias)?;
            }
        }
    }

    store.upload("output_norm.weight", norm.weight())?;
    if let Some(bias) = norm.bias() {
        store.upload("output_norm.bias", bias)?;
    }

    upload_weight(&mut store, "output.weight", &output.weight)?;
    if let Some(ref bias) = output.bias {
        store.upload("output.bias", bias)?;
    }

    let vram_mb = store.vram_usage() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "Upload complete: {} weights, {:.1} MB VRAM",
        store.len(),
        vram_mb,
    );

    Ok(store)
}

/// Upload a single expert weight to GPU for on-the-fly MoE streaming.
///
/// Returns a `QuantizedGpuWeight` or `GpuWeight` that can be used for one
/// matmul, then dropped. The caller should reuse a name-keyed scratch slot
/// to avoid repeated allocations.
pub fn upload_expert_weight_to_store(
    store: &mut GpuWeightStore,
    name: &str,
    tensor: &Tensor,
) -> BackendResult<()> {
    upload_weight(store, name, tensor)
}
