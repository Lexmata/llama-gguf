//! GGUF model quantization API
//!
//! Provides functions to re-quantize GGUF models from one format to another.

use std::path::Path;

use bytemuck;
use half::f16;

use super::constants::GGUF_DEFAULT_ALIGNMENT;
use super::types::MetadataValue;
use super::{GgufBuilder, GgufError, GgufFile, GgmlType, TensorToWrite};
use crate::tensor::quant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0_blocks, dequantize_q4_1,
    dequantize_q4_k, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k,
    dequantize_q8_0_blocks, dequantize_q8_1, dequantize_q8_k,
    quantize_q2_k, quantize_q3_k, quantize_q4_0, quantize_q4_1, quantize_q4_k,
    quantize_q5_0, quantize_q5_1, quantize_q5_k, quantize_q6_k, quantize_q8_0, quantize_q8_1,
};
use crate::tensor::quant::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K,
    BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K,
};

/// Quantization options
#[derive(Debug, Clone)]
pub struct QuantizeOptions {
    /// Target quantization type
    pub target_type: GgmlType,
    /// Number of threads for parallel quantization
    pub threads: usize,
    /// Only quantize weight tensors (skip embeddings, norms)
    pub weights_only: bool,
    /// Minimum tensor elements to quantize (small tensors stay F32)
    pub min_elements: usize,
}

impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            target_type: GgmlType::Q4_0,
            threads: 4,
            weights_only: true,
            min_elements: 256,
        }
    }
}

/// Progress callback for quantization
pub type QuantizeProgressFn = Box<dyn Fn(usize, usize, &str) + Send>;

/// Quantization statistics
#[derive(Debug, Clone, Default)]
pub struct QuantizeStats {
    pub tensors_total: usize,
    pub tensors_quantized: usize,
    pub tensors_skipped: usize,
    pub bytes_original: usize,
    pub bytes_quantized: usize,
}

/// Quantize a GGUF model file to a new format
pub fn quantize_model(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: &QuantizeOptions,
    progress: Option<QuantizeProgressFn>,
) -> Result<QuantizeStats, GgufError> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    // 1. Open input GGUF file
    let gguf = GgufFile::open(input_path)?;
    let total_tensors = gguf.data.tensors.len();

    let alignment = gguf
        .data
        .metadata
        .get("general.alignment")
        .and_then(|v| match v {
            MetadataValue::Uint32(a) => Some(*a as usize),
            MetadataValue::Uint64(a) => Some(*a as usize),
            _ => None,
        })
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    // 2. Create GgufBuilder with same metadata
    let mut builder = GgufBuilder::new()
        .version(gguf.data.header.version)
        .alignment(alignment);

    for (key, value) in &gguf.data.metadata {
        builder = builder.metadata(key.clone(), value.clone());
    }

    let mut stats = QuantizeStats {
        tensors_total: total_tensors,
        tensors_quantized: 0,
        tensors_skipped: 0,
        bytes_original: 0,
        bytes_quantized: 0,
    };

    // 3. For each tensor: process and add to builder
    for (idx, tensor_info) in gguf.data.tensors.iter().enumerate() {
        if let Some(ref prog) = progress {
            prog(idx, total_tensors, &tensor_info.name);
        }

        let raw_data = gguf
            .tensor_data(&tensor_info.name)
            .ok_or_else(|| GgufError::InvalidData(format!("Missing tensor data: {}", tensor_info.name)))?;

        let n_elements = tensor_info.n_elements() as usize;
        stats.bytes_original += raw_data.len();

        let should_quantize = should_quantize_tensor(&tensor_info.name, options)
            && n_elements >= options.min_elements
            && options.target_type != tensor_info.dtype;

        let (output_data, output_dtype) = if should_quantize {
            let f32_data = dequantize_to_f32(raw_data, tensor_info.dtype, n_elements)?;
            let quantized = quantize_from_f32(&f32_data, options.target_type)?;
            stats.tensors_quantized += 1;
            stats.bytes_quantized += quantized.len();
            (quantized, options.target_type)
        } else {
            stats.tensors_skipped += 1;
            stats.bytes_quantized += raw_data.len();
            (raw_data.to_vec(), tensor_info.dtype)
        };

        let tensor = TensorToWrite::new(
            tensor_info.name.clone(),
            tensor_info.dims.clone(),
            output_dtype,
            output_data,
        );
        builder = builder.tensor(tensor);
    }

    // 4. Write output file
    builder.write_to_file(output_path)?;

    Ok(stats)
}

/// Check if a tensor should be quantized based on its name
fn should_quantize_tensor(name: &str, options: &QuantizeOptions) -> bool {
    if !options.weights_only {
        return true;
    }
    // Only quantize weight matrices, skip norms, embeddings, biases
    name.contains("weight")
        && !name.contains("norm")
        && !name.contains("embed")
        && !name.contains("bias")
}

/// Dequantize raw tensor data to F32
fn dequantize_to_f32(data: &[u8], dtype: GgmlType, n_elements: usize) -> Result<Vec<f32>, GgufError> {
    let mut output = vec![0.0f32; n_elements];

    match dtype {
        GgmlType::F32 => {
            if data.len() < n_elements * 4 {
                return Err(GgufError::InvalidData("F32 data too short".into()));
            }
            let f32_slice = bytemuck::cast_slice::<u8, f32>(data);
            output.copy_from_slice(&f32_slice[..n_elements]);
        }
        GgmlType::F16 => {
            if data.len() < n_elements * 2 {
                return Err(GgufError::InvalidData("F16 data too short".into()));
            }
            let f16_slice = bytemuck::cast_slice::<u8, f16>(data);
            for (i, &h) in f16_slice.iter().take(n_elements).enumerate() {
                output[i] = h.to_f32();
            }
        }
        GgmlType::BF16 => {
            if data.len() < n_elements * 2 {
                return Err(GgufError::InvalidData("BF16 data too short".into()));
            }
            for i in 0..n_elements {
                let offset = i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                output[i] = half::bf16::from_bits(bits).to_f32();
            }
        }
        GgmlType::Q4_0 => {
            let blocks: &[BlockQ4_0] = bytemuck::cast_slice(data);
            if blocks.len() * 32 != n_elements {
                return Err(GgufError::InvalidData("Q4_0 block count mismatch".into()));
            }
            dequantize_q4_0_blocks(blocks, &mut output);
        }
        GgmlType::Q4_1 => {
            let blocks: &[BlockQ4_1] = bytemuck::cast_slice(data);
            if blocks.len() * 32 != n_elements {
                return Err(GgufError::InvalidData("Q4_1 block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 32] = (&mut output[i * 32..(i + 1) * 32])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q4_1(block, out_slice);
            }
        }
        GgmlType::Q5_0 => {
            let blocks: &[BlockQ5_0] = bytemuck::cast_slice(data);
            if blocks.len() * 32 != n_elements {
                return Err(GgufError::InvalidData("Q5_0 block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 32] = (&mut output[i * 32..(i + 1) * 32])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q5_0(block, out_slice);
            }
        }
        GgmlType::Q5_1 => {
            let blocks: &[BlockQ5_1] = bytemuck::cast_slice(data);
            if blocks.len() * 32 != n_elements {
                return Err(GgufError::InvalidData("Q5_1 block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 32] = (&mut output[i * 32..(i + 1) * 32])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q5_1(block, out_slice);
            }
        }
        GgmlType::Q8_0 => {
            let blocks: &[BlockQ8_0] = bytemuck::cast_slice(data);
            if blocks.len() * 32 != n_elements {
                return Err(GgufError::InvalidData("Q8_0 block count mismatch".into()));
            }
            dequantize_q8_0_blocks(blocks, &mut output);
        }
        GgmlType::Q8_1 => {
            let blocks: &[BlockQ8_1] = bytemuck::cast_slice(data);
            if blocks.len() * 32 != n_elements {
                return Err(GgufError::InvalidData("Q8_1 block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 32] = (&mut output[i * 32..(i + 1) * 32])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q8_1(block, out_slice);
            }
        }
        GgmlType::Q2K => {
            let blocks: &[BlockQ2K] = bytemuck::cast_slice(data);
            if blocks.len() * 256 != n_elements {
                return Err(GgufError::InvalidData("Q2K block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 256] = (&mut output[i * 256..(i + 1) * 256])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q2_k(block, out_slice);
            }
        }
        GgmlType::Q3K => {
            let blocks: &[BlockQ3K] = bytemuck::cast_slice(data);
            if blocks.len() * 256 != n_elements {
                return Err(GgufError::InvalidData("Q3K block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 256] = (&mut output[i * 256..(i + 1) * 256])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q3_k(block, out_slice);
            }
        }
        GgmlType::Q4K => {
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(data);
            if blocks.len() * 256 != n_elements {
                return Err(GgufError::InvalidData("Q4K block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 256] = (&mut output[i * 256..(i + 1) * 256])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q4_k(block, out_slice);
            }
        }
        GgmlType::Q5K => {
            let blocks: &[BlockQ5K] = bytemuck::cast_slice(data);
            if blocks.len() * 256 != n_elements {
                return Err(GgufError::InvalidData("Q5K block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 256] = (&mut output[i * 256..(i + 1) * 256])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q5_k(block, out_slice);
            }
        }
        GgmlType::Q6K => {
            let blocks: &[BlockQ6K] = bytemuck::cast_slice(data);
            if blocks.len() * 256 != n_elements {
                return Err(GgufError::InvalidData("Q6K block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 256] = (&mut output[i * 256..(i + 1) * 256])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q6_k(block, out_slice);
            }
        }
        GgmlType::Q8K => {
            let blocks: &[BlockQ8K] = bytemuck::cast_slice(data);
            if blocks.len() * 256 != n_elements {
                return Err(GgufError::InvalidData("Q8K block count mismatch".into()));
            }
            for (i, block) in blocks.iter().enumerate() {
                let out_slice: &mut [f32; 256] = (&mut output[i * 256..(i + 1) * 256])
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Slice conversion failed".into()))?;
                dequantize_q8_k(block, out_slice);
            }
        }
        _ => {
            return Err(GgufError::InvalidData(format!(
                "Unsupported input type for dequantization: {:?}",
                dtype
            )));
        }
    }

    Ok(output)
}

/// Quantize F32 data to target format
fn quantize_from_f32(data: &[f32], target: GgmlType) -> Result<Vec<u8>, GgufError> {
    let n_elements = data.len();
    let block_size = target.block_size();

    if n_elements % block_size != 0 {
        return Err(GgufError::InvalidData(format!(
            "Element count {} not divisible by block size {}",
            n_elements, block_size
        )));
    }

    let output = match target {
        GgmlType::F32 => {
            return Err(GgufError::InvalidData(
                "Target type F32 cannot be used for quantization".into(),
            ));
        }
        GgmlType::Q4_0 => {
            let n_blocks = n_elements / 32;
            let mut out = Vec::with_capacity(n_blocks * BlockQ4_0::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q4_0(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q4_1 => {
            let n_blocks = n_elements / 32;
            let mut out = Vec::with_capacity(n_blocks * BlockQ4_1::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q4_1(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q5_0 => {
            let n_blocks = n_elements / 32;
            let mut out = Vec::with_capacity(n_blocks * BlockQ5_0::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q5_0(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q5_1 => {
            let n_blocks = n_elements / 32;
            let mut out = Vec::with_capacity(n_blocks * BlockQ5_1::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q5_1(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q8_0 => {
            let n_blocks = n_elements / 32;
            let mut out = Vec::with_capacity(n_blocks * BlockQ8_0::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q8_0(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q8_1 => {
            let n_blocks = n_elements / 32;
            let mut out = Vec::with_capacity(n_blocks * BlockQ8_1::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q8_1(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q2K => {
            let n_blocks = n_elements / 256;
            let mut out = Vec::with_capacity(n_blocks * BlockQ2K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q2_k(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q3K => {
            let n_blocks = n_elements / 256;
            let mut out = Vec::with_capacity(n_blocks * BlockQ3K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q3_k(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q4K => {
            let n_blocks = n_elements / 256;
            let mut out = Vec::with_capacity(n_blocks * BlockQ4K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q4_k(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q5K => {
            let n_blocks = n_elements / 256;
            let mut out = Vec::with_capacity(n_blocks * BlockQ5K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q5_k(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        GgmlType::Q6K => {
            let n_blocks = n_elements / 256;
            let mut out = Vec::with_capacity(n_blocks * BlockQ6K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| GgufError::InvalidData("Block slice conversion failed".into()))?;
                let block = quantize_q6_k(block_data);
                out.extend_from_slice(bytemuck::bytes_of(&block));
            }
            out
        }
        _ => {
            return Err(GgufError::InvalidData(format!(
                "Unsupported target quantization type: {:?}",
                target
            )));
        }
    };

    Ok(output)
}
