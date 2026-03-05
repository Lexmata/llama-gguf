//! GPU-side quantization
//!
//! Quantize F32 tensors directly on GPU without downloading to host.
//! Supports Q8_0, Q4_K, and Q6_K on all GPU backends.

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// Supported GPU quantization formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuQuantFormat {
    Q8_0,
    Q4K,
    Q6K,
}

impl GpuQuantFormat {
    pub fn to_dtype(&self) -> DType {
        match self {
            Self::Q8_0 => DType::Q8_0,
            Self::Q4K => DType::Q4K,
            Self::Q6K => DType::Q6K,
        }
    }
}

/// Trait for backends that support GPU-side quantization
pub trait GpuQuantize: Backend {
    /// Quantize an F32 tensor to the given format on the GPU
    fn quantize_on_device(&self, input: &Tensor, format: GpuQuantFormat) -> BackendResult<Tensor>;

    /// Check if this backend supports GPU quantization for a given format
    fn supports_gpu_quantize(&self, format: GpuQuantFormat) -> bool;
}

/// CPU fallback implementation of GPU quantization
pub fn quantize_cpu_fallback(input: &Tensor, format: GpuQuantFormat) -> BackendResult<Tensor> {
    let data = input
        .as_f32()
        .map_err(|e| BackendError::OperationFailed(e.to_string()))?;
    let target_dtype = format.to_dtype();
    let block_size = target_dtype.block_size();
    let n_elements = data.len();

    if n_elements % block_size != 0 {
        return Err(BackendError::InvalidArgument(format!(
            "Input size {} not divisible by block size {}",
            n_elements, block_size
        )));
    }

    let quantized_bytes = match format {
        GpuQuantFormat::Q8_0 => {
            use crate::tensor::quant::{quantize_q8_0, BlockQ8_0};
            let n_blocks = n_elements / 32;
            let mut output = Vec::with_capacity(n_blocks * BlockQ8_0::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 32] = data[i * 32..(i + 1) * 32]
                    .try_into()
                    .map_err(|_| BackendError::OperationFailed("Slice conversion failed".into()))?;
                let block = quantize_q8_0(block_data);
                output.extend_from_slice(bytemuck::bytes_of(&block));
            }
            output
        }
        GpuQuantFormat::Q4K => {
            use crate::tensor::quant::{quantize_q4_k, BlockQ4K};
            let n_blocks = n_elements / 256;
            let mut output = Vec::with_capacity(n_blocks * BlockQ4K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| BackendError::OperationFailed("Slice conversion failed".into()))?;
                let block = quantize_q4_k(block_data);
                output.extend_from_slice(bytemuck::bytes_of(&block));
            }
            output
        }
        GpuQuantFormat::Q6K => {
            use crate::tensor::quant::{quantize_q6_k, BlockQ6K};
            let n_blocks = n_elements / 256;
            let mut output = Vec::with_capacity(n_blocks * BlockQ6K::TYPE_SIZE);
            for i in 0..n_blocks {
                let block_data: &[f32; 256] = data[i * 256..(i + 1) * 256]
                    .try_into()
                    .map_err(|_| BackendError::OperationFailed("Slice conversion failed".into()))?;
                let block = quantize_q6_k(block_data);
                output.extend_from_slice(bytemuck::bytes_of(&block));
            }
            output
        }
    };

    Tensor::new(quantized_bytes, vec![n_elements], target_dtype)
        .map_err(|e| BackendError::OperationFailed(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_quant_format_to_dtype() {
        assert_eq!(GpuQuantFormat::Q8_0.to_dtype(), DType::Q8_0);
        assert_eq!(GpuQuantFormat::Q4K.to_dtype(), DType::Q4K);
        assert_eq!(GpuQuantFormat::Q6K.to_dtype(), DType::Q6K);
    }

    #[test]
    fn test_cpu_fallback_q8_0() {
        let f32_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::from_f32(&f32_data, vec![64]).unwrap();
        let result = quantize_cpu_fallback(&input, GpuQuantFormat::Q8_0).unwrap();
        assert_eq!(result.dtype(), DType::Q8_0);
        assert_eq!(result.shape(), &[64]);
        let expected_size = DType::Q8_0.size_for_elements(64);
        assert_eq!(result.data().len(), expected_size);
    }
}
