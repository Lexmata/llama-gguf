//! Tensor data types

use crate::gguf::GgmlType;

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    // Floating point types
    F32,
    F16,
    BF16,
    F64,
    // Integer types
    I8,
    I16,
    I32,
    I64,
    U8,
    // Legacy quantized types (block size 32)
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    // K-quant types (block size 256)
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // IQ (importance-weighted) quant types
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4XS,
    IQ4NL,
    IQ1S,
}

impl DType {
    /// Block size for this type (number of elements per block)
    pub const fn block_size(&self) -> usize {
        match self {
            // Non-quantized types have block size 1
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 => 1,
            // Legacy quants: 32 elements per block
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            // K-quants: 256 elements per block
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            // IQ types: 256 elements per block
            Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ2S
            | Self::IQ3XXS
            | Self::IQ3S
            | Self::IQ4XS
            | Self::IQ4NL
            | Self::IQ1S => 256,
        }
    }

    /// Bytes per block for this type
    pub const fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 | Self::U8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ4XS => 136,
            Self::IQ4NL => 132,
            Self::IQ1S => 50,
        }
    }

    /// Returns true if this is a quantized type
    pub const fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::U8
        )
    }

    /// Calculate the byte size needed for a given number of elements
    pub const fn size_for_elements(&self, n_elements: usize) -> usize {
        let block_size = self.block_size();
        let block_bytes = self.block_bytes();
        // For quantized types, elements must be a multiple of block_size
        // We round up to handle partial blocks
        let n_blocks = n_elements.div_ceil(block_size);
        n_blocks * block_bytes
    }
}

impl From<GgmlType> for DType {
    fn from(ggml_type: GgmlType) -> Self {
        match ggml_type {
            GgmlType::F32 => DType::F32,
            GgmlType::F16 => DType::F16,
            GgmlType::BF16 => DType::BF16,
            GgmlType::F64 => DType::F64,
            GgmlType::I8 => DType::I8,
            GgmlType::I16 => DType::I16,
            GgmlType::I32 => DType::I32,
            GgmlType::I64 => DType::I64,
            GgmlType::Q4_0 => DType::Q4_0,
            GgmlType::Q4_1 => DType::Q4_1,
            GgmlType::Q5_0 => DType::Q5_0,
            GgmlType::Q5_1 => DType::Q5_1,
            GgmlType::Q8_0 => DType::Q8_0,
            GgmlType::Q8_1 => DType::Q8_1,
            GgmlType::Q2K => DType::Q2K,
            GgmlType::Q3K => DType::Q3K,
            GgmlType::Q4K => DType::Q4K,
            GgmlType::Q5K => DType::Q5K,
            GgmlType::Q6K => DType::Q6K,
            GgmlType::Q8K => DType::Q8K,
            GgmlType::IQ2XXS => DType::IQ2XXS,
            GgmlType::IQ2XS => DType::IQ2XS,
            GgmlType::IQ2S => DType::IQ2S,
            GgmlType::IQ3XXS => DType::IQ3XXS,
            GgmlType::IQ3S => DType::IQ3S,
            GgmlType::IQ4XS => DType::IQ4XS,
            GgmlType::IQ4NL => DType::IQ4NL,
            GgmlType::IQ1S => DType::IQ1S,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_for_elements() {
        // F32: 4 bytes per element
        assert_eq!(DType::F32.size_for_elements(10), 40);

        // Q4_0: 18 bytes per 32 elements
        assert_eq!(DType::Q4_0.size_for_elements(32), 18);
        assert_eq!(DType::Q4_0.size_for_elements(64), 36);

        // Q4K: 144 bytes per 256 elements
        assert_eq!(DType::Q4K.size_for_elements(256), 144);
        assert_eq!(DType::Q4K.size_for_elements(512), 288);
    }

    #[test]
    fn test_is_quantized() {
        assert!(!DType::F32.is_quantized());
        assert!(!DType::F16.is_quantized());
        assert!(!DType::I32.is_quantized());
        assert!(DType::Q4_0.is_quantized());
        assert!(DType::Q4K.is_quantized());
        assert!(DType::IQ2XXS.is_quantized());
    }

    #[test]
    fn test_from_ggml_type() {
        assert_eq!(DType::from(GgmlType::F32), DType::F32);
        assert_eq!(DType::from(GgmlType::Q4_0), DType::Q4_0);
        assert_eq!(DType::from(GgmlType::Q4K), DType::Q4K);
    }
}
