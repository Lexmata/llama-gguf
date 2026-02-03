//! CPU backend implementation
//!
//! This module provides a CPU-based implementation of the `Backend` trait
//! using SIMD operations where possible and rayon for parallelism.

pub mod flash_attn;
mod ops;
pub mod simd;

use crate::backend::{Backend, BackendResult};
use crate::tensor::{DType, Tensor};

/// CPU backend using SIMD operations and rayon for parallelism
pub struct CpuBackend {
    /// Number of threads to use for parallel operations
    num_threads: usize,
    /// Whether AVX2 is available
    has_avx2: bool,
    /// Whether AVX-512 is available
    has_avx512: bool,
    /// Whether NEON is available
    has_neon: bool,
}

impl CpuBackend {
    /// Create a new CPU backend with default thread count
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            has_avx2: simd::has_avx2(),
            has_avx512: simd::has_avx512(),
            has_neon: simd::has_neon(),
        }
    }

    /// Create a CPU backend with a specific thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            num_threads,
            has_avx2: simd::has_avx2(),
            has_avx512: simd::has_avx512(),
            has_neon: simd::has_neon(),
        }
    }

    /// Get the number of threads used by this backend
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Check if AVX2 SIMD is available
    pub fn has_avx2(&self) -> bool {
        self.has_avx2
    }

    /// Check if AVX-512 SIMD is available
    pub fn has_avx512(&self) -> bool {
        self.has_avx512
    }

    /// Check if NEON SIMD is available
    pub fn has_neon(&self) -> bool {
        self.has_neon
    }

    /// Get a string describing the SIMD capabilities
    pub fn simd_info(&self) -> String {
        let mut features = Vec::new();
        if self.has_avx512 {
            features.push("AVX-512");
        }
        if self.has_avx2 {
            features.push("AVX2");
        }
        if self.has_neon {
            features.push("NEON");
        }
        if features.is_empty() {
            "scalar".to_string()
        } else {
            features.join(", ")
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor> {
        Ok(Tensor::zeros(shape.to_vec(), dtype))
    }

    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor> {
        // CPU to CPU is just a clone
        Ok(tensor.clone())
    }

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::add(a, b, out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::mul(a, b, out)
    }

    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
        ops::scale(a, scalar, out)
    }

    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::silu(x, out)
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::gelu(x, out)
    }

    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::softmax(x, out)
    }

    fn rms_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> BackendResult<()> {
        ops::rms_norm(x, weight, eps, out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::matmul(a, b, out)
    }

    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::matvec(a, b, out)
    }

    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::vec_mat(a, b, out)
    }

    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::dequantize(src, out)
    }

    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::matvec_q(a, b, out)
    }

    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::vec_mat_q(a, b, out)
    }

    fn rope(
        &self,
        q: &mut Tensor,
        k: &mut Tensor,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
        use_neox: bool,
    ) -> BackendResult<()> {
        ops::rope(q, k, pos, freq_base, freq_scale, use_neox)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
    ) -> BackendResult<()> {
        ops::attention(q, k, v, out, scale)
    }

    fn flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
        causal: bool,
    ) -> BackendResult<()> {
        flash_attn::flash_attention(q, k, v, out, scale, causal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert!(backend.is_available());
        assert!(backend.num_threads() > 0);
    }

    #[test]
    fn test_cpu_backend_alloc() {
        let backend = CpuBackend::new();
        let tensor = backend.alloc(&[4, 4], DType::F32).unwrap();
        assert_eq!(tensor.shape(), &[4, 4]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.numel(), 16);
    }
}
