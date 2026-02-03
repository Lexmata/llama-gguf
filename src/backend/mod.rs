//! Hardware backends for tensor operations
//!
//! This module defines the `Backend` trait which provides an abstraction
//! over different hardware implementations (CPU, CUDA, Vulkan, Metal, etc.)

mod error;
pub mod cpu;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use error::BackendError;

use crate::tensor::{DType, Tensor};

/// Result type for backend operations
pub type BackendResult<T> = Result<T, BackendError>;

/// Hardware backend trait for tensor operations
///
/// This trait defines all the operations needed for LLM inference.
/// Each backend (CPU, CUDA, etc.) implements this trait.
pub trait Backend: Send + Sync {
    /// Get the name of this backend
    fn name(&self) -> &str;

    /// Check if this backend is available on the current system
    fn is_available(&self) -> bool;

    // =========================================================================
    // Memory operations
    // =========================================================================

    /// Allocate a tensor with the given shape and dtype
    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor>;

    /// Copy a tensor to this backend (may be a no-op for CPU)
    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor>;

    // =========================================================================
    // Element-wise operations
    // =========================================================================

    /// Element-wise addition: out = a + b
    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Element-wise multiplication: out = a * b
    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Scale tensor by scalar: out = a * scalar
    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Activation functions
    // =========================================================================

    /// SiLU activation: out = x * sigmoid(x)
    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// GELU activation
    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Softmax along last dimension
    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Normalization
    // =========================================================================

    /// RMS normalization: out = x / rms(x) * weight
    fn rms_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> BackendResult<()>;

    // =========================================================================
    // Matrix operations
    // =========================================================================

    /// Matrix multiplication: out = a @ b
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Matrix-vector multiplication: out = a @ b where a is 2D, b is 1D
    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Vector-matrix multiplication: out = a @ b where a is 1D, b is 2D
    /// This computes y = x @ W where x is [k] and W is [k, n], giving y [n]
    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Quantization
    // =========================================================================

    /// Dequantize tensor to f32
    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Quantized matrix-vector multiply (fused dequant + matvec for performance)
    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Quantized vector-matrix multiply (fused dequant + vec_mat for performance)
    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Position embeddings
    // =========================================================================

    /// Apply Rotary Position Embedding (RoPE) to query and key tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [num_heads, seq_len, head_dim]
    /// * `k` - Key tensor of shape [num_kv_heads, seq_len, head_dim]
    /// * `pos` - Starting position for RoPE
    /// * `freq_base` - Base frequency (typically 10000.0)
    /// * `freq_scale` - Frequency scale factor (typically 1.0)
    /// * `use_neox` - Use NeoX style (first half with second half) vs Normal (consecutive pairs)
    fn rope(
        &self,
        q: &mut Tensor,
        k: &mut Tensor,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
        use_neox: bool,
    ) -> BackendResult<()>;

    // =========================================================================
    // Attention operations
    // =========================================================================

    /// Compute causal self-attention
    ///
    /// # Arguments
    /// * `q` - Query tensor [num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [num_kv_heads, kv_len, head_dim]
    /// * `v` - Value tensor [num_kv_heads, kv_len, head_dim]
    /// * `out` - Output tensor [num_heads, seq_len, head_dim]
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
    ) -> BackendResult<()>;

    /// Compute Flash Attention (memory-efficient tiled attention)
    ///
    /// Flash Attention computes attention using tiling to reduce memory usage
    /// from O(n²) to O(n). This is especially beneficial for long sequences.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_kv_heads, kv_len, head_dim]
    /// * `v` - Value tensor [batch, num_kv_heads, kv_len, head_dim]
    /// * `out` - Output tensor [batch, num_heads, seq_len, head_dim]
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
    /// * `causal` - Whether to apply causal masking
    fn flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
        _causal: bool,
    ) -> BackendResult<()> {
        // Default implementation falls back to standard attention
        // Backends can override this with optimized implementations
        self.attention(q, k, v, out, scale)
    }
}

/// Get the default backend (CPU)
pub fn default_backend() -> Box<dyn Backend> {
    Box::new(cpu::CpuBackend::new())
}
