//! CPU backend tensor operations
//!
//! This module implements all tensor operations for the CPU backend.
//! Operations use rayon for parallelism where beneficial.

use crate::backend::{BackendError, BackendResult};
use crate::tensor::quant::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8_1, BlockQ8K, dequantize_q2_k, dequantize_q3_k, dequantize_q4_0,
    dequantize_q4_1, dequantize_q4_k, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k,
    dequantize_q6_k, dequantize_q8_0, dequantize_q8_1, dequantize_q8_k,
};
use crate::tensor::{DType, Tensor};
use rayon::prelude::*;

// =============================================================================
// Element-wise Operations
// =============================================================================

// Threshold for using parallel execution (avoid rayon overhead for small arrays)
const PARALLEL_THRESHOLD: usize = 8192;

/// Element-wise addition: out = a + b
pub fn add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(a, b)?;
    check_same_shape(a, out)?;
    check_dtype(a, DType::F32)?;

    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    if out_data.len() >= PARALLEL_THRESHOLD {
        out_data
            .par_iter_mut()
            .zip(a_data.par_iter().zip(b_data.par_iter()))
            .for_each(|(o, (&a, &b))| *o = a + b);
    } else {
        // Sequential with SIMD
        add_f32_simd(a_data, b_data, out_data);
    }

    Ok(())
}

/// SIMD-optimized element-wise add
fn add_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::simd::has_avx2() {
        unsafe { add_f32_avx2(a, b, out) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_f32_neon(a, b, out) };
        return;
    }

    // Scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for ((o, &a_val), &b_val) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = a_val + b_val;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            let vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out_ptr.add(offset), vr);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let vr = vaddq_f32(va, vb);
        vst1q_f32(out_ptr.add(offset), vr);
    }

    for i in (chunks * 4)..n {
        *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
    }
}

/// Element-wise multiplication: out = a * b
pub fn mul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(a, b)?;
    check_same_shape(a, out)?;
    check_dtype(a, DType::F32)?;

    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    if out_data.len() >= PARALLEL_THRESHOLD {
        out_data
            .par_iter_mut()
            .zip(a_data.par_iter().zip(b_data.par_iter()))
            .for_each(|(o, (&a, &b))| *o = a * b);
    } else {
        mul_f32_simd(a_data, b_data, out_data);
    }

    Ok(())
}

/// SIMD-optimized element-wise mul
fn mul_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::simd::has_avx2() {
        unsafe { mul_f32_avx2(a, b, out) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { mul_f32_neon(a, b, out) };
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for ((o, &a_val), &b_val) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = a_val * b_val;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            let vr = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out_ptr.add(offset), vr);
        }

        for i in (chunks * 8)..n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let vr = vmulq_f32(va, vb);
        vst1q_f32(out_ptr.add(offset), vr);
    }

    for i in (chunks * 4)..n {
        *out.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
    }
}

/// Scale by scalar: out = a * scalar
pub fn scale(a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(a, out)?;
    check_dtype(a, DType::F32)?;

    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;

    if out_data.len() >= PARALLEL_THRESHOLD {
        out_data
            .par_iter_mut()
            .zip(a_data.par_iter())
            .for_each(|(o, &a)| *o = a * scalar);
    } else {
        scale_f32_simd(a_data, scalar, out_data);
    }

    Ok(())
}

/// SIMD-optimized scale
fn scale_f32_simd(a: &[f32], scalar: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::simd::has_avx2() {
        unsafe { scale_f32_avx2(a, scalar, out) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { scale_f32_neon(a, scalar, out) };
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for (o, &a_val) in out.iter_mut().zip(a.iter()) {
        *o = a_val * scalar;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_f32_avx2(a: &[f32], scalar: f32, out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let vscalar = _mm256_set1_ps(scalar);

        let a_ptr = a.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vr = _mm256_mul_ps(va, vscalar);
            _mm256_storeu_ps(out_ptr.add(offset), vr);
        }

        for i in (chunks * 8)..n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * scalar;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn scale_f32_neon(a: &[f32], scalar: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let vscalar = vdupq_n_f32(scalar);

    let a_ptr = a.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vr = vmulq_f32(va, vscalar);
        vst1q_f32(out_ptr.add(offset), vr);
    }

    for i in (chunks * 4)..n {
        *out.get_unchecked_mut(i) = *a.get_unchecked(i) * scalar;
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

/// SiLU (Swish) activation: out = x * sigmoid(x)
pub fn silu(x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;

    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;

    if out_data.len() >= PARALLEL_THRESHOLD {
        out_data
            .par_iter_mut()
            .zip(x_data.par_iter())
            .for_each(|(o, &x)| {
                *o = x / (1.0 + (-x).exp());
            });
    } else {
        // Sequential - avoid rayon overhead
        for (o, &x) in out_data.iter_mut().zip(x_data.iter()) {
            *o = x / (1.0 + (-x).exp());
        }
    }

    Ok(())
}

/// GELU activation (approximation used in transformers)
pub fn gelu(x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;

    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;

    out_data
        .par_iter_mut()
        .zip(x_data.par_iter())
        .for_each(|(o, &x)| {
            let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
            *o = 0.5 * x * (1.0 + inner.tanh());
        });

    Ok(())
}

/// Softmax along last dimension (SIMD-optimized)
pub fn softmax(x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;

    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let last_dim = *x.shape().last().unwrap_or(&1);
    let n_rows = x.numel() / last_dim;

    // Process each row
    for row in 0..n_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_x = &x_data[start..end];
        let row_out = &mut out_data[start..end];

        // Use SIMD-optimized max
        let max = super::simd::max_f32(row_x);

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for (o, &x) in row_out.iter_mut().zip(row_x.iter()) {
            *o = (x - max).exp();
            sum += *o;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for o in row_out.iter_mut() {
            *o *= inv_sum;
        }
    }

    Ok(())
}

// =============================================================================
// Normalization
// =============================================================================

/// RMS normalization: out = x / rms(x) * weight (SIMD-optimized)
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;
    check_dtype(weight, DType::F32)?;

    let x_data = x.as_f32()?;
    let w_data = weight.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let hidden_size = *x.shape().last().unwrap_or(&1);
    let n_rows = x.numel() / hidden_size;

    if w_data.len() != hidden_size {
        return Err(BackendError::ShapeMismatch {
            expected: vec![hidden_size],
            got: weight.shape().to_vec(),
        });
    }

    for row in 0..n_rows {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let row_x = &x_data[start..end];
        let row_out = &mut out_data[start..end];

        // Use SIMD-optimized RMS norm
        super::simd::rms_norm(row_x, w_data, eps, row_out);
    }

    Ok(())
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// Matrix multiplication: out = a @ b (2D matrices)
pub fn matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(a, DType::F32)?;
    check_dtype(b, DType::F32)?;
    check_dtype(out, DType::F32)?;

    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(BackendError::InvalidArgument(
            "matmul requires 2D tensors".into(),
        ));
    }

    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);

    if k1 != k2 {
        return Err(BackendError::ShapeMismatch {
            expected: vec![m, k1],
            got: vec![k2, n],
        });
    }

    if out.shape() != [m, n] {
        return Err(BackendError::ShapeMismatch {
            expected: vec![m, n],
            got: out.shape().to_vec(),
        });
    }

    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    // Use different strategies based on matrix size
    let total_ops = m * k1 * n;

    if total_ops < 256 * 256 * 256 {
        // Simple parallel implementation for small matrices
        matmul_simple(a_data, b_data, out_data, m, k1, n);
    } else {
        // Tiled matmul for large matrices
        matmul_tiled(a_data, b_data, out_data, m, k1, n);
    }

    Ok(())
}

/// Simple parallel matmul for small matrices
fn matmul_simple(a: &[f32], b: &[f32], c: &mut [f32], _m: usize, k: usize, n: usize) {
    c.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        for (j, out_val) in row_out.iter_mut().enumerate().take(n) {
            let mut sum = 0.0f32;
            let a_row = i * k;
            for kk in 0..k {
                sum += unsafe { *a.get_unchecked(a_row + kk) * *b.get_unchecked(kk * n + j) };
            }
            *out_val = sum;
        }
    });
}

/// Tiled matrix multiplication for better cache utilization (large matrices)
fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], _m: usize, k: usize, n: usize) {
    // Tile sizes tuned for L2 cache (~256KB)
    const _TILE_M: usize = 32;
    const TILE_N: usize = 256;
    const TILE_K: usize = 32;

    // Initialize output to zero
    c.iter_mut().for_each(|x| *x = 0.0);

    // Process rows in parallel
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        for kk in (0..k).step_by(TILE_K) {
            let k_end = (kk + TILE_K).min(k);

            for jj in (0..n).step_by(TILE_N) {
                let j_end = (jj + TILE_N).min(n);

                let a_row = i * k;

                for (j, c_val) in c_row.iter_mut().enumerate().take(j_end).skip(jj) {
                    let mut sum = *c_val;

                    for kk_inner in kk..k_end {
                        sum += unsafe {
                            *a.get_unchecked(a_row + kk_inner) * *b.get_unchecked(kk_inner * n + j)
                        };
                    }

                    *c_val = sum;
                }
            }
        }
    });
}

/// Matrix-vector multiplication: out = a @ b where b is 1D (SIMD-optimized)
///
/// Computes y = A @ x where A is [m, k] and x is [k], giving y [m]
///
/// Uses row-major storage (standard Rust/C convention).
/// For shape [m, k], element (i, j) is at index i * k + j.
pub fn matvec(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(a, DType::F32)?;
    check_dtype(b, DType::F32)?;
    check_dtype(out, DType::F32)?;

    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(BackendError::InvalidArgument(
            "matvec requires 2D matrix and 1D vector".into(),
        ));
    }

    let (m, k) = (a.shape()[0], a.shape()[1]);
    if b.shape()[0] != k {
        return Err(BackendError::ShapeMismatch {
            expected: vec![k],
            got: b.shape().to_vec(),
        });
    }

    if out.shape() != [m] {
        return Err(BackendError::ShapeMismatch {
            expected: vec![m],
            got: out.shape().to_vec(),
        });
    }

    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    // Parallel over rows with SIMD dot product
    out_data.par_iter_mut().enumerate().for_each(|(i, o)| {
        let row_start = i * k;
        let row_end = row_start + k;
        *o = super::simd::dot_f32(&a_data[row_start..row_end], b_data);
    });

    Ok(())
}

// =============================================================================
// Quantization Operations
// =============================================================================

/// Dequantize a quantized tensor to f32
pub fn dequantize(src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(out, DType::F32)?;

    match src.dtype() {
        DType::Q4_0 => {
            let blocks: &[BlockQ4_0] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 32 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 32],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 32];
                dequantize_q4_0(block, &mut tmp);
                let start = i * 32;
                // Safe because we're writing to non-overlapping regions
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        32,
                    );
                }
            });

            Ok(())
        }
        DType::Q4_1 => {
            let blocks: &[BlockQ4_1] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 32 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 32],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 32];
                dequantize_q4_1(block, &mut tmp);
                let start = i * 32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        32,
                    );
                }
            });

            Ok(())
        }
        DType::Q5_0 => {
            let blocks: &[BlockQ5_0] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 32 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 32],
                    got: vec![out_data.len()],
                });
            }

            // Parallel dequantization
            out_data
                .par_chunks_mut(32)
                .zip(blocks.par_iter())
                .for_each(|(chunk, block)| {
                    let mut tmp = [0.0f32; 32];
                    dequantize_q5_0(block, &mut tmp);
                    chunk.copy_from_slice(&tmp);
                });

            Ok(())
        }
        DType::Q5_1 => {
            let blocks: &[BlockQ5_1] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 32 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 32],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 32];
                dequantize_q5_1(block, &mut tmp);
                let start = i * 32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        32,
                    );
                }
            });

            Ok(())
        }
        DType::Q8_0 => {
            let blocks: &[BlockQ8_0] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 32 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 32],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 32];
                dequantize_q8_0(block, &mut tmp);
                let start = i * 32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        32,
                    );
                }
            });

            Ok(())
        }
        DType::Q8_1 => {
            let blocks: &[BlockQ8_1] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 32 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 32],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 32];
                dequantize_q8_1(block, &mut tmp);
                let start = i * 32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        32,
                    );
                }
            });

            Ok(())
        }
        DType::Q2K => {
            let blocks: &[BlockQ2K] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 256 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 256],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 256];
                dequantize_q2_k(block, &mut tmp);
                let start = i * 256;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        256,
                    );
                }
            });

            Ok(())
        }
        DType::Q3K => {
            let blocks: &[BlockQ3K] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 256 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 256],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 256];
                dequantize_q3_k(block, &mut tmp);
                let start = i * 256;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        256,
                    );
                }
            });

            Ok(())
        }
        DType::Q4K => {
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 256 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 256],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 256];
                dequantize_q4_k(block, &mut tmp);
                let start = i * 256;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        256,
                    );
                }
            });

            Ok(())
        }
        DType::Q5K => {
            let blocks: &[BlockQ5K] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 256 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 256],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 256];
                dequantize_q5_k(block, &mut tmp);
                let start = i * 256;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        256,
                    );
                }
            });

            Ok(())
        }
        DType::Q6K => {
            let blocks: &[BlockQ6K] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 256 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 256],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 256];
                dequantize_q6_k(block, &mut tmp);
                let start = i * 256;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        256,
                    );
                }
            });

            Ok(())
        }
        DType::Q8K => {
            let blocks: &[BlockQ8K] = bytemuck::cast_slice(src.data());
            let out_data = out.as_f32_mut()?;

            if out_data.len() != blocks.len() * 256 {
                return Err(BackendError::ShapeMismatch {
                    expected: vec![blocks.len() * 256],
                    got: vec![out_data.len()],
                });
            }

            blocks.par_iter().enumerate().for_each(|(i, block)| {
                let mut tmp = [0.0f32; 256];
                dequantize_q8_k(block, &mut tmp);
                let start = i * 256;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr(),
                        out_data.as_ptr().add(start) as *mut f32,
                        256,
                    );
                }
            });

            Ok(())
        }
        DType::F32 => {
            // Just copy
            let src_data = src.as_f32()?;
            let out_data = out.as_f32_mut()?;
            out_data.copy_from_slice(src_data);
            Ok(())
        }
        dtype => Err(BackendError::UnsupportedDType(dtype)),
    }
}

/// Quantized matrix-vector multiply
///
/// For now, this dequantizes and uses regular matvec.
/// TODO: Implement fused quantized matmul for better performance.
pub fn matvec_q(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    // For now, dequantize first then use regular matvec
    // This is not optimal but correct

    // Dequantize the matrix first, then use regular matvec
    let mut a_f32 = Tensor::zeros(a.shape().to_vec(), DType::F32);

    dequantize(a, &mut a_f32)?;
    matvec(&a_f32, b, out)
}

/// Vector-matrix multiplication: out = a @ b where a is 1D, b is 2D
///
/// Computes y = x @ W where x is [k] and W is [k, n], giving y [n]
///
/// GGUF stores matrices in column-major order (ne[0] is contiguous).
/// For shape [k, n], element (i, j) is at index i + j * k.
pub fn vec_mat(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(a, DType::F32)?;
    check_dtype(b, DType::F32)?;
    check_dtype(out, DType::F32)?;

    if a.ndim() != 1 || b.ndim() != 2 {
        return Err(BackendError::InvalidArgument(
            "vec_mat requires 1D vector and 2D matrix".into(),
        ));
    }

    let k = a.shape()[0];
    let (k2, n) = (b.shape()[0], b.shape()[1]);

    if k != k2 {
        return Err(BackendError::ShapeMismatch {
            expected: vec![k],
            got: vec![k2],
        });
    }

    if out.shape() != [n] {
        return Err(BackendError::ShapeMismatch {
            expected: vec![n],
            got: out.shape().to_vec(),
        });
    }

    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    // Compute y[j] = sum_i x[i] * W[i,j] for each j
    // GGUF column-major: W[i,j] is at index i + j * k
    out_data.par_iter_mut().enumerate().for_each(|(j, o)| {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += a_data[i] * b_data[i + j * k];
        }
        *o = sum;
    });

    Ok(())
}

/// Quantized vector-matrix multiply
///
/// For now, this dequantizes and uses regular vec_mat.
/// TODO: Implement fused quantized vecmat for better performance.
pub fn vec_mat_q(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    // Dequantize the matrix first, then use regular vec_mat
    let mut b_f32 = Tensor::zeros(b.shape().to_vec(), DType::F32);

    dequantize(b, &mut b_f32)?;
    vec_mat(a, &b_f32, out)
}

// =============================================================================
// Position Embeddings
// =============================================================================

/// Apply Rotary Position Embedding (RoPE) to query and key tensors
///
/// RoPE applies rotation in 2D subspaces of the embedding based on position.
/// This enables the model to learn relative positions through the attention mechanism.
///
/// # Arguments
/// * `q` - Query tensor, modified in place. Shape: [num_heads, seq_len, head_dim]
/// * `k` - Key tensor, modified in place. Shape: [num_kv_heads, seq_len, head_dim]
/// * `pos` - Starting position in the sequence
/// * `freq_base` - Base frequency (typically 10000.0)
/// * `freq_scale` - Frequency scale factor (typically 1.0)
pub fn rope(
    q: &mut Tensor,
    k: &mut Tensor,
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
    use_neox: bool,
) -> BackendResult<()> {
    check_dtype(q, DType::F32)?;
    check_dtype(k, DType::F32)?;

    if q.ndim() != 3 || k.ndim() != 3 {
        return Err(BackendError::InvalidArgument(
            "RoPE requires 3D tensors [num_heads, seq_len, head_dim]".into(),
        ));
    }

    // Capture shape values before mutable borrow
    let (q_num_heads, q_seq_len, q_head_dim) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let (k_num_kv_heads, k_seq_len, k_head_dim) = (k.shape()[0], k.shape()[1], k.shape()[2]);

    if k_seq_len != q_seq_len || k_head_dim != q_head_dim {
        return Err(BackendError::InvalidArgument(
            "Q and K must have same seq_len and head_dim".into(),
        ));
    }

    // Apply RoPE to Q
    {
        let q_data = q.as_f32_mut()?;
        apply_rope_to_tensor(
            q_data,
            q_num_heads,
            q_seq_len,
            q_head_dim,
            pos,
            freq_base,
            freq_scale,
            use_neox,
        );
    }

    // Apply RoPE to K
    {
        let k_data = k.as_f32_mut()?;
        apply_rope_to_tensor(
            k_data,
            k_num_kv_heads,
            k_seq_len,
            k_head_dim,
            pos,
            freq_base,
            freq_scale,
            use_neox,
        );
    }

    Ok(())
}

/// Internal function to apply RoPE to a tensor
///
/// Supports two RoPE styles:
/// - LLaMA/Standard (GGML_ROPE_TYPE_NORMAL): consecutive pairs (x[2i], x[2i+1])
/// - NeoX (GGML_ROPE_TYPE_NEOX): first half paired with second half (x[i], x[i+d/2])
///
/// Qwen2 uses NeoX style RoPE.
#[allow(clippy::too_many_arguments)]
fn apply_rope_to_tensor(
    data: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
    use_neox: bool,
) {
    let half_dim = head_dim / 2;

    for head in 0..num_heads {
        for s in 0..seq_len {
            // For linear scaling, positions are divided by the scale factor
            // (e.g., if scale_linear=4.0, position 4 becomes 1)
            let position = (pos + s) as f32 / freq_scale;
            let head_offset = head * seq_len * head_dim + s * head_dim;

            for i in 0..half_dim {
                // Compute the rotation frequency for this dimension
                // Frequency decreases as dimension index increases
                let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
                let theta = position * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                if use_neox {
                    // NeoX style: pairs are (i, i+half_dim)
                    let idx0 = head_offset + i;
                    let idx1 = head_offset + i + half_dim;

                    let x0 = data[idx0];
                    let x1 = data[idx1];

                    // Apply rotation
                    data[idx0] = x0 * cos_theta - x1 * sin_theta;
                    data[idx1] = x0 * sin_theta + x1 * cos_theta;
                } else {
                    // Normal/LLaMA style: consecutive pairs (2i, 2i+1)
                    let idx0 = head_offset + 2 * i;
                    let idx1 = head_offset + 2 * i + 1;

                    let x0 = data[idx0];
                    let x1 = data[idx1];

                    // Apply rotation
                    data[idx0] = x0 * cos_theta - x1 * sin_theta;
                    data[idx1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
}

// =============================================================================
// Attention Operations
// =============================================================================

/// Compute causal self-attention
///
/// Computes: softmax(Q @ K^T / scale) @ V with causal masking
///
/// # Arguments
/// * `q` - Query tensor [num_heads, seq_len, head_dim]
/// * `k` - Key tensor [num_kv_heads, kv_len, head_dim]
/// * `v` - Value tensor [num_kv_heads, kv_len, head_dim]
/// * `out` - Output tensor [num_heads, seq_len, head_dim]
/// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
    scale: f32,
) -> BackendResult<()> {
    check_dtype(q, DType::F32)?;
    check_dtype(k, DType::F32)?;
    check_dtype(v, DType::F32)?;
    check_dtype(out, DType::F32)?;

    if q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 {
        return Err(BackendError::InvalidArgument(
            "Attention requires 3D tensors".into(),
        ));
    }

    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    let num_heads = q_shape[0];
    let seq_len = q_shape[1];
    let head_dim = q_shape[2];
    let num_kv_heads = k_shape[0];
    let kv_len = k_shape[1];

    if k_shape[2] != head_dim
        || v_shape[0] != num_kv_heads
        || v_shape[1] != kv_len
        || v_shape[2] != head_dim
    {
        return Err(BackendError::InvalidArgument(
            "Attention tensor dimension mismatch".into(),
        ));
    }

    // Handle Grouped Query Attention (GQA): num_heads may be > num_kv_heads
    let num_queries_per_kv = num_heads / num_kv_heads;

    let q_data = q.as_f32()?;
    let k_data = k.as_f32()?;
    let v_data = v.as_f32()?;
    let out_data = out.as_f32_mut()?;

    // Process each head
    for head in 0..num_heads {
        let kv_head = head / num_queries_per_kv; // Map to KV head for GQA

        for s in 0..seq_len {
            // Compute attention scores for this query position
            let q_offset = head * seq_len * head_dim + s * head_dim;
            let q_vec = &q_data[q_offset..q_offset + head_dim];

            // Attention scores: Q @ K^T
            let mut scores = vec![0.0f32; kv_len];
            for (kv_pos, score) in scores.iter_mut().enumerate() {
                // Causal mask: only attend to positions <= current position
                // For inference: current query is at position (kv_len - seq_len + s)
                let q_abs_pos = kv_len.saturating_sub(seq_len) + s;
                if kv_pos > q_abs_pos {
                    *score = f32::NEG_INFINITY;
                    continue;
                }

                let k_offset = kv_head * kv_len * head_dim + kv_pos * head_dim;
                let k_vec = &k_data[k_offset..k_offset + head_dim];

                // Dot product
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_vec[d];
                }
                *score = dot * scale;
            }

            // Softmax on scores
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let inv_sum = 1.0 / sum;
            for score in &mut scores {
                *score *= inv_sum;
            }

            // Weighted sum of values
            let out_offset = head * seq_len * head_dim + s * head_dim;
            let out_vec = &mut out_data[out_offset..out_offset + head_dim];
            out_vec.fill(0.0);

            for (kv_pos, &score_val) in scores.iter().enumerate() {
                if score_val > 0.0 {
                    let v_offset = kv_head * kv_len * head_dim + kv_pos * head_dim;
                    let v_vec = &v_data[v_offset..v_offset + head_dim];

                    for d in 0..head_dim {
                        out_vec[d] += score_val * v_vec[d];
                    }
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn check_same_shape(a: &Tensor, b: &Tensor) -> BackendResult<()> {
    if a.shape() != b.shape() {
        return Err(BackendError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

fn check_dtype(t: &Tensor, expected: DType) -> BackendResult<()> {
    if t.dtype() != expected {
        return Err(BackendError::DTypeMismatch {
            expected,
            got: t.dtype(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        add(&a, &b, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        mul(&a, &b, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        assert_eq!(result, &[2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        scale(&a, 2.5, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        assert_eq!(result, &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_silu() {
        let x = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        silu(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // SiLU(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-6);
        // SiLU(1) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.01);
        // SiLU(-1) ≈ -0.269
        assert!((result[2] - (-0.269)).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        softmax(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    #[test]
    fn test_rms_norm() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let weight = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        rms_norm(&x, &weight, 1e-5, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.74
        // So output ≈ [0.37, 0.73, 1.10, 1.46]
        assert!((result[0] - 0.365).abs() < 0.01);
        assert!((result[3] - 1.46).abs() < 0.01);
    }

    #[test]
    fn test_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let mut out = Tensor::zeros(vec![2, 2], DType::F32);

        matmul(&a, &b, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
        // = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        // = [[22, 28], [49, 64]]
        assert_eq!(result, &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matvec() {
        // 3x4 @ 4 = 3
        let a = Tensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4],
        )
        .unwrap();
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![3], DType::F32);

        matvec(&a, &b, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // [1,2,3,4] · [1,2,3,4] = 30
        // [5,6,7,8] · [1,2,3,4] = 70
        // [9,10,11,12] · [1,2,3,4] = 110
        assert_eq!(result, &[30.0, 70.0, 110.0]);
    }

    #[test]
    fn test_rope() {
        // Create Q and K tensors with shape [2 heads, 1 seq, 4 head_dim]
        let q_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let k_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];

        let mut q = Tensor::from_f32(&q_data, vec![2, 1, 4]).unwrap();
        let mut k = Tensor::from_f32(&k_data, vec![2, 1, 4]).unwrap();

        // Apply RoPE at position 0 (use NeoX style for this test)
        rope(&mut q, &mut k, 0, 10000.0, 1.0, true).unwrap();

        // At position 0, rotation is by angle 0 for all dimensions
        // cos(0) = 1, sin(0) = 0, so values should be unchanged
        let q_result = q.as_f32().unwrap();
        assert!((q_result[0] - 1.0).abs() < 1e-5);
        assert!((q_result[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_position() {
        // Test that position affects the rotation
        let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];

        let mut q = Tensor::from_f32(&q_data, vec![1, 1, 4]).unwrap();
        let mut k = Tensor::from_f32(&k_data, vec![1, 1, 4]).unwrap();

        // Apply RoPE at position 1 (use Normal/LLaMA style for this test)
        rope(&mut q, &mut k, 1, 10000.0, 1.0, false).unwrap();

        let q_result = q.as_f32().unwrap();
        // At position 1, there should be some rotation
        // For consecutive pairing (LLaMA style): pairs are (0,1) and (2,3)
        // theta = 1 / 10000^0 = 1 for the first pair
        // x'[0] = x[0]*cos(1) - x[1]*sin(1) = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.54
        assert!((q_result[0] - 0.54).abs() < 0.02);
    }

    #[test]
    fn test_rope_consecutive_pairing() {
        // Test that RoPE uses consecutive pairing (LLaMA style)
        // Input: [1, 2, 3, 4] for head_dim=4
        // With NeoX pairing: pairs are (0,2) and (1,3) (first half paired with second half)
        // At position 1:
        //   theta_0 = 1 / 10000^0 = 1
        //   theta_1 = 1 / 10000^(2/4) = 1/100 = 0.01
        //
        // With x = [1, 2, 3, 4]:
        //   x'[0] = x[0]*cos(1) - x[2]*sin(1) = 1*0.5403 - 3*0.8415 ≈ -1.98
        //   x'[2] = x[0]*sin(1) + x[2]*cos(1) = 1*0.8415 + 3*0.5403 ≈ 2.46
        //   x'[1] = x[1]*cos(0.01) - x[3]*sin(0.01) = 2*1 - 4*0.01 ≈ 1.96
        //   x'[3] = x[1]*sin(0.01) + x[3]*cos(0.01) = 2*0.01 + 4*1 ≈ 4.02

        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let k_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let mut q = Tensor::from_f32(&q_data, vec![1, 1, 4]).unwrap();
        let mut k = Tensor::from_f32(&k_data, vec![1, 1, 4]).unwrap();

        // Use NeoX style RoPE for this test
        rope(&mut q, &mut k, 1, 10000.0, 1.0, true).unwrap();

        let q_result = q.as_f32().unwrap();

        // Verify NeoX pairing behavior
        // First pair (0,2): significant rotation (theta=1)
        assert!(
            (q_result[0] - (-1.98)).abs() < 0.05,
            "q[0]={} expected ~-1.98",
            q_result[0]
        );
        assert!(
            (q_result[2] - 2.46).abs() < 0.05,
            "q[2]={} expected ~2.46",
            q_result[2]
        );

        // Second pair (1,3): minimal rotation (theta=0.01)
        assert!(
            (q_result[1] - 1.96).abs() < 0.05,
            "q[1]={} expected ~1.96",
            q_result[1]
        );
        assert!(
            (q_result[3] - 4.02).abs() < 0.05,
            "q[3]={} expected ~4.02",
            q_result[3]
        );
    }

    #[test]
    fn test_attention_simple() {
        // Simple attention test: 1 head, 2 positions, 4 head_dim
        let q = Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![1, 2, 4]).unwrap();
        let k = Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![1, 2, 4]).unwrap();
        let v = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 4]).unwrap();
        let mut out = Tensor::zeros(vec![1, 2, 4], DType::F32);

        let scale = 1.0 / 2.0f32.sqrt(); // 1/sqrt(head_dim)
        attention(&q, &k, &v, &mut out, scale).unwrap();

        // With causal mask:
        // Position 0: can only attend to position 0, so output = v[0]
        // Position 1: can attend to both, weights depend on Q@K^T
        let result = out.as_f32().unwrap();
        assert!((result[0] - 1.0).abs() < 0.1); // Should be close to v[0]
    }

    #[test]
    fn test_attention_gqa() {
        // Test Grouped Query Attention: 4 query heads, 2 KV heads
        let q = Tensor::from_f32(&vec![1.0f32; 4 * 1 * 4], vec![4, 1, 4]).unwrap();
        let k = Tensor::from_f32(&vec![1.0f32; 2 * 1 * 4], vec![2, 1, 4]).unwrap();
        let v = Tensor::from_f32(&vec![1.0f32; 2 * 1 * 4], vec![2, 1, 4]).unwrap();
        let mut out = Tensor::zeros(vec![4, 1, 4], DType::F32);

        attention(&q, &k, &v, &mut out, 0.5).unwrap();

        // Should complete without error (GQA handled correctly)
        let result = out.as_f32().unwrap();
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_vec_mat_gguf_layout() {
        // Test vec_mat with GGUF-style memory layout
        //
        // Simulates a PyTorch Linear layer with:
        //   - in_features = 3
        //   - out_features = 2
        //   - PyTorch weight shape: [out_features, in_features] = [2, 3]
        //   - PyTorch weight (row-major):
        //       W = [[1, 2, 3],    <- weights for output 0
        //            [4, 5, 6]]   <- weights for output 1
        //     Flat: [1, 2, 3, 4, 5, 6]
        //
        // After GGUF conversion (dimensions reversed, data unchanged):
        //   - GGUF dims: [in_features, out_features] = [3, 2]
        //   - Data still: [1, 2, 3, 4, 5, 6]
        //
        // For input x = [1, 1, 1]:
        //   y[0] = 1*1 + 2*1 + 3*1 = 6  (output 0)
        //   y[1] = 4*1 + 5*1 + 6*1 = 15 (output 1)

        // GGUF weight: shape [3, 2], data is PyTorch row-major
        let weight_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = Tensor::from_f32(&weight_data, vec![3, 2]).unwrap();

        let x = Tensor::from_f32(&[1.0, 1.0, 1.0], vec![3]).unwrap();
        let mut out = Tensor::zeros(vec![2], DType::F32);

        vec_mat(&x, &weight, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // y[j] = sum_i x[i] * W[i + j * k]
        // j=0: x[0]*W[0] + x[1]*W[1] + x[2]*W[2] = 1*1 + 1*2 + 1*3 = 6
        // j=1: x[0]*W[3] + x[1]*W[4] + x[2]*W[5] = 1*4 + 1*5 + 1*6 = 15
        assert_eq!(result, &[6.0, 15.0]);
    }

    #[test]
    fn test_vec_mat_identity_pattern() {
        // Test that vec_mat produces correct results for identity-like weights
        // PyTorch: W = [[1,0,0], [0,1,0]] (2 outputs, 3 inputs), selects first 2 inputs
        // Flat (row-major): [1, 0, 0, 0, 1, 0]
        // GGUF shape: [3, 2]

        let weight_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let weight = Tensor::from_f32(&weight_data, vec![3, 2]).unwrap();

        let x = Tensor::from_f32(&[7.0, 8.0, 9.0], vec![3]).unwrap();
        let mut out = Tensor::zeros(vec![2], DType::F32);

        vec_mat(&x, &weight, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // Should select first two inputs: [7, 8]
        assert_eq!(result, &[7.0, 8.0]);
    }
}
