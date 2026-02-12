//! SIMD-optimized operations for CPU backend
//!
//! This module provides optimized implementations using:
//! - AVX2 (256-bit vectors, 8 floats)
//! - AVX-512 (512-bit vectors, 16 floats) - when available
//! - NEON (128-bit vectors, 4 floats) - for ARM
//!
//! Runtime feature detection is used to select the best implementation.

// Allow unsafe operations in unsafe functions (Rust 2024 compatibility)
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// Feature Detection
// =============================================================================

/// Check if AVX2 is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Check if AVX-512F is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Check if NEON is available (always true on aarch64)
#[cfg(target_arch = "aarch64")]
pub fn has_neon() -> bool {
    true
}

/// AVX2 is not available on aarch64
#[cfg(target_arch = "aarch64")]
pub fn has_avx2() -> bool {
    false
}

/// AVX-512 is not available on aarch64
#[cfg(target_arch = "aarch64")]
pub fn has_avx512() -> bool {
    false
}

/// Check if NEON is available (always false on x86_64)
#[cfg(target_arch = "x86_64")]
pub fn has_neon() -> bool {
    false
}

// Fallback for other architectures (not x86_64, not aarch64)
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn has_avx2() -> bool {
    false
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn has_avx512() -> bool {
    false
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn has_neon() -> bool {
    false
}

// =============================================================================
// Dot Product
// =============================================================================

/// Compute dot product of two f32 slices using best available SIMD
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { dot_f32_avx512(a, b) };
        }
        if has_avx2() {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_f32_neon(a, b) };
    }

    // Scalar fallback
    dot_f32_scalar(a, b)
}

/// Scalar dot product (fallback)
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// AVX2 dot product (8 floats at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of 8 floats
    let mut result = hsum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }

    result
}

/// AVX-512 dot product (16 floats at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 16;

    let mut sum = _mm512_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a_ptr.add(offset));
        let vb = _mm512_loadu_ps(b_ptr.add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    // Reduce 512-bit to scalar
    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    for i in (chunks * 16)..n {
        result += a[i] * b[i];
    }

    result
}

/// NEON dot product (4 floats at a time)
#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }

    result
}

/// Horizontal sum for AVX2 (sum 8 floats to 1)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    // Add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);

    // Now sum 4 floats
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);

    _mm_cvtss_f32(sum32)
}

// =============================================================================
// Vector Operations
// =============================================================================

/// Element-wise multiply-add: out = a * b + c
pub fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                fma_f32_avx2(a, b, c, out);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            fma_f32_neon(a, b, c, out);
        }
        return;
    }

    // Scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for i in 0..a.len() {
        out[i] = a[i] * b[i] + c[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vc = _mm256_loadu_ps(c.as_ptr().add(offset));
        let result = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        out[i] = a[i] * b[i] + c[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fma_f32_neon(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let vc = vld1q_f32(c.as_ptr().add(offset));
        let result = vfmaq_f32(vc, va, vb);
        vst1q_f32(out.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        out[i] = a[i] * b[i] + c[i];
    }
}

/// Scale a vector: out = a * scalar
pub fn scale_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                scale_f32_avx2(a, scalar, out);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            scale_f32_neon(a, scalar, out);
        }
        return;
    }

    // Scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for i in 0..a.len() {
        out[i] = a[i] * scalar;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_f32_avx2(a: &[f32], scalar: f32, out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    let vscalar = _mm256_set1_ps(scalar);

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let result = _mm256_mul_ps(va, vscalar);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 8)..n {
        out[i] = a[i] * scalar;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn scale_f32_neon(a: &[f32], scalar: f32, out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;
    let vscalar = vdupq_n_f32(scalar);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let result = vmulq_f32(va, vscalar);
        vst1q_f32(out.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        out[i] = a[i] * scalar;
    }
}

/// Sum all elements in a slice
pub fn sum_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { sum_f32_avx2(a) };
        }
        a.iter().sum()
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sum_f32_neon(a) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    a.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f32_avx2(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        sum = _mm256_add_ps(sum, va);
    }

    let mut result = hsum_avx2(sum);

    for item in a.iter().take(n).skip(chunks * 8) {
        result += item;
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn sum_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let mut sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        sum = vaddq_f32(sum, va);
    }

    let mut result = vaddvq_f32(sum);

    for i in (chunks * 4)..n {
        result += a[i];
    }

    result
}

/// Find maximum value in a slice
pub fn max_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { max_f32_avx2(a) };
        }
        a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { max_f32_neon(a) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_f32_avx2(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }

    let chunks = n / 8;
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        vmax = _mm256_max_ps(vmax, va);
    }

    // Reduce to scalar
    let high = _mm256_extractf128_ps(vmax, 1);
    let low = _mm256_castps256_ps128(vmax);
    let max128 = _mm_max_ps(high, low);

    let shuf = _mm_movehdup_ps(max128);
    let max64 = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(max64, max64);
    let max32 = _mm_max_ss(max64, shuf2);

    let mut result = _mm_cvtss_f32(max32);

    for item in a.iter().take(n).skip(chunks * 8) {
        result = result.max(*item);
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn max_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }

    let chunks = n / 4;
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        vmax = vmaxq_f32(vmax, va);
    }

    let mut result = vmaxvq_f32(vmax);

    for i in (chunks * 4)..n {
        result = result.max(a[i]);
    }

    result
}

// =============================================================================
// Softmax (SIMD-optimized)
// =============================================================================

/// Compute softmax in-place with SIMD
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = max_f32(x);

    // Subtract max and compute exp
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                softmax_inplace_avx2(x, max_val);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            softmax_inplace_neon(x, max_val);
        }
        return;
    }

    // Scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_inplace_avx2(x: &mut [f32], max_val: f32) {
    let n = x.len();
    let _vmax = _mm256_set1_ps(max_val);

    // Compute exp(x - max)
    // Note: We use a fast exp approximation here
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Divide by sum
    let inv_sum = 1.0 / sum;
    let vinv = _mm256_set1_ps(inv_sum);
    let chunks = n / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        let result = _mm256_mul_ps(vx, vinv);
        _mm256_storeu_ps(x.as_mut_ptr().add(offset), result);
    }

    for item in x.iter_mut().take(n).skip(chunks * 8) {
        *item *= inv_sum;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn softmax_inplace_neon(x: &mut [f32], max_val: f32) {
    let n = x.len();

    // Compute exp(x - max) - still use scalar for exp() as NEON doesn't have native exp
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Divide by sum using NEON
    let inv_sum = 1.0 / sum;
    let vinv = vdupq_n_f32(inv_sum);
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(offset));
        let result = vmulq_f32(vx, vinv);
        vst1q_f32(x.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        x[i] *= inv_sum;
    }
}

// =============================================================================
// RMS Norm (SIMD-optimized)
// =============================================================================

/// Compute sum of squares for RMS norm
pub fn sum_of_squares(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { sum_of_squares_avx2(x) };
        }
        x.iter().map(|&v| v * v).sum()
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sum_of_squares_neon(x) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    x.iter().map(|&v| v * v).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_of_squares_avx2(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(vx, vx, sum);
    }

    let mut result = hsum_avx2(sum);

    for item in x.iter().take(n).skip(chunks * 8) {
        result += item * item;
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn sum_of_squares_neon(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks = n / 4;
    let mut sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(offset));
        sum = vfmaq_f32(sum, vx, vx);
    }

    let mut result = vaddvq_f32(sum);

    for i in (chunks * 4)..n {
        result += x[i] * x[i];
    }

    result
}

/// Apply RMS normalization: out = x / rms * weight
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let n = x.len();
    let ss = sum_of_squares(x);
    let rms = (ss / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                rms_norm_avx2(x, weight, inv_rms, out);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            rms_norm_neon(x, weight, inv_rms, out);
        }
        return;
    }

    // Scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rms_norm_avx2(x: &[f32], weight: &[f32], inv_rms: f32, out: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;
    let vinv_rms = _mm256_set1_ps(inv_rms);

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        let vw = _mm256_loadu_ps(weight.as_ptr().add(offset));
        let scaled = _mm256_mul_ps(vx, vinv_rms);
        let result = _mm256_mul_ps(scaled, vw);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 8)..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], inv_rms: f32, out: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;
    let vinv_rms = vdupq_n_f32(inv_rms);

    for i in 0..chunks {
        let offset = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(offset));
        let vw = vld1q_f32(weight.as_ptr().add(offset));
        let scaled = vmulq_f32(vx, vinv_rms);
        let result = vmulq_f32(scaled, vw);
        vst1q_f32(out.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// =============================================================================
// Quantized Dot Product (SIMD-optimized Q4_0)
// =============================================================================

use crate::tensor::quant::BlockQ4_0;

/// SIMD-optimized dot product with Q4_0 quantized weights
pub fn dot_q4_0(weights: &[BlockQ4_0], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { dot_q4_0_avx2(weights, x) };
        }
        dot_q4_0_scalar(weights, x)
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_q4_0_neon(weights, x) };
    }

    // Scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    dot_q4_0_scalar(weights, x)
}

fn dot_q4_0_scalar(weights: &[BlockQ4_0], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut offset = 0;

    for block in weights {
        let d = block.d.to_f32();

        for i in 0..16 {
            let byte = block.qs[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;

            sum += lo * d * x[offset + i];
            sum += hi * d * x[offset + i + 16];
        }

        offset += 32;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_0_avx2(weights: &[BlockQ4_0], x: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut offset = 0;

    let _mask_lo = _mm256_set1_epi8(0x0F);
    let _sub8 = _mm256_set1_epi8(8);

    for block in weights {
        let d = block.d.to_f32();
        let _vd = _mm256_set1_ps(d);

        // Load 16 bytes of quantized data
        let q_bytes = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);

        // Expand to 32 int8 values
        let _q_lo = _mm256_cvtepu8_epi16(q_bytes);

        // Extract low and high nibbles - this is simplified, full impl would be more complex
        // For now, use scalar for the inner loop
        for i in 0..16 {
            let byte = block.qs[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;

            let x_lo = x[offset + i];
            let x_hi = x[offset + i + 16];

            // Accumulate
            let contrib_lo = lo * d * x_lo;
            let contrib_hi = hi * d * x_hi;

            sum = _mm256_add_ps(sum, _mm256_set1_ps(contrib_lo + contrib_hi));
        }

        offset += 32;
    }

    hsum_avx2(sum)
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4_0_neon(weights: &[BlockQ4_0], x: &[f32]) -> f32 {
    let mut sum = vdupq_n_f32(0.0);
    let mut offset = 0;

    for block in weights {
        let d = block.d.to_f32();
        let vd = vdupq_n_f32(d);

        // Process 4 elements at a time from each half (lo/hi nibbles)
        // Each block has 16 bytes = 32 nibbles = 32 values
        for chunk in 0..4 {
            let chunk_offset = chunk * 4;

            // Extract 4 lo nibbles and 4 hi nibbles
            let mut lo_vals = [0.0f32; 4];
            let mut hi_vals = [0.0f32; 4];

            for i in 0..4 {
                let byte = block.qs[chunk_offset + i];
                lo_vals[i] = ((byte & 0x0F) as i32 - 8) as f32;
                hi_vals[i] = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            }

            // Load x values
            let x_lo = vld1q_f32(x.as_ptr().add(offset + chunk_offset));
            let x_hi = vld1q_f32(x.as_ptr().add(offset + chunk_offset + 16));

            // Load quantized values as f32
            let q_lo = vld1q_f32(lo_vals.as_ptr());
            let q_hi = vld1q_f32(hi_vals.as_ptr());

            // Compute: sum += d * q * x
            let scaled_lo = vmulq_f32(q_lo, vd);
            let scaled_hi = vmulq_f32(q_hi, vd);

            sum = vfmaq_f32(sum, scaled_lo, x_lo);
            sum = vfmaq_f32(sum, scaled_hi, x_hi);
        }

        offset += 32;
    }

    vaddvq_f32(sum)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = dot_f32(&a, &b);
        assert!((result - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = sum_f32(&a);
        assert!((result - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_max() {
        let a = vec![1.0, 5.0, 3.0, 9.0, 2.0, 8.0, 4.0, 7.0, 6.0];
        let result = max_f32(&a);
        assert!((result - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];

        rms_norm(&x, &weight, 1e-6, &mut out);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        // Each output should be x[i] / rms
        let rms = (30.0f32 / 4.0).sqrt();
        for i in 0..4 {
            let expected = x[i] / rms;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_scale() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 8];

        scale_f32(&a, 2.0, &mut out);

        for i in 0..8 {
            assert!((out[i] - a[i] * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_feature_detection() {
        // Just ensure these don't panic
        println!("AVX2: {}", has_avx2());
        println!("AVX-512: {}", has_avx512());
    }
}
