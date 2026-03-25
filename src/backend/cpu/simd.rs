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

/// Fused multiply-add accumulate: y[i] += alpha * x[i] (SIMD accelerated)
///
/// This is the BLAS-like AXPY operation, critical for attention weighted V accumulation.
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            unsafe {
                axpy_f32_avx512(alpha, x, y);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                axpy_f32_avx2(alpha, x, y);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            axpy_f32_neon(alpha, x, y);
        }
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_f32_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;
    let valpha = _mm256_set1_ps(alpha);

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        let vy = _mm256_loadu_ps(y.as_ptr().add(offset));
        let result = _mm256_fmadd_ps(valpha, vx, vy);
        _mm256_storeu_ps(y.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 8)..n {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn axpy_f32_avx512(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 16;
    let valpha = _mm512_set1_ps(alpha);

    for i in 0..chunks {
        let offset = i * 16;
        let vx = _mm512_loadu_ps(x.as_ptr().add(offset));
        let vy = _mm512_loadu_ps(y.as_ptr().add(offset));
        let result = _mm512_fmadd_ps(valpha, vx, vy);
        _mm512_storeu_ps(y.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 16)..n {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn axpy_f32_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;
    let valpha = vdupq_n_f32(alpha);

    for i in 0..chunks {
        let offset = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(offset));
        let vy = vld1q_f32(y.as_ptr().add(offset));
        let result = vfmaq_f32(vy, valpha, vx);
        vst1q_f32(y.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        y[i] += alpha * x[i];
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
// Fused SiLU * Multiply
// =============================================================================

/// Fused SiLU activation and element-wise multiply: gate[i] = silu(gate[i]) * up[i]
///
/// Computes `gate[i] = gate[i] * sigmoid(gate[i]) * up[i]` in a single pass.
/// This eliminates two intermediate tensor allocations in the FFN hot path.
pub fn silu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                silu_mul_inplace_avx2(gate, up);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            silu_mul_inplace_neon(gate, up);
        }
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for i in 0..gate.len() {
        let x = gate[i];
        gate[i] = x / (1.0 + (-x).exp()) * up[i];
    }
}

/// AVX2: fused silu * mul — SIMD for the multiply, scalar for sigmoid (no native exp)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_mul_inplace_avx2(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    let chunks = n / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let g = gate.as_mut_ptr().add(offset);
        for j in 0..8 {
            let x = *g.add(j);
            *g.add(j) = x / (1.0 + (-x).exp());
        }
        let vg = _mm256_loadu_ps(g);
        let vu = _mm256_loadu_ps(up.as_ptr().add(offset));
        _mm256_storeu_ps(g, _mm256_mul_ps(vg, vu));
    }

    for i in (chunks * 8)..n {
        let x = gate[i];
        gate[i] = x / (1.0 + (-x).exp()) * up[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn silu_mul_inplace_neon(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let g = gate.as_mut_ptr().add(offset);
        for j in 0..4 {
            let x = *g.add(j);
            *g.add(j) = x / (1.0 + (-x).exp());
        }
        let vg = vld1q_f32(g);
        let vu = vld1q_f32(up.as_ptr().add(offset));
        vst1q_f32(g, vmulq_f32(vg, vu));
    }

    for i in (chunks * 4)..n {
        let x = gate[i];
        gate[i] = x / (1.0 + (-x).exp()) * up[i];
    }
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
// Fused Quantized Dot Products
// =============================================================================
//
// These compute dot(dequant(blocks), x) without materializing the full
// dequantized vector, saving both the allocation and the extra memory read.

use crate::tensor::quant::{BlockQ4_0, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8K};

/// Fused dot product: Q4_0 blocks against f32 vector (32 elements/block)
pub fn dot_q4_0(weights: &[BlockQ4_0], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut offset = 0;

    for block in weights {
        let d = block.d.to_f32();
        let mut acc_lo = 0.0f32;
        let mut acc_hi = 0.0f32;

        for i in 0..16 {
            let byte = block.qs[i];
            acc_lo += ((byte & 0x0F) as i32 - 8) as f32 * x[offset + i];
            acc_hi += (((byte >> 4) & 0x0F) as i32 - 8) as f32 * x[offset + i + 16];
        }

        sum += d * (acc_lo + acc_hi);
        offset += 32;
    }

    sum
}

/// Fused dot product: Q8_0 blocks against f32 vector (32 elements/block)
pub fn dot_q8_0(weights: &[BlockQ8_0], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut offset = 0;

    for block in weights {
        let d = block.d.to_f32();
        let mut acc = 0.0f32;

        for i in 0..32 {
            acc += block.qs[i] as f32 * x[offset + i];
        }

        sum += d * acc;
        offset += 32;
    }

    sum
}

/// Fused dot product: Q4_K blocks against f32 vector (256 elements/block)
///
/// Separates the scale*q and dmin*min terms so that the inner loops only
/// accumulate integer-weighted sums, multiplying by the per-sub-block
/// scale/min factors once per 32-element group.
pub fn dot_q4_k(weights: &[BlockQ4K], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut x_off = 0;

    for block in weights {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..4 {
            scales[j] = block.scales[j] & 0x3F;
            mins[j] = block.scales[j + 4] & 0x3F;
        }
        for j in 4..8 {
            scales[j] = (block.scales[j + 4] & 0x0F) | ((block.scales[j - 4] >> 6) << 4);
            mins[j] = ((block.scales[j + 4] >> 4) & 0x0F) | ((block.scales[j] >> 6) << 4);
        }

        let mut qs_ptr = 0;
        let mut is = 0;

        for _ in 0..4 {
            let d1 = d * scales[is] as f32;
            let m1 = dmin * mins[is] as f32;
            let d2 = d * scales[is + 1] as f32;
            let m2 = dmin * mins[is + 1] as f32;

            let mut q_acc1 = 0.0f32;
            let mut x_acc1 = 0.0f32;
            for l in 0..32 {
                let q = (block.qs[qs_ptr + l] & 0x0F) as f32;
                q_acc1 += q * x[x_off + l];
                x_acc1 += x[x_off + l];
            }
            sum += d1 * q_acc1 - m1 * x_acc1;
            x_off += 32;

            let mut q_acc2 = 0.0f32;
            let mut x_acc2 = 0.0f32;
            for l in 0..32 {
                let q = ((block.qs[qs_ptr + l] >> 4) & 0x0F) as f32;
                q_acc2 += q * x[x_off + l];
                x_acc2 += x[x_off + l];
            }
            sum += d2 * q_acc2 - m2 * x_acc2;
            x_off += 32;

            qs_ptr += 32;
            is += 2;
        }
    }

    sum
}

/// Fused dot product: Q5_K blocks against f32 vector (256 elements/block)
pub fn dot_q5_k(weights: &[BlockQ5K], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut x_off = 0;

    for block in weights {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..4 {
            scales[j] = block.scales[j] & 0x3F;
            mins[j] = block.scales[j + 4] & 0x3F;
        }
        for j in 4..8 {
            scales[j] = (block.scales[j + 4] & 0x0F) | ((block.scales[j - 4] >> 6) << 4);
            mins[j] = ((block.scales[j + 4] >> 4) & 0x0F) | ((block.scales[j] >> 6) << 4);
        }

        let mut ql_ptr = 0;
        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _ in 0..4 {
            let d1 = d * scales[is] as f32;
            let m1 = dmin * mins[is] as f32;
            let d2 = d * scales[is + 1] as f32;
            let m2 = dmin * mins[is + 1] as f32;

            let mut q_acc1 = 0.0f32;
            let mut x_acc1 = 0.0f32;
            for l in 0..32 {
                let lo4 = (block.qs[ql_ptr + l] & 0x0F) as f32;
                let hi5 = if block.qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                q_acc1 += (lo4 + hi5) * x[x_off + l];
                x_acc1 += x[x_off + l];
            }
            sum += d1 * q_acc1 - m1 * x_acc1;
            x_off += 32;

            let mut q_acc2 = 0.0f32;
            let mut x_acc2 = 0.0f32;
            for l in 0..32 {
                let hi4 = ((block.qs[ql_ptr + l] >> 4) & 0x0F) as f32;
                let hi5 = if block.qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                q_acc2 += (hi4 + hi5) * x[x_off + l];
                x_acc2 += x[x_off + l];
            }
            sum += d2 * q_acc2 - m2 * x_acc2;
            x_off += 32;

            ql_ptr += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    sum
}

/// Fused dot product: Q6_K blocks against f32 vector (256 elements/block)
pub fn dot_q6_k(weights: &[BlockQ6K], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut x_off = 0;

    for block in weights {
        let d = block.d.to_f32();

        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;

            for l in 0..32 {
                let is = l / 16;

                let q1 = ((block.ql[ql_base + l] & 0x0F)
                    | ((block.qh[qh_base + l] & 0x03) << 4)) as i32
                    - 32;
                let q2 = ((block.ql[ql_base + l + 32] & 0x0F)
                    | (((block.qh[qh_base + l] >> 2) & 0x03) << 4))
                    as i32
                    - 32;
                let q3 = ((block.ql[ql_base + l] >> 4)
                    | (((block.qh[qh_base + l] >> 4) & 0x03) << 4))
                    as i32
                    - 32;
                let q4 = ((block.ql[ql_base + l + 32] >> 4)
                    | (((block.qh[qh_base + l] >> 6) & 0x03) << 4))
                    as i32
                    - 32;

                let out_base = n * 128;
                let s1 = block.scales[sc_base + is] as f32;
                let s2 = block.scales[sc_base + is + 2] as f32;
                let s3 = block.scales[sc_base + is + 4] as f32;
                let s4 = block.scales[sc_base + is + 6] as f32;

                sum += d * s1 * q1 as f32 * x[x_off + out_base + l];
                sum += d * s2 * q2 as f32 * x[x_off + out_base + l + 32];
                sum += d * s3 * q3 as f32 * x[x_off + out_base + l + 64];
                sum += d * s4 * q4 as f32 * x[x_off + out_base + l + 96];
            }
        }

        x_off += 256;
    }

    sum
}

/// Fused dot product: Q8_K blocks against f32 vector (256 elements/block)
pub fn dot_q8_k(weights: &[BlockQ8K], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut offset = 0;

    for block in weights {
        let d = block.d;
        let mut acc = 0.0f32;

        for i in 0..256 {
            acc += block.qs[i] as f32 * x[offset + i];
        }

        sum += d * acc;
        offset += 256;
    }

    sum
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
    fn test_silu_mul_inplace() {
        let mut gate = vec![1.0, -1.0, 2.0, 0.0, 0.5, -0.5, 3.0, -2.0];
        let up = vec![2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 0.5, 1.0];
        let original_gate = gate.clone();

        silu_mul_inplace(&mut gate, &up);

        for i in 0..gate.len() {
            let x = original_gate[i];
            let expected = x / (1.0 + (-x).exp()) * up[i];
            assert!(
                (gate[i] - expected).abs() < 1e-5,
                "mismatch at {}: {} vs {}",
                i,
                gate[i],
                expected
            );
        }
    }

    #[test]
    fn test_axpy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        axpy_f32(2.0, &x, &mut y);

        let expected = vec![12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0];
        for i in 0..8 {
            assert!(
                (y[i] - expected[i]).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                y[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_feature_detection() {
        // Just ensure these don't panic
        println!("AVX2: {}", has_avx2());
        println!("AVX-512: {}", has_avx512());
    }

    #[test]
    fn test_dot_sign_bits() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let mut bits = vec![0u64; 2];
        for i in 0..128 {
            if i % 2 == 0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        let result = dot_with_sign_bits_fast(&values, &bits, 128);
        let mut expected = 0.0f32;
        for i in 0..128 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            expected += values[i] * sign;
        }
        assert!(
            (result - expected).abs() < 1e-3,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_sign_bits_fast_basic() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let bits: Vec<u64> = vec![0b1010]; // bits: idx0=0(-1), idx1=1(+1), idx2=0(-1), idx3=1(+1)
        let result = super::dot_with_sign_bits_fast(&values, &bits, 4);
        let expected = -1.0 + 2.0 - 3.0 + 4.0; // 2.0
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_sign_bits_fast_all_ones() {
        let n = 128;
        let values: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let bits: Vec<u64> = vec![u64::MAX; (n + 63) / 64];
        let result = super::dot_with_sign_bits_fast(&values, &bits, n);
        let expected: f32 = values.iter().sum();
        assert!(
            (result - expected).abs() < 1e-2,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_sign_bits_fast_all_zeros() {
        let n = 64;
        let values: Vec<f32> = vec![1.0; n];
        let bits: Vec<u64> = vec![0u64];
        let result = super::dot_with_sign_bits_fast(&values, &bits, n);
        let expected = -(n as f32);
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_sign_bits_fast_odd_count() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let bits: Vec<u64> = vec![0b1111111]; // all +1
        let result = super::dot_with_sign_bits_fast(&values, &bits, 7);
        let expected: f32 = values.iter().sum();
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }
}

// =============================================================================
// TurboQuant SIMD helpers
// =============================================================================

/// SIMD-optimized dot product between f32 values and packed sign bits.
///
/// Each bit in `bits` encodes +1 (set) or -1 (clear). This is the hot loop
/// for QJL inner product estimation.
pub fn dot_with_sign_bits_fast(values: &[f32], bits: &[u64], count: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { dot_sign_bits_avx2(values, bits, count) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_sign_bits_neon(values, bits, count) };
    }
    #[allow(unreachable_code)]
    dot_sign_bits_scalar(values, bits, count)
}

fn dot_sign_bits_scalar(values: &[f32], bits: &[u64], count: usize) -> f32 {
    crate::model::turboquant::qjl::dot_with_sign_bits(values, bits, count)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_sign_bits_avx2(values: &[f32], bits: &[u64], count: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut scalar_tail = 0.0f32;
    let mut pos = 0;

    for &word in bits.iter() {
        let remaining = count - pos;
        if remaining == 0 {
            break;
        }
        let n = remaining.min(64);
        let mut bit_idx = 0;
        while bit_idx + 8 <= n && pos + 8 <= count {
            let v = _mm256_loadu_ps(values.as_ptr().add(pos));
            let sign_vec = _mm256_set_ps(
                if (word >> (bit_idx + 7)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> (bit_idx + 6)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> (bit_idx + 5)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> (bit_idx + 4)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> (bit_idx + 3)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> (bit_idx + 2)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> (bit_idx + 1)) & 1 == 1 { 1.0 } else { -1.0 },
                if (word >> bit_idx) & 1 == 1 { 1.0 } else { -1.0 },
            );
            acc = _mm256_fmadd_ps(v, sign_vec, acc);
            bit_idx += 8;
            pos += 8;
        }
        // Scalar tail
        while bit_idx < n && pos < count {
            let sign = if (word >> bit_idx) & 1 == 1 { 1.0f32 } else { -1.0f32 };
            scalar_tail += values[pos] * sign;
            bit_idx += 1;
            pos += 1;
        }
    }

    // Horizontal sum
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), acc);
    sum_arr.iter().sum::<f32>() + scalar_tail
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_sign_bits_neon(values: &[f32], bits: &[u64], count: usize) -> f32 {
    let ones = vdupq_n_f32(1.0);
    let neg_ones = vdupq_n_f32(-1.0);
    let mut acc = vdupq_n_f32(0.0);
    let mut pos = 0;

    for &word in bits.iter() {
        let remaining = count - pos;
        if remaining == 0 {
            break;
        }
        let n = remaining.min(64);
        let mut bit_idx = 0;
        while bit_idx + 4 <= n && pos + 4 <= count {
            let v = vld1q_f32(values.as_ptr().add(pos));
            let mask: [u32; 4] = [
                if (word >> bit_idx) & 1 == 1 { 0xFFFFFFFF } else { 0 },
                if (word >> (bit_idx + 1)) & 1 == 1 { 0xFFFFFFFF } else { 0 },
                if (word >> (bit_idx + 2)) & 1 == 1 { 0xFFFFFFFF } else { 0 },
                if (word >> (bit_idx + 3)) & 1 == 1 { 0xFFFFFFFF } else { 0 },
            ];
            let mask_v = vld1q_u32(mask.as_ptr());
            let signs = vbslq_f32(mask_v, ones, neg_ones);
            acc = vfmaq_f32(acc, v, signs);
            bit_idx += 4;
            pos += 4;
        }
        while bit_idx < n && pos < count {
            let sign = if (word >> bit_idx) & 1 == 1 { 1.0f32 } else { -1.0f32 };
            let sv = vdupq_n_f32(values[pos] * sign);
            acc = vaddq_f32(acc, sv);
            bit_idx += 1;
            pos += 1;
        }
    }

    vaddvq_f32(acc)
}
