//! CUDA kernel implementations for tensor operations
//!
//! This module contains PTX kernels for GPU-accelerated operations.
//! Kernels are compiled at runtime using cudarc's nvrtc support.

use cudarc::driver::{CudaDevice, CudaFunction};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};

/// CUDA kernel source code
pub const KERNEL_SOURCE: &str = r#"
// FP16 support
#include <cuda_fp16.h>

// Define infinity for CUDA
#define CUDART_INF_F __int_as_float(0x7f800000)
#define MY_INFINITY CUDART_INF_F

// Helper to convert f16 (as unsigned short) to f32
__device__ __forceinline__ float half_to_float(unsigned short h) {
    // Simple f16 to f32 conversion
    unsigned int sign = (h >> 15) & 0x1;
    unsigned int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormal
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        unsigned int f = (sign << 31) | 0x7F800000 | (mant << 13);
        return __int_as_float(f);
    }
    
    unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return __int_as_float(f);
}

extern "C" {

// ============================================================================
// Element-wise operations
// ============================================================================

__global__ void add_f32(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void mul_f32(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void scale_f32(const float* a, float scalar, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

// ============================================================================
// Activation functions
// ============================================================================

__global__ void silu_f32(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void gelu_f32(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // GELU approximation
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float GELU_COEF = 0.044715f;
        float inner = SQRT_2_OVER_PI * (val + GELU_COEF * val * val * val);
        out[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

// ============================================================================
// Normalization
// ============================================================================

// RMS normalization - two-pass algorithm (legacy, kept for reference)
__global__ void rms_norm_sum_sq(const float* x, float* sum_sq, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? x[idx] * x[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(sum_sq, sdata[0]);
    }
}

__global__ void rms_norm_scale(const float* x, const float* weight, float* out, 
                                float rms_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * rms_inv * weight[idx];
    }
}

// Fused RMS normalization — single kernel, no CPU round-trip.
// One block processes the entire vector.  Shared memory holds partial
// sums-of-squares; after reduction we broadcast rms_inv and each
// thread writes its output element.
__global__ void rms_norm_fused(const float* x, const float* weight,
                                float* out, float eps, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Step 1: Each thread accumulates partial sum-of-squares.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float v = x[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Step 2: Parallel reduction over shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Step 3: Compute rms_inv and broadcast via shared memory.
    if (tid == 0) {
        float rms = sqrtf(sdata[0] / (float)n + eps);
        sdata[0] = 1.0f / rms;  // reuse sdata[0] for rms_inv
    }
    __syncthreads();
    
    float rms_inv = sdata[0];
    
    // Step 4: Scale output — each thread writes its elements.
    for (int i = tid; i < n; i += stride) {
        out[i] = x[i] * rms_inv * weight[i];
    }
}

// ============================================================================
// Softmax
// ============================================================================

__global__ void softmax_max(const float* x, float* max_val, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? x[idx] : -MY_INFINITY;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Atomic max for floats using int representation
        int* max_int = (int*)max_val;
        int old = *max_int;
        int assumed;
        do {
            assumed = old;
            float old_f = __int_as_float(assumed);
            float new_f = fmaxf(old_f, sdata[0]);
            old = atomicCAS(max_int, assumed, __float_as_int(new_f));
        } while (assumed != old);
    }
}

__global__ void softmax_exp_sum(const float* x, float* out, float* sum, float max_val, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    if (idx < n) {
        val = expf(x[idx] - max_val);
        out[idx] = val;
    }
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void softmax_div(float* out, float sum_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] *= sum_inv;
    }
}

// Fused softmax — single kernel using online softmax algorithm.
// One block processes the entire vector.  No CPU round-trips.
__global__ void softmax_fused(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    // sdata layout: [blockDim.x] for max, then [blockDim.x] for sum
    float* smax = sdata;
    float* ssum = sdata + blockDim.x;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Step 1: Find maximum (parallel reduction).
    float local_max = -MY_INFINITY;
    for (int i = tid; i < n; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }
    smax[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    float max_val = smax[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum (parallel reduction).
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float e = expf(x[i] - max_val);
        out[i] = e;
        local_sum += e;
    }
    ssum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];
    __syncthreads();
    
    // Step 3: Normalize.
    for (int i = tid; i < n; i += stride) {
        out[i] *= inv_sum;
    }
}

// ============================================================================
// Matrix operations
// ============================================================================

// Vector-matrix multiplication: out = vec @ mat
// vec: [k], mat: [k, n], out: [n]
// vec_mat: y[j] = sum_i x[i] * W[i,j]
// GGUF stores weights in column-major order: W[i,j] is at index i + j * k
// vec: [k], mat: [k, n] (stored column-major), out: [n]
__global__ void vec_mat_f32(const float* vec, const float* mat, float* out,
                            int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        float sum = 0.0f;
        // Column-major indexing: mat[i, col] = mat[i + col * k]
        for (int i = 0; i < k; i++) {
            sum += vec[i] * mat[i + col * k];
        }
        out[col] = sum;
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

// RoPE for LLaMA-style (consecutive pairs)
// q, k: [num_heads * head_dim] for single position
__global__ void rope_single_pos(float* q, float* k, 
                                 int num_heads, int num_kv_heads, int head_dim,
                                 int pos, float freq_base, float freq_scale,
                                 int use_neox) {
    int head = blockIdx.x;
    int i = threadIdx.x;  // pair index
    int half_dim = head_dim / 2;
    
    if (head >= num_heads || i >= half_dim) return;
    
    // Compute frequency
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float position = (float)pos / freq_scale;
    float theta = position * freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    
    int q_base = head * head_dim;
    int q_idx0, q_idx1;
    
    if (use_neox) {
        q_idx0 = q_base + i;
        q_idx1 = q_base + i + half_dim;
    } else {
        q_idx0 = q_base + 2 * i;
        q_idx1 = q_base + 2 * i + 1;
    }
    
    // Rotate Q (always — all heads)
    float q0 = q[q_idx0];
    float q1 = q[q_idx1];
    q[q_idx0] = q0 * cos_theta - q1 * sin_theta;
    q[q_idx1] = q0 * sin_theta + q1 * cos_theta;
    
    // Rotate K only for KV heads (skip for GQA heads beyond num_kv_heads)
    if (head < num_kv_heads) {
        int k_base = head * head_dim;
        int k_idx0, k_idx1;
        if (use_neox) {
            k_idx0 = k_base + i;
            k_idx1 = k_base + i + half_dim;
        } else {
            k_idx0 = k_base + 2 * i;
            k_idx1 = k_base + 2 * i + 1;
        }
        float k0 = k[k_idx0];
        float k1 = k[k_idx1];
        k[k_idx0] = k0 * cos_theta - k1 * sin_theta;
        k[k_idx1] = k0 * sin_theta + k1 * cos_theta;
    }
}

// ============================================================================
// Quantized Operations - Q4_K (most common for good quality/size)
// ============================================================================

// Q4_K block layout (144 bytes for 256 values):
// - d: f16 (2 bytes) - scale
// - dmin: f16 (2 bytes) - min scale  
// - scales: [12] u8 - packed 6-bit scales/mins
// - qs: [128] u8 - 256 4-bit values

// Fused dequantize + vec_mat for Q4_K
// Each thread handles one output column
__global__ void vec_mat_q4k(const unsigned char* weight,  // [num_blocks, 144]
                            const float* vec,              // [k]
                            float* out,                    // [n]
                            int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    int num_blocks = k / 256;
    float sum = 0.0f;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Pointer to this block's data for this output column
        // Layout: blocks are stored as [num_blocks, n, 144] for coalesced access
        const unsigned char* block = weight + (block_idx * n + col) * 144;
        
        // Read d and dmin (f16)
        unsigned short d_bits = block[0] | (block[1] << 8);
        unsigned short dmin_bits = block[2] | (block[3] << 8);
        float d = half_to_float(d_bits);
        float dmin = half_to_float(dmin_bits);
        
        // Decode scales and mins from 12 bytes
        float scales[8], mins[8];
        for (int j = 0; j < 4; j++) {
            scales[j] = (float)(block[4 + j] & 0x3F);
            mins[j] = (float)(block[4 + j + 4] & 0x3F);
        }
        for (int j = 4; j < 8; j++) {
            scales[j] = (float)((block[4 + j + 4] & 0x0F) | ((block[4 + j - 4] >> 6) << 4));
            mins[j] = (float)(((block[4 + j + 4] >> 4) & 0x0F) | ((block[4 + j] >> 6) << 4));
        }
        
        // Process 256 values
        const unsigned char* qs = block + 16;  // After d, dmin, scales
        int vec_base = block_idx * 256;
        int qs_idx = 0;
        int is = 0;
        
        for (int group = 0; group < 4; group++) {
            float d1 = d * scales[is];
            float m1 = dmin * mins[is];
            float d2 = d * scales[is + 1];
            float m2 = dmin * mins[is + 1];
            
            // First 32: low nibbles
            for (int l = 0; l < 32; l++) {
                float q = (float)(qs[qs_idx + l] & 0x0F);
                float val = d1 * q - m1;
                sum += vec[vec_base] * val;
                vec_base++;
            }
            
            // Next 32: high nibbles
            for (int l = 0; l < 32; l++) {
                float q = (float)((qs[qs_idx + l] >> 4) & 0x0F);
                float val = d2 * q - m2;
                sum += vec[vec_base] * val;
                vec_base++;
            }
            
            qs_idx += 32;
            is += 2;
        }
    }
    
    out[col] = sum;
}

// ============================================================================
// Quantized Operations - Q8_0 (high quality)
// ============================================================================

// Q8_0 block layout (34 bytes for 32 values):
// - d: f16 (2 bytes) - scale
// - qs: [32] i8 - 32 signed 8-bit values

__global__ void vec_mat_q8_0(const unsigned char* weight,  // [num_blocks, n, 34]
                              const float* vec,             // [k]
                              float* out,                   // [n]
                              int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    int num_blocks = k / 32;
    float sum = 0.0f;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const unsigned char* block = weight + (block_idx * n + col) * 34;
        
        // Read d (f16)
        unsigned short d_bits = block[0] | (block[1] << 8);
        float d = half_to_float(d_bits);
        
        const signed char* qs = (const signed char*)(block + 2);
        int vec_base = block_idx * 32;
        
        for (int i = 0; i < 32; i++) {
            float val = d * (float)qs[i];
            sum += vec[vec_base + i] * val;
        }
    }
    
    out[col] = sum;
}

// ============================================================================
// Quantized Operations - Q4_0 (legacy, smaller models)
// ============================================================================

// Q4_0 block layout (18 bytes for 32 values):
// - d: f16 (2 bytes)
// - qs: [16] u8 - 32 4-bit values packed

__global__ void vec_mat_q4_0(const unsigned char* weight,  // [num_blocks, n, 18]
                              const float* vec,             // [k]
                              float* out,                   // [n]
                              int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    int num_blocks = k / 32;
    float sum = 0.0f;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const unsigned char* block = weight + (block_idx * n + col) * 18;
        
        unsigned short d_bits = block[0] | (block[1] << 8);
        float d = half_to_float(d_bits);
        
        const unsigned char* qs = block + 2;
        int vec_base = block_idx * 32;
        
        for (int i = 0; i < 16; i++) {
            unsigned char byte = qs[i];
            // Low nibble (first half)
            float q_lo = (float)((byte & 0x0F) - 8);
            // High nibble (second half)
            float q_hi = (float)(((byte >> 4) & 0x0F) - 8);
            
            sum += vec[vec_base + i] * (d * q_lo);
            sum += vec[vec_base + i + 16] * (d * q_hi);
        }
    }
    
    out[col] = sum;
}

// ============================================================================
// Quantized Operations - Q6_K (high quality K-quant)
// ============================================================================

// Q6_K block layout (210 bytes for 256 values):
// - ql: [128] u8 - low 4 bits
// - qh: [64] u8 - high 2 bits
// - scales: [16] i8 - signed per-group scales
// - d: f16 (2 bytes) - super scale

__global__ void vec_mat_q6k(const unsigned char* weight,
                            const float* vec,
                            float* out,
                            int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    int num_blocks = k / 256;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const unsigned char* block = weight + (block_idx * n + col) * 210;

        // d is at offset 208 (after ql[128] + qh[64] + scales[16])
        unsigned short d_bits = block[208] | (block[209] << 8);
        float d = half_to_float(d_bits);

        int vec_base = block_idx * 256;

        // Process 256 elements in two groups of 128
        for (int ng = 0; ng < 2; ng++) {
            int ql_base = ng * 64;
            int qh_base = 128 + ng * 32;
            int sc_base = 192 + ng * 8;
            int o_base = ng * 128;

            for (int l = 0; l < 32; l++) {
                int is_idx = l / 16;

                int q1 = ((block[ql_base + l] & 0x0F) | ((block[qh_base + l] & 0x03) << 4)) - 32;
                int q2 = ((block[ql_base + l + 32] & 0x0F) | (((block[qh_base + l] >> 2) & 0x03) << 4)) - 32;
                int q3 = ((block[ql_base + l] >> 4) | (((block[qh_base + l] >> 4) & 0x03) << 4)) - 32;
                int q4 = ((block[ql_base + l + 32] >> 4) | (((block[qh_base + l] >> 6) & 0x03) << 4)) - 32;

                float sc1 = d * (float)((signed char)block[sc_base + is_idx]);
                float sc2 = d * (float)((signed char)block[sc_base + is_idx + 2]);
                float sc3 = d * (float)((signed char)block[sc_base + is_idx + 4]);
                float sc4 = d * (float)((signed char)block[sc_base + is_idx + 6]);

                sum += vec[vec_base + o_base + l]      * (sc1 * (float)q1);
                sum += vec[vec_base + o_base + l + 32]  * (sc2 * (float)q2);
                sum += vec[vec_base + o_base + l + 64]  * (sc3 * (float)q3);
                sum += vec[vec_base + o_base + l + 96]  * (sc4 * (float)q4);
            }
        }
    }

    out[col] = sum;
}

// ============================================================================
// Quantized Operations - Q5_K (5-bit K-quant)
// ============================================================================

// Q5_K block layout (176 bytes for 256 values):
// - d: f16 (2 bytes) at offset 0
// - dmin: f16 (2 bytes) at offset 2
// - scales: [12] u8 at offset 4
// - qh: [32] u8 at offset 16  (high bits)
// - qs: [128] u8 at offset 48 (low 4 bits)

__global__ void vec_mat_q5k(const unsigned char* weight,
                            const float* vec,
                            float* out,
                            int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    int num_blocks = k / 256;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const unsigned char* block = weight + (block_idx * n + col) * 176;

        unsigned short d_bits = block[0] | (block[1] << 8);
        unsigned short dmin_bits = block[2] | (block[3] << 8);
        float d = half_to_float(d_bits);
        float dmin = half_to_float(dmin_bits);

        // Decode scales and mins from 12 bytes at offset 4
        float scales[8], mins[8];
        for (int j = 0; j < 4; j++) {
            scales[j] = (float)(block[4 + j] & 0x3F);
            mins[j] = (float)(block[4 + j + 4] & 0x3F);
        }
        for (int j = 4; j < 8; j++) {
            scales[j] = (float)((block[4 + j + 4] & 0x0F) | ((block[4 + j - 4] >> 6) << 4));
            mins[j] = (float)(((block[4 + j + 4] >> 4) & 0x0F) | ((block[4 + j] >> 6) << 4));
        }

        const unsigned char* qh = block + 16;   // [32] high bits
        const unsigned char* qs = block + 48;    // [128] low nibbles

        int vec_base = block_idx * 256;
        int qs_idx = 0;
        int is = 0;
        unsigned char u1 = 1;
        unsigned char u2 = 2;

        for (int group = 0; group < 4; group++) {
            float d1 = d * scales[is];
            float m1 = dmin * mins[is];
            float d2 = d * scales[is + 1];
            float m2 = dmin * mins[is + 1];

            // First 32: low nibbles + high bit u1
            for (int l = 0; l < 32; l++) {
                float lo4 = (float)(qs[qs_idx + l] & 0x0F);
                float hi5 = (qh[l] & u1) ? 16.0f : 0.0f;
                sum += vec[vec_base] * (d1 * (lo4 + hi5) - m1);
                vec_base++;
            }

            // Next 32: high nibbles + high bit u2
            for (int l = 0; l < 32; l++) {
                float hi4 = (float)((qs[qs_idx + l] >> 4) & 0x0F);
                float hi5 = (qh[l] & u2) ? 16.0f : 0.0f;
                sum += vec[vec_base] * (d2 * (hi4 + hi5) - m2);
                vec_base++;
            }

            qs_idx += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    out[col] = sum;
}

// ============================================================================
// Attention
// ============================================================================

// Single-head attention for one query position
// Computes: softmax(q @ K^T / sqrt(d)) @ V
__global__ void attention_single_head(const float* q,        // [head_dim]
                                       const float* k_cache, // [kv_len, head_dim]
                                       const float* v_cache, // [kv_len, head_dim]
                                       float* out,           // [head_dim]
                                       int head_dim, int kv_len, int q_pos,
                                       float scale) {
    extern __shared__ float shared[];
    float* scores = shared;  // [kv_len]
    
    int tid = threadIdx.x;
    int dim = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Step 1: Compute attention scores (q @ K^T)
    if (dim < kv_len) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k_cache[dim * head_dim + d];
        }
        score *= scale;
        
        // Apply causal mask
        if (dim > q_pos) {
            score = -MY_INFINITY;
        }
        scores[dim] = score;
    }
    __syncthreads();
    
    // Step 2: Softmax
    // Find max
    if (tid == 0) {
        float max_val = -MY_INFINITY;
        for (int i = 0; i < kv_len; i++) {
            max_val = fmaxf(max_val, scores[i]);
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < kv_len; i++) {
            scores[i] = expf(scores[i] - max_val);
            sum += scores[i];
        }
        
        // Normalize
        for (int i = 0; i < kv_len; i++) {
            scores[i] /= sum;
        }
    }
    __syncthreads();
    
    // Step 3: Weighted sum of values (scores @ V)
    if (dim < head_dim) {
        float sum = 0.0f;
        for (int i = 0; i < kv_len; i++) {
            sum += scores[i] * v_cache[i * head_dim + dim];
        }
        out[dim] = sum;
    }
}

// Copy a single KV pair to the cache at position pos
// k, v: [num_kv_heads * head_dim]
// k_cache, v_cache: [num_kv_heads * max_seq_len * head_dim]
__global__ void update_kv_cache(const float* k, const float* v,
                                 float* k_cache, float* v_cache,
                                 int num_kv_heads, int head_dim,
                                 int max_seq_len, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    
    if (idx < total) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        
        // Cache layout: [num_kv_heads, max_seq_len, head_dim]
        int cache_idx = head * max_seq_len * head_dim + pos * head_dim + d;
        
        k_cache[cache_idx] = k[idx];
        v_cache[cache_idx] = v[idx];
    }
}

// Multi-head attention with GQA support
// q: [num_heads * head_dim]
// k_cache, v_cache: [num_kv_heads * max_seq_len * head_dim]
// out: [num_heads * head_dim]
// One block per query head
//
// Shared memory layout: scores[kv_len] + reduce[blockDim.x]
__global__ void attention_multihead(const float* q,
                                     const float* k_cache,
                                     const float* v_cache,
                                     float* out,
                                     int num_heads, int num_kv_heads,
                                     int head_dim, int max_seq_len,
                                     int kv_len, float scale) {
    extern __shared__ float shared[];
    float* scores = shared;            // [kv_len]
    float* reduce = shared + kv_len;   // [blockDim.x]
    
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    
    // GQA: map query head to KV head
    int heads_per_kv = num_heads / num_kv_heads;
    int kv_head = head / heads_per_kv;
    
    // Offset into Q for this head
    const float* q_head = q + head * head_dim;
    // Offset into KV cache for this KV head
    const float* k_head = k_cache + kv_head * max_seq_len * head_dim;
    const float* v_head = v_cache + kv_head * max_seq_len * head_dim;
    
    // Step 1: Compute attention scores (parallel over kv_len)
    for (int pos = tid; pos < kv_len; pos += nthreads) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_head[d] * k_head[pos * head_dim + d];
        }
        scores[pos] = score * scale;
    }
    __syncthreads();
    
    // Step 2: Parallel softmax over scores[0..kv_len]
    // 2a: Find max via parallel reduction.
    float local_max = -MY_INFINITY;
    for (int i = tid; i < kv_len; i += nthreads) {
        local_max = fmaxf(local_max, scores[i]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        }
        __syncthreads();
    }
    float max_val = reduce[0];
    __syncthreads();
    
    // 2b: Compute exp(score - max) and sum via parallel reduction.
    float local_sum = 0.0f;
    for (int i = tid; i < kv_len; i += nthreads) {
        float e = expf(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce[tid] += reduce[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];
    __syncthreads();
    
    // 2c: Normalize.
    for (int i = tid; i < kv_len; i += nthreads) {
        scores[i] *= inv_sum;
    }
    __syncthreads();
    
    // Step 3: Weighted sum of values (parallel over head_dim)
    float* out_head = out + head * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float sum = 0.0f;
        for (int pos = 0; pos < kv_len; pos++) {
            sum += scores[pos] * v_head[pos * head_dim + d];
        }
        out_head[d] = sum;
    }
}

// ============================================================================
// Weighted accumulation: out[i] += scale * x[i]
// ============================================================================

__global__ void scaled_add_f32(float* out, const float* x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] += scale * x[idx];
    }
}

// ============================================================================
// DeltaNet: Depthwise 1D Convolution + SiLU
// ============================================================================
//
// conv_state: [(kernel_size-1), channels] ring buffer
// qkv_in: [channels] current input
// conv_w: [channels * kernel_size] (GGML: weight[ch * ks + ki])
// conv_out: [channels] = silu(conv(state, qkv))
// Updates conv_state in-place.

__global__ void deltanet_conv1d_silu(
    float* conv_state,
    const float* qkv_in,
    const float* conv_w,
    float* conv_out,
    int channels,
    int kernel_size
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int buf_len = kernel_size - 1;

    float sum = 0.0f;
    for (int ki = 0; ki < buf_len; ki++) {
        sum += conv_state[ki * channels + ch] * conv_w[ch * kernel_size + ki];
    }
    sum += qkv_in[ch] * conv_w[ch * kernel_size + (kernel_size - 1)];

    // SiLU
    float sig = 1.0f / (1.0f + expf(-sum));
    conv_out[ch] = sum * sig;

    // Update state: shift left and append current input
    // Each thread handles its own channel - no race
    for (int ki = 0; ki < buf_len - 1; ki++) {
        conv_state[ki * channels + ch] = conv_state[(ki + 1) * channels + ch];
    }
    conv_state[(buf_len - 1) * channels + ch] = qkv_in[ch];
}

// ============================================================================
// DeltaNet: Full recurrent state update + gated RMS norm output
// ============================================================================
//
// One block per value head. Single thread per block handles all per-head math
// (head dimensions are small ~64-128, so GPU parallelism is across heads).
//
// conv_out: [qkv_dim] - output of conv1d+silu
// gate_z: [d_inner] - output of gate projection
// ba_raw: [num_k_heads * ba_per_group] - beta/alpha from ssm_ba projection
// ssm_a: [num_v_heads] - decay multiplier
// dt_bias: [num_v_heads] - decay bias
// norm_w: [head_v_dim] - RMS norm weights
// ssm_state: [num_v_heads * head_v_dim * head_k_dim] - recurrent state (modified)
// output: [d_inner] - final output before output projection

// config_buf: [num_v_heads, num_k_heads, head_v_dim, head_k_dim, kv_ratio, d_inner, qkv_dim]
// norm_eps passed as separate float parameter
__global__ void deltanet_recurrent(
    float* ssm_state,
    const float* conv_out,
    const float* gate_z,
    const float* ba_raw,
    const float* ssm_a,
    const float* dt_bias,
    const float* norm_w,
    float* output,
    const int* config_buf,
    float norm_eps
) {
    int num_v_heads = config_buf[0];
    int num_k_heads = config_buf[1];
    int head_v_dim  = config_buf[2];
    int head_k_dim  = config_buf[3];
    int kv_ratio    = config_buf[4];

    int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    if (threadIdx.x != 0) return;

    // Beta/alpha uses interleaved group layout for the packed ba_raw buffer
    int ba_kh = vh / kv_ratio;
    int ba_r = vh % kv_ratio;

    int ba_per_group = 2 * kv_ratio;
    int group_offset = ba_kh * ba_per_group;
    float beta_raw = ba_raw[group_offset + ba_r];
    float alpha_raw = ba_raw[group_offset + kv_ratio + ba_r];

    float beta = 1.0f / (1.0f + expf(-beta_raw));

    // Gate: softplus(alpha + dt_bias) * ssm_a
    float gate_val = alpha_raw + dt_bias[vh];
    float gate;
    if (gate_val > 20.0f) gate = gate_val;
    else gate = logf(1.0f + expf(gate_val));
    gate *= ssm_a[vh];

    // Q, K, V offsets into conv_out
    // Layout: [Q: num_k_heads*head_k_dim | K: num_k_heads*head_k_dim | V: d_inner]
    // Q and K are tiled (ggml_repeat pattern): v-head maps to k-head via modulo
    int q_dim = num_k_heads * head_k_dim;
    int qk_head = vh % num_k_heads;
    int q_offset = qk_head * head_k_dim;
    int k_offset = q_dim + qk_head * head_k_dim;
    int v_offset = q_dim + q_dim + vh * head_v_dim;

    // L2 normalize Q (per k-head)
    float q_norm_sq = 0.0f;
    for (int i = 0; i < head_k_dim; i++) {
        float val = conv_out[q_offset + i];
        q_norm_sq += val * val;
    }
    float q_inv_norm = rsqrtf(q_norm_sq + 1e-6f);
    float q_scale = rsqrtf((float)head_k_dim);

    // L2 normalize K (per k-head)
    float k_norm_sq = 0.0f;
    for (int i = 0; i < head_k_dim; i++) {
        float val = conv_out[k_offset + i];
        k_norm_sq += val * val;
    }
    float k_inv_norm = rsqrtf(k_norm_sq + 1e-6f);

    // State pointer for this head
    int s_offset = vh * head_v_dim * head_k_dim;
    float* s = ssm_state + s_offset;

    // Decay state: s *= exp(gate)
    float decay = expf(gate);
    if (decay > 1e10f) decay = 1e10f;
    for (int i = 0; i < head_v_dim * head_k_dim; i++) {
        s[i] *= decay;
    }

    // sk = S^T @ k_normalized → [head_v_dim]
    // S is [head_v_dim, head_k_dim], k is [head_k_dim]
    // sk[vi] = sum_ki(S[vi, ki] * k[ki])
    for (int vi = 0; vi < head_v_dim; vi++) {
        float dot = 0.0f;
        for (int ki = 0; ki < head_k_dim; ki++) {
            float k_val = conv_out[k_offset + ki] * k_inv_norm;
            dot += s[vi * head_k_dim + ki] * k_val;
        }

        // delta = (v - sk) * beta
        float v_val = conv_out[v_offset + vi];
        float delta = (v_val - dot) * beta;

        // State update: s[vi, :] += delta * k^T
        for (int ki = 0; ki < head_k_dim; ki++) {
            float k_val = conv_out[k_offset + ki] * k_inv_norm;
            s[vi * head_k_dim + ki] += delta * k_val;
        }
    }

    // Output: o = S @ q_normalized → [head_v_dim]
    int o_offset = vh * head_v_dim;
    for (int vi = 0; vi < head_v_dim; vi++) {
        float dot = 0.0f;
        for (int ki = 0; ki < head_k_dim; ki++) {
            float q_val = conv_out[q_offset + ki] * q_inv_norm * q_scale;
            dot += s[vi * head_k_dim + ki] * q_val;
        }
        output[o_offset + vi] = dot;
    }

    // Gated RMS norm: output = rms_norm(output) * silu(gate_z)
    float ss = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float val = output[o_offset + d];
        ss += val * val;
    }
    float rms = sqrtf(ss / (float)head_v_dim + norm_eps);
    float rms_inv = 1.0f / rms;

    int norm_w_len = head_v_dim;
    for (int d = 0; d < head_v_dim; d++) {
        float normed = output[o_offset + d] * rms_inv * norm_w[d % norm_w_len];
        float z = gate_z[o_offset + d];
        float silu_z = z / (1.0f + expf(-z));
        output[o_offset + d] = normed * silu_z;
    }
}

// Per-head RMS normalization for QK norm (Qwen3)
// data: [num_heads * kl] in-place
// norm_weight: [norm_dim]
// One block per head, threads cooperate on the reduction
__global__ void qk_norm_per_head(float* data, const float* norm_weight,
                                  float eps, int num_heads, int kl, int norm_dim) {
    extern __shared__ float shared[];
    float* reduce = shared;

    int head = blockIdx.x;
    if (head >= num_heads) return;

    float* head_data = data + head * kl;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    // Sum of squares
    float local_ss = 0.0f;
    for (int d = tid; d < norm_dim; d += nt) {
        local_ss += head_data[d] * head_data[d];
    }
    reduce[tid] = local_ss;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(reduce[0] / (float)norm_dim + eps);

    // Scale
    for (int d = tid; d < norm_dim && d < kl; d += nt) {
        head_data[d] = head_data[d] * rms_inv * norm_weight[d];
    }
}

// Partial RoPE: apply RoPE only to the last rope_dims dimensions of each head
// q: [num_q_heads * kl], k: [num_kv_heads * kl], both in-place
// Dispatch: (num_q_heads + num_kv_heads) blocks
__global__ void partial_rope(float* q, float* k,
                              int num_q_heads, int num_kv_heads,
                              int kl, int rope_dims,
                              int pos, float freq_base, float freq_scale,
                              int use_neox) {
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int half_dim = rope_dims / 2;
    if (tid >= half_dim) return;

    float* head_ptr;
    int nope_dims = kl - rope_dims;

    if (block < num_q_heads) {
        head_ptr = q + block * kl + nope_dims;
    } else {
        int kv_head = block - num_q_heads;
        if (kv_head >= num_kv_heads) return;
        head_ptr = k + kv_head * kl + nope_dims;
    }

    float freq = 1.0f / powf(freq_base, (float)(2 * tid) / (float)rope_dims);
    float position = (float)pos / freq_scale;
    float theta = position * freq;
    float cos_val = cosf(theta);
    float sin_val = sinf(theta);

    if (use_neox) {
        float x0 = head_ptr[tid];
        float x1 = head_ptr[tid + half_dim];
        head_ptr[tid]            = x0 * cos_val - x1 * sin_val;
        head_ptr[tid + half_dim] = x1 * cos_val + x0 * sin_val;
    } else {
        float x0 = head_ptr[2 * tid];
        float x1 = head_ptr[2 * tid + 1];
        head_ptr[2 * tid]     = x0 * cos_val - x1 * sin_val;
        head_ptr[2 * tid + 1] = x1 * cos_val + x0 * sin_val;
    }
}

// Attention gate: out[i] = sigmoid(gate[i]) * attn[i]
__global__ void attention_gate_sigmoid(const float* gate, const float* attn,
                                        float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = 1.0f / (1.0f + expf(-gate[idx]));
        out[idx] = g * attn[idx];
    }
}

// Split Q output into q_proper and gate per head
// q_raw: [num_heads * (kl + vl)] — per head: [q(kl) | gate(vl)]
// q_out: [num_heads * kl], gate_out: [num_heads * vl]
__global__ void split_q_gate(const float* q_raw, float* q_out, float* gate_out,
                              int num_heads, int kl, int vl) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int per_head = kl + vl;

    for (int d = tid; d < kl; d += nt) {
        q_out[head * kl + d] = q_raw[head * per_head + d];
    }
    for (int d = tid; d < vl; d += nt) {
        gate_out[head * vl + d] = q_raw[head * per_head + kl + d];
    }
}

// Full attention with causal masking (for non-cached attention)
// q: [num_heads, seq_len, head_dim], k: [num_kv_heads, kv_len, head_dim]
// v: [num_kv_heads, kv_len, head_dim], out: [num_heads, seq_len, head_dim]
// Online softmax — uses O(head_dim + blockDim.x) shared memory
// Dispatch: grid(num_heads, seq_len), block(256)
__global__ void attention_full(const float* q, const float* k, const float* v,
                                float* out,
                                int num_heads, int num_kv_heads,
                                int seq_len, int kv_len, int head_dim,
                                float scale) {
    extern __shared__ float shared[];
    float* accum     = shared;              // [head_dim]
    float* reduction = shared + head_dim;   // [blockDim.x]
    // 4 extra floats at shared + head_dim + blockDim.x
    float* s_vals    = shared + head_dim + blockDim.x;

    int head = blockIdx.x;
    int s    = blockIdx.y;
    int tid  = threadIdx.x;
    int nt   = blockDim.x;

    int kv_head = head / (num_heads / num_kv_heads);
    int q_abs_pos = kv_len - seq_len + s;

    for (int d = tid; d < head_dim; d += nt) {
        accum[d] = 0.0f;
    }
    if (tid == 0) {
        s_vals[0] = -CUDART_INF_F; // max_score
        s_vals[1] = 0.0f;          // sum_exp
    }
    __syncthreads();

    int q_base = head * seq_len * head_dim + s * head_dim;

    for (int kv_pos = 0; kv_pos <= q_abs_pos && kv_pos < kv_len; kv_pos++) {
        // Dot product Q · K
        float local_dot = 0.0f;
        int k_base = kv_head * kv_len * head_dim + kv_pos * head_dim;
        for (int d = tid; d < head_dim; d += nt) {
            local_dot += q[q_base + d] * k[k_base + d];
        }
        reduction[tid] = local_dot;
        __syncthreads();
        for (int stride = nt / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }

        float score = reduction[0] * scale;

        if (tid == 0) {
            float old_max = s_vals[0];
            if (score > old_max) {
                s_vals[3] = expf(old_max - score); // correction
                s_vals[1] *= s_vals[3];
                s_vals[0] = score;
            } else {
                s_vals[3] = 1.0f;
            }
            s_vals[2] = expf(score - s_vals[0]); // weight
            s_vals[1] += s_vals[2];
        }
        __syncthreads();

        float w = s_vals[2];
        float c = s_vals[3];
        int v_base = kv_head * kv_len * head_dim + kv_pos * head_dim;
        for (int d = tid; d < head_dim; d += nt) {
            accum[d] = accum[d] * c + w * v[v_base + d];
        }
        __syncthreads();
    }

    float inv_sum = (s_vals[1] > 0.0f) ? 1.0f / s_vals[1] : 0.0f;
    int out_base = head * seq_len * head_dim + s * head_dim;
    for (int d = tid; d < head_dim; d += nt) {
        out[out_base + d] = accum[d] * inv_sum;
    }
}

// FP16 conversion: f32 -> f16
__global__ void f32_to_f16(const float* input, half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

// FP16 conversion: f16 -> f32
__global__ void f16_to_f32(const half* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// FP16 vector-matrix multiply: out = x @ W^T (FP16 compute, FP32 accumulation)
// x: [in_dim] (f16), W: [out_dim, in_dim] (f16), out: [out_dim] (f16)
// Each thread block computes one output element
__global__ void vec_mat_f16(const half* x, const half* weight,
                             half* out, int in_dim, int out_dim) {
    extern __shared__ float reduce[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    if (row >= out_dim) return;

    float sum = 0.0f;
    const half* w_row = weight + row * in_dim;
    for (int i = tid; i < in_dim; i += nt) {
        sum += __half2float(x[i]) * __half2float(w_row[i]);
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = __float2half(reduce[0]);
    }
}

// FP16 element-wise add
__global__ void add_f16(const half* a, const half* b, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// FP16 SiLU activation
__global__ void silu_f16(const half* input, half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

// FP16 element-wise multiply
__global__ void mul_f16(const half* a, const half* b, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

// FP16 RMS norm (fused: normalize and scale)
__global__ void rms_norm_f16(const half* x, const half* weight,
                              half* out, int n, float eps) {
    extern __shared__ float reduce[];
    int tid = threadIdx.x;
    int nt  = blockDim.x;

    float local_ss = 0.0f;
    for (int i = tid; i < n; i += nt) {
        float xi = __half2float(x[i]);
        local_ss += xi * xi;
    }
    reduce[tid] = local_ss;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(reduce[0] / (float)n + eps);
    for (int i = tid; i < n; i += nt) {
        out[i] = __float2half(__half2float(x[i]) * rms_inv * __half2float(weight[i]));
    }
}

// Flash Attention (cached) — O(head_dim) shared memory, supports any kv_len
// Replaces attention_multihead for generation (single query per head)
// Uses online softmax: no materialization of full score array
// Dispatch: grid(num_heads), block(256)
__global__ void flash_attention_cached(
    const float* q,         // [num_heads * head_dim]
    const float* k_cache,   // [num_kv_heads * max_seq_len * head_dim]
    const float* v_cache,   // [num_kv_heads * max_seq_len * head_dim]
    float* out,             // [num_heads * head_dim]
    int num_heads, int num_kv_heads,
    int head_dim, int max_seq_len,
    int kv_len, float scale
) {
    extern __shared__ float smem[];
    float* accum   = smem;                         // [head_dim]
    float* reduce  = smem + head_dim;              // [blockDim.x]
    float* s_vals  = smem + head_dim + blockDim.x; // [4]: max, sum, weight, correction

    int head = blockIdx.x;
    int tid  = threadIdx.x;
    int nt   = blockDim.x;
    int kv_head = head / (num_heads / num_kv_heads);

    for (int d = tid; d < head_dim; d += nt) accum[d] = 0.0f;
    if (tid == 0) { s_vals[0] = -CUDART_INF_F; s_vals[1] = 0.0f; }
    __syncthreads();

    const float* q_head = q + head * head_dim;
    int k_off = kv_head * max_seq_len * head_dim;
    int v_off = kv_head * max_seq_len * head_dim;

    for (int pos = 0; pos < kv_len; pos++) {
        float local_dot = 0.0f;
        for (int d = tid; d < head_dim; d += nt)
            local_dot += q_head[d] * k_cache[k_off + pos * head_dim + d];
        reduce[tid] = local_dot;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        float score = reduce[0] * scale;

        if (tid == 0) {
            float old_max = s_vals[0];
            if (score > old_max) {
                s_vals[3] = expf(old_max - score);
                s_vals[1] *= s_vals[3];
                s_vals[0] = score;
            } else {
                s_vals[3] = 1.0f;
            }
            s_vals[2] = expf(score - s_vals[0]);
            s_vals[1] += s_vals[2];
        }
        __syncthreads();

        float w = s_vals[2], c = s_vals[3];
        for (int d = tid; d < head_dim; d += nt)
            accum[d] = accum[d] * c + w * v_cache[v_off + pos * head_dim + d];
        __syncthreads();
    }

    float inv = (s_vals[1] > 0.0f) ? 1.0f / s_vals[1] : 0.0f;
    int out_base = head * head_dim;
    for (int d = tid; d < head_dim; d += nt)
        out[out_base + d] = accum[d] * inv;
}

// Batched RoPE: apply RoPE to [num_heads, seq_len, head_dim] tensors
// Each position gets a different frequency based on (start_pos + s)
// Dispatch: grid(max(num_heads, num_kv_heads), seq_len), block(half_dim)
__global__ void rope_batch(float* q, float* k,
                            int num_heads, int num_kv_heads, int head_dim,
                            int seq_len, int start_pos,
                            float freq_base, float freq_scale,
                            int use_neox) {
    int head = blockIdx.x;
    int s    = blockIdx.y;
    int i    = threadIdx.x;
    int half_dim = head_dim / 2;

    if (i >= half_dim) return;

    int pos = start_pos + s;
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float position = (float)pos / freq_scale;
    float theta = position * freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    if (head < num_heads) {
        int base = head * seq_len * head_dim + s * head_dim;
        int idx0, idx1;
        if (use_neox) {
            idx0 = base + i;
            idx1 = base + i + half_dim;
        } else {
            idx0 = base + 2 * i;
            idx1 = base + 2 * i + 1;
        }
        float q0 = q[idx0];
        float q1 = q[idx1];
        q[idx0] = q0 * cos_theta - q1 * sin_theta;
        q[idx1] = q0 * sin_theta + q1 * cos_theta;
    }

    if (head < num_kv_heads) {
        int base = head * seq_len * head_dim + s * head_dim;
        int idx0, idx1;
        if (use_neox) {
            idx0 = base + i;
            idx1 = base + i + half_dim;
        } else {
            idx0 = base + 2 * i;
            idx1 = base + 2 * i + 1;
        }
        float k0 = k[idx0];
        float k1 = k[idx1];
        k[idx0] = k0 * cos_theta - k1 * sin_theta;
        k[idx1] = k0 * sin_theta + k1 * cos_theta;
    }
}

// Batched KV cache update: write seq_len positions to cache starting at start_pos
// k_new: [num_kv_heads, seq_len, head_dim]
// k_cache: [num_kv_heads, max_seq_len, head_dim]
// Dispatch: grid((total + 255) / 256), block(256)
__global__ void update_kv_cache_batch(const float* k_new, const float* v_new,
                                       float* k_cache, float* v_cache,
                                       int num_kv_heads, int head_dim,
                                       int max_seq_len, int seq_len,
                                       int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * seq_len * head_dim;
    if (idx >= total) return;

    int head = idx / (seq_len * head_dim);
    int remainder = idx % (seq_len * head_dim);
    int s = remainder / head_dim;
    int d = remainder % head_dim;

    int cache_pos = start_pos + s;
    int cache_idx = head * max_seq_len * head_dim + cache_pos * head_dim + d;

    k_cache[cache_idx] = k_new[idx];
    v_cache[cache_idx] = v_new[idx];
}

// Batched matrix-vector multiply for linear projections on batch of tokens
// x: [batch_size, in_dim], weight: [out_dim, in_dim], out: [batch_size, out_dim]
// Dispatch: grid(out_dim, batch_size), block(256)
__global__ void batched_linear(const float* x, const float* weight,
                                float* out, int batch_size,
                                int in_dim, int out_dim) {
    int row = blockIdx.x;   // output dimension
    int b   = blockIdx.y;   // batch index
    int tid = threadIdx.x;
    int nt  = blockDim.x;

    if (row >= out_dim || b >= batch_size) return;

    extern __shared__ float reduce[];

    float sum = 0.0f;
    const float* x_ptr = x + b * in_dim;
    const float* w_ptr = weight + row * in_dim;
    for (int i = tid; i < in_dim; i += nt) {
        sum += x_ptr[i] * w_ptr[i];
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[b * out_dim + row] = reduce[0];
    }
}

} // extern "C"
"#;

/// Compiled CUDA kernels
#[allow(dead_code)]
pub struct CudaKernels {
    // Element-wise
    pub add_f32: CudaFunction,
    pub mul_f32: CudaFunction,
    pub scale_f32: CudaFunction,

    // Activations
    pub silu_f32: CudaFunction,
    pub gelu_f32: CudaFunction,

    // Normalization (legacy two-pass)
    pub rms_norm_sum_sq: CudaFunction,
    pub rms_norm_scale: CudaFunction,
    // Normalization (fused single-pass)
    pub rms_norm_fused: CudaFunction,

    // Softmax (legacy three-pass)
    pub softmax_max: CudaFunction,
    pub softmax_exp_sum: CudaFunction,
    pub softmax_div: CudaFunction,
    // Softmax (fused single-pass)
    pub softmax_fused: CudaFunction,

    // Matrix ops
    pub vec_mat_f32: CudaFunction,

    // RoPE
    pub rope_single_pos: CudaFunction,

    // Quantized ops
    pub vec_mat_q4k: CudaFunction,
    pub vec_mat_q8_0: CudaFunction,
    pub vec_mat_q6k: CudaFunction,
    pub vec_mat_q5k: CudaFunction,

    // KV cache
    pub update_kv_cache: CudaFunction,
    pub attention_multihead: CudaFunction,
    pub vec_mat_q4_0: CudaFunction,

    // Attention
    pub attention_single_head: CudaFunction,
    pub attention_full: CudaFunction,

    // GPU attention support
    pub qk_norm_per_head: CudaFunction,
    pub partial_rope: CudaFunction,
    pub attention_gate_sigmoid: CudaFunction,
    pub split_q_gate: CudaFunction,

    // FP16 compute
    pub f32_to_f16: CudaFunction,
    pub f16_to_f32: CudaFunction,
    pub vec_mat_f16: CudaFunction,
    pub add_f16: CudaFunction,
    pub silu_f16: CudaFunction,
    pub mul_f16: CudaFunction,
    pub rms_norm_f16: CudaFunction,

    // Flash Attention
    pub flash_attention_cached: CudaFunction,

    // Batched inference
    pub rope_batch: CudaFunction,
    pub update_kv_cache_batch: CudaFunction,
    pub batched_linear: CudaFunction,

    // DeltaNet
    pub deltanet_conv1d_silu: CudaFunction,
    pub deltanet_recurrent: CudaFunction,

    // MoE accumulation
    pub scaled_add_f32: CudaFunction,
}

impl CudaKernels {
    /// Compile and load all CUDA kernels
    pub fn new(device: Arc<CudaDevice>) -> BackendResult<Self> {
        let mut include_paths = Vec::new();
        for candidate in &[
            "/usr/local/cuda/include",
            "/opt/cuda/targets/x86_64-linux/include",
            "/opt/cuda/include",
            "/usr/include",
        ] {
            if std::path::Path::new(candidate).join("cuda_fp16.h").exists() {
                include_paths.push(candidate.to_string());
                break;
            }
        }

        let opts = cudarc::nvrtc::CompileOptions {
            include_paths,
            ..Default::default()
        };

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SOURCE, opts).map_err(|e| {
            BackendError::InitializationFailed(format!("NVRTC compile failed: {}", e))
        })?;

        // Load module
        device
            .load_ptx(
                ptx,
                "llama_kernels",
                &[
                    "add_f32",
                    "mul_f32",
                    "scale_f32",
                    "silu_f32",
                    "gelu_f32",
                    "rms_norm_sum_sq",
                    "rms_norm_scale",
                    "rms_norm_fused",
                    "softmax_max",
                    "softmax_exp_sum",
                    "softmax_div",
                    "softmax_fused",
                    "vec_mat_f32",
                    "rope_single_pos",
                    "vec_mat_q4k",
                    "vec_mat_q8_0",
                    "vec_mat_q4_0",
                    "vec_mat_q6k",
                    "vec_mat_q5k",
                    "attention_single_head",
                    "attention_full",
                    "update_kv_cache",
                    "attention_multihead",
                    "qk_norm_per_head",
                    "partial_rope",
                    "attention_gate_sigmoid",
                    "split_q_gate",
                    "f32_to_f16",
                    "f16_to_f32",
                    "vec_mat_f16",
                    "add_f16",
                    "silu_f16",
                    "mul_f16",
                    "rms_norm_f16",
                    "flash_attention_cached",
                    "rope_batch",
                    "update_kv_cache_batch",
                    "batched_linear",
                    "scaled_add_f32",
                    "deltanet_conv1d_silu",
                    "deltanet_recurrent",
                ],
            )
            .map_err(|e| BackendError::InitializationFailed(format!("PTX load failed: {}", e)))?;

        // Get function handles
        Ok(Self {
            add_f32: device.get_func("llama_kernels", "add_f32").ok_or_else(|| {
                BackendError::InitializationFailed("Kernel 'add_f32' not found".into())
            })?,
            mul_f32: device.get_func("llama_kernels", "mul_f32").ok_or_else(|| {
                BackendError::InitializationFailed("Kernel 'mul_f32' not found".into())
            })?,
            scale_f32: device
                .get_func("llama_kernels", "scale_f32")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'scale_f32' not found".into())
                })?,
            silu_f32: device
                .get_func("llama_kernels", "silu_f32")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'silu_f32' not found".into())
                })?,
            gelu_f32: device
                .get_func("llama_kernels", "gelu_f32")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'gelu_f32' not found".into())
                })?,
            rms_norm_sum_sq: device
                .get_func("llama_kernels", "rms_norm_sum_sq")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'rms_norm_sum_sq' not found".into())
                })?,
            rms_norm_scale: device
                .get_func("llama_kernels", "rms_norm_scale")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'rms_norm_scale' not found".into())
                })?,
            rms_norm_fused: device
                .get_func("llama_kernels", "rms_norm_fused")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'rms_norm_fused' not found".into())
                })?,
            softmax_max: device
                .get_func("llama_kernels", "softmax_max")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'softmax_max' not found".into())
                })?,
            softmax_exp_sum: device
                .get_func("llama_kernels", "softmax_exp_sum")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'softmax_exp_sum' not found".into())
                })?,
            softmax_div: device
                .get_func("llama_kernels", "softmax_div")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'softmax_div' not found".into())
                })?,
            softmax_fused: device
                .get_func("llama_kernels", "softmax_fused")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'softmax_fused' not found".into())
                })?,
            vec_mat_f32: device
                .get_func("llama_kernels", "vec_mat_f32")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'vec_mat_f32' not found".into())
                })?,
            rope_single_pos: device
                .get_func("llama_kernels", "rope_single_pos")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'rope_single_pos' not found".into())
                })?,
            vec_mat_q4k: device
                .get_func("llama_kernels", "vec_mat_q4k")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'vec_mat_q4k' not found".into())
                })?,
            vec_mat_q8_0: device
                .get_func("llama_kernels", "vec_mat_q8_0")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'vec_mat_q8_0' not found".into())
                })?,
            vec_mat_q4_0: device
                .get_func("llama_kernels", "vec_mat_q4_0")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'vec_mat_q4_0' not found".into())
                })?,
            vec_mat_q6k: device
                .get_func("llama_kernels", "vec_mat_q6k")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'vec_mat_q6k' not found".into())
                })?,
            vec_mat_q5k: device
                .get_func("llama_kernels", "vec_mat_q5k")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'vec_mat_q5k' not found".into())
                })?,
            attention_single_head: device
                .get_func("llama_kernels", "attention_single_head")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'attention_single_head' not found".into(),
                    )
                })?,
            attention_full: device
                .get_func("llama_kernels", "attention_full")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'attention_full' not found".into(),
                    )
                })?,
            qk_norm_per_head: device
                .get_func("llama_kernels", "qk_norm_per_head")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'qk_norm_per_head' not found".into(),
                    )
                })?,
            partial_rope: device
                .get_func("llama_kernels", "partial_rope")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'partial_rope' not found".into(),
                    )
                })?,
            attention_gate_sigmoid: device
                .get_func("llama_kernels", "attention_gate_sigmoid")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'attention_gate_sigmoid' not found".into(),
                    )
                })?,
            split_q_gate: device
                .get_func("llama_kernels", "split_q_gate")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'split_q_gate' not found".into(),
                    )
                })?,
            f32_to_f16: device
                .get_func("llama_kernels", "f32_to_f16")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'f32_to_f16' not found".into(),
                    )
                })?,
            f16_to_f32: device
                .get_func("llama_kernels", "f16_to_f32")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'f16_to_f32' not found".into(),
                    )
                })?,
            vec_mat_f16: device
                .get_func("llama_kernels", "vec_mat_f16")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'vec_mat_f16' not found".into(),
                    )
                })?,
            add_f16: device
                .get_func("llama_kernels", "add_f16")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'add_f16' not found".into(),
                    )
                })?,
            silu_f16: device
                .get_func("llama_kernels", "silu_f16")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'silu_f16' not found".into(),
                    )
                })?,
            mul_f16: device
                .get_func("llama_kernels", "mul_f16")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'mul_f16' not found".into(),
                    )
                })?,
            rms_norm_f16: device
                .get_func("llama_kernels", "rms_norm_f16")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'rms_norm_f16' not found".into(),
                    )
                })?,
            flash_attention_cached: device
                .get_func("llama_kernels", "flash_attention_cached")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'flash_attention_cached' not found".into(),
                    )
                })?,
            rope_batch: device
                .get_func("llama_kernels", "rope_batch")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'rope_batch' not found".into(),
                    )
                })?,
            update_kv_cache_batch: device
                .get_func("llama_kernels", "update_kv_cache_batch")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'update_kv_cache_batch' not found".into(),
                    )
                })?,
            batched_linear: device
                .get_func("llama_kernels", "batched_linear")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'batched_linear' not found".into(),
                    )
                })?,
            update_kv_cache: device
                .get_func("llama_kernels", "update_kv_cache")
                .ok_or_else(|| {
                    BackendError::InitializationFailed("Kernel 'update_kv_cache' not found".into())
                })?,
            attention_multihead: device
                .get_func("llama_kernels", "attention_multihead")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'attention_multihead' not found".into(),
                    )
                })?,
            scaled_add_f32: device
                .get_func("llama_kernels", "scaled_add_f32")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'scaled_add_f32' not found".into(),
                    )
                })?,
            deltanet_conv1d_silu: device
                .get_func("llama_kernels", "deltanet_conv1d_silu")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'deltanet_conv1d_silu' not found".into(),
                    )
                })?,
            deltanet_recurrent: device
                .get_func("llama_kernels", "deltanet_recurrent")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'deltanet_recurrent' not found".into(),
                    )
                })?,
        })
    }
}
