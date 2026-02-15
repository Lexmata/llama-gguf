//! CUDA kernel implementations for tensor operations
//!
//! This module contains PTX kernels for GPU-accelerated operations.
//! Kernels are compiled at runtime using cudarc's nvrtc support.

use cudarc::driver::{CudaDevice, CudaFunction};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};

/// CUDA kernel source code
pub const KERNEL_SOURCE: &str = r#"
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
                                 int num_heads, int head_dim,
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
    
    int base = head * head_dim;
    int idx0, idx1;
    
    if (use_neox) {
        // NeoX: (i, i + half_dim)
        idx0 = base + i;
        idx1 = base + i + half_dim;
    } else {
        // LLaMA: (2*i, 2*i+1)
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }
    
    // Rotate Q
    float q0 = q[idx0];
    float q1 = q[idx1];
    q[idx0] = q0 * cos_theta - q1 * sin_theta;
    q[idx1] = q0 * sin_theta + q1 * cos_theta;
    
    // Rotate K
    float k0 = k[idx0];
    float k1 = k[idx1];
    k[idx0] = k0 * cos_theta - k1 * sin_theta;
    k[idx1] = k0 * sin_theta + k1 * cos_theta;
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

} // extern "C"
"#;

/// Compiled CUDA kernels
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

    // KV cache
    pub update_kv_cache: CudaFunction,
    pub attention_multihead: CudaFunction,
    pub vec_mat_q4_0: CudaFunction,

    // Attention
    pub attention_single_head: CudaFunction,
}

impl CudaKernels {
    /// Compile and load all CUDA kernels
    pub fn new(device: Arc<CudaDevice>) -> BackendResult<Self> {
        // Compile PTX
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE).map_err(|e| {
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
                    "attention_single_head",
                    "update_kv_cache",
                    "attention_multihead",
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
            attention_single_head: device
                .get_func("llama_kernels", "attention_single_head")
                .ok_or_else(|| {
                    BackendError::InitializationFailed(
                        "Kernel 'attention_single_head' not found".into(),
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
        })
    }
}
