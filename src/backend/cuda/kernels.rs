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

// RMS normalization - two-pass algorithm
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

// ============================================================================
// Matrix operations
// ============================================================================

// Vector-matrix multiplication: out = vec @ mat
// vec: [k], mat: [k, n], out: [n]
__global__ void vec_mat_f32(const float* vec, const float* mat, float* out,
                            int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += vec[i] * mat[i * n + col];
        }
        out[col] = sum;
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
    
    // Normalization
    pub rms_norm_sum_sq: CudaFunction,
    pub rms_norm_scale: CudaFunction,
    
    // Softmax
    pub softmax_max: CudaFunction,
    pub softmax_exp_sum: CudaFunction,
    pub softmax_div: CudaFunction,
    
    // Matrix ops
    pub vec_mat_f32: CudaFunction,
}

impl CudaKernels {
    /// Compile and load all CUDA kernels
    pub fn new(device: Arc<CudaDevice>) -> BackendResult<Self> {
        // Compile PTX
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)
            .map_err(|e| BackendError::InitializationFailed(format!("NVRTC compile failed: {}", e)))?;
        
        // Load module
        device.load_ptx(ptx, "llama_kernels", &[
            "add_f32", "mul_f32", "scale_f32",
            "silu_f32", "gelu_f32",
            "rms_norm_sum_sq", "rms_norm_scale",
            "softmax_max", "softmax_exp_sum", "softmax_div",
            "vec_mat_f32",
        ]).map_err(|e| BackendError::InitializationFailed(format!("PTX load failed: {}", e)))?;
        
        // Get function handles
        Ok(Self {
            add_f32: device.get_func("llama_kernels", "add_f32")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'add_f32' not found".into()))?,
            mul_f32: device.get_func("llama_kernels", "mul_f32")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'mul_f32' not found".into()))?,
            scale_f32: device.get_func("llama_kernels", "scale_f32")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'scale_f32' not found".into()))?,
            silu_f32: device.get_func("llama_kernels", "silu_f32")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'silu_f32' not found".into()))?,
            gelu_f32: device.get_func("llama_kernels", "gelu_f32")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'gelu_f32' not found".into()))?,
            rms_norm_sum_sq: device.get_func("llama_kernels", "rms_norm_sum_sq")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'rms_norm_sum_sq' not found".into()))?,
            rms_norm_scale: device.get_func("llama_kernels", "rms_norm_scale")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'rms_norm_scale' not found".into()))?,
            softmax_max: device.get_func("llama_kernels", "softmax_max")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'softmax_max' not found".into()))?,
            softmax_exp_sum: device.get_func("llama_kernels", "softmax_exp_sum")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'softmax_exp_sum' not found".into()))?,
            softmax_div: device.get_func("llama_kernels", "softmax_div")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'softmax_div' not found".into()))?,
            vec_mat_f32: device.get_func("llama_kernels", "vec_mat_f32")
                .ok_or_else(|| BackendError::InitializationFailed("Kernel 'vec_mat_f32' not found".into()))?,
        })
    }
}
