//! CUDA GPU backend for tensor operations
//!
//! This module provides a CUDA-based GPU implementation of the Backend trait
//! for NVIDIA GPUs.
//!
//! # Features
//! - High-performance matrix operations via cuBLAS
//! - Custom CUDA kernels for quantized operations
//! - Efficient memory management
//!
//! # Requirements
//! - NVIDIA GPU with compute capability 6.0+
//! - CUDA Toolkit 11.0+
//! - Build with `--features cuda`

#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
mod memory;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use kernels::CudaKernels;

/// CUDA backend configuration
#[derive(Debug, Clone)]
pub struct CudaConfig {
    /// Device index to use (0 = first GPU)
    pub device_index: usize,
    /// Whether to use TensorCores (if available)
    pub use_tensor_cores: bool,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            use_tensor_cores: true,
        }
    }
}

/// CUDA GPU backend
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    config: CudaConfig,
    // CPU backend for fallback operations that aren't yet GPU-accelerated
    cpu_backend: crate::backend::cpu::CpuBackend,
}

#[cfg(not(feature = "cuda"))]
pub struct CudaBackend {
    config: CudaConfig,
}

impl CudaBackend {
    /// Create a new CUDA backend with default configuration
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(CudaConfig::default())
    }

    /// Create a CUDA backend with custom configuration
    #[cfg(feature = "cuda")]
    pub fn with_config(config: CudaConfig) -> Result<Self, BackendError> {
        let device = CudaDevice::new(config.device_index)
            .map_err(|e| BackendError::InitializationFailed(format!("CUDA init failed: {}", e)))?;
        
        let kernels = CudaKernels::new(device.clone())?;
        
        Ok(Self {
            device,
            kernels,
            config,
            cpu_backend: crate::backend::cpu::CpuBackend::new(),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn with_config(_config: CudaConfig) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "CUDA support not compiled. Build with --features cuda".to_string(),
        ))
    }

    /// Get device name
    #[cfg(feature = "cuda")]
    pub fn device_name(&self) -> String {
        format!("CUDA Device {}", self.config.device_index)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn device_name(&self) -> String {
        "CUDA disabled".to_string()
    }

    /// Allocate GPU memory and copy data
    #[cfg(feature = "cuda")]
    fn to_device(&self, data: &[f32]) -> Result<CudaSlice<f32>, BackendError> {
        self.device
            .htod_sync_copy(data)
            .map_err(|e| BackendError::AllocationFailed(format!("GPU copy failed: {}", e)))
    }

    /// Copy data back from GPU
    #[cfg(feature = "cuda")]
    fn from_device(&self, slice: &CudaSlice<f32>) -> Result<Vec<f32>, BackendError> {
        self.device
            .dtoh_sync_copy(slice)
            .map_err(|e| BackendError::OperationFailed(format!("GPU read failed: {}", e)))
    }
    
    /// Allocate GPU buffer
    #[cfg(feature = "cuda")]
    fn alloc_gpu(&self, size: usize) -> Result<CudaSlice<f32>, BackendError> {
        self.device
            .alloc_zeros::<f32>(size)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            #[cfg(feature = "cuda")]
            panic!("Failed to create CUDA backend");
            #[cfg(not(feature = "cuda"))]
            Self {
                config: CudaConfig::default(),
            }
        })
    }
}

/// Helper function to create a 1D launch configuration
#[cfg(feature = "cuda")]
fn launch_config_1d(n: usize, block_size: usize) -> LaunchConfig {
    let grid_size = (n + block_size - 1) / block_size;
    LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Helper function with shared memory
#[cfg(feature = "cuda")]
fn launch_config_1d_shared(n: usize, block_size: usize, shared_bytes: usize) -> LaunchConfig {
    let grid_size = (n + block_size - 1) / block_size;
    LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    }
}

#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor> {
        Ok(Tensor::zeros(shape.to_vec(), dtype))
    }

    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor> {
        Ok(tensor.clone())
    }

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = a_data.len();

        // Upload to GPU
        let a_gpu = self.to_device(a_data)?;
        let b_gpu = self.to_device(b_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;

        // Launch kernel
        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.add_f32.clone().launch(config, (&a_gpu, &b_gpu, &mut out_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("add kernel failed: {}", e)))?;

        // Copy back
        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = a_data.len();

        let a_gpu = self.to_device(a_data)?;
        let b_gpu = self.to_device(b_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;

        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.mul_f32.clone().launch(config, (&a_gpu, &b_gpu, &mut out_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("mul kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
        let a_data = a.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = a_data.len();

        let a_gpu = self.to_device(a_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;

        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.scale_f32.clone().launch(config, (&a_gpu, scalar, &mut out_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("scale kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        let x_data = x.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = x_data.len();

        let x_gpu = self.to_device(x_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;

        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.silu_f32.clone().launch(config, (&x_gpu, &mut out_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("silu kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        let x_data = x.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = x_data.len();

        let x_gpu = self.to_device(x_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;

        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.gelu_f32.clone().launch(config, (&x_gpu, &mut out_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("gelu kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        let x_data = x.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = x_data.len();

        let x_gpu = self.to_device(x_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;
        let mut max_gpu = self.device.alloc_zeros::<f32>(1)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        let mut sum_gpu = self.device.alloc_zeros::<f32>(1)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        // Initialize max to negative infinity
        self.device.htod_sync_copy_into(&[f32::NEG_INFINITY], &mut max_gpu)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;

        let block_size = 256;
        let config = launch_config_1d_shared(n, block_size, block_size * 4);

        // Pass 1: Find max
        unsafe {
            self.kernels.softmax_max.clone().launch(config.clone(), (&x_gpu, &mut max_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("softmax_max kernel failed: {}", e)))?;

        // Read max value
        let max_val = self.from_device(&max_gpu)?[0];

        // Pass 2: Compute exp and sum
        unsafe {
            self.kernels.softmax_exp_sum.clone().launch(config.clone(), (&x_gpu, &mut out_gpu, &mut sum_gpu, max_val, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("softmax_exp_sum kernel failed: {}", e)))?;

        // Read sum
        let sum_val = self.from_device(&sum_gpu)?[0];
        let sum_inv = 1.0 / sum_val;

        // Pass 3: Normalize
        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.softmax_div.clone().launch(config, (&mut out_gpu, sum_inv, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("softmax_div kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) -> BackendResult<()> {
        let x_data = x.as_f32()?;
        let w_data = weight.as_f32()?;
        let out_data = out.as_f32_mut()?;
        let n = x_data.len();

        let x_gpu = self.to_device(x_data)?;
        let w_gpu = self.to_device(w_data)?;
        let mut out_gpu = self.alloc_gpu(n)?;
        let mut sum_sq_gpu = self.device.alloc_zeros::<f32>(1)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;

        let block_size = 256;
        let config = launch_config_1d_shared(n, block_size, block_size * 4);

        // Pass 1: Compute sum of squares
        unsafe {
            self.kernels.rms_norm_sum_sq.clone().launch(config, (&x_gpu, &mut sum_sq_gpu, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("rms_norm_sum_sq kernel failed: {}", e)))?;

        // Read sum and compute RMS inverse
        let sum_sq = self.from_device(&sum_sq_gpu)?[0];
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let rms_inv = 1.0 / rms;

        // Pass 2: Normalize and scale
        let config = launch_config_1d(n, 256);
        unsafe {
            self.kernels.rms_norm_scale.clone().launch(config, (&x_gpu, &w_gpu, &mut out_gpu, rms_inv, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("rms_norm_scale kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);
        
        Ok(())
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for matmul until cuBLAS is properly integrated
        self.cpu_backend.matmul(a, b, out)
    }

    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.matvec(a, b, out)
    }

    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Use GPU kernel for vec_mat (critical path)
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out_data = out.as_f32_mut()?;

        let k = b.shape()[0];
        let n_out = b.shape()[1];

        let a_gpu = self.to_device(a_data)?;
        let b_gpu = self.to_device(b_data)?;
        let mut out_gpu = self.alloc_gpu(n_out)?;

        // Use our custom kernel for vec_mat
        let config = launch_config_1d(n_out, 256);
        unsafe {
            self.kernels.vec_mat_f32.clone().launch(config, (&a_gpu, &b_gpu, &mut out_gpu, k as i32, n_out as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("vec_mat kernel failed: {}", e)))?;

        let result = self.from_device(&out_gpu)?;
        out_data.copy_from_slice(&result);

        Ok(())
    }

    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.dequantize(src, out)
    }

    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.matvec_q(a, b, out)
    }

    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for quantized ops
        self.cpu_backend.vec_mat_q(a, b, out)
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
        // Use CPU for RoPE - the GPU kernel has complex shape requirements
        self.cpu_backend.rope(q, k, pos, freq_base, freq_scale, use_neox)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
    ) -> BackendResult<()> {
        // Use CPU for attention - GPU attention has complex implementation
        self.cpu_backend.attention(q, k, v, out, scale)
    }
}

#[cfg(not(feature = "cuda"))]
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn rms_norm(&self, _x: &Tensor, _weight: &Tensor, _eps: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn rope(&self, _q: &mut Tensor, _k: &mut Tensor, _pos: usize, _freq_base: f32, _freq_scale: f32, _use_neox: bool) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn attention(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor, _out: &mut Tensor, _scale: f32) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_config_default() {
        let config = CudaConfig::default();
        assert_eq!(config.device_index, 0);
        assert!(config.use_tensor_cores);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_backend_creation() {
        match CudaBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "cuda");
                assert!(backend.is_available());
                println!("CUDA backend created: {}", backend.device_name());
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
            }
        }
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_add() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[0.5, 0.5, 0.5, 0.5], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        backend.add(&a, &b, &mut out).unwrap();
        
        let result = out.as_f32().unwrap();
        assert!((result[0] - 1.5).abs() < 1e-5);
        assert!((result[1] - 2.5).abs() < 1e-5);
        assert!((result[2] - 3.5).abs() < 1e-5);
        assert!((result[3] - 4.5).abs() < 1e-5);
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_silu() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        
        let x = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        backend.silu(&x, &mut out).unwrap();
        
        let result = out.as_f32().unwrap();
        // SiLU(0) = 0
        assert!(result[0].abs() < 1e-5);
        // SiLU(1) = 1 / (1 + e^-1) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.01);
    }
}
