//! GPU memory management for CUDA backend
//!
//! This module provides GPU-resident tensors and memory management utilities.

use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::tensor::DType;

/// A tensor stored in GPU memory
#[allow(dead_code)]
pub struct GpuTensor {
    /// GPU memory buffer
    data: GpuBuffer,
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
}

/// GPU memory buffer (type-erased for flexibility)
#[allow(dead_code)]
pub enum GpuBuffer {
    F32(CudaSlice<f32>),
    F16(CudaSlice<u16>),  // f16 stored as u16
    U8(CudaSlice<u8>),    // for quantized data
}

#[allow(dead_code)]
impl GpuTensor {
    /// Create a new GPU tensor with uninitialized memory
    pub fn alloc(device: &Arc<CudaDevice>, shape: Vec<usize>, dtype: DType) -> BackendResult<Self> {
        let numel: usize = shape.iter().product();
        
        let data = match dtype {
            DType::F32 => {
                let slice = device.alloc_zeros::<f32>(numel)
                    .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
                GpuBuffer::F32(slice)
            }
            DType::F16 => {
                let slice = device.alloc_zeros::<u16>(numel)
                    .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
                GpuBuffer::F16(slice)
            }
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | 
            DType::Q8_0 | DType::Q8_1 | 
            DType::Q2K | DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K | DType::Q8K => {
                // Quantized types: allocate as raw bytes
                let bytes = quantized_bytes(numel, dtype);
                let slice = device.alloc_zeros::<u8>(bytes)
                    .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
                GpuBuffer::U8(slice)
            }
            _ => return Err(BackendError::UnsupportedDType(dtype)),
        };
        
        Ok(Self { data, shape, dtype })
    }
    
    /// Create GPU tensor from CPU data (f32)
    pub fn from_f32(device: &Arc<CudaDevice>, data: &[f32], shape: Vec<usize>) -> BackendResult<Self> {
        let slice = device.htod_sync_copy(data)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        
        Ok(Self {
            data: GpuBuffer::F32(slice),
            shape,
            dtype: DType::F32,
        })
    }
    
    /// Create GPU tensor from raw bytes (for quantized data)
    pub fn from_bytes(device: &Arc<CudaDevice>, data: &[u8], shape: Vec<usize>, dtype: DType) -> BackendResult<Self> {
        let slice = device.htod_sync_copy(data)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        
        Ok(Self {
            data: GpuBuffer::U8(slice),
            shape,
            dtype,
        })
    }
    
    /// Copy tensor data back to CPU
    pub fn to_f32(&self, device: &Arc<CudaDevice>) -> BackendResult<Vec<f32>> {
        match &self.data {
            GpuBuffer::F32(slice) => {
                device.dtoh_sync_copy(slice)
                    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
            }
            _ => Err(BackendError::DTypeMismatch {
                expected: DType::F32,
                got: self.dtype,
            }),
        }
    }
    
    /// Get a reference to the underlying GPU slice (f32)
    pub fn as_f32_slice(&self) -> BackendResult<&CudaSlice<f32>> {
        match &self.data {
            GpuBuffer::F32(slice) => Ok(slice),
            _ => Err(BackendError::DTypeMismatch {
                expected: DType::F32,
                got: self.dtype,
            }),
        }
    }
    
    /// Get a mutable reference to the underlying GPU slice (f32)
    pub fn as_f32_slice_mut(&mut self) -> BackendResult<&mut CudaSlice<f32>> {
        match &mut self.data {
            GpuBuffer::F32(slice) => Ok(slice),
            _ => Err(BackendError::DTypeMismatch {
                expected: DType::F32,
                got: self.dtype,
            }),
        }
    }
    
    /// Get a reference to the underlying GPU slice (u8)
    pub fn as_u8_slice(&self) -> BackendResult<&CudaSlice<u8>> {
        match &self.data {
            GpuBuffer::U8(slice) => Ok(slice),
            _ => Err(BackendError::DTypeMismatch {
                expected: DType::Q4K,  // Generic quantized
                got: self.dtype,
            }),
        }
    }
    
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Calculate bytes needed for quantized data
#[allow(dead_code)]
fn quantized_bytes(numel: usize, dtype: DType) -> usize {
    match dtype {
        DType::Q4_0 => (numel / 32) * 18,  // 32 values -> 18 bytes
        DType::Q4_1 => (numel / 32) * 20,
        DType::Q5_0 => (numel / 32) * 22,
        DType::Q5_1 => (numel / 32) * 24,
        DType::Q8_0 => (numel / 32) * 34,
        DType::Q8_1 => (numel / 32) * 36,
        DType::Q2K => (numel / 256) * 84,
        DType::Q3K => (numel / 256) * 110,
        DType::Q4K => (numel / 256) * 144,
        DType::Q5K => (numel / 256) * 176,
        DType::Q6K => (numel / 256) * 210,
        DType::Q8K => (numel / 256) * 292,
        _ => numel * 4,  // Assume f32 for unknown
    }
}

/// Weight cache for GPU - stores model weights on GPU
#[allow(dead_code)]
pub struct GpuWeightCache {
    device: Arc<CudaDevice>,
    /// Cached weights by layer name/index
    weights: std::collections::HashMap<String, GpuTensor>,
    /// Total bytes allocated
    total_bytes: usize,
}

#[allow(dead_code)]
impl GpuWeightCache {
    /// Create a new weight cache
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            weights: std::collections::HashMap::new(),
            total_bytes: 0,
        }
    }
    
    /// Upload a weight tensor to GPU
    pub fn upload_f32(&mut self, name: String, data: &[f32], shape: Vec<usize>) -> BackendResult<()> {
        let gpu_tensor = GpuTensor::from_f32(&self.device, data, shape)?;
        self.total_bytes += data.len() * 4;
        self.weights.insert(name, gpu_tensor);
        Ok(())
    }
    
    /// Upload quantized weight to GPU
    pub fn upload_quantized(&mut self, name: String, data: &[u8], shape: Vec<usize>, dtype: DType) -> BackendResult<()> {
        let gpu_tensor = GpuTensor::from_bytes(&self.device, data, shape, dtype)?;
        self.total_bytes += data.len();
        self.weights.insert(name, gpu_tensor);
        Ok(())
    }
    
    /// Get a cached weight
    pub fn get(&self, name: &str) -> Option<&GpuTensor> {
        self.weights.get(name)
    }
    
    /// Get total bytes allocated
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}
