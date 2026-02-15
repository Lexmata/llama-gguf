//! Vulkan GPU backend for tensor operations
//!
//! This module provides a Vulkan-based GPU implementation of the Backend trait.
//! Vulkan enables cross-platform GPU compute on Windows, Linux, macOS (via MoltenVK),
//! and Android.
//!
//! # Features
//! - Cross-platform GPU compute via Vulkan 1.2
//! - SPIR-V compute shaders compiled at build time
//! - Memory management via gpu-allocator
//! - Full Backend trait implementation with CPU fallback for complex operations
//!
//! # Requirements
//! - Vulkan 1.2+ capable GPU and driver
//! - Vulkan SDK (for shader compilation at build time)
//! - Build with `--features vulkan`

#[cfg(feature = "vulkan")]
pub(crate) mod context;
#[cfg(feature = "vulkan")]
pub(crate) mod ops;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// Vulkan backend configuration
#[derive(Debug, Clone)]
pub struct VulkanConfig {
    /// Device index to use (0 = first GPU)
    pub device_index: usize,
    /// Enable validation layers (for debugging)
    pub enable_validation: bool,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            enable_validation: cfg!(debug_assertions),
        }
    }
}

/// Vulkan GPU backend
///
/// Implements the Backend trait using Vulkan compute shaders for GPU-accelerated
/// tensor operations. Falls back to the CPU backend for operations not yet
/// implemented on GPU (attention, dequantization).
#[cfg(feature = "vulkan")]
pub struct VulkanBackend {
    ctx: context::VulkanContext,
    cpu_fallback: crate::backend::cpu::CpuBackend,
    device_name: String,
}

#[cfg(not(feature = "vulkan"))]
pub struct VulkanBackend {
    _config: VulkanConfig,
}

impl VulkanBackend {
    /// Create a new Vulkan backend with default configuration
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(VulkanConfig::default())
    }

    /// Create a Vulkan backend with custom configuration
    #[cfg(feature = "vulkan")]
    pub fn with_config(config: VulkanConfig) -> Result<Self, BackendError> {
        let ctx = context::VulkanContext::new(config.device_index, config.enable_validation)?;

        let device_name = unsafe {
            std::ffi::CStr::from_ptr(ctx.device_properties.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };

        Ok(Self {
            ctx,
            cpu_fallback: crate::backend::cpu::CpuBackend::new(),
            device_name,
        })
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn with_config(_config: VulkanConfig) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "Vulkan support not compiled. Build with --features vulkan".to_string(),
        ))
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        #[cfg(feature = "vulkan")]
        {
            &self.device_name
        }
        #[cfg(not(feature = "vulkan"))]
        {
            "Vulkan disabled"
        }
    }

    /// Enumerate available Vulkan devices
    #[cfg(feature = "vulkan")]
    pub fn enumerate_devices() -> Vec<VulkanDeviceInfo> {
        unsafe {
            let Ok(entry) = ash::Entry::load() else {
                return vec![];
            };
            let app_info = ash::vk::ApplicationInfo::default()
                .api_version(ash::vk::make_api_version(0, 1, 2, 0));
            let instance_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);

            let Ok(instance) = entry.create_instance(&instance_info, None) else {
                return vec![];
            };

            let devices = instance.enumerate_physical_devices().unwrap_or_default();

            let mut result = Vec::new();
            for pd in devices {
                let props = instance.get_physical_device_properties(pd);
                let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .to_string();

                let device_type = match props.device_type {
                    ash::vk::PhysicalDeviceType::DISCRETE_GPU => VulkanDeviceType::DiscreteGpu,
                    ash::vk::PhysicalDeviceType::INTEGRATED_GPU => VulkanDeviceType::IntegratedGpu,
                    ash::vk::PhysicalDeviceType::VIRTUAL_GPU => VulkanDeviceType::VirtualGpu,
                    ash::vk::PhysicalDeviceType::CPU => VulkanDeviceType::Cpu,
                    _ => VulkanDeviceType::Other,
                };

                let mem_props = instance.get_physical_device_memory_properties(pd);
                let vram_bytes: u64 = mem_props
                    .memory_heaps
                    .iter()
                    .take(mem_props.memory_heap_count as usize)
                    .filter(|h| h.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|h| h.size)
                    .sum();

                result.push(VulkanDeviceInfo {
                    name,
                    device_type,
                    vram_bytes,
                    api_version: (
                        ash::vk::api_version_major(props.api_version),
                        ash::vk::api_version_minor(props.api_version),
                        ash::vk::api_version_patch(props.api_version),
                    ),
                    driver_version: props.driver_version,
                });
            }

            instance.destroy_instance(None);
            result
        }
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn enumerate_devices() -> Vec<VulkanDeviceInfo> {
        vec![]
    }
}

/// Information about a Vulkan-capable device
#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    /// Device name
    pub name: String,
    /// Device type (discrete, integrated, etc.)
    pub device_type: VulkanDeviceType,
    /// Available VRAM in bytes
    pub vram_bytes: u64,
    /// Vulkan API version supported
    pub api_version: (u32, u32, u32),
    /// Driver version
    pub driver_version: u32,
}

/// Vulkan device type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanDeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

// =============================================================================
// Backend trait implementation
// =============================================================================

#[cfg(feature = "vulkan")]
impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor> {
        // Allocate on CPU side (tensors are CPU-resident, data is uploaded to GPU per-op)
        Ok(Tensor::zeros(shape.to_vec(), dtype))
    }

    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor> {
        Ok(tensor.clone())
    }

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::add(&self.ctx, a, b, out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::mul(&self.ctx, a, b, out)
    }

    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
        ops::scale(&self.ctx, a, scalar, out)
    }

    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::silu(&self.ctx, x, out)
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::gelu(&self.ctx, x, out)
    }

    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for small tensors
        match ops::softmax(&self.ctx, x, out) {
            Err(BackendError::Unsupported(_)) => self.cpu_fallback.softmax(x, out),
            other => other,
        }
    }

    fn rms_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> BackendResult<()> {
        ops::rms_norm(&self.ctx, x, weight, eps, out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for full matrix multiply (complex GPU implementation)
        self.cpu_fallback.matmul(a, b, out)
    }

    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_fallback.matvec(a, b, out)
    }

    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::vec_mat(&self.ctx, a, b, out)
    }

    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for dequantization (requires quantized format support in shaders)
        self.cpu_fallback.dequantize(src, out)
    }

    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for quantized ops
        self.cpu_fallback.matvec_q(a, b, out)
    }

    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // For quantized vector-matrix multiply, dequantize on CPU then use GPU vec_mat
        // This is a hybrid approach until we have dequant shaders
        if b.dtype() == DType::F32 {
            return ops::vec_mat(&self.ctx, a, b, out);
        }

        // CPU fallback for quantized weights
        self.cpu_fallback.vec_mat_q(a, b, out)
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
        match ops::rope(&self.ctx, q, k, pos, freq_base, freq_scale, use_neox) {
            Err(BackendError::Unsupported(_)) => self
                .cpu_fallback
                .rope(q, k, pos, freq_base, freq_scale, use_neox),
            other => other,
        }
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
    ) -> BackendResult<()> {
        // Fall back to CPU for attention (complex multi-step GPU implementation)
        self.cpu_fallback.attention(q, k, v, out, scale)
    }

    fn attention_cached(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        scale: f32,
        kv_len: usize,
    ) -> BackendResult<()> {
        // Route directly to CPU strided attention — avoids the default trait
        // impl that copies the cache into contiguous tensors first.
        self.cpu_fallback
            .attention_cached(q, k_cache, v_cache, out, scale, kv_len)
    }
}

#[cfg(not(feature = "vulkan"))]
impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn rms_norm(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _eps: f32,
        _out: &mut Tensor,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn rope(
        &self,
        _q: &mut Tensor,
        _k: &mut Tensor,
        _pos: usize,
        _freq_base: f32,
        _freq_scale: f32,
        _use_neox: bool,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }

    fn attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _out: &mut Tensor,
        _scale: f32,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Vulkan".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helper: create backend or skip test if Vulkan unavailable
    // =========================================================================

    #[cfg(feature = "vulkan")]
    fn vk_backend() -> Option<VulkanBackend> {
        VulkanBackend::new().ok()
    }

    /// Assert two f32 slices are approximately equal.
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {} differs: gpu={} expected={} (diff={}, tol={})",
                i,
                x,
                y,
                (x - y).abs(),
                tol,
            );
        }
    }

    /// Reference SiLU: x * sigmoid(x)
    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Reference GELU (tanh approximation)
    fn ref_gelu(x: f32) -> f32 {
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
    }

    // =========================================================================
    // Config / enumeration
    // =========================================================================

    #[test]
    fn test_vulkan_config_default() {
        let config = VulkanConfig::default();
        assert_eq!(config.device_index, 0);
    }

    #[test]
    fn test_vulkan_config_custom() {
        let config = VulkanConfig {
            device_index: 1,
            enable_validation: false,
        };
        assert_eq!(config.device_index, 1);
        assert!(!config.enable_validation);
    }

    #[test]
    fn test_vulkan_enumerate_devices() {
        let devices = VulkanBackend::enumerate_devices();
        println!("Found {} Vulkan devices", devices.len());
        for (i, dev) in devices.iter().enumerate() {
            println!(
                "  [{}] {} ({:?}, {:.0} MB VRAM, API {}.{}.{})",
                i,
                dev.name,
                dev.device_type,
                dev.vram_bytes as f64 / 1024.0 / 1024.0,
                dev.api_version.0,
                dev.api_version.1,
                dev.api_version.2,
            );
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_enumerate_devices_fields() {
        let devices = VulkanBackend::enumerate_devices();
        if let Some(dev) = devices.first() {
            assert!(!dev.name.is_empty());
            assert!(dev.api_version.0 >= 1);
            assert!(dev.vram_bytes > 0);
        }
    }

    // =========================================================================
    // Backend creation / trait
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_backend_creation() {
        match VulkanBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "vulkan");
                assert!(backend.is_available());
                println!("Vulkan backend: {}", backend.device_name());
            }
            Err(e) => {
                println!("Vulkan not available: {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_invalid_device_index() {
        let result = VulkanBackend::with_config(VulkanConfig {
            device_index: 999,
            enable_validation: false,
        });
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_alloc() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let tensor = backend.alloc(&[2, 3], DType::F32).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);

        // Should be zero-initialized
        let data = tensor.as_f32().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_copy_to() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let src = Tensor::from_f32(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let copy = backend.copy_to(&src).unwrap();

        assert_eq!(copy.shape(), src.shape());
        let src_data = src.as_f32().unwrap();
        let copy_data = copy.as_f32().unwrap();
        assert_eq!(src_data, copy_data);
    }

    // =========================================================================
    // Element-wise: add
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_add() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[0.5, 0.5, 0.5, 0.5], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.add(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[1.5, 2.5, 3.5, 4.5], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_add_negatives() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[-1.0, -2.0, 3.0, 0.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[1.0, -3.0, -3.0, 0.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.add(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[0.0, -5.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_add_large() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 4096;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let expected: Vec<f32> = vec![n as f32; n];

        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.add(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &expected, 1e-3);
    }

    // =========================================================================
    // Element-wise: mul
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_mul() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.mul(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[2.0, 6.0, 12.0, 20.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_mul_zeros() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[0.0, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.mul(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[0.0, 0.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_mul_negatives() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[-1.0, 2.0, -3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[2.0, -3.0, -4.0, 5.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.mul(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[-2.0, -6.0, 12.0, 20.0], 1e-5);
    }

    // =========================================================================
    // Scale
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_scale() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.scale(&a, 2.5, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[2.5, 5.0, 7.5, 10.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_scale_zero() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.scale(&a, 0.0, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[0.0, 0.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_scale_negative() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, -2.0, 3.0, -4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.scale(&a, -0.5, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[-0.5, 1.0, -1.5, 2.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_scale_large() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 2048;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected: Vec<f32> = a_data.iter().map(|x| x * 3.14).collect();

        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.scale(&a, 3.14, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &expected, 1e-3);
    }

    // =========================================================================
    // SiLU activation
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_silu() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.silu(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        assert!(result[0].abs() < 1e-5); // SiLU(0) = 0
        assert!((result[1] - 0.731).abs() < 0.01); // SiLU(1) ≈ 0.731
        assert!((result[2] - (-0.269)).abs() < 0.01); // SiLU(-1) ≈ -0.269
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_silu_large() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 1024;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let expected: Vec<f32> = x_data.iter().map(|&v| ref_silu(v)).collect();

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.silu(&x, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &expected, 1e-4);
    }

    // =========================================================================
    // GELU activation
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_gelu() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.gelu(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        let expected: Vec<f32> = [0.0, 1.0, -1.0, 2.0].iter().map(|&v| ref_gelu(v)).collect();
        assert_approx_eq(result, &expected, 1e-4);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_gelu_symmetry() {
        // GELU is NOT symmetric: gelu(-x) != -gelu(x), but gelu(0) = 0
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[0.0], vec![1]).unwrap();
        let mut out = Tensor::zeros(vec![1], DType::F32);

        backend.gelu(&x, &mut out).unwrap();
        assert!(out.as_f32().unwrap()[0].abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_gelu_large() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 1024;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let expected: Vec<f32> = x_data.iter().map(|&v| ref_gelu(v)).collect();

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.gelu(&x, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &expected, 1e-4);
    }

    // =========================================================================
    // Softmax
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_softmax_gpu_path() {
        // Large enough to use the GPU path (>= 1024)
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 2048;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.softmax(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();

        // Sum should be ~1.0
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {} (expected 1.0)",
            sum
        );

        // All values should be positive
        assert!(result.iter().all(|&v| v >= 0.0));

        // Should be monotonically increasing (inputs are increasing)
        for i in 1..n {
            assert!(
                result[i] >= result[i - 1],
                "not monotonic at {}: {} < {}",
                i,
                result[i],
                result[i - 1]
            );
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_softmax_uniform() {
        // Uniform inputs => all outputs equal
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 2048;
        let x_data = vec![1.0f32; n];

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.softmax(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        let expected_val = 1.0 / n as f32;
        for &v in result {
            assert!(
                (v - expected_val).abs() < 1e-5,
                "expected {} got {}",
                expected_val,
                v
            );
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_softmax_cpu_fallback() {
        // Small tensor should fallback to CPU (< 1024)
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        // Should succeed via CPU fallback
        backend.softmax(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    // =========================================================================
    // RMS Norm
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rms_norm() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let weight = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.rms_norm(&x, &weight, 1e-5, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.7386
        // normalized: [0.3651, 0.7303, 1.0954, 1.4606]
        assert!((result[0] - 0.365).abs() < 0.01);
        assert!((result[1] - 0.730).abs() < 0.01);
        assert!((result[2] - 1.095).abs() < 0.01);
        assert!((result[3] - 1.461).abs() < 0.01);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rms_norm_with_weights() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let weight = Tensor::from_f32(&[2.0, 0.5, 1.0, 0.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.rms_norm(&x, &weight, 1e-5, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        // rms = sqrt(30/4) ≈ 2.7386
        // normalized x[i] = x[i] / rms * weight[i]
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[0] - 1.0 / rms * 2.0).abs() < 0.01);
        assert!((result[1] - 2.0 / rms * 0.5).abs() < 0.01);
        assert!((result[3]).abs() < 1e-5); // weight=0 => output=0
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rms_norm_large() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 4096;
        let x_data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let w_data = vec![1.0f32; n];

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let weight = Tensor::from_f32(&w_data, vec![n]).unwrap();
        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.rms_norm(&x, &weight, 1e-5, &mut gpu_out).unwrap();

        // Compute reference on CPU
        let cpu = crate::backend::cpu::CpuBackend::new();
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);
        cpu.rms_norm(&x, &weight, 1e-5, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-3);
    }

    // =========================================================================
    // vec_mat
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vec_mat() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        // a = [1, 2], b = [[1, 0], [0, 1]] (identity), out should be [1, 2]
        let a = Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let mut out = Tensor::zeros(vec![2], DType::F32);

        backend.vec_mat(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[1.0, 2.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vec_mat_non_square() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        // a = [1, 2, 3], shape [3]
        // W = [[1, 0, 0, 1],   shape [3, 4] (k=3, n=4)
        //      [0, 1, 0, 0],
        //      [0, 0, 1, 1]]
        // out[j] = sum_i a[i] * W[i,j]
        // out[0] = 1*1 + 2*0 + 3*0 = 1
        // out[1] = 1*0 + 2*1 + 3*0 = 2
        // out[2] = 1*0 + 2*0 + 3*1 = 3
        // out[3] = 1*1 + 2*0 + 3*1 = 4
        //
        // GGUF column-major storage: W[i,j] at index i + j*k
        // col0: W[0,0]=1, W[1,0]=0, W[2,0]=0 → [1, 0, 0]
        // col1: W[0,1]=0, W[1,1]=1, W[2,1]=0 → [0, 1, 0]
        // col2: W[0,2]=0, W[1,2]=0, W[2,2]=1 → [0, 0, 1]
        // col3: W[0,3]=1, W[1,3]=0, W[2,3]=1 → [1, 0, 1]
        // flat: [1,0,0, 0,1,0, 0,0,1, 1,0,1]
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![3, 4],
        )
        .unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.vec_mat(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vec_mat_large() {
        // GPU and CPU both use GGUF column-major: W[i,j] at b[i + j*k]
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let k = 128;
        let n = 512;
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let a = Tensor::from_f32(&a_data, vec![k]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![k, n]).unwrap();
        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.vec_mat(&a, &b, &mut gpu_out).unwrap();

        // Compare with CPU backend (both use column-major now)
        let cpu = crate::backend::cpu::CpuBackend::new();
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);
        cpu.vec_mat(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-2);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vec_mat_large_consistency() {
        // Run vec_mat twice with same input, verify identical output
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let k = 64;
        let n = 256;
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.01).cos()).collect();

        let a = Tensor::from_f32(&a_data, vec![k]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![k, n]).unwrap();

        let mut out1 = Tensor::zeros(vec![n], DType::F32);
        let mut out2 = Tensor::zeros(vec![n], DType::F32);

        backend.vec_mat(&a, &b, &mut out1).unwrap();
        backend.vec_mat(&a, &b, &mut out2).unwrap();

        assert_approx_eq(out1.as_f32().unwrap(), out2.as_f32().unwrap(), 1e-6);
    }

    // =========================================================================
    // RoPE
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rope() {
        // Compare GPU vs CPU RoPE. CPU requires 3D [num_heads, seq_len, head_dim].
        // Vulkan accepts 2D or 3D with seq_len=1.
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let head_dim = 8;
        let num_heads = 2;
        let seq_len = 1;
        let n = num_heads * seq_len * head_dim;

        let q_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.2).collect();

        let mut q_gpu = Tensor::from_f32(&q_data, vec![num_heads, seq_len, head_dim]).unwrap();
        let mut k_gpu = Tensor::from_f32(&k_data, vec![num_heads, seq_len, head_dim]).unwrap();

        let mut q_cpu = Tensor::from_f32(&q_data, vec![num_heads, seq_len, head_dim]).unwrap();
        let mut k_cpu = Tensor::from_f32(&k_data, vec![num_heads, seq_len, head_dim]).unwrap();

        let pos = 5;
        let freq_base = 10000.0;
        let freq_scale = 1.0;

        backend
            .rope(&mut q_gpu, &mut k_gpu, pos, freq_base, freq_scale, false)
            .unwrap();

        let cpu = crate::backend::cpu::CpuBackend::new();
        cpu.rope(&mut q_cpu, &mut k_cpu, pos, freq_base, freq_scale, false)
            .unwrap();

        assert_approx_eq(q_gpu.as_f32().unwrap(), q_cpu.as_f32().unwrap(), 1e-3);
        assert_approx_eq(k_gpu.as_f32().unwrap(), k_cpu.as_f32().unwrap(), 1e-3);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rope_pos_zero() {
        // At position 0 with freq_scale=1.0, RoPE should have minimal rotation
        // (angle = 0 for all frequencies => cos(0)=1, sin(0)=0 => identity)
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let head_dim = 8;
        let num_heads = 1;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();

        let mut q = Tensor::from_f32(&data, vec![num_heads, 1, head_dim]).unwrap();
        let mut k = Tensor::from_f32(&data, vec![num_heads, 1, head_dim]).unwrap();

        backend
            .rope(&mut q, &mut k, 0, 10000.0, 1.0, false)
            .unwrap();

        // At pos=0, cos(0)=1, sin(0)=0, so the output should match the input
        assert_approx_eq(q.as_f32().unwrap(), &data, 1e-4);
        assert_approx_eq(k.as_f32().unwrap(), &data, 1e-4);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rope_preserves_magnitude() {
        // RoPE is a rotation, so it should preserve vector magnitude
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let head_dim = 16;
        let num_heads = 4;
        let n = num_heads * head_dim;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut q = Tensor::from_f32(&data, vec![num_heads, 1, head_dim]).unwrap();
        let mut k = Tensor::from_f32(&data, vec![num_heads, 1, head_dim]).unwrap();

        // Compute magnitudes before
        let q_mag_before: Vec<f32> = (0..num_heads)
            .map(|h| {
                let start = h * head_dim;
                let end = start + head_dim;
                data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();

        backend
            .rope(&mut q, &mut k, 7, 10000.0, 1.0, false)
            .unwrap();

        // Compute magnitudes after
        let q_data = q.as_f32().unwrap();
        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let mag_after: f32 = q_data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (mag_after - q_mag_before[h]).abs() < 0.1,
                "head {} magnitude changed: {} -> {}",
                h,
                q_mag_before[h],
                mag_after,
            );
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_rope_multiple_positions() {
        // Different positions should produce different rotations
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let head_dim = 8;
        let data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let mut q0 = Tensor::from_f32(&data, vec![1, 1, head_dim]).unwrap();
        let mut k0 = Tensor::from_f32(&data, vec![1, 1, head_dim]).unwrap();
        let mut q1 = Tensor::from_f32(&data, vec![1, 1, head_dim]).unwrap();
        let mut k1 = Tensor::from_f32(&data, vec![1, 1, head_dim]).unwrap();

        backend
            .rope(&mut q0, &mut k0, 0, 10000.0, 1.0, false)
            .unwrap();
        backend
            .rope(&mut q1, &mut k1, 10, 10000.0, 1.0, false)
            .unwrap();

        // Position 0 and 10 should give different results
        let q0_data = q0.as_f32().unwrap();
        let q1_data = q1.as_f32().unwrap();
        let diff: f32 = q0_data
            .iter()
            .zip(q1_data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "positions 0 and 10 should produce different rotations"
        );
    }

    // =========================================================================
    // CPU fallback operations (matmul, matvec, attention, dequantize)
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_matmul_cpu_fallback() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        // 2x3 @ 3x2 = 2x2
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let mut out = Tensor::zeros(vec![2, 2], DType::F32);

        backend.matmul(&a, &b, &mut out).unwrap();

        // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        assert_approx_eq(out.as_f32().unwrap(), &[22.0, 28.0, 49.0, 64.0], 1e-4);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_matvec_cpu_fallback() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4],
        )
        .unwrap();
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![3], DType::F32);

        backend.matvec(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[30.0, 70.0, 110.0], 1e-4);
    }

    // =========================================================================
    // GPU vs CPU agreement: run same operation on both backends, compare results
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vs_cpu_add() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        let n = 1000;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.71).cos()).collect();

        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.add(&a, &b, &mut gpu_out).unwrap();
        cpu.add(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vs_cpu_mul() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        let n = 1000;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.71).cos()).collect();

        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.mul(&a, &b, &mut gpu_out).unwrap();
        cpu.mul(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vs_cpu_silu() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        let n = 1000;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.01).collect();
        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();

        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.silu(&x, &mut gpu_out).unwrap();
        cpu.silu(&x, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vs_cpu_gelu() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        let n = 1000;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.01).collect();
        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();

        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.gelu(&x, &mut gpu_out).unwrap();
        cpu.gelu(&x, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_vs_cpu_scale() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        let n = 1000;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();

        let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.scale(&a, 2.718, &mut gpu_out).unwrap();
        cpu.scale(&a, 2.718, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_add_single_element() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[42.0], vec![1]).unwrap();
        let b = Tensor::from_f32(&[58.0], vec![1]).unwrap();
        let mut out = Tensor::zeros(vec![1], DType::F32);

        backend.add(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[100.0], 1e-5);
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_ops_non_power_of_two() {
        // Test with sizes that don't align to workgroup size (256)
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        for n in [1, 3, 7, 17, 127, 255, 257, 511, 513, 1023, 1025] {
            let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();

            let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
            let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

            let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
            let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

            backend.add(&a, &b, &mut gpu_out).unwrap();
            cpu.add(&a, &b, &mut cpu_out).unwrap();

            assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_mul_non_power_of_two() {
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };
        let cpu = crate::backend::cpu::CpuBackend::new();

        for n in [1, 3, 17, 255, 257, 1023] {
            let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();

            let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
            let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

            let mut gpu_out = Tensor::zeros(vec![n], DType::F32);
            let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

            backend.mul(&a, &b, &mut gpu_out).unwrap();
            cpu.mul(&a, &b, &mut cpu_out).unwrap();

            assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
        }
    }

    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_repeated_ops() {
        // Ensure repeated operations on the same backend don't leak or corrupt state
        let backend = match vk_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        for i in 0..50 {
            backend.add(&a, &b, &mut out).unwrap();
            let result = out.as_f32().unwrap();
            assert_approx_eq(result, &[2.0, 3.0, 4.0, 5.0], 1e-5);

            backend.mul(&a, &b, &mut out).unwrap();
            let result = out.as_f32().unwrap();
            assert_approx_eq(result, &[1.0, 2.0, 3.0, 4.0], 1e-5);

            backend.scale(&a, 0.5, &mut out).unwrap();
            let result = out.as_f32().unwrap();
            assert_approx_eq(result, &[0.5, 1.0, 1.5, 2.0], 1e-5);

            if i % 10 == 0 {
                println!("iteration {} ok", i);
            }
        }
    }
}
