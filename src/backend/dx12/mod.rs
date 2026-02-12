//! DirectX 12 GPU backend for tensor operations
//!
//! This module provides a DX12-based GPU implementation of the Backend trait.
//! DX12 enables native Windows GPU compute with full hardware access.
//!
//! # Features
//! - Native Windows GPU compute via DirectX 12
//! - HLSL compute shaders compiled at build time (CSO bytecode)
//! - Explicit resource management with root descriptors
//! - Fence-based GPU synchronization
//! - Full Backend trait implementation with CPU fallback for complex operations
//!
//! # Requirements
//! - Windows 10+ with DirectX 12 capable GPU
//! - DirectX Shader Compiler (dxc.exe) for shader compilation at build time
//! - Build with `--features dx12`

#[cfg(all(feature = "dx12", target_os = "windows"))]
pub(crate) mod context;
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub(crate) mod ops;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// DX12 backend configuration
#[derive(Debug, Clone, Default)]
pub struct Dx12Config {
    /// Device index to use (0 = first GPU)
    pub device_index: usize,
    /// Prefer WARP (Windows Advanced Rasterization Platform) software adapter
    pub prefer_warp: bool,
}

/// Information about a DX12-capable device
#[derive(Debug, Clone)]
pub struct Dx12DeviceInfo {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: Dx12DeviceType,
    /// Dedicated video memory in bytes
    pub dedicated_video_memory: u64,
    /// Shared system memory in bytes
    pub shared_system_memory: u64,
    /// Feature level supported (e.g., "12.0", "12.1")
    pub feature_level: String,
}

/// DX12 device type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dx12DeviceType {
    Discrete,
    Integrated,
    Warp,
    Other,
}

/// DirectX 12 GPU backend
///
/// Implements the Backend trait using DX12 compute shaders for GPU-accelerated
/// tensor operations. Falls back to the CPU backend for operations not yet
/// implemented on GPU (attention, dequantization).
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub struct Dx12Backend {
    ctx: context::Dx12Context,
    cpu_fallback: crate::backend::cpu::CpuBackend,
    device_name: String,
}

#[cfg(not(all(feature = "dx12", target_os = "windows")))]
pub struct Dx12Backend {
    _config: Dx12Config,
}

impl Dx12Backend {
    /// Create a new DX12 backend with default configuration
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(Dx12Config::default())
    }

    /// Create a DX12 backend with custom configuration
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    pub fn with_config(config: Dx12Config) -> Result<Self, BackendError> {
        let ctx = context::Dx12Context::new(config.device_index, config.prefer_warp)?;

        // Get the device name via DXGI
        let device_name = Self::get_device_name(config.device_index, config.prefer_warp)
            .unwrap_or_else(|| "DX12 Device".to_string());

        Ok(Self {
            ctx,
            cpu_fallback: crate::backend::cpu::CpuBackend::new(),
            device_name,
        })
    }

    #[cfg(not(all(feature = "dx12", target_os = "windows")))]
    pub fn with_config(_config: Dx12Config) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "DX12 support not compiled. Build on Windows with --features dx12".to_string(),
        ))
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            &self.device_name
        }
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        {
            "DX12 disabled"
        }
    }

    /// Enumerate available DX12 devices
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    pub fn enumerate_devices() -> Vec<Dx12DeviceInfo> {
        unsafe {
            use windows::Win32::Graphics::Dxgi::*;

            let Ok(factory): Result<IDXGIFactory4, _> =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return vec![];
            };

            let mut idx = 0u32;
            let mut result = Vec::new();

            loop {
                match factory.EnumAdapters1(idx) {
                    Ok(adapter) => {
                        if let Ok(desc) = adapter.GetDesc1() {
                            let name = String::from_utf16_lossy(
                                &desc.Description[..desc
                                    .Description
                                    .iter()
                                    .position(|&c| c == 0)
                                    .unwrap_or(desc.Description.len())],
                            );

                            let is_software =
                                (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
                            let device_type = if is_software {
                                Dx12DeviceType::Warp
                            } else if desc.DedicatedVideoMemory > 0 {
                                Dx12DeviceType::Discrete
                            } else {
                                Dx12DeviceType::Integrated
                            };

                            result.push(Dx12DeviceInfo {
                                name: name.trim().to_string(),
                                device_type,
                                dedicated_video_memory: desc.DedicatedVideoMemory as u64,
                                shared_system_memory: desc.SharedSystemMemory as u64,
                                feature_level: "12.0".to_string(),
                            });
                        }
                        idx += 1;
                    }
                    Err(_) => break,
                }
            }

            result
        }
    }

    #[cfg(not(all(feature = "dx12", target_os = "windows")))]
    pub fn enumerate_devices() -> Vec<Dx12DeviceInfo> {
        vec![]
    }

    /// Get device name from DXGI adapter.
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn get_device_name(device_index: usize, prefer_warp: bool) -> Option<String> {
        unsafe {
            use windows::Win32::Graphics::Dxgi::*;

            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;

            if prefer_warp {
                let adapter: IDXGIAdapter1 = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.GetDesc1().ok()?;
                let name = String::from_utf16_lossy(
                    &desc.Description[..desc
                        .Description
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(desc.Description.len())],
                );
                return Some(name.trim().to_string());
            }

            let mut idx = 0u32;
            let mut hw_index = 0usize;
            loop {
                match factory.EnumAdapters1(idx) {
                    Ok(adapter) => {
                        let desc = adapter.GetDesc1().ok()?;
                        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0 {
                            if hw_index == device_index {
                                let name = String::from_utf16_lossy(
                                    &desc.Description[..desc
                                        .Description
                                        .iter()
                                        .position(|&c| c == 0)
                                        .unwrap_or(desc.Description.len())],
                                );
                                return Some(name.trim().to_string());
                            }
                            hw_index += 1;
                        }
                        idx += 1;
                    }
                    Err(_) => break,
                }
            }
            None
        }
    }
}

// =============================================================================
// Backend trait implementation
// =============================================================================

#[cfg(all(feature = "dx12", target_os = "windows"))]
impl Backend for Dx12Backend {
    fn name(&self) -> &str {
        "dx12"
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
}

#[cfg(not(all(feature = "dx12", target_os = "windows")))]
impl Backend for Dx12Backend {
    fn name(&self) -> &str {
        "dx12"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn rms_norm(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _eps: f32,
        _out: &mut Tensor,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
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
        Err(BackendError::NotAvailable("DX12".to_string()))
    }

    fn attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _out: &mut Tensor,
        _scale: f32,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("DX12".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helper: create backend or skip test if DX12 unavailable
    // =========================================================================

    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn dx12_backend() -> Option<Dx12Backend> {
        Dx12Backend::new().ok()
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
    #[allow(dead_code)]
    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Reference GELU (tanh approximation)
    #[allow(dead_code)]
    fn ref_gelu(x: f32) -> f32 {
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
    }

    // =========================================================================
    // Config / enumeration
    // =========================================================================

    #[test]
    fn test_dx12_config_default() {
        let config = Dx12Config::default();
        assert_eq!(config.device_index, 0);
        assert!(!config.prefer_warp);
    }

    #[test]
    fn test_dx12_config_custom() {
        let config = Dx12Config {
            device_index: 1,
            prefer_warp: true,
        };
        assert_eq!(config.device_index, 1);
        assert!(config.prefer_warp);
    }

    #[test]
    fn test_dx12_enumerate_devices() {
        let devices = Dx12Backend::enumerate_devices();
        println!("Found {} DX12 devices", devices.len());
        for (i, dev) in devices.iter().enumerate() {
            println!(
                "  [{}] {} ({:?}, {:.0} MB dedicated, {:.0} MB shared, FL {})",
                i,
                dev.name,
                dev.device_type,
                dev.dedicated_video_memory as f64 / 1024.0 / 1024.0,
                dev.shared_system_memory as f64 / 1024.0 / 1024.0,
                dev.feature_level,
            );
        }
    }

    // =========================================================================
    // Backend creation / trait
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_backend_creation() {
        match Dx12Backend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "dx12");
                assert!(backend.is_available());
                println!("DX12 backend: {}", backend.device_name());
            }
            Err(e) => {
                println!("DX12 not available: {}", e);
            }
        }
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_invalid_device_index() {
        let result = Dx12Backend::with_config(Dx12Config {
            device_index: 999,
            prefer_warp: false,
        });
        assert!(result.is_err());
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_alloc() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_copy_to() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_add() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_add_negatives() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_add_large() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_mul() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_mul_zeros() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[0.0, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.mul(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[0.0, 0.0, 0.0, 0.0], 1e-5);
    }

    // =========================================================================
    // Scale
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_scale() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.scale(&a, 2.5, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[2.5, 5.0, 7.5, 10.0], 1e-5);
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_scale_zero() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.scale(&a, 0.0, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[0.0, 0.0, 0.0, 0.0], 1e-5);
    }

    // =========================================================================
    // SiLU activation
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_silu() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.silu(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        assert!(result[0].abs() < 1e-5); // SiLU(0) = 0
        assert!((result[1] - 0.731).abs() < 0.01); // SiLU(1) ~ 0.731
        assert!((result[2] - (-0.269)).abs() < 0.01); // SiLU(-1) ~ -0.269
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_silu_large() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_gelu() {
        let backend = match dx12_backend() {
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

    // =========================================================================
    // Softmax
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_softmax_gpu_path() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let n = 2048;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let mut out = Tensor::zeros(vec![n], DType::F32);

        backend.softmax(&x, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {} (expected 1.0)",
            sum
        );
        assert!(result.iter().all(|&v| v >= 0.0));
    }

    // =========================================================================
    // RMS Norm
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_rms_norm() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let weight = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.rms_norm(&x, &weight, 1e-5, &mut out).unwrap();

        let result = out.as_f32().unwrap();
        assert!((result[0] - 0.365).abs() < 0.01);
        assert!((result[1] - 0.730).abs() < 0.01);
        assert!((result[2] - 1.095).abs() < 0.01);
        assert!((result[3] - 1.461).abs() < 0.01);
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_rms_norm_large() {
        let backend = match dx12_backend() {
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

        let cpu = crate::backend::cpu::CpuBackend::new();
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);
        cpu.rms_norm(&x, &weight, 1e-5, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-3);
    }

    // =========================================================================
    // vec_mat
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_vec_mat() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let mut out = Tensor::zeros(vec![2], DType::F32);

        backend.vec_mat(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[1.0, 2.0], 1e-5);
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_vec_mat_large() {
        let backend = match dx12_backend() {
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

        let cpu = crate::backend::cpu::CpuBackend::new();
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);
        cpu.vec_mat(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-2);
    }

    // =========================================================================
    // RoPE
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_rope() {
        let backend = match dx12_backend() {
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
        backend
            .rope(&mut q_gpu, &mut k_gpu, pos, 10000.0, 1.0, false)
            .unwrap();

        let cpu = crate::backend::cpu::CpuBackend::new();
        cpu.rope(&mut q_cpu, &mut k_cpu, pos, 10000.0, 1.0, false)
            .unwrap();

        assert_approx_eq(q_gpu.as_f32().unwrap(), q_cpu.as_f32().unwrap(), 1e-3);
        assert_approx_eq(k_gpu.as_f32().unwrap(), k_cpu.as_f32().unwrap(), 1e-3);
    }

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_rope_pos_zero() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let head_dim = 8;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();

        let mut q = Tensor::from_f32(&data, vec![1, 1, head_dim]).unwrap();
        let mut k = Tensor::from_f32(&data, vec![1, 1, head_dim]).unwrap();

        backend
            .rope(&mut q, &mut k, 0, 10000.0, 1.0, false)
            .unwrap();

        assert_approx_eq(q.as_f32().unwrap(), &data, 1e-4);
        assert_approx_eq(k.as_f32().unwrap(), &data, 1e-4);
    }

    // =========================================================================
    // GPU vs CPU agreement
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_vs_cpu_add() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_vs_cpu_silu() {
        let backend = match dx12_backend() {
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

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_add_single_element() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_ops_non_power_of_two() {
        let backend = match dx12_backend() {
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
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    fn test_dx12_repeated_ops() {
        let backend = match dx12_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        for i in 0..50 {
            backend.add(&a, &b, &mut out).unwrap();
            assert_approx_eq(out.as_f32().unwrap(), &[2.0, 3.0, 4.0, 5.0], 1e-5);

            backend.mul(&a, &b, &mut out).unwrap();
            assert_approx_eq(out.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5);

            backend.scale(&a, 0.5, &mut out).unwrap();
            assert_approx_eq(out.as_f32().unwrap(), &[0.5, 1.0, 1.5, 2.0], 1e-5);

            if i % 10 == 0 {
                println!("iteration {} ok", i);
            }
        }
    }
}
