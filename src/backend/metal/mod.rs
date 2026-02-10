//! Metal GPU backend for tensor operations (macOS / Apple Silicon)
//!
//! This module provides a Metal-based GPU implementation of the Backend trait.
//! Metal is Apple's native GPU API, providing the best performance on macOS
//! and Apple Silicon hardware with unified memory architecture.
//!
//! # Features
//! - Native macOS GPU compute via Metal
//! - Metal Shading Language kernels compiled at build time
//! - Unified memory on Apple Silicon (zero-copy CPU<->GPU)
//! - Full Backend trait implementation with CPU fallback for complex operations
//!
//! # Requirements
//! - macOS 10.15+ with Metal-capable GPU
//! - Xcode Command Line Tools (for shader compilation at build time)
//! - Build with `--features metal`

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod context;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod ops;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// Metal backend configuration
#[derive(Debug, Clone, Default)]
pub struct MetalConfig {
    /// Device index to use (0 = system default GPU)
    pub device_index: usize,
}

/// Metal GPU backend
///
/// Implements the Backend trait using Metal compute shaders for GPU-accelerated
/// tensor operations. Falls back to the CPU backend for operations not yet
/// implemented on GPU (attention, dequantization).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalBackend {
    ctx: context::MetalContext,
    cpu_fallback: crate::backend::cpu::CpuBackend,
    device_name: String,
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub struct MetalBackend {
    _config: MetalConfig,
}

impl MetalBackend {
    /// Create a new Metal backend with default configuration
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(MetalConfig::default())
    }

    /// Create a Metal backend with custom configuration
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn with_config(config: MetalConfig) -> Result<Self, BackendError> {
        let ctx = context::MetalContext::new(config.device_index)?;
        let device_name = ctx.device_name().to_string();

        Ok(Self {
            ctx,
            cpu_fallback: crate::backend::cpu::CpuBackend::new(),
            device_name,
        })
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn with_config(_config: MetalConfig) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "Metal support not available. Requires macOS with --features metal".to_string(),
        ))
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            &self.device_name
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            "Metal disabled"
        }
    }

    /// Enumerate available Metal devices
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn enumerate_devices() -> Vec<MetalDeviceInfo> {
        use metal::Device;

        let all_devices = Device::all();
        all_devices
            .iter()
            .enumerate()
            .map(|(idx, dev)| MetalDeviceInfo {
                name: dev.name().to_string(),
                index: idx,
                is_low_power: dev.is_low_power(),
                has_unified_memory: dev.has_unified_memory(),
                max_buffer_length: dev.max_buffer_length(),
                recommended_max_working_set_size: dev.recommended_max_working_set_size(),
            })
            .collect()
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn enumerate_devices() -> Vec<MetalDeviceInfo> {
        vec![]
    }
}

/// Information about a Metal-capable device
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name
    pub name: String,
    /// Device index
    pub index: usize,
    /// Whether this is a low-power (integrated) GPU
    pub is_low_power: bool,
    /// Whether the device has unified memory (Apple Silicon)
    pub has_unified_memory: bool,
    /// Maximum buffer size in bytes
    pub max_buffer_length: u64,
    /// Recommended maximum working set size
    pub recommended_max_working_set_size: u64,
}

// =============================================================================
// Backend trait implementation
// =============================================================================

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor> {
        // Allocate on CPU side (tensors are CPU-resident, data is uploaded to GPU per-op)
        // On Apple Silicon with unified memory this is effectively zero-copy
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
        // Fall back to CPU for full matrix multiply
        self.cpu_fallback.matmul(a, b, out)
    }

    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_fallback.matvec(a, b, out)
    }

    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::vec_mat(&self.ctx, a, b, out)
    }

    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for dequantization
        self.cpu_fallback.dequantize(src, out)
    }

    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // Fall back to CPU for quantized ops
        self.cpu_fallback.matvec_q(a, b, out)
    }

    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // For quantized vector-matrix multiply, use GPU for F32 or CPU fallback
        if b.dtype() == DType::F32 {
            return ops::vec_mat(&self.ctx, a, b, out);
        }
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
        // Fall back to CPU for attention
        self.cpu_fallback.attention(q, k, v, out, scale)
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn rms_norm(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _eps: f32,
        _out: &mut Tensor,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
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
        Err(BackendError::NotAvailable("Metal".to_string()))
    }

    fn attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _out: &mut Tensor,
        _scale: f32,
    ) -> BackendResult<()> {
        Err(BackendError::NotAvailable("Metal".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helper: create backend or skip test if Metal unavailable
    // =========================================================================

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn metal_backend() -> Option<MetalBackend> {
        MetalBackend::new().ok()
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
    fn test_metal_config_default() {
        let config = MetalConfig::default();
        assert_eq!(config.device_index, 0);
    }

    #[test]
    fn test_metal_enumerate_devices() {
        let devices = MetalBackend::enumerate_devices();
        println!("Found {} Metal devices", devices.len());
        for dev in &devices {
            println!(
                "  [{}] {} (unified_mem={}, low_power={}, max_buf={:.0} MB)",
                dev.index,
                dev.name,
                dev.has_unified_memory,
                dev.is_low_power,
                dev.max_buffer_length as f64 / 1024.0 / 1024.0,
            );
        }
    }

    // =========================================================================
    // Backend creation / trait
    // =========================================================================

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_backend_creation() {
        match MetalBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "metal");
                assert!(backend.is_available());
                println!("Metal backend: {}", backend.device_name());
            }
            Err(e) => {
                println!("Metal not available: {}", e);
            }
        }
    }

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_alloc() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_copy_to() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_add() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_add_large() {
        let backend = match metal_backend() {
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
    // Scale
    // =========================================================================

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_scale() {
        let backend = match metal_backend() {
            Some(b) => b,
            None => return,
        };

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        backend.scale(&a, 2.5, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[2.5, 5.0, 7.5, 10.0], 1e-5);
    }

    // =========================================================================
    // SiLU / GELU activations
    // =========================================================================

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_silu() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_gelu() {
        let backend = match metal_backend() {
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
    // vec_mat
    // =========================================================================

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_vec_mat() {
        let backend = match metal_backend() {
            Some(b) => b,
            None => return,
        };

        // Identity matrix test
        let a = Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let mut out = Tensor::zeros(vec![2], DType::F32);

        backend.vec_mat(&a, &b, &mut out).unwrap();
        assert_approx_eq(out.as_f32().unwrap(), &[1.0, 2.0], 1e-5);
    }

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_vec_mat_large() {
        let backend = match metal_backend() {
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

        // Compare with CPU backend
        let cpu = crate::backend::cpu::CpuBackend::new();
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);
        cpu.vec_mat(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(gpu_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-2);
    }

    // =========================================================================
    // RoPE
    // =========================================================================

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_rope() {
        let backend = match metal_backend() {
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

    // =========================================================================
    // GPU vs CPU agreement
    // =========================================================================

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_vs_cpu_add() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_vs_cpu_silu() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_ops_non_power_of_two() {
        let backend = match metal_backend() {
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_repeated_ops() {
        let backend = match metal_backend() {
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
