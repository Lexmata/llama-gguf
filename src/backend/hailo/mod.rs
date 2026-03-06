#[cfg(feature = "hailo")]
pub(crate) mod context;
pub mod config;
#[cfg(feature = "hailo")]
pub(crate) mod compiler;
#[cfg(feature = "hailo")]
pub mod gpu_only;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

pub use config::{HailoConfig, HailoQuantization, HefManifest};

#[cfg(feature = "hailo")]
pub use context::HailoContext;

pub struct HailoBackend {
    _config: HailoConfig,
}

impl HailoBackend {
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(HailoConfig::default())
    }

    #[cfg(feature = "hailo")]
    pub fn with_config(config: HailoConfig) -> Result<Self, BackendError> {
        context::check_device_available()?;
        Ok(Self { _config: config })
    }

    #[cfg(not(feature = "hailo"))]
    pub fn with_config(_config: HailoConfig) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "Hailo support not compiled. Build with --features hailo".to_string(),
        ))
    }
}

impl Backend for HailoBackend {
    fn name(&self) -> &str {
        "hailo"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn rms_norm(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _eps: f32,
        _out: &mut Tensor,
    ) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
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
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }

    fn attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _out: &mut Tensor,
        _scale: f32,
    ) -> BackendResult<()> {
        Err(BackendError::Unsupported(
            "Hailo does not support individual tensor ops".into(),
        ))
    }
}
