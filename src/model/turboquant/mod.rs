//! TurboQuant KV cache compression algorithm
//!
//! Implements Google's TurboQuant (ICLR 2026) for memory-efficient
//! KV caches in transformer inference. Two variants:
//!
//! - **TurboQuant_MSE** (biased): Randomized Hadamard rotation + Lloyd-Max
//!   scalar quantization at 2-3 bits. Simpler, lower overhead.
//!
//! - **TurboQuant_prod** (unbiased): Same as MSE plus a QJL 1-bit
//!   correction on the quantization residual. Higher accuracy for
//!   attention computation at the cost of ~1 extra bit per dimension.

pub mod codebook;
pub mod qjl;
pub mod quant;
pub mod rotation;

pub use codebook::Codebook;
pub use qjl::QjlProjector;
pub use quant::{CompressedVector, TurboQuantConfig, TurboQuantEngine};
pub use rotation::HadamardRotation;
