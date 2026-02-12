//! Tensor module for llama-rs
//!
//! This module provides tensor types for representing multi-dimensional arrays
//! with support for various data types including quantized formats.

mod core;
mod dtype;
mod error;
pub mod quant;
mod storage;

pub use core::{Tensor, compute_strides};
pub use dtype::DType;
pub use error::TensorError;
pub use storage::TensorStorage;
