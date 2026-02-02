//! llama-rs: A Rust implementation of llama.cpp

pub mod gguf;
pub mod tensor;
pub mod backend;

pub use gguf::GgufFile;
pub use tensor::{Tensor, DType};
pub use backend::Backend;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("GGUF error: {0}")]
    Gguf(#[from] gguf::GgufError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] tensor::TensorError),
    #[error("Backend error: {0}")]
    Backend(#[from] backend::BackendError),
}

pub type Result<T> = std::result::Result<T, Error>;
