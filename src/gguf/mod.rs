//! GGUF file format parser

mod constants;
mod error;
mod types;

pub use constants::{GgmlType, GgufMetadataValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};
pub use error::GgufError;
pub use types::{GgufData, GgufHeader, MetadataArray, MetadataValue, TensorInfo};

/// Placeholder for GGUF file handle
pub struct GgufFile;
