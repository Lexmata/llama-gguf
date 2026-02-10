//! ONNX model format support
//!
//! This module provides functionality to load model weights from ONNX files
//! exported by HuggingFace Optimum or similar tools. Weights are extracted
//! from ONNX initializers and loaded into the existing model infrastructure.
//!
//! # Supported formats
//!
//! - HuggingFace Optimum exports (model.onnx + config.json + tokenizer.json)
//! - Generic ONNX models with companion config.json
//!
//! # Example
//!
//! ```no_run
//! use llama_gguf::onnx::OnnxModelLoader;
//!
//! let loader = OnnxModelLoader::load("model_dir/model.onnx").unwrap();
//! let model = loader.build_model().unwrap();
//! ```

pub mod config;
pub mod error;
pub mod loader;
pub(crate) mod proto;
pub mod reader;

pub use config::HfConfig;
pub use error::{OnnxError, OnnxResult};
pub use loader::OnnxModelLoader;
pub use reader::{OnnxFile, OnnxMetadata, OnnxTensorInfo};
