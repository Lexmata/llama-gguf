//! llama-rs: A Rust implementation of llama.cpp
//!
//! High-performance LLM inference engine with support for GGUF and ONNX models.
//!
//! # Features
//!
//! - Full GGUF file format support (v1, v2, v3)
//! - ONNX model loading (with `onnx` feature) for HuggingFace Optimum exports
//! - All quantization formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, K-quants)
//! - Memory-mapped model loading
//! - CPU backend with SIMD and parallel operations
//! - LLaMA model architecture support
//!
//! # Example
//!
//! ```no_run
//! use llama_gguf::{GgufFile, default_backend};
//!
//! // Load a GGUF model
//! let file = GgufFile::open("model.gguf").unwrap();
//! println!("Model architecture: {:?}", file.data.get_string("general.architecture"));
//!
//! // Get the default backend
//! let backend = default_backend();
//! println!("Using backend: {}", backend.name());
//! ```

pub mod backend;
#[cfg(feature = "client")]
pub mod client;
pub mod config;
pub mod engine;
pub mod gguf;
#[cfg(feature = "huggingface")]
pub mod huggingface;
pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod rag;
pub mod sampling;
#[cfg(feature = "server")]
pub mod server;
pub mod tensor;
pub mod tokenizer;

// Re-export main types
pub use backend::{Backend, BackendError, default_backend};
pub use config::{Config, ConfigError};
pub use engine::{ChatEngine, ChatTemplate, Engine, EngineConfig, EngineError};
pub use gguf::{GgufBuilder, GgufData, GgufFile, GgufReader, GgufWriter, TensorToWrite};
#[cfg(feature = "huggingface")]
pub use huggingface::{HfClient, HfError, HfFileInfo, format_bytes};
pub use model::{
    Architecture,
    // Prompt cache
    CachedPrefix,
    // Embeddings
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingExtractor,
    InferenceContext,
    KVCache,
    LlamaModel,
    // LoRA
    LoraAdapter,
    LoraAdapters,
    LoraConfig,
    Model,
    ModelConfig,
    ModelError,
    ModelLoader,
    // MoE
    MoeConfig,
    MoeExpert,
    MoeLayer,
    MoeRouter,
    MoeStats,
    PoolingStrategy,
    PrefixId,
    PrefixSharing,
    PromptCache,
    PromptCacheConfig,
    PromptCacheStats,
    // Speculative decoding
    SpeculativeConfig,
    SpeculativeDecoder,
    SpeculativeStats,
    TruncationStrategy,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    find_nearest,
    load_llama_model,
};
#[cfg(feature = "onnx")]
pub use onnx::{HfConfig, OnnxError, OnnxFile, OnnxMetadata, OnnxModelLoader, OnnxTensorInfo};
#[cfg(feature = "rag")]
pub use rag::{
    Document, NewDocument, RagConfig, RagContextBuilder, RagError, RagResult, RagStore, TextChunker,
};
pub use sampling::{
    GbnfGrammar, Grammar, GrammarSampler, JsonGrammar, MirostatConfig, RegexGrammar, Sampler,
    SamplerConfig,
};
pub use tensor::{DType, Tensor, TensorError, TensorStorage};
pub use tokenizer::{Tokenizer, TokenizerError};

/// Library-wide error type
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
