//! llama-rs: A Rust implementation of llama.cpp
//!
//! High-performance LLM inference engine with support for GGUF models.
//!
//! # Features
//!
//! - Full GGUF file format support (v1, v2, v3)
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
pub mod engine;
pub mod gguf;
#[cfg(feature = "huggingface")]
pub mod huggingface;
pub mod model;
pub mod rag;
pub mod sampling;
#[cfg(feature = "server")]
pub mod server;
pub mod tensor;
pub mod tokenizer;

// Re-export main types
pub use engine::{ChatEngine, ChatTemplate, Engine, EngineConfig, EngineError};
pub use backend::{default_backend, Backend, BackendError};
pub use gguf::{GgufBuilder, GgufData, GgufFile, GgufReader, GgufWriter, TensorToWrite};
pub use model::{
    Architecture, InferenceContext, KVCache, LlamaModel, Model, ModelConfig, ModelError,
    ModelLoader, load_llama_model,
    // LoRA
    LoraAdapter, LoraAdapters, LoraConfig,
    // MoE
    MoeConfig, MoeExpert, MoeLayer, MoeRouter, MoeStats,
    // Speculative decoding
    SpeculativeConfig, SpeculativeDecoder, SpeculativeStats,
    // Embeddings
    EmbeddingConfig, EmbeddingError, EmbeddingExtractor, PoolingStrategy, TruncationStrategy,
    cosine_similarity, dot_product, euclidean_distance, find_nearest,
    // Prompt cache
    CachedPrefix, PrefixId, PrefixSharing, PromptCache, PromptCacheConfig, PromptCacheStats,
};
pub use sampling::{
    Grammar, GrammarSampler, GbnfGrammar, JsonGrammar, RegexGrammar,
    MirostatConfig, Sampler, SamplerConfig,
};
pub use tensor::{DType, Tensor, TensorError, TensorStorage};
pub use tokenizer::{Tokenizer, TokenizerError};
#[cfg(feature = "huggingface")]
pub use huggingface::{HfClient, HfError, HfFileInfo, format_bytes};
#[cfg(feature = "rag")]
pub use rag::{RagConfig, RagStore, RagError, RagResult, Document, NewDocument, RagContextBuilder, TextChunker};

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
