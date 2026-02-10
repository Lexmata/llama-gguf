//! RAG (Retrieval-Augmented Generation) support with pgvector
//!
//! This module provides integration with PostgreSQL + pgvector for
//! semantic search and context retrieval during inference, with an
//! AWS Bedrock Knowledge Bases-style API.
//!
//! # Knowledge Base API (Recommended)
//!
//! ```rust,ignore
//! use llama_gguf::rag::{KnowledgeBase, KnowledgeBaseBuilder, DataSource, ChunkingStrategy};
//!
//! // Create a knowledge base with the builder
//! let kb = KnowledgeBaseBuilder::new("my-kb")
//!     .description("Documentation knowledge base")
//!     .fixed_size_chunking(300, 20)  // 300 tokens, 20% overlap
//!     .max_results(5)
//!     .create()
//!     .await?;
//!
//! // Ingest documents
//! kb.ingest(DataSource::Directory {
//!     path: "./docs".into(),
//!     pattern: Some("**/*.md".into()),
//!     recursive: true,
//! }).await?;
//!
//! // Retrieve and generate
//! let response = kb.retrieve_and_generate("What is the main feature?", None).await?;
//! println!("Answer: {}", response.output);
//!
//! // Print citations
//! for citation in response.citations {
//!     println!("Source: {} (score: {:.2})", citation.source.uri, citation.score);
//! }
//! ```
//!
//! # Low-Level Store API
//!
//! For direct vector store access:
//!
//! ```rust,ignore
//! use llama_gguf::rag::{RagConfig, RagStore, MetadataFilter};
//!
//! let config = RagConfig::load(Some("rag.toml"))?;
//! let store = RagStore::connect(config).await?;
//!
//! // Search with filters
//! let filter = MetadataFilter::eq("type", "documentation");
//! let results = store.search_with_filter(&query_embedding, Some(5), Some(filter)).await?;
//! ```
//!
//! # Setup
//!
//! 1. Install pgvector extension in PostgreSQL:
//!    ```sql
//!    CREATE EXTENSION vector;
//!    ```
//!
//! 2. Configure connection in `rag.toml` or via environment:
//!    ```toml
//!    [database]
//!    connection_string = "postgres://user:pass@localhost/db"
//!    
//!    [embeddings]
//!    dimension = 384
//!    ```

#[cfg(feature = "rag")]
mod config;
#[cfg(feature = "rag")]
mod embedding;
#[cfg(feature = "rag")]
mod knowledge_base;
#[cfg(feature = "rag")]
mod store;

#[cfg(feature = "rag")]
pub use config::example_config;
#[cfg(feature = "rag")]
pub use config::*;
#[cfg(feature = "rag")]
pub use embedding::*;
#[cfg(feature = "rag")]
pub use knowledge_base::*;
#[cfg(feature = "rag")]
pub use store::*;

use thiserror::Error;

/// RAG-related errors
#[derive(Error, Debug)]
pub enum RagError {
    #[error("Database connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Query failed: {0}")]
    QueryFailed(String),

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type RagResult<T> = Result<T, RagError>;
