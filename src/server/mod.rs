//! HTTP server with OpenAI-compatible API
//!
//! This module provides an HTTP server that implements the OpenAI API format
//! for chat completions and text completions, plus AWS Bedrock-style RAG APIs.
//!
//! # RAG Endpoints
//!
//! When the server is started with `--rag-database-url`, the following
//! Bedrock-style Knowledge Base endpoints are available:
//!
//! - `POST /v1/rag/retrieve` - Retrieve relevant chunks from a knowledge base
//! - `POST /v1/rag/retrieveAndGenerate` - Full RAG pipeline
//! - `POST /v1/rag/ingest` - Ingest documents into a knowledge base
//! - `POST /v1/rag/knowledgebases` - List knowledge bases
//! - `GET /v1/rag/knowledgebases/:id` - Get knowledge base details
//! - `DELETE /v1/rag/knowledgebases/:id` - Delete a knowledge base

mod api;
pub mod batch;
mod handlers;
mod types;

pub use api::{ServerConfig, run_server};
pub use batch::{
    BatchConfig, BatchScheduler, FinishReason, GenerationEvent, GenerationRequest, RequestId,
    SharedBatchScheduler, new_batch_scheduler,
};
pub use types::*;

#[cfg(feature = "rag")]
pub use handlers::RagState;
