//! HTTP server with OpenAI-compatible API
//!
//! This module provides an HTTP server that implements the OpenAI API format
//! for chat completions, text completions, and embeddings, plus AWS Bedrock-style RAG APIs.
//!
//! # Features
//!
//! - **Embeddings** (`POST /v1/embeddings`) — OpenAI-compatible embedding extraction
//! - **Function calling** — Tool definitions in chat completions with prompt injection + JSON grammar
//! - **Token usage** — Reported in all responses including streaming
//! - **Request queuing** — FIFO queue with configurable depth; rejects with 429 when full
//! - **Model hot-swap** — `POST /v1/models/load` or `SIGHUP` to reload without restart
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
