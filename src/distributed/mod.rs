//! Distributed pipeline-parallel inference
//!
//! This module implements pipeline parallelism for LLM inference across
//! multiple machines. A coordinator node splits a model's transformer layers
//! across shard nodes, streaming hidden states between them over gRPC.
//!
//! # Architecture
//!
//! - **Coordinator**: Loads the GGUF model, partitions layers across shards,
//!   holds token embedding and output projection locally, orchestrates the
//!   forward pass pipeline.
//! - **Shard**: Receives assigned layers from the coordinator, runs forward
//!   passes on its local backend (CPU/GPU), manages its own KV cache.
//!
//! # Usage
//!
//! Start shard servers on worker nodes:
//! ```bash
//! llama-gguf shard --port 50051 --gpu
//! ```
//!
//! Run distributed inference from the coordinator:
//! ```bash
//! llama-gguf run model.gguf --distributed cluster.toml -p "Hello"
//! ```

pub mod config;
pub mod coordinator;
pub mod model;
pub mod pipeline;
pub mod shard;
pub mod tensor_transfer;

#[allow(clippy::all)]
pub mod proto {
    tonic::include_proto!("distributed");
}

pub use config::{ClusterConfig, ShardSpec};
pub use coordinator::Coordinator;
pub use model::DistributedModel;
pub use pipeline::PipelineExecutor;
pub use shard::ShardServer;

/// Errors that can occur during distributed inference.
#[derive(thiserror::Error, Debug)]
pub enum DistributedError {
    #[error("gRPC transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("gRPC status error: {0}")]
    Status(#[from] tonic::Status),

    #[error("shard error: {0}")]
    Shard(String),

    #[error("coordinator error: {0}")]
    Coordinator(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("tensor serialization error: {0}")]
    TensorTransfer(String),

    #[error("model error: {0}")]
    Model(#[from] crate::model::ModelError),

    #[error("backend error: {0}")]
    Backend(#[from] crate::backend::BackendError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF error: {0}")]
    Gguf(#[from] crate::gguf::GgufError),

    #[error("pipeline has no shards")]
    NoShards,

    #[error("shard {name} failed health check: {reason}")]
    HealthCheckFailed { name: String, reason: String },

    #[error("layer assignment mismatch: model has {model_layers} layers but shards cover {assigned_layers}")]
    LayerMismatch {
        model_layers: usize,
        assigned_layers: usize,
    },
}

pub type DistributedResult<T> = std::result::Result<T, DistributedError>;
