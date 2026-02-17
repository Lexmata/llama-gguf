//! Pipeline executor for distributed inference
//!
//! Chains forward passes across ordered shard nodes, sending hidden states
//! through the pipeline from first shard to last.

use crate::tensor::Tensor;

use super::proto::shard_service_client::ShardServiceClient;
use super::proto::{ForwardRequest, ResetRequest};
use super::tensor_transfer::{tensor_from_proto, tensor_to_proto};
use super::{DistributedError, DistributedResult};

/// A connected shard client with metadata.
pub struct ShardConnection {
    /// gRPC client for this shard
    pub client: ShardServiceClient<tonic::transport::Channel>,
    /// Human-readable shard name
    pub name: String,
    /// Layer range this shard handles
    pub layer_start: usize,
    pub layer_end: usize,
}

/// Executes the distributed forward pass pipeline by chaining
/// hidden state transfers through an ordered sequence of shards.
pub struct PipelineExecutor {
    /// Ordered shard connections (first shard processes earliest layers)
    shards: Vec<ShardConnection>,
}

impl PipelineExecutor {
    /// Create a new pipeline executor from a list of connected shards.
    pub fn new(shards: Vec<ShardConnection>) -> DistributedResult<Self> {
        if shards.is_empty() {
            return Err(DistributedError::NoShards);
        }
        Ok(Self { shards })
    }

    /// Number of shards in the pipeline.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Run the forward pass through all shards in sequence.
    ///
    /// Takes the hidden state from the embedding layer, sends it through
    /// each shard's layers, and returns the final hidden state for
    /// norm + logits computation.
    pub async fn forward(
        &mut self,
        hidden: &Tensor,
        position: usize,
    ) -> DistributedResult<Tensor> {
        let mut current_hidden = hidden.clone();

        for shard in &mut self.shards {
            let hidden_proto = tensor_to_proto(&current_hidden);

            let request = ForwardRequest {
                hidden_state: Some(hidden_proto),
                position: position as u32,
                seq_len: (position + 1) as u32,
            };

            let response = shard
                .client
                .forward(request)
                .await
                .map_err(|e| {
                    DistributedError::Shard(format!(
                        "forward failed on shard '{}' (layers {}..{}): {}",
                        shard.name, shard.layer_start, shard.layer_end, e
                    ))
                })?
                .into_inner();

            if !response.success {
                return Err(DistributedError::Shard(format!(
                    "shard '{}' returned error: {}",
                    shard.name, response.error
                )));
            }

            let output_proto = response.hidden_state.ok_or_else(|| {
                DistributedError::Shard(format!(
                    "shard '{}' returned empty hidden state",
                    shard.name
                ))
            })?;

            current_hidden = tensor_from_proto(&output_proto)?;
        }

        Ok(current_hidden)
    }

    /// Reset KV caches on all shards (for new sequence).
    pub async fn reset_kv_caches(&mut self) -> DistributedResult<()> {
        for shard in &mut self.shards {
            shard
                .client
                .reset_kv_cache(ResetRequest {})
                .await
                .map_err(|e| {
                    DistributedError::Shard(format!(
                        "failed to reset KV cache on shard '{}': {}",
                        shard.name, e
                    ))
                })?;
        }
        Ok(())
    }
}
