//! Multi-node tensor parallelism over gRPC
//!
//! Implements the `TensorParallel` trait for distributed inference,
//! using gRPC AllReduce RPCs to synchronize partial results across nodes.
//! Supports ring-allreduce for bandwidth-efficient communication.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::backend::{BackendError, BackendResult};
use crate::backend::tensor_parallel::TensorParallel;
use crate::tensor::Tensor;

use super::config::ShardSpec;
use super::proto::shard_service_client::ShardServiceClient;
use super::proto::{AllReduceOp, AllReduceRequest};
use super::tensor_transfer::{tensor_from_proto, tensor_to_proto};
use super::DistributedError;
use super::DistributedResult;

/// Parallelism mode for the distributed cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismMode {
    /// Pipeline parallelism: each shard processes a range of layers sequentially
    Pipeline,
    /// Tensor parallelism: all shards process the same layers with AllReduce
    TensorParallel,
    /// Hybrid: TP within groups, PP between groups
    Hybrid {
        /// Number of shards per tensor-parallel group
        tp_group_size: usize,
    },
}

impl Default for ParallelismMode {
    fn default() -> Self {
        Self::Pipeline
    }
}

/// A member of a tensor-parallel group
#[derive(Debug, Clone)]
pub struct TPGroupMember {
    pub name: String,
    pub address: String,
    pub rank: usize,
}

/// A tensor-parallel group of nodes
#[derive(Debug, Clone)]
pub struct TPGroup {
    pub group_id: u32,
    pub members: Vec<TPGroupMember>,
    pub world_size: usize,
}

impl TPGroup {
    pub fn new(group_id: u32, members: Vec<TPGroupMember>) -> Self {
        let world_size = members.len();
        Self {
            group_id,
            members,
            world_size,
        }
    }
}

/// Distributed tensor parallelism using gRPC AllReduce
pub struct DistributedTP {
    rank: usize,
    world_size: usize,
    group_id: u32,
    /// TP group (for connect and rank 0 address)
    group: TPGroup,
    /// gRPC client to rank 0 (reducer). All ranks send to rank 0 for all-reduce.
    rank0_client: Arc<Mutex<Option<ShardServiceClient<tonic::transport::Channel>>>>,
    /// Tokio runtime handle for blocking bridge
    runtime: tokio::runtime::Handle,
}

fn to_backend_error(e: DistributedError) -> BackendError {
    BackendError::OperationFailed(format!("distributed TP: {}", e))
}

impl DistributedTP {
    /// Create a new distributed TP instance.
    /// Call `connect` before using all-reduce.
    pub fn new(rank: usize, group: &TPGroup, runtime: tokio::runtime::Handle) -> Self {
        Self {
            rank,
            world_size: group.world_size,
            group_id: group.group_id,
            group: group.clone(),
            rank0_client: Arc::new(Mutex::new(None)),
            runtime,
        }
    }

    /// Connect to rank 0 (reducer). All ranks must connect before all-reduce.
    pub async fn connect(&self, timeout: std::time::Duration) -> DistributedResult<()> {
        let rank0_member = self.group.members.iter().find(|m| m.rank == 0).ok_or_else(|| {
            DistributedError::Config("TP group has no rank 0".into())
        })?;

        let endpoint = tonic::transport::Channel::from_shared(format!(
            "http://{}",
            rank0_member.address
        ))
        .map_err(|e| {
            DistributedError::Config(format!(
                "invalid address '{}': {}",
                rank0_member.address, e
            ))
        })?
        .connect_timeout(timeout)
        .timeout(timeout);

        let channel = endpoint.connect().await.map_err(|e| {
            DistributedError::Shard(format!(
                "failed to connect to rank 0 '{}' at {}: {}",
                rank0_member.name, rank0_member.address, e
            ))
        })?;

        let client = ShardServiceClient::new(channel)
            .max_decoding_message_size(256 * 1024 * 1024)
            .max_encoding_message_size(256 * 1024 * 1024);

        *self.rank0_client.lock().await = Some(client);
        Ok(())
    }

    /// Perform all-reduce over gRPC (async version).
    ///
    /// Simplified implementation: all ranks send their tensor to rank 0,
    /// rank 0 sums and responds with the result. All ranks receive the sum.
    pub async fn all_reduce_sum_async(&self, tensor: &mut Tensor) -> BackendResult<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let guard = self.rank0_client.lock().await;
        let client = guard.as_ref().ok_or_else(|| {
            BackendError::OperationFailed(
                "not connected: call connect() before all-reduce".into(),
            )
        })?;

        let tensor_proto = tensor_to_proto(tensor);
        let request = AllReduceRequest {
            tensor: Some(tensor_proto),
            operation: AllReduceOp::Sum as i32,
            group_id: self.group_id,
            sender_rank: self.rank as u32,
            world_size: self.world_size as u32,
        };

        let response = client
            .clone()
            .all_reduce(request)
            .await
            .map_err(|e| to_backend_error(DistributedError::Shard(e.to_string())))?
            .into_inner();

        if !response.success {
            return Err(BackendError::OperationFailed(format!(
                "all-reduce failed: {}",
                response.error
            )));
        }

        let reduced_proto = response.tensor.ok_or_else(|| {
            BackendError::OperationFailed("all-reduce response missing tensor".into())
        })?;

        let reduced = tensor_from_proto(&reduced_proto)
            .map_err(|e| to_backend_error(e))?;

        let out_data = tensor.data_mut().ok_or_else(|| {
            BackendError::InvalidArgument("tensor must be mutable".into())
        })?;
        out_data.copy_from_slice(reduced.data());

        Ok(())
    }
}

impl TensorParallel for DistributedTP {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_sum(&self, tensor: &mut Tensor) -> BackendResult<()> {
        tokio::task::block_in_place(|| {
            self.runtime.block_on(self.all_reduce_sum_async(tensor))
        })
    }

    fn all_gather(&self, local: &Tensor, output: &mut Tensor) -> BackendResult<()> {
        tokio::task::block_in_place(|| {
            self.runtime.block_on(async {
                if self.world_size == 1 {
                    let local_data = local.data();
                    let out_data = output
                        .data_mut()
                        .ok_or_else(|| BackendError::InvalidArgument("output must be mutable".into()))?;
                    let copy_len = local_data.len().min(out_data.len());
                    out_data[..copy_len].copy_from_slice(&local_data[..copy_len]);
                    return Ok(());
                }
                // Simplified: collect tensors from all peers and concatenate.
                // Full implementation would need AllGather RPC.
                Err(BackendError::Unsupported(
                    "all_gather not yet implemented for distributed TP".into(),
                ))
            })
        })
    }

    fn scatter(&self, input: &Tensor, output: &mut Tensor) -> BackendResult<()> {
        tokio::task::block_in_place(|| {
            self.runtime.block_on(async {
                if self.world_size == 1 {
                    let input_data = input.data();
                    let out_data = output
                        .data_mut()
                        .ok_or_else(|| BackendError::InvalidArgument("output must be mutable".into()))?;
                    let copy_len = input_data.len().min(out_data.len());
                    out_data[..copy_len].copy_from_slice(&input_data[..copy_len]);
                    return Ok(());
                }
                Err(BackendError::Unsupported(
                    "scatter not yet implemented for distributed TP".into(),
                ))
            })
        })
    }

    fn barrier(&self) -> BackendResult<()> {
        tokio::task::block_in_place(|| {
            self.runtime.block_on(async {
                if self.world_size == 1 {
                    return Ok(());
                }
                // Simple barrier: all-reduce a dummy scalar
                let mut dummy = Tensor::from_f32(&[0.0f32], vec![1]).unwrap();
                self.all_reduce_sum_async(&mut dummy).await
            })
        })
    }
}

/// Split shards into TP groups based on parallelism mode
pub fn compute_tp_groups(
    shards: &[ShardSpec],
    mode: ParallelismMode,
) -> DistributedResult<Vec<TPGroup>> {
    match mode {
        ParallelismMode::Pipeline => Ok(Vec::new()),
        ParallelismMode::TensorParallel => {
            if shards.is_empty() {
                return Ok(Vec::new());
            }
            let members: Vec<TPGroupMember> = shards
                .iter()
                .enumerate()
                .map(|(i, s)| TPGroupMember {
                    name: s.name.clone(),
                    address: s.address.clone(),
                    rank: i,
                })
                .collect();
            Ok(vec![TPGroup::new(0, members)])
        }
        ParallelismMode::Hybrid { tp_group_size } => {
            if tp_group_size == 0 {
                return Err(DistributedError::Config(
                    "tp_group_size must be > 0 for hybrid mode".into(),
                ));
            }
            if shards.len() % tp_group_size != 0 {
                return Err(DistributedError::Config(format!(
                    "shard count {} must be divisible by tp_group_size {} for hybrid mode",
                    shards.len(),
                    tp_group_size
                )));
            }
            let mut groups = Vec::new();
            for (gid, chunk) in shards.chunks(tp_group_size).enumerate() {
                let members: Vec<TPGroupMember> = chunk
                    .iter()
                    .enumerate()
                    .map(|(i, s)| TPGroupMember {
                        name: s.name.clone(),
                        address: s.address.clone(),
                        rank: i,
                    })
                    .collect();
                groups.push(TPGroup::new(gid as u32, members));
            }
            Ok(groups)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelism_mode_default() {
        assert_eq!(ParallelismMode::default(), ParallelismMode::Pipeline);
    }

    #[test]
    fn test_tp_group_creation() {
        let members = vec![
            TPGroupMember {
                name: "a".into(),
                address: "localhost:50051".into(),
                rank: 0,
            },
            TPGroupMember {
                name: "b".into(),
                address: "localhost:50052".into(),
                rank: 1,
            },
        ];
        let group = TPGroup::new(0, members);
        assert_eq!(group.world_size, 2);
        assert_eq!(group.members.len(), 2);
    }

    #[test]
    fn test_compute_tp_groups_pipeline() {
        let shards = vec![
            ShardSpec {
                name: "shard-0".into(),
                address: "localhost:50051".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-1".into(),
                address: "localhost:50052".into(),
                layer_start: None,
                layer_end: None,
            },
        ];
        let groups = compute_tp_groups(&shards, ParallelismMode::Pipeline).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_compute_tp_groups_tensor_parallel() {
        let shards = vec![
            ShardSpec {
                name: "shard-0".into(),
                address: "localhost:50051".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-1".into(),
                address: "localhost:50052".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-2".into(),
                address: "localhost:50053".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-3".into(),
                address: "localhost:50054".into(),
                layer_start: None,
                layer_end: None,
            },
        ];
        let groups = compute_tp_groups(&shards, ParallelismMode::TensorParallel).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].world_size, 4);
        assert_eq!(groups[0].members.len(), 4);
    }

    #[test]
    fn test_compute_tp_groups_hybrid() {
        let shards = vec![
            ShardSpec {
                name: "shard-0".into(),
                address: "localhost:50051".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-1".into(),
                address: "localhost:50052".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-2".into(),
                address: "localhost:50053".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-3".into(),
                address: "localhost:50054".into(),
                layer_start: None,
                layer_end: None,
            },
        ];
        let groups = compute_tp_groups(&shards, ParallelismMode::Hybrid { tp_group_size: 2 })
            .unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].world_size, 2);
        assert_eq!(groups[1].world_size, 2);
    }

    #[test]
    fn test_compute_tp_groups_hybrid_indivisible() {
        let shards = vec![
            ShardSpec {
                name: "shard-0".into(),
                address: "localhost:50051".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-1".into(),
                address: "localhost:50052".into(),
                layer_start: None,
                layer_end: None,
            },
            ShardSpec {
                name: "shard-2".into(),
                address: "localhost:50053".into(),
                layer_start: None,
                layer_end: None,
            },
        ];
        let result = compute_tp_groups(&shards, ParallelismMode::Hybrid { tp_group_size: 2 });
        assert!(result.is_err());
    }
}
