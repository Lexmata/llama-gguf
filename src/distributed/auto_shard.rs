//! Automatic VRAM-based model sharding
//!
//! Probes shard hardware capabilities via gRPC and assigns layers
//! proportional to available VRAM.

use std::ops::Range;

use super::proto::shard_service_client::ShardServiceClient;
use super::proto::CapabilitiesRequest;
use super::{DistributedError, DistributedResult};

/// Hardware capabilities of a shard node, collected via gRPC probe
#[derive(Debug, Clone)]
pub struct ShardCapabilities {
    pub shard_name: String,
    pub address: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub gpu_name: String,
    pub backend_name: String,
    pub num_gpu_devices: u32,
    pub total_ram_bytes: u64,
    pub free_ram_bytes: u64,
}

/// Estimated memory requirements for a model
#[derive(Debug, Clone)]
pub struct ModelMemoryEstimate {
    /// Bytes per transformer layer (weights + KV cache at max_seq_len)
    pub bytes_per_layer: u64,
    /// Bytes for embedding table
    pub embedding_bytes: u64,
    /// Bytes for output projection + norm
    pub output_bytes: u64,
    /// Total number of layers
    pub num_layers: usize,
}

/// Probe a single shard's capabilities via gRPC
pub async fn probe_shard(
    address: &str,
    name: &str,
    timeout: std::time::Duration,
) -> DistributedResult<ShardCapabilities> {
    let endpoint = tonic::transport::Channel::from_shared(format!("http://{}", address))
        .map_err(|e| {
            DistributedError::Config(format!("invalid shard address '{}': {}", address, e))
        })?
        .connect_timeout(timeout)
        .timeout(timeout);

    let channel = endpoint.connect().await.map_err(|e| {
        DistributedError::Shard(format!(
            "failed to connect to shard '{}' at {}: {}",
            name, address, e
        ))
    })?;

    let mut client = ShardServiceClient::new(channel);
    let response = client
        .get_capabilities(CapabilitiesRequest {})
        .await
        .map_err(|e| {
            DistributedError::Shard(format!(
                "get_capabilities failed on shard '{}': {}",
                name, e
            ))
        })?
        .into_inner();

    Ok(ShardCapabilities {
        shard_name: name.to_string(),
        address: address.to_string(),
        total_vram_bytes: response.total_vram_bytes,
        free_vram_bytes: response.free_vram_bytes,
        gpu_name: response.gpu_name,
        backend_name: response.backend_name,
        num_gpu_devices: response.num_gpu_devices,
        total_ram_bytes: response.total_ram_bytes,
        free_ram_bytes: response.free_ram_bytes,
    })
}

/// Probe all shards in the cluster and return their capabilities
pub async fn probe_all_shards(
    shards: &[super::config::ShardSpec],
    timeout: std::time::Duration,
) -> DistributedResult<Vec<ShardCapabilities>> {
    let futures: Vec<_> = shards
        .iter()
        .map(|s| probe_shard(&s.address, &s.name, timeout))
        .collect();
    futures::future::join_all(futures)
        .await
        .into_iter()
        .collect()
}

/// Estimate memory requirements per layer from model config
pub fn estimate_model_memory(
    config: &crate::model::ModelConfig,
    max_seq_len: usize,
) -> ModelMemoryEstimate {
    let h = config.hidden_size as u64;
    let i = config.intermediate_size as u64;
    let kv_heads = config.num_kv_heads as u64;
    let head_dim = config.head_dim as u64;
    let max_seq = max_seq_len as u64;
    let vocab = config.vocab_size as u64;

    // Attention: wq + wk + wv + wo = 4 * hidden_size^2 * 2 (f16)
    let attn_bytes = 4 * h * h * 2;
    // FFN: gate + up + down = 3 * hidden_size * intermediate_size * 2 (f16)
    let ffn_bytes = 3 * h * i * 2;
    // KV cache: 2 * num_kv_heads * head_dim * max_seq_len * 4 (f32)
    let kv_bytes = 2 * kv_heads * head_dim * max_seq * 4;

    let bytes_per_layer = attn_bytes + ffn_bytes + kv_bytes;
    let embedding_bytes = vocab * h * 2;
    let output_bytes = vocab * h * 2 + h * 4;

    ModelMemoryEstimate {
        bytes_per_layer,
        embedding_bytes,
        output_bytes,
        num_layers: config.num_layers,
    }
}

/// Compute VRAM-proportional layer assignments
///
/// Assigns more layers to shards with more free VRAM.
/// Falls back to even distribution if all shards report 0 VRAM (CPU-only).
pub fn compute_vram_assignments(
    capabilities: &[ShardCapabilities],
    estimate: &ModelMemoryEstimate,
) -> DistributedResult<Vec<Range<usize>>> {
    let num_layers = estimate.num_layers;
    let n_shards = capabilities.len();

    if n_shards == 0 {
        return Err(DistributedError::Config("no shards to assign".into()));
    }

    if num_layers < n_shards {
        return Err(DistributedError::Config(format!(
            "model has {} layers but {} shards; each shard needs at least 1 layer",
            num_layers, n_shards
        )));
    }

    // Determine which memory metric to use
    let all_zero_vram = capabilities.iter().all(|c| c.free_vram_bytes == 0);
    let use_ram = all_zero_vram;

    let memories: Vec<u64> = if use_ram {
        capabilities
            .iter()
            .map(|c| {
                if c.free_ram_bytes == 0 && c.total_ram_bytes == 0 {
                    1u64 // Fallback for even split when no RAM info
                } else {
                    c.free_ram_bytes.max(1)
                }
            })
            .collect()
    } else {
        capabilities
            .iter()
            .map(|c| c.free_vram_bytes.max(1))
            .collect()
    };

    let total_memory: u64 = memories.iter().sum();
    let all_zero = total_memory == memories.len() as u64; // All 1s from fallback

    let layer_counts = if all_zero {
        // Even split
        let base = num_layers / n_shards;
        let remainder = num_layers % n_shards;
        (0..n_shards)
            .map(|i| base + if i < remainder { 1 } else { 0 })
            .collect::<Vec<_>>()
    } else {
        // Proportional assignment
        let mut counts: Vec<usize> = memories
            .iter()
            .map(|&m| {
                let proportion = m as f64 / total_memory as f64;
                (proportion * num_layers as f64).floor() as usize
            })
            .collect();

        let mut remaining = num_layers - counts.iter().sum::<usize>();

        // Distribute remaining layers to shards with most capacity
        while remaining > 0 {
            let mut best_idx = 0;
            let mut best_ratio = 0.0f64;
            for (i, &mem) in memories.iter().enumerate() {
                let ratio = mem as f64 / (counts[i] + 1) as f64;
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_idx = i;
                }
            }
            counts[best_idx] += 1;
            remaining -= 1;
        }

        counts
    };

    // Ensure every shard gets at least 1 layer (should be guaranteed by num_layers >= n_shards check)
    for (i, &c) in layer_counts.iter().enumerate() {
        if c == 0 {
            return Err(DistributedError::Config(format!(
                "shard '{}' would get 0 layers; need at least {} layers for {} shards",
                capabilities[i].shard_name, n_shards, n_shards
            )));
        }
    }

    // Convert counts to ranges
    let mut offset = 0;
    let ranges: Vec<Range<usize>> = layer_counts
        .iter()
        .map(|&count| {
            let start = offset;
            offset += count;
            start..offset
        })
        .collect();

    Ok(ranges)
}

/// High-level auto-sharding: probe capabilities + compute assignments
pub async fn auto_shard(
    shards: &[super::config::ShardSpec],
    model_config: &crate::model::ModelConfig,
    max_seq_len: usize,
    timeout: std::time::Duration,
) -> DistributedResult<(Vec<ShardCapabilities>, Vec<Range<usize>>)> {
    let capabilities = probe_all_shards(shards, timeout).await?;
    let estimate = estimate_model_memory(model_config, max_seq_len);
    let assignments = compute_vram_assignments(&capabilities, &estimate)?;
    Ok((capabilities, assignments))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 2048,
            ..Default::default()
        }
    }

    #[test]
    fn test_estimate_model_memory() {
        let config = test_config();
        let estimate = estimate_model_memory(&config, 2048);
        assert!(estimate.bytes_per_layer > 0);
        assert!(estimate.embedding_bytes > 0);
        assert!(estimate.output_bytes > 0);
        assert_eq!(estimate.num_layers, 32);

        // Sanity check: bytes_per_layer should be in hundreds of MB for 7B-scale
        assert!(estimate.bytes_per_layer > 10_000_000);
    }

    #[test]
    fn test_vram_proportional_assignment() {
        let estimate = ModelMemoryEstimate {
            bytes_per_layer: 100_000_000,
            embedding_bytes: 256_000_000,
            output_bytes: 256_000_000,
            num_layers: 30,
        };
        let capabilities = vec![
            ShardCapabilities {
                shard_name: "gpu1".into(),
                address: "host1:50051".into(),
                total_vram_bytes: 24 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024, // 16GB
                gpu_name: "RTX 4090".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
            ShardCapabilities {
                shard_name: "gpu2".into(),
                address: "host2:50051".into(),
                total_vram_bytes: 12 * 1024 * 1024 * 1024,
                free_vram_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                gpu_name: "RTX 3080".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 32 * 1024 * 1024 * 1024,
                free_ram_bytes: 16 * 1024 * 1024 * 1024,
            },
        ];
        let assignments = compute_vram_assignments(&capabilities, &estimate).unwrap();
        assert_eq!(assignments.len(), 2);
        // 16GB vs 8GB -> 2:1 ratio -> ~20 vs ~10 layers
        assert!(assignments[0].len() >= 18 && assignments[0].len() <= 22);
        assert!(assignments[1].len() >= 8 && assignments[1].len() <= 12);
        assert_eq!(assignments[0].len() + assignments[1].len(), 30);
    }

    #[test]
    fn test_vram_even_fallback() {
        let estimate = ModelMemoryEstimate {
            bytes_per_layer: 100_000_000,
            embedding_bytes: 256_000_000,
            output_bytes: 256_000_000,
            num_layers: 30,
        };
        // All shards have 0 VRAM and 0 free RAM -> even split
        let capabilities = vec![
            ShardCapabilities {
                shard_name: "cpu1".into(),
                address: "host1:50051".into(),
                total_vram_bytes: 0,
                free_vram_bytes: 0,
                gpu_name: String::new(),
                backend_name: "cpu".into(),
                num_gpu_devices: 0,
                total_ram_bytes: 0,
                free_ram_bytes: 0,
            },
            ShardCapabilities {
                shard_name: "cpu2".into(),
                address: "host2:50051".into(),
                total_vram_bytes: 0,
                free_vram_bytes: 0,
                gpu_name: String::new(),
                backend_name: "cpu".into(),
                num_gpu_devices: 0,
                total_ram_bytes: 0,
                free_ram_bytes: 0,
            },
        ];
        let assignments = compute_vram_assignments(&capabilities, &estimate).unwrap();
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].len(), 15);
        assert_eq!(assignments[1].len(), 15);
    }

    #[test]
    fn test_vram_minimum_one_layer_error() {
        let estimate = ModelMemoryEstimate {
            bytes_per_layer: 100_000_000,
            embedding_bytes: 256_000_000,
            output_bytes: 256_000_000,
            num_layers: 2,
        };
        let capabilities = vec![
            ShardCapabilities {
                shard_name: "s1".into(),
                address: "h1:50051".into(),
                total_vram_bytes: 16 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
                gpu_name: "GPU".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
            ShardCapabilities {
                shard_name: "s2".into(),
                address: "h2:50051".into(),
                total_vram_bytes: 16 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
                gpu_name: "GPU".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
            ShardCapabilities {
                shard_name: "s3".into(),
                address: "h3:50051".into(),
                total_vram_bytes: 16 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
                gpu_name: "GPU".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
        ];
        let result = compute_vram_assignments(&capabilities, &estimate);
        assert!(result.is_err());
    }

    #[test]
    fn test_vram_minimum_one_layer_ok() {
        let estimate = ModelMemoryEstimate {
            bytes_per_layer: 100_000_000,
            embedding_bytes: 256_000_000,
            output_bytes: 256_000_000,
            num_layers: 3,
        };
        let capabilities = vec![
            ShardCapabilities {
                shard_name: "s1".into(),
                address: "h1:50051".into(),
                total_vram_bytes: 16 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
                gpu_name: "GPU".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
            ShardCapabilities {
                shard_name: "s2".into(),
                address: "h2:50051".into(),
                total_vram_bytes: 16 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
                gpu_name: "GPU".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
            ShardCapabilities {
                shard_name: "s3".into(),
                address: "h3:50051".into(),
                total_vram_bytes: 16 * 1024 * 1024 * 1024,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
                gpu_name: "GPU".into(),
                backend_name: "cuda".into(),
                num_gpu_devices: 1,
                total_ram_bytes: 64 * 1024 * 1024 * 1024,
                free_ram_bytes: 32 * 1024 * 1024 * 1024,
            },
        ];
        let assignments = compute_vram_assignments(&capabilities, &estimate).unwrap();
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0], 0..1);
        assert_eq!(assignments[1], 1..2);
        assert_eq!(assignments[2], 2..3);
    }

    #[test]
    fn test_vram_single_shard() {
        let estimate = ModelMemoryEstimate {
            bytes_per_layer: 100_000_000,
            embedding_bytes: 256_000_000,
            output_bytes: 256_000_000,
            num_layers: 32,
        };
        let capabilities = vec![ShardCapabilities {
            shard_name: "single".into(),
            address: "host:50051".into(),
            total_vram_bytes: 24 * 1024 * 1024 * 1024,
            free_vram_bytes: 20 * 1024 * 1024 * 1024,
            gpu_name: "RTX 4090".into(),
            backend_name: "cuda".into(),
            num_gpu_devices: 1,
            total_ram_bytes: 64 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
        }];
        let assignments = compute_vram_assignments(&capabilities, &estimate).unwrap();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0], 0..32);
    }
}
