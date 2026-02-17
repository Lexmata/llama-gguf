//! Cluster configuration for distributed inference
//!
//! Defines the topology: which shards exist, their network addresses,
//! and how model layers are partitioned across them.

use std::ops::Range;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::DistributedError;

/// Configuration for a distributed inference cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Path to the GGUF model file (on the coordinator node)
    pub model_path: String,

    /// Ordered list of shard nodes
    pub shards: Vec<ShardSpec>,

    /// Timeout for initial connection to shards (seconds)
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,

    /// Timeout for individual RPC requests (seconds)
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Whether to use GPU on shards when available
    #[serde(default = "default_true")]
    pub use_gpu: bool,

    /// Maximum sequence length override (0 = use model default)
    #[serde(default)]
    pub max_seq_len: usize,
}

fn default_connect_timeout() -> u64 {
    10
}

fn default_request_timeout() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

/// Specification for a single shard node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardSpec {
    /// Human-readable name for this shard
    pub name: String,

    /// Network address in "host:port" format
    pub address: String,

    /// Layer range assigned to this shard. If not specified,
    /// the coordinator will auto-partition layers evenly.
    #[serde(default)]
    pub layer_start: Option<usize>,

    /// End of layer range (exclusive). Must be set if layer_start is set.
    #[serde(default)]
    pub layer_end: Option<usize>,
}

impl ShardSpec {
    /// Get the manually assigned layer range, if both start and end are set.
    pub fn layer_range(&self) -> Option<Range<usize>> {
        match (self.layer_start, self.layer_end) {
            (Some(start), Some(end)) => Some(start..end),
            _ => None,
        }
    }
}

impl ClusterConfig {
    /// Load cluster configuration from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, DistributedError> {
        let contents = std::fs::read_to_string(path.as_ref()).map_err(DistributedError::Io)?;
        let config: Self = toml::from_str(&contents).map_err(|e| {
            DistributedError::Config(format!("failed to parse cluster config: {}", e))
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Connection timeout as a Duration.
    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.connect_timeout_secs)
    }

    /// Request timeout as a Duration.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), DistributedError> {
        if self.shards.is_empty() {
            return Err(DistributedError::Config(
                "cluster must have at least one shard".into(),
            ));
        }

        for shard in &self.shards {
            if shard.address.is_empty() {
                return Err(DistributedError::Config(format!(
                    "shard '{}' has empty address",
                    shard.name
                )));
            }

            // If one of start/end is set, both must be set
            if shard.layer_start.is_some() != shard.layer_end.is_some() {
                return Err(DistributedError::Config(format!(
                    "shard '{}': layer_start and layer_end must both be set or both omitted",
                    shard.name
                )));
            }

            if let (Some(start), Some(end)) = (shard.layer_start, shard.layer_end) {
                if start >= end {
                    return Err(DistributedError::Config(format!(
                        "shard '{}': layer_start ({}) must be less than layer_end ({})",
                        shard.name, start, end
                    )));
                }
            }
        }

        Ok(())
    }

    /// Compute layer assignments for all shards.
    ///
    /// If shards have manually assigned ranges, those are used.
    /// Otherwise, layers are partitioned evenly across shards.
    ///
    /// Returns a Vec of `Range<usize>` in the same order as `self.shards`.
    pub fn compute_layer_assignments(
        &self,
        num_layers: usize,
    ) -> Result<Vec<Range<usize>>, DistributedError> {
        let all_manual = self.shards.iter().all(|s| s.layer_range().is_some());
        let all_auto = self.shards.iter().all(|s| s.layer_range().is_none());

        if !all_manual && !all_auto {
            return Err(DistributedError::Config(
                "either all shards must have manual layer assignments or none".into(),
            ));
        }

        if all_manual {
            let ranges: Vec<Range<usize>> = self
                .shards
                .iter()
                .map(|s| s.layer_range().unwrap())
                .collect();

            // Verify complete coverage
            let total_assigned: usize = ranges.iter().map(|r| r.len()).sum();
            if total_assigned != num_layers {
                return Err(DistributedError::LayerMismatch {
                    model_layers: num_layers,
                    assigned_layers: total_assigned,
                });
            }

            return Ok(ranges);
        }

        // Auto-partition: divide layers as evenly as possible
        let n_shards = self.shards.len();
        let base_layers = num_layers / n_shards;
        let remainder = num_layers % n_shards;

        let mut assignments = Vec::with_capacity(n_shards);
        let mut offset = 0;

        for i in 0..n_shards {
            let count = base_layers + if i < remainder { 1 } else { 0 };
            assignments.push(offset..offset + count);
            offset += count;
        }

        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_shard(name: &str, addr: &str) -> ShardSpec {
        ShardSpec {
            name: name.into(),
            address: addr.into(),
            layer_start: None,
            layer_end: None,
        }
    }

    #[test]
    fn test_auto_partition_even() {
        let config = ClusterConfig {
            model_path: "model.gguf".into(),
            shards: vec![
                test_shard("a", "host1:50051"),
                test_shard("b", "host2:50051"),
            ],
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            use_gpu: true,
            max_seq_len: 0,
        };

        let assignments = config.compute_layer_assignments(32).unwrap();
        assert_eq!(assignments, vec![0..16, 16..32]);
    }

    #[test]
    fn test_auto_partition_uneven() {
        let config = ClusterConfig {
            model_path: "model.gguf".into(),
            shards: vec![
                test_shard("a", "h1:50051"),
                test_shard("b", "h2:50051"),
                test_shard("c", "h3:50051"),
            ],
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            use_gpu: true,
            max_seq_len: 0,
        };

        let assignments = config.compute_layer_assignments(10).unwrap();
        // 10 / 3 = 3 remainder 1 -> first shard gets 4, rest get 3
        assert_eq!(assignments, vec![0..4, 4..7, 7..10]);
    }

    #[test]
    fn test_manual_partition() {
        let config = ClusterConfig {
            model_path: "model.gguf".into(),
            shards: vec![
                ShardSpec {
                    name: "a".into(),
                    address: "h1:50051".into(),
                    layer_start: Some(0),
                    layer_end: Some(10),
                },
                ShardSpec {
                    name: "b".into(),
                    address: "h2:50051".into(),
                    layer_start: Some(10),
                    layer_end: Some(32),
                },
            ],
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            use_gpu: true,
            max_seq_len: 0,
        };

        let assignments = config.compute_layer_assignments(32).unwrap();
        assert_eq!(assignments, vec![0..10, 10..32]);
    }

    #[test]
    fn test_manual_partition_mismatch() {
        let config = ClusterConfig {
            model_path: "model.gguf".into(),
            shards: vec![
                ShardSpec {
                    name: "a".into(),
                    address: "h1:50051".into(),
                    layer_start: Some(0),
                    layer_end: Some(10),
                },
            ],
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            use_gpu: true,
            max_seq_len: 0,
        };

        let result = config.compute_layer_assignments(32);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_empty_shards() {
        let config = ClusterConfig {
            model_path: "model.gguf".into(),
            shards: vec![],
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            use_gpu: true,
            max_seq_len: 0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_parse() {
        let toml_str = r#"
model_path = "model.gguf"
connect_timeout_secs = 15

[[shards]]
name = "gpu1"
address = "192.168.1.10:50051"

[[shards]]
name = "gpu2"
address = "192.168.1.11:50051"
"#;
        let config: ClusterConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.shards.len(), 2);
        assert_eq!(config.connect_timeout_secs, 15);
        assert_eq!(config.request_timeout_secs, 30); // default
    }
}
