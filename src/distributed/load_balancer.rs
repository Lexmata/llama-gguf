//! Load balancing across heterogeneous hardware
//!
//! Collects per-shard performance metrics and suggests rebalancing
//! when workload distribution is uneven.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Performance metrics for a single shard
#[derive(Debug, Clone, Default)]
pub struct ShardMetrics {
    pub shard_name: String,
    pub layer_count: usize,
    pub total_forwards: u64,
    pub total_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub last_latency_us: u64,
    pub last_update: Option<Instant>,
}

impl ShardMetrics {
    /// Create new metrics for a shard
    pub fn new(name: impl Into<String>, layer_count: usize) -> Self {
        Self {
            shard_name: name.into(),
            layer_count,
            total_forwards: 0,
            total_latency_us: 0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
            last_latency_us: 0,
            last_update: None,
        }
    }

    /// Record a forward pass latency measurement
    pub fn record(&mut self, latency_us: u64) {
        self.total_forwards += 1;
        self.total_latency_us += latency_us;
        if latency_us < self.min_latency_us {
            self.min_latency_us = latency_us;
        }
        if latency_us > self.max_latency_us {
            self.max_latency_us = latency_us;
        }
        self.last_latency_us = latency_us;
        self.last_update = Some(Instant::now());
    }

    /// Average latency in microseconds
    pub fn avg_latency_us(&self) -> u64 {
        if self.total_forwards == 0 {
            0
        } else {
            self.total_latency_us / self.total_forwards
        }
    }

    /// Average latency per layer (for fair comparison across different layer counts)
    pub fn avg_latency_per_layer_us(&self) -> u64 {
        if self.layer_count == 0 || self.total_forwards == 0 {
            0
        } else {
            self.avg_latency_us() / self.layer_count as u64
        }
    }
}

/// Configuration for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalanceConfig {
    /// Minimum number of forwards before considering rebalance (default: 100)
    pub min_samples: u64,
    /// Imbalance ratio threshold to trigger rebalance suggestion (default: 2.0)
    /// If slowest shard is >ratio x faster shard per-layer latency, suggest rebalance
    pub imbalance_threshold: f64,
    /// Minimum interval between rebalance evaluations in seconds (default: 60)
    pub eval_interval_secs: u64,
}

impl Default for LoadBalanceConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            imbalance_threshold: 2.0,
            eval_interval_secs: 60,
        }
    }
}

/// Decision from load balancer evaluation
#[derive(Debug, Clone)]
pub enum RebalanceDecision {
    /// No rebalance needed
    Balanced,
    /// Rebalance suggested with new layer assignments
    Rebalance {
        reason: String,
        current_assignments: Vec<(String, usize)>,
        suggested_assignments: Vec<(String, usize)>,
        estimated_speedup: f64,
    },
    /// Not enough data to evaluate
    InsufficientData,
}

/// Collects per-shard metrics and suggests layer rebalancing.
pub struct LoadBalancer {
    config: LoadBalanceConfig,
    metrics: HashMap<String, ShardMetrics>,
    last_eval: Option<Instant>,
}

impl LoadBalancer {
    /// Create a new load balancer with the given configuration.
    pub fn new(config: LoadBalanceConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
            last_eval: None,
        }
    }

    /// Initialize metrics for shards
    pub fn register_shards(&mut self, shards: &[(String, usize)]) {
        for (name, layer_count) in shards {
            self.metrics
                .insert(name.clone(), ShardMetrics::new(name.clone(), *layer_count));
        }
    }

    /// Record a forward pass measurement for a shard
    pub fn record(&mut self, shard_name: &str, latency_us: u64) {
        if let Some(m) = self.metrics.get_mut(shard_name) {
            m.record(latency_us);
        }
    }

    /// Evaluate if rebalancing is needed
    pub fn evaluate(&mut self) -> RebalanceDecision {
        if self.metrics.is_empty() {
            return RebalanceDecision::InsufficientData;
        }

        // Check if enough samples on all shards
        let min_samples = self.config.min_samples;
        for m in self.metrics.values() {
            if m.total_forwards < min_samples {
                return RebalanceDecision::InsufficientData;
            }
        }

        // Check eval interval
        if let Some(last) = self.last_eval {
            if last.elapsed().as_secs() < self.config.eval_interval_secs {
                return RebalanceDecision::Balanced;
            }
        }
        self.last_eval = Some(Instant::now());

        // Compute per-layer latency for each shard
        let per_layer: Vec<(String, u64, usize)> = self
            .metrics
            .iter()
            .map(|(name, m)| {
                (
                    name.clone(),
                    m.avg_latency_per_layer_us(),
                    m.layer_count,
                )
            })
            .collect();

        let total_layers: usize = per_layer.iter().map(|(_, _, lc)| lc).sum();

        // Find min/max per-layer latency (excluding zero to avoid div by zero)
        let non_zero: Vec<_> = per_layer
            .iter()
            .filter(|(_, lat, _)| *lat > 0)
            .collect();

        if non_zero.is_empty() {
            return RebalanceDecision::Balanced;
        }

        let min_per_layer = non_zero.iter().map(|(_, lat, _)| lat).min().copied().unwrap_or(0);
        let max_per_layer = non_zero.iter().map(|(_, lat, _)| lat).max().copied().unwrap_or(0);

        if min_per_layer == 0 {
            return RebalanceDecision::Balanced;
        }

        let ratio = max_per_layer as f64 / min_per_layer as f64;
        if ratio <= self.config.imbalance_threshold {
            return RebalanceDecision::Balanced;
        }

        // Suggested: total_layers * (1/shard_latency) / sum(1/latency) per shard
        let inv_sum: f64 = non_zero
            .iter()
            .map(|(_, lat, _)| 1.0 / (*lat as f64))
            .sum();

        let current_assignments: Vec<(String, usize)> = per_layer
            .iter()
            .map(|(name, _, lc)| (name.clone(), *lc))
            .collect();

        let mut suggested_assignments: Vec<(String, usize)> = Vec::with_capacity(per_layer.len());
        let mut assigned: usize = 0;

        for (i, (name, lat_per_layer, _)) in per_layer.iter().enumerate() {
            let layers = if *lat_per_layer > 0 {
                let ideal = total_layers as f64 * (1.0 / *lat_per_layer as f64) / inv_sum;
                ideal.round() as usize
            } else {
                0
            };

            // Last shard gets remainder to ensure exact total
            let layers = if i == per_layer.len() - 1 {
                total_layers.saturating_sub(assigned)
            } else {
                layers.min(total_layers.saturating_sub(assigned))
            };

            assigned += layers;
            suggested_assignments.push((name.clone(), layers));
        }

        // Current bottleneck: max of (layer_count * latency_per_layer) per shard
        let current_bottleneck_us: u64 = per_layer
            .iter()
            .map(|(_, lat, lc)| *lat * (*lc as u64))
            .max()
            .unwrap_or(0);

        // New bottleneck with suggested assignment
        let new_bottleneck_us: u64 = suggested_assignments
            .iter()
            .zip(per_layer.iter())
            .map(|((_, new_lc), (_, lat, _))| *lat * (*new_lc as u64))
            .max()
            .unwrap_or(1);

        let estimated_speedup = if new_bottleneck_us > 0 {
            current_bottleneck_us as f64 / new_bottleneck_us as f64
        } else {
            1.0
        };

        RebalanceDecision::Rebalance {
            reason: format!(
                "per-layer latency ratio {:.2} exceeds threshold {:.2}",
                ratio, self.config.imbalance_threshold
            ),
            current_assignments,
            suggested_assignments,
            estimated_speedup,
        }
    }

    /// Get current metrics for all shards
    pub fn metrics(&self) -> &HashMap<String, ShardMetrics> {
        &self.metrics
    }

    /// Get metrics for a specific shard
    pub fn shard_metrics(&self, name: &str) -> Option<&ShardMetrics> {
        self.metrics.get(name)
    }

    /// Reset all metrics (e.g., after rebalancing)
    pub fn reset(&mut self) {
        for m in self.metrics.values_mut() {
            m.total_forwards = 0;
            m.total_latency_us = 0;
            m.min_latency_us = u64::MAX;
            m.max_latency_us = 0;
            m.last_latency_us = 0;
            m.last_update = None;
        }
        self.last_eval = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_metrics_recording() {
        let mut m = ShardMetrics::new("s1", 8);
        m.record(100);
        m.record(200);
        m.record(150);

        assert_eq!(m.total_forwards, 3);
        assert_eq!(m.total_latency_us, 450);
        assert_eq!(m.min_latency_us, 100);
        assert_eq!(m.max_latency_us, 200);
        assert_eq!(m.avg_latency_us(), 150);
    }

    #[test]
    fn test_shard_metrics_per_layer() {
        let mut m = ShardMetrics::new("s1", 8);
        m.record(800); // 100 us per layer
        m.record(800);

        assert_eq!(m.avg_latency_us(), 800);
        assert_eq!(m.avg_latency_per_layer_us(), 100);
    }

    #[test]
    fn test_load_balance_config_defaults() {
        let config = LoadBalanceConfig::default();
        assert_eq!(config.min_samples, 100);
        assert_eq!(config.imbalance_threshold, 2.0);
        assert_eq!(config.eval_interval_secs, 60);
    }

    #[test]
    fn test_evaluate_balanced() {
        let mut lb = LoadBalancer::new(LoadBalanceConfig {
            min_samples: 10,
            imbalance_threshold: 2.0,
            eval_interval_secs: 0,
        });
        lb.register_shards(&[
            ("a".into(), 8),
            ("b".into(), 8),
        ]);
        for _ in 0..20 {
            lb.record("a", 800);
            lb.record("b", 900);
        }

        match lb.evaluate() {
            RebalanceDecision::Balanced => {}
            other => panic!("expected Balanced, got {:?}", other),
        }
    }

    #[test]
    fn test_evaluate_imbalanced() {
        let mut lb = LoadBalancer::new(LoadBalanceConfig {
            min_samples: 10,
            imbalance_threshold: 2.0,
            eval_interval_secs: 0,
        });
        lb.register_shards(&[
            ("fast".into(), 8),
            ("slow".into(), 8),
        ]);
        for _ in 0..20 {
            lb.record("fast", 80);   // 10 us per layer
            lb.record("slow", 8000); // 1000 us per layer - 100x slower
        }

        match lb.evaluate() {
            RebalanceDecision::Rebalance { .. } => {}
            other => panic!("expected Rebalance, got {:?}", other),
        }
    }

    #[test]
    fn test_evaluate_insufficient_data() {
        let mut lb = LoadBalancer::new(LoadBalanceConfig {
            min_samples: 100,
            imbalance_threshold: 2.0,
            eval_interval_secs: 0,
        });
        lb.register_shards(&[
            ("a".into(), 8),
            ("b".into(), 8),
        ]);
        for _ in 0..10 {
            lb.record("a", 800);
            lb.record("b", 8000);
        }

        match lb.evaluate() {
            RebalanceDecision::InsufficientData => {}
            other => panic!("expected InsufficientData, got {:?}", other),
        }
    }

    #[test]
    fn test_rebalance_suggestion_distributes_inversely() {
        let mut lb = LoadBalancer::new(LoadBalanceConfig {
            min_samples: 50,
            imbalance_threshold: 2.0,
            eval_interval_secs: 0,
        });
        lb.register_shards(&[
            ("fast".into(), 8),
            ("slow".into(), 8),
        ]);
        for _ in 0..100 {
            lb.record("fast", 80);   // 10 us per layer
            lb.record("slow", 800); // 100 us per layer - 10x slower
        }

        match lb.evaluate() {
            RebalanceDecision::Rebalance {
                suggested_assignments,
                ..
            } => {
                let fast_layers = suggested_assignments
                    .iter()
                    .find(|(n, _)| n == "fast")
                    .map(|(_, l)| *l)
                    .unwrap();
                let slow_layers = suggested_assignments
                    .iter()
                    .find(|(n, _)| n == "slow")
                    .map(|(_, l)| *l)
                    .unwrap();
                // Fast shard should get more layers than slow
                assert!(
                    fast_layers > slow_layers,
                    "fast should get more layers (got fast={}, slow={})",
                    fast_layers,
                    slow_layers
                );
            }
            other => panic!("expected Rebalance, got {:?}", other),
        }
    }
}
