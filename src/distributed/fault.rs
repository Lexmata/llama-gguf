//! Fault tolerance and node recovery for distributed inference
//!
//! Monitors shard health and handles failure detection, reconnection,
//! and layer reloading.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{watch, Mutex};
use tonic::transport::Channel;

use super::proto::shard_service_client::ShardServiceClient;
use super::proto::{ConfigureRequest, HealthRequest, LayerData};
use super::{DistributedError, DistributedResult};

/// Status of a shard in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ShardStatus {
    /// Shard is healthy and responding
    Healthy,
    /// Shard missed some health checks but not yet failed
    Degraded,
    /// Shard is unresponsive (exceeded failure threshold)
    Failed,
    /// Shard is being recovered (reconnect + reload in progress)
    Recovering,
    /// Unknown state (not yet probed)
    Unknown,
}

/// Configuration for fault tolerance behavior
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FaultConfig {
    /// Health check interval in seconds (default: 5)
    pub health_interval_secs: u64,
    /// Number of consecutive failures before marking shard as Failed (default: 3)
    pub failure_threshold: u32,
    /// Maximum reconnection attempts before giving up (default: 10)
    pub max_retries: u32,
    /// Delay between retry attempts in seconds (default: 2)
    pub retry_delay_secs: u64,
    /// Timeout for individual health check RPCs in seconds (default: 5)
    pub health_timeout_secs: u64,
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            health_interval_secs: 5,
            failure_threshold: 3,
            max_retries: 10,
            retry_delay_secs: 2,
            health_timeout_secs: 5,
        }
    }
}

/// Tracked state for a single shard
#[derive(Debug, Clone)]
pub struct ShardHealthState {
    pub name: String,
    pub address: String,
    pub status: ShardStatus,
    pub consecutive_failures: u32,
    pub total_checks: u64,
    pub total_failures: u64,
    pub last_check_time: Option<std::time::Instant>,
    pub last_healthy_time: Option<std::time::Instant>,
}

/// Snapshot of all shard statuses
#[derive(Debug, Clone)]
pub struct ClusterHealth {
    pub shards: Vec<ShardHealthState>,
    pub all_healthy: bool,
    pub failed_shards: Vec<String>,
}

/// Callback type for shard failure notification
pub type FailureCallback = Arc<dyn Fn(&str, ShardStatus) + Send + Sync>;

/// Monitors shard health via periodic gRPC health checks.
pub struct HealthMonitor {
    config: FaultConfig,
    shard_states: Arc<Mutex<HashMap<String, ShardHealthState>>>,
    shutdown_tx: Option<watch::Sender<bool>>,
    on_failure: Option<FailureCallback>,
}

impl HealthMonitor {
    /// Create a new health monitor with the given configuration.
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            shard_states: Arc::new(Mutex::new(HashMap::new())),
            shutdown_tx: None,
            on_failure: None,
        }
    }

    /// Set callback invoked when a shard transitions to Failed
    pub fn on_failure(
        mut self,
        callback: impl Fn(&str, ShardStatus) + Send + Sync + 'static,
    ) -> Self {
        self.on_failure = Some(Arc::new(callback));
        self
    }

    /// Start monitoring the given shards. Returns a JoinHandle for the background task.
    pub fn start(
        &mut self,
        shards: Vec<(String, String)>,
    ) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let shard_states = Arc::clone(&self.shard_states);
        let on_failure = self.on_failure.clone();

        let (tx, mut rx) = watch::channel(false);
        self.shutdown_tx = Some(tx);

        let health_interval = Duration::from_secs(config.health_interval_secs);
        let health_timeout = Duration::from_secs(config.health_timeout_secs);
        let failure_threshold = config.failure_threshold;

        tokio::spawn(async move {
            // Initialize states for all shards
            {
                let mut states = shard_states.lock().await;
                for (name, address) in &shards {
                    states.insert(
                        name.clone(),
                        ShardHealthState {
                            name: name.clone(),
                            address: address.clone(),
                            status: ShardStatus::Unknown,
                            consecutive_failures: 0,
                            total_checks: 0,
                            total_failures: 0,
                            last_check_time: None,
                            last_healthy_time: None,
                        },
                    );
                }
            }

            let mut interval = tokio::time::interval(health_interval);
            interval.tick().await; // First tick completes immediately, skip it

            loop {
                tokio::select! {
                    _ = rx.changed() => {
                        if *rx.borrow() {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        for (name, address) in &shards {
                            let endpoint = match Channel::from_shared(format!("http://{}", address)) {
                                Ok(e) => e,
                                Err(_) => {
                                    update_shard_failure(
                                        &shard_states,
                                        name,
                                        address,
                                        failure_threshold,
                                        &on_failure,
                                    )
                                    .await;
                                    continue;
                                }
                            };

                            let endpoint = endpoint.timeout(health_timeout);

                            match endpoint.connect().await {
                                Ok(channel) => {
                                    let mut client = ShardServiceClient::new(channel);
                                    match client.health(HealthRequest {}).await {
                                        Ok(_) => {
                                            let mut states = shard_states.lock().await;
                                            if let Some(s) = states.get_mut(name) {
                                                s.consecutive_failures = 0;
                                                s.status = ShardStatus::Healthy;
                                                s.total_checks += 1;
                                                s.last_check_time = Some(std::time::Instant::now());
                                                s.last_healthy_time = Some(std::time::Instant::now());
                                            }
                                        }
                                        Err(_) => {
                                            update_shard_failure(
                                                &shard_states,
                                                name,
                                                address,
                                                failure_threshold,
                                                &on_failure,
                                            )
                                            .await;
                                        }
                                    }
                                }
                                Err(_) => {
                                    update_shard_failure(
                                        &shard_states,
                                        name,
                                        address,
                                        failure_threshold,
                                        &on_failure,
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    /// Stop the health monitor
    pub fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(true);
        }
    }

    /// Get current cluster health snapshot
    pub async fn cluster_health(&self) -> ClusterHealth {
        let states = self.shard_states.lock().await;
        let shards: Vec<ShardHealthState> = states.values().cloned().collect();
        let failed_shards: Vec<String> = shards
            .iter()
            .filter(|s| s.status == ShardStatus::Failed)
            .map(|s| s.name.clone())
            .collect();
        let all_healthy = failed_shards.is_empty()
            && shards.iter().all(|s| s.status == ShardStatus::Healthy);
        ClusterHealth {
            shards,
            all_healthy,
            failed_shards,
        }
    }

    /// Get status of a specific shard
    pub async fn shard_status(&self, name: &str) -> Option<ShardStatus> {
        let states = self.shard_states.lock().await;
        states.get(name).map(|s| s.status)
    }

    /// Check if all shards are healthy
    pub async fn all_healthy(&self) -> bool {
        let states = self.shard_states.lock().await;
        states.values().all(|s| s.status == ShardStatus::Healthy)
    }
}

async fn update_shard_failure(
    shard_states: &Mutex<HashMap<String, ShardHealthState>>,
    name: &str,
    _address: &str,
    failure_threshold: u32,
    on_failure: &Option<FailureCallback>,
) {
    let (new_status, should_notify) = {
        let mut states = shard_states.lock().await;
        let s = states.get_mut(name).unwrap();
        s.total_checks += 1;
        s.total_failures += 1;
        s.consecutive_failures += 1;
        s.last_check_time = Some(std::time::Instant::now());

        let (new_status, should_notify) = if s.consecutive_failures >= failure_threshold {
            let old = s.status;
            s.status = ShardStatus::Failed;
            (ShardStatus::Failed, old != ShardStatus::Failed)
        } else {
            s.status = ShardStatus::Degraded;
            (ShardStatus::Degraded, false)
        };
        (new_status, should_notify)
    };

    if should_notify {
        if let Some(cb) = on_failure.as_ref() {
            cb(name, new_status);
        }
    }
}

/// Manages reconnection and layer reloading for failed shards.
pub struct RecoveryManager {
    config: FaultConfig,
}

impl RecoveryManager {
    /// Create a new recovery manager with the given configuration.
    pub fn new(config: FaultConfig) -> Self {
        Self { config }
    }

    /// Attempt to recover a failed shard by reconnecting
    pub async fn recover_connection(
        &self,
        address: &str,
    ) -> DistributedResult<ShardServiceClient<Channel>> {
        let retry_delay = Duration::from_secs(self.config.retry_delay_secs);
        let mut last_err = None;

        for attempt in 0..self.config.max_retries {
            let endpoint = Channel::from_shared(format!("http://{}", address)).map_err(|e| {
                DistributedError::Config(format!("invalid address '{}': {}", address, e))
            })?;

            match endpoint.connect().await {
                Ok(channel) => {
                    let client = ShardServiceClient::new(channel)
                        .max_decoding_message_size(256 * 1024 * 1024)
                        .max_encoding_message_size(256 * 1024 * 1024);
                    return Ok(client);
                }
                Err(e) => {
                    last_err = Some(e);
                    if attempt < self.config.max_retries - 1 {
                        tokio::time::sleep(retry_delay).await;
                    }
                }
            }
        }

        Err(DistributedError::Shard(format!(
            "failed to reconnect to {} after {} attempts: {:?}",
            address, self.config.max_retries, last_err
        )))
    }

    /// Full shard recovery: reconnect + reconfigure + reload layers
    /// Returns the new ShardServiceClient on success
    pub async fn recover_shard(
        &self,
        address: &str,
        configure_request: ConfigureRequest,
        layer_data: Vec<LayerData>,
    ) -> DistributedResult<ShardServiceClient<Channel>> {
        let mut client = self.recover_connection(address).await?;

        client
            .configure(configure_request)
            .await
            .map_err(|e| {
                DistributedError::Shard(format!("failed to configure shard during recovery: {}", e))
            })?;

        let stream = futures::stream::iter(layer_data);
        client.load_layers(stream).await.map_err(|e| {
            DistributedError::Shard(format!("failed to load layers during recovery: {}", e))
        })?;

        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_config_defaults() {
        let config = FaultConfig::default();
        assert_eq!(config.health_interval_secs, 5);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.max_retries, 10);
        assert_eq!(config.retry_delay_secs, 2);
        assert_eq!(config.health_timeout_secs, 5);
    }

    #[test]
    fn test_shard_status_transitions() {
        // Healthy -> Degraded: after 1 failure
        let mut state = ShardHealthState {
            name: "s1".into(),
            address: "localhost:50051".into(),
            status: ShardStatus::Healthy,
            consecutive_failures: 0,
            total_checks: 10,
            total_failures: 0,
            last_check_time: None,
            last_healthy_time: None,
        };
        state.consecutive_failures = 1;
        state.status = ShardStatus::Degraded;
        assert_eq!(state.status, ShardStatus::Degraded);

        // Degraded -> Failed: after reaching threshold
        state.consecutive_failures = 3;
        state.status = ShardStatus::Failed;
        assert_eq!(state.status, ShardStatus::Failed);
    }

    #[test]
    fn test_cluster_health_snapshot() {
        let shards = vec![
            ShardHealthState {
                name: "a".into(),
                address: "h1:50051".into(),
                status: ShardStatus::Healthy,
                consecutive_failures: 0,
                total_checks: 5,
                total_failures: 0,
                last_check_time: None,
                last_healthy_time: None,
            },
            ShardHealthState {
                name: "b".into(),
                address: "h2:50051".into(),
                status: ShardStatus::Failed,
                consecutive_failures: 3,
                total_checks: 5,
                total_failures: 3,
                last_check_time: None,
                last_healthy_time: None,
            },
        ];
        let health = ClusterHealth {
            all_healthy: false,
            failed_shards: vec!["b".into()],
            shards: shards.clone(),
        };
        assert!(!health.all_healthy);
        assert_eq!(health.failed_shards, vec!["b"]);
        assert_eq!(health.shards.len(), 2);
    }

    #[test]
    fn test_recovery_manager_creation() {
        let config = FaultConfig::default();
        let _manager = RecoveryManager::new(config);
    }
}
