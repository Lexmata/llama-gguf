//! Coordinator for distributed inference
//!
//! The coordinator loads the GGUF model, partitions layers across shards,
//! streams weight tensors to each shard, and builds the `DistributedModel`
//! for inference.

use std::ops::Range;

use crate::model::{LlamaModel, Model, ModelLoader, RopeType};
use crate::model::layers::{Linear, RMSNorm};

use super::config::ClusterConfig;
use super::model::DistributedModel;
use super::pipeline::{PipelineExecutor, ShardConnection};
use super::proto::shard_service_client::ShardServiceClient;
use super::proto::{ConfigureRequest, HealthRequest, LayerData, NamedTensor};
use super::tensor_transfer::tensor_to_proto;
use super::{DistributedError, DistributedResult};

/// Orchestrates the distributed inference cluster.
///
/// Responsible for:
/// - Connecting to shard servers
/// - Loading the model and distributing layers
/// - Building the `DistributedModel` for use with `Engine`
pub struct Coordinator {
    config: ClusterConfig,
}

impl Coordinator {
    /// Create a new coordinator from cluster configuration.
    pub fn new(config: ClusterConfig) -> Self {
        Self { config }
    }

    /// Connect to all shards, load the model, distribute layers,
    /// and return a `DistributedModel` ready for inference.
    pub async fn setup(&self) -> DistributedResult<DistributedModel> {
        tracing::info!(
            "Setting up distributed cluster with {} shards",
            self.config.shards.len()
        );

        // Load the GGUF model on the coordinator
        let loader = ModelLoader::load(&self.config.model_path)?;
        let model = loader.build_model()?;
        let model_config = model.config().clone();

        let num_layers = model_config.num_layers;
        let architecture = model.architecture();

        // Compute layer assignments
        let assignments = self.config.compute_layer_assignments(num_layers)?;

        tracing::info!("Layer assignments:");
        for (i, range) in assignments.iter().enumerate() {
            tracing::info!(
                "  {} -> layers {}..{}",
                self.config.shards[i].name,
                range.start,
                range.end
            );
        }

        // Connect to all shards and configure them
        let mut shard_connections = Vec::new();

        for (i, shard_spec) in self.config.shards.iter().enumerate() {
            let range = &assignments[i];

            tracing::info!(
                "Connecting to shard '{}' at {}",
                shard_spec.name,
                shard_spec.address
            );

            let endpoint = tonic::transport::Channel::from_shared(format!(
                "http://{}",
                shard_spec.address
            ))
            .map_err(|e| {
                DistributedError::Config(format!(
                    "invalid shard address '{}': {}",
                    shard_spec.address, e
                ))
            })?
            .connect_timeout(self.config.connect_timeout())
            .timeout(self.config.request_timeout());

            let channel = endpoint.connect().await.map_err(|e| {
                DistributedError::Shard(format!(
                    "failed to connect to shard '{}' at {}: {}",
                    shard_spec.name, shard_spec.address, e
                ))
            })?;

            let mut client = ShardServiceClient::new(channel)
                .max_decoding_message_size(256 * 1024 * 1024) // 256MB for weight transfer
                .max_encoding_message_size(256 * 1024 * 1024);

            // Health check
            let health = client.health(HealthRequest {}).await.map_err(|e| {
                DistributedError::HealthCheckFailed {
                    name: shard_spec.name.clone(),
                    reason: e.to_string(),
                }
            })?;

            let health_resp = health.into_inner();
            if !health_resp.healthy {
                return Err(DistributedError::HealthCheckFailed {
                    name: shard_spec.name.clone(),
                    reason: "shard reported unhealthy".into(),
                });
            }

            tracing::info!(
                "Shard '{}' healthy (backend: {})",
                shard_spec.name,
                health_resp.backend_name
            );

            // Configure the shard
            let use_neox = matches!(
                model_config.rope_config.rope_type,
                RopeType::NeoX
            );

            let max_seq = if self.config.max_seq_len > 0 {
                self.config.max_seq_len
            } else {
                model_config.max_seq_len
            };

            client
                .configure(ConfigureRequest {
                    hidden_size: model_config.hidden_size as u32,
                    intermediate_size: model_config.intermediate_size as u32,
                    num_layers: num_layers as u32,
                    num_heads: model_config.num_heads as u32,
                    num_kv_heads: model_config.num_kv_heads as u32,
                    head_dim: model_config.head_dim as u32,
                    max_seq_len: max_seq as u32,
                    norm_eps: model_config.norm_eps,
                    rope_freq_base: model_config.rope_config.freq_base,
                    rope_freq_scale: model_config.rope_config.freq_scale,
                    use_neox_rope: use_neox,
                    layer_start: range.start as u32,
                    layer_end: range.end as u32,
                    use_gpu: self.config.use_gpu,
                })
                .await
                .map_err(|e| {
                    DistributedError::Shard(format!(
                        "failed to configure shard '{}': {}",
                        shard_spec.name, e
                    ))
                })?;

            // Stream layer weights to this shard
            self.send_layers(&model, &mut client, range).await?;

            shard_connections.push(ShardConnection {
                client,
                name: shard_spec.name.clone(),
                layer_start: range.start,
                layer_end: range.end,
            });
        }

        // Build the pipeline
        let pipeline = PipelineExecutor::new(shard_connections)?;

        // Extract coordinator-local components (embedding, norm, output)
        let token_embedding = model.token_embedding().clone();
        let norm = RMSNorm::new(model.norm().weight.clone(), model.norm().eps)
            .map_err(DistributedError::Model)?;
        let output = Linear::new(model.output().weight.clone(), model.output().bias.clone())
            .map_err(DistributedError::Model)?;

        let distributed_model = DistributedModel::new(
            token_embedding,
            norm,
            output,
            pipeline,
            model_config,
            architecture,
        );

        tracing::info!("Distributed cluster setup complete");
        Ok(distributed_model)
    }

    /// Stream all layer tensors for the given range to a shard.
    async fn send_layers(
        &self,
        model: &LlamaModel,
        client: &mut ShardServiceClient<tonic::transport::Channel>,
        range: &Range<usize>,
    ) -> DistributedResult<()> {
        let layers: Vec<LayerData> = model
            .layers()
            .iter()
            .filter(|l| range.contains(&l.layer_idx))
            .map(|layer| {
                let mut tensors = Vec::new();

                // Attention norm
                tensors.push(NamedTensor {
                    name: "attn_norm.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.attn_norm.weight)),
                });

                // Attention Q/K/V/O weights
                tensors.push(NamedTensor {
                    name: "attn_q.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.attention.wq.weight)),
                });
                if let Some(ref bias) = layer.attention.wq.bias {
                    tensors.push(NamedTensor {
                        name: "attn_q.bias".into(),
                        tensor: Some(tensor_to_proto(bias)),
                    });
                }

                tensors.push(NamedTensor {
                    name: "attn_k.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.attention.wk.weight)),
                });
                if let Some(ref bias) = layer.attention.wk.bias {
                    tensors.push(NamedTensor {
                        name: "attn_k.bias".into(),
                        tensor: Some(tensor_to_proto(bias)),
                    });
                }

                tensors.push(NamedTensor {
                    name: "attn_v.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.attention.wv.weight)),
                });
                if let Some(ref bias) = layer.attention.wv.bias {
                    tensors.push(NamedTensor {
                        name: "attn_v.bias".into(),
                        tensor: Some(tensor_to_proto(bias)),
                    });
                }

                tensors.push(NamedTensor {
                    name: "attn_output.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.attention.wo.weight)),
                });
                if let Some(ref bias) = layer.attention.wo.bias {
                    tensors.push(NamedTensor {
                        name: "attn_output.bias".into(),
                        tensor: Some(tensor_to_proto(bias)),
                    });
                }

                // FFN norm
                tensors.push(NamedTensor {
                    name: "ffn_norm.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.ffn_norm.weight)),
                });

                // FFN gate/up/down
                tensors.push(NamedTensor {
                    name: "ffn_gate.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.ffn.w_gate.weight)),
                });
                tensors.push(NamedTensor {
                    name: "ffn_up.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.ffn.w_up.weight)),
                });
                tensors.push(NamedTensor {
                    name: "ffn_down.weight".into(),
                    tensor: Some(tensor_to_proto(&layer.ffn.w_down.weight)),
                });

                LayerData {
                    layer_index: layer.layer_idx as u32,
                    tensors,
                }
            })
            .collect();

        let num_layers = layers.len();
        let stream = futures::stream::iter(layers);

        client
            .load_layers(stream)
            .await
            .map_err(|e| {
                DistributedError::Shard(format!("failed to stream layers: {}", e))
            })?;

        tracing::info!("Streamed {} layers to shard", num_layers);
        Ok(())
    }
}
