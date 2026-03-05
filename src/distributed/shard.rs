//! Shard server for distributed inference
//!
//! Each shard runs a gRPC server that receives model layers from the
//! coordinator and executes forward passes on its local backend.

use std::ops::Range;
use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status, Streaming};

use crate::backend::Backend;
use crate::model::{
    KVCache, ModelConfig, RopeConfig, RopeScalingType, RopeType,
};
use crate::model::layers::{
    Attention, AttentionLayer, FeedForward, FfnLayer, Linear, RMSNorm, TransformerLayer,
};
use crate::tensor::Tensor;

use super::proto::shard_service_server::{ShardService, ShardServiceServer};
use super::proto::{
    AllReduceRequest, AllReduceResponse,
    CapabilitiesRequest, CapabilitiesResponse,
    ConfigureRequest, ConfigureResponse, ForwardRequest, ForwardResponse, HealthRequest,
    HealthResponse, LayerData, LoadResponse, ResetRequest, ResetResponse,
};
use super::tensor_transfer::{tensor_from_proto, tensor_to_proto};

/// Internal state of a shard server, protected by a mutex for concurrent access.
struct ShardState {
    /// Transformer layers assigned to this shard
    layers: Vec<TransformerLayer>,
    /// KV cache for this shard's layers
    kv_cache: Option<KVCache>,
    /// Computation backend (CPU or GPU)
    backend: Arc<dyn Backend>,
    /// Layer range this shard is responsible for
    layer_range: Range<usize>,
    /// Model configuration (set during Configure)
    model_config: Option<ModelConfig>,
    /// Whether configuration has been received
    configured: bool,
}

/// A gRPC shard server that participates in distributed pipeline-parallel inference.
pub struct ShardServer {
    state: Arc<Mutex<ShardState>>,
    name: String,
    use_gpu: bool,
}

/// Select the best available GPU backend for this shard node.
///
/// Priority: CUDA > Metal > DX12 > Vulkan > CPU fallback.
/// Unlike `Engine::select_gpu_backend`, this doesn't need a `LlamaModel`
/// reference since shard nodes receive layers over gRPC, not from a local
/// GGUF file. GPU weight preloading (e.g. CUDA `load_model_weights`) is
/// skipped — the per-op compute path is used instead.
#[allow(unused_variables)]
fn select_shard_gpu_backend() -> Arc<dyn Backend> {
    #[cfg(feature = "cuda")]
    {
        match crate::backend::cuda::CudaBackend::new() {
            Ok(cuda) => {
                tracing::info!("Shard using CUDA backend: {}", cuda.device_name());
                return Arc::new(cuda);
            }
            Err(e) => {
                tracing::info!("CUDA not available on shard ({}), trying Metal...", e);
            }
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match crate::backend::metal::MetalBackend::new() {
            Ok(metal) => {
                tracing::info!("Shard using Metal backend: {}", metal.device_name());
                return Arc::new(metal);
            }
            Err(e) => {
                tracing::info!("Metal not available on shard ({}), trying DX12...", e);
            }
        }
    }

    #[cfg(all(feature = "dx12", target_os = "windows"))]
    {
        match crate::backend::dx12::Dx12Backend::new() {
            Ok(dx12) => {
                tracing::info!("Shard using DX12 backend: {}", dx12.device_name());
                return Arc::new(dx12);
            }
            Err(e) => {
                tracing::info!("DX12 not available on shard ({}), trying Vulkan...", e);
            }
        }
    }

    #[cfg(feature = "vulkan")]
    {
        match crate::backend::vulkan::VulkanBackend::new() {
            Ok(vk) => {
                tracing::info!("Shard using Vulkan backend: {}", vk.device_name());
                return Arc::new(vk);
            }
            Err(e) => {
                tracing::warn!("Vulkan not available on shard ({}), falling back to CPU", e);
            }
        }
    }

    tracing::info!("Shard using CPU backend");
    Arc::new(crate::backend::cpu::CpuBackend::new())
}

impl ShardServer {
    /// Create a new shard server.
    pub fn new(name: impl Into<String>, use_gpu: bool) -> Self {
        let backend: Arc<dyn Backend> = Arc::new(crate::backend::cpu::CpuBackend::new());

        Self {
            state: Arc::new(Mutex::new(ShardState {
                layers: Vec::new(),
                kv_cache: None,
                backend,
                layer_range: 0..0,
                model_config: None,
                configured: false,
            })),
            name: name.into(),
            use_gpu,
        }
    }

    /// Convert into a tonic service for serving.
    pub fn into_service(self) -> ShardServiceServer<Self> {
        ShardServiceServer::new(self)
    }

    /// Start the shard server on the given address.
    pub async fn serve(self, addr: impl Into<std::net::SocketAddr>) -> Result<(), tonic::transport::Error> {
        let addr = addr.into();
        tracing::info!("Shard '{}' listening on {}", self.name, addr);

        tonic::transport::Server::builder()
            .add_service(self.into_service())
            .serve(addr)
            .await
    }
}

/// Reconstruct a `TransformerLayer` from streamed tensor data.
fn build_layer_from_tensors(
    layer_idx: usize,
    tensors: &std::collections::HashMap<String, Tensor>,
    config: &ModelConfig,
) -> Result<TransformerLayer, Status> {
    let get = |name: &str| -> Result<Tensor, Status> {
        tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Status::invalid_argument(format!("missing tensor: {}", name)))
    };

    let get_opt = |name: &str| -> Option<Tensor> { tensors.get(name).cloned() };

    // Attention norm
    let attn_norm = RMSNorm::new(get("attn_norm.weight")?, config.norm_eps)
        .map_err(|e| Status::internal(format!("failed to build attn_norm: {}", e)))?;

    // Attention projections
    let wq = Linear::new(get("attn_q.weight")?, get_opt("attn_q.bias"))
        .map_err(|e| Status::internal(format!("failed to build wq: {}", e)))?;
    let wk = Linear::new(get("attn_k.weight")?, get_opt("attn_k.bias"))
        .map_err(|e| Status::internal(format!("failed to build wk: {}", e)))?;
    let wv = Linear::new(get("attn_v.weight")?, get_opt("attn_v.bias"))
        .map_err(|e| Status::internal(format!("failed to build wv: {}", e)))?;
    let wo = Linear::new(get("attn_output.weight")?, get_opt("attn_output.bias"))
        .map_err(|e| Status::internal(format!("failed to build wo: {}", e)))?;

    let use_neox_rope = matches!(config.rope_config.rope_type, RopeType::NeoX);
    let attention = Attention::with_rope_type(
        wq,
        wk,
        wv,
        wo,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        use_neox_rope,
    );

    // FFN norm
    let ffn_norm = RMSNorm::new(get("ffn_norm.weight")?, config.norm_eps)
        .map_err(|e| Status::internal(format!("failed to build ffn_norm: {}", e)))?;

    // Feed-forward
    let w_gate = Linear::new(get("ffn_gate.weight")?, None)
        .map_err(|e| Status::internal(format!("failed to build ffn_gate: {}", e)))?;
    let w_up = Linear::new(get("ffn_up.weight")?, None)
        .map_err(|e| Status::internal(format!("failed to build ffn_up: {}", e)))?;
    let w_down = Linear::new(get("ffn_down.weight")?, None)
        .map_err(|e| Status::internal(format!("failed to build ffn_down: {}", e)))?;

    let ffn = FeedForward::new(w_gate, w_up, w_down);

    Ok(TransformerLayer {
        attn_norm,
        attn_layer: AttentionLayer::FullAttention(attention),
        post_attn_norm: None,
        ffn_norm,
        ffn_layer: FfnLayer::Dense(ffn),
        post_ffn_norm: None,
        layer_idx,
    })
}

#[tonic::async_trait]
impl ShardService for ShardServer {
    async fn configure(
        &self,
        request: Request<ConfigureRequest>,
    ) -> Result<Response<ConfigureResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.lock().await;

        let rope_config = RopeConfig {
            freq_base: req.rope_freq_base,
            freq_scale: req.rope_freq_scale,
            n_dims: req.head_dim as usize,
            scaling_type: RopeScalingType::None,
            original_max_position_embeddings: req.max_seq_len as usize,
            rope_type: if req.use_neox_rope {
                RopeType::NeoX
            } else {
                RopeType::Normal
            },
        };

        let model_config = ModelConfig {
            vocab_size: 0, // Not needed on shard
            hidden_size: req.hidden_size as usize,
            intermediate_size: req.intermediate_size as usize,
            num_layers: req.num_layers as usize,
            num_heads: req.num_heads as usize,
            num_kv_heads: req.num_kv_heads as usize,
            head_dim: req.head_dim as usize,
            max_seq_len: req.max_seq_len as usize,
            norm_eps: req.norm_eps,
            rope_config,
            ..Default::default()
        };

        // Initialize KV cache for assigned layers
        let layer_count = (req.layer_end - req.layer_start) as usize;
        let max_seq = req.max_seq_len as usize;

        let kv_cache = KVCache::new(
            layer_count,
            req.num_kv_heads as usize,
            max_seq,
            req.head_dim as usize,
        );

        // Select backend: GPU if both the shard and the coordinator request it
        let backend: Arc<dyn Backend> = if self.use_gpu && req.use_gpu {
            select_shard_gpu_backend()
        } else {
            tracing::info!("Shard '{}' using CPU backend (GPU not requested)", self.name);
            Arc::new(crate::backend::cpu::CpuBackend::new())
        };

        let backend_name = backend.name().to_string();

        state.model_config = Some(model_config);
        state.kv_cache = Some(kv_cache);
        state.backend = backend;
        state.layer_range = (req.layer_start as usize)..(req.layer_end as usize);
        state.configured = true;

        tracing::info!(
            "Shard '{}' configured: layers {}..{}, backend: {}",
            self.name,
            req.layer_start,
            req.layer_end,
            backend_name
        );

        Ok(Response::new(ConfigureResponse {
            success: true,
            message: format!(
                "Configured for layers {}..{} with {} backend",
                req.layer_start, req.layer_end, backend_name
            ),
            backend_name,
        }))
    }

    async fn load_layers(
        &self,
        request: Request<Streaming<LayerData>>,
    ) -> Result<Response<LoadResponse>, Status> {
        let mut stream = request.into_inner();
        let mut layers_loaded = 0u32;

        let mut state = self.state.lock().await;
        if !state.configured {
            return Err(Status::failed_precondition(
                "shard must be configured before loading layers",
            ));
        }

        while let Some(layer_data) = stream
            .message()
            .await
            .map_err(|e| Status::internal(format!("stream error: {}", e)))?
        {
            let layer_idx = layer_data.layer_index as usize;

            // Deserialize all tensors for this layer
            let mut tensors = std::collections::HashMap::new();
            for named in &layer_data.tensors {
                let tensor = tensor_from_proto(
                    named.tensor.as_ref().ok_or_else(|| {
                        Status::invalid_argument(format!(
                            "layer {} tensor '{}' has no data",
                            layer_idx, named.name
                        ))
                    })?,
                )
                .map_err(|e| Status::internal(format!("tensor deserialization failed: {}", e)))?;

                tensors.insert(named.name.clone(), tensor);
            }

            // Build the transformer layer from tensors
            let config = state.model_config.as_ref().ok_or_else(|| {
                Status::internal("model config not available")
            })?;
            let layer = build_layer_from_tensors(layer_idx, &tensors, config)?;
            state.layers.push(layer);
            layers_loaded += 1;

            tracing::debug!(
                "Shard '{}': loaded layer {} ({} tensors)",
                self.name,
                layer_idx,
                tensors.len()
            );
        }

        // Sort layers by index for correct execution order
        state.layers.sort_by_key(|l| l.layer_idx);

        tracing::info!(
            "Shard '{}': loaded {} layers ({}..{})",
            self.name,
            layers_loaded,
            state.layer_range.start,
            state.layer_range.end
        );

        Ok(Response::new(LoadResponse {
            success: true,
            message: format!("loaded {} layers", layers_loaded),
            layers_loaded,
        }))
    }

    async fn forward(
        &self,
        request: Request<ForwardRequest>,
    ) -> Result<Response<ForwardResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();

        let mut state = self.state.lock().await;
        if state.layers.is_empty() {
            return Err(Status::failed_precondition("no layers loaded"));
        }

        let config = state.model_config.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard not configured")
        })?;

        let hidden_proto = req.hidden_state.ok_or_else(|| {
            Status::invalid_argument("missing hidden_state")
        })?;
        let mut hidden = tensor_from_proto(&hidden_proto)
            .map_err(|e| Status::internal(format!("failed to deserialize hidden state: {}", e)))?;

        let position = req.position as usize;
        let freq_base = config.rope_config.freq_base;
        let freq_scale = config.rope_config.freq_scale;

        let ShardState {
            ref layers,
            ref mut kv_cache,
            ref backend,
            ..
        } = *state;

        let kv_cache = kv_cache.as_mut().ok_or_else(|| {
            Status::internal("KV cache not initialized")
        })?;

        for (local_idx, layer) in layers.iter().enumerate() {
            hidden = layer
                .forward(
                    &hidden,
                    &mut kv_cache.k_cache[local_idx],
                    &mut kv_cache.v_cache[local_idx],
                    position,
                    freq_base,
                    freq_scale,
                    backend.as_ref(),
                    None,
                )
                .map_err(|e| {
                    Status::internal(format!(
                        "forward failed at layer {}: {}",
                        layer.layer_idx, e
                    ))
                })?;
        }

        kv_cache.seq_len = position + 1;

        let output_proto = tensor_to_proto(&hidden);
        let latency_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(ForwardResponse {
            hidden_state: Some(output_proto),
            success: true,
            error: String::new(),
            forward_latency_us: latency_us,
        }))
    }

    async fn reset_kv_cache(
        &self,
        _request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        let mut state = self.state.lock().await;
        if let Some(ref mut kv_cache) = state.kv_cache {
            kv_cache.reset();
        }
        tracing::debug!("Shard '{}': KV cache reset", self.name);

        Ok(Response::new(ResetResponse { success: true }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let state = self.state.lock().await;

        Ok(Response::new(HealthResponse {
            healthy: true,
            shard_name: self.name.clone(),
            backend_name: state.backend.name().to_string(),
            layer_start: state.layer_range.start as u32,
            layer_end: state.layer_range.end as u32,
            layers_loaded: state.layers.len() as u32,
            memory_used: state
                .kv_cache
                .as_ref()
                .map(|kv| kv.memory_usage() as u64)
                .unwrap_or(0),
            gpu_available: state.backend.name() != "cpu",
        }))
    }

    async fn get_capabilities(
        &self,
        _request: Request<CapabilitiesRequest>,
    ) -> Result<Response<CapabilitiesResponse>, Status> {
        let state = self.state.lock().await;
        let backend_name = state.backend.name().to_string();
        let _gpu_available = backend_name != "cpu";

        #[allow(unused_mut)]
        let mut total_vram: u64 = 0;
        #[allow(unused_mut)]
        let mut free_vram: u64 = 0;
        #[allow(unused_mut)]
        let mut gpu_name = String::new();
        #[allow(unused_mut)]
        let mut num_devices: u32 = 0;

        #[cfg(feature = "cuda")]
        if gpu_available {
            if let Ok(cuda) = crate::backend::cuda::CudaBackend::new() {
                total_vram = cuda.total_memory() as u64;
                free_vram = cuda.free_memory() as u64;
                gpu_name = cuda.device_name().to_string();
                num_devices = 1;
            }
        }

        let sys_info = sys_memory_info();

        Ok(Response::new(CapabilitiesResponse {
            total_vram_bytes: total_vram,
            free_vram_bytes: free_vram,
            gpu_name,
            backend_name,
            num_gpu_devices: num_devices,
            total_ram_bytes: sys_info.0,
            free_ram_bytes: sys_info.1,
        }))
    }

    async fn all_reduce(
        &self,
        request: Request<AllReduceRequest>,
    ) -> Result<Response<AllReduceResponse>, Status> {
        let req = request.into_inner();

        let tensor_proto = req.tensor.ok_or_else(|| {
            Status::invalid_argument("missing tensor in AllReduceRequest")
        })?;

        let tensor = tensor_from_proto(&tensor_proto)
            .map_err(|e| Status::internal(format!("tensor deserialization failed: {}", e)))?;

        let output_proto = tensor_to_proto(&tensor);

        Ok(Response::new(AllReduceResponse {
            tensor: Some(output_proto),
            success: true,
            error: String::new(),
        }))
    }
}

/// Get system RAM info (total, free) in bytes.
fn sys_memory_info() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total = 0u64;
            let mut available = 0u64;
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total = parse_meminfo_kb(line) * 1024;
                } else if line.starts_with("MemAvailable:") {
                    available = parse_meminfo_kb(line) * 1024;
                }
            }
            return (total, available);
        }
    }
    (0, 0)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
