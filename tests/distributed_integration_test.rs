//! Integration tests for distributed pipeline-parallel inference.
//!
//! These tests start real gRPC shard servers on localhost and exercise
//! the full pipeline: configure -> load layers -> forward -> reset.
//!
//! No real GGUF model is required; tests construct synthetic layers
//! and verify the pipeline mechanics end-to-end.

#![cfg(feature = "distributed")]

use std::net::SocketAddr;
use std::time::Duration;

use llama_gguf::distributed::proto::shard_service_client::ShardServiceClient;
use llama_gguf::distributed::proto::{
    ConfigureRequest, ForwardRequest, HealthRequest, LayerData, NamedTensor, ResetRequest,
};
use llama_gguf::distributed::tensor_transfer::{tensor_from_proto, tensor_to_proto};
use llama_gguf::distributed::ShardServer;
use llama_gguf::tensor::{DType, Tensor};

const HIDDEN_SIZE: usize = 32;
const INTERMEDIATE_SIZE: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 8; // HIDDEN_SIZE / NUM_HEADS
const MAX_SEQ_LEN: usize = 64;

/// Find an available port by binding to port 0.
fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Start a shard server on the given port and return a handle to stop it.
async fn start_shard(name: &str, port: u16) -> tokio::task::JoinHandle<()> {
    let name = name.to_string();
    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();

    tokio::spawn(async move {
        let server = ShardServer::new(&name, false);
        // Ignore the error when the server is shut down
        let _ = server.serve(addr).await;
    })
}

/// Create a synthetic tensor with deterministic values for testing.
fn make_weight(shape: Vec<usize>, seed: f32) -> Tensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.001 + seed) * 0.01).collect();
    Tensor::from_f32(&data, shape).unwrap()
}

/// Build synthetic layer data for a transformer layer.
fn make_layer_data(layer_idx: u32) -> LayerData {
    let seed = layer_idx as f32;

    let mut tensors = Vec::new();

    // Attention norm weight [HIDDEN_SIZE]
    tensors.push(NamedTensor {
        name: "attn_norm.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(vec![HIDDEN_SIZE], seed))),
    });

    // Q/K/V/O weights [HIDDEN_SIZE, NUM_HEADS * HEAD_DIM] or [HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM]
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    tensors.push(NamedTensor {
        name: "attn_q.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![HIDDEN_SIZE, q_dim],
            seed + 0.1,
        ))),
    });
    tensors.push(NamedTensor {
        name: "attn_k.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![HIDDEN_SIZE, kv_dim],
            seed + 0.2,
        ))),
    });
    tensors.push(NamedTensor {
        name: "attn_v.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![HIDDEN_SIZE, kv_dim],
            seed + 0.3,
        ))),
    });
    tensors.push(NamedTensor {
        name: "attn_output.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![q_dim, HIDDEN_SIZE],
            seed + 0.4,
        ))),
    });

    // FFN norm weight [HIDDEN_SIZE]
    tensors.push(NamedTensor {
        name: "ffn_norm.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(vec![HIDDEN_SIZE], seed + 0.5))),
    });

    // FFN gate/up [HIDDEN_SIZE, INTERMEDIATE_SIZE], down [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    tensors.push(NamedTensor {
        name: "ffn_gate.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![HIDDEN_SIZE, INTERMEDIATE_SIZE],
            seed + 0.6,
        ))),
    });
    tensors.push(NamedTensor {
        name: "ffn_up.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![HIDDEN_SIZE, INTERMEDIATE_SIZE],
            seed + 0.7,
        ))),
    });
    tensors.push(NamedTensor {
        name: "ffn_down.weight".into(),
        tensor: Some(tensor_to_proto(&make_weight(
            vec![INTERMEDIATE_SIZE, HIDDEN_SIZE],
            seed + 0.8,
        ))),
    });

    LayerData {
        layer_index: layer_idx,
        tensors,
    }
}

/// Wait for a shard server to become available.
async fn wait_for_shard(addr: &str, timeout: Duration) -> ShardServiceClient<tonic::transport::Channel> {
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > timeout {
            panic!("Timed out waiting for shard at {}", addr);
        }

        let endpoint = tonic::transport::Channel::from_shared(format!("http://{}", addr))
            .unwrap()
            .connect_timeout(Duration::from_millis(500));

        if let Ok(channel) = endpoint.connect().await {
            let mut client = ShardServiceClient::new(channel);
            if client.health(HealthRequest {}).await.is_ok() {
                return client;
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn test_shard_health_check() {
    let port = find_free_port();
    let _handle = start_shard("test-health", port).await;

    let mut client = wait_for_shard(&format!("127.0.0.1:{}", port), Duration::from_secs(5)).await;

    let resp = client.health(HealthRequest {}).await.unwrap().into_inner();
    assert!(resp.healthy);
    assert_eq!(resp.shard_name, "test-health");
    assert_eq!(resp.layers_loaded, 0);
}

#[tokio::test]
async fn test_shard_configure() {
    let port = find_free_port();
    let _handle = start_shard("test-configure", port).await;

    let mut client = wait_for_shard(&format!("127.0.0.1:{}", port), Duration::from_secs(5)).await;

    let resp = client
        .configure(ConfigureRequest {
            hidden_size: HIDDEN_SIZE as u32,
            intermediate_size: INTERMEDIATE_SIZE as u32,
            num_layers: 4,
            num_heads: NUM_HEADS as u32,
            num_kv_heads: NUM_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            max_seq_len: MAX_SEQ_LEN as u32,
            norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            use_neox_rope: false,
            layer_start: 0,
            layer_end: 2,
            use_gpu: false,
        })
        .await
        .unwrap()
        .into_inner();

    assert!(resp.success);
    assert_eq!(resp.backend_name, "cpu");
}

#[tokio::test]
async fn test_shard_load_and_forward() {
    let port = find_free_port();
    let _handle = start_shard("test-forward", port).await;

    let mut client = wait_for_shard(&format!("127.0.0.1:{}", port), Duration::from_secs(5)).await;

    // Configure for 1 layer
    client
        .configure(ConfigureRequest {
            hidden_size: HIDDEN_SIZE as u32,
            intermediate_size: INTERMEDIATE_SIZE as u32,
            num_layers: 1,
            num_heads: NUM_HEADS as u32,
            num_kv_heads: NUM_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            max_seq_len: MAX_SEQ_LEN as u32,
            norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            use_neox_rope: false,
            layer_start: 0,
            layer_end: 1,
            use_gpu: false,
        })
        .await
        .unwrap();

    // Load 1 layer
    let layer_data = make_layer_data(0);
    let stream = futures::stream::iter(vec![layer_data]);
    let load_resp = client.load_layers(stream).await.unwrap().into_inner();
    assert!(load_resp.success);
    assert_eq!(load_resp.layers_loaded, 1);

    // Health check should show 1 layer
    let health = client.health(HealthRequest {}).await.unwrap().into_inner();
    assert_eq!(health.layers_loaded, 1);

    // Forward pass with a synthetic hidden state
    let hidden = Tensor::from_f32(&vec![0.1f32; HIDDEN_SIZE], vec![HIDDEN_SIZE]).unwrap();
    let hidden_proto = tensor_to_proto(&hidden);

    let fwd_resp = client
        .forward(ForwardRequest {
            hidden_state: Some(hidden_proto),
            position: 0,
            seq_len: 1,
        })
        .await
        .unwrap()
        .into_inner();

    assert!(fwd_resp.success);
    let output = tensor_from_proto(&fwd_resp.hidden_state.unwrap()).unwrap();
    assert_eq!(output.shape(), &[HIDDEN_SIZE]);
    assert_eq!(output.dtype(), DType::F32);

    // Output should be different from input (layers transform it)
    let output_data = output.as_f32().unwrap();
    let input_data = hidden.as_f32().unwrap();
    assert_ne!(output_data, input_data, "forward pass should transform the hidden state");
}

#[tokio::test]
async fn test_two_shard_pipeline() {
    let port1 = find_free_port();
    let port2 = find_free_port();

    let _handle1 = start_shard("shard-0", port1).await;
    let _handle2 = start_shard("shard-1", port2).await;

    let mut client1 =
        wait_for_shard(&format!("127.0.0.1:{}", port1), Duration::from_secs(5)).await;
    let mut client2 =
        wait_for_shard(&format!("127.0.0.1:{}", port2), Duration::from_secs(5)).await;

    // Configure both shards
    let base_config = ConfigureRequest {
        hidden_size: HIDDEN_SIZE as u32,
        intermediate_size: INTERMEDIATE_SIZE as u32,
        num_layers: 2,
        num_heads: NUM_HEADS as u32,
        num_kv_heads: NUM_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        max_seq_len: MAX_SEQ_LEN as u32,
        norm_eps: 1e-5,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        use_neox_rope: false,
        layer_start: 0,
        layer_end: 1,
        use_gpu: false,
    };

    client1.configure(base_config.clone()).await.unwrap();
    client2
        .configure(ConfigureRequest {
            layer_start: 1,
            layer_end: 2,
            ..base_config
        })
        .await
        .unwrap();

    // Load layer 0 on shard 0, layer 1 on shard 1
    let stream1 = futures::stream::iter(vec![make_layer_data(0)]);
    client1.load_layers(stream1).await.unwrap();

    let stream2 = futures::stream::iter(vec![make_layer_data(1)]);
    client2.load_layers(stream2).await.unwrap();

    // Pipeline: hidden -> shard0 -> shard1 -> output
    let hidden = Tensor::from_f32(&vec![0.1f32; HIDDEN_SIZE], vec![HIDDEN_SIZE]).unwrap();

    // Forward through shard 0
    let fwd1 = client1
        .forward(ForwardRequest {
            hidden_state: Some(tensor_to_proto(&hidden)),
            position: 0,
            seq_len: 1,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(fwd1.success);

    // Forward through shard 1
    let fwd2 = client2
        .forward(ForwardRequest {
            hidden_state: fwd1.hidden_state,
            position: 0,
            seq_len: 1,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(fwd2.success);

    let final_output = tensor_from_proto(&fwd2.hidden_state.unwrap()).unwrap();
    assert_eq!(final_output.shape(), &[HIDDEN_SIZE]);

    // Result should differ from single-shard forward (two layers applied)
    let single_shard_output = tensor_from_proto(
        &client1
            .forward(ForwardRequest {
                hidden_state: Some(tensor_to_proto(&hidden)),
                position: 1,
                seq_len: 2,
            })
            .await
            .unwrap()
            .into_inner()
            .hidden_state
            .unwrap(),
    )
    .unwrap();

    // Two-shard pipeline output should differ from re-running shard 0 alone
    // (because shard 1 applies different weights)
    assert_ne!(
        final_output.as_f32().unwrap(),
        single_shard_output.as_f32().unwrap(),
        "pipeline through 2 shards should differ from single shard"
    );
}

#[tokio::test]
async fn test_kv_cache_reset() {
    let port = find_free_port();
    let _handle = start_shard("test-reset", port).await;

    let mut client = wait_for_shard(&format!("127.0.0.1:{}", port), Duration::from_secs(5)).await;

    // Configure and load
    client
        .configure(ConfigureRequest {
            hidden_size: HIDDEN_SIZE as u32,
            intermediate_size: INTERMEDIATE_SIZE as u32,
            num_layers: 1,
            num_heads: NUM_HEADS as u32,
            num_kv_heads: NUM_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            max_seq_len: MAX_SEQ_LEN as u32,
            norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            use_neox_rope: false,
            layer_start: 0,
            layer_end: 1,
            use_gpu: false,
        })
        .await
        .unwrap();

    let stream = futures::stream::iter(vec![make_layer_data(0)]);
    client.load_layers(stream).await.unwrap();

    let hidden = Tensor::from_f32(&vec![0.1f32; HIDDEN_SIZE], vec![HIDDEN_SIZE]).unwrap();

    // Forward at position 0
    let fwd1 = client
        .forward(ForwardRequest {
            hidden_state: Some(tensor_to_proto(&hidden)),
            position: 0,
            seq_len: 1,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(fwd1.success);

    // Reset KV cache
    let reset_resp = client.reset_kv_cache(ResetRequest {}).await.unwrap().into_inner();
    assert!(reset_resp.success);

    // Forward at position 0 again should give the same result
    // (KV cache was reset, so position 0 is fresh)
    let fwd2 = client
        .forward(ForwardRequest {
            hidden_state: Some(tensor_to_proto(&hidden)),
            position: 0,
            seq_len: 1,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(fwd2.success);

    let out1 = tensor_from_proto(&fwd1.hidden_state.unwrap()).unwrap();
    let out2 = tensor_from_proto(&fwd2.hidden_state.unwrap()).unwrap();

    assert_eq!(
        out1.as_f32().unwrap(),
        out2.as_f32().unwrap(),
        "after reset, same input at same position should produce same output"
    );
}

#[tokio::test]
async fn test_forward_before_configure_fails() {
    let port = find_free_port();
    let _handle = start_shard("test-noconfig", port).await;

    let mut client = wait_for_shard(&format!("127.0.0.1:{}", port), Duration::from_secs(5)).await;

    let hidden = Tensor::from_f32(&vec![0.1f32; 32], vec![32]).unwrap();

    // Forward without configuring should fail
    let result = client
        .forward(ForwardRequest {
            hidden_state: Some(tensor_to_proto(&hidden)),
            position: 0,
            seq_len: 1,
        })
        .await;

    assert!(result.is_err(), "forward without configuration should fail");
}

#[tokio::test]
async fn test_load_before_configure_fails() {
    let port = find_free_port();
    let _handle = start_shard("test-noconfig-load", port).await;

    let mut client = wait_for_shard(&format!("127.0.0.1:{}", port), Duration::from_secs(5)).await;

    let stream = futures::stream::iter(vec![make_layer_data(0)]);
    let result = client.load_layers(stream).await;

    assert!(result.is_err(), "loading layers without configuration should fail");
}
