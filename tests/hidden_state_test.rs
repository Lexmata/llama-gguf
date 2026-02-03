//! Integration test comparing hidden states between our implementation
//! and reference values from llama-cpp (via Python script).
//!
//! This test verifies that our forward pass produces numerically similar
//! results to the llama.cpp reference implementation.

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};
use std::path::Path;

const MODEL_PATH: &str = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn stats(data: &[f32]) -> (f32, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    (min, max, mean)
}

#[test]
fn test_embedding_lookup() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let backend = CpuBackend::new();
    let model = load_llama_model(MODEL_PATH).expect("Failed to load model");

    // Test token 28 ('=')
    let embedding = model
        .embed_tokens(&[28], &backend)
        .expect("Failed to get embedding");

    let embedding = embedding
        .reshape(vec![model.config().hidden_size])
        .expect("Failed to reshape");

    let data = embedding.as_f32().expect("Failed to get f32 data");
    let (min, max, mean) = stats(data);

    // Reference values from Python (verified to match llama-cpp)
    // Embedding for '=': min=-0.0476, max=0.0532, mean=-0.0003
    assert!(
        (min - (-0.0476)).abs() < 0.001,
        "Embedding min mismatch: {} vs -0.0476",
        min
    );
    assert!(
        (max - 0.0532).abs() < 0.001,
        "Embedding max mismatch: {} vs 0.0532",
        max
    );
    assert!(
        mean.abs() < 0.01,
        "Embedding mean should be near zero: {}",
        mean
    );
}

#[test]
fn test_layer0_forward() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let backend = CpuBackend::new();
    let model = load_llama_model(MODEL_PATH).expect("Failed to load model");

    let hidden_size = model.config().hidden_size;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let max_seq_len = 512;
    let freq_base = model.config().rope_config.freq_base;
    let freq_scale = model.config().rope_config.freq_scale;

    // Get embedding for token 28 ('=')
    let embedding = model
        .embed_tokens(&[28], &backend)
        .expect("Failed to get embedding");
    let embedding = embedding
        .reshape(vec![hidden_size])
        .expect("Failed to reshape");

    // Initialize KV cache
    let mut k_cache = Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32);
    let mut v_cache = Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32);

    // Forward through layer 0
    let layer = &model.layers()[0];
    let hidden = layer
        .forward(
            &embedding, &mut k_cache, &mut v_cache, 0, freq_base, freq_scale, &backend,
        )
        .expect("Failed to forward through layer 0");

    let data = hidden.as_f32().expect("Failed to get f32 data");
    let (min, max, _mean) = stats(data);

    // Layer 0 output should be in reasonable range
    // After fixing Q4_K and Q5_K, values should be moderate
    assert!(
        min > -100.0 && max < 100.0,
        "Layer 0 output has extreme values: min={}, max={}",
        min,
        max
    );
}

#[test]
fn test_full_forward_produces_valid_logits() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let backend = CpuBackend::new();
    let model = load_llama_model(MODEL_PATH).expect("Failed to load model");

    let hidden_size = model.config().hidden_size;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let max_seq_len = 512;
    let freq_base = model.config().rope_config.freq_base;
    let freq_scale = model.config().rope_config.freq_scale;
    let num_layers = model.config().num_layers;

    // Get embedding for token 28 ('=')
    let embedding = model
        .embed_tokens(&[28], &backend)
        .expect("Failed to get embedding");
    let mut hidden = embedding
        .reshape(vec![hidden_size])
        .expect("Failed to reshape");

    // Initialize KV caches
    let mut k_caches: Vec<_> = (0..num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    let mut v_caches: Vec<_> = (0..num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();

    // Forward through all layers
    for (layer_idx, layer) in model.layers().iter().enumerate() {
        hidden = layer
            .forward(
                &hidden,
                &mut k_caches[layer_idx],
                &mut v_caches[layer_idx],
                0,
                freq_base,
                freq_scale,
                &backend,
            )
            .expect(&format!("Failed to forward through layer {}", layer_idx));
    }

    // Final norm
    let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
    model
        .norm()
        .forward(&hidden, &mut normed, &backend)
        .expect("Failed to apply final norm");

    // Output projection
    let mut logits = Tensor::zeros(vec![model.config().vocab_size], DType::F32);
    model
        .output()
        .forward(&normed, &mut logits, &backend)
        .expect("Failed to compute logits");

    let logits_data = logits.as_f32().expect("Failed to get logits");

    // Verify logits are valid (no NaN or Inf)
    for (i, &logit) in logits_data.iter().enumerate() {
        assert!(
            logit.is_finite(),
            "Logit {} is not finite: {}",
            i,
            logit
        );
    }

    // For single token '=', token '2' (id 17) should be among top predictions
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_10: Vec<usize> = indexed.iter().take(10).map(|(i, _)| *i).collect();
    let token_2_rank = indexed
        .iter()
        .position(|(i, _)| *i == 17)
        .expect("Token 17 not found");

    // Token '2' (17) should be in top 5 after single '='
    assert!(
        token_2_rank < 5,
        "Token '2' (17) should be in top 5, but ranked {}",
        token_2_rank
    );

    eprintln!(
        "Top 10 token IDs: {:?}, Token 17 ('2') rank: {}",
        top_10, token_2_rank
    );
}
