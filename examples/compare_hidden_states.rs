//! Compare hidden states with Python reference implementation
//!
//! This example outputs hidden states after each layer in JSON format
//! for comparison with the Python reference.
//!
//! Usage: cargo run --example compare_hidden_states -- <token_id>

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};
use serde_json::{json, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let token_id: u32 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(28); // Default to '=' token
    
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let backend = CpuBackend::new();
    let model = load_llama_model(model_path)?;
    
    let hidden_size = model.config().hidden_size;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let max_seq_len = 512;
    let freq_base = model.config().rope_config.freq_base;
    let freq_scale = model.config().rope_config.freq_scale;
    let num_layers = model.config().num_layers;
    
    eprintln!("Processing token {} through {} layers...", token_id, num_layers);
    
    let mut states: HashMap<String, Vec<f32>> = HashMap::new();
    
    // Get embedding
    let embedding = model.embed_tokens(&[token_id], &backend)?;
    let embedding = embedding.reshape(vec![hidden_size])?;
    let emb_data = embedding.as_f32()?.to_vec();
    states.insert("embedding".to_string(), emb_data.clone());
    
    eprintln!("Embedding: min={:.4}, max={:.4}", 
        emb_data.iter().cloned().fold(f32::INFINITY, f32::min),
        emb_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // Process through layers
    let mut hidden = embedding;
    
    // Initialize KV caches
    let mut k_caches: Vec<_> = (0..num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    let mut v_caches: Vec<_> = (0..num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    
    // Only process first 3 layers for comparison
    let layers_to_process = 3.min(num_layers);
    
    for layer_idx in 0..layers_to_process {
        let layer = &model.layers()[layer_idx];
        
        hidden = layer.forward(
            &hidden,
            &mut k_caches[layer_idx],
            &mut v_caches[layer_idx],
            0,  // pos
            freq_base,
            freq_scale,
            &backend,
        )?;
        
        let h_data = hidden.as_f32()?.to_vec();
        states.insert(format!("layer_{}", layer_idx), h_data.clone());
        
        eprintln!("Layer {}: min={:.4}, max={:.4}",
            layer_idx,
            h_data.iter().cloned().fold(f32::INFINITY, f32::min),
            h_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }
    
    // Output JSON
    let output: Value = states.into_iter()
        .map(|(k, v)| (k, json!(v)))
        .collect();
    
    println!("{}", serde_json::to_string_pretty(&output)?);
    
    Ok(())
}
