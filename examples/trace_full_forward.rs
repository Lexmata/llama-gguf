//! Trace full forward pass to see where values diverge
//! 
//! This runs token "=" through all layers and shows intermediate stats

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading model from {}...", model_path);
    let backend = CpuBackend::new();
    let model = load_llama_model(model_path)?;
    
    // Token "=" is token 28
    let token_id = 28u32;
    let pos = 0usize;
    
    // Get model dimensions
    let hidden_size = model.config().hidden_size;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let max_seq_len = 512;
    let freq_base = model.config().rope_config.freq_base;
    let freq_scale = model.config().rope_config.freq_scale;
    
    println!("\n=== Testing token {} ('=') at position {} ===", token_id, pos);
    
    // Step 1: Get embedding
    println!("\n--- Embedding ---");
    let embedding = model.embed_tokens(&[token_id], &backend)?;
    let embedding = embedding.reshape(vec![hidden_size])?;
    let emb_data = embedding.as_f32()?;
    
    let stats = compute_stats(emb_data);
    println!("After embedding: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", 
        stats.0, stats.1, stats.2, stats.3);
    println!("First 5: {:?}", &emb_data[..5]);
    
    // Process through each layer
    let mut hidden = embedding;
    
    // Initialize KV caches for all layers
    let mut k_caches: Vec<_> = (0..model.config().num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    let mut v_caches: Vec<_> = (0..model.config().num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    
    for (layer_idx, layer) in model.layers().iter().enumerate() {
        hidden = layer.forward(
            &hidden,
            &mut k_caches[layer_idx],
            &mut v_caches[layer_idx],
            pos,
            freq_base,
            freq_scale,
            &backend,
        )?;
        
        let h_data = hidden.as_f32()?;
        let stats = compute_stats(h_data);
        
        if layer_idx < 5 || layer_idx >= model.config().num_layers - 2 {
            println!("After layer {:2}: min={:8.2}, max={:8.2}, mean={:8.4}, std={:8.4}", 
                layer_idx, stats.0, stats.1, stats.2, stats.3);
            println!("            first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);
        } else if layer_idx == 5 {
            println!("  ... (layers 5-{}) ...", model.config().num_layers - 3);
        }
    }
    
    // Final norm + output projection
    println!("\n--- Final Processing ---");
    
    // Apply final norm
    let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
    model.norm().forward(&hidden, &mut normed, &backend)?;
    
    let normed_data = normed.as_f32()?;
    let stats = compute_stats(normed_data);
    println!("After final norm: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", 
        stats.0, stats.1, stats.2, stats.3);
    println!("First 5: {:?}", &normed_data[..5]);
    
    // Apply output projection
    let mut logits = Tensor::zeros(vec![model.config().vocab_size], DType::F32);
    model.output().forward(&normed, &mut logits, &backend)?;
    
    let logits_data = logits.as_f32()?;
    let stats = compute_stats(logits_data);
    println!("Logits: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", 
        stats.0, stats.1, stats.2, stats.3);
    
    // Find top 5 tokens
    let mut indexed: Vec<_> = logits_data.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    
    println!("\nTop 5 predictions:");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} with logit {:.4}", i+1, idx, logit);
    }
    
    // Check token 17 (='2')
    let token_17_rank = indexed.iter().position(|(idx, _)| *idx == 17).unwrap();
    println!("\nToken 17 ('2') rank: {} with logit {:.4}", token_17_rank + 1, logits_data[17]);
    
    // Compare with llama-cpp reference
    println!("\n=== COMPARISON WITH LLAMA-CPP ===");
    println!("llama-cpp final hidden state for '=':");
    println!("  min=-101.79, max=62.94, mean=0.174");
    println!("  first 5: [8.03, 0.21, 1.87, 6.41, -5.38]");
    
    Ok(())
}

fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std = variance.sqrt();
    (min, max, mean, std)
}
