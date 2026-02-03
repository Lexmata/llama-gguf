//! Trace the actual model forward pass with detailed debugging.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::model::{InferenceContext, Model, ModelLoader};
use std::sync::Arc;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading model...");
    let loader = ModelLoader::load(model_path).expect("Failed to load model");
    let config = loader.config().clone();
    let model = loader.build_model().expect("Failed to build model");
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    
    println!("Model config:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_heads: {}", config.num_heads);
    println!("  num_kv_heads: {}", config.num_kv_heads);
    println!("  num_layers: {}", config.num_layers);
    println!("  vocab_size: {}", config.vocab_size);
    println!();
    
    // Create inference context
    let mut ctx = InferenceContext::new(&config, backend.clone());
    
    // Test with "="  alone (single token)
    println!("=== Test 1: Single token '=' ===");
    ctx.reset();
    let tokens_single: Vec<u32> = vec![28];  // "="
    let logits_single = model.forward(&tokens_single, &mut ctx).expect("Forward failed");
    
    let logits_data = logits_single.as_f32().unwrap();
    let logit_17 = logits_data[17];
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_17 = indexed.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("Token 17 ('2') logit: {:.4}, rank: {}", logit_17, rank_17);
    println!("Top 5:");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", i + 1, idx, logit);
    }
    println!();
    
    // Test with "1=" (two tokens)
    println!("=== Test 2: Two tokens '1=' ===");
    ctx.reset();
    let tokens_two: Vec<u32> = vec![16, 28];  // "1="
    let logits_two = model.forward(&tokens_two, &mut ctx).expect("Forward failed");
    
    let logits_data = logits_two.as_f32().unwrap();
    let logit_17 = logits_data[17];
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_17 = indexed.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("Token 17 ('2') logit: {:.4}, rank: {}", logit_17, rank_17);
    println!("Top 5:");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", i + 1, idx, logit);
    }
    println!();
    
    // Test with "1+1=" (four tokens)
    println!("=== Test 3: Four tokens '1+1=' ===");
    ctx.reset();
    let tokens_four: Vec<u32> = vec![16, 10, 16, 28];  // "1+1="
    let logits_four = model.forward(&tokens_four, &mut ctx).expect("Forward failed");
    
    let logits_data = logits_four.as_f32().unwrap();
    let logit_17 = logits_data[17];
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_17 = indexed.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("Token 17 ('2') logit: {:.4}, rank: {}", logit_17, rank_17);
    println!("Top 5:");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", i + 1, idx, logit);
    }
}
