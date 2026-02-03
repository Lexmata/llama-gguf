//! Test if the issue is specifically with position 1.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::model::{InferenceContext, Model, ModelLoader};
use std::sync::Arc;

fn test_position(model: &dyn Model, backend: Arc<dyn Backend>, config: &llama_rs::model::ModelConfig, tokens: &[u32], desc: &str) {
    let mut ctx = InferenceContext::new(config, backend);
    let logits = model.forward(tokens, &mut ctx).expect("Forward failed");
    
    let logits_data = logits.as_f32().unwrap();
    let logit_17 = logits_data[17];
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_17 = indexed.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("{:<40} | {:>8.4} | {:>6}", desc, logit_17, rank_17);
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let loader = ModelLoader::load(model_path).expect("Failed to load model");
    let config = loader.config().clone();
    let model = loader.build_model().expect("Failed to build model");
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    
    eprintln!("Running tests...");
    eprintln!();
    
    println!("Token sequence                           | Logit 17 | Rank 17");
    println!("-----------------------------------------+----------+--------");
    
    // Test single "=" at position 0
    test_position(&model, backend.clone(), &config, &[28], "'=' at pos 0");
    
    // Test "1" at pos 0, "=" at pos 1
    test_position(&model, backend.clone(), &config, &[16, 28], "'1=' - '=' at pos 1");
    
    // Test "+" at pos 0, "=" at pos 1 (different first token)
    test_position(&model, backend.clone(), &config, &[10, 28], "'+=' - '=' at pos 1");
    
    // Test "a" at pos 0, "=" at pos 1 (different first token)
    test_position(&model, backend.clone(), &config, &[64, 28], "'a=' - '=' at pos 1");
    
    // Test " " (space) at pos 0, "=" at pos 1
    test_position(&model, backend.clone(), &config, &[220, 28], "' =' - '=' at pos 1");
    
    // What about "=" at pos 0, "=" at pos 1?
    test_position(&model, backend.clone(), &config, &[28, 28], "'==' - '=' at pos 1");
    
    // Multiple "=" tokens
    test_position(&model, backend.clone(), &config, &[28, 28, 28], "'===' - '=' at pos 2");
    test_position(&model, backend.clone(), &config, &[28, 28, 28, 28], "'====' - '=' at pos 3");
    
    // Original "1+1="
    test_position(&model, backend.clone(), &config, &[16, 10, 16, 28], "'1+1=' - '=' at pos 3");
    
    // What about "11"? (token 16 twice)
    test_position(&model, backend.clone(), &config, &[16, 16], "'11' - '1' at pos 1");
    
    println!();
    println!("Analysis:");
    println!("- If rank is similar across different first tokens, issue is in RoPE/position handling");
    println!("- If rank depends on first token, issue is in how attention combines information");
}
