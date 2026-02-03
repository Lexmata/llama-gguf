//! Check top predictions from our model for various inputs.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::model::{InferenceContext, Model, ModelLoader};
use llama_rs::tokenizer::Tokenizer;
use llama_rs::gguf::GgufFile;
use std::sync::Arc;
use std::path::Path;

fn get_top_predictions(model: &dyn Model, tokenizer: &Tokenizer, backend: Arc<dyn Backend>, config: &llama_rs::model::ModelConfig, tokens: &[u32], desc: &str, n: usize) {
    let mut ctx = InferenceContext::new(config, backend);
    let logits = model.forward(tokens, &mut ctx).expect("Forward failed");
    
    let logits_data = logits.as_f32().unwrap();
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("{}:", desc);
    for i in 0..n {
        let (tok_id, logit) = indexed[i];
        let tok_str = tokenizer.decode(&[tok_id as u32]).unwrap_or_else(|_| format!("<{}>", tok_id));
        println!("  {}: token {} ({:?}): {:.4}", i + 1, tok_id, tok_str, logit);
    }
    println!();
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    
    let loader = ModelLoader::load(model_path).expect("Failed to load model");
    let config = loader.config().clone();
    let model = loader.build_model().expect("Failed to build model");
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    
    eprintln!();
    println!("=== Top Predictions ===");
    println!("(llama.cpp reference in parentheses)");
    println!();
    
    get_top_predictions(&model, &tokenizer, backend.clone(), &config, &[28], "'=' at pos 0 (llama.cpp: '')", 5);
    get_top_predictions(&model, &tokenizer, backend.clone(), &config, &[16, 28], "'1=' (llama.cpp: '1')", 5);
    get_top_predictions(&model, &tokenizer, backend.clone(), &config, &[10, 28], "'+=' (llama.cpp: '1')", 5);
    get_top_predictions(&model, &tokenizer, backend.clone(), &config, &[16, 10, 16, 28], "'1+1=' (llama.cpp: '2')", 5);
    get_top_predictions(&model, &tokenizer, backend.clone(), &config, &[17, 10, 17, 28], "'2+2=' (llama.cpp: '4')", 5);
    get_top_predictions(&model, &tokenizer, backend.clone(), &config, &[16, 16], "'11' (llama.cpp: '、')", 5);
}
