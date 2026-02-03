//! Trace hidden states through all layers

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    gguf::GgufFile,
    model::load_llama_model,
    tensor::{DType, Tensor},
    tokenizer::Tokenizer,
};

fn stats(data: &[f32]) -> (f32, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    (min, max, mean)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("=");
    
    eprintln!("Loading model from: {}", model_path);
    let backend = CpuBackend::new();
    
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let model = load_llama_model(model_path)?;
    
    let hidden_size = model.config().hidden_size;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let max_seq_len = 512;
    let freq_base = model.config().rope_config.freq_base;
    let freq_scale = model.config().rope_config.freq_scale;
    let num_layers = model.config().num_layers;
    
    eprintln!("Config: hidden_size={}, num_layers={}, freq_base={}, freq_scale={}", 
        hidden_size, num_layers, freq_base, freq_scale);
    
    // Encode the prompt
    let add_bos = gguf.data.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(true);
    let tokens = tokenizer.encode(prompt, add_bos)?;
    eprintln!("Prompt '{}' -> tokens: {:?}", prompt, tokens);
    
    // Use last token for single-token tracing
    let token_id = *tokens.last().unwrap();
    
    println!("\nTracing token {} through {} layers...\n", token_id, num_layers);
    
    // Get embedding
    let embedding = model.embed_tokens(&[token_id], &backend)?;
    let embedding = embedding.reshape(vec![hidden_size])?;
    let (min, max, mean) = stats(embedding.as_f32()?);
    println!("Embedding: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
    
    // Initialize KV caches
    let mut k_caches: Vec<_> = (0..num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    let mut v_caches: Vec<_> = (0..num_layers)
        .map(|_| Tensor::zeros(vec![num_kv_heads, max_seq_len, head_dim], DType::F32))
        .collect();
    
    let mut hidden = embedding;
    
    for layer_idx in 0..num_layers {
        let layer = &model.layers()[layer_idx];
        
        hidden = layer.forward(
            &hidden,
            &mut k_caches[layer_idx],
            &mut v_caches[layer_idx],
            0,
            freq_base,
            freq_scale,
            &backend,
        )?;
        
        let (min, max, mean) = stats(hidden.as_f32()?);
        
        // Flag if values are getting too large or NaN
        let flag = if min.is_nan() || max.is_nan() {
            " [NaN!]"
        } else if min.abs() > 1000.0 || max.abs() > 1000.0 {
            " [LARGE!]"
        } else if min.abs() > 100.0 || max.abs() > 100.0 {
            " [growing]"
        } else {
            ""
        };
        
        println!("Layer {:2}: min={:9.4}, max={:9.4}, mean={:9.4}{}", 
            layer_idx, min, max, mean, flag);
    }
    
    // Final norm
    let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
    model.norm().forward(&hidden, &mut normed, &backend)?;
    let (min, max, mean) = stats(normed.as_f32()?);
    println!("\nFinal norm: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
    
    // Output projection (get logits)
    let mut logits = Tensor::zeros(vec![model.config().vocab_size], DType::F32);
    model.output().forward(&normed, &mut logits, &backend)?;
    
    let logits_data = logits.as_f32()?;
    let (min, max, mean) = stats(logits_data);
    println!("Logits: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
    
    // Top 5 predictions
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 5 predictions:");
    for (idx, score) in indexed.iter().take(5) {
        println!("  {} ({:.4})", idx, score);
    }
    
    // Check token '2' (token 17)
    let token_2_logit = logits_data[17];
    let token_2_rank = indexed.iter().position(|(i, _)| *i == 17).unwrap_or(999999);
    println!("\nToken 17 ('2'): logit={:.4}, rank={}", token_2_logit, token_2_rank);
    
    Ok(())
}
