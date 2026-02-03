//! Verify Q5_0 dequantization

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    // Load token embedding tensor
    let info = gguf.data.get_tensor("token_embd.weight").expect("tensor not found");
    let data = gguf.tensor_data("token_embd.weight").expect("data not found");
    let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    
    println!("token_embd.weight:");
    println!("  GGUF shape (ne): {:?}", shape);  // [896, 151936]
    println!("  dtype: {:?}", DType::from(info.dtype));  // Should be Q5_0
    println!("  raw data len: {} bytes", data.len());
    
    // Create tensor and dequantize
    let tensor = Tensor::new(data.to_vec(), shape.clone(), DType::from(info.dtype)).unwrap();
    let total_elements = 896 * 151936;
    let mut out = Tensor::zeros(vec![total_elements], DType::F32);
    backend.dequantize(&tensor, &mut out).expect("dequantize failed");
    
    let dequant = out.as_f32().unwrap();
    
    // Check embedding for token 28
    let hidden_size = 896;
    let emb_28 = &dequant[28 * hidden_size..(28 + 1) * hidden_size];
    
    println!("\nToken 28 ('=') embedding:");
    println!("  First 10: {:?}", &emb_28[..10]);
    println!("  Sum: {:.6}", emb_28.iter().sum::<f32>());
    println!("  Min: {:.6}, Max: {:.6}", 
        emb_28.iter().cloned().fold(f32::INFINITY, f32::min),
        emb_28.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // Check if repeated values suggest wrong dequantization
    let mut unique_count = std::collections::HashSet::new();
    for &v in emb_28 {
        unique_count.insert(v.to_bits());
    }
    println!("  Unique values: {} / {}", unique_count.len(), hidden_size);
    
    // Also check token 17 ('2')
    let emb_17 = &dequant[17 * hidden_size..(17 + 1) * hidden_size];
    println!("\nToken 17 ('2') embedding:");
    println!("  First 5: {:?}", &emb_17[..5]);
    println!("  Sum: {:.6}", emb_17.iter().sum::<f32>());
    
    // And token 16 ('1')
    let emb_16 = &dequant[16 * hidden_size..(16 + 1) * hidden_size];
    println!("\nToken 16 ('1') embedding:");
    println!("  First 5: {:?}", &emb_16[..5]);
    println!("  Sum: {:.6}", emb_16.iter().sum::<f32>());
    
    // The key test: if our embeddings are correct, then embedding -> output proj
    // should give reasonable logits. Let's check that token 28 self-predicts
    println!("\n=== Testing embedding -> output projection ===");
    
    let output_info = gguf.data.get_tensor("output.weight").expect("tensor not found");
    let output_data = gguf.tensor_data("output.weight").expect("data not found");
    let output_shape: Vec<usize> = output_info.dims.iter().map(|&d| d as usize).collect();
    let output_tensor = Tensor::new(output_data.to_vec(), output_shape.clone(), DType::from(output_info.dtype)).unwrap();
    
    let vocab_size = 151936;
    let mut output_dequant = Tensor::zeros(vec![hidden_size * vocab_size], DType::F32);
    backend.dequantize(&output_tensor, &mut output_dequant).expect("dequantize failed");
    let output_weights = output_dequant.as_f32().unwrap();
    
    // Compute logits: logits[j] = sum_i(emb_28[i] * output[j * hidden + i])
    // For token 28 specifically
    let logit_28: f32 = (0..hidden_size)
        .map(|i| emb_28[i] * output_weights[28 * hidden_size + i])
        .sum();
    
    println!("Logit for token 28 (self): {:.6}", logit_28);
    
    // For token 17
    let logit_17: f32 = (0..hidden_size)
        .map(|i| emb_28[i] * output_weights[17 * hidden_size + i])
        .sum();
    
    println!("Logit for token 17 ('2'): {:.6}", logit_17);
    
    // Find top logit
    let mut top_logit = f32::NEG_INFINITY;
    let mut top_token = 0;
    for j in 0..vocab_size {
        let logit: f32 = (0..hidden_size)
            .map(|i| emb_28[i] * output_weights[j * hidden_size + i])
            .sum();
        if logit > top_logit {
            top_logit = logit;
            top_token = j;
        }
    }
    println!("Top prediction: token {} with logit {:.6}", top_token, top_logit);
    println!("(Expected: token 28 should be near the top since embedding self-similarity)");
}
