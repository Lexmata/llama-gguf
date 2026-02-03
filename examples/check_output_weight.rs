//! Check output weight dimensions and relationship to embeddings.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    println!("=== Weight Analysis ===\n");

    // Check embedding weight
    if let Some(emb_info) = gguf.data.get_tensor("token_embd.weight") {
        println!("token_embd.weight:");
        println!("  dims: {:?}", emb_info.dims);
        println!("  dtype: {:?}", emb_info.dtype);
        
        // Expected: [hidden_size, vocab_size] = [896, 151936] in GGUF
        // For embedding lookup, we want embedding[token] = emb_weight[:, token]
        // In column-major, column i is at offset i * hidden_size
    }
    println!();

    // Check output weight
    if let Some(out_info) = gguf.data.get_tensor("output.weight") {
        println!("output.weight:");
        println!("  dims: {:?}", out_info.dims);
        println!("  dtype: {:?}", out_info.dtype);
        
        // If tied to embeddings, this would be the same tensor
        // If not tied, shape should be [hidden_size, vocab_size] for logits = hidden @ W
    }
    println!();

    // Check if they're the same tensor (weight tying)
    let emb = gguf.data.get_tensor("token_embd.weight").expect("No emb");
    let out = gguf.data.get_tensor("output.weight").expect("No out");
    
    println!("Are output.weight and token_embd.weight the same?");
    println!("  Same dims: {}", emb.dims == out.dims);
    println!("  Same dtype: {}", emb.dtype == out.dtype);
    println!("  Same offset: {}", emb.offset == out.offset);
    println!();

    // Check actual values
    let emb_data = gguf.tensor_data("token_embd.weight").unwrap();
    let out_data = gguf.tensor_data("output.weight").unwrap();
    
    // Check first 16 bytes
    println!("First 16 bytes of token_embd.weight: {:02x?}", &emb_data[..16.min(emb_data.len())]);
    println!("First 16 bytes of output.weight: {:02x?}", &out_data[..16.min(out_data.len())]);
    
    // If they're tied, the bytes should be identical
    let same_data = emb_data.len() == out_data.len() && 
        emb_data.iter().zip(out_data.iter()).all(|(a, b)| a == b);
    println!();
    println!("Same tensor data: {}", same_data);

    // Dequantize and compare statistics
    let emb_tensor = {
        let tensor_info = gguf.data.get_tensor("token_embd.weight").unwrap();
        let tensor_data = gguf.tensor_data("token_embd.weight").unwrap();
        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        let dtype = DType::from(tensor_info.dtype);
        Tensor::new(tensor_data.to_vec(), shape, dtype).unwrap()
    };
    
    let out_tensor = {
        let tensor_info = gguf.data.get_tensor("output.weight").unwrap();
        let tensor_data = gguf.tensor_data("output.weight").unwrap();
        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        let dtype = DType::from(tensor_info.dtype);
        Tensor::new(tensor_data.to_vec(), shape, dtype).unwrap()
    };
    
    let numel = emb_tensor.numel();
    let mut emb_f32 = Tensor::zeros(vec![numel], DType::F32);
    let mut out_f32 = Tensor::zeros(vec![numel], DType::F32);
    
    backend.dequantize(&emb_tensor, &mut emb_f32).unwrap();
    backend.dequantize(&out_tensor, &mut out_f32).unwrap();
    
    let emb_vals = emb_f32.as_f32().unwrap();
    let out_vals = out_f32.as_f32().unwrap();
    
    println!();
    println!("Dequantized comparison:");
    println!("  First 5 emb values: {:?}", &emb_vals[..5]);
    println!("  First 5 out values: {:?}", &out_vals[..5]);
    
    // Check if they're related by transpose or identical
    let max_diff: f32 = emb_vals.iter()
        .zip(out_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Max diff (same position): {:.9}", max_diff);
    
    // Check statistics
    let emb_mean: f32 = emb_vals.iter().sum::<f32>() / emb_vals.len() as f32;
    let out_mean: f32 = out_vals.iter().sum::<f32>() / out_vals.len() as f32;
    let emb_rms = (emb_vals.iter().map(|x| x * x).sum::<f32>() / emb_vals.len() as f32).sqrt();
    let out_rms = (out_vals.iter().map(|x| x * x).sum::<f32>() / out_vals.len() as f32).sqrt();
    
    println!();
    println!("Statistics:");
    println!("  Embedding: mean={:.6}, rms={:.4}", emb_mean, emb_rms);
    println!("  Output:    mean={:.6}, rms={:.4}", out_mean, out_rms);
}
