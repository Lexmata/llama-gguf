// Test embedding lookup and compare
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{Tensor, DType};
use llama_rs::backend::cpu::CpuBackend;
use llama_rs::Backend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = CpuBackend::new();
    let gguf = GgufFile::open("/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
    
    // Get token embedding tensor
    let emb_info = gguf.data.get_tensor("token_embd.weight").unwrap();
    let emb_data = gguf.tensor_data("token_embd.weight").unwrap();
    
    println!("Embedding tensor: dims={:?}, dtype={:?}", emb_info.dims, emb_info.dtype);
    
    // Dims are [896, 151936] in GGUF
    // ne[0] = 896 (hidden_size, contiguous)
    // ne[1] = 151936 (vocab_size)
    
    let hidden_size = emb_info.dims[0] as usize;
    let vocab_size = emb_info.dims[1] as usize;
    
    println!("hidden_size={}, vocab_size={}", hidden_size, vocab_size);
    
    // Create tensor from raw data
    let emb_tensor = Tensor::new(emb_data.to_vec(), vec![hidden_size, vocab_size], 
        llama_rs::tensor::DType::from(emb_info.dtype))?;
    
    // Dequantize
    let mut emb_f32 = Tensor::zeros(vec![hidden_size, vocab_size], DType::F32);
    backend.dequantize(&emb_tensor, &mut emb_f32)?;
    
    let emb_data = emb_f32.as_f32()?;
    
    // Get embedding for token 16 ("1")
    // In GGUF column-major: embedding for token t is at indices t*hidden_size..(t+1)*hidden_size
    let token = 16;
    let start = token * hidden_size;
    let end = start + hidden_size;
    let token_emb = &emb_data[start..end];
    
    println!("\nEmbedding for token {} (first 10):", token);
    for (i, v) in token_emb.iter().take(10).enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }
    
    // Also get embedding for token 17 ("2")
    let token2 = 17;
    let start2 = token2 * hidden_size;
    let end2 = start2 + hidden_size;
    let token_emb2 = &emb_data[start2..end2];
    
    println!("\nEmbedding for token {} (first 10):", token2);
    for (i, v) in token_emb2.iter().take(10).enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }
    
    // Check if they're different
    let diff: f32 = token_emb.iter().zip(token_emb2.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("\nSum of absolute differences: {}", diff);
    
    Ok(())
}
