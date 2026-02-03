//! Compare single layer output with expected values
//! 
//! The hypothesis is that something in our layer computation is wrong.
//! Let's trace very carefully what should happen.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;
use std::fs::File;
use std::io::Write;

fn load_tensor(gguf: &GgufFile, name: &str) -> Tensor {
    let info = gguf.data.get_tensor(name).unwrap();
    let data = gguf.tensor_data(name).unwrap();
    let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    Tensor::new(data.to_vec(), shape, DType::from(info.dtype)).unwrap()
}

fn try_load_tensor(gguf: &GgufFile, name: &str) -> Option<Tensor> {
    let info = gguf.data.get_tensor(name)?;
    let data = gguf.tensor_data(name)?;
    let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    Tensor::new(data.to_vec(), shape, DType::from(info.dtype)).ok()
}

fn dequant(backend: &CpuBackend, t: &Tensor) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        t.as_f32().unwrap().to_vec()
    } else {
        let mut out = Tensor::zeros(vec![t.numel()], DType::F32);
        backend.dequantize(t, &mut out).unwrap();
        out.as_f32().unwrap().to_vec()
    }
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    let hidden_size = 896;
    
    // Load just the embedding and output layers (no transformer layers)
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let output_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let output_w = dequant(&backend, &load_tensor(&gguf, "output.weight"));
    
    println!("=== Embedding verification ===");
    
    // Get embedding for token 28 ("=")
    let token = 28usize;
    let embedding: Vec<f32> = emb[token * hidden_size..(token + 1) * hidden_size].to_vec();
    
    println!("Token {} embedding:", token);
    println!("  First 10: {:?}", &embedding[..10]);
    println!("  Sum: {:.6}", embedding.iter().sum::<f32>());
    println!("  L2 norm: {:.6}", embedding.iter().map(|x| x*x).sum::<f32>().sqrt());
    
    // Save embedding to file for Python comparison
    let mut f = File::create("/tmp/rust_embedding_28.txt").unwrap();
    for (i, &v) in embedding.iter().enumerate() {
        writeln!(f, "{} {:.8}", i, v).unwrap();
    }
    println!("\nSaved embedding to /tmp/rust_embedding_28.txt");
    
    // Now let's think about what could be wrong:
    // 1. Embedding lookup - we're taking row [token] which is data[token*896..(token+1)*896]
    // 2. This assumes Q4_K layout is [vocab, hidden] which matches output.weight
    
    // Let me verify by checking embedding for token 0 vs a token with known patterns
    println!("\n=== Cross-check with token 0 ===");
    let emb_0: Vec<f32> = emb[0..hidden_size].to_vec();
    let emb_1: Vec<f32> = emb[hidden_size..2*hidden_size].to_vec();
    
    println!("Token 0 sum: {:.6}", emb_0.iter().sum::<f32>());
    println!("Token 1 sum: {:.6}", emb_1.iter().sum::<f32>());
    println!("Token 28 sum: {:.6}", embedding.iter().sum::<f32>());
    
    // If the layout was wrong (column-major instead of row-major),
    // we'd see a pattern like emb[0], emb[vocab_size], emb[2*vocab_size], etc.
    // Let's check if that could explain things
    
    println!("\n=== Testing alternative layout ===");
    let vocab_size = 151936;
    // If layout is [hidden, vocab] (column major):
    // emb[token] = data[0 + token * stride], data[1 + token * stride], ...
    // where stride could be vocab_size
    
    // Actually, the dequantized data is a flat array of vocab_size * hidden_size floats
    // Index i corresponds to some (vocab, hidden) or (hidden, vocab) pair
    
    // Current assumption: data[vocab * hidden_size + hidden_dim] = emb[vocab][hidden_dim]
    // Alternative: data[hidden_dim * vocab_size + vocab] = emb[vocab][hidden_dim]
    
    // Let's try the alternative
    let mut alt_embedding = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        if i * vocab_size + token < emb.len() {
            alt_embedding[i] = emb[i * vocab_size + token];
        }
    }
    
    println!("Alternative layout embedding for token {}:", token);
    println!("  First 10: {:?}", &alt_embedding[..10]);
    println!("  Sum: {:.6}", alt_embedding.iter().sum::<f32>());
    
    // If the alternative makes more sense, we have found the bug!
    // The embedding would be column-major in GGUF
    
    // Save alternative for comparison
    let mut f = File::create("/tmp/rust_embedding_28_alt.txt").unwrap();
    for (i, &v) in alt_embedding.iter().enumerate() {
        writeln!(f, "{} {:.8}", i, v).unwrap();
    }
    println!("Saved alternative embedding to /tmp/rust_embedding_28_alt.txt");
}
