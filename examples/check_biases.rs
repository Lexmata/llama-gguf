//! Check Qwen2 bias magnitudes

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

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

fn stats(name: &str, data: &[f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let sq_mean = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let std = (sq_mean - mean * mean).sqrt();
    println!("{}: len={}, min={:.4}, max={:.4}, mean={:.4}, std={:.4}", 
             name, data.len(), min, max, mean, std);
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    println!("=== Checking Qwen2 bias magnitudes ===\n");
    
    // Check biases for layer 0
    for layer in [0, 11, 23] {
        println!("--- Layer {} ---", layer);
        
        if let Some(q_bias) = try_load_tensor(&gguf, &format!("blk.{}.attn_q.bias", layer)) {
            stats("Q bias", &dequant(&backend, &q_bias));
        }
        
        if let Some(k_bias) = try_load_tensor(&gguf, &format!("blk.{}.attn_k.bias", layer)) {
            stats("K bias", &dequant(&backend, &k_bias));
        }
        
        if let Some(v_bias) = try_load_tensor(&gguf, &format!("blk.{}.attn_v.bias", layer)) {
            stats("V bias", &dequant(&backend, &v_bias));
        }
        println!();
    }
    
    // The Q and K biases are large!
    // But in llama.cpp and transformers, these biases are added AFTER the 
    // linear projection but BEFORE RoPE in the attention mechanism.
    //
    // Key question: are these biases supposed to be this large?
    // Let's also check the projection weights to see if the magnitudes make sense.
    
    println!("=== Checking layer 0 weight magnitudes ===\n");
    
    let load_tensor = |name: &str| {
        let info = gguf.data.get_tensor(name).unwrap();
        let data = gguf.tensor_data(name).unwrap();
        let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        Tensor::new(data.to_vec(), shape, DType::from(info.dtype)).unwrap()
    };
    
    stats("attn_q.weight", &dequant(&backend, &load_tensor("blk.0.attn_q.weight")));
    stats("attn_k.weight", &dequant(&backend, &load_tensor("blk.0.attn_k.weight")));
    stats("attn_v.weight", &dequant(&backend, &load_tensor("blk.0.attn_v.weight")));
    
    println!("\n=== Analysis ===");
    println!("Q/K biases have large magnitudes (up to ~130 in absolute value)");
    println!("This is normal for Qwen2 - the biases encode learned positional patterns");
    println!("But these large values need to be handled correctly in attention");
}
