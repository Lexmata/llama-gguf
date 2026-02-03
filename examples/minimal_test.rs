//! Minimal test: just embedding -> output projection (no layers)
//! This tests if our weight loading and matrix multiply are correct.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn load_tensor(gguf: &GgufFile, name: &str) -> Tensor {
    let info = gguf.data.get_tensor(name).unwrap();
    let data = gguf.tensor_data(name).unwrap();
    let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    Tensor::new(data.to_vec(), shape, DType::from(info.dtype)).unwrap()
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

fn vec_mat(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for j in 0..n {
        for i in 0..k {
            out[j] += x[i] * w[i + j * k];
        }
    }
    out
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter().zip(w.iter()).map(|(v, wt)| v * inv_rms * wt).collect()
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    // Load weights
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let output_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let output_w = dequant(&backend, &load_tensor(&gguf, "output.weight"));
    
    println!("Embedding dequantized size: {}", emb.len());
    println!("Output weight dequantized size: {}", output_w.len());
    
    let hidden_size = 896;
    let vocab_size = 151936;
    let eps = 1e-6f32;
    
    // Test with token 28 ("=")
    let token = 28u32;
    println!("\n=== Testing minimal path for token {} ('=') ===", token);
    
    // 1. Get embedding
    let embedding = &emb[token as usize * hidden_size..(token as usize + 1) * hidden_size];
    println!("Embedding stats:");
    println!("  First 5: {:?}", &embedding[..5]);
    println!("  Sum: {:.6}", embedding.iter().sum::<f32>());
    
    // 2. Skip all layers - just apply final norm directly
    // (In real model, hidden state goes through 24 layers first)
    let normed = rms_norm(embedding, &output_norm_w, eps);
    println!("\nAfter output_norm (direct from embedding):");
    println!("  First 5: {:?}", &normed[..5]);
    println!("  Sum: {:.6}", normed.iter().sum::<f32>());
    
    // 3. Project to logits
    let logits = vec_mat(&normed, &output_w, hidden_size, vocab_size);
    
    // 4. Check results
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nLogits (skipping all transformer layers):");
    println!("  Min: {:.4}", logits.iter().cloned().fold(f32::INFINITY, f32::min));
    println!("  Max: {:.4}", logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("  Mean: {:.4}", logits.iter().sum::<f32>() / logits.len() as f32);
    
    println!("\nTop 5 predictions:");
    for (idx, logit) in indexed.iter().take(5) {
        println!("  Token {}: {:.4}", idx, logit);
    }
    
    // Check numeric tokens
    println!("\nNumeric tokens:");
    for i in 16..21 {
        let rank = indexed.iter().position(|(idx, _)| *idx == i).unwrap() + 1;
        println!("  Token {} ('{}'), rank {}: {:.4}", i, i - 15, rank, logits[i]);
    }
    
    // Now test if THIS matches what llama-cpp would produce if we skip layers
    // (Of course llama-cpp doesn't have a skip-layers option, so we can't directly compare)
    // But we can verify the computation is sane
    
    println!("\n=== Verifying output projection weights ===");
    // The logit for token j is: sum_i(normed[i] * output_w[j * 896 + i])
    // Let's manually compute for token 17
    let logit_17_manual: f32 = normed.iter()
        .zip(&output_w[17 * hidden_size..18 * hidden_size])
        .map(|(n, w)| n * w)
        .sum();
    println!("Manually computed logit for token 17: {:.4}", logit_17_manual);
    println!("Vec_mat computed logit for token 17: {:.4}", logits[17]);
    
    if (logit_17_manual - logits[17]).abs() < 1e-4 {
        println!("✓ Manual computation matches vec_mat");
    } else {
        println!("✗ Mismatch between manual and vec_mat!");
    }
    
    // Also verify that our understanding of the weight layout is correct
    // by checking that output_w has the expected structure
    println!("\n=== Output weight verification ===");
    // If layout is [vocab, hidden], then output_w[j * 896 : (j+1) * 896] = weights for vocab j
    // Sum of weights for vocab 0
    let sum_vocab_0: f32 = output_w[0..hidden_size].iter().sum();
    // Sum of weights for vocab 17
    let sum_vocab_17: f32 = output_w[17 * hidden_size..18 * hidden_size].iter().sum();
    println!("Sum of weights for vocab 0: {:.6}", sum_vocab_0);
    println!("Sum of weights for vocab 17: {:.6}", sum_vocab_17);
    println!("(Expected from Python: vocab 0 ~= 0.227, vocab 17 ~= 0.220)");
}
