//! Compare attention patterns between positions.

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

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter().zip(w.iter()).map(|(v, wt)| v * inv_rms * wt).collect()
}

fn vec_mat(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += x[i] * w[i + j * k];
        }
        out[j] = sum;
    }
    out
}

fn apply_rope(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    let position = pos as f32;
    for i in 0..half_dim {
        let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let theta = position * freq;
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * cos_t - x1 * sin_t;
        data[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

fn softmax(scores: &mut [f32]) {
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let queries_per_kv = num_heads / num_kv_heads;  // 7

    println!("=== Attention Pattern Analysis ===");
    println!();
    println!("Configuration:");
    println!("  num_heads = {}, num_kv_heads = {}", num_heads, num_kv_heads);
    println!("  queries_per_kv = {} (heads 0-6 use KV head 0, heads 7-13 use KV head 1)", queries_per_kv);
    println!("  scale = 1/sqrt({}) = {:.6}", head_dim, scale);
    println!();
    
    // Load layer 0 weights
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));

    // Process "1+1=" [16, 10, 16, 28]
    let tokens: Vec<u32> = vec![16, 10, 16, 28];
    
    // Compute K/V for each position (Layer 0 only)
    let mut k_cache: Vec<Vec<f32>> = vec![]; // [pos][kv_head * head_dim]
    let mut v_cache: Vec<Vec<f32>> = vec![];
    
    println!("Layer 0 K/V computation for each position:");
    println!("-----------------------------------------");
    
    for (pos, &token) in tokens.iter().enumerate() {
        let emb_vec = &emb[token as usize * hidden_size..(token as usize + 1) * hidden_size];
        let normed = rms_norm(emb_vec, &attn_norm_w, eps);
        
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        
        // RoPE for K at this position
        for kv_head in 0..num_kv_heads {
            apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
        }
        
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        
        println!("Position {} (token {}): K[0..5]={:.3?}", pos, token, &k[..5]);
        
        k_cache.push(k);
        v_cache.push(v);
    }
    
    println!();
    
    // Now compute Q for the last position and show attention patterns
    let last_pos = tokens.len() - 1;
    let last_token = tokens[last_pos];
    
    let emb_vec = &emb[last_token as usize * hidden_size..(last_token as usize + 1) * hidden_size];
    let normed = rms_norm(emb_vec, &attn_norm_w, eps);
    
    let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
    
    // RoPE for Q at position 3
    for head in 0..num_heads {
        apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], last_pos, head_dim, freq_base);
    }
    
    if let Some(ref bias) = q_bias {
        for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
    }
    
    println!("Q at position {} (token {}):", last_pos, last_token);
    println!("  Q head 0 [0..5] = {:.3?}", &q[..5]);
    println!("  Q head 7 [0..5] = {:.3?}", &q[7*head_dim..7*head_dim+5]);
    println!();
    
    // Compute attention scores for head 0 (uses KV head 0)
    println!("Attention scores for head 0 (uses KV head 0):");
    println!("---------------------------------------------");
    
    let q_head0 = &q[0..head_dim];
    let mut scores = vec![];
    
    for kv_pos in 0..tokens.len() {
        let k_vec = &k_cache[kv_pos][0..head_dim];  // KV head 0
        let dot: f32 = q_head0.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
        let scaled = dot * scale;
        scores.push(scaled);
        println!("  Score[{}] (token {}) = {:.2} * {:.4} = {:.4}", kv_pos, tokens[kv_pos], dot, scale, scaled);
    }
    
    let mut weights = scores.clone();
    softmax(&mut weights);
    println!();
    println!("  Softmax weights: {:?}", weights);
    println!();
    
    // Compute attention scores for head 7 (uses KV head 1)
    println!("Attention scores for head 7 (uses KV head 1):");
    println!("---------------------------------------------");
    
    let q_head7 = &q[7*head_dim..8*head_dim];
    let mut scores7 = vec![];
    
    for kv_pos in 0..tokens.len() {
        let k_vec = &k_cache[kv_pos][head_dim..2*head_dim];  // KV head 1
        let dot: f32 = q_head7.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
        let scaled = dot * scale;
        scores7.push(scaled);
        println!("  Score[{}] (token {}) = {:.2} * {:.4} = {:.4}", kv_pos, tokens[kv_pos], dot, scale, scaled);
    }
    
    let mut weights7 = scores7.clone();
    softmax(&mut weights7);
    println!();
    println!("  Softmax weights: {:?}", weights7);
    println!();
    
    // Analysis
    println!("=== Analysis ===");
    println!();
    println!("For '1+1=', to correctly predict '2', the model should:");
    println!("1. Attend to both '1' tokens (positions 0 and 2)");
    println!("2. Recognize the '+' and '=' operators");
    println!("3. Output an embedding similar to token '2'");
    println!();
    println!("Head 0 attention pattern: {:?}", weights);
    println!("  This head attends mostly to position {} (weight {:.2}%)", 
        weights.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0,
        weights.iter().cloned().fold(0.0f32, f32::max) * 100.0);
    
    println!();
    println!("If the attention is dominated by large biases, the model may not");
    println!("be able to properly distinguish between different inputs.");
}
