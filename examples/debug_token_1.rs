//! Debug why token "1" (ID 16) causes attention issues.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn load_tensor(gguf: &GgufFile, name: &str) -> Tensor {
    let tensor_info = gguf.data.get_tensor(name).expect(&format!("No tensor: {}", name));
    let tensor_data = gguf.tensor_data(name).expect(&format!("No data: {}", name));
    let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
    let dtype = DType::from(tensor_info.dtype);
    Tensor::new(tensor_data.to_vec(), shape, dtype).expect("Failed to create tensor")
}

fn try_load_tensor(gguf: &GgufFile, name: &str) -> Option<Tensor> {
    let tensor_info = gguf.data.get_tensor(name)?;
    let tensor_data = gguf.tensor_data(name)?;
    let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
    let dtype = DType::from(tensor_info.dtype);
    Tensor::new(tensor_data.to_vec(), shape, dtype).ok()
}

fn dequant(backend: &CpuBackend, t: &Tensor) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        t.as_f32().unwrap().to_vec()
    } else {
        let numel = t.numel();
        let mut out = Tensor::zeros(vec![numel], DType::F32);
        backend.dequantize(t, &mut out).expect("dequant failed");
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
        let idx0 = 2 * i;
        let idx1 = 2 * i + 1;
        let x0 = data[idx0];
        let x1 = data[idx1];
        data[idx0] = x0 * cos_t - x1 * sin_t;
        data[idx1] = x0 * sin_t + x1 * cos_t;
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

fn stats(x: &[f32]) -> (f32, f32, f32) {
    let min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    (min, max, mean)
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

    println!("=== Token '1' vs '+' K/V Analysis ===");
    println!();

    // Load weights
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));
    
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));

    let tokens = [
        (16u32, "1"),
        (10, "+"),
        (28, "="),
    ];

    println!("K/V values at position 0 for different tokens:");
    println!("----------------------------------------------");
    
    for (tok_id, tok_str) in &tokens {
        let emb_vec = &emb[*tok_id as usize * hidden_size..(*tok_id as usize + 1) * hidden_size];
        let normed = rms_norm(emb_vec, &attn_norm_w, eps);
        
        // K/V projections
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        // Apply biases
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        
        // Apply RoPE at position 0 (identity)
        for kv_head in 0..num_kv_heads {
            let offset = kv_head * head_dim;
            apply_rope(&mut k[offset..offset + head_dim], 0, head_dim, freq_base);
        }
        
        // Apply K bias after RoPE
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        
        let (k_min, k_max, k_mean) = stats(&k);
        let (v_min, v_max, v_mean) = stats(&v);
        
        println!("Token '{}' (ID {})", tok_str, tok_id);
        println!("  K: min={:8.4}, max={:8.4}, mean={:8.4}", k_min, k_max, k_mean);
        println!("  V: min={:8.4}, max={:8.4}, mean={:8.4}", v_min, v_max, v_mean);
        println!("  K first 5 (KV head 0): {:?}", &k[..5]);
        println!("  V first 5 (KV head 0): {:?}", &v[..5]);
        println!();
    }

    // Now simulate what happens when we have "1=" vs "+="
    println!("=== Attention Simulation: X= (where X is first token) ===");
    println!();

    for (first_tok_id, first_tok_str) in &tokens[..2] {
        println!("Sequence: '{}=' (tokens [{}, 28])", first_tok_str, first_tok_id);
        
        // Position 0: compute K, V for first token
        let emb0 = &emb[*first_tok_id as usize * hidden_size..(*first_tok_id as usize + 1) * hidden_size];
        let normed0 = rms_norm(emb0, &attn_norm_w, eps);
        
        let mut k0 = vec_mat(&normed0, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v0 = vec_mat(&normed0, &wv, hidden_size, num_kv_heads * head_dim);
        
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v0.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        for kv_head in 0..num_kv_heads {
            let offset = kv_head * head_dim;
            apply_rope(&mut k0[offset..offset + head_dim], 0, head_dim, freq_base);
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k0.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        
        // Position 1: compute Q, K, V for "="
        let emb1 = &emb[28 * hidden_size..29 * hidden_size];
        let normed1 = rms_norm(emb1, &attn_norm_w, eps);
        
        let mut q1 = vec_mat(&normed1, &wq, hidden_size, num_heads * head_dim);
        let mut k1 = vec_mat(&normed1, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v1 = vec_mat(&normed1, &wv, hidden_size, num_kv_heads * head_dim);
        
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v1.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        
        // Apply RoPE at position 1
        for head in 0..num_heads {
            let offset = head * head_dim;
            apply_rope(&mut q1[offset..offset + head_dim], 1, head_dim, freq_base);
        }
        for kv_head in 0..num_kv_heads {
            let offset = kv_head * head_dim;
            apply_rope(&mut k1[offset..offset + head_dim], 1, head_dim, freq_base);
        }
        
        // Apply Q/K bias after RoPE
        if let Some(ref bias) = q_bias {
            for (qi, bi) in q1.iter_mut().zip(bias.iter()) { *qi += *bi; }
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k1.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        
        // Compute attention for head 0 (uses KV head 0)
        let q_head0 = &q1[0..head_dim];
        let k0_head0 = &k0[0..head_dim];
        let k1_head0 = &k1[0..head_dim];
        let v0_head0 = &v0[0..head_dim];
        let v1_head0 = &v1[0..head_dim];
        
        let score0: f32 = q_head0.iter().zip(k0_head0.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
        let score1: f32 = q_head0.iter().zip(k1_head0.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
        
        let mut scores = vec![score0, score1];
        println!("  Head 0 scores: [{:.4}, {:.4}]", score0, score1);
        
        softmax(&mut scores);
        println!("  Head 0 weights: [{:.4}, {:.4}]", scores[0], scores[1]);
        
        // Weighted sum of V
        let mut attn_out: Vec<f32> = vec![0.0; head_dim];
        for d in 0..head_dim {
            attn_out[d] = scores[0] * v0_head0[d] + scores[1] * v1_head0[d];
        }
        
        let (out_min, out_max, out_mean) = stats(&attn_out);
        println!("  Head 0 output: min={:.4}, max={:.4}, mean={:.4}", out_min, out_max, out_mean);
        println!("  Output first 5: {:?}", &attn_out[..5]);
        println!();
    }
}
