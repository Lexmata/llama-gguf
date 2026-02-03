//! Trace attention output projection to verify it's working correctly.

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

    println!("=== Attention Output Projection Trace ===");
    println!();

    // Load weights
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    let wo = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_output.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));
    
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));

    // Check Wo shape
    let wo_info = gguf.data.get_tensor("blk.0.attn_output.weight").unwrap();
    println!("Wo shape (GGUF): {:?}", wo_info.dims);
    println!("Expected: [{}, {}]", num_heads * head_dim, hidden_size);
    println!();

    // Test with single token "=" at position 0
    let token = 28u32;  // "="
    let emb_vec = &emb[token as usize * hidden_size..(token as usize + 1) * hidden_size];
    let normed = rms_norm(emb_vec, &attn_norm_w, eps);
    
    // Q/K/V projections
    let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
    let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
    let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
    
    // Apply V bias
    if let Some(ref bias) = v_bias {
        for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
    }
    
    // Apply RoPE at position 0 (identity for position 0)
    for head in 0..num_heads {
        let offset = head * head_dim;
        apply_rope(&mut q[offset..offset + head_dim], 0, head_dim, freq_base);
    }
    for kv_head in 0..num_kv_heads {
        let offset = kv_head * head_dim;
        apply_rope(&mut k[offset..offset + head_dim], 0, head_dim, freq_base);
    }
    
    // Apply Q/K biases after RoPE
    if let Some(ref bias) = q_bias {
        for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
    }
    if let Some(ref bias) = k_bias {
        for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
    }
    
    let (q_min, q_max, q_mean) = stats(&q);
    let (k_min, k_max, k_mean) = stats(&k);
    let (v_min, v_max, v_mean) = stats(&v);
    
    println!("After projections (single token '=' at pos 0):");
    println!("  Q: min={:.4}, max={:.4}, mean={:.4}, len={}", q_min, q_max, q_mean, q.len());
    println!("  K: min={:.4}, max={:.4}, mean={:.4}, len={}", k_min, k_max, k_mean, k.len());
    println!("  V: min={:.4}, max={:.4}, mean={:.4}, len={}", v_min, v_max, v_mean, v.len());
    println!();
    
    // With only 1 position, attention output is just V (weighted sum with weight 1.0)
    // But we need to map through GQA
    let queries_per_kv = num_heads / num_kv_heads;
    let mut attn_out = vec![0.0f32; num_heads * head_dim];
    
    for head in 0..num_heads {
        let kv_head = head / queries_per_kv;
        let q_vec = &q[head * head_dim..(head + 1) * head_dim];
        let k_vec = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
        let v_vec = &v[kv_head * head_dim..(kv_head + 1) * head_dim];
        
        // Only one position, attention is 100% on it
        // Score doesn't matter since there's only one position
        let out_offset = head * head_dim;
        attn_out[out_offset..out_offset + head_dim].copy_from_slice(v_vec);
        
        if head < 2 {
            // Compute dot product to verify
            let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
            println!("Head {}: Q @ K dot = {:.4}, scaled = {:.4}", head, dot, dot * scale);
            println!("  V (which becomes attn output): {:?}", &v_vec[..5]);
        }
    }
    
    let (attn_min, attn_max, attn_mean) = stats(&attn_out);
    println!();
    println!("Attention output (before Wo): min={:.4}, max={:.4}, mean={:.4}, len={}", 
        attn_min, attn_max, attn_mean, attn_out.len());
    println!("  First 5: {:?}", &attn_out[..5]);
    println!();
    
    // Apply output projection Wo
    // Wo shape: [num_heads * head_dim, hidden_size]
    // attn_out shape: [num_heads * head_dim]
    // output shape: [hidden_size]
    let projected = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
    
    let (proj_min, proj_max, proj_mean) = stats(&projected);
    println!("After Wo projection: min={:.4}, max={:.4}, mean={:.4}, len={}", 
        proj_min, proj_max, proj_mean, projected.len());
    println!("  First 5: {:?}", &projected[..5]);
    println!();
    
    // Add residual
    let residual: Vec<f32> = emb_vec.iter().zip(projected.iter())
        .map(|(a, b)| a + b)
        .collect();
    
    let (res_min, res_max, res_mean) = stats(&residual);
    println!("After residual connection: min={:.4}, max={:.4}, mean={:.4}", 
        res_min, res_max, res_mean);
    println!("  First 5: {:?}", &residual[..5]);
}
