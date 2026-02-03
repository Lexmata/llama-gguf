//! Debug just layer 0 in detail

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
        for i in 0..k {
            out[j] += x[i] * w[i + j * k];
        }
    }
    out
}

fn apply_rope_neox(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = data[i];
        let x1 = data[i + half_dim];
        data[i] = x0 * cos_t - x1 * sin_t;
        data[i + half_dim] = x0 * sin_t + x1 * cos_t;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn stats(name: &str, data: &[f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    println!("{}: min={:.6}, max={:.6}, sum={:.6}, mean={:.6}", name, min, max, sum, mean);
    println!("  First 5: {:?}", &data[..5.min(data.len())].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let queries_per_kv = num_heads / num_kv_heads;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    
    // Token 28 ("="), position 0
    let token = 28u32;
    let pos = 0usize;
    
    println!("=== Debugging layer 0 for token {} ('=') at position {} ===\n", token, pos);
    
    // Get embedding
    let hidden = emb[token as usize * hidden_size..(token as usize + 1) * hidden_size].to_vec();
    stats("Input (embedding)", &hidden);
    
    // Load layer 0 weights
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    let wo = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_output.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));
    
    let ffn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_norm.weight"));
    let w_gate = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_gate.weight"));
    let w_up = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_up.weight"));
    let w_down = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_down.weight"));
    
    println!("\n--- Attention ---");
    
    // Attention norm
    let normed = rms_norm(&hidden, &attn_norm_w, eps);
    stats("After attn_norm", &normed);
    
    // QKV projections
    let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
    stats("Q (before bias/RoPE)", &q);
    
    let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
    stats("K (before bias/RoPE)", &k);
    
    let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
    
    // Apply biases BEFORE RoPE
    if let Some(ref bias) = q_bias {
        for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
        stats("Q (after bias, before RoPE)", &q);
    }
    if let Some(ref bias) = k_bias {
        for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        stats("K (after bias, before RoPE)", &k);
    }
    if let Some(ref bias) = v_bias {
        for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
    }
    stats("V (after bias)", &v);
    
    // Apply RoPE
    for head in 0..num_heads {
        apply_rope_neox(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base);
    }
    for kv_head in 0..num_kv_heads {
        apply_rope_neox(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
    }
    stats("Q (after RoPE)", &q);
    stats("K (after RoPE)", &k);
    
    // For single token at position 0:
    // - Attention scores: Q * K^T / sqrt(d)
    // - Softmax of single value = 1.0
    // - Output = V (weighted by softmax, which is just 1.0)
    
    // Actually, let me compute this properly
    let mut attn_out = vec![0.0f32; num_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    for head in 0..num_heads {
        let kv_head = head / queries_per_kv;
        let q_vec = &q[head * head_dim..(head + 1) * head_dim];
        let k_vec = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
        let v_vec = &v[kv_head * head_dim..(kv_head + 1) * head_dim];
        
        // Score = Q * K / sqrt(d)
        let score: f32 = q_vec.iter().zip(k_vec).map(|(qi, ki)| qi * ki).sum::<f32>() * scale;
        
        // Single position softmax = 1.0
        // Output = 1.0 * V
        for d in 0..head_dim {
            attn_out[head * head_dim + d] = v_vec[d];
        }
        
        if head == 0 {
            println!("Head 0: score={:.6} (softmax=1.0)", score);
        }
    }
    stats("Attention output (concat of V)", &attn_out);
    
    // Output projection
    let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
    stats("Attention projection", &attn_proj);
    
    // Residual
    let h: Vec<f32> = hidden.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();
    stats("After attention residual", &h);
    
    println!("\n--- FFN ---");
    
    // FFN norm
    let ffn_normed = rms_norm(&h, &ffn_norm_w, eps);
    stats("After ffn_norm", &ffn_normed);
    
    // Gate and Up
    let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
    let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
    stats("Gate", &gate);
    stats("Up", &up);
    
    // SiLU(gate) * up
    let intermediate: Vec<f32> = gate.iter().zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    stats("SiLU(gate)*up", &intermediate);
    
    // Down projection
    let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);
    stats("FFN output", &ffn_out);
    
    // Final residual
    let final_h: Vec<f32> = h.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
    stats("\nLayer 0 output", &final_h);
}
