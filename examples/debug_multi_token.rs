//! Debug multi-token attention

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
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let queries_per_kv = num_heads / num_kv_heads;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Process "1+1=" = [16, 10, 16, 28]
    let tokens = [16u32, 10, 16, 28];
    println!("=== Processing tokens {:?} ===\n", tokens);
    
    // Load layer 0 weights
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));
    
    // Store KV for all positions
    let mut k_cache: Vec<Vec<f32>> = Vec::new();  // [num_kv_heads * head_dim] per position
    let mut v_cache: Vec<Vec<f32>> = Vec::new();
    
    // Process each position
    for (pos, &token) in tokens.iter().enumerate() {
        let hidden = emb[token as usize * hidden_size..(token as usize + 1) * hidden_size].to_vec();
        let normed = rms_norm(&hidden, &attn_norm_w, eps);
        
        let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        // Apply biases BEFORE RoPE
        if let Some(ref bias) = q_bias {
            for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        let v: Vec<f32> = if let Some(ref bias) = v_bias {
            v.iter().zip(bias.iter()).map(|(vi, bi)| vi + bi).collect()
        } else {
            v
        };
        
        // Apply RoPE
        for head in 0..num_heads {
            apply_rope_neox(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base);
        }
        for kv_head in 0..num_kv_heads {
            apply_rope_neox(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
        }
        
        k_cache.push(k);
        v_cache.push(v);
        
        // Show attention scores for head 0 from this position
        if pos == 3 {  // Last position (token 28, "=")
            println!("=== Position {} (token {}) - Head 0 Attention ===", pos, token);
            
            let q_head = &q[0..head_dim];
            let kv_head = 0;  // head 0 uses kv_head 0
            
            let mut scores = Vec::new();
            for kv_pos in 0..=pos {
                let k_head = &k_cache[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim];
                let dot: f32 = q_head.iter().zip(k_head).map(|(qi, ki)| qi * ki).sum();
                scores.push(dot * scale);
            }
            
            println!("Raw scores: {:?}", scores);
            
            softmax(&mut scores);
            println!("Softmax weights: {:?}", scores);
            
            // Weighted V
            let mut attn_out = vec![0.0f32; head_dim];
            for kv_pos in 0..=pos {
                let v_head = &v_cache[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim];
                for d in 0..head_dim {
                    attn_out[d] += scores[kv_pos] * v_head[d];
                }
            }
            
            println!("Attention output sum: {:.4}", attn_out.iter().sum::<f32>());
        }
    }
    
    // Now show what llama-cpp's attention pattern should look like
    println!("\n=== Analysis ===");
    println!("With large K biases, attention scores are dominated by bias terms.");
    println!("The attention pattern may not reflect content similarity at all,");
    println!("but rather learned positional/structural patterns.");
    println!("This is how Qwen2's attention likely works - biases encode structure.");
}
