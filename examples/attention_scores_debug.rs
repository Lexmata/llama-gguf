//! Debug attention scores across multiple positions to understand bias impact.

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

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    // Config
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Tokens: "1+1=" = [16, 10, 16, 28]
    let tokens: Vec<u32> = vec![16, 10, 16, 28];

    // Load layer 0 weights
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));

    println!("=== Attention Scores Analysis (Layer 0) ===\n");

    // Compute Q, K, V for each token
    let mut all_q: Vec<Vec<f32>> = Vec::new();
    let mut all_k: Vec<Vec<f32>> = Vec::new();
    let mut all_v: Vec<Vec<f32>> = Vec::new();

    for (pos, &tok) in tokens.iter().enumerate() {
        // Get embedding
        let start = tok as usize * hidden_size;
        let h: Vec<f32> = emb[start..start + hidden_size].to_vec();
        
        // RMS norm
        let normed = rms_norm(&h, &attn_norm_w, eps);
        
        // Q, K, V projections
        let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        // Add biases
        if let Some(ref bias) = q_bias {
            for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        
        // Apply RoPE
        for head in 0..num_heads {
            let offset = head * head_dim;
            apply_rope(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
        }
        for head in 0..num_kv_heads {
            let offset = head * head_dim;
            apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
        }
        
        all_q.push(q);
        all_k.push(k);
        all_v.push(v);
    }

    // Analyze attention scores at position 3 (4 tokens to attend to)
    let pos = 3;
    let q = &all_q[pos];
    
    println!("Position 3: attending to 4 tokens (\"1\", \"+\", \"1\", \"=\")");
    println!("scale = {}", scale);
    println!();

    let queries_per_kv = num_heads / num_kv_heads;
    
    for head in 0..3 {  // Just first 3 heads
        let kv_head = head / queries_per_kv;
        let q_offset = head * head_dim;
        let q_vec = &q[q_offset..q_offset + head_dim];
        
        println!("Head {} (KV head {}):", head, kv_head);
        
        let mut scores = vec![0.0f32; 4];
        for kv_pos in 0..4 {
            let k_vec = &all_k[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim];
            let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
            scores[kv_pos] = dot * scale;
        }
        
        println!("  Raw scores (scaled): {:?}", scores.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
        
        // Softmax
        let mut probs = scores.clone();
        softmax(&mut probs);
        println!("  Softmax probs: {:?}", probs.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
        
        // Compute V @ probs weighted sum
        let mut attn_out = vec![0.0f32; head_dim];
        for kv_pos in 0..4 {
            let v_vec = &all_v[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim];
            for d in 0..head_dim {
                attn_out[d] += probs[kv_pos] * v_vec[d];
            }
        }
        let ao_sum: f32 = attn_out.iter().sum();
        let ao_sum_sq: f32 = attn_out.iter().map(|x| x * x).sum();
        println!("  Attention out: sum={:.4}, sum_sq={:.4}", ao_sum, ao_sum_sq);
        println!();
    }

    // Compare score differences
    println!("=== Score Analysis ===");
    println!();
    
    let head = 0;
    let kv_head = 0;
    let q_vec = &q[head * head_dim..(head + 1) * head_dim];
    
    // What would scores be WITHOUT biases?
    println!("Without biases:");
    let q_no_bias = &all_q[pos].iter().enumerate()
        .map(|(i, &v)| v - q_bias.as_ref().map(|b| b[i]).unwrap_or(0.0))
        .collect::<Vec<_>>()[head * head_dim..(head + 1) * head_dim];
    
    for kv_pos in 0..4 {
        let k_no_bias: Vec<f32> = all_k[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim]
            .iter()
            .enumerate()
            .map(|(i, &v)| v - k_bias.as_ref().map(|b| b[i]).unwrap_or(0.0))
            .collect();
        
        let dot: f32 = q_no_bias.iter().zip(k_no_bias.iter()).map(|(a, b)| a * b).sum();
        println!("  Pos {}: Q@K = {:.2}, scaled = {:.2}", kv_pos, dot, dot * scale);
    }
    println!();
    
    println!("Bias contribution:");
    if let (Some(qb), Some(kb)) = (&q_bias, &k_bias) {
        let qb_head = &qb[head * head_dim..(head + 1) * head_dim];
        let kb_head = &kb[kv_head * head_dim..(kv_head + 1) * head_dim];
        
        let bias_dot: f32 = qb_head.iter().zip(kb_head.iter()).map(|(a, b)| a * b).sum();
        println!("  Q_bias @ K_bias = {:.2} (constant across positions)", bias_dot);
        println!("  Scaled = {:.2}", bias_dot * scale);
    }
}
