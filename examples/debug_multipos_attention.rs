//! Debug multi-position attention to find why single token works but multi-token doesn't.

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

    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Just first layer
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias").map(|t| dequant(&backend, &t));

    // Embedding
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));

    // Tokens: "1=" (tokens 16, 28)
    let tokens: Vec<u32> = vec![16, 28];
    let embeddings: Vec<Vec<f32>> = tokens.iter()
        .map(|&tok| emb[tok as usize * hidden_size..(tok as usize + 1) * hidden_size].to_vec())
        .collect();

    println!("=== Multi-Position Attention Debug (Layer 0) ===");
    println!("Tokens: {:?}", tokens);
    println!("num_heads={}, head_dim={}, num_kv_heads={}", num_heads, head_dim, num_kv_heads);
    println!("queries_per_kv_head={}", num_heads / num_kv_heads);
    println!();

    // Store K and V for each position (only storing kv_head 0 for simplicity)
    let mut k_cache: Vec<Vec<f32>> = Vec::new();
    let mut v_cache: Vec<Vec<f32>> = Vec::new();

    // Process each position
    for (pos, emb_vec) in embeddings.iter().enumerate() {
        println!("=== Position {} ===", pos);

        // Normalize
        let normed = rms_norm(emb_vec, &attn_norm_w, eps);
        println!("  Normed first 5: {:?}", &normed[..5]);

        // Q/K/V projections (no bias yet)
        let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        // Apply V bias directly
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        
        println!("  Q (before RoPE) first 5: {:?}", &q[..5]);
        println!("  K (before RoPE) first 5: {:?}", &k[..5]);

        // Apply RoPE to each head
        for head in 0..num_heads {
            let offset = head * head_dim;
            apply_rope(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
        }
        for kv_head in 0..num_kv_heads {
            let offset = kv_head * head_dim;
            apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
        }

        println!("  Q (after RoPE) first 5: {:?}", &q[..5]);
        println!("  K (after RoPE) first 5: {:?}", &k[..5]);

        // Apply Q/K biases AFTER RoPE
        if let Some(ref bias) = q_bias {
            for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }

        println!("  Q (after bias) first 5: {:?}", &q[..5]);
        println!("  K (after bias) first 5: {:?}", &k[..5]);
        println!("  V first 5: {:?}", &v[..5]);

        // Store K and V (just KV head 0)
        k_cache.push(k[0..head_dim].to_vec());
        v_cache.push(v[0..head_dim].to_vec());

        // Now compute attention for head 0 (which uses KV head 0)
        let q_head0 = &q[0..head_dim];
        
        println!();
        println!("  === Attention (Head 0) ===");
        
        // Compute attention scores for all positions 0..=pos
        let kv_len = pos + 1;
        let mut scores = vec![0.0f32; kv_len];
        
        for kv_pos in 0..kv_len {
            let k_vec = &k_cache[kv_pos];
            let dot: f32 = q_head0.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
            scores[kv_pos] = dot * scale;
            println!("    Score[{}] = Q @ K = {:.4} * scale = {:.4}", kv_pos, dot, scores[kv_pos]);
        }
        
        // Softmax
        println!();
        println!("  Raw scores: {:?}", scores);
        softmax(&mut scores);
        println!("  After softmax: {:?}", scores);
        
        // Weighted sum of values
        let mut attn_out = vec![0.0f32; head_dim];
        for kv_pos in 0..kv_len {
            let v_vec = &v_cache[kv_pos];
            for d in 0..head_dim {
                attn_out[d] += scores[kv_pos] * v_vec[d];
            }
        }
        println!("  Attention output first 5: {:?}", &attn_out[..5]);
        println!();
    }

    // Now let's compare what happens when we run just "=" (single token)
    println!("=== Single Token Comparison ===");
    let single_emb = &embeddings[1]; // Token "="
    let normed = rms_norm(single_emb, &attn_norm_w, eps);
    
    let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
    let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
    let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
    
    if let Some(ref bias) = v_bias {
        for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
    }

    // Apply RoPE at position 0 (as if it's the only token)
    for head in 0..num_heads {
        let offset = head * head_dim;
        apply_rope(&mut q[offset..offset + head_dim], 0, head_dim, freq_base);
    }
    for kv_head in 0..num_kv_heads {
        let offset = kv_head * head_dim;
        apply_rope(&mut k[offset..offset + head_dim], 0, head_dim, freq_base);
    }

    if let Some(ref bias) = q_bias {
        for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
    }
    if let Some(ref bias) = k_bias {
        for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
    }

    println!("Single '=' at pos 0:");
    println!("  Q (after RoPE+bias) first 5: {:?}", &q[..5]);
    println!("  K (after RoPE+bias) first 5: {:?}", &k[..5]);
    
    // With only 1 position, attention is trivially 100% on position 0
    // Output = V[0]
    println!("  Attention output (single pos) first 5: {:?}", &v[..5]);
    
    // Compare with multi-position case
    println!();
    println!("Multi-pos '1=' final attention output at pos 1:");
    println!("  This attends to both position 0 ('1') and position 1 ('=')");
    println!("  The RoPE at position 1 changes the Q/K values significantly");
    println!();

    // Key insight: RoPE at different positions creates different Q/K values
    // So the same token "=" gets different Q/K when at position 0 vs position 1
    println!("=== RoPE Effect ===");
    let normed_eq = rms_norm(&embeddings[1], &attn_norm_w, eps); // "=" embedding
    let mut q0 = vec_mat(&normed_eq, &wq, hidden_size, num_heads * head_dim);
    let mut q1 = q0.clone();
    
    apply_rope(&mut q0[0..head_dim], 0, head_dim, freq_base);
    apply_rope(&mut q1[0..head_dim], 1, head_dim, freq_base);
    
    println!("Q for '=' at position 0, first 5: {:?}", &q0[..5]);
    println!("Q for '=' at position 1, first 5: {:?}", &q1[..5]);
    
    let diff: Vec<f32> = q0[..5].iter().zip(q1[..5].iter()).map(|(a, b)| a - b).collect();
    println!("Difference (pos0 - pos1): {:?}", diff);
}
