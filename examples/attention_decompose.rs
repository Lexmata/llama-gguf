//! Decompose attention scores into bias components.
//!
//! score[i,j] = (Q[i] + Q_bias) @ (K[j] + K_bias) * scale
//!            = Q[i] @ K[j] * scale           -- content-content (varies by i,j)
//!            + Q[i] @ K_bias * scale         -- content query x bias key (varies by i, same for all j)
//!            + Q_bias @ K[j] * scale         -- bias query x content key (varies by j, same for all i) 
//!            + Q_bias @ K_bias * scale       -- bias-bias (constant)
//!
//! For softmax, adding a constant (same for all j) doesn't change the result.
//! So the effective score is:
//!   Q[i] @ K[j] + Q_bias @ K[j] = (Q[i] + Q_bias) @ K[j]
//!
//! But wait, that's just what we compute anyway. So the biases should work correctly.
//! Let me verify by computing WITHOUT biases and seeing if the softmax ordering is preserved.

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

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    exp_scores.iter().map(|&e| e / sum).collect()
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
    
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias").map(|t| dequant(&backend, &t));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias").map(|t| dequant(&backend, &t));

    println!("=== Attention Score Decomposition (Layer 0, Head 0, Position 3) ===\n");

    // Compute Q, K for each position (without and with biases)
    let mut q_no_bias: Vec<Vec<f32>> = Vec::new();
    let mut k_no_bias: Vec<Vec<f32>> = Vec::new();
    let mut q_with_bias: Vec<Vec<f32>> = Vec::new();
    let mut k_with_bias: Vec<Vec<f32>> = Vec::new();

    for (pos, &tok) in tokens.iter().enumerate() {
        let start = tok as usize * hidden_size;
        let h: Vec<f32> = emb[start..start + hidden_size].to_vec();
        let normed = rms_norm(&h, &attn_norm_w, eps);
        
        // Without bias
        let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        
        // Apply RoPE (before bias addition, to match what we should do)
        for head in 0..num_heads {
            let offset = head * head_dim;
            apply_rope(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
        }
        for head in 0..num_kv_heads {
            let offset = head * head_dim;
            apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
        }
        
        q_no_bias.push(q.clone());
        k_no_bias.push(k.clone());
        
        // With bias (add after RoPE)
        if let Some(ref bias) = q_bias {
            for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        
        q_with_bias.push(q);
        k_with_bias.push(k);
    }

    let query_pos = 3;
    let head = 0;
    let kv_head = 0;

    println!("Query position: {}", query_pos);
    println!("Head: {} (KV head: {})", head, kv_head);
    println!();

    // Decompose scores for query at position 3
    let q_vec_no_bias = &q_no_bias[query_pos][head * head_dim..(head + 1) * head_dim];
    let q_vec_with_bias = &q_with_bias[query_pos][head * head_dim..(head + 1) * head_dim];
    
    let qb = q_bias.as_ref().map(|b| &b[head * head_dim..(head + 1) * head_dim]);
    let kb = k_bias.as_ref().map(|b| &b[kv_head * head_dim..(kv_head + 1) * head_dim]);

    println!("Score decomposition:");
    println!("  score[q,k] = Q @ K + Q @ Kb + Qb @ K + Qb @ Kb");
    println!();

    let mut scores_no_bias = vec![0.0f32; 4];
    let mut scores_with_bias = vec![0.0f32; 4];
    let mut q_dot_kb = vec![0.0f32; 4];  // same for all key positions
    let mut qb_dot_k = vec![0.0f32; 4];
    let qb_dot_kb: f32;

    // Q @ K_bias (same for all key positions)
    if let Some(kb) = kb {
        let dot: f32 = q_vec_no_bias.iter().zip(kb.iter()).map(|(a, b)| a * b).sum();
        for v in q_dot_kb.iter_mut() { *v = dot; }
        println!("Q @ K_bias = {:.2} (constant for all key positions)", dot * scale);
    }

    // Q_bias @ K_bias
    if let (Some(qb), Some(kb)) = (qb, kb) {
        qb_dot_kb = qb.iter().zip(kb.iter()).map(|(a, b)| a * b).sum();
        println!("Q_bias @ K_bias = {:.2} (constant)", qb_dot_kb * scale);
    } else {
        qb_dot_kb = 0.0;
    }
    println!();

    println!("{:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "kpos", "Q@K", "Q@Kb", "Qb@K", "Qb@Kb", "Total", "NoBias");
    println!("{:-<5}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", "", "", "");

    for kv_pos in 0..4 {
        let k_vec_no_bias = &k_no_bias[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim];
        let k_vec_with_bias = &k_with_bias[kv_pos][kv_head * head_dim..(kv_head + 1) * head_dim];

        // Q @ K (content-content)
        let q_dot_k: f32 = q_vec_no_bias.iter().zip(k_vec_no_bias.iter()).map(|(a, b)| a * b).sum();
        
        // Q_bias @ K (varies by key position)
        let qb_dot_k_val: f32 = if let Some(qb) = qb {
            qb.iter().zip(k_vec_no_bias.iter()).map(|(a, b)| a * b).sum()
        } else {
            0.0
        };
        qb_dot_k[kv_pos] = qb_dot_k_val;

        scores_no_bias[kv_pos] = q_dot_k * scale;
        scores_with_bias[kv_pos] = (q_dot_k + q_dot_kb[kv_pos] + qb_dot_k_val + qb_dot_kb) * scale;

        println!("{:>5} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2}",
            kv_pos,
            q_dot_k * scale,
            q_dot_kb[kv_pos] * scale,
            qb_dot_k_val * scale,
            qb_dot_kb * scale,
            scores_with_bias[kv_pos],
            scores_no_bias[kv_pos]);
    }

    println!();
    println!("Softmax comparison:");
    
    let probs_no_bias = softmax(&scores_no_bias);
    let probs_with_bias = softmax(&scores_with_bias);
    
    println!("  Without bias: {:?}", probs_no_bias.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("  With bias:    {:?}", probs_with_bias.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    
    // Check: do they match?
    let diff: f32 = probs_no_bias.iter().zip(probs_with_bias.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("  Total abs diff: {:.6}", diff);

    println!();
    println!("=== Analysis ===");
    println!();
    println!("The 'Qb@K' term varies by key position and is NOT cancelled by softmax.");
    println!("This is expected behavior - the bias affects attention patterns.");
    println!();
    
    // Check if this causes issues
    let max_qb_k_diff = qb_dot_k.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        - qb_dot_k.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("Qb@K range: {:.2} (difference between key positions)", max_qb_k_diff * scale);
    
    let max_qk_diff = scores_no_bias.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        - scores_no_bias.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("Q@K range:  {:.2} (content-based differences)", max_qk_diff);
    
    println!();
    if max_qb_k_diff * scale > max_qk_diff * 10.0 {
        println!("WARNING: Bias-content interaction dominates content-content interaction!");
        println!("This may explain why our implementation produces different results.");
    }
}
