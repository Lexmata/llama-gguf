//! Test whether bias should be applied BEFORE or AFTER RoPE.
//! 
//! This tests the hypothesis that llama.cpp applies bias BEFORE RoPE
//! (i.e., Q = RoPE(Wx + b)), while our implementation applies bias AFTER.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::model::{InferenceContext, Model, ModelLoader};
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;
use std::sync::Arc;

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

fn apply_rope(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn run_full_forward(
    tokens: &[u32],
    gguf: &GgufFile,
    backend: &CpuBackend,
    emb: &[f32],
    output_norm_w: &[f32],
    output_w: &[f32],
    eps: f32,
    freq_base: f32,
    bias_before_rope: bool,  // Key parameter!
) -> Vec<f32> {
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let queries_per_kv = num_heads / num_kv_heads;
    let max_seq_len = 512;
    let n_layers = 24;
    
    let mut k_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
        n_layers
    ];
    let mut v_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
        n_layers
    ];
    
    let mut final_hidden = vec![];
    
    for (pos, &token) in tokens.iter().enumerate() {
        let mut hidden = emb[token as usize * hidden_size..(token as usize + 1) * hidden_size].to_vec();
        
        for layer in 0..n_layers {
            let prefix = format!("blk.{}", layer);
            
            let attn_norm_w = dequant(backend, &load_tensor(gguf, &format!("{}.attn_norm.weight", prefix)));
            let wq = dequant(backend, &load_tensor(gguf, &format!("{}.attn_q.weight", prefix)));
            let wk = dequant(backend, &load_tensor(gguf, &format!("{}.attn_k.weight", prefix)));
            let wv = dequant(backend, &load_tensor(gguf, &format!("{}.attn_v.weight", prefix)));
            let wo = dequant(backend, &load_tensor(gguf, &format!("{}.attn_output.weight", prefix)));
            
            let q_bias = try_load_tensor(gguf, &format!("{}.attn_q.bias", prefix)).map(|t| dequant(backend, &t));
            let k_bias = try_load_tensor(gguf, &format!("{}.attn_k.bias", prefix)).map(|t| dequant(backend, &t));
            let v_bias = try_load_tensor(gguf, &format!("{}.attn_v.bias", prefix)).map(|t| dequant(backend, &t));
            
            let ffn_norm_w = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_norm.weight", prefix)));
            let w_gate = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_gate.weight", prefix)));
            let w_up = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_up.weight", prefix)));
            let w_down = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_down.weight", prefix)));
            
            let normed = rms_norm(&hidden, &attn_norm_w, eps);
            let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
            let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
            let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
            
            // Always apply V bias immediately (no RoPE for V)
            if let Some(ref bias) = v_bias { 
                for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; } 
            }
            
            if bias_before_rope {
                // OPTION A: Apply bias BEFORE RoPE (like llama.cpp might do)
                if let Some(ref bias) = q_bias {
                    for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
                }
                if let Some(ref bias) = k_bias {
                    for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
                }
                
                // Then apply RoPE
                for head in 0..num_heads {
                    apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base);
                }
                for kv_head in 0..num_kv_heads {
                    apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
                }
            } else {
                // OPTION B: Apply RoPE BEFORE bias (our current implementation)
                for head in 0..num_heads {
                    apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base);
                }
                for kv_head in 0..num_kv_heads {
                    apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
                }
                
                // Then apply bias
                if let Some(ref bias) = q_bias {
                    for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
                }
                if let Some(ref bias) = k_bias {
                    for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
                }
            }
            
            // Store K/V in cache
            for kv_head in 0..num_kv_heads {
                k_caches[layer][kv_head][pos] = k[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec();
                v_caches[layer][kv_head][pos] = v[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec();
            }
            
            // Compute attention
            let kv_len = pos + 1;
            let mut attn_out = vec![0.0f32; num_heads * head_dim];
            
            for head in 0..num_heads {
                let kv_head = head / queries_per_kv;
                let q_vec = &q[head * head_dim..(head + 1) * head_dim];
                
                let mut scores = vec![0.0f32; kv_len];
                for kv_pos in 0..kv_len {
                    let k_vec = &k_caches[layer][kv_head][kv_pos];
                    let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                    scores[kv_pos] = dot * scale;
                }
                
                softmax(&mut scores);
                
                for kv_pos in 0..kv_len {
                    let v_vec = &v_caches[layer][kv_head][kv_pos];
                    for d in 0..head_dim {
                        attn_out[head * head_dim + d] += scores[kv_pos] * v_vec[d];
                    }
                }
            }
            
            let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
            let h: Vec<f32> = hidden.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();
            
            let ffn_normed = rms_norm(&h, &ffn_norm_w, eps);
            let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
            let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
            let intermediate: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
            let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);
            
            hidden = h.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
        }
        
        final_hidden = hidden;
    }
    
    // Compute final logits
    let normed_final = rms_norm(&final_hidden, output_norm_w, eps);
    vec_mat(&normed_final, output_w, 896, 151936)
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    let hidden_size = 896;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let output_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let output_w = dequant(&backend, &load_tensor(&gguf, "output.weight"));
    
    // Test with 4 tokens: "1+1="
    let tokens: Vec<u32> = vec![16, 10, 16, 28];  // 1, +, 1, =
    
    println!("Testing bias ordering with {} tokens: {:?}", tokens.len(), tokens);
    println!();
    
    // Option A: Bias BEFORE RoPE
    println!("=== Option A: Bias BEFORE RoPE ===");
    let logits_a = run_full_forward(
        &tokens, &gguf, &backend, &emb, &output_norm_w, &output_w,
        eps, freq_base, true
    );
    
    let mut indexed_a: Vec<(usize, f32)> = logits_a.iter().cloned().enumerate().collect();
    indexed_a.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_a = indexed_a.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    let top5_a: Vec<(usize, f32)> = indexed_a.iter().take(5).cloned().collect();
    
    println!("  Token 17 ('2') rank: {}", rank_a);
    println!("  Token 17 logit: {:.4}", logits_a[17]);
    println!("  Top 5 predictions:");
    for (idx, logit) in &top5_a {
        println!("    Token {} (logit {:.4})", idx, logit);
    }
    println!();
    
    // Option B: Bias AFTER RoPE (our current implementation)
    println!("=== Option B: Bias AFTER RoPE ===");
    let logits_b = run_full_forward(
        &tokens, &gguf, &backend, &emb, &output_norm_w, &output_w,
        eps, freq_base, false
    );
    
    let mut indexed_b: Vec<(usize, f32)> = logits_b.iter().cloned().enumerate().collect();
    indexed_b.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_b = indexed_b.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    let top5_b: Vec<(usize, f32)> = indexed_b.iter().take(5).cloned().collect();
    
    println!("  Token 17 ('2') rank: {}", rank_b);
    println!("  Token 17 logit: {:.4}", logits_b[17]);
    println!("  Top 5 predictions:");
    for (idx, logit) in &top5_b {
        println!("    Token {} (logit {:.4})", idx, logit);
    }
    println!();
    
    // Summary
    println!("=== Summary ===");
    println!("  Bias BEFORE RoPE: Token '2' rank = {}", rank_a);
    println!("  Bias AFTER RoPE:  Token '2' rank = {}", rank_b);
    if rank_a < rank_b {
        println!("\n  ** BIAS BEFORE ROPE is better! **");
    } else if rank_b < rank_a {
        println!("\n  ** BIAS AFTER ROPE is better (current implementation) **");
    } else {
        println!("\n  Both produce the same rank.");
    }
}
