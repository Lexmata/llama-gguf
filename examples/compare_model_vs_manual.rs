//! Compare actual model hidden states with manual computation.

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

fn compare(a: &[f32], b: &[f32], name: &str) {
    let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    let max_diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    println!("  {} diff: sum={:.6}, max={:.6}", name, diff, max_diff);
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    let backend_arc: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    
    let loader = ModelLoader::load(model_path).expect("Failed to load model");
    let config = loader.config().clone();
    let model = loader.build_model().expect("Failed to build model");
    
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let queries_per_kv = num_heads / num_kv_heads;
    let max_seq_len = 512;
    
    // Test with single token first
    let token: u32 = 28; // "="
    
    println!("=== Single Token Test (token={}) ===", token);
    println!();
    
    // Run through actual model
    let mut ctx = InferenceContext::new(&config, backend_arc.clone());
    let logits_actual = model.forward(&[token], &mut ctx).expect("Forward failed");
    let logits_actual_data = logits_actual.as_f32().unwrap();
    
    // Manual computation
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let output_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let output_w = dequant(&backend, &load_tensor(&gguf, "output.weight"));
    
    let mut hidden_manual = emb[token as usize * hidden_size..(token as usize + 1) * hidden_size].to_vec();
    
    // Process just layer 0 manually
    {
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
        
        // Attention
        let normed = rms_norm(&hidden_manual, &attn_norm_w, eps);
        let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        if let Some(ref bias) = v_bias {
            for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; }
        }
        
        // RoPE at position 0 (identity)
        for head in 0..num_heads {
            apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], 0, head_dim, freq_base);
        }
        for kv_head in 0..num_kv_heads {
            apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], 0, head_dim, freq_base);
        }
        
        if let Some(ref bias) = q_bias {
            for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
        }
        if let Some(ref bias) = k_bias {
            for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
        }
        
        // Single position attention: output = V (mapped through GQA)
        let mut attn_out = vec![0.0f32; num_heads * head_dim];
        for head in 0..num_heads {
            let kv_head = head / queries_per_kv;
            attn_out[head * head_dim..(head + 1) * head_dim]
                .copy_from_slice(&v[kv_head * head_dim..(kv_head + 1) * head_dim]);
        }
        
        let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
        let h: Vec<f32> = hidden_manual.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();
        
        // FFN
        let ffn_normed = rms_norm(&h, &ffn_norm_w, eps);
        let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
        let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
        let intermediate: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
        let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);
        
        hidden_manual = h.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
    }
    
    // Now extract the actual model's hidden state after layer 0
    // We can't easily do this, so let's compare final logits instead
    
    // Process remaining layers manually
    for layer in 1..24 {
        let prefix = format!("blk.{}", layer);
        
        let attn_norm_w = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_norm.weight", prefix)));
        let wq = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_q.weight", prefix)));
        let wk = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_k.weight", prefix)));
        let wv = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_v.weight", prefix)));
        let wo = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_output.weight", prefix)));
        
        let q_bias = try_load_tensor(&gguf, &format!("{}.attn_q.bias", prefix)).map(|t| dequant(&backend, &t));
        let k_bias = try_load_tensor(&gguf, &format!("{}.attn_k.bias", prefix)).map(|t| dequant(&backend, &t));
        let v_bias = try_load_tensor(&gguf, &format!("{}.attn_v.bias", prefix)).map(|t| dequant(&backend, &t));
        
        let ffn_norm_w = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_norm.weight", prefix)));
        let w_gate = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_gate.weight", prefix)));
        let w_up = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_up.weight", prefix)));
        let w_down = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_down.weight", prefix)));
        
        let normed = rms_norm(&hidden_manual, &attn_norm_w, eps);
        let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
        let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
        let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
        
        if let Some(ref bias) = v_bias { for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; } }
        for head in 0..num_heads { apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], 0, head_dim, freq_base); }
        for kv_head in 0..num_kv_heads { apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], 0, head_dim, freq_base); }
        if let Some(ref bias) = q_bias { for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; } }
        if let Some(ref bias) = k_bias { for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; } }
        
        let mut attn_out = vec![0.0f32; num_heads * head_dim];
        for head in 0..num_heads {
            let kv_head = head / queries_per_kv;
            attn_out[head * head_dim..(head + 1) * head_dim].copy_from_slice(&v[kv_head * head_dim..(kv_head + 1) * head_dim]);
        }
        
        let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
        let h: Vec<f32> = hidden_manual.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();
        
        let ffn_normed = rms_norm(&h, &ffn_norm_w, eps);
        let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
        let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
        let intermediate: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
        let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);
        
        hidden_manual = h.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
    }
    
    // Compute final logits manually
    let normed_final = rms_norm(&hidden_manual, &output_norm_w, eps);
    let logits_manual = vec_mat(&normed_final, &output_w, hidden_size, 151936);
    
    // Compare logits
    println!("Comparing logits (single token):");
    compare(&logits_actual_data, &logits_manual, "logits");
    
    // Check rank of token 17
    let mut indexed_actual: Vec<(usize, f32)> = logits_actual_data.iter().cloned().enumerate().collect();
    indexed_actual.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_actual = indexed_actual.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    let mut indexed_manual: Vec<(usize, f32)> = logits_manual.iter().cloned().enumerate().collect();
    indexed_manual.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_manual = indexed_manual.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("  Token 17 rank: actual={}, manual={}", rank_actual, rank_manual);
    println!("  Token 17 logit: actual={:.4}, manual={:.4}", logits_actual_data[17], logits_manual[17]);
    println!();
    
    if rank_actual == rank_manual {
        println!("SUCCESS: Actual model matches manual computation for single token!");
    } else {
        println!("MISMATCH: Rankings differ!");
    }
    
    // Now test with two tokens
    println!();
    println!("=== Two Token Test (tokens=[16, 28] = '1=') ===");
    println!();
    
    let tokens: Vec<u32> = vec![16, 28];
    
    // Run through actual model
    let mut ctx2 = InferenceContext::new(&config, backend_arc.clone());
    let logits_actual2 = model.forward(&tokens, &mut ctx2).expect("Forward failed");
    let logits_actual2_data = logits_actual2.as_f32().unwrap();
    
    // Manual computation with KV cache
    let mut k_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
        24
    ];
    let mut v_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
        24
    ];
    
    let mut final_hidden = vec![];
    
    for (pos, &token) in tokens.iter().enumerate() {
        let mut hidden = emb[token as usize * hidden_size..(token as usize + 1) * hidden_size].to_vec();
        
        for layer in 0..24 {
            let prefix = format!("blk.{}", layer);
            
            let attn_norm_w = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_norm.weight", prefix)));
            let wq = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_q.weight", prefix)));
            let wk = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_k.weight", prefix)));
            let wv = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_v.weight", prefix)));
            let wo = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_output.weight", prefix)));
            
            let q_bias = try_load_tensor(&gguf, &format!("{}.attn_q.bias", prefix)).map(|t| dequant(&backend, &t));
            let k_bias = try_load_tensor(&gguf, &format!("{}.attn_k.bias", prefix)).map(|t| dequant(&backend, &t));
            let v_bias = try_load_tensor(&gguf, &format!("{}.attn_v.bias", prefix)).map(|t| dequant(&backend, &t));
            
            let ffn_norm_w = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_norm.weight", prefix)));
            let w_gate = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_gate.weight", prefix)));
            let w_up = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_up.weight", prefix)));
            let w_down = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_down.weight", prefix)));
            
            let normed = rms_norm(&hidden, &attn_norm_w, eps);
            let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
            let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
            let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
            
            if let Some(ref bias) = v_bias { for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; } }
            
            for head in 0..num_heads { apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base); }
            for kv_head in 0..num_kv_heads { apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base); }
            
            if let Some(ref bias) = q_bias { for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; } }
            if let Some(ref bias) = k_bias { for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; } }
            
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
    
    // Compute final logits manually
    let normed_final2 = rms_norm(&final_hidden, &output_norm_w, eps);
    let logits_manual2 = vec_mat(&normed_final2, &output_w, hidden_size, 151936);
    
    // Compare logits
    println!("Comparing logits (two tokens):");
    compare(&logits_actual2_data, &logits_manual2, "logits");
    
    // Check rank of token 17
    let mut indexed_actual2: Vec<(usize, f32)> = logits_actual2_data.iter().cloned().enumerate().collect();
    indexed_actual2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_actual2 = indexed_actual2.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    let mut indexed_manual2: Vec<(usize, f32)> = logits_manual2.iter().cloned().enumerate().collect();
    indexed_manual2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_manual2 = indexed_manual2.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("  Token 17 rank: actual={}, manual={}", rank_actual2, rank_manual2);
    println!("  Token 17 logit: actual={:.4}, manual={:.4}", logits_actual2_data[17], logits_manual2[17]);
    
    if rank_actual2 == rank_manual2 {
        println!();
        println!("Model and manual computation match for multi-token too!");
        println!("The issue is consistent between implementations.");
    }
}
