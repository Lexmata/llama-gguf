//! Test different RoPE frequency formulas.

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

/// NeoX-style RoPE with the standard formula from our implementation
fn apply_rope_neox_v1(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        // Formula: freq = 1 / (base ^ (2*i / head_dim))
        let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = data[i];
        let x1 = data[i + half_dim];
        data[i] = x0 * cos_t - x1 * sin_t;
        data[i + half_dim] = x0 * sin_t + x1 * cos_t;
    }
}

/// NeoX-style RoPE with llama.cpp formula (uses i instead of 2*i in exponent)
fn apply_rope_neox_v2(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        // Alternative formula: freq = 1 / (base ^ (i / half_dim))
        // This is because in some implementations, the exponent ranges from 0 to 1 
        // over the half_dim instead of 0 to 1 over head_dim/2
        let freq = 1.0 / freq_base.powf(i as f32 / half_dim as f32);
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
    apply_rope: fn(&mut [f32], usize, usize, f32),
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
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    
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
            
            // V bias applied directly
            if let Some(ref bias) = v_bias { 
                for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; } 
            }
            
            // Bias BEFORE RoPE for Q and K
            if let Some(ref bias) = q_bias {
                for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
            }
            if let Some(ref bias) = k_bias {
                for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
            }
            
            // Apply RoPE
            for head in 0..num_heads {
                apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base);
            }
            for kv_head in 0..num_kv_heads {
                apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
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
    
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
    let output_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let output_w = dequant(&backend, &load_tensor(&gguf, "output.weight"));
    
    let tokens: Vec<u32> = vec![16, 10, 16, 28];  // "1+1="
    
    println!("Testing RoPE frequency formulas (with NeoX style, bias BEFORE RoPE)");
    println!("Tokens: {:?}", tokens);
    println!();
    
    // Test V1: freq = 1 / (base ^ (2*i / head_dim))
    let logits_v1 = run_full_forward(
        &tokens, &gguf, &backend, &emb, &output_norm_w, &output_w,
        apply_rope_neox_v1
    );
    let mut indexed_v1: Vec<(usize, f32)> = logits_v1.iter().cloned().enumerate().collect();
    indexed_v1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_v1 = indexed_v1.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("=== V1: freq = 1 / (base ^ (2*i / head_dim)) ===");
    
    // Show logit distribution
    let min_logit: f32 = logits_v1.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_logit: f32 = logits_v1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_logit: f32 = logits_v1.iter().sum::<f32>() / logits_v1.len() as f32;
    let sum_logit: f32 = logits_v1.iter().sum();
    
    println!("  Logit stats: min={:.4}, max={:.4}, mean={:.4}, sum={:.4}", min_logit, max_logit, mean_logit, sum_logit);
    println!("  Sample [0:5]: {:?}", &logits_v1[..5].iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
    println!("  Sample [16:20] (numeric): {:?}", &logits_v1[16..20].iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
    
    println!("  Token 17 ('2') rank: {}", rank_v1);
    println!("  Token 17 logit: {:.4}", logits_v1[17]);
    println!("  Top 5: {:?}", indexed_v1.iter().take(5).map(|(i, l)| (*i, format!("{:.2}", l))).collect::<Vec<_>>());
    println!();
    
    // Test V2: freq = 1 / (base ^ (i / half_dim))
    let logits_v2 = run_full_forward(
        &tokens, &gguf, &backend, &emb, &output_norm_w, &output_w,
        apply_rope_neox_v2
    );
    let mut indexed_v2: Vec<(usize, f32)> = logits_v2.iter().cloned().enumerate().collect();
    indexed_v2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_v2 = indexed_v2.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    
    println!("=== V2: freq = 1 / (base ^ (i / half_dim)) ===");
    println!("  Token 17 ('2') rank: {}", rank_v2);
    println!("  Token 17 logit: {:.4}", logits_v2[17]);
    println!("  Top 5: {:?}", indexed_v2.iter().take(5).map(|(i, l)| (*i, format!("{:.2}", l))).collect::<Vec<_>>());
    println!();
    
    // Compare with llama-cpp-python reference
    println!("=== llama-cpp-python reference ===");
    println!("  Token 17 ('2') rank: 1");
    println!("  Token 17 logit: 17.99");
    println!("  Top 5: [(17, \"17.99\"), (16, \"17.54\"), (18, \"17.16\"), (20, \"15.98\"), (9909, \"15.86\")]");
}
