//! Layer-by-layer debug tool for comparing with llama.cpp
//!
//! This example runs inference and dumps intermediate values at each layer
//! to help identify where our implementation diverges from llama.cpp.
//!
//! Usage:
//!   cargo run --example layer_by_layer_debug
//!
//! Output can be compared with the Python reference implementation.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;
use std::sync::Arc;

fn load_tensor(gguf: &GgufFile, name: &str) -> Tensor {
    let tensor_info = gguf
        .data
        .get_tensor(name)
        .unwrap_or_else(|| panic!("No tensor: {}", name));
    let tensor_data = gguf
        .tensor_data(name)
        .unwrap_or_else(|| panic!("No data for: {}", name));
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

fn tensor_stats(data: &[f32]) -> (f32, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    (min, max, mean)
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(w.iter())
        .map(|(v, wt)| v * inv_rms * wt)
        .collect()
}

/// vec_mat: y = x @ W where x is [k], W is [k, n] in column-major, output is [n]
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

fn add_bias(x: &mut [f32], bias: &[f32]) {
    for (x_i, b_i) in x.iter_mut().zip(bias.iter()) {
        *x_i += *b_i;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax(scores: &mut [f32]) {
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    let inv_sum = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

fn apply_rope(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    let position = pos as f32;

    for i in 0..half_dim {
        let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let theta = position * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let idx0 = 2 * i;
        let idx1 = 2 * i + 1;

        let x0 = data[idx0];
        let x1 = data[idx1];

        data[idx0] = x0 * cos_theta - x1 * sin_theta;
        data[idx1] = x0 * sin_theta + x1 * cos_theta;
    }
}

fn main() {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());

    let gguf = GgufFile::open(Path::new(&model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    // Model config
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let num_layers = 24;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt(); // 0.125

    // Tokens: "1+1=" = [16, 10, 16, 28]
    let tokens: Vec<u32> = vec![16, 10, 16, 28];
    let seq_len = tokens.len();

    println!("=== Layer-by-Layer Debug ===");
    println!("Model: Qwen2.5-0.5B-Instruct");
    println!("Tokens: {:?} (seq_len={})", tokens, seq_len);
    println!("Config: hidden={}, heads={}, kv_heads={}, head_dim={}", 
        hidden_size, num_heads, num_kv_heads, head_dim);
    println!();

    // Load embeddings
    let emb_tensor = load_tensor(&gguf, "token_embd.weight");
    let emb_data = dequant(&backend, &emb_tensor);

    // Process each token position
    // We'll track hidden states for each position
    let mut hidden_states: Vec<Vec<f32>> = Vec::new();

    // KV cache: [layer][kv_head][pos][head_dim]
    let mut k_cache: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; seq_len]; num_kv_heads]; 
        num_layers
    ];
    let mut v_cache: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; seq_len]; num_kv_heads]; 
        num_layers
    ];

    // Get embeddings for all tokens
    for &token in &tokens {
        let start = token as usize * hidden_size;
        hidden_states.push(emb_data[start..start + hidden_size].to_vec());
    }

    println!("--- Embeddings ---");
    for (pos, h) in hidden_states.iter().enumerate() {
        let (min, max, mean) = tensor_stats(h);
        println!("Pos {}: min={:.6}, max={:.6}, mean={:.6}, first5={:?}", 
            pos, min, max, mean, &h[..5]);
    }
    println!();

    // Process tokens one at a time (causal)
    for pos in 0..seq_len {
        println!("=== Processing Position {} (token {}) ===", pos, tokens[pos]);
        
        let mut h = hidden_states[pos].clone();

        // Process each layer
        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{}", layer_idx);

            // Load layer weights
            let attn_norm_w = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_norm.weight", prefix)));
            let wq = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_q.weight", prefix)));
            let wk = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_k.weight", prefix)));
            let wv = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_v.weight", prefix)));
            let wo = dequant(&backend, &load_tensor(&gguf, &format!("{}.attn_output.weight", prefix)));

            let q_bias = try_load_tensor(&gguf, &format!("{}.attn_q.bias", prefix))
                .map(|t| dequant(&backend, &t));
            let k_bias = try_load_tensor(&gguf, &format!("{}.attn_k.bias", prefix))
                .map(|t| dequant(&backend, &t));
            let v_bias = try_load_tensor(&gguf, &format!("{}.attn_v.bias", prefix))
                .map(|t| dequant(&backend, &t));

            let ffn_norm_w = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_norm.weight", prefix)));
            let w_gate = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_gate.weight", prefix)));
            let w_up = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_up.weight", prefix)));
            let w_down = dequant(&backend, &load_tensor(&gguf, &format!("{}.ffn_down.weight", prefix)));

            // ===== Attention =====
            // 1. RMSNorm
            let normed = rms_norm(&h, &attn_norm_w, eps);

            // 2. Q, K, V projections
            let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
            let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
            let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);

            // Add biases if present
            if let Some(ref bias) = q_bias {
                add_bias(&mut q, bias);
            }
            if let Some(ref bias) = k_bias {
                add_bias(&mut k, bias);
            }
            if let Some(ref bias) = v_bias {
                add_bias(&mut v, bias);
            }

            // 3. Apply RoPE to Q and K (per head)
            for head in 0..num_heads {
                let offset = head * head_dim;
                apply_rope(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
            }
            for head in 0..num_kv_heads {
                let offset = head * head_dim;
                apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
            }

            // 4. Store K, V in cache
            for kv_head in 0..num_kv_heads {
                let offset = kv_head * head_dim;
                k_cache[layer_idx][kv_head][pos] = k[offset..offset + head_dim].to_vec();
                v_cache[layer_idx][kv_head][pos] = v[offset..offset + head_dim].to_vec();
            }

            // 5. Compute attention
            let kv_len = pos + 1;
            let queries_per_kv = num_heads / num_kv_heads;
            let mut attn_out = vec![0.0f32; num_heads * head_dim];

            for head in 0..num_heads {
                let kv_head = head / queries_per_kv;
                let q_offset = head * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                // Compute attention scores
                let mut scores = vec![0.0f32; kv_len];
                for kv_pos in 0..kv_len {
                    let k_vec = &k_cache[layer_idx][kv_head][kv_pos];
                    let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                    scores[kv_pos] = dot * scale;
                }

                // Causal mask (not needed here since we only attend to past)
                // Softmax
                softmax(&mut scores);

                // Weighted sum of values
                let out_offset = head * head_dim;
                for kv_pos in 0..kv_len {
                    let v_vec = &v_cache[layer_idx][kv_head][kv_pos];
                    for d in 0..head_dim {
                        attn_out[out_offset + d] += scores[kv_pos] * v_vec[d];
                    }
                }
            }

            // 6. Output projection
            let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);

            // 7. Residual connection
            for i in 0..hidden_size {
                h[i] += attn_proj[i];
            }

            // ===== FFN =====
            // 1. RMSNorm
            let ffn_normed = rms_norm(&h, &ffn_norm_w, eps);

            // 2. Gate and up projections
            let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
            let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);

            // 3. SiLU(gate) * up
            let mut intermediate: Vec<f32> = gate.iter().zip(up.iter())
                .map(|(g, u)| silu(*g) * u)
                .collect();

            // 4. Down projection
            let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);

            // 5. Residual connection
            for i in 0..hidden_size {
                h[i] += ffn_out[i];
            }

            // Print stats for first few layers and last layer
            if layer_idx < 3 || layer_idx == num_layers - 1 {
                let (h_min, h_max, h_mean) = tensor_stats(&h);
                println!("  L{:02} hidden: min={:.4}, max={:.4}, mean={:.6}, first5={:.4?}", 
                    layer_idx, h_min, h_max, h_mean, 
                    &h[..5].iter().map(|x| (*x * 10000.0).round() / 10000.0).collect::<Vec<_>>());
            } else if layer_idx == 3 {
                println!("  ... (layers 3-{} omitted) ...", num_layers - 2);
            }
        }

        // Store final hidden state for this position
        hidden_states[pos] = h;
    }

    println!();
    println!("=== Final Output ===");

    // Final norm
    let final_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let final_hidden = &hidden_states[seq_len - 1];
    let normed_final = rms_norm(final_hidden, &final_norm_w, eps);

    let (n_min, n_max, n_mean) = tensor_stats(&normed_final);
    println!("Final normed: min={:.4}, max={:.4}, mean={:.6}", n_min, n_max, n_mean);
    println!("First 10: {:?}", &normed_final[..10].iter().map(|x| (*x * 10000.0).round() / 10000.0).collect::<Vec<_>>());

    // Output projection
    let output_weight = load_tensor(&gguf, "output.weight");
    let output_data = dequant(&backend, &output_weight);
    let vocab_size = 151936;

    // Compute logits: normed @ output_weight
    // output_weight shape: [hidden_size, vocab_size] in GGUF column-major
    let logits = vec_mat(&normed_final, &output_data, hidden_size, vocab_size);

    let (l_min, l_max, l_mean) = tensor_stats(&logits);
    println!("Logits: min={:.4}, max={:.4}, mean={:.6}", l_min, l_max, l_mean);

    // Find top predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("Top 10 predictions:");
    for (idx, logit) in indexed.iter().take(10) {
        println!("  Token {}: logit={:.4}", idx, logit);
    }

    // Check token 17 ("2")
    let token_2_rank = indexed.iter().position(|(idx, _)| *idx == 17).unwrap_or(vocab_size);
    println!();
    println!("Token 17 ('2'): logit={:.4}, rank={}", logits[17], token_2_rank + 1);
}
