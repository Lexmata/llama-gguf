//! Test actual model Layer 0 vs manual computation for multi-token input.

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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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
    let intermediate_size = 4864;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_seq_len = 512;

    // Tokens: "1+1=" = [16, 10, 16, 28]
    let tokens: Vec<u32> = vec![16, 10, 16, 28];

    println!("=== Testing Actual Model Layer 0 vs Manual ===");
    println!("Tokens: {:?}", tokens);
    println!();

    // Load weights for manual computation
    let emb = dequant(&backend, &load_tensor(&gguf, "token_embd.weight"));
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

    // Manual KV cache
    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];

    // Process each position and collect layer 0 outputs
    let mut manual_outputs: Vec<Vec<f32>> = Vec::new();

    for (pos, &tok) in tokens.iter().enumerate() {
        // Get embedding
        let h: Vec<f32> = emb[tok as usize * hidden_size..(tok as usize + 1) * hidden_size].to_vec();

        // === ATTENTION ===
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
        for kv_head in 0..num_kv_heads {
            let offset = kv_head * head_dim;
            apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
        }

        // Store K, V in cache
        for kv_head in 0..num_kv_heads {
            let offset = kv_head * head_dim;
            k_cache[kv_head][pos] = k[offset..offset + head_dim].to_vec();
            v_cache[kv_head][pos] = v[offset..offset + head_dim].to_vec();
        }

        // Compute attention
        let kv_len = pos + 1;
        let queries_per_kv = num_heads / num_kv_heads;
        let mut attn_out = vec![0.0f32; num_heads * head_dim];

        for head in 0..num_heads {
            let kv_head = head / queries_per_kv;
            let q_vec = &q[head * head_dim..(head + 1) * head_dim];
            
            let mut scores = vec![0.0f32; kv_len];
            for kv_pos in 0..kv_len {
                let k_vec = &k_cache[kv_head][kv_pos];
                let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                scores[kv_pos] = dot * scale;
            }
            
            softmax(&mut scores);
            
            let out_offset = head * head_dim;
            for kv_pos in 0..kv_len {
                let v_vec = &v_cache[kv_head][kv_pos];
                for d in 0..head_dim {
                    attn_out[out_offset + d] += scores[kv_pos] * v_vec[d];
                }
            }
        }

        // Output projection
        let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);

        // Residual
        let mut h_after_attn: Vec<f32> = h.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

        // === FFN ===
        let ffn_normed = rms_norm(&h_after_attn, &ffn_norm_w, eps);
        let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
        let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
        let intermediate: Vec<f32> = gate.iter().zip(up.iter())
            .map(|(g, u)| silu(*g) * u)
            .collect();
        let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);

        // Residual
        let h_final: Vec<f32> = h_after_attn.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        manual_outputs.push(h_final);
    }

    println!("\n=== Final Comparison ===");
    println!();
    
    // Print manual layer 0 outputs for the last position
    let manual_last = &manual_outputs[tokens.len() - 1];
    let (m_min, m_max) = (
        manual_last.iter().cloned().fold(f32::INFINITY, f32::min),
        manual_last.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("Manual Layer 0 output (pos 3): min={:.4}, max={:.4}", m_min, m_max);
    println!("First 5: {:?}", &manual_last[..5].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    
    // Print what layer_by_layer_debug showed
    println!();
    println!("From layer_by_layer_debug.rs (pos 3, layer 0):");
    println!("  L00 hidden: min=-0.4688, max=0.4737");
    println!("  first5=[0.0412, 0.0617, -0.0334, 0.1000, 0.0196]");
    
    println!();
    println!("If these match, our manual layer 0 computation is correct.");
    println!("If they differ, there's a bug in how we process multi-token sequences.");
}
