//! Detailed layer 0 analysis for debugging.
//!
//! This outputs exact intermediate values at each step of layer 0
//! for comparison with llama.cpp.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

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

fn tensor_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let sum_sq: f32 = data.iter().map(|x| x * x).sum();
    let rms = (sum_sq / data.len() as f32).sqrt();
    (min, max, mean, rms)
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    println!("  RMS norm: sum_sq={:.6}, n={}, rms={:.6}, inv_rms={:.6}", 
        sum_sq, x.len(), rms, inv_rms);
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
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Single token: "1" = token 16
    let token_id = 16u32;
    let pos = 0usize;

    println!("=== Layer 0 Detailed Analysis ===");
    println!("Token: {} at position {}", token_id, pos);
    println!("Config: hidden={}, heads={}, kv_heads={}, head_dim={}", 
        hidden_size, num_heads, num_kv_heads, head_dim);
    println!();

    // ===== Step 1: Embedding Lookup =====
    println!("===== Step 1: Embedding Lookup =====");
    let emb_tensor = load_tensor(&gguf, "token_embd.weight");
    let emb_data = dequant(&backend, &emb_tensor);
    
    let start = token_id as usize * hidden_size;
    let h: Vec<f32> = emb_data[start..start + hidden_size].to_vec();
    
    let (min, max, mean, rms) = tensor_stats(&h);
    println!("Embedding: min={:.6}, max={:.6}, mean={:.6}, rms={:.6}", min, max, mean, rms);
    println!("First 10: {:?}", &h[..10]);
    println!();

    // ===== Step 2: Attention Norm =====
    println!("===== Step 2: Attention Norm =====");
    let attn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_norm.weight"));
    let (w_min, w_max, w_mean, _) = tensor_stats(&attn_norm_w);
    println!("Norm weights: min={:.6}, max={:.6}, mean={:.6}", w_min, w_max, w_mean);
    println!("First 10: {:?}", &attn_norm_w[..10]);
    
    let normed = rms_norm(&h, &attn_norm_w, eps);
    let (n_min, n_max, n_mean, n_rms) = tensor_stats(&normed);
    println!("After norm: min={:.6}, max={:.6}, mean={:.6}, rms={:.6}", n_min, n_max, n_mean, n_rms);
    println!("First 10: {:?}", &normed[..10]);
    println!();

    // ===== Step 3: Q Projection =====
    println!("===== Step 3: Q Projection =====");
    let wq = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_q.weight"));
    let q_bias = try_load_tensor(&gguf, "blk.0.attn_q.bias")
        .map(|t| dequant(&backend, &t));
    
    let (wq_min, wq_max, wq_mean, _) = tensor_stats(&wq);
    println!("Wq: min={:.6}, max={:.6}, mean={:.9}", wq_min, wq_max, wq_mean);
    
    let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
    let (q_min, q_max, q_mean, _) = tensor_stats(&q);
    println!("Q (before bias): min={:.6}, max={:.6}, mean={:.9}", q_min, q_max, q_mean);
    println!("Q first 10: {:?}", &q[..10]);
    
    if let Some(ref bias) = q_bias {
        let (b_min, b_max, b_mean, _) = tensor_stats(bias);
        println!("Q bias: min={:.4}, max={:.4}, mean={:.6}", b_min, b_max, b_mean);
        println!("Q bias first 10: {:?}", &bias[..10]);
        add_bias(&mut q, bias);
        let (q_min, q_max, q_mean, _) = tensor_stats(&q);
        println!("Q (after bias): min={:.4}, max={:.4}, mean={:.6}", q_min, q_max, q_mean);
        println!("Q first 10 (after bias): {:?}", &q[..10]);
    }
    println!();

    // ===== Step 4: K Projection =====
    println!("===== Step 4: K Projection =====");
    let wk = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_k.weight"));
    let k_bias = try_load_tensor(&gguf, "blk.0.attn_k.bias")
        .map(|t| dequant(&backend, &t));
    
    let (wk_min, wk_max, wk_mean, _) = tensor_stats(&wk);
    println!("Wk: min={:.6}, max={:.6}, mean={:.9}", wk_min, wk_max, wk_mean);
    
    let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
    let (k_min, k_max, k_mean, _) = tensor_stats(&k);
    println!("K (before bias): min={:.6}, max={:.6}, mean={:.9}", k_min, k_max, k_mean);
    println!("K first 10: {:?}", &k[..10]);
    
    if let Some(ref bias) = k_bias {
        let (b_min, b_max, b_mean, _) = tensor_stats(bias);
        println!("K bias: min={:.4}, max={:.4}, mean={:.6}", b_min, b_max, b_mean);
        println!("K bias first 10: {:?}", &bias[..10]);
        add_bias(&mut k, bias);
        let (k_min, k_max, k_mean, _) = tensor_stats(&k);
        println!("K (after bias): min={:.4}, max={:.4}, mean={:.6}", k_min, k_max, k_mean);
    }
    println!();

    // ===== Step 5: V Projection =====
    println!("===== Step 5: V Projection =====");
    let wv = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_v.weight"));
    let v_bias = try_load_tensor(&gguf, "blk.0.attn_v.bias")
        .map(|t| dequant(&backend, &t));
    
    let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
    let (v_min, v_max, v_mean, _) = tensor_stats(&v);
    println!("V (before bias): min={:.6}, max={:.6}, mean={:.9}", v_min, v_max, v_mean);
    
    if let Some(ref bias) = v_bias {
        add_bias(&mut v, bias);
        let (v_min, v_max, v_mean, _) = tensor_stats(&v);
        println!("V (after bias): min={:.6}, max={:.6}, mean={:.9}", v_min, v_max, v_mean);
    }
    println!();

    // ===== Step 6: RoPE =====
    println!("===== Step 6: RoPE (pos=0, should be identity) =====");
    let q_before_rope = q.clone();
    let k_before_rope = k.clone();
    
    for head in 0..num_heads {
        let offset = head * head_dim;
        apply_rope(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
    }
    for head in 0..num_kv_heads {
        let offset = head * head_dim;
        apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
    }
    
    let q_diff: f32 = q.iter().zip(q_before_rope.iter()).map(|(a, b)| (a - b).abs()).sum();
    let k_diff: f32 = k.iter().zip(k_before_rope.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("Q change after RoPE (should be 0 at pos=0): {:.9}", q_diff);
    println!("K change after RoPE (should be 0 at pos=0): {:.9}", k_diff);
    println!();

    // ===== Step 7: Attention =====
    println!("===== Step 7: Attention (single token, kv_len=1) =====");
    let kv_len = 1;
    let queries_per_kv = num_heads / num_kv_heads;
    let mut attn_out = vec![0.0f32; num_heads * head_dim];

    println!("scale = 1/sqrt({}) = {}", head_dim, scale);
    
    for head in 0..num_heads {
        let kv_head = head / queries_per_kv;
        let q_offset = head * head_dim;
        let k_offset = kv_head * head_dim;
        let v_offset = kv_head * head_dim;
        
        let q_vec = &q[q_offset..q_offset + head_dim];
        let k_vec = &k[k_offset..k_offset + head_dim];
        let v_vec = &v[v_offset..v_offset + head_dim];
        
        // Q @ K dot product
        let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
        let scaled = dot * scale;
        
        // Softmax over single element = 1.0
        // So attention output = V
        let out_offset = head * head_dim;
        for d in 0..head_dim {
            attn_out[out_offset + d] = v_vec[d];
        }
        
        if head < 2 {
            println!("Head {}: Q@K dot={:.4}, scaled={:.4}", head, dot, scaled);
            println!("  Q[0:5] = {:?}", &q_vec[..5]);
            println!("  K[0:5] = {:?}", &k_vec[..5]);
            println!("  V[0:5] = {:?}", &v_vec[..5]);
        }
    }
    
    let (ao_min, ao_max, ao_mean, _) = tensor_stats(&attn_out);
    println!("Attention output: min={:.6}, max={:.6}, mean={:.9}", ao_min, ao_max, ao_mean);
    println!();

    // ===== Step 8: Output Projection =====
    println!("===== Step 8: Output Projection =====");
    let wo = dequant(&backend, &load_tensor(&gguf, "blk.0.attn_output.weight"));
    let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
    
    let (ap_min, ap_max, ap_mean, _) = tensor_stats(&attn_proj);
    println!("Attn projection: min={:.6}, max={:.6}, mean={:.9}", ap_min, ap_max, ap_mean);
    println!("First 10: {:?}", &attn_proj[..10]);
    println!();

    // ===== Step 9: Residual =====
    println!("===== Step 9: Residual Connection =====");
    let mut h2: Vec<f32> = h.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();
    let (h2_min, h2_max, h2_mean, _) = tensor_stats(&h2);
    println!("After attention residual: min={:.6}, max={:.6}, mean={:.9}", h2_min, h2_max, h2_mean);
    println!("First 10: {:?}", &h2[..10]);
    println!();

    // ===== Step 10: FFN Norm =====
    println!("===== Step 10: FFN Norm =====");
    let ffn_norm_w = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_norm.weight"));
    let ffn_normed = rms_norm(&h2, &ffn_norm_w, eps);
    let (fn_min, fn_max, fn_mean, _) = tensor_stats(&ffn_normed);
    println!("After FFN norm: min={:.6}, max={:.6}, mean={:.9}", fn_min, fn_max, fn_mean);
    println!();

    // ===== Step 11: FFN =====
    println!("===== Step 11: FFN =====");
    let w_gate = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_gate.weight"));
    let w_up = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_up.weight"));
    let w_down = dequant(&backend, &load_tensor(&gguf, "blk.0.ffn_down.weight"));
    
    let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
    let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
    
    let (g_min, g_max, _, _) = tensor_stats(&gate);
    let (u_min, u_max, _, _) = tensor_stats(&up);
    println!("Gate: min={:.6}, max={:.6}", g_min, g_max);
    println!("Up: min={:.6}, max={:.6}", u_min, u_max);
    
    let intermediate: Vec<f32> = gate.iter().zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    
    let (i_min, i_max, _, _) = tensor_stats(&intermediate);
    println!("SiLU(gate) * up: min={:.6}, max={:.6}", i_min, i_max);
    
    let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);
    let (fo_min, fo_max, fo_mean, _) = tensor_stats(&ffn_out);
    println!("FFN output: min={:.6}, max={:.6}, mean={:.9}", fo_min, fo_max, fo_mean);
    println!();

    // ===== Step 12: Final Residual =====
    println!("===== Step 12: Final Residual =====");
    let h_final: Vec<f32> = h2.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
    let (hf_min, hf_max, hf_mean, _) = tensor_stats(&h_final);
    println!("Layer 0 output: min={:.6}, max={:.6}, mean={:.9}", hf_min, hf_max, hf_mean);
    println!("First 10: {:?}", &h_final[..10]);
}
