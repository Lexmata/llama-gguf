//! Test running inference WITHOUT attention biases to see if that's the issue.

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

fn run_inference(gguf: &GgufFile, backend: &CpuBackend, use_bias: bool) -> (f32, usize, f32) {
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let num_layers = 24;
    let vocab_size = 151936;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_seq_len = 512;

    let tokens: Vec<u32> = vec![16, 10, 16, 28]; // "1+1="

    let emb = dequant(backend, &load_tensor(gguf, "token_embd.weight"));
    let output_norm_w = dequant(backend, &load_tensor(gguf, "output_norm.weight"));
    let output_w = dequant(backend, &load_tensor(gguf, "output.weight"));

    let mut k_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
        num_layers
    ];
    let mut v_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![
        vec![vec![vec![0.0; head_dim]; max_seq_len]; num_kv_heads];
        num_layers
    ];

    let mut hidden_states: Vec<Vec<f32>> = tokens.iter()
        .map(|&tok| emb[tok as usize * hidden_size..(tok as usize + 1) * hidden_size].to_vec())
        .collect();

    for pos in 0..tokens.len() {
        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{}", layer_idx);
            
            let attn_norm_w = dequant(backend, &load_tensor(gguf, &format!("{}.attn_norm.weight", prefix)));
            let wq = dequant(backend, &load_tensor(gguf, &format!("{}.attn_q.weight", prefix)));
            let wk = dequant(backend, &load_tensor(gguf, &format!("{}.attn_k.weight", prefix)));
            let wv = dequant(backend, &load_tensor(gguf, &format!("{}.attn_v.weight", prefix)));
            let wo = dequant(backend, &load_tensor(gguf, &format!("{}.attn_output.weight", prefix)));
            
            let q_bias = if use_bias {
                try_load_tensor(gguf, &format!("{}.attn_q.bias", prefix)).map(|t| dequant(backend, &t))
            } else {
                None
            };
            let k_bias = if use_bias {
                try_load_tensor(gguf, &format!("{}.attn_k.bias", prefix)).map(|t| dequant(backend, &t))
            } else {
                None
            };
            let v_bias = if use_bias {
                try_load_tensor(gguf, &format!("{}.attn_v.bias", prefix)).map(|t| dequant(backend, &t))
            } else {
                None
            };

            let ffn_norm_w = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_norm.weight", prefix)));
            let w_gate = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_gate.weight", prefix)));
            let w_up = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_up.weight", prefix)));
            let w_down = dequant(backend, &load_tensor(gguf, &format!("{}.ffn_down.weight", prefix)));

            let h = &hidden_states[pos];
            let normed = rms_norm(h, &attn_norm_w, eps);
            
            let mut q = vec_mat(&normed, &wq, hidden_size, num_heads * head_dim);
            let mut k = vec_mat(&normed, &wk, hidden_size, num_kv_heads * head_dim);
            let mut v = vec_mat(&normed, &wv, hidden_size, num_kv_heads * head_dim);
            
            if let Some(ref bias) = q_bias { for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; } }
            if let Some(ref bias) = k_bias { for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; } }
            if let Some(ref bias) = v_bias { for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; } }

            for head in 0..num_heads {
                let offset = head * head_dim;
                apply_rope(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
            }
            for kv_head in 0..num_kv_heads {
                let offset = kv_head * head_dim;
                apply_rope(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
            }

            for kv_head in 0..num_kv_heads {
                let offset = kv_head * head_dim;
                k_caches[layer_idx][kv_head][pos] = k[offset..offset + head_dim].to_vec();
                v_caches[layer_idx][kv_head][pos] = v[offset..offset + head_dim].to_vec();
            }

            let kv_len = pos + 1;
            let queries_per_kv = num_heads / num_kv_heads;
            let mut attn_out = vec![0.0f32; num_heads * head_dim];

            for head in 0..num_heads {
                let kv_head = head / queries_per_kv;
                let q_vec = &q[head * head_dim..(head + 1) * head_dim];
                
                let mut scores = vec![0.0f32; kv_len];
                for kv_pos in 0..kv_len {
                    let k_vec = &k_caches[layer_idx][kv_head][kv_pos];
                    let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                    scores[kv_pos] = dot * scale;
                }
                softmax(&mut scores);
                for kv_pos in 0..kv_len {
                    let v_vec = &v_caches[layer_idx][kv_head][kv_pos];
                    for d in 0..head_dim {
                        attn_out[head * head_dim + d] += scores[kv_pos] * v_vec[d];
                    }
                }
            }

            let attn_proj = vec_mat(&attn_out, &wo, num_heads * head_dim, hidden_size);
            let mut h_after_attn: Vec<f32> = h.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

            let ffn_normed = rms_norm(&h_after_attn, &ffn_norm_w, eps);
            let gate = vec_mat(&ffn_normed, &w_gate, hidden_size, intermediate_size);
            let up = vec_mat(&ffn_normed, &w_up, hidden_size, intermediate_size);
            let intermediate: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
            let ffn_out = vec_mat(&intermediate, &w_down, intermediate_size, hidden_size);

            let h_final: Vec<f32> = h_after_attn.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
            hidden_states[pos] = h_final;
        }
    }

    let final_hidden = &hidden_states[tokens.len() - 1];
    let normed = rms_norm(final_hidden, &output_norm_w, eps);
    let logits = vec_mat(&normed, &output_w, hidden_size, vocab_size);

    let token_17_logit = logits[17];
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank = indexed.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;

    (token_17_logit, rank, max_logit)
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    println!("=== Bias Impact Test ===");
    println!();
    
    let (logit_with, rank_with, max_with) = run_inference(&gguf, &backend, true);
    let (logit_without, rank_without, max_without) = run_inference(&gguf, &backend, false);
    
    println!("                   | WITH bias  | WITHOUT bias | Diff");
    println!("-------------------+------------+--------------+------");
    println!("Token 17 logit:    | {:>10.4} | {:>12.4} | {:>6.2}", logit_with, logit_without, logit_with - logit_without);
    println!("Token 17 rank:     | {:>10} | {:>12} |", rank_with, rank_without);
    println!("Max logit:         | {:>10.4} | {:>12.4} |", max_with, max_without);
    println!();
    println!("Reference: llama.cpp (with bias) produces rank 1, logit ~18");
}
