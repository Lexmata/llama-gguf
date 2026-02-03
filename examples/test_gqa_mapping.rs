//! Test different GQA (Grouped Query Attention) mappings.
//!
//! Our implementation: heads 0-6 use KV head 0, heads 7-13 use KV head 1 (contiguous)
//! Alternative: interleaved mapping

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

enum GqaMapping {
    Contiguous,  // heads 0-6 -> KV 0, heads 7-13 -> KV 1
    Interleaved, // heads 0,2,4,6,8,10,12 -> KV 0, heads 1,3,5,7,9,11,13 -> KV 1
}

fn get_kv_head(head: usize, mapping: &GqaMapping, num_heads: usize, num_kv_heads: usize) -> usize {
    match mapping {
        GqaMapping::Contiguous => head / (num_heads / num_kv_heads),
        GqaMapping::Interleaved => head % num_kv_heads,
    }
}

fn run_full_forward(
    tokens: &[u32],
    gguf: &GgufFile,
    backend: &CpuBackend,
    emb: &[f32],
    output_norm_w: &[f32],
    output_w: &[f32],
    mapping: GqaMapping,
) -> Vec<f32> {
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let scale = 1.0 / (head_dim as f32).sqrt();
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
            
            if let Some(ref bias) = v_bias { 
                for (vi, bi) in v.iter_mut().zip(bias.iter()) { *vi += *bi; } 
            }
            
            for head in 0..num_heads {
                apply_rope(&mut q[head * head_dim..(head + 1) * head_dim], pos, head_dim, freq_base);
            }
            for kv_head in 0..num_kv_heads {
                apply_rope(&mut k[kv_head * head_dim..(kv_head + 1) * head_dim], pos, head_dim, freq_base);
            }
            
            if let Some(ref bias) = q_bias {
                for (qi, bi) in q.iter_mut().zip(bias.iter()) { *qi += *bi; }
            }
            if let Some(ref bias) = k_bias {
                for (ki, bi) in k.iter_mut().zip(bias.iter()) { *ki += *bi; }
            }
            
            for kv_head in 0..num_kv_heads {
                k_caches[layer][kv_head][pos] = k[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec();
                v_caches[layer][kv_head][pos] = v[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec();
            }
            
            let kv_len = pos + 1;
            let mut attn_out = vec![0.0f32; num_heads * head_dim];
            
            for head in 0..num_heads {
                let kv_head = get_kv_head(head, &mapping, num_heads, num_kv_heads);
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
    
    println!("Testing GQA mappings with tokens: {:?}", tokens);
    println!();
    
    // Contiguous mapping
    println!("=== Contiguous GQA Mapping ===");
    println!("  heads 0-6 -> KV head 0, heads 7-13 -> KV head 1");
    let logits_cont = run_full_forward(
        &tokens, &gguf, &backend, &emb, &output_norm_w, &output_w,
        GqaMapping::Contiguous
    );
    let mut indexed_cont: Vec<(usize, f32)> = logits_cont.iter().cloned().enumerate().collect();
    indexed_cont.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_cont = indexed_cont.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    println!("  Token 17 ('2') rank: {}", rank_cont);
    println!();
    
    // Interleaved mapping
    println!("=== Interleaved GQA Mapping ===");
    println!("  heads 0,2,4,6,8,10,12 -> KV 0, heads 1,3,5,7,9,11,13 -> KV 1");
    let logits_inter = run_full_forward(
        &tokens, &gguf, &backend, &emb, &output_norm_w, &output_w,
        GqaMapping::Interleaved
    );
    let mut indexed_inter: Vec<(usize, f32)> = logits_inter.iter().cloned().enumerate().collect();
    indexed_inter.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank_inter = indexed_inter.iter().position(|(idx, _)| *idx == 17).unwrap() + 1;
    println!("  Token 17 ('2') rank: {}", rank_inter);
    println!();
    
    // Summary
    println!("=== Summary ===");
    println!("  Contiguous:   Token '2' rank = {}", rank_cont);
    println!("  Interleaved:  Token '2' rank = {}", rank_inter);
    if rank_inter < rank_cont {
        println!("\n  ** INTERLEAVED mapping is better! **");
    } else if rank_cont < rank_inter {
        println!("\n  ** CONTIGUOUS mapping is better (current) **");
    } else {
        println!("\n  Both produce the same rank.");
    }
}
