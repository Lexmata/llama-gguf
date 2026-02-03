//! Trace through full layer 0 with 4 tokens to debug attention

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn load_tensor(gguf: &GgufFile, name: &str) -> Tensor {
    let tensor_info = gguf.data.get_tensor(name).expect(&format!("No tensor: {}", name));
    let tensor_data = gguf.tensor_data(name).expect(&format!("No data for: {}", name));
    let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
    let dtype = DType::from(tensor_info.dtype);
    Tensor::new(tensor_data.to_vec(), shape, dtype).expect("Failed to create tensor")
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
    let rms_inv = 1.0 / rms;
    x.iter().zip(w.iter()).map(|(v, wt)| v * rms_inv * wt).collect()
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
    let num_kv_heads = 2;  // Qwen2 uses GQA with 2 KV heads
    let kv_head_dim = head_dim;
    
    // Tokens: "1+1=" = [16, 10, 16, 28]
    let tokens: Vec<u32> = vec![16, 10, 16, 28];
    let seq_len = tokens.len();
    
    println!("Config: hidden={}, heads={}, head_dim={}, kv_heads={}", 
        hidden_size, num_heads, head_dim, num_kv_heads);
    println!("Tokens: {:?}, seq_len={}", tokens, seq_len);
    
    // Load embeddings
    let emb_tensor = load_tensor(&gguf, "token_embd.weight");
    let emb_data = dequant(&backend, &emb_tensor);
    
    // Get embeddings for all tokens [seq_len, hidden_size]
    let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * hidden_size);
    for &token in &tokens {
        let start = token as usize * hidden_size;
        hidden.extend_from_slice(&emb_data[start..start + hidden_size]);
    }
    
    println!("\nInput hidden shape: [{}x{}]", seq_len, hidden_size);
    println!("Token 0 first 5: {:?}", &hidden[..5]);
    println!("Token 3 first 5: {:?}", &hidden[3*hidden_size..3*hidden_size+5]);
    
    // Load layer 0 weights
    let attn_norm_tensor = load_tensor(&gguf, "blk.0.attn_norm.weight");
    let attn_norm = dequant(&backend, &attn_norm_tensor);
    
    let q_proj = load_tensor(&gguf, "blk.0.attn_q.weight");
    let k_proj = load_tensor(&gguf, "blk.0.attn_k.weight");
    let v_proj = load_tensor(&gguf, "blk.0.attn_v.weight");
    let out_proj = load_tensor(&gguf, "blk.0.attn_output.weight");
    
    let q_weights = dequant(&backend, &q_proj);
    let k_weights = dequant(&backend, &k_proj);
    let v_weights = dequant(&backend, &v_proj);
    let out_weights = dequant(&backend, &out_proj);
    
    println!("\nWeights loaded:");
    println!("  Q: [{}, {}]", q_proj.shape()[0], q_proj.shape()[1]);
    println!("  K: [{}, {}]", k_proj.shape()[0], k_proj.shape()[1]);
    println!("  V: [{}, {}]", v_proj.shape()[0], v_proj.shape()[1]);
    println!("  Out: [{}, {}]", out_proj.shape()[0], out_proj.shape()[1]);
    
    // Apply RMSNorm and compute QKV for each position
    let mut q_all = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k_all = vec![0.0f32; seq_len * num_kv_heads * kv_head_dim];
    let mut v_all = vec![0.0f32; seq_len * num_kv_heads * kv_head_dim];
    
    for pos in 0..seq_len {
        let h = &hidden[pos*hidden_size..(pos+1)*hidden_size];
        let normed = rms_norm(h, &attn_norm, 1e-6);
        
        // Q projection
        for i in 0..num_heads*head_dim {
            let mut sum = 0.0f32;
            for j in 0..hidden_size {
                sum += normed[j] * q_weights[i * hidden_size + j];
            }
            q_all[pos * num_heads * head_dim + i] = sum;
        }
        
        // K projection
        for i in 0..num_kv_heads*kv_head_dim {
            let mut sum = 0.0f32;
            for j in 0..hidden_size {
                sum += normed[j] * k_weights[i * hidden_size + j];
            }
            k_all[pos * num_kv_heads * kv_head_dim + i] = sum;
        }
        
        // V projection  
        for i in 0..num_kv_heads*kv_head_dim {
            let mut sum = 0.0f32;
            for j in 0..hidden_size {
                sum += normed[j] * v_weights[i * hidden_size + j];
            }
            v_all[pos * num_kv_heads * kv_head_dim + i] = sum;
        }
    }
    
    println!("\nQ/K/V computed:");
    println!("  Q[0] first 5: {:?}", &q_all[..5]);
    println!("  K[0] first 5: {:?}", &k_all[..5]);
    println!("  V[0] first 5: {:?}", &v_all[..5]);
    
    // Apply RoPE (simplified - real impl is more complex)
    // For now, skip RoPE to see if the issue is there
    
    // Compute attention for head 0, position 3 (the "=" token)
    // Q[3], K[0..4], V[0..4]
    let q_pos3_head0: Vec<f32> = q_all[3*num_heads*head_dim..3*num_heads*head_dim+head_dim].to_vec();
    
    // Get K for all positions, KV head 0 (shared among heads)
    let k_heads: Vec<Vec<f32>> = (0..seq_len).map(|p| {
        k_all[p*num_kv_heads*kv_head_dim..p*num_kv_heads*kv_head_dim+kv_head_dim].to_vec()
    }).collect();
    
    let v_heads: Vec<Vec<f32>> = (0..seq_len).map(|p| {
        v_all[p*num_kv_heads*kv_head_dim..p*num_kv_heads*kv_head_dim+kv_head_dim].to_vec()
    }).collect();
    
    // Compute attention scores for position 3, head 0
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    
    for k_pos in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q_pos3_head0[d] * k_heads[k_pos][d];
        }
        scores[k_pos] = dot * scale;
    }
    
    println!("\nAttention scores for pos=3, head=0 (before softmax): {:?}", scores);
    
    // Apply causal mask (pos 3 can attend to 0, 1, 2, 3)
    // All positions are valid, no masking needed
    
    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    let attn_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
    
    println!("Attention weights for pos=3, head=0: {:?}", attn_weights);
    
    // Compute attention output
    let mut attn_out = vec![0.0f32; head_dim];
    for k_pos in 0..seq_len {
        for d in 0..head_dim {
            attn_out[d] += attn_weights[k_pos] * v_heads[k_pos][d];
        }
    }
    
    println!("Attention output for pos=3, head=0 first 10: {:?}", &attn_out[..10]);
}
