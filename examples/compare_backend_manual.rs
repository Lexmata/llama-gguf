//! Compare backend operations with manual computation.
//! 
//! This test verifies that:
//! 1. backend.vec_mat_q produces same output as manual dequant + vec_mat
//! 2. backend.rms_norm produces same output as manual rms_norm
//! 3. backend.rope produces same output as manual rope

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

fn manual_dequant(backend: &CpuBackend, t: &Tensor) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        t.as_f32().unwrap().to_vec()
    } else {
        let numel = t.numel();
        let mut out = Tensor::zeros(vec![numel], DType::F32);
        backend.dequantize(t, &mut out).expect("dequant failed");
        out.as_f32().unwrap().to_vec()
    }
}

fn manual_vec_mat(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
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

fn manual_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter().zip(w.iter()).map(|(v, wt)| v * inv_rms * wt).collect()
}

fn manual_rope(data: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
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

fn compare_vecs(name: &str, manual: &[f32], backend_out: &[f32], tolerance: f32) {
    assert_eq!(manual.len(), backend_out.len(), "{} length mismatch", name);
    
    let max_diff: f32 = manual.iter()
        .zip(backend_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    let mean_diff: f32 = manual.iter()
        .zip(backend_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / manual.len() as f32;
    
    println!("{}: max_diff={:.9}, mean_diff={:.9}", name, max_diff, mean_diff);
    
    if max_diff > tolerance {
        println!("  WARNING: max_diff exceeds tolerance {}!", tolerance);
        
        // Find worst elements
        let mut diffs: Vec<(usize, f32, f32, f32)> = manual.iter()
            .zip(backend_out.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, *a, *b, (a - b).abs()))
            .collect();
        diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        
        println!("  Worst 5 elements:");
        for (idx, m, b, diff) in diffs.iter().take(5) {
            println!("    [{}]: manual={:.6}, backend={:.6}, diff={:.9}", idx, m, b, diff);
        }
    }
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let eps = 1e-6f32;
    let freq_base = 1000000.0f32;

    println!("=== Backend vs Manual Comparison ===\n");

    // Load test data
    let emb = load_tensor(&gguf, "token_embd.weight");
    let emb_data = manual_dequant(&backend, &emb);
    let tok_emb: Vec<f32> = emb_data[16 * hidden_size..(16 + 1) * hidden_size].to_vec();
    
    let attn_norm_w = load_tensor(&gguf, "blk.0.attn_norm.weight");
    let wq = load_tensor(&gguf, "blk.0.attn_q.weight");

    println!("--- Test 1: RMS Norm ---");
    
    // Manual
    let norm_w_data = manual_dequant(&backend, &attn_norm_w);
    let manual_normed = manual_rms_norm(&tok_emb, &norm_w_data, eps);
    
    // Backend
    let mut input_tensor = Tensor::from_f32(&tok_emb, vec![hidden_size]).unwrap();
    let mut backend_normed = Tensor::zeros(vec![hidden_size], DType::F32);
    backend.rms_norm(&input_tensor, &attn_norm_w, eps, &mut backend_normed).unwrap();
    let backend_normed_data = backend_normed.as_f32().unwrap();
    
    compare_vecs("RMS Norm", &manual_normed, backend_normed_data, 1e-5);
    println!();

    println!("--- Test 2: vec_mat_q (Q projection) ---");
    
    // Manual: dequant weights, then vec_mat
    let wq_data = manual_dequant(&backend, &wq);
    let manual_q = manual_vec_mat(&manual_normed, &wq_data, hidden_size, num_heads * head_dim);
    
    // Backend: vec_mat_q directly
    let normed_tensor = Tensor::from_f32(&manual_normed, vec![hidden_size]).unwrap();
    let mut backend_q = Tensor::zeros(vec![num_heads * head_dim], DType::F32);
    backend.vec_mat_q(&normed_tensor, &wq, &mut backend_q).unwrap();
    let backend_q_data = backend_q.as_f32().unwrap();
    
    compare_vecs("vec_mat_q (Q)", &manual_q, backend_q_data, 1e-3);  // Larger tolerance for quantized
    
    println!("\n  Q stats:");
    let q_min = manual_q.iter().cloned().fold(f32::INFINITY, f32::min);
    let q_max = manual_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Manual: min={:.4}, max={:.4}, first5={:?}", q_min, q_max, &manual_q[..5]);
    let bq_min = backend_q_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let bq_max = backend_q_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Backend: min={:.4}, max={:.4}, first5={:?}", bq_min, bq_max, &backend_q_data[..5]);
    println!();

    println!("--- Test 3: RoPE (position 1) ---");
    
    let pos = 1;
    
    // Manual: apply rope to first head
    let mut manual_rope_out = backend_q_data[..head_dim].to_vec();
    manual_rope(&mut manual_rope_out, pos, head_dim, freq_base);
    
    // Backend: apply rope
    let mut backend_q_rope = Tensor::from_f32(backend_q_data, vec![num_heads, 1, head_dim]).unwrap();
    let mut dummy_k = Tensor::zeros(vec![1, 1, head_dim], DType::F32);
    backend.rope(&mut backend_q_rope, &mut dummy_k, pos, freq_base, 1.0, true).unwrap();
    let backend_rope_data = backend_q_rope.as_f32().unwrap();
    let backend_rope_head0 = &backend_rope_data[..head_dim];
    
    compare_vecs("RoPE (head 0, pos 1)", &manual_rope_out, backend_rope_head0, 1e-5);
    
    println!("\n  RoPE first 10 values:");
    println!("  Manual: {:?}", &manual_rope_out[..10].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("  Backend: {:?}", &backend_rope_head0[..10].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!();

    println!("--- Test 4: Full Linear forward (with bias) ---");
    
    // Load bias
    if let Some(q_bias_info) = gguf.data.get_tensor("blk.0.attn_q.bias") {
        let q_bias = load_tensor(&gguf, "blk.0.attn_q.bias");
        let q_bias_data = q_bias.as_f32().unwrap();
        
        // Manual: Q + bias
        let manual_q_biased: Vec<f32> = manual_q.iter().zip(q_bias_data.iter())
            .map(|(q, b)| q + b)
            .collect();
        
        // Backend: use Linear layer
        use llama_rs::model::layers::Linear;
        let normed_for_linear = Tensor::from_f32(&manual_normed, vec![hidden_size]).unwrap();
        let linear = Linear::new(wq.clone(), Some(q_bias.clone())).expect("Failed to create Linear");
        let mut backend_q_biased = Tensor::zeros(vec![num_heads * head_dim], DType::F32);
        linear.forward(&normed_for_linear, &mut backend_q_biased, &backend).unwrap();
        let backend_q_biased_data = backend_q_biased.as_f32().unwrap();
        
        compare_vecs("Linear (Q + bias)", &manual_q_biased, backend_q_biased_data, 1e-3);
        
        println!("\n  Q with bias stats:");
        let mqb_min = manual_q_biased.iter().cloned().fold(f32::INFINITY, f32::min);
        let mqb_max = manual_q_biased.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  Manual: min={:.4}, max={:.4}", mqb_min, mqb_max);
        let bqb_min = backend_q_biased_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let bqb_max = backend_q_biased_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  Backend: min={:.4}, max={:.4}", bqb_min, bqb_max);
    }

    println!("\n=== Summary ===");
    println!("If all tests pass with small differences, the backend operations");
    println!("match manual computation and the bug is elsewhere.");
}
