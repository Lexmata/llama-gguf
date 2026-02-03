//! Verify attention computation produces mathematically correct results.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::tensor::{DType, Tensor};

fn main() {
    let backend = CpuBackend::new();
    
    println!("=== Attention Math Verification ===");
    println!();
    
    // Simple test case: 2 heads, 4-dim, 2 KV positions
    let num_heads = 2;
    let seq_len = 1;  // Query has 1 position
    let head_dim = 4;
    let num_kv_heads = 2;  // No GQA for simplicity
    let kv_len = 2;  // 2 KV positions
    let scale = 1.0 / (head_dim as f32).sqrt();  // 0.5
    
    // Create Q: [num_heads, seq_len, head_dim] = [2, 1, 4]
    let q_data = vec![
        // Head 0, position 0
        1.0, 2.0, 3.0, 4.0,
        // Head 1, position 0
        -1.0, 0.0, 1.0, 2.0,
    ];
    let q = Tensor::from_f32(&q_data, vec![num_heads, seq_len, head_dim]).unwrap();
    
    // Create K: [num_kv_heads, kv_len, head_dim] = [2, 2, 4]
    let k_data = vec![
        // KV head 0, position 0
        1.0, 0.0, 0.0, 0.0,
        // KV head 0, position 1
        0.0, 1.0, 0.0, 0.0,
        // KV head 1, position 0
        0.0, 0.0, 1.0, 0.0,
        // KV head 1, position 1
        0.0, 0.0, 0.0, 1.0,
    ];
    let k = Tensor::from_f32(&k_data, vec![num_kv_heads, kv_len, head_dim]).unwrap();
    
    // Create V: [num_kv_heads, kv_len, head_dim] = [2, 2, 4]
    let v_data = vec![
        // KV head 0, position 0
        10.0, 0.0, 0.0, 0.0,
        // KV head 0, position 1
        0.0, 10.0, 0.0, 0.0,
        // KV head 1, position 0
        0.0, 0.0, 10.0, 0.0,
        // KV head 1, position 1
        0.0, 0.0, 0.0, 10.0,
    ];
    let v = Tensor::from_f32(&v_data, vec![num_kv_heads, kv_len, head_dim]).unwrap();
    
    // Create output tensor
    let mut out = Tensor::zeros(vec![num_heads, seq_len, head_dim], DType::F32);
    
    println!("Input shapes:");
    println!("  Q: {:?}", q.shape());
    println!("  K: {:?}", k.shape());
    println!("  V: {:?}", v.shape());
    println!("  scale: {}", scale);
    println!();
    
    // Expected computation:
    // Head 0 uses KV head 0 (since num_heads == num_kv_heads, no GQA)
    //   Q0 = [1, 2, 3, 4]
    //   K0 (pos 0) = [1, 0, 0, 0] -> score = 1*1 = 1, scaled = 0.5
    //   K0 (pos 1) = [0, 1, 0, 0] -> score = 2*1 = 2, scaled = 1.0
    //   softmax([0.5, 1.0]) = [exp(0.5-1), exp(1-1)] / sum = [0.6065, 1.0] / 1.6065 = [0.378, 0.622]
    //   output = 0.378 * V0_p0 + 0.622 * V0_p1 = 0.378*[10,0,0,0] + 0.622*[0,10,0,0]
    //          = [3.78, 6.22, 0, 0]
    
    // Head 1 uses KV head 1
    //   Q1 = [-1, 0, 1, 2]
    //   K1 (pos 0) = [0, 0, 1, 0] -> score = 1*1 = 1, scaled = 0.5
    //   K1 (pos 1) = [0, 0, 0, 1] -> score = 2*1 = 2, scaled = 1.0
    //   Same softmax: [0.378, 0.622]
    //   output = 0.378 * V1_p0 + 0.622 * V1_p1 = 0.378*[0,0,10,0] + 0.622*[0,0,0,10]
    //          = [0, 0, 3.78, 6.22]
    
    println!("Expected computation for head 0:");
    let score00 = (1.0 * 1.0 + 2.0 * 0.0 + 3.0 * 0.0 + 4.0 * 0.0) * scale;
    let score01 = (1.0 * 0.0 + 2.0 * 1.0 + 3.0 * 0.0 + 4.0 * 0.0) * scale;
    println!("  Score(pos0) = {} * {} = {}", 1.0, scale, score00);
    println!("  Score(pos1) = {} * {} = {}", 2.0, scale, score01);
    
    let max_score = score00.max(score01);
    let exp0 = (score00 - max_score).exp();
    let exp1 = (score01 - max_score).exp();
    let sum = exp0 + exp1;
    let w0 = exp0 / sum;
    let w1 = exp1 / sum;
    println!("  Softmax weights: [{:.4}, {:.4}]", w0, w1);
    
    let expected_h0 = [w0 * 10.0, w1 * 10.0, 0.0, 0.0];
    println!("  Expected output: {:?}", expected_h0);
    
    println!();
    println!("Expected computation for head 1:");
    let score10 = (-1.0 * 0.0 + 0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 0.0) * scale;
    let score11 = (-1.0 * 0.0 + 0.0 * 0.0 + 1.0 * 0.0 + 2.0 * 1.0) * scale;
    println!("  Score(pos0) = {} * {} = {}", 1.0, scale, score10);
    println!("  Score(pos1) = {} * {} = {}", 2.0, scale, score11);
    
    let max_score = score10.max(score11);
    let exp0 = (score10 - max_score).exp();
    let exp1 = (score11 - max_score).exp();
    let sum = exp0 + exp1;
    let w0 = exp0 / sum;
    let w1 = exp1 / sum;
    println!("  Softmax weights: [{:.4}, {:.4}]", w0, w1);
    
    let expected_h1 = [0.0, 0.0, w0 * 10.0, w1 * 10.0];
    println!("  Expected output: {:?}", expected_h1);
    
    // Run backend attention
    println!();
    println!("Running backend attention...");
    backend.attention(&q, &k, &v, &mut out, scale).unwrap();
    
    let out_data = out.as_f32().unwrap();
    println!();
    println!("Backend output:");
    println!("  Head 0: {:?}", &out_data[0..4]);
    println!("  Head 1: {:?}", &out_data[4..8]);
    
    // Check if they match
    let h0_match = out_data[0..4].iter().zip(expected_h0.iter())
        .all(|(a, b)| (a - b).abs() < 0.001);
    let h1_match = out_data[4..8].iter().zip(expected_h1.iter())
        .all(|(a, b)| (a - b).abs() < 0.001);
    
    println!();
    println!("Head 0 matches expected: {}", if h0_match { "✓" } else { "✗" });
    println!("Head 1 matches expected: {}", if h1_match { "✓" } else { "✗" });
    
    // Test with GQA
    println!();
    println!("=== GQA Test (4 query heads, 2 KV heads) ===");
    
    let num_heads = 4;
    let num_kv_heads = 2;
    // Heads 0, 1 share KV head 0
    // Heads 2, 3 share KV head 1
    
    let q_data = vec![
        // Head 0: [1, 0, 0, 0]
        1.0, 0.0, 0.0, 0.0,
        // Head 1: [0, 1, 0, 0]
        0.0, 1.0, 0.0, 0.0,
        // Head 2: [0, 0, 1, 0]
        0.0, 0.0, 1.0, 0.0,
        // Head 3: [0, 0, 0, 1]
        0.0, 0.0, 0.0, 1.0,
    ];
    let q = Tensor::from_f32(&q_data, vec![num_heads, 1, head_dim]).unwrap();
    
    let mut out = Tensor::zeros(vec![num_heads, 1, head_dim], DType::F32);
    
    backend.attention(&q, &k, &v, &mut out, scale).unwrap();
    let out_data = out.as_f32().unwrap();
    
    println!("GQA output (heads 0,1 use KV head 0; heads 2,3 use KV head 1):");
    for h in 0..num_heads {
        let start = h * head_dim;
        println!("  Head {}: {:?}", h, &out_data[start..start+head_dim]);
    }
}
