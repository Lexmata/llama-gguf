//! Validate that our inference produces sensible intermediate values.

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn load_tensor(gguf: &GgufFile, name: &str) -> Tensor {
    let tensor_info = gguf.data.get_tensor(name).unwrap();
    let tensor_data = gguf.tensor_data(name).unwrap();
    let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
    let dtype = DType::from(tensor_info.dtype);
    Tensor::new(tensor_data.to_vec(), shape, dtype).unwrap()
}

fn dequant(backend: &CpuBackend, t: &Tensor) -> Vec<f32> {
    let numel = t.numel();
    let mut out = Tensor::zeros(vec![numel], DType::F32);
    backend.dequantize(t, &mut out).unwrap();
    out.as_f32().unwrap().to_vec()
}

fn stats(x: &[f32]) -> (f32, f32, f32, f32) {
    let min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let variance: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let std = variance.sqrt();
    (min, max, mean, std)
}

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    println!("=== Validating Embedding ===");
    
    let emb_tensor = load_tensor(&gguf, "token_embd.weight");
    println!("Embedding tensor shape: {:?}, dtype: {:?}", emb_tensor.shape(), emb_tensor.dtype());
    
    let emb = dequant(&backend, &emb_tensor);
    let (e_min, e_max, e_mean, e_std) = stats(&emb);
    println!("Embedding stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", e_min, e_max, e_mean, e_std);
    
    // Check a few specific embeddings
    let hidden_size = 896;
    let vocab_size = 151936;
    
    println!();
    println!("Sample embeddings:");
    for tok in [0, 1, 16, 17, 28, 100, 1000] {
        let start = tok * hidden_size;
        let end = start + hidden_size;
        let tok_emb = &emb[start..end];
        let (t_min, t_max, t_mean, t_std) = stats(tok_emb);
        println!("  Token {:5}: min={:+.4}, max={:+.4}, mean={:+.4}, std={:.4}, first3={:.4?}", 
            tok, t_min, t_max, t_mean, t_std, &tok_emb[..3]);
    }
    
    // Verify that different tokens have different embeddings
    println!();
    println!("Verifying token embeddings are distinct:");
    let emb_16 = &emb[16 * hidden_size..(16 + 1) * hidden_size];
    let emb_17 = &emb[17 * hidden_size..(17 + 1) * hidden_size];
    let emb_28 = &emb[28 * hidden_size..(28 + 1) * hidden_size];
    
    let diff_16_17: f32 = emb_16.iter().zip(emb_17.iter()).map(|(a, b)| (a - b).abs()).sum();
    let diff_16_28: f32 = emb_16.iter().zip(emb_28.iter()).map(|(a, b)| (a - b).abs()).sum();
    let diff_17_28: f32 = emb_17.iter().zip(emb_28.iter()).map(|(a, b)| (a - b).abs()).sum();
    
    println!("  |emb[16] - emb[17]| = {:.4} (should be > 0)", diff_16_17);
    println!("  |emb[16] - emb[28]| = {:.4} (should be > 0)", diff_16_28);
    println!("  |emb[17] - emb[28]| = {:.4} (should be > 0)", diff_17_28);
    
    // Check output projection
    println!();
    println!("=== Validating Output Projection ===");
    
    let out_tensor = load_tensor(&gguf, "output.weight");
    println!("Output weight shape: {:?}, dtype: {:?}", out_tensor.shape(), out_tensor.dtype());
    
    let out_w = dequant(&backend, &out_tensor);
    let (o_min, o_max, o_mean, o_std) = stats(&out_w);
    println!("Output weight stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", o_min, o_max, o_mean, o_std);
    
    // Check specific rows (for tokens 16, 17, 28)
    // For GGUF shape [896, 151936], row j (for output token j) is at data[j*896..(j+1)*896]
    println!();
    println!("Sample output rows (for computing logits):");
    for tok in [16, 17, 28, 0, 1] {
        let start = tok * hidden_size;
        let end = start + hidden_size;
        let row = &out_w[start..end];
        let (r_min, r_max, r_mean, r_std) = stats(row);
        println!("  Token {:5}: min={:+.4}, max={:+.4}, mean={:+.4}, std={:.4}", 
            tok, r_min, r_max, r_mean, r_std);
    }
    
    // Verify output weights vary between tokens
    let row_16 = &out_w[16 * hidden_size..17 * hidden_size];
    let row_17 = &out_w[17 * hidden_size..18 * hidden_size];
    let diff_rows: f32 = row_16.iter().zip(row_17.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!();
    println!("  |out_row[16] - out_row[17]| = {:.4} (should be > 0)", diff_rows);
    
    // Manual logit computation test
    println!();
    println!("=== Manual Logit Test ===");
    
    // Use a random-ish hidden state
    let test_hidden: Vec<f32> = (0..hidden_size).map(|i| ((i as f32 * 0.01).sin())).collect();
    let (h_min, h_max, h_mean, _) = stats(&test_hidden);
    println!("Test hidden: min={:.4}, max={:.4}, mean={:.4}", h_min, h_max, h_mean);
    
    // Compute logit for token 17
    let logit_17: f32 = test_hidden.iter().zip(row_17.iter()).map(|(h, w)| h * w).sum();
    println!("Logit for token 17: {:.4}", logit_17);
    
    // Compute all logits
    let mut logits = vec![0.0f32; vocab_size];
    for j in 0..vocab_size {
        let row_start = j * hidden_size;
        let mut sum = 0.0f32;
        for i in 0..hidden_size {
            sum += test_hidden[i] * out_w[row_start + i];
        }
        logits[j] = sum;
    }
    
    let (l_min, l_max, l_mean, l_std) = stats(&logits);
    println!("All logits: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", l_min, l_max, l_mean, l_std);
    
    // Find top predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 5 predictions:");
    for i in 0..5 {
        println!("  {}: token {} = {:.4}", i + 1, indexed[i].0, indexed[i].1);
    }
}
