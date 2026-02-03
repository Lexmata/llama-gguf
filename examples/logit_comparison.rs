//! Compare logit distributions to understand the difference.

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

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    let hidden_size = 896;
    let vocab_size = 151936;
    let eps = 1e-6f32;

    println!("=== Logit Analysis ===");
    println!();

    // Load final layer weights
    let output_norm_w = dequant(&backend, &load_tensor(&gguf, "output_norm.weight"));
    let output_w = dequant(&backend, &load_tensor(&gguf, "output.weight"));

    // Our hidden state from layer_by_layer_debug (position 3, after all layers)
    // From the output: L23 hidden: min=-10.4084, max=10.5080
    // Let me re-run to get the exact values

    println!("Loading layer_by_layer_debug output values...");
    
    // From layer_by_layer_debug.rs final output:
    // Final normed: min=-24.6017, max=31.2524, mean=0.164492
    // First 10: [-0.6544, -8.7116, -4.5684, -6.2143, 6.2871, -9.2824, 12.671, 3.2369, 0.708, -8.9773]
    
    // Reference from llama.cpp:
    // Logits: min=-8.6063, max=17.9867, mean=-0.382661
    
    println!();
    println!("Comparison of logit statistics:");
    println!("                     Our impl     llama.cpp    Diff");
    println!("  Logits min:        -16.89       -8.61        -8.28");
    println!("  Logits max:         11.69       17.99        -6.30");
    println!("  Logits mean:        -1.72       -0.38        -1.34");
    println!();
    
    println!("Key observations:");
    println!("1. Our logits are shifted DOWN by ~1.3 on average");
    println!("2. Our max logit is 6.3 lower than llama.cpp");
    println!("3. Our min logit is 8.3 lower than llama.cpp");
    println!();
    
    println!("This systematic shift suggests either:");
    println!("a) Our hidden state has wrong direction/magnitude");
    println!("b) Our output projection is computed differently");
    println!("c) There's a missing normalization or scaling step");
    println!();

    // Let me check the output projection statistics
    println!("Output projection analysis:");
    
    let (w_min, w_max) = (
        output_w.iter().cloned().fold(f32::INFINITY, f32::min),
        output_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    let w_mean: f32 = output_w.iter().sum::<f32>() / output_w.len() as f32;
    let w_rms = (output_w.iter().map(|x| x * x).sum::<f32>() / output_w.len() as f32).sqrt();
    
    println!("  Output weight: min={:.4}, max={:.4}, mean={:.6}, rms={:.4}", 
        w_min, w_max, w_mean, w_rms);
    println!("  Output weight shape: [hidden_size={}, vocab_size={}]", hidden_size, vocab_size);
    
    // Check norm weight
    let (nw_min, nw_max) = (
        output_norm_w.iter().cloned().fold(f32::INFINITY, f32::min),
        output_norm_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    let nw_mean: f32 = output_norm_w.iter().sum::<f32>() / output_norm_w.len() as f32;
    println!("  Output norm weight: min={:.4}, max={:.4}, mean={:.4}", nw_min, nw_max, nw_mean);
    
    println!();
    println!("Token ranking comparison:");
    println!("  Token   Our logit  Our rank   llama.cpp logit  llama.cpp rank");
    println!("  ----------------------------------------------------------------");
    println!("  17(2)   -3.57      114226     17.99            1");
    println!("  16(1)   ?          ?          17.54            2");
    println!("  18(3)   ?          ?          17.16            3");
    println!("  1402(AM) 11.69     1          3.85             ?");
    println!();
    
    println!("The difference between token 17 and 1402:");
    println!("  Our impl:   17 vs 1402 = -3.57 - 11.69 = -15.26");
    println!("  llama.cpp:  17 vs 1402 = 17.99 - 3.85  = +14.14");
    println!();
    println!("The relative ordering is COMPLETELY INVERTED!");
    println!();
    
    println!("This suggests the hidden state direction is fundamentally wrong,");
    println!("not just a simple scaling issue.");
}
