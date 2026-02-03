//! Verify Q6_K dequantization against known values

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let backend = CpuBackend::new();
    let model = load_llama_model(model_path)?;
    
    // Get layer 0's FFN down weight
    let layer = &model.layers()[0];
    let w_down = &layer.ffn.w_down.weight;
    
    println!("FFN down weight:");
    println!("  Shape: {:?}", w_down.shape());
    println!("  DType: {:?}", w_down.dtype());
    println!("  Data size: {} bytes", w_down.data().len());
    
    // Dequantize
    let mut w_down_f32 = Tensor::zeros(w_down.shape().to_vec(), DType::F32);
    backend.dequantize(w_down, &mut w_down_f32)?;
    
    let data = w_down_f32.as_f32()?;
    
    // Statistics
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    
    println!("\nDequantized stats:");
    println!("  min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
    println!("  Total elements: {}", data.len());
    
    // Print first row (should be input_features = 4864 elements for each output)
    // Shape should be [in_features, out_features] = [4864, 896] based on GGUF convention
    let in_features = w_down.shape()[0];
    let out_features = w_down.shape()[1];
    println!("\n  in_features (shape[0]): {}", in_features);
    println!("  out_features (shape[1]): {}", out_features);
    
    // First few elements of row 0 (first out_features for in=0)
    // In column-major: W[i,j] = data[i + j * in_features]
    println!("\nFirst row (in=0) first 10 elements:");
    for j in 0..10.min(out_features) {
        let idx = 0 + j * in_features;
        print!("{:.6}, ", data[idx]);
    }
    println!();
    
    // First few elements of column 0 (first in_features for out=0)
    println!("\nFirst column (out=0) first 10 elements:");
    for i in 0..10.min(in_features) {
        let idx = i + 0 * in_features;
        print!("{:.6}, ", data[idx]);
    }
    println!();
    
    // Now let's test the vec_mat computation
    // Create a test input of [4864] and check the output
    println!("\n=== Testing vec_mat with FFN down ===");
    
    let intermediate_size = in_features;  // 4864
    let hidden_size = out_features;  // 896
    
    // Create a simple test input (all ones)
    let mut test_input = Tensor::zeros(vec![intermediate_size], DType::F32);
    {
        let inp_data = test_input.as_f32_mut()?;
        for i in 0..intermediate_size {
            inp_data[i] = 1.0;  // Simple: sum of each column
        }
    }
    
    let mut test_output = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.ffn.w_down.forward(&test_input, &mut test_output, &backend)?;
    
    let out_data = test_output.as_f32()?;
    let out_min = out_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let out_max = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out_mean = out_data.iter().sum::<f32>() / out_data.len() as f32;
    
    println!("Output from vec_mat with all-ones input:");
    println!("  Shape: {}", out_data.len());
    println!("  min={:.6}, max={:.6}, mean={:.6}", out_min, out_max, out_mean);
    println!("  First 10: {:?}", &out_data[..10]);
    
    // The output[j] should be sum of column j, which is sum_i(W[i,j])
    // Let's verify by computing manually
    println!("\nManual verification of first output:");
    let mut manual_sum = 0.0f32;
    for i in 0..intermediate_size {
        manual_sum += data[i + 0 * in_features];  // Column 0
    }
    println!("  Manual sum of column 0: {:.6}", manual_sum);
    println!("  vec_mat output[0]: {:.6}", out_data[0]);
    
    if (manual_sum - out_data[0]).abs() < 0.01 {
        println!("  MATCH!");
    } else {
        println!("  MISMATCH!");
    }
    
    Ok(())
}
