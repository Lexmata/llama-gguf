//! Debug Q6_K data layout

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
    
    let layer = &model.layers()[0];
    let w_down = &layer.ffn.w_down.weight;
    
    println!("FFN down weight:");
    println!("  Shape: {:?}", w_down.shape());
    println!("  DType: {:?}", w_down.dtype());
    
    // Dequantize
    let mut w_down_f32 = Tensor::zeros(w_down.shape().to_vec(), DType::F32);
    backend.dequantize(w_down, &mut w_down_f32)?;
    
    let data = w_down_f32.as_f32()?;
    
    // GGUF shape is [4864, 896]
    // In GGUF column-major, element (i, j) is at index i + j * 4864
    let in_features = 4864;
    let out_features = 896;
    
    println!("\nFirst 256 dequantized elements (should be first Q6_K block):");
    for i in 0..256 {
        if i < 10 {
            print!("{:.6}, ", data[i]);
        }
    }
    println!("...");
    
    // Statistics of first block
    let first_block: Vec<f32> = data[..256].to_vec();
    let min = first_block.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = first_block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("First block stats: min={:.6}, max={:.6}", min, max);
    
    // Compare with Python which shows:
    // First block first 10: [-0.006, -0.217, -0.196, -0.219, -0.005, -0.007, -0.018, -0.221, -0.219, -0.221]
    println!("\nPython first block first 10:");
    println!("[-0.006, -0.217, -0.196, -0.219, -0.005, -0.007, -0.018, -0.221, -0.219, -0.221]");
    
    Ok(())
}
