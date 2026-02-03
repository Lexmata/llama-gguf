
use llama_rs::gguf::GgufFile;
use std::path::Path;

fn main() {
    let gguf = GgufFile::open(Path::new("/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        .expect("Failed to open GGUF");
    
    // Check if bias tensor exists and read values
    let bias_name = "blk.0.attn_q.bias";
    if let Some(info) = gguf.data.get_tensor(bias_name) {
        println!("Found tensor '{}': dims={:?}, dtype={:?}", bias_name, info.dims, info.dtype);
        
        // Get tensor data
        if let Some(data) = gguf.tensor_data(bias_name) {
            // It's F32, so interpret as floats
            let floats: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const f32,
                    data.len() / 4
                )
            };
            
            let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = floats.iter().sum::<f32>() / floats.len() as f32;
            
            println!("Q bias: len={}, min={:.4}, max={:.4}, mean={:.6}", 
                floats.len(), min, max, mean);
            println!("Q bias first 10: {:?}", &floats[..10.min(floats.len())]);
        } else {
            println!("Could not get tensor data for {}", bias_name);
        }
    } else {
        println!("Tensor '{}' NOT FOUND", bias_name);
    }
    
    // Also check K bias
    let k_bias_name = "blk.0.attn_k.bias";
    if let Some(info) = gguf.data.get_tensor(k_bias_name) {
        println!("\nFound tensor '{}': dims={:?}, dtype={:?}", k_bias_name, info.dims, info.dtype);
        
        if let Some(data) = gguf.tensor_data(k_bias_name) {
            let floats: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const f32,
                    data.len() / 4
                )
            };
            
            let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            println!("K bias: len={}, min={:.4}, max={:.4}", floats.len(), min, max);
            println!("K bias first 10: {:?}", &floats[..10.min(floats.len())]);
        }
    }
    
    println!("\nTotal tensors: {}", gguf.data.tensors.len());
}
