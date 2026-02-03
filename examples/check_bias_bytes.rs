//! Check raw bias bytes to verify we're reading them correctly.

use llama_rs::gguf::GgufFile;
use std::path::Path;

fn main() {
    let gguf = GgufFile::open(Path::new("/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        .expect("Failed to open GGUF");
    
    let bias_name = "blk.0.attn_q.bias";
    let info = gguf.data.get_tensor(bias_name).expect("No tensor found");
    
    println!("Tensor: {}", bias_name);
    println!("Dims: {:?}", info.dims);
    println!("DType: {:?} (0=F32, 1=F16, ...)", info.dtype);
    
    let data = gguf.tensor_data(bias_name).expect("No data");
    println!("Data length: {} bytes", data.len());
    
    // Show first 40 bytes as hex
    println!("\nFirst 40 bytes (hex):");
    for (i, byte) in data[..40.min(data.len())].iter().enumerate() {
        print!("{:02x} ", byte);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();
    
    // Interpret as f32
    println!("\nFirst 10 values as f32:");
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
    };
    for (i, &val) in floats[..10].iter().enumerate() {
        let bytes = val.to_le_bytes();
        println!("  [{}]: {:.6} (bytes: {:02x} {:02x} {:02x} {:02x})", 
            i, val, bytes[0], bytes[1], bytes[2], bytes[3]);
    }
    
    // Check for any patterns
    let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = floats.iter().sum::<f32>() / floats.len() as f32;
    let sum_sq: f32 = floats.iter().map(|x| x * x).sum();
    let rms = (sum_sq / floats.len() as f32).sqrt();
    
    println!("\nStatistics:");
    println!("  Count: {}", floats.len());
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}", max);
    println!("  Mean: {:.6}", mean);
    println!("  RMS: {:.6}", rms);
    
    // Count large values
    let large_count = floats.iter().filter(|&&x| x.abs() > 10.0).count();
    println!("  Values with |x| > 10: {} ({:.1}%)", 
        large_count, 100.0 * large_count as f32 / floats.len() as f32);
    
    // Print some of the large values and their indices
    println!("\nLarge values (|x| > 10):");
    for (i, &val) in floats.iter().enumerate() {
        if val.abs() > 10.0 {
            println!("  [{}]: {:.4}", i, val);
            if i > 20 {
                println!("  ... (truncated)");
                break;
            }
        }
    }
    
    // Also check K bias
    println!("\n=== K Bias ===");
    let k_bias_name = "blk.0.attn_k.bias";
    if let Some(info) = gguf.data.get_tensor(k_bias_name) {
        println!("Dims: {:?}, DType: {:?}", info.dims, info.dtype);
        
        if let Some(data) = gguf.tensor_data(k_bias_name) {
            let k_floats: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
            };
            
            let min = k_floats.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = k_floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let large_count = k_floats.iter().filter(|&&x| x.abs() > 10.0).count();
            
            println!("Min: {:.4}, Max: {:.4}", min, max);
            println!("Values with |x| > 10: {} ({:.1}%)", 
                large_count, 100.0 * large_count as f32 / k_floats.len() as f32);
            
            // Print first 10
            println!("First 10: {:?}", &k_floats[..10.min(k_floats.len())].iter()
                .map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
        }
    }
}
