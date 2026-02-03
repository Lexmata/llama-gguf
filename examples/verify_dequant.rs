//! Verify our dequantization matches Python

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;
use std::io::Read;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    // Load output.weight tensor
    let info = gguf.data.get_tensor("output.weight").expect("tensor not found");
    let data = gguf.tensor_data("output.weight").expect("data not found");
    let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    
    println!("output.weight tensor:");
    println!("  GGUF shape (ne): {:?}", shape);  // Should be [896, 151936]
    println!("  dtype: {:?}", DType::from(info.dtype));
    println!("  raw data len: {} bytes", data.len());
    
    // Expected: 896 * 151936 / 32 * 34 = 144643072 bytes for Q8_0
    let expected_bytes = 896 * 151936 / 32 * 34;
    println!("  expected bytes (Q8_0): {}", expected_bytes);
    
    // Create tensor
    let tensor = Tensor::new(data.to_vec(), shape.clone(), DType::from(info.dtype)).unwrap();
    
    // Dequantize
    let total_elements = 896 * 151936;
    let mut out = Tensor::zeros(vec![total_elements], DType::F32);
    backend.dequantize(&tensor, &mut out).expect("dequantize failed");
    
    let dequant = out.as_f32().unwrap();
    println!("  dequantized len: {}", dequant.len());
    
    // Our dequant is a flat array
    // Question: what's the indexing?
    // Option A: dequant[j * 896 + i] = weight for vocab j, hidden i
    // Option B: dequant[i * 151936 + j] = weight for hidden i, vocab j
    
    // Let's check by looking at the first 896 values (which would be vocab 0 under option A)
    println!("\n=== Checking dequantization layout ===");
    
    // First 5 dequantized values
    println!("First 5 values: {:?}", &dequant[..5]);
    
    // Load Python reference
    let ref_path = "/tmp/output_weight_row0.npy";
    if let Ok(mut file) = std::fs::File::open(ref_path) {
        let mut npy_data = Vec::new();
        file.read_to_end(&mut npy_data).unwrap();
        
        // Simple NPY parsing - skip header and read f64 values
        // NPY header ends with '\n', then data follows
        let header_end = npy_data.iter().position(|&b| b == b'\n').unwrap_or(0) + 1;
        
        // Actually NPY header can be more complex. Let's just read the float data
        // The header describes dtype and shape. For float64, each value is 8 bytes
        // For float32, each value is 4 bytes
        
        // Python saves as float64 by default
        if npy_data.len() > 128 {  // Sanity check
            // Try to find the actual data after the header
            // NPY v1.0 header: 6 bytes magic + 2 bytes version + 2 bytes header_len + header + data
            let magic = &npy_data[..6];
            if magic == b"\x93NUMPY" {
                let _version = (npy_data[6], npy_data[7]);
                let header_len = u16::from_le_bytes([npy_data[8], npy_data[9]]) as usize;
                let data_start = 10 + header_len;
                
                // Data is float64
                let float_data: Vec<f64> = npy_data[data_start..]
                    .chunks_exact(8)
                    .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                
                println!("\nPython reference (row 0 = vocab token 0):");
                println!("  First 5: {:?}", &float_data[..5]);
                
                // Compare our first 896 values with Python's row 0
                let our_row0: Vec<f64> = dequant[..896].iter().map(|&x| x as f64).collect();
                let py_row0: Vec<f64> = float_data[..896].to_vec();
                
                // Check if they match
                let mut match_count = 0;
                let mut max_diff = 0.0f64;
                for (i, (&ours, &theirs)) in our_row0.iter().zip(py_row0.iter()).enumerate() {
                    let diff = (ours - theirs).abs();
                    if diff < 1e-4 {
                        match_count += 1;
                    }
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if i < 5 {
                        println!("  [{}] ours={:.6}, py={:.6}, diff={:.6}", i, ours, theirs, diff);
                    }
                }
                
                println!("\n  Matching values (< 1e-4 diff): {}/{}", match_count, 896);
                println!("  Max difference: {:.6}", max_diff);
                
                if match_count == 896 {
                    println!("\n  ✓ Layout confirmed: dequant[j * 896 + i] = weight for vocab j, hidden i");
                } else {
                    println!("\n  ✗ Layout mismatch! Let's check alternative indexing...");
                    
                    // Try alternative: maybe our layout is [hidden, vocab] instead of [vocab, hidden]
                    // Then dequant[i + j * hidden_size] for hidden i, vocab j
                    // To get vocab 0's weights: dequant[0], dequant[896], dequant[2*896], ...
                    let alt_row0: Vec<f64> = (0..896).map(|i| dequant[i * 151936] as f64).collect();
                    
                    let mut alt_match = 0;
                    for (&ours, &theirs) in alt_row0.iter().zip(py_row0.iter()) {
                        if (ours - theirs).abs() < 1e-4 {
                            alt_match += 1;
                        }
                    }
                    println!("  Alternative layout match: {}/{}", alt_match, 896);
                }
            }
        }
    } else {
        println!("Could not load Python reference - run compare_dequant_values.py first");
    }
    
    // Also compute what the logit for vocab 0 would be with a simple hidden state
    println!("\n=== Test logit computation ===");
    // Use a simple hidden state of all 1.0s
    let hidden: Vec<f32> = vec![1.0; 896];
    
    // Compute logit for vocab 0 using first 896 dequantized weights
    let logit_0: f32 = hidden.iter().zip(&dequant[..896]).map(|(h, w)| h * w).sum();
    println!("Logit for vocab 0 with hidden=[1.0; 896]: {:.4}", logit_0);
    
    // Also compute for vocab 17
    let logit_17: f32 = hidden.iter().zip(&dequant[17*896..18*896]).map(|(h, w)| h * w).sum();
    println!("Logit for vocab 17 with hidden=[1.0; 896]: {:.4}", logit_17);
    
    // These should match the Python row sums
    println!("\n(Python row 0 sum was ~0.227, row 17 sum was ~0.220)");
}
