//! Check if Q/K heads are organized as contiguous or interleaved.
//!
//! Contiguous: [head0_dim0, head0_dim1, ..., head0_dim63, head1_dim0, ...]
//! Interleaved: [head0_dim0, head1_dim0, ..., headN_dim0, head0_dim1, ...]

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();

    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;

    println!("=== Q Weight Layout Analysis ===");
    println!();

    // Load Q weight
    let wq_info = gguf.data.get_tensor("blk.0.attn_q.weight").unwrap();
    let wq_data = gguf.tensor_data("blk.0.attn_q.weight").unwrap();
    let wq_shape: Vec<usize> = wq_info.dims.iter().map(|&d| d as usize).collect();
    let wq = Tensor::new(wq_data.to_vec(), wq_shape, DType::from(wq_info.dtype)).unwrap();
    
    let mut wq_f32 = Tensor::zeros(vec![wq.numel()], DType::F32);
    backend.dequantize(&wq, &mut wq_f32).unwrap();
    let wq_vals = wq_f32.as_f32().unwrap();

    println!("Wq shape (GGUF): {:?}", wq_info.dims);
    println!("Total elements: {}", wq_vals.len());
    println!();

    // The weight matrix is [hidden_size, num_heads * head_dim] = [896, 896]
    // For y = x @ W, the j-th output element is sum_i(x[i] * W[i + j * hidden_size])
    //
    // If heads are CONTIGUOUS:
    //   - Output columns 0-63 belong to head 0
    //   - Output columns 64-127 belong to head 1
    //   - etc.
    //
    // If heads are INTERLEAVED:
    //   - Column 0, 14, 28, ... belong to dimension 0 of each head
    //   - Column 1, 15, 29, ... belong to dimension 1 of each head

    // To check: look at the pattern of the weights
    // For contiguous layout, weights for head 0 (columns 0-63) should be
    // independent of weights for head 1 (columns 64-127)

    // Let's compute the mean absolute value for each "head" under both assumptions
    
    println!("Under CONTIGUOUS assumption (columns 0-63 = head 0, etc.):");
    for head in 0..3 {
        let start_col = head * head_dim;
        let mut sum = 0.0f32;
        let mut count = 0;
        for col in start_col..start_col + head_dim {
            for row in 0..hidden_size {
                sum += wq_vals[row + col * hidden_size].abs();
                count += 1;
            }
        }
        println!("  Head {}: mean|w| = {:.6}", head, sum / count as f32);
    }
    println!();

    println!("Under INTERLEAVED assumption (every 14th column = same dim position):");
    for dim_pos in 0..3 {
        let mut sum = 0.0f32;
        let mut count = 0;
        for head in 0..num_heads {
            let col = dim_pos * num_heads + head;
            for row in 0..hidden_size {
                sum += wq_vals[row + col * hidden_size].abs();
                count += 1;
            }
        }
        println!("  Dim {}: mean|w| = {:.6}", dim_pos, sum / count as f32);
    }
    println!();

    // Actually, let's check the bias pattern which is clearer
    println!("=== Q Bias Pattern ===");
    let qb_info = gguf.data.get_tensor("blk.0.attn_q.bias").unwrap();
    let qb_data = gguf.tensor_data("blk.0.attn_q.bias").unwrap();
    let qb: &[f32] = unsafe {
        std::slice::from_raw_parts(qb_data.as_ptr() as *const f32, qb_data.len() / 4)
    };

    println!("Q bias shape: {:?}", qb_info.dims);
    println!();

    // Under CONTIGUOUS: bias[0:64] = head 0, bias[64:128] = head 1, ...
    println!("Under CONTIGUOUS (bias[0:64] = head 0):");
    for head in 0..3 {
        let start = head * head_dim;
        let slice = &qb[start..start + head_dim];
        let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        println!("  Head {}: min={:.4}, max={:.4}, mean={:.4}", head, min, max, mean);
    }
    println!();

    // Under INTERLEAVED: bias[0], bias[14], bias[28], ... = dim 0 of each head
    println!("Under INTERLEAVED (bias[0,14,28,...] = dim 0 of each head):");
    for dim_pos in 0..3 {
        let values: Vec<f32> = (0..num_heads).map(|h| qb[dim_pos * num_heads + h]).collect();
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        println!("  Dim {} across heads: min={:.4}, max={:.4}, mean={:.4}", dim_pos, min, max, mean);
        println!("    Values: {:?}", values);
    }
    println!();

    // Look at the large bias values
    println!("Large bias values (|x| > 10):");
    for (i, &val) in qb.iter().enumerate() {
        if val.abs() > 10.0 {
            let head_if_contig = i / head_dim;
            let dim_if_contig = i % head_dim;
            let head_if_interleaved = i % num_heads;
            let dim_if_interleaved = i / num_heads;
            println!("  bias[{}] = {:.2}", i, val);
            println!("    Contiguous: head {}, dim {}", head_if_contig, dim_if_contig);
            println!("    Interleaved: head {}, dim {}", head_if_interleaved, dim_if_interleaved);
            if i > 20 {
                println!("  ...");
                break;
            }
        }
    }
}
