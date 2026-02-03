//! Check weight dimensions and first values

use llama_rs::backend::cpu::CpuBackend;
use llama_rs::backend::Backend;
use llama_rs::gguf::GgufFile;
use llama_rs::tensor::{DType, Tensor};
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(&model_path)).expect("Failed to open GGUF");
    let backend = CpuBackend::new();
    
    let tensors_to_check = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ];
    
    for name in &tensors_to_check {
        if let Some(info) = gguf.data.get_tensor(name) {
            println!("{}: dims={:?}, dtype={:?}", name, info.dims, info.dtype);
        } else {
            println!("{}: NOT FOUND", name);
        }
    }
    
    // Now check the actual data for attn_output.weight
    if let Some(info) = gguf.data.get_tensor("blk.0.attn_output.weight") {
        if let Some(data) = gguf.tensor_data("blk.0.attn_output.weight") {
            let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
            let dtype = DType::from(info.dtype);
            let tensor = Tensor::new(data.to_vec(), shape.clone(), dtype).unwrap();
            
            let numel = tensor.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            backend.dequantize(&tensor, &mut dequant).expect("dequant failed");
            let weights = dequant.as_f32().unwrap();
            
            println!("\nattn_output.weight dequantized:");
            println!("  First 10: {:?}", &weights[..10]);
            println!("  Min: {:.6}, Max: {:.6}", 
                weights.iter().cloned().fold(f32::INFINITY, f32::min),
                weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }
    }
}
