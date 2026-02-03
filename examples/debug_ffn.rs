//! Debug FFN computation to find the issue

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};

fn print_stats(name: &str, data: &[f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    println!("{}: min={:.4}, max={:.4}, mean={:.4}", name, min, max, mean);
    println!("  first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        data[0], data[1], data[2], data[3], data[4]);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    eprintln!("Loading model...");
    let backend = CpuBackend::new();
    let model = load_llama_model(model_path)?;
    
    let hidden_size = model.config().hidden_size;
    
    let token_id = 28u32;
    
    println!("=== Debugging FFN for token {} ===\n", token_id);
    
    // Get the input to FFN (this is the normed hidden state after attention)
    // For simplicity, let's create a test input similar to what we saw:
    // After ffn_norm: min=-3.0927, max=5.0278
    
    // First, compute the actual input
    let embedding = model.embed_tokens(&[token_id], &backend)?;
    let embedding = embedding.reshape(vec![hidden_size])?;
    
    let layer = &model.layers()[0];
    
    // Attention norm
    let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.attn_norm.forward(&embedding, &mut normed, &backend)?;
    
    // Get V and attention output
    let x_vec = normed.reshape(vec![hidden_size])?;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    
    let mut v = Tensor::zeros(vec![num_kv_heads * head_dim], DType::F32);
    layer.attention.wv.forward(&x_vec, &mut v, &backend)?;
    
    // At pos=0, attention output = V replicated
    let v_data = v.as_f32()?;
    let v_heads: Vec<_> = v_data.chunks(head_dim).collect();
    
    let mut attn_out = vec![0.0f32; num_heads * head_dim];
    let num_q_per_kv = num_heads / num_kv_heads;
    for h in 0..num_heads {
        let kv_h = h / num_q_per_kv;
        let src_start = kv_h * head_dim;
        let dst_start = h * head_dim;
        attn_out[dst_start..dst_start + head_dim].copy_from_slice(&v_data[src_start..src_start + head_dim]);
    }
    
    // Output projection
    let attn_out_tensor = Tensor::from_f32(&attn_out, vec![num_heads * head_dim])?;
    let mut output = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.attention.wo.forward(&attn_out_tensor, &mut output, &backend)?;
    
    // First residual
    let emb_data = embedding.as_f32()?;
    let output_data = output.as_f32()?;
    let hidden_after_attn: Vec<f32> = emb_data.iter().zip(output_data.iter())
        .map(|(e, o)| e + o).collect();
    
    let hidden_after_attn_tensor = Tensor::from_f32(&hidden_after_attn, vec![hidden_size])?;
    
    // FFN norm
    let mut normed_ffn = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.ffn_norm.forward(&hidden_after_attn_tensor, &mut normed_ffn, &backend)?;
    
    let normed_ffn_data = normed_ffn.as_f32()?;
    print_stats("FFN Input (after ffn_norm)", normed_ffn_data);
    
    // Now trace through FFN step by step
    println!("\n--- FFN Step-by-Step ---");
    
    let intermediate_size = layer.ffn.intermediate_size;
    println!("intermediate_size = {}", intermediate_size);
    
    // Gate projection
    let mut gate_out = Tensor::zeros(vec![intermediate_size], DType::F32);
    layer.ffn.w_gate.forward(&normed_ffn, &mut gate_out, &backend)?;
    
    let gate_data = gate_out.as_f32()?;
    print_stats("Gate projection", gate_data);
    
    // Up projection  
    let mut up_out = Tensor::zeros(vec![intermediate_size], DType::F32);
    layer.ffn.w_up.forward(&normed_ffn, &mut up_out, &backend)?;
    
    let up_data = up_out.as_f32()?;
    print_stats("Up projection", up_data);
    
    // SiLU activation on gate
    let silu_gate: Vec<f32> = gate_data.iter()
        .map(|&x| x / (1.0 + (-x).exp()))
        .collect();
    print_stats("SiLU(gate)", &silu_gate);
    
    // Element-wise multiply: silu(gate) * up
    let swiglu: Vec<f32> = silu_gate.iter().zip(up_data.iter())
        .map(|(g, u)| g * u)
        .collect();
    print_stats("SwiGLU (silu(gate) * up)", &swiglu);
    
    // Down projection
    let swiglu_tensor = Tensor::from_f32(&swiglu, vec![intermediate_size])?;
    let mut down_out = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.ffn.w_down.forward(&swiglu_tensor, &mut down_out, &backend)?;
    
    let down_data = down_out.as_f32()?;
    print_stats("Down projection (FFN output)", down_data);
    
    // What does the full FFN.forward produce?
    println!("\n--- Full FFN.forward result ---");
    let mut ffn_full_out = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.ffn.forward(&normed_ffn, &mut ffn_full_out, &backend)?;
    
    let ffn_full_data = ffn_full_out.as_f32()?;
    print_stats("FFN.forward output", ffn_full_data);
    
    // Compare step-by-step vs forward
    let diff: f32 = down_data.iter().zip(ffn_full_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("\nDifference (step-by-step vs forward): {:.6}", diff);
    
    // Python reference
    println!("\n=== PYTHON REFERENCE ===");
    println!("After layer 0: min=-4.4456, max=4.0055");
    println!("Expected FFN to produce large values, not small ones!");
    
    Ok(())
}
