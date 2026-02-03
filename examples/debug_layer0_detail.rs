//! Debug layer 0 in detail to find divergence point
//!
//! Prints values at each step within layer 0

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};

fn print_stats(name: &str, data: &[f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
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
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let freq_base = model.config().rope_config.freq_base;
    
    let token_id = 28u32;
    let pos = 0usize;
    
    println!("=== Debugging Layer 0 for token {} ===\n", token_id);
    
    // Get embedding
    let embedding = model.embed_tokens(&[token_id], &backend)?;
    let embedding = embedding.reshape(vec![hidden_size])?;
    let emb_data = embedding.as_f32()?;
    print_stats("Embedding", emb_data);
    
    // Get layer 0
    let layer = &model.layers()[0];
    
    // Step 1: Attention norm
    println!("\n--- Step 1: Attention RMSNorm ---");
    let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.attn_norm.forward(&embedding, &mut normed, &backend)?;
    let normed_data = normed.as_f32()?;
    print_stats("After attn_norm", normed_data);
    
    // Step 2: Q, K, V projections with bias
    println!("\n--- Step 2: Q, K, V Projections ---");
    let x_vec = normed.reshape(vec![hidden_size])?;
    
    let mut q = Tensor::zeros(vec![num_heads * head_dim], DType::F32);
    let mut k = Tensor::zeros(vec![num_kv_heads * head_dim], DType::F32);
    let mut v = Tensor::zeros(vec![num_kv_heads * head_dim], DType::F32);
    
    layer.attention.wq.forward(&x_vec, &mut q, &backend)?;
    layer.attention.wk.forward(&x_vec, &mut k, &backend)?;
    layer.attention.wv.forward(&x_vec, &mut v, &backend)?;
    
    print_stats("Q (with bias)", q.as_f32()?);
    print_stats("K (with bias)", k.as_f32()?);
    print_stats("V", v.as_f32()?);
    
    // Step 3: RoPE (at pos=0, should be identity)
    println!("\n--- Step 3: RoPE ---");
    let mut q_reshaped = q.reshape(vec![num_heads, 1, head_dim])?;
    let mut k_reshaped = k.reshape(vec![num_kv_heads, 1, head_dim])?;
    
    backend.rope(&mut q_reshaped, &mut k_reshaped, pos, freq_base, 1.0, true)?;
    
    println!("At pos=0, RoPE should be identity (cos(0)=1, sin(0)=0)");
    let q_rope_data = q_reshaped.as_f32()?;
    let k_rope_data = k_reshaped.as_f32()?;
    print_stats("Q after RoPE", q_rope_data);
    print_stats("K after RoPE", k_rope_data);
    
    // Step 4: Attention (at pos=0, output = V replicated)
    println!("\n--- Step 4: Attention ---");
    let v_reshaped = v.reshape(vec![num_kv_heads, 1, head_dim])?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let mut attn_out = Tensor::zeros(vec![num_heads, 1, head_dim], DType::F32);
    backend.attention(&q_reshaped, &k_reshaped, &v_reshaped, &mut attn_out, scale)?;
    
    let attn_out_data = attn_out.as_f32()?;
    print_stats("Attention output", attn_out_data);
    
    // Expected: at pos=0, attention output should equal V (replicated)
    let v_data = v.as_f32()?;
    let num_q_per_kv = num_heads / num_kv_heads;
    println!("\n  Expected (V head 0 replicated): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        v_data[0], v_data[1], v_data[2], v_data[3], v_data[4]);
    
    // Step 5: Output projection
    println!("\n--- Step 5: Output Projection ---");
    let attn_out_flat = attn_out.reshape(vec![num_heads * head_dim])?;
    let mut output = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.attention.wo.forward(&attn_out_flat, &mut output, &backend)?;
    
    let output_data = output.as_f32()?;
    print_stats("Output projection", output_data);
    
    // Step 6: First residual
    println!("\n--- Step 6: First Residual ---");
    let mut hidden_after_attn = Tensor::zeros(vec![hidden_size], DType::F32);
    {
        let h_data = hidden_after_attn.as_f32_mut()?;
        for i in 0..hidden_size {
            h_data[i] = emb_data[i] + output_data[i];
        }
    }
    let hidden_after_attn_data = hidden_after_attn.as_f32()?;
    print_stats("After first residual (emb + attn_out)", hidden_after_attn_data);
    
    // Step 7: FFN norm
    println!("\n--- Step 7: FFN RMSNorm ---");
    let mut normed_ffn = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.ffn_norm.forward(&hidden_after_attn, &mut normed_ffn, &backend)?;
    let normed_ffn_data = normed_ffn.as_f32()?;
    print_stats("After ffn_norm", normed_ffn_data);
    
    // Step 8: FFN (gate, up, down)
    println!("\n--- Step 8: FFN ---");
    let intermediate_size = layer.ffn.intermediate_size;
    println!("  intermediate_size = {}", intermediate_size);
    
    let mut ffn_out = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.ffn.forward(&normed_ffn, &mut ffn_out, &backend)?;
    
    let ffn_out_data = ffn_out.as_f32()?;
    print_stats("FFN output", ffn_out_data);
    
    // Step 9: Second residual (final layer output)
    println!("\n--- Step 9: Second Residual (Layer 0 Output) ---");
    let mut layer_output = Tensor::zeros(vec![hidden_size], DType::F32);
    {
        let l_data = layer_output.as_f32_mut()?;
        for i in 0..hidden_size {
            l_data[i] = hidden_after_attn_data[i] + ffn_out_data[i];
        }
    }
    let layer_output_data = layer_output.as_f32()?;
    print_stats("FINAL Layer 0 output", layer_output_data);
    
    // Compare with Python reference
    println!("\n=== COMPARISON WITH PYTHON REFERENCE ===");
    println!("Python layer_0: min=-4.4456, max=4.0055, mean=-0.0860");
    println!("Python first 5: [-1.5666, 0.0268, -1.2803, -0.5673, 1.2690]");
    
    Ok(())
}
