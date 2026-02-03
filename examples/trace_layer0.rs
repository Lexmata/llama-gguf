//! Trace layer 0 forward pass step by step
//! 
//! This example prints intermediate values at each step of layer 0
//! to help debug divergence from llama.cpp

use llama_rs::{
    backend::{cpu::CpuBackend, Backend},
    model::load_llama_model,
    tensor::{DType, Tensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading model from {}...", model_path);
    let backend = CpuBackend::new();
    let model = load_llama_model(model_path)?;
    
    // Token "=" is token 28
    let token_id = 28u32;
    let pos = 0usize;
    
    println!("\n=== Testing token {} ('=') at position {} ===", token_id, pos);
    
    // Get model dimensions
    let hidden_size = model.config().hidden_size;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim;
    let freq_base = model.config().rope_config.freq_base;
    
    println!("Model config:");
    println!("  hidden_size: {}", hidden_size);
    println!("  num_heads: {}", num_heads);
    println!("  num_kv_heads: {}", num_kv_heads);
    println!("  head_dim: {}", head_dim);
    println!("  freq_base: {}", freq_base);
    
    // Step 1: Get embedding
    println!("\n--- Step 1: Token Embedding ---");
    let embedding = model.embed_tokens(&[token_id], &backend)?;
    let emb_data = embedding.as_f32()?;
    
    println!("Embedding shape: {:?}", embedding.shape());
    let emb_min = emb_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let emb_max = emb_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let emb_sum: f32 = emb_data.iter().sum();
    println!("Embedding stats: min={:.6}, max={:.6}, sum={:.6}", emb_min, emb_max, emb_sum);
    println!("Embedding first 10: {:?}", &emb_data[..10.min(emb_data.len())]);
    
    // Step 2: Apply attention norm (layer 0)
    println!("\n--- Step 2: Attention RMSNorm ---");
    let layer = &model.layers()[0];
    
    // Make sure embedding is contiguous by reshaping to same shape
    let embedding = embedding.reshape(vec![hidden_size])?;
    let emb_data = embedding.as_f32()?; // Re-get after reshape
    
    let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.attn_norm.forward(&embedding, &mut normed, &backend)?;
    let normed_data = normed.as_f32()?;
    
    let norm_min = normed_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm_max = normed_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("After norm stats: min={:.6}, max={:.6}", norm_min, norm_max);
    println!("After norm first 10: {:?}", &normed_data[..10.min(normed_data.len())]);
    
    // Step 3: Compute Q, K, V projections
    println!("\n--- Step 3: Q, K, V Projections ---");
    
    // Flatten normed for linear layer
    let x_vec = normed.reshape(vec![hidden_size])?;
    
    // Q projection
    let mut q = Tensor::zeros(vec![num_heads * head_dim], DType::F32);
    layer.attention.wq.forward(&x_vec, &mut q, &backend)?;
    let q_data = q.as_f32()?;
    
    let q_min = q_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let q_max = q_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Q (after bias) stats: min={:.2}, max={:.2}", q_min, q_max);
    println!("Q first 10: {:?}", &q_data[..10.min(q_data.len())]);
    
    // K projection
    let mut k = Tensor::zeros(vec![num_kv_heads * head_dim], DType::F32);
    layer.attention.wk.forward(&x_vec, &mut k, &backend)?;
    let k_data = k.as_f32()?;
    
    let k_min = k_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let k_max = k_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("K (after bias) stats: min={:.2}, max={:.2}", k_min, k_max);
    println!("K first 10: {:?}", &k_data[..10.min(k_data.len())]);
    
    // V projection
    let mut v = Tensor::zeros(vec![num_kv_heads * head_dim], DType::F32);
    layer.attention.wv.forward(&x_vec, &mut v, &backend)?;
    let v_data = v.as_f32()?;
    
    let v_min = v_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = v_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("V stats: min={:.4}, max={:.4}", v_min, v_max);
    println!("V first 10: {:?}", &v_data[..10.min(v_data.len())]);
    
    // Step 4: Apply RoPE
    println!("\n--- Step 4: Apply RoPE ---");
    let mut q_reshaped = q.reshape(vec![num_heads, 1, head_dim])?;
    let mut k_reshaped = k.reshape(vec![num_kv_heads, 1, head_dim])?;
    
    backend.rope(&mut q_reshaped, &mut k_reshaped, pos, freq_base, 1.0, true)?;
    
    let q_rope_data = q_reshaped.as_f32()?;
    let k_rope_data = k_reshaped.as_f32()?;
    
    println!("At pos=0, RoPE should be identity (cos(0)=1, sin(0)=0)");
    println!("Q after RoPE first 10: {:?}", &q_rope_data[..10.min(q_rope_data.len())]);
    println!("K after RoPE first 10: {:?}", &k_rope_data[..10.min(k_rope_data.len())]);
    
    // Check if RoPE changed anything at pos=0
    let q_diff: f32 = q_data.iter().zip(q_rope_data.iter()).map(|(a, b)| (a - b).abs()).sum();
    let k_diff: f32 = k_data.iter().zip(k_rope_data.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("Q diff from pre-RoPE: {:.6}", q_diff);
    println!("K diff from pre-RoPE: {:.6}", k_diff);
    
    // Step 5: Self-attention at position 0
    println!("\n--- Step 5: Self-Attention (pos=0) ---");
    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("Attention scale: {}", scale);
    
    // At position 0, each Q head attends only to its corresponding KV head
    // With softmax over a single element, the weight is always 1.0
    // So attention output = V
    
    // Compute attention scores for head 0
    let q_head0: f32 = q_rope_data[..head_dim].iter().zip(k_rope_data[..head_dim].iter())
        .map(|(qi, ki)| qi * ki).sum();
    let score_head0 = q_head0 * scale;
    println!("Head 0 raw attention score: {:.4}", score_head0);
    
    // V for head 0 (which is KV head 0)
    let v_head0 = &v_data[..head_dim];
    println!("V head 0 first 10: {:?}", &v_head0[..10.min(v_head0.len())]);
    
    // Attention output for head 0 = softmax(score) * V = 1.0 * V = V
    println!("Attention output head 0 = V head 0 (softmax of single element = 1)");
    
    // Step 6: Full attention computation
    println!("\n--- Step 6: Full Attention Output ---");
    
    // Create K and V tensors for attention
    let v_reshaped = v.reshape(vec![num_kv_heads, 1, head_dim])?;
    
    // Run full attention
    let mut attn_out = Tensor::zeros(vec![num_heads, 1, head_dim], DType::F32);
    backend.attention(
        &q_reshaped,
        &k_reshaped,
        &v_reshaped,
        &mut attn_out,
        scale,
    )?;
    
    let attn_out_data = attn_out.as_f32()?;
    let attn_min = attn_out_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let attn_max = attn_out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Attention output stats: min={:.4}, max={:.4}", attn_min, attn_max);
    println!("Attention output first 10: {:?}", &attn_out_data[..10.min(attn_out_data.len())]);
    
    // At position 0, attention output should equal V (replicated for all Q heads)
    // Heads 0-6 use KV head 0, heads 7-13 use KV head 1
    let expected_attn_out: Vec<f32> = (0..num_heads).flat_map(|h| {
        let kv_h = h / (num_heads / num_kv_heads);
        v_data[kv_h * head_dim..(kv_h + 1) * head_dim].to_vec()
    }).collect();
    
    let attn_diff: f32 = attn_out_data.iter().zip(expected_attn_out.iter())
        .map(|(a, b)| (a - b).abs()).sum();
    println!("Attention output diff from expected (V): {:.6}", attn_diff);
    
    // Step 7: Output projection
    println!("\n--- Step 7: Output Projection ---");
    let attn_out_flat = attn_out.reshape(vec![num_heads * head_dim])?;
    let mut output = Tensor::zeros(vec![hidden_size], DType::F32);
    layer.attention.wo.forward(&attn_out_flat, &mut output, &backend)?;
    
    let out_data = output.as_f32()?;
    let out_min = out_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let out_max = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Output projection stats: min={:.4}, max={:.4}", out_min, out_max);
    println!("Output projection first 10: {:?}", &out_data[..10.min(out_data.len())]);
    
    // Step 8: Residual connection
    println!("\n--- Step 8: Residual Connection ---");
    let mut hidden_after_attn = Tensor::zeros(vec![hidden_size], DType::F32);
    {
        let hidden_data = hidden_after_attn.as_f32_mut()?;
        for i in 0..hidden_size {
            hidden_data[i] = emb_data[i] + out_data[i];
        }
    }
    
    let hidden_data = hidden_after_attn.as_f32()?;
    let hidden_min = hidden_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let hidden_max = hidden_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let hidden_sum: f32 = hidden_data.iter().sum();
    println!("After attention+residual stats: min={:.4}, max={:.4}, sum={:.4}", hidden_min, hidden_max, hidden_sum);
    println!("After attention+residual first 10: {:?}", &hidden_data[..10.min(hidden_data.len())]);
    
    println!("\n=== SUMMARY ===");
    println!("Embedding sum: {:.4}", emb_sum);
    println!("After attention sum: {:.4}", hidden_sum);
    println!("Output projection sum: {:.4}", out_data.iter().sum::<f32>());
    
    Ok(())
}
