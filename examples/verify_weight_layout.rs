//! Verify weight matrix layout and dimensions.

use llama_rs::gguf::GgufFile;
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading GGUF file...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    
    // Expected model dimensions
    let hidden_size = 896;
    let num_heads = 14;
    let num_kv_heads = 2;
    let head_dim = 64;
    let intermediate_size = 4864;
    let vocab_size = 151936;
    
    println!("\n=== Expected Dimensions ===");
    println!("  hidden_size: {}", hidden_size);
    println!("  num_heads: {}", num_heads);
    println!("  num_kv_heads: {}", num_kv_heads);
    println!("  head_dim: {}", head_dim);
    println!("  Q dim: {} x {} (hidden -> num_heads*head_dim)", hidden_size, num_heads * head_dim);
    println!("  K dim: {} x {} (hidden -> num_kv_heads*head_dim)", hidden_size, num_kv_heads * head_dim);
    println!("  V dim: {} x {} (hidden -> num_kv_heads*head_dim)", hidden_size, num_kv_heads * head_dim);
    println!("  O dim: {} x {} (num_heads*head_dim -> hidden)", num_heads * head_dim, hidden_size);
    println!();
    
    // Check actual tensor dimensions from GGUF
    println!("=== Actual GGUF Tensor Dimensions ===");
    
    let tensors_to_check = vec![
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q.bias",
        "blk.0.attn_k.weight",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.weight",
        "blk.0.attn_v.bias",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ];
    
    for name in tensors_to_check {
        if let Some(info) = gguf.data.get_tensor(name) {
            let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
            let numel: usize = dims.iter().product();
            println!("  {}: {:?} (numel: {})", name, dims, numel);
        } else {
            println!("  {}: NOT FOUND", name);
        }
    }
    
    println!();
    println!("=== GGUF Dimension Interpretation ===");
    println!("  In GGUF, ne[0] is the contiguous/fast dimension (columns in row-major)");
    println!("  For matrix W in y = x @ W:");
    println!("    - x has shape [k]");
    println!("    - W has shape [k, n] in our notation (ne[0]=k, ne[1]=n in GGUF)");
    println!("    - y has shape [n]");
    println!();
    
    if let Some(info) = gguf.data.get_tensor("blk.0.attn_q.weight") {
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        println!("  Q weight: ne[0]={}, ne[1]={}", dims[0], dims[1]);
        println!("    -> For y = x @ W_q:");
        println!("       x has shape [{}] (hidden)", dims[0]);
        println!("       y has shape [{}] (num_heads * head_dim = {} * {} = {})", 
                dims[1], num_heads, head_dim, num_heads * head_dim);
        if dims[0] == hidden_size && dims[1] == num_heads * head_dim {
            println!("    MATCHES expected: hidden -> num_heads*head_dim");
        } else {
            println!("    WARNING: Does not match expected dimensions!");
        }
    }
    
    println!();
    if let Some(info) = gguf.data.get_tensor("blk.0.attn_output.weight") {
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        println!("  Output weight: ne[0]={}, ne[1]={}", dims[0], dims[1]);
        println!("    -> For y = x @ W_o:");
        println!("       x has shape [{}] (num_heads * head_dim)", dims[0]);
        println!("       y has shape [{}] (hidden)", dims[1]);
        if dims[0] == num_heads * head_dim && dims[1] == hidden_size {
            println!("    MATCHES expected: num_heads*head_dim -> hidden");
        } else {
            println!("    WARNING: Does not match expected dimensions!");
        }
    }
    
    println!();
    if let Some(info) = gguf.data.get_tensor("output.weight") {
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        println!("  LM head weight: ne[0]={}, ne[1]={}", dims[0], dims[1]);
        println!("    -> For logits = hidden @ W_lm:");
        println!("       hidden has shape [{}]", dims[0]);
        println!("       logits has shape [{}]", dims[1]);
        if dims[0] == hidden_size && dims[1] == vocab_size {
            println!("    MATCHES expected: hidden -> vocab_size");
        } else {
            println!("    WARNING: Does not match expected dimensions!");
        }
    }
    
    // Check if bias exists
    println!();
    println!("=== Bias Tensor Analysis ===");
    if let Some(info) = gguf.data.get_tensor("blk.0.attn_q.bias") {
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        println!("  Q bias dims: {:?}", dims);
        println!("    Expected: [{}] (num_heads * head_dim)", num_heads * head_dim);
    }
}
