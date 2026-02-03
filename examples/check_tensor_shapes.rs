//! Check tensor shapes and validate our understanding.

use llama_rs::gguf::GgufFile;
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    
    println!("=== Tensor Shape Analysis ===");
    println!();
    
    // Key tensors to check
    let tensors_to_check = [
        ("token_embd.weight", "Embedding matrix"),
        ("blk.0.attn_q.weight", "Q projection"),
        ("blk.0.attn_k.weight", "K projection"),
        ("blk.0.attn_v.weight", "V projection"),
        ("blk.0.attn_output.weight", "Output projection"),
        ("blk.0.ffn_gate.weight", "FFN gate"),
        ("blk.0.ffn_up.weight", "FFN up"),
        ("blk.0.ffn_down.weight", "FFN down"),
        ("output.weight", "LM head"),
    ];
    
    // Model dimensions
    let hidden_size = 896;
    let num_heads = 14;
    let head_dim = 64;
    let num_kv_heads = 2;
    let intermediate_size = 4864;
    let vocab_size = 151936;
    
    println!("Model dimensions:");
    println!("  hidden_size = {}", hidden_size);
    println!("  num_heads = {}", num_heads);
    println!("  head_dim = {}", head_dim);
    println!("  num_kv_heads = {}", num_kv_heads);
    println!("  intermediate_size = {}", intermediate_size);
    println!("  vocab_size = {}", vocab_size);
    println!();
    
    println!("In GGUF, ne[0] is the fastest-changing (contiguous) dimension.");
    println!("For y = x @ W, where x is [k], W is [k, n], y is [n]:");
    println!("  GGUF stores W with shape [n_rows, n_cols] = [k, n]");
    println!("  But ne[0] = k (input), ne[1] = n (output)");
    println!();
    
    for (name, desc) in tensors_to_check.iter() {
        if let Some(info) = gguf.data.get_tensor(name) {
            let dims: Vec<u64> = info.dims.to_vec();
            println!("{} ({}):", desc, name);
            println!("  GGUF shape (ne): {:?}", dims);
            println!("  dtype: {:?}", info.dtype);
            
            // Interpret the shape
            match dims.len() {
                2 => {
                    let ne0 = dims[0];
                    let ne1 = dims[1];
                    println!("  Interpretation: [{} rows x {} cols]", ne0, ne1);
                    println!("  For y = x @ W: input dim = {}, output dim = {}", ne0, ne1);
                }
                1 => {
                    println!("  1D tensor, size = {}", dims[0]);
                }
                _ => {
                    println!("  {}D tensor", dims.len());
                }
            }
            println!();
        } else {
            println!("{} ({}) - NOT FOUND", desc, name);
            println!();
        }
    }
    
    println!("=== Expected Shapes ===");
    println!();
    println!("Embedding: [{}, {}] -> lookup token t gives row t (size {})", vocab_size, hidden_size, hidden_size);
    println!("Q projection: [{}, {}] -> x @ W_q gives [{}]", hidden_size, num_heads * head_dim, num_heads * head_dim);
    println!("K projection: [{}, {}] -> x @ W_k gives [{}]", hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim);
    println!("V projection: [{}, {}] -> x @ W_v gives [{}]", hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim);
    println!("Output proj: [{}, {}] -> attn @ W_o gives [{}]", num_heads * head_dim, hidden_size, hidden_size);
    println!("FFN gate: [{}, {}] -> x @ W_gate gives [{}]", hidden_size, intermediate_size, intermediate_size);
    println!("FFN up: [{}, {}] -> x @ W_up gives [{}]", hidden_size, intermediate_size, intermediate_size);
    println!("FFN down: [{}, {}] -> inter @ W_down gives [{}]", intermediate_size, hidden_size, hidden_size);
    println!("LM head: [{}, {}] -> hidden @ W_lm gives [{}]", hidden_size, vocab_size, vocab_size);
}
