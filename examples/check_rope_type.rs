//! Check the RoPE type from GGUF metadata.

use llama_rs::gguf::GgufFile;
use std::path::Path;

fn main() {
    let model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading GGUF file...");
    let gguf = GgufFile::open(Path::new(model_path)).expect("Failed to open GGUF");
    
    println!("\n=== RoPE Related Metadata ===");
    
    // Check all metadata for rope-related entries
    for (key, value) in gguf.data.metadata.iter() {
        let key_lower = key.to_lowercase();
        if key_lower.contains("rope") || key_lower.contains("freq") || key_lower.contains("rot") {
            println!("  {}: {:?}", key, value);
        }
    }
    
    println!("\n=== All Metadata (for reference) ===");
    for (key, value) in gguf.data.metadata.iter() {
        println!("  {}: {:?}", key, value);
    }
}
