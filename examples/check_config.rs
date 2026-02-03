
use llama_rs::model::load_llama_model;
use llama_rs::Model;
use std::path::Path;

fn main() {
    let model = load_llama_model(Path::new("/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        .expect("Failed to load model");
    
    let config = model.config();
    println!("norm_eps = {:e}", config.norm_eps);
    println!("hidden_size = {}", config.hidden_size);
    println!("num_layers = {}", config.num_layers);
    println!("num_heads = {}", config.num_heads);
    println!("num_kv_heads = {}", config.num_kv_heads);
    println!("rope freq_base = {}", config.rope_config.freq_base);
}
