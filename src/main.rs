use std::io::{self, BufRead, Write};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use llama_rs::gguf::{GgufFile, MetadataValue};
use llama_rs::huggingface::{format_bytes, HfClient};
use llama_rs::model::{InferenceContext, ModelLoader};
use llama_rs::sampling::{Sampler, SamplerConfig};
use llama_rs::tokenizer::Tokenizer;
use llama_rs::Model;

#[derive(Parser)]
#[command(name = "llama-rs")]
#[command(about = "Rust implementation of llama.cpp", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show information about a GGUF model file
    Info {
        /// Path to the GGUF model file
        model: String,

        /// Show all metadata (verbose)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run inference on a model
    Run {
        /// Path to the GGUF model file
        model: String,

        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Number of tokens to generate
        #[arg(short, long, default_value = "128")]
        n_predict: usize,

        /// Temperature for sampling (0.0 = greedy)
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,

        /// Top-K sampling (0 = disabled)
        #[arg(long, default_value = "40")]
        top_k: usize,

        /// Top-P (nucleus) sampling
        #[arg(long, default_value = "0.95")]
        top_p: f32,

        /// Repetition penalty
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Interactive chat mode
    Chat {
        /// Path to the GGUF model file
        model: String,

        /// System prompt (instructions for the model)
        #[arg(long)]
        system: Option<String>,

        /// Maximum tokens to generate per response
        #[arg(short, long, default_value = "512")]
        n_predict: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Top-K sampling
        #[arg(long, default_value = "40")]
        top_k: usize,

        /// Top-P (nucleus) sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Repetition penalty
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Start HTTP server with OpenAI-compatible API
    #[cfg(feature = "server")]
    Serve {
        /// Path to the GGUF model file
        model: String,

        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Quantize a model to a different format
    Quantize {
        /// Path to the input GGUF model file
        input: String,

        /// Path to the output GGUF model file
        output: String,

        /// Target quantization type (q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k)
        #[arg(short = 't', long, default_value = "q4_0")]
        qtype: String,

        /// Number of threads to use
        #[arg(long)]
        threads: Option<usize>,
    },

    /// Show system information and capabilities
    SysInfo,

    /// Benchmark model performance
    Bench {
        /// Path to the GGUF model file
        model: String,

        /// Number of prompt tokens to process
        #[arg(short = 'p', long, default_value = "512")]
        n_prompt: usize,

        /// Number of tokens to generate
        #[arg(short = 'n', long, default_value = "128")]
        n_gen: usize,

        /// Number of repetitions for averaging
        #[arg(short, long, default_value = "3")]
        repetitions: usize,

        /// Number of threads to use
        #[arg(long)]
        threads: Option<usize>,
    },

    /// Extract embeddings from text
    Embed {
        /// Path to the GGUF model file
        model: String,

        /// Text to embed
        #[arg(short, long)]
        text: String,

        /// Output format (json, raw)
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Download a model from HuggingFace Hub
    Download {
        /// HuggingFace repository (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        repo: String,

        /// Specific file to download (optional, lists available files if not specified)
        #[arg(short, long)]
        file: Option<String>,

        /// Download directory (uses system cache by default)
        #[arg(short, long)]
        output: Option<String>,

        /// Force re-download even if file exists
        #[arg(long)]
        force: bool,
    },

    /// Manage local model cache
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// List all cached models
    List,

    /// Search for models on HuggingFace Hub
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show cache information
    CacheInfo,

    /// Clear the model cache
    ClearCache {
        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },

    /// List GGUF files in a HuggingFace repository
    ListFiles {
        /// HuggingFace repository (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        repo: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { model, verbose } => {
            if let Err(e) = show_info(&model, verbose) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Run {
            model,
            prompt,
            n_predict,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
        } => {
            if let Err(e) = run_inference(
                &model,
                prompt.as_deref(),
                n_predict,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                seed,
            ) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Chat {
            model,
            system,
            n_predict,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
        } => {
            if let Err(e) = run_chat(
                &model,
                system.as_deref(),
                n_predict,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                seed,
            ) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        #[cfg(feature = "server")]
        Commands::Serve { model, host, port } => {
            if let Err(e) = run_server(&model, &host, port) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Quantize {
            input,
            output,
            qtype,
            threads,
        } => {
            if let Err(e) = run_quantize(&input, &output, &qtype, threads) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::SysInfo => {
            show_sysinfo();
        }
        Commands::Bench {
            model,
            n_prompt,
            n_gen,
            repetitions,
            threads,
        } => {
            if let Err(e) = run_benchmark(&model, n_prompt, n_gen, repetitions, threads) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Embed {
            model,
            text,
            format,
        } => {
            if let Err(e) = run_embed(&model, &text, &format) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Download {
            repo,
            file,
            output,
            force,
        } => {
            if let Err(e) = run_download(&repo, file.as_deref(), output.as_deref(), force) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Models { action } => {
            if let Err(e) = run_models_command(action) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

#[cfg(feature = "server")]
fn run_server(model_path: &str, host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        llama_rs::server::run_server(llama_rs::server::ServerConfig {
            host: host.to_string(),
            port,
            model_path: model_path.to_string(),
        })
        .await
    })
}

fn run_inference(
    model_path: &str,
    prompt: Option<&str>,
    n_predict: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model from: {}", model_path);

    // Load GGUF file
    let gguf = GgufFile::open(model_path)?;

    // Load tokenizer
    eprintln!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    eprintln!("Vocabulary size: {}", tokenizer.vocab_size);

    // Load model
    eprintln!("Loading model weights...");
    let loader = ModelLoader::load(model_path)?;
    let config = loader.config().clone();
    eprintln!(
        "Model: {} layers, {} heads, {} hidden dim",
        config.num_layers, config.num_heads, config.hidden_size
    );

    let model = loader.build_model()?;

    // Create backend and context
    let backend: Arc<dyn llama_rs::Backend> = Arc::new(llama_rs::backend::cpu::CpuBackend::new());
    let mut ctx = InferenceContext::new(&config, backend);

    // Configure sampler
    let sampler_config = SamplerConfig {
        temperature,
        top_k,
        top_p,
        repeat_penalty,
        seed,
        ..Default::default()
    };
    let mut sampler = Sampler::new(sampler_config, config.vocab_size);

    // Encode prompt
    let prompt_text = prompt.unwrap_or("Hello");
    eprintln!("Prompt: {}", prompt_text);

    let mut tokens = tokenizer.encode(prompt_text, true)?;
    eprintln!("Prompt tokens: {} tokens", tokens.len());
    eprintln!();

    // Print prompt
    print!("{}", prompt_text);
    io::stdout().flush()?;

    // Generation loop
    for _ in 0..n_predict {
        // Check if we hit EOS
        if let Some(&last) = tokens.last() {
            if last == tokenizer.special_tokens.eos_token_id {
                break;
            }
        }

        // Run forward pass (just the last token for incremental generation)
        let input_tokens = if ctx.position == 0 {
            &tokens[..]
        } else {
            &tokens[tokens.len() - 1..]
        };

        let logits = model.forward(input_tokens, &mut ctx)?;

        // Sample next token
        let next_token = sampler.sample(&logits, &tokens);

        // Decode and print
        if let Ok(text) = tokenizer.decode(&[next_token]) {
            print!("{}", text);
            io::stdout().flush()?;
        }

        tokens.push(next_token);

        // Check for EOS
        if next_token == tokenizer.special_tokens.eos_token_id {
            break;
        }
    }

    println!();
    eprintln!();
    eprintln!("Generated {} tokens", tokens.len());

    Ok(())
}

/// Interactive chat mode
fn run_chat(
    model_path: &str,
    system_prompt: Option<&str>,
    n_predict: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model from: {}", model_path);

    // Load GGUF file
    let gguf = GgufFile::open(model_path)?;

    // Load tokenizer
    eprintln!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    eprintln!("Vocabulary size: {}", tokenizer.vocab_size);

    // Load model
    eprintln!("Loading model weights...");
    let loader = ModelLoader::load(model_path)?;
    let config = loader.config().clone();
    eprintln!(
        "Model: {} layers, {} heads, {} hidden dim",
        config.num_layers, config.num_heads, config.hidden_size
    );
    eprintln!("Max context: {} tokens", config.max_seq_len);

    let model = loader.build_model()?;

    // Create backend and context
    let backend: Arc<dyn llama_rs::Backend> = Arc::new(llama_rs::backend::cpu::CpuBackend::new());
    let mut ctx = InferenceContext::new(&config, backend);

    // Configure sampler
    let sampler_config = SamplerConfig {
        temperature,
        top_k,
        top_p,
        repeat_penalty,
        seed,
        ..Default::default()
    };
    let mut sampler = Sampler::new(sampler_config, config.vocab_size);

    // Conversation history as tokens
    let mut conversation_tokens: Vec<u32> = Vec::new();

    // Format system prompt if provided
    let system_text = system_prompt.unwrap_or("You are a helpful AI assistant.");

    // Print chat header
    eprintln!();
    eprintln!("╭─────────────────────────────────────────────────────────────────╮");
    eprintln!("│                     Interactive Chat Mode                        │");
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Commands:                                                        │");
    eprintln!("│   /clear  - Clear conversation history                           │");
    eprintln!("│   /system - Show/set system prompt                               │");
    eprintln!("│   /help   - Show this help                                       │");
    eprintln!("│   /quit   - Exit chat                                            │");
    eprintln!("╰─────────────────────────────────────────────────────────────────╯");
    eprintln!();
    eprintln!("System: {}", system_text);
    eprintln!();

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Print prompt
        print!("You: ");
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            // EOF
            break;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input.to_lowercase().as_str() {
                "/quit" | "/exit" | "/q" => {
                    eprintln!("Goodbye!");
                    break;
                }
                "/clear" => {
                    conversation_tokens.clear();
                    ctx.reset();
                    sampler.reset();
                    eprintln!("Conversation cleared.");
                    continue;
                }
                "/help" => {
                    eprintln!("Commands:");
                    eprintln!("  /clear  - Clear conversation history");
                    eprintln!("  /system - Show system prompt");
                    eprintln!("  /quit   - Exit chat");
                    continue;
                }
                "/system" => {
                    eprintln!("System prompt: {}", system_text);
                    continue;
                }
                _ => {
                    eprintln!("Unknown command: {}. Type /help for available commands.", input);
                    continue;
                }
            }
        }

        // Format the conversation for the model
        // Using a simple chat format: [INST] user message [/INST] assistant response
        let formatted_input = if conversation_tokens.is_empty() {
            // First message - include system prompt
            format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]", system_text, input)
        } else {
            // Continuation
            format!(" [INST] {} [/INST]", input)
        };

        // Encode user input
        let new_tokens = tokenizer.encode(&formatted_input, conversation_tokens.is_empty())?;

        // Check context length
        let total_len = conversation_tokens.len() + new_tokens.len() + n_predict;
        if total_len > config.max_seq_len {
            // Need to truncate - shift the KV cache
            let excess = total_len - config.max_seq_len + 100; // Leave some buffer
            if excess >= conversation_tokens.len() {
                // Too much - just reset
                eprintln!("(Context full, resetting conversation)");
                conversation_tokens.clear();
                ctx.reset();
            } else {
                eprintln!("(Trimming {} tokens from context)", excess);
                conversation_tokens = conversation_tokens[excess..].to_vec();
                ctx.kv_cache.shift_left(excess);
                ctx.position = ctx.position.saturating_sub(excess);
            }
        }

        // Add new tokens to conversation
        conversation_tokens.extend(&new_tokens);

        // Process the new tokens
        let start_pos = ctx.position;
        for (i, &token) in new_tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos < config.max_seq_len {
                let _ = model.forward(&[token], &mut ctx);
            }
        }

        // Generate response
        print!("\nAssistant: ");
        io::stdout().flush()?;

        let mut response_tokens = Vec::new();
        let mut generated_text = String::new();

        for _ in 0..n_predict {
            // Check for end of response patterns
            if generated_text.contains("[INST]") || generated_text.contains("</s>") {
                break;
            }

            // Get last token for generation
            let last_token = *conversation_tokens.last().unwrap_or(&tokenizer.special_tokens.bos_token_id);

            // Forward pass
            let logits = model.forward(&[last_token], &mut ctx)?;

            // Sample next token
            let next_token = sampler.sample(&logits, &conversation_tokens);

            // Check for EOS
            if next_token == tokenizer.special_tokens.eos_token_id {
                break;
            }

            // Decode and print
            if let Ok(text) = tokenizer.decode(&[next_token]) {
                print!("{}", text);
                io::stdout().flush()?;
                generated_text.push_str(&text);
            }

            conversation_tokens.push(next_token);
            response_tokens.push(next_token);
        }

        println!();
        println!();
    }

    Ok(())
}

fn show_info(path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let file = GgufFile::open(path)?;
    let data = &file.data;

    println!("╭─────────────────────────────────────────────────────────────────╮");
    println!("│                        GGUF Model Info                          │");
    println!("╰─────────────────────────────────────────────────────────────────╯");
    println!();
    println!("File: {}", path);
    println!("GGUF Version: {}", data.header.version);
    println!("Tensor count: {}", data.header.tensor_count);
    println!("Metadata entries: {}", data.header.metadata_kv_count);
    println!();

    // General information
    println!("┌─ General ─────────────────────────────────────────────────────┐");
    if let Some(arch) = data.get_string("general.architecture") {
        println!("│ Architecture: {:<50} │", arch);
    }
    if let Some(name) = data.get_string("general.name") {
        println!("│ Name: {:<57} │", truncate(name, 57));
    }
    if let Some(author) = data.get_string("general.author") {
        println!("│ Author: {:<55} │", truncate(author, 55));
    }
    if let Some(quant) = data.get_string("general.quantization_version") {
        println!("│ Quantization: {:<49} │", quant);
    }
    if let Some(file_type) = data.get_u32("general.file_type") {
        println!("│ File type: {:<52} │", file_type);
    }
    println!("└───────────────────────────────────────────────────────────────┘");
    println!();

    // Model parameters
    let arch = data.get_string("general.architecture").unwrap_or("llama");

    println!("┌─ Model Parameters ─────────────────────────────────────────────┐");

    if let Some(v) = data.get_u32(&format!("{}.context_length", arch)) {
        println!("│ Context length: {:<47} │", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.embedding_length", arch)) {
        println!("│ Embedding size: {:<47} │", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.feed_forward_length", arch)) {
        println!("│ Feed-forward size: {:<44} │", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.block_count", arch)) {
        println!("│ Layers: {:<55} │", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.attention.head_count", arch)) {
        println!("│ Attention heads: {:<46} │", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.attention.head_count_kv", arch)) {
        println!("│ KV heads: {:<53} │", v);
    }
    if let Some(v) = data.get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch)) {
        println!("│ RMS norm epsilon: {:<45} │", format!("{:.2e}", v));
    }
    if let Some(v) = data.get_f32(&format!("{}.rope.freq_base", arch)) {
        println!("│ RoPE freq base: {:<47} │", v);
    }

    println!("└───────────────────────────────────────────────────────────────┘");
    println!();

    // Tokenizer info
    println!("┌─ Tokenizer ────────────────────────────────────────────────────┐");
    if let Some(model) = data.get_string("tokenizer.ggml.model") {
        println!("│ Model: {:<56} │", model);
    }
    if let Some(bos) = data.get_u32("tokenizer.ggml.bos_token_id") {
        println!("│ BOS token ID: {:<49} │", bos);
    }
    if let Some(eos) = data.get_u32("tokenizer.ggml.eos_token_id") {
        println!("│ EOS token ID: {:<49} │", eos);
    }
    if let Some(pad) = data.get_u32("tokenizer.ggml.padding_token_id") {
        println!("│ PAD token ID: {:<49} │", pad);
    }
    println!("└───────────────────────────────────────────────────────────────┘");
    println!();

    // Show tensors
    println!("┌─ Tensors ──────────────────────────────────────────────────────┐");
    let max_tensors = if verbose { data.tensors.len() } else { 10 };
    for tensor in data.tensors.iter().take(max_tensors) {
        let dims_str = format!("{:?}", tensor.dims);
        let dtype_str = format!("{:?}", tensor.dtype);
        let name_truncated = truncate(&tensor.name, 30);
        println!(
            "│ {:<30} {:>12} {:>8} │",
            name_truncated, dims_str, dtype_str
        );
    }
    if data.tensors.len() > max_tensors {
        println!("│ ... and {} more tensors{:>29} │", data.tensors.len() - max_tensors, "");
    }
    println!("└───────────────────────────────────────────────────────────────┘");

    // Verbose: show all metadata
    if verbose {
        println!();
        println!("┌─ All Metadata ────────────────────────────────────────────────┐");
        let mut keys: Vec<_> = data.metadata.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(value) = data.metadata.get(key) {
                let value_str = format_metadata_value(value);
                println!("│ {}: {}", key, truncate(&value_str, 50));
            }
        }
        println!("└───────────────────────────────────────────────────────────────┘");
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

fn format_metadata_value(value: &MetadataValue) -> String {
    match value {
        MetadataValue::Uint8(v) => format!("{}", v),
        MetadataValue::Int8(v) => format!("{}", v),
        MetadataValue::Uint16(v) => format!("{}", v),
        MetadataValue::Int16(v) => format!("{}", v),
        MetadataValue::Uint32(v) => format!("{}", v),
        MetadataValue::Int32(v) => format!("{}", v),
        MetadataValue::Float32(v) => format!("{}", v),
        MetadataValue::Bool(v) => format!("{}", v),
        MetadataValue::String(v) => format!("\"{}\"", truncate(v, 40)),
        MetadataValue::Array(arr) => format!("[array of {} items]", arr.values.len()),
        MetadataValue::Uint64(v) => format!("{}", v),
        MetadataValue::Int64(v) => format!("{}", v),
        MetadataValue::Float64(v) => format!("{}", v),
    }
}

/// Show system information and capabilities
fn show_sysinfo() {
    use llama_rs::backend::cpu::CpuBackend;

    let backend = CpuBackend::new();

    println!("╭─────────────────────────────────────────────────────────────────╮");
    println!("│                     System Information                          │");
    println!("╰─────────────────────────────────────────────────────────────────╯");
    println!();

    // CPU info
    println!("┌─ CPU ────────────────────────────────────────────────────────────┐");
    println!("│ Threads: {:<54} │", backend.num_threads());
    println!("│ SIMD: {:<57} │", backend.simd_info());
    println!("│ AVX2: {:<57} │", if backend.has_avx2() { "yes" } else { "no" });
    println!("│ AVX-512: {:<54} │", if backend.has_avx512() { "yes" } else { "no" });
    println!("│ NEON: {:<57} │", if backend.has_neon() { "yes" } else { "no" });
    println!("└───────────────────────────────────────────────────────────────────┘");
    println!();

    // Supported formats
    println!("┌─ Supported Quantization Formats ────────────────────────────────┐");
    println!("│ Basic: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1                        │");
    println!("│ K-quants: Q2K, Q3K, Q4K, Q5K, Q6K, Q8K                           │");
    println!("└───────────────────────────────────────────────────────────────────┘");
    println!();

    // Features
    println!("┌─ Features ────────────────────────────────────────────────────────┐");
    #[cfg(feature = "server")]
    println!("│ HTTP Server: enabled                                              │");
    #[cfg(not(feature = "server"))]
    println!("│ HTTP Server: disabled (compile with --features server)            │");
    println!("└───────────────────────────────────────────────────────────────────┘");
}

/// Run model quantization
fn run_quantize(
    input_path: &str,
    output_path: &str,
    qtype: &str,
    threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    use llama_rs::tensor::DType;

    // Parse target quantization type
    let target_dtype = match qtype.to_lowercase().as_str() {
        "q4_0" => DType::Q4_0,
        "q4_1" => DType::Q4_1,
        "q5_0" => DType::Q5_0,
        "q5_1" => DType::Q5_1,
        "q8_0" => DType::Q8_0,
        "q2_k" | "q2k" => DType::Q2K,
        "q3_k" | "q3k" => DType::Q3K,
        "q4_k" | "q4k" => DType::Q4K,
        "q5_k" | "q5k" => DType::Q5K,
        "q6_k" | "q6k" => DType::Q6K,
        _ => {
            return Err(format!(
                "Unknown quantization type: {}. Supported: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k",
                qtype
            )
            .into());
        }
    };

    // Set thread count if specified
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    eprintln!("╭─────────────────────────────────────────────────────────────────╮");
    eprintln!("│                     Model Quantization                          │");
    eprintln!("╰─────────────────────────────────────────────────────────────────╯");
    eprintln!();
    eprintln!("Input: {}", input_path);
    eprintln!("Output: {}", output_path);
    eprintln!("Target type: {:?}", target_dtype);
    eprintln!();

    // Load input model
    eprintln!("Loading input model...");
    let gguf = GgufFile::open(input_path)?;

    eprintln!("Model has {} tensors", gguf.data.tensors.len());

    // Count tensors by type
    let mut dtype_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for tensor in &gguf.data.tensors {
        let dtype_str = format!("{:?}", tensor.dtype);
        *dtype_counts.entry(dtype_str).or_insert(0) += 1;
    }

    eprintln!("Tensor types:");
    let mut sorted_types: Vec<_> = dtype_counts.iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));
    for (dtype, count) in sorted_types {
        eprintln!("  {}: {}", dtype, count);
    }

    // Note: Full quantization would require:
    // 1. Reading each tensor
    // 2. Converting to F32 if needed
    // 3. Quantizing to target format
    // 4. Writing new GGUF file
    //
    // This is a complex operation that requires a GGUF writer (not yet implemented)
    eprintln!();
    eprintln!("Note: Full quantization requires GGUF writer (not yet implemented).");
    eprintln!("This command currently only analyzes the model for quantization.");
    eprintln!();

    // Calculate estimated sizes
    let mut current_size = 0usize;
    let mut estimated_size = 0usize;

    for tensor in &gguf.data.tensors {
        let n_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();

        // Current size
        let current_dtype = DType::from(tensor.dtype);
        current_size += current_dtype.size_for_elements(n_elements);

        // Estimated size after quantization
        // Only quantize weight tensors (not embeddings, norms, etc.)
        let should_quantize = tensor.name.contains("weight")
            && !tensor.name.contains("norm")
            && !tensor.name.contains("embed");

        if should_quantize && !current_dtype.is_quantized() {
            estimated_size += target_dtype.size_for_elements(n_elements);
        } else {
            estimated_size += current_dtype.size_for_elements(n_elements);
        }
    }

    eprintln!("Size analysis:");
    eprintln!("  Current model size: {:.2} MB", current_size as f64 / 1024.0 / 1024.0);
    eprintln!("  Estimated quantized size: {:.2} MB", estimated_size as f64 / 1024.0 / 1024.0);
    eprintln!("  Estimated reduction: {:.1}%", 
              (1.0 - estimated_size as f64 / current_size as f64) * 100.0);

    Ok(())
}

/// Run model benchmark
fn run_benchmark(
    model_path: &str,
    n_prompt: usize,
    n_gen: usize,
    repetitions: usize,
    threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    // Set thread count if specified
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    eprintln!("╭─────────────────────────────────────────────────────────────────╮");
    eprintln!("│                      Model Benchmark                            │");
    eprintln!("╰─────────────────────────────────────────────────────────────────╯");
    eprintln!();

    // Load model
    eprintln!("Loading model: {}", model_path);
    let start = Instant::now();
    let loader = ModelLoader::load(model_path)?;
    let config = loader.config().clone();
    let model = loader.build_model()?;
    let load_time = start.elapsed();
    eprintln!("Model loaded in {:.2}s", load_time.as_secs_f64());
    eprintln!();

    // Create inference context
    let backend: Arc<dyn llama_rs::Backend> = Arc::new(llama_rs::backend::cpu::CpuBackend::new());
    let mut ctx = InferenceContext::new(&config, backend.clone());

    // Generate dummy prompt tokens
    let prompt_tokens: Vec<u32> = (0..n_prompt as u32).map(|i| i % 32000).collect();

    eprintln!("Configuration:");
    eprintln!("  Prompt tokens: {}", n_prompt);
    eprintln!("  Generation tokens: {}", n_gen);
    eprintln!("  Repetitions: {}", repetitions);
    eprintln!("  Threads: {}", rayon::current_num_threads());
    eprintln!();

    let mut prompt_times = Vec::with_capacity(repetitions);
    let mut gen_times = Vec::with_capacity(repetitions);

    for rep in 0..repetitions {
        eprintln!("Run {}/{}...", rep + 1, repetitions);
        ctx.reset();

        // Benchmark prompt processing (prefill)
        let start = Instant::now();
        for &token in &prompt_tokens {
            let _ = model.forward(&[token], &mut ctx)?;
        }
        let prompt_time = start.elapsed();
        prompt_times.push(prompt_time);

        // Benchmark generation (decode)
        let start = Instant::now();
        let mut last_token = *prompt_tokens.last().unwrap_or(&1);
        for _ in 0..n_gen {
            let logits = model.forward(&[last_token], &mut ctx)?;
            // Simple argmax instead of full sampling
            let logits_data = logits.as_f32()?;
            last_token = logits_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }
        let gen_time = start.elapsed();
        gen_times.push(gen_time);
    }

    // Calculate statistics
    let avg_prompt_time: f64 = prompt_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / repetitions as f64;
    let avg_gen_time: f64 = gen_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / repetitions as f64;

    let prompt_tps = n_prompt as f64 / avg_prompt_time;
    let gen_tps = n_gen as f64 / avg_gen_time;

    eprintln!();
    eprintln!("┌─ Results ─────────────────────────────────────────────────────────┐");
    eprintln!("│ Prompt processing (prefill):                                      │");
    eprintln!("│   Time: {:.3}s                                                     │", avg_prompt_time);
    eprintln!("│   Speed: {:.2} tokens/sec                                          │", prompt_tps);
    eprintln!("├───────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Text generation (decode):                                         │");
    eprintln!("│   Time: {:.3}s                                                     │", avg_gen_time);
    eprintln!("│   Speed: {:.2} tokens/sec                                          │", gen_tps);
    eprintln!("└───────────────────────────────────────────────────────────────────┘");

    // JSON output for scripting
    println!();
    println!("{{");
    println!("  \"prompt_tokens\": {},", n_prompt);
    println!("  \"gen_tokens\": {},", n_gen);
    println!("  \"prompt_time_s\": {:.4},", avg_prompt_time);
    println!("  \"gen_time_s\": {:.4},", avg_gen_time);
    println!("  \"prompt_tokens_per_sec\": {:.2},", prompt_tps);
    println!("  \"gen_tokens_per_sec\": {:.2}", gen_tps);
    println!("}}");

    Ok(())
}

/// Extract embeddings from text
fn run_embed(
    model_path: &str,
    text: &str,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use llama_rs::model::{EmbeddingConfig, EmbeddingExtractor};

    eprintln!("Loading model: {}", model_path);
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let loader = ModelLoader::load(model_path)?;
    let config = loader.config().clone();
    let model = loader.build_model()?;

    let backend: Arc<dyn llama_rs::Backend> = Arc::new(llama_rs::backend::cpu::CpuBackend::new());
    let mut ctx = InferenceContext::new(&config, backend.clone());

    let embed_config = EmbeddingConfig::default();
    let extractor = EmbeddingExtractor::new(embed_config, &config);

    eprintln!("Extracting embeddings for: \"{}\"", text);
    let embedding = extractor.embed_text(&model, &tokenizer, &mut ctx, text)?;

    match format {
        "json" => {
            println!("{{");
            println!("  \"text\": {:?},", text);
            println!("  \"dimension\": {},", embedding.len());
            println!("  \"embedding\": [");
            for (i, &val) in embedding.iter().enumerate() {
                if i < embedding.len() - 1 {
                    println!("    {:.6},", val);
                } else {
                    println!("    {:.6}", val);
                }
            }
            println!("  ]");
            println!("}}");
        }
        "raw" => {
            for val in &embedding {
                println!("{:.6}", val);
            }
        }
        _ => {
            eprintln!("Unknown format: {}. Using json.", format);
            // Fall back to json
            println!("{:?}", embedding);
        }
    }

    Ok(())
}

/// Download a model from HuggingFace Hub
fn run_download(
    repo: &str,
    file: Option<&str>,
    output_dir: Option<&str>,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse repository ID
    let repo_id = HfClient::parse_repo_id(repo)?;
    println!("Repository: {}", repo_id);

    // Create client
    let client = if let Some(dir) = output_dir {
        HfClient::with_cache_dir(std::path::PathBuf::from(dir))
    } else {
        HfClient::new()
    };

    println!("Cache directory: {}", client.cache_dir().display());

    // If no file specified, list available GGUF files
    if file.is_none() {
        println!("\nFetching available GGUF files...");
        let files = client.list_gguf_files(&repo_id)?;

        println!("\nAvailable GGUF files:");
        println!("{:<60} {:>12}", "Filename", "Size");
        println!("{}", "-".repeat(74));

        for f in &files {
            let size_str = f.file_size().map(format_bytes).unwrap_or_else(|| "?".to_string());
            println!("{:<60} {:>12}", f.path, size_str);
        }

        println!("\nTo download, run:");
        println!("  llama-rs download {} --file <filename>", repo);
        return Ok(());
    }

    let filename = file.unwrap();

    // Check if already cached (unless force flag is set)
    if !force && client.is_cached(&repo_id, filename) {
        let cached_path = client.get_cached_path(&repo_id, filename);
        println!("File already downloaded: {}", cached_path.display());
        println!("\nUse --force to re-download");
        return Ok(());
    }

    // Delete existing file if force flag is set
    if force {
        let cached_path = client.get_cached_path(&repo_id, filename);
        if cached_path.exists() {
            std::fs::remove_file(&cached_path)?;
            println!("Removed existing file");
        }
    }

    // Download the file
    println!("\nDownloading: {}", filename);
    let path = client.download_file(&repo_id, filename, true)?;

    println!("\nDownload complete!");
    println!("Model saved to: {}", path.display());
    println!("\nTo run inference:");
    println!("  llama-rs run {}", path.display());

    Ok(())
}

/// Handle models subcommands
fn run_models_command(action: ModelAction) -> Result<(), Box<dyn std::error::Error>> {
    let client = HfClient::new();

    match action {
        ModelAction::List => {
            println!("Cached models:");
            println!("{}", "-".repeat(80));

            let cached = client.list_cached()?;
            if cached.is_empty() {
                println!("No models cached yet.");
                println!("\nDownload a model with:");
                println!("  llama-rs download <repo> --file <filename>");
            } else {
                for (repo, path) in &cached {
                    let size = path.metadata().map(|m| format_bytes(m.len())).unwrap_or_default();
                    println!("{}", repo);
                    println!("  {} ({})", path.display(), size);
                }
            }
        }

        ModelAction::Search { query, limit } => {
            println!("Searching HuggingFace Hub for: \"{}\"", query);
            println!();

            let results = client.search_models(&query, limit)?;

            if results.is_empty() {
                println!("No models found matching your query.");
                return Ok(());
            }

            println!("{:<50} {:>10} {:>8}", "Repository", "Downloads", "Likes");
            println!("{}", "-".repeat(70));

            for model in &results {
                let id = model.model_id.as_ref().unwrap_or(&model.id);
                let downloads = model.downloads.map(|d| format!("{}", d)).unwrap_or_default();
                let likes = model.likes.map(|l| format!("{}", l)).unwrap_or_default();
                println!("{:<50} {:>10} {:>8}", id, downloads, likes);
            }

            println!("\nTo see available files:");
            println!("  llama-rs models list-files <repo>");
        }

        ModelAction::CacheInfo => {
            let cache_dir = client.cache_dir();
            println!("Cache directory: {}", cache_dir.display());

            let total_size = client.cache_size()?;
            println!("Total cache size: {}", format_bytes(total_size));

            let cached = client.list_cached()?;
            println!("Cached models: {}", cached.len());
        }

        ModelAction::ClearCache { yes } => {
            if !yes {
                print!("Are you sure you want to clear the model cache? [y/N] ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;

                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            let size_before = client.cache_size()?;
            client.clear_cache()?;
            println!("Cache cleared. Freed {}", format_bytes(size_before));
        }

        ModelAction::ListFiles { repo } => {
            let repo_id = HfClient::parse_repo_id(&repo)?;
            println!("Fetching files from: {}", repo_id);
            println!();

            let files = client.list_gguf_files(&repo_id)?;

            println!("{:<60} {:>12}", "Filename", "Size");
            println!("{}", "-".repeat(74));

            for f in &files {
                let size_str = f.file_size().map(format_bytes).unwrap_or_else(|| "?".to_string());
                let cached = if client.is_cached(&repo_id, &f.path) {
                    " [cached]"
                } else {
                    ""
                };
                println!("{:<60} {:>12}{}", f.path, size_str, cached);
            }

            println!("\nTo download:");
            println!("  llama-rs download {} --file <filename>", repo);
        }
    }

    Ok(())
}
