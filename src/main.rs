use std::io::{self, BufRead, Write};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use llama_gguf::Model;
use llama_gguf::config::Config;
use llama_gguf::engine::{ChatEngine, Engine, EngineConfig};
use llama_gguf::gguf::{GgufFile, MetadataValue};
#[cfg(feature = "huggingface")]
use llama_gguf::huggingface::{HfClient, format_bytes};
use llama_gguf::model::{InferenceContext, ModelLoader};

#[derive(Parser)]
#[command(name = "llama-gguf")]
#[command(about = "High-performance LLM inference engine with GGUF support", long_about = None)]
#[command(version)]
struct Cli {
    /// Path to TOML configuration file.
    /// Searches llama-gguf.toml, config/llama-gguf.toml, .llama-gguf.toml by default.
    #[arg(short, long, global = true, env = "LLAMA_CONFIG")]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show information about a model file (GGUF or ONNX)
    Info {
        /// Path to the model file (.gguf or .onnx)
        model: String,

        /// Show all metadata (verbose)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run inference on a model
    Run {
        /// Path to the model file (.gguf or .onnx)
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

        /// Use GPU acceleration (CUDA)
        #[arg(long)]
        gpu: bool,

        /// Path to external tokenizer (tokenizer.json or .gguf file)
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Interactive chat mode (local model or remote server)
    Chat {
        /// Path to the model file (not needed when using --server)
        model: Option<String>,

        /// Connect to a remote OpenAI-compatible server instead of loading a local model.
        /// Example: --server http://192.168.1.4:8080
        #[arg(long, env = "LLAMA_SERVER")]
        server: Option<String>,

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

        /// Path to external tokenizer (tokenizer.json or .gguf file)
        #[arg(long)]
        tokenizer: Option<String>,
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

        /// Enable RAG with PostgreSQL/pgvector database URL
        /// Format: postgres://user:pass@host:port/database
        #[cfg(feature = "rag")]
        #[arg(long, env = "RAG_DATABASE_URL")]
        rag_database_url: Option<String>,

        /// Path to RAG config file (alternative to --rag-database-url)
        #[cfg(feature = "rag")]
        #[arg(long)]
        rag_config: Option<String>,
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
    #[cfg(feature = "huggingface")]
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
    #[cfg(feature = "huggingface")]
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// RAG (Retrieval-Augmented Generation) operations
    #[cfg(feature = "rag")]
    Rag {
        #[command(subcommand)]
        action: RagAction,
    },

    /// Generate an example TOML configuration file
    #[command(name = "init-config")]
    InitConfig {
        /// Output path for the configuration file
        #[arg(short, long, default_value = "llama-gguf.toml")]
        output: String,
    },

    /// Generate and install man pages
    Manpages {
        /// Output directory for man pages.
        /// Typical locations: /usr/local/share/man/man1, ~/.local/share/man/man1
        #[arg(default_value = ".")]
        output_dir: String,
    },
}

/// RAG subcommands
#[cfg(feature = "rag")]
#[derive(Subcommand)]
enum RagAction {
    /// Initialize RAG database table
    Init {
        /// Path to TOML config file (optional, also checks rag.toml)
        #[arg(short, long)]
        config: Option<String>,

        /// PostgreSQL connection string (overrides config file)
        #[arg(long, env = "RAG_DATABASE_URL")]
        database_url: Option<String>,

        /// Embeddings table name (overrides config file)
        #[arg(long)]
        table: Option<String>,

        /// Embedding dimension (overrides config file)
        #[arg(long)]
        dim: Option<usize>,
    },

    /// Index documents into the vector store
    Index {
        /// Path to file or directory to index
        path: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// PostgreSQL connection string (overrides config file)
        #[arg(long, env = "RAG_DATABASE_URL")]
        database_url: Option<String>,

        /// Embeddings table name (overrides config file)
        #[arg(long)]
        table: Option<String>,

        /// Chunk size in characters
        #[arg(long, default_value = "500")]
        chunk_size: usize,

        /// Chunk overlap in characters
        #[arg(long, default_value = "50")]
        chunk_overlap: usize,
    },

    /// Search the vector store
    Search {
        /// Search query
        query: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// PostgreSQL connection string (overrides config file)
        #[arg(long, env = "RAG_DATABASE_URL")]
        database_url: Option<String>,

        /// Embeddings table name (overrides config file)
        #[arg(long)]
        table: Option<String>,

        /// Number of results (overrides config file)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Metadata filters (can be specified multiple times)
        /// Format: field=value, field!=value, field>value, field>=value,
        ///         field<value, field<=value, field~value (contains),
        ///         field^value (starts with), field$value (ends with),
        ///         field? (exists), !field? (not exists)
        #[arg(short = 'f', long = "filter", value_name = "FILTER")]
        filters: Vec<String>,
    },

    /// List unique values for a metadata field
    ListValues {
        /// Metadata field to list values for
        field: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// PostgreSQL connection string (overrides config file)
        #[arg(long, env = "RAG_DATABASE_URL")]
        database_url: Option<String>,

        /// Embeddings table name (overrides config file)
        #[arg(long)]
        table: Option<String>,

        /// Maximum values to return
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },

    /// Delete documents matching a filter
    Delete {
        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// PostgreSQL connection string (overrides config file)
        #[arg(long, env = "RAG_DATABASE_URL")]
        database_url: Option<String>,

        /// Embeddings table name (overrides config file)
        #[arg(long)]
        table: Option<String>,

        /// Metadata filters (required, can be specified multiple times)
        #[arg(short = 'f', long = "filter", value_name = "FILTER", required = true)]
        filters: Vec<String>,

        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },

    /// Show RAG database statistics
    Stats {
        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// PostgreSQL connection string (overrides config file)
        #[arg(long, env = "RAG_DATABASE_URL")]
        database_url: Option<String>,

        /// Embeddings table name (overrides config file)
        #[arg(long)]
        table: Option<String>,
    },

    /// Generate an example configuration file
    GenConfig {
        /// Output path for the config file
        #[arg(short, long, default_value = "rag.toml")]
        output: String,
    },

    // =========================================================================
    // Knowledge Base Commands (Bedrock-style API)
    // =========================================================================
    /// Create a new knowledge base
    #[command(name = "kb-create")]
    KbCreate {
        /// Knowledge base name
        name: String,

        /// Description
        #[arg(short, long)]
        description: Option<String>,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// Chunking strategy: none, fixed, semantic, hierarchical
        #[arg(long, default_value = "fixed")]
        chunking: String,

        /// Max tokens per chunk (for fixed/semantic chunking)
        #[arg(long, default_value = "300")]
        max_tokens: usize,

        /// Overlap percentage (for fixed/hierarchical chunking)
        #[arg(long, default_value = "20")]
        overlap: u8,
    },

    /// Ingest data into a knowledge base
    #[command(name = "kb-ingest")]
    KbIngest {
        /// Knowledge base name
        #[arg(short, long)]
        name: String,

        /// Path to file or directory to ingest
        path: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// File pattern for directories (e.g., "*.md")
        #[arg(long)]
        pattern: Option<String>,

        /// Recursive directory search
        #[arg(long, default_value = "true")]
        recursive: bool,
    },

    /// Query a knowledge base (retrieve only)
    #[command(name = "kb-retrieve")]
    KbRetrieve {
        /// Query text
        query: String,

        /// Knowledge base name
        #[arg(short, long)]
        name: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Minimum similarity score
        #[arg(long, default_value = "0.5")]
        min_score: f32,
    },

    /// Retrieve and generate (RAG pipeline)
    #[command(name = "kb-rag")]
    KbRetrieveAndGenerate {
        /// Query text
        query: String,

        /// Knowledge base name
        #[arg(short, long)]
        name: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// Maximum results to retrieve
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Custom prompt template (use {context} and {query} placeholders)
        #[arg(long)]
        prompt_template: Option<String>,

        /// Show citations
        #[arg(long, default_value = "true")]
        citations: bool,
    },

    /// Show knowledge base statistics
    #[command(name = "kb-stats")]
    KbStats {
        /// Knowledge base name
        #[arg(short, long)]
        name: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Delete a knowledge base
    #[command(name = "kb-delete")]
    KbDelete {
        /// Knowledge base name
        #[arg(short, long)]
        name: String,

        /// Path to TOML config file
        #[arg(short, long)]
        config: Option<String>,

        /// Skip confirmation
        #[arg(long)]
        force: bool,
    },
}

#[cfg(feature = "huggingface")]
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

    // Load configuration with precedence: CLI > env > config file > defaults
    let cfg = match Config::load(cli.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            // Only fatal if the user explicitly passed --config
            if cli.config.is_some() {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            }
            Config::default()
        }
    };

    match cli.command {
        Commands::Info { model, verbose } => {
            let model_path = resolve_model_path(&model, &cfg);
            if let Err(e) = show_info(&model_path, verbose) {
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
            gpu,
            tokenizer,
        } => {
            // CLI args override config file values; clap defaults are used when
            // the user didn't explicitly pass an argument (we detect this by
            // checking if the value matches clap's default and the config differs).
            let ec = cfg.to_engine_config(Some(&model));
            if let Err(e) = run_inference(
                &resolve_model_path(&model, &cfg),
                prompt.as_deref(),
                cli_or_config(n_predict, 128, ec.max_tokens),
                cli_or_config_f32(temperature, 0.8, ec.temperature),
                cli_or_config(top_k, 40, ec.top_k),
                cli_or_config_f32(top_p, 0.95, ec.top_p),
                cli_or_config_f32(repeat_penalty, 1.1, ec.repeat_penalty),
                seed.or(ec.seed),
                gpu || ec.use_gpu,
                tokenizer,
            ) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Chat {
            model,
            server,
            system,
            n_predict,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            tokenizer,
        } => {
            // Determine server URL: --server flag > config file > none
            let server_url = server.or_else(|| cfg.server.host_url());

            if let Some(url) = server_url {
                // Remote mode: connect to server
                #[cfg(feature = "client")]
                {
                    let system_prompt = system.or(cfg.chat.system_prompt.clone());
                    if let Err(e) = run_remote_chat(
                        &url,
                        system_prompt.as_deref(),
                        cli_or_config(n_predict, 512, cfg.chat.max_tokens.unwrap_or(512)),
                        cli_or_config_f32(temperature, 0.7, cfg.chat.temperature.unwrap_or(0.7)),
                        cli_or_config_f32(top_p, 0.9, cfg.chat.top_p.unwrap_or(0.9)),
                    ) {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                }
                #[cfg(not(feature = "client"))]
                {
                    eprintln!("Error: Remote chat requires the 'client' feature.");
                    eprintln!("Compile with: cargo build --features client");
                    std::process::exit(1);
                }
            } else {
                // Local mode: load model
                let model = model.unwrap_or_else(|| {
                    cfg.model.path.clone().unwrap_or_else(|| {
                        eprintln!(
                            "Error: No model specified. Provide a model path or use --server."
                        );
                        eprintln!(
                            "Usage: llama-gguf chat <MODEL> or llama-gguf chat --server <URL>"
                        );
                        std::process::exit(1);
                    })
                });
                let ec = cfg.to_chat_engine_config(Some(&model));
                let system_prompt = system.or(cfg.chat.system_prompt.clone());
                if let Err(e) = run_chat(
                    &resolve_model_path(&model, &cfg),
                    system_prompt.as_deref(),
                    cli_or_config(n_predict, 512, ec.max_tokens),
                    cli_or_config_f32(temperature, 0.7, ec.temperature),
                    cli_or_config(top_k, 40, ec.top_k),
                    cli_or_config_f32(top_p, 0.9, ec.top_p),
                    cli_or_config_f32(repeat_penalty, 1.1, ec.repeat_penalty),
                    seed.or(ec.seed),
                    tokenizer,
                ) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        #[cfg(feature = "server")]
        Commands::Serve {
            model,
            host,
            port,
            #[cfg(feature = "rag")]
            rag_database_url,
            #[cfg(feature = "rag")]
            rag_config,
        } => {
            let model_path = resolve_model_path(&model, &cfg);
            let host = if host == "127.0.0.1" && cfg.server.host != "127.0.0.1" {
                cfg.server.host.clone()
            } else {
                host
            };
            let port = if port == 8080 && cfg.server.port != 8080 {
                cfg.server.port
            } else {
                port
            };

            // Determine RAG database URL
            #[cfg(feature = "rag")]
            let rag_url = if let Some(url) = rag_database_url {
                Some(url)
            } else if let Some(config_path) = rag_config {
                match llama_gguf::rag::RagConfig::load(Some(&config_path)) {
                    Ok(config) => Some(config.connection_string().to_string()),
                    Err(e) => {
                        eprintln!("Warning: Failed to load RAG config: {}", e);
                        cfg.server.rag_database_url.clone()
                    }
                }
            } else if cfg.server.rag_database_url.is_some() {
                cfg.server.rag_database_url.clone()
            } else {
                match llama_gguf::rag::RagConfig::load(None::<&str>) {
                    Ok(config) if !config.connection_string().is_empty() => {
                        Some(config.connection_string().to_string())
                    }
                    _ => None,
                }
            };

            #[cfg(not(feature = "rag"))]
            let rag_url: Option<String> = None;

            if let Err(e) = run_server(&model_path, &host, port, rag_url) {
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
            let qtype = if qtype == "q4_0" && cfg.quantize.output_type != "q4_0" {
                cfg.quantize.output_type.clone()
            } else {
                qtype
            };
            let threads = threads.or(cfg.quantize.threads);
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
            if let Err(e) = run_benchmark(
                &resolve_model_path(&model, &cfg),
                cli_or_config(n_prompt, 512, cfg.bench.n_prompt),
                cli_or_config(n_gen, 128, cfg.bench.n_gen),
                cli_or_config(repetitions, 3, cfg.bench.repetitions),
                threads.or(cfg.bench.threads),
            ) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Embed {
            model,
            text,
            format,
        } => {
            let format = if format == "json" && cfg.embed.format != "json" {
                cfg.embed.format.clone()
            } else {
                format
            };
            if let Err(e) = run_embed(&resolve_model_path(&model, &cfg), &text, &format) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        #[cfg(feature = "huggingface")]
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
        #[cfg(feature = "huggingface")]
        Commands::Models { action } => {
            if let Err(e) = run_models_command(action) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        #[cfg(feature = "rag")]
        Commands::Rag { action } => {
            if let Err(e) = run_rag_command(action) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::InitConfig { output } => {
            if std::path::Path::new(&output).exists() {
                eprintln!("Config file already exists: {}", output);
                eprintln!("Remove it first or choose a different path with --output");
                std::process::exit(1);
            }
            if let Err(e) = std::fs::write(&output, llama_gguf::config::example_config()) {
                eprintln!("Error writing config: {}", e);
                std::process::exit(1);
            }
            eprintln!("Created configuration file: {}", output);
            eprintln!("Edit it to set your model path and preferred defaults.");
        }
        Commands::Manpages { output_dir } => {
            if let Err(e) = generate_manpages(&output_dir) {
                eprintln!("Error generating man pages: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Generate man pages using clap_mangen
fn generate_manpages(output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    use clap::CommandFactory;
    use clap_mangen::Man;
    use std::fs;
    use std::path::Path;

    let out_dir = Path::new(output_dir);

    // Create output directory if it doesn't exist
    fs::create_dir_all(out_dir)?;

    // Generate main command man page
    let cmd = Cli::command();
    let man = Man::new(cmd.clone());
    let mut buffer = Vec::new();
    man.render(&mut buffer)?;
    let main_path = out_dir.join("llama-gguf.1");
    fs::write(&main_path, buffer)?;
    eprintln!("Generated: {}", main_path.display());

    // Generate man pages for each subcommand
    for subcommand in cmd.get_subcommands() {
        let name = subcommand.get_name();
        // Skip hidden commands
        if subcommand.is_hide_set() {
            continue;
        }

        let sub_man = Man::new(subcommand.clone());
        let mut buffer = Vec::new();
        sub_man.render(&mut buffer)?;

        let filename = format!("llama-gguf-{}.1", name);
        let path = out_dir.join(&filename);
        fs::write(&path, buffer)?;
        eprintln!("Generated: {}", path.display());

        // Handle nested subcommands (like "models list", "rag init", etc.)
        for nested in subcommand.get_subcommands() {
            if nested.is_hide_set() {
                continue;
            }
            let nested_name = nested.get_name();
            let nested_man = Man::new(nested.clone());
            let mut buffer = Vec::new();
            nested_man.render(&mut buffer)?;

            let filename = format!("llama-gguf-{}-{}.1", name, nested_name);
            let path = out_dir.join(&filename);
            fs::write(&path, buffer)?;
            eprintln!("Generated: {}", path.display());
        }
    }

    eprintln!();
    eprintln!("Man pages generated in: {}", out_dir.display());
    eprintln!();
    eprintln!("To install system-wide (requires sudo):");
    eprintln!(
        "  sudo cp {}/*.1 /usr/local/share/man/man1/",
        out_dir.display()
    );
    eprintln!("  sudo mandb");
    eprintln!();
    eprintln!("Or install for current user:");
    eprintln!("  mkdir -p ~/.local/share/man/man1");
    eprintln!("  cp {}/*.1 ~/.local/share/man/man1/", out_dir.display());
    eprintln!("  mandb -u");

    Ok(())
}

// ============================================================================
// Config merge helpers
// ============================================================================

/// If the CLI value equals clap's default, prefer the config file value.
/// Otherwise the user explicitly passed it, so use the CLI value.
fn cli_or_config<T: PartialEq>(cli_val: T, clap_default: T, config_val: T) -> T {
    if cli_val == clap_default {
        config_val
    } else {
        cli_val
    }
}

/// Float variant (uses approximate equality for f32 defaults).
fn cli_or_config_f32(cli_val: f32, clap_default: f32, config_val: f32) -> f32 {
    if (cli_val - clap_default).abs() < f32::EPSILON {
        config_val
    } else {
        cli_val
    }
}

/// Resolve the model path: CLI positional arg wins, then config file, then empty.
fn resolve_model_path(cli_model: &str, cfg: &Config) -> String {
    if !cli_model.is_empty() {
        cli_model.to_string()
    } else {
        cfg.model.path.clone().unwrap_or_default()
    }
}

#[cfg(feature = "server")]
fn run_server(
    model_path: &str,
    host: &str,
    port: u16,
    rag_database_url: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        llama_gguf::server::run_server(llama_gguf::server::ServerConfig {
            host: host.to_string(),
            port,
            model_path: model_path.to_string(),
            #[cfg(feature = "rag")]
            rag_database_url,
        })
        .await
    })
}

#[allow(clippy::too_many_arguments)]
fn run_inference(
    model_path: &str,
    prompt: Option<&str>,
    n_predict: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
    use_gpu: bool,
    tokenizer_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::load(EngineConfig {
        model_path: model_path.to_string(),
        tokenizer_path,
        temperature,
        top_k,
        top_p,
        repeat_penalty,
        max_tokens: n_predict,
        seed,
        use_gpu,
    })?;

    let raw_prompt = prompt.unwrap_or("Hello");

    // Print the prompt before streaming the response
    print!("{}", raw_prompt);
    io::stdout().flush()?;

    // Stream tokens as they are generated
    let mut count = 0;
    for token_result in engine.generate_streaming(raw_prompt, n_predict) {
        let text = token_result?;
        print!("{}", text);
        io::stdout().flush()?;
        count += 1;
    }

    println!();
    eprintln!("\nGenerated {} tokens", count);

    Ok(())
}

/// Interactive chat mode
#[allow(clippy::too_many_arguments)]
fn run_chat(
    model_path: &str,
    system_prompt: Option<&str>,
    n_predict: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
    tokenizer_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::load(EngineConfig {
        model_path: model_path.to_string(),
        tokenizer_path,
        temperature,
        top_k,
        top_p,
        repeat_penalty,
        max_tokens: n_predict,
        seed,
        use_gpu: false,
    })?;

    let system_text = system_prompt.unwrap_or("You are a helpful AI assistant.");
    let mut chat = ChatEngine::new(engine, Some(system_text.to_string()));

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
                    chat.clear_history();
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
                    eprintln!("System prompt: {}", chat.system_prompt());
                    continue;
                }
                _ => {
                    eprintln!(
                        "Unknown command: {}. Type /help for available commands.",
                        input
                    );
                    continue;
                }
            }
        }

        // Generate and stream the response
        print!("\nAssistant: ");
        io::stdout().flush()?;

        let stream = chat.chat_streaming(input)?;
        for token_result in stream {
            let text = token_result?;
            print!("{}", text);
            io::stdout().flush()?;
        }

        println!();
        println!();
    }

    Ok(())
}

/// Remote chat mode: connect to an OpenAI-compatible server
#[cfg(feature = "client")]
fn run_remote_chat(
    server_url: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use llama_gguf::client::RemoteChatClient;

    let mut client =
        RemoteChatClient::new(server_url, system_prompt, temperature, max_tokens, top_p)?;

    let system_text = client.system_prompt().to_string();
    let model_name = client.model_name().to_string();

    // Print chat header
    eprintln!();
    eprintln!("╭─────────────────────────────────────────────────────────────────╮");
    eprintln!("│                 Interactive Chat Mode (Remote)                    │");
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Server: {:<56}│", server_url);
    eprintln!("│ Model:  {:<56}│", &model_name[..model_name.len().min(56)]);
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Commands:                                                        │");
    eprintln!("│   /clear  - Clear conversation history                           │");
    eprintln!("│   /system - Show system prompt                                   │");
    eprintln!("│   /help   - Show this help                                       │");
    eprintln!("│   /quit   - Exit chat                                            │");
    eprintln!("╰─────────────────────────────────────────────────────────────────╯");
    eprintln!();
    eprintln!("System: {}", system_text);
    eprintln!();

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
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
                    client.clear_history();
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
                    eprintln!("System prompt: {}", client.system_prompt());
                    continue;
                }
                _ => {
                    eprintln!(
                        "Unknown command: {}. Type /help for available commands.",
                        input
                    );
                    continue;
                }
            }
        }

        // Generate and stream the response
        print!("\nAssistant: ");
        io::stdout().flush()?;

        match client.chat_streaming(input) {
            Ok(_) => {
                println!();
                println!();
            }
            Err(e) => {
                eprintln!("\nError: {}", e);
                eprintln!();
            }
        }
    }

    Ok(())
}

fn show_info(path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect format
    if path.ends_with(".onnx") {
        return show_onnx_info(path, verbose);
    }

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
        println!(
            "│ ... and {} more tensors{:>29} │",
            data.tensors.len() - max_tensors,
            ""
        );
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

fn show_onnx_info(path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "onnx")]
    {
        use llama_gguf::onnx::reader::onnx_dtype_name;
        use llama_gguf::onnx::{OnnxFile, OnnxModelLoader};

        let onnx = OnnxFile::open(path)?;
        let meta = onnx.metadata();

        println!("╭─────────────────────────────────────────────────────────────────╮");
        println!("│                        ONNX Model Info                          │");
        println!("╰─────────────────────────────────────────────────────────────────╯");
        println!();
        println!("File: {}", path);
        println!("IR Version: {}", meta.ir_version);
        println!("Producer: {} {}", meta.producer_name, meta.producer_version);
        if !meta.domain.is_empty() {
            println!("Domain: {}", meta.domain);
        }
        println!();

        // Opset info
        println!("┌─ Opset Imports ─────────────────────────────────────────────────┐");
        for (domain, version) in &meta.opset_imports {
            let domain_display = if domain.is_empty() { "ai.onnx" } else { domain };
            println!("│ {:<40} v{:<21} │", domain_display, version);
        }
        println!("└───────────────────────────────────────────────────────────────┘");
        println!();

        // Try to load config.json for architecture info
        let model_dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let config_path = model_dir.join("config.json");
        if config_path.exists()
            && let Ok(loader) = OnnxModelLoader::load(path)
        {
            let config = loader.config();
            println!("┌─ Model Configuration (from config.json) ─────────────────────┐");
            println!(
                "│ Architecture: {:<49} │",
                format!("{:?}", loader.architecture())
            );
            println!("│ Hidden size: {:<50} │", config.hidden_size);
            println!("│ Intermediate size: {:<44} │", config.intermediate_size);
            println!("│ Layers: {:<55} │", config.num_layers);
            println!("│ Attention heads: {:<46} │", config.num_heads);
            println!("│ KV heads: {:<53} │", config.num_kv_heads);
            println!("│ Head dim: {:<53} │", config.head_dim);
            println!("│ Max seq len: {:<50} │", config.max_seq_len);
            println!("│ Vocab size: {:<51} │", config.vocab_size);
            println!("│ RoPE freq base: {:<47} │", config.rope_config.freq_base);
            println!("│ Norm eps: {:<53} │", format!("{:.2e}", config.norm_eps));
            println!("└───────────────────────────────────────────────────────────────┘");
            println!();
        }

        // Graph inputs/outputs
        if let Ok(inputs) = onnx.inputs() {
            println!("┌─ Graph Inputs ─────────────────────────────────────────────────┐");
            for (name, dims) in &inputs {
                let dims_str = format!("{:?}", dims);
                println!("│ {:<40} {:>22} │", truncate(name, 40), dims_str);
            }
            println!("└───────────────────────────────────────────────────────────────┘");
            println!();
        }

        if let Ok(outputs) = onnx.outputs() {
            println!("┌─ Graph Outputs ────────────────────────────────────────────────┐");
            for name in &outputs {
                println!("│ {:<63} │", truncate(name, 63));
            }
            println!("└───────────────────────────────────────────────────────────────┘");
            println!();
        }

        // Show initializer tensors (weights)
        if let Ok(infos) = onnx.tensor_infos() {
            println!("┌─ Initializer Tensors (Weights) ────────────────────────────────┐");
            let max_tensors = if verbose { infos.len() } else { 15 };
            for info in infos.iter().take(max_tensors) {
                let dims_str = format!("{:?}", info.dims);
                let dtype_str = onnx_dtype_name(info.data_type);
                println!(
                    "│ {:<30} {:>16} {:>8} │",
                    truncate(&info.name, 30),
                    dims_str,
                    dtype_str
                );
            }
            if infos.len() > max_tensors {
                println!(
                    "│ ... and {} more tensors{:>29} │",
                    infos.len() - max_tensors,
                    ""
                );
            }
            println!("│                                                               │");
            println!("│ Total initializers: {:<43} │", infos.len());
            println!("└───────────────────────────────────────────────────────────────┘");
        }

        // Show graph nodes if verbose
        if verbose && let Ok(nodes) = onnx.nodes() {
            println!();
            println!(
                "┌─ Graph Nodes ({}) ─────────────────────────────────────────┐",
                nodes.len()
            );
            for node in nodes.iter().take(50) {
                println!(
                    "│ {:<20} {:<43} │",
                    truncate(&node.op_type, 20),
                    truncate(&node.name, 43)
                );
            }
            if nodes.len() > 50 {
                println!("│ ... and {} more nodes{:>30} │", nodes.len() - 50, "");
            }
            println!("└───────────────────────────────────────────────────────────────┘");
        }

        Ok(())
    }

    #[cfg(not(feature = "onnx"))]
    {
        let _ = (path, verbose);
        Err(
            "ONNX support requires the `onnx` feature. Build with: cargo build --features onnx"
                .into(),
        )
    }
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
    use llama_gguf::backend::cpu::CpuBackend;

    let backend = CpuBackend::new();

    println!("╭─────────────────────────────────────────────────────────────────╮");
    println!("│                     System Information                          │");
    println!("╰─────────────────────────────────────────────────────────────────╯");
    println!();

    // CPU info
    println!("┌─ CPU ────────────────────────────────────────────────────────────┐");
    println!("│ Threads: {:<54} │", backend.num_threads());
    println!("│ SIMD: {:<57} │", backend.simd_info());
    println!(
        "│ AVX2: {:<57} │",
        if backend.has_avx2() { "yes" } else { "no" }
    );
    println!(
        "│ AVX-512: {:<54} │",
        if backend.has_avx512() { "yes" } else { "no" }
    );
    println!(
        "│ NEON: {:<57} │",
        if backend.has_neon() { "yes" } else { "no" }
    );
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
    use llama_gguf::tensor::DType;

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
    let mut dtype_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
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
    eprintln!(
        "  Current model size: {:.2} MB",
        current_size as f64 / 1024.0 / 1024.0
    );
    eprintln!(
        "  Estimated quantized size: {:.2} MB",
        estimated_size as f64 / 1024.0 / 1024.0
    );
    eprintln!(
        "  Estimated reduction: {:.1}%",
        (1.0 - estimated_size as f64 / current_size as f64) * 100.0
    );

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
    let backend: Arc<dyn llama_gguf::Backend> =
        Arc::new(llama_gguf::backend::cpu::CpuBackend::new());
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
    let avg_prompt_time: f64 =
        prompt_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / repetitions as f64;
    let avg_gen_time: f64 =
        gen_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / repetitions as f64;

    let prompt_tps = n_prompt as f64 / avg_prompt_time;
    let gen_tps = n_gen as f64 / avg_gen_time;

    eprintln!();
    eprintln!("┌─ Results ─────────────────────────────────────────────────────────┐");
    eprintln!("│ Prompt processing (prefill):                                      │");
    eprintln!(
        "│   Time: {:.3}s                                                     │",
        avg_prompt_time
    );
    eprintln!(
        "│   Speed: {:.2} tokens/sec                                          │",
        prompt_tps
    );
    eprintln!("├───────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Text generation (decode):                                         │");
    eprintln!(
        "│   Time: {:.3}s                                                     │",
        avg_gen_time
    );
    eprintln!(
        "│   Speed: {:.2} tokens/sec                                          │",
        gen_tps
    );
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
fn run_embed(model_path: &str, text: &str, format: &str) -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::load(EngineConfig {
        model_path: model_path.to_string(),
        ..Default::default()
    })?;

    eprintln!("Extracting embeddings for: \"{}\"", text);
    let embedding = engine.embed(text)?;

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
#[cfg(feature = "huggingface")]
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
            let size_str = f
                .file_size()
                .map(format_bytes)
                .unwrap_or_else(|| "?".to_string());
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
#[cfg(feature = "huggingface")]
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
                    let size = path
                        .metadata()
                        .map(|m| format_bytes(m.len()))
                        .unwrap_or_default();
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
                let downloads = model
                    .downloads
                    .map(|d| format!("{}", d))
                    .unwrap_or_default();
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
                let size_str = f
                    .file_size()
                    .map(format_bytes)
                    .unwrap_or_else(|| "?".to_string());
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

/// Handle RAG subcommands
#[cfg(feature = "rag")]
fn run_rag_command(action: RagAction) -> Result<(), Box<dyn std::error::Error>> {
    use llama_gguf::rag::{RagConfig, RagContextBuilder, RagStore, example_config};

    // Create tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    match action {
        RagAction::Init {
            config,
            database_url,
            table,
            dim,
        } => {
            rt.block_on(async {
                // Load config with precedence: CLI args > env vars > config file > defaults
                let mut cfg = RagConfig::load(config.as_deref())?;

                // Apply CLI overrides
                if let Some(url) = database_url {
                    cfg.database.connection_string = url;
                }
                if let Some(t) = table {
                    cfg.embeddings.table_name = t;
                }
                if let Some(d) = dim {
                    cfg.embeddings.dimension = d;
                }

                println!("Initializing RAG database...");
                println!("  Table: {}", cfg.table_name());
                println!("  Embedding dimension: {}", cfg.embedding_dim());

                let store = RagStore::connect(cfg).await?;
                store.create_table().await?;

                println!("\nDatabase initialized successfully!");
                println!("\nTo index documents:");
                println!("  llama-gguf rag index <path>");

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::Index {
            path,
            config,
            database_url,
            table,
            chunk_size,
            chunk_overlap,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::{NewDocument, TextChunker};
                use std::path::Path;

                // Load config
                let mut cfg = RagConfig::load(config.as_deref())?;

                // Apply CLI overrides
                if let Some(url) = database_url {
                    cfg.database.connection_string = url;
                }
                if let Some(t) = table {
                    cfg.embeddings.table_name = t;
                }

                println!("Indexing documents from: {}", path);

                let store = RagStore::connect(cfg).await?;

                let chunker = TextChunker::new(chunk_size).with_overlap(chunk_overlap);
                let path = Path::new(&path);

                let mut total_chunks = 0;

                if path.is_file() {
                    let content = std::fs::read_to_string(path)?;
                    let chunks = chunker.chunk(&content);

                    for chunk in chunks {
                        let embedding = vec![0.0f32; store.config().embedding_dim()];

                        let doc = NewDocument {
                            content: chunk,
                            embedding,
                            metadata: Some(serde_json::json!({
                                "source": path.to_string_lossy()
                            })),
                        };

                        store.insert(doc).await?;
                        total_chunks += 1;
                    }
                } else if path.is_dir() {
                    for entry in std::fs::read_dir(path)? {
                        let entry = entry?;
                        let file_path = entry.path();

                        if file_path.is_file()
                            && let Ok(content) = std::fs::read_to_string(&file_path) {
                                let chunks = chunker.chunk(&content);

                                for chunk in chunks {
                                    let embedding = vec![0.0f32; store.config().embedding_dim()];

                                    let doc = NewDocument {
                                        content: chunk,
                                        embedding,
                                        metadata: Some(serde_json::json!({
                                            "source": file_path.to_string_lossy()
                                        })),
                                    };

                                    store.insert(doc).await?;
                                    total_chunks += 1;
                                }
                            }
                    }
                }

                println!("\nIndexed {} chunks", total_chunks);
                println!("\nNote: Using placeholder embeddings. For production, integrate a real embedding model.");

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::Search {
            query,
            config,
            database_url,
            table,
            limit,
            filters,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::MetadataFilter;

                // Load config
                let mut cfg = RagConfig::load(config.as_deref())?;

                // Apply CLI overrides
                if let Some(url) = database_url {
                    cfg.database.connection_string = url;
                }
                if let Some(t) = table {
                    cfg.embeddings.table_name = t;
                }
                if let Some(l) = limit {
                    cfg.search.max_results = l;
                }

                // Parse filters
                let filter = if filters.is_empty() {
                    None
                } else {
                    let parsed: Result<Vec<MetadataFilter>, _> =
                        filters.iter().map(|f| MetadataFilter::parse(f)).collect();

                    match parsed {
                        Ok(fs) if fs.len() == 1 => Some(fs.into_iter().next().unwrap()),
                        Ok(fs) => Some(MetadataFilter::and(fs)),
                        Err(e) => {
                            eprintln!("Error parsing filter: {}", e);
                            return Err(e.into());
                        }
                    }
                };

                println!("Searching for: \"{}\"", query);
                if !filters.is_empty() {
                    println!("Filters: {:?}", filters);
                }
                println!();

                let store = RagStore::connect(cfg).await?;

                // Generate query embedding (placeholder)
                let query_embedding = vec![0.0f32; store.config().embedding_dim()];

                let results = store
                    .search_with_filter(&query_embedding, limit, filter)
                    .await?;

                if results.is_empty() {
                    println!("No results found.");
                } else {
                    println!("Found {} results:\n", results.len());

                    for (i, doc) in results.iter().enumerate() {
                        let score = doc.score.map(|s| format!("{:.4}", s)).unwrap_or_default();
                        let preview: String = doc.content.chars().take(200).collect();

                        println!("{}. [{}] {}", i + 1, score, preview);
                        if let Some(meta) = &doc.metadata {
                            println!(
                                "   Metadata: {}",
                                serde_json::to_string_pretty(meta).unwrap_or_default()
                            );
                        }
                        println!();
                    }

                    // Show context builder example
                    let context = RagContextBuilder::new(results).with_scores(true).build();

                    println!("--- Combined Context ---");
                    println!("{}", &context[..context.len().min(500)]);
                    if context.len() > 500 {
                        println!("... (truncated)");
                    }
                }

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::ListValues {
            field,
            config,
            database_url,
            table,
            limit,
        } => {
            rt.block_on(async {
                // Load config
                let mut cfg = RagConfig::load(config.as_deref())?;

                if let Some(url) = database_url {
                    cfg.database.connection_string = url;
                }
                if let Some(t) = table {
                    cfg.embeddings.table_name = t;
                }

                let store = RagStore::connect(cfg).await?;
                let values = store.list_metadata_values(&field, Some(limit)).await?;

                println!("Unique values for '{}':", field);
                println!("{}", "-".repeat(40));

                if values.is_empty() {
                    println!("(no values found)");
                } else {
                    for value in &values {
                        println!("  {}", value);
                    }
                    println!("\nTotal: {} unique values", values.len());
                }

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::Delete {
            config,
            database_url,
            table,
            filters,
            force,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::MetadataFilter;

                // Load config
                let mut cfg = RagConfig::load(config.as_deref())?;

                if let Some(url) = database_url {
                    cfg.database.connection_string = url;
                }
                if let Some(t) = table {
                    cfg.embeddings.table_name = t;
                }

                // Parse filters
                let parsed: Result<Vec<MetadataFilter>, _> =
                    filters.iter().map(|f| MetadataFilter::parse(f)).collect();

                let filter = match parsed {
                    Ok(fs) if fs.len() == 1 => fs.into_iter().next().unwrap(),
                    Ok(fs) => MetadataFilter::and(fs),
                    Err(e) => {
                        eprintln!("Error parsing filter: {}", e);
                        return Err(e.into());
                    }
                };

                let store = RagStore::connect(cfg).await?;

                // Count documents to be deleted
                let count = store.count_with_filter(Some(filter.clone())).await?;

                if count == 0 {
                    println!("No documents match the filter.");
                    return Ok(());
                }

                println!("Documents matching filter: {}", count);

                if !force {
                    print!("Delete {} documents? [y/N] ", count);
                    use std::io::{self, Write};
                    io::stdout().flush()?;

                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;

                    if !input.trim().eq_ignore_ascii_case("y") {
                        println!("Cancelled.");
                        return Ok(());
                    }
                }

                let deleted = store.delete_with_filter(filter).await?;
                println!("Deleted {} documents.", deleted);

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::Stats {
            config,
            database_url,
            table,
        } => {
            rt.block_on(async {
                // Load config
                let mut cfg = RagConfig::load(config.as_deref())?;

                // Apply CLI overrides
                if let Some(url) = database_url {
                    cfg.database.connection_string = url;
                }
                if let Some(t) = table {
                    cfg.embeddings.table_name = t;
                }

                let store = RagStore::connect(cfg).await?;

                let count = store.count().await?;

                println!("RAG Database Statistics");
                println!("{}", "-".repeat(40));
                println!("Table: {}", store.config().table_name());
                println!("Documents: {}", count);
                println!("Embedding dimension: {}", store.config().embedding_dim());
                println!("Distance metric: {:?}", store.config().distance_metric());

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::GenConfig { output } => {
            std::fs::write(&output, example_config())?;
            println!("Generated example configuration: {}", output);
            println!("\nEdit this file to configure your RAG database connection.");
            println!("Then use: llama-gguf rag init --config {}", output);
        }

        // =====================================================================
        // Knowledge Base Commands
        // =====================================================================
        RagAction::KbCreate {
            name,
            description,
            config,
            chunking,
            max_tokens,
            overlap,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::{ChunkingStrategy, KnowledgeBaseBuilder};

                // Load base config
                let storage = RagConfig::load(config.as_deref())?;

                // Parse chunking strategy
                let chunking_strategy = match chunking.to_lowercase().as_str() {
                    "none" => ChunkingStrategy::None,
                    "fixed" => ChunkingStrategy::FixedSize {
                        max_tokens,
                        overlap_percentage: overlap.min(50),
                    },
                    "semantic" => ChunkingStrategy::Semantic {
                        max_tokens,
                        buffer_size: 100,
                    },
                    "hierarchical" => ChunkingStrategy::Hierarchical {
                        parent_max_tokens: max_tokens * 2,
                        child_max_tokens: max_tokens,
                        child_overlap_percentage: overlap.min(50),
                    },
                    _ => {
                        eprintln!("Unknown chunking strategy: {}. Using 'fixed'.", chunking);
                        ChunkingStrategy::FixedSize {
                            max_tokens,
                            overlap_percentage: overlap.min(50),
                        }
                    }
                };

                let mut builder = KnowledgeBaseBuilder::new(&name)
                    .storage(storage)
                    .chunking(chunking_strategy);

                if let Some(desc) = description {
                    builder = builder.description(desc);
                }

                let kb = builder.create().await?;

                println!("Created knowledge base: {}", kb.name());
                println!("  Chunking: {:?}", kb.config().chunking);
                println!("  Embedding dim: {}", kb.config().storage.embedding_dim());
                println!("\nNext steps:");
                println!("  llama-gguf rag kb-ingest -n {} <path>", name);

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::KbIngest {
            name,
            path,
            config,
            pattern,
            recursive,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::{DataSource, KnowledgeBase, KnowledgeBaseConfig};
                use std::path::Path;

                let storage = RagConfig::load(config.as_deref())?;
                let kb_config = KnowledgeBaseConfig {
                    name: name.clone(),
                    storage,
                    ..Default::default()
                };

                let kb = KnowledgeBase::connect(kb_config).await?;

                let source_path = Path::new(&path);
                let source = if source_path.is_file() {
                    DataSource::File {
                        path: source_path.to_path_buf(),
                    }
                } else if source_path.is_dir() {
                    DataSource::Directory {
                        path: source_path.to_path_buf(),
                        pattern,
                        recursive,
                    }
                } else {
                    return Err(format!("Path not found: {}", path).into());
                };

                println!("Ingesting from: {}", path);
                let result = kb.ingest(source).await?;

                println!("\nIngestion complete:");
                println!("  Documents processed: {}", result.documents_processed);
                println!("  Chunks created: {}", result.chunks_created);
                println!("  Total characters: {}", result.metadata.total_characters);

                if !result.failures.is_empty() {
                    println!("\nFailures:");
                    for (path, err) in &result.failures {
                        println!("  {}: {}", path, err);
                    }
                }

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::KbRetrieve {
            query,
            name,
            config,
            limit,
            min_score,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::{KnowledgeBase, KnowledgeBaseConfig, RetrievalConfig};

                let storage = RagConfig::load(config.as_deref())?;
                let kb_config = KnowledgeBaseConfig {
                    name: name.clone(),
                    storage,
                    ..Default::default()
                };

                let kb = KnowledgeBase::connect(kb_config).await?;

                let retrieval_config = RetrievalConfig {
                    max_results: limit,
                    min_score,
                    ..Default::default()
                };

                println!("Querying knowledge base '{}': \"{}\"", name, query);
                println!();

                let response = kb.retrieve(&query, Some(retrieval_config)).await?;

                if response.chunks.is_empty() {
                    println!("No results found.");
                } else {
                    println!("Found {} results:\n", response.chunks.len());

                    for (i, chunk) in response.chunks.iter().enumerate() {
                        println!(
                            "{}. [score: {:.4}] {}",
                            i + 1,
                            chunk.score,
                            chunk.source.uri
                        );
                        let preview: String = chunk.content.chars().take(200).collect();
                        println!("   {}", preview);
                        if chunk.content.len() > 200 {
                            println!("   ...");
                        }
                        println!();
                    }
                }

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::KbRetrieveAndGenerate {
            query,
            name,
            config,
            limit,
            prompt_template,
            citations,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::{KnowledgeBase, KnowledgeBaseConfig, RetrievalConfig};

                let storage = RagConfig::load(config.as_deref())?;
                let kb_config = KnowledgeBaseConfig {
                    name: name.clone(),
                    storage,
                    ..Default::default()
                };

                let kb = KnowledgeBase::connect(kb_config).await?;

                let retrieval_config = RetrievalConfig {
                    max_results: limit,
                    prompt_template,
                    ..Default::default()
                };

                println!("Retrieve and Generate from '{}': \"{}\"", name, query);
                println!();

                let response = kb
                    .retrieve_and_generate(&query, Some(retrieval_config))
                    .await?;

                println!("=== Generated Prompt ===");
                println!("{}", response.output);
                println!();

                if citations && !response.citations.is_empty() {
                    println!("=== Citations ===");
                    for (i, citation) in response.citations.iter().enumerate() {
                        println!(
                            "{}. {} (score: {:.4})",
                            i + 1,
                            citation.source.uri,
                            citation.score
                        );
                    }
                }

                println!("\n[Note: Pass this prompt to your LLM for generation]");

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::KbStats { name, config } => {
            rt.block_on(async {
                use llama_gguf::rag::{KnowledgeBase, KnowledgeBaseConfig};

                let storage = RagConfig::load(config.as_deref())?;
                let kb_config = KnowledgeBaseConfig {
                    name: name.clone(),
                    storage,
                    ..Default::default()
                };

                let kb = KnowledgeBase::connect(kb_config).await?;
                let stats = kb.stats().await?;

                println!("Knowledge Base: {}", stats.name);
                println!("{}", "-".repeat(40));
                println!("Documents: {}", stats.document_count);
                println!("Embedding dimension: {}", stats.embedding_dimension);
                println!("Chunking strategy: {}", stats.chunking_strategy);
                println!(
                    "Hybrid search: {}",
                    if stats.hybrid_search_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }

        RagAction::KbDelete {
            name,
            config,
            force,
        } => {
            rt.block_on(async {
                use llama_gguf::rag::{KnowledgeBase, KnowledgeBaseConfig};

                let storage = RagConfig::load(config.as_deref())?;
                let kb_config = KnowledgeBaseConfig {
                    name: name.clone(),
                    storage,
                    ..Default::default()
                };

                let kb = KnowledgeBase::connect(kb_config).await?;
                let stats = kb.stats().await?;

                println!(
                    "Knowledge base '{}' contains {} documents.",
                    name, stats.document_count
                );

                if !force {
                    print!("Delete all documents? [y/N] ");
                    use std::io::{self, Write};
                    io::stdout().flush()?;

                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;

                    if !input.trim().eq_ignore_ascii_case("y") {
                        println!("Cancelled.");
                        return Ok(());
                    }
                }

                kb.delete().await?;
                println!("Knowledge base '{}' deleted.", name);

                Ok::<_, Box<dyn std::error::Error>>(())
            })?;
        }
    }

    Ok(())
}
