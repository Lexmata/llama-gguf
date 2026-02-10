//! HTTP server setup and routing

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::routing::{delete, get, post};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::engine::{ChatTemplate, Engine, EngineConfig};
use crate::gguf::GgufFile;
use crate::model::ModelLoader;
use crate::tokenizer::Tokenizer;

use super::handlers::{self, AppState};

/// Server configuration
pub struct ServerConfig {
    /// Host address to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Path to the GGUF model file
    pub model_path: String,
    /// RAG database URL (optional)
    #[cfg(feature = "rag")]
    pub rag_database_url: Option<String>,
}

/// Run the HTTP server
pub async fn run_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model from: {}", config.model_path);

    // Load GGUF file and tokenizer
    let gguf = GgufFile::open(&config.model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    eprintln!("Tokenizer loaded: {} tokens", tokenizer.vocab_size);

    // Detect chat template from GGUF metadata
    let chat_template = ChatTemplate::detect(&gguf);
    eprintln!("Chat template: {:?}", chat_template);

    // Load model
    let loader = ModelLoader::load(&config.model_path)?;
    let model_config = loader.config().clone();
    eprintln!(
        "Model config: {} layers, {} heads, {} dim",
        model_config.num_layers, model_config.num_heads, model_config.hidden_size
    );

    let model = loader.build_model()?;
    eprintln!("Model loaded successfully");

    // Select backend (GPU if LLAMA_GPU=1, otherwise CPU)
    let use_gpu = std::env::var("LLAMA_GPU")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    let backend: Arc<dyn crate::Backend> = if use_gpu {
        Engine::select_gpu_backend(&model)
    } else {
        Arc::new(crate::backend::cpu::CpuBackend::new())
    };

    // Extract model name from path
    let model_name = std::path::Path::new(&config.model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llama")
        .to_string();

    // Create shared state
    let app_state = Arc::new(AppState {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
        config: model_config,
        model_name,
        chat_template,
        backend,
        inference_lock: Mutex::new(()),
    });

    // Setup CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build base router with OpenAI-compatible endpoints
    let mut app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/models", get(handlers::list_models))
        // Health and status
        .route("/health", get(handlers::health))
        .route("/", get(|| async { "llama-gguf server" }))
        .with_state(app_state.clone());

    // Add RAG endpoints if configured
    #[cfg(feature = "rag")]
    let rag_enabled = config.rag_database_url.is_some();
    #[cfg(not(feature = "rag"))]
    let rag_enabled = false;

    #[cfg(feature = "rag")]
    if let Some(ref db_url) = config.rag_database_url {
        use super::handlers::RagState;
        use crate::rag::RagConfig;

        eprintln!("RAG enabled with database connection");

        let rag_config = RagConfig::new(db_url);
        let rag_state = Arc::new(RagState::new(rag_config));

        // RAG-only routes
        let rag_routes = Router::new()
            // Bedrock-style Knowledge Base APIs
            .route("/knowledgebases", post(handlers::list_knowledge_bases))
            .route("/knowledgebases/:kb_id", get(handlers::get_knowledge_base))
            .route(
                "/knowledgebases/:kb_id",
                delete(handlers::delete_knowledge_base),
            )
            // Retrieval APIs
            .route("/retrieve", post(handlers::retrieve))
            .route("/ingest", post(handlers::ingest))
            .with_state(rag_state.clone());

        // RAG + Model routes (need both states)
        let rag_gen_routes = Router::new()
            .route(
                "/retrieveAndGenerate",
                post(handlers::retrieve_and_generate),
            )
            .with_state((app_state.clone(), rag_state));

        app = app
            .nest("/v1/rag", rag_routes)
            .nest("/v1/rag", rag_gen_routes);
    }

    app = app.layer(cors);

    let addr = format!("{}:{}", config.host, config.port);
    let socket_addr: SocketAddr = addr.parse()?;

    eprintln!();
    eprintln!("╭─────────────────────────────────────────────────────────────────╮");
    eprintln!("│                     llama-gguf Server                            │");
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Listening on: http://{:<44} │", addr);
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Endpoints:                                                       │");
    eprintln!("│   POST /v1/chat/completions  - Chat completions (OpenAI API)     │");
    eprintln!("│   POST /v1/completions       - Text completions (OpenAI API)     │");
    eprintln!("│   GET  /v1/models            - List models                       │");
    eprintln!("│   GET  /health               - Health check                      │");
    if rag_enabled {
        eprintln!("├─────────────────────────────────────────────────────────────────┤");
        eprintln!("│ RAG / Knowledge Base Endpoints (Bedrock-style):                 │");
        eprintln!("│   POST /v1/rag/retrieve            - Retrieve from KB           │");
        eprintln!("│   POST /v1/rag/retrieveAndGenerate - RAG pipeline               │");
        eprintln!("│   POST /v1/rag/ingest              - Ingest documents           │");
        eprintln!("│   POST /v1/rag/knowledgebases      - List knowledge bases       │");
        eprintln!("│   GET  /v1/rag/knowledgebases/:id  - Get KB details             │");
        eprintln!("│   DEL  /v1/rag/knowledgebases/:id  - Delete KB                  │");
    }
    eprintln!("╰─────────────────────────────────────────────────────────────────╯");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(socket_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
