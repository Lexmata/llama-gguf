//! HTTP server setup and routing

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tokio::sync::{RwLock, Semaphore};
use tower_http::cors::{Any, CorsLayer};

use crate::engine::{ChatTemplate, Engine, EngineConfig};
use crate::gguf::GgufFile;
use crate::model::ModelLoader;
use crate::tokenizer::Tokenizer;

use super::handlers::{self, AppState, RequestQueue};

/// Server configuration
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: String,
    /// Maximum concurrent inference requests (default: 1)
    pub max_concurrent: usize,
    /// Maximum queued requests before rejecting with 429 (default: 64)
    pub max_queue_depth: usize,
    #[cfg(feature = "rag")]
    pub rag_database_url: Option<String>,
}

/// Run the HTTP server
pub async fn run_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model from: {}", config.model_path);

    let gguf = GgufFile::open(&config.model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    eprintln!("Tokenizer loaded: {} tokens", tokenizer.vocab_size);

    let chat_template = ChatTemplate::detect(&gguf);
    eprintln!("Chat template: {:?}", chat_template);

    let loader = ModelLoader::load(&config.model_path)?;
    let model_config = loader.config().clone();
    eprintln!(
        "Model config: {} layers, {} heads, {} dim",
        model_config.num_layers, model_config.num_heads, model_config.hidden_size
    );

    let model = loader.build_model()?;
    eprintln!("Model loaded successfully");

    let use_gpu = std::env::var("LLAMA_GPU")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    let backend: Arc<dyn crate::Backend> = if use_gpu {
        Engine::select_gpu_backend(&model)
    } else {
        Arc::new(crate::backend::cpu::CpuBackend::new())
    };

    let model_name = std::path::Path::new(&config.model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llama")
        .to_string();

    let max_concurrent = if config.max_concurrent == 0 {
        1
    } else {
        config.max_concurrent
    };
    let max_queue_depth = if config.max_queue_depth == 0 {
        64
    } else {
        config.max_queue_depth
    };

    let app_state = Arc::new(AppState {
        model: RwLock::new(Arc::new(model)),
        tokenizer: RwLock::new(Arc::new(tokenizer)),
        config: RwLock::new(model_config),
        model_name: RwLock::new(model_name),
        model_path: RwLock::new(config.model_path.clone()),
        chat_template: RwLock::new(chat_template),
        backend: RwLock::new(backend),
        inference_semaphore: Arc::new(Semaphore::new(max_concurrent)),
        request_queue: RequestQueue::new(max_queue_depth, max_concurrent),
    });

    // Spawn SIGHUP handler for model hot-reload (Unix only)
    #[cfg(unix)]
    {
        let state_for_signal = app_state.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};

            let mut stream = match signal(SignalKind::hangup()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Failed to register SIGHUP handler: {}", e);
                    return;
                }
            };

            loop {
                stream.recv().await;
                tracing::info!("SIGHUP received: reloading model...");
                let path = state_for_signal.model_path.read().await.clone();
                match handlers::reload_model_from_path(&state_for_signal, &path).await {
                    Ok((name, ctx)) => {
                        tracing::info!("Model reloaded via SIGHUP: {} (ctx={})", name, ctx);
                    }
                    Err(e) => {
                        tracing::error!("Model reload via SIGHUP failed: {}", e);
                    }
                }
            }
        });
    }

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let mut app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/models", get(handlers::list_models))
        // Server management
        .route("/v1/models/load", post(handlers::load_model))
        .route("/v1/queue/status", get(handlers::queue_status))
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

        let rag_routes = Router::new()
            .route("/knowledgebases", post(handlers::list_knowledge_bases))
            .route("/knowledgebases/:kb_id", get(handlers::get_knowledge_base))
            .route(
                "/knowledgebases/:kb_id",
                axum::routing::delete(handlers::delete_knowledge_base),
            )
            .route("/retrieve", post(handlers::retrieve))
            .route("/ingest", post(handlers::ingest))
            .with_state(rag_state.clone());

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
    eprintln!("╭────────────────────────────────────────────────────────────────────╮");
    eprintln!("│                        llama-gguf Server                           │");
    eprintln!("├────────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Listening on: http://{:<48}│", addr);
    eprintln!("│ Concurrency:  {} concurrent, {} max queued{:<27}│", max_concurrent, max_queue_depth, "");
    eprintln!("├────────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Endpoints:                                                         │");
    eprintln!("│   POST /v1/chat/completions  - Chat completions (OpenAI API)       │");
    eprintln!("│   POST /v1/completions       - Text completions (OpenAI API)       │");
    eprintln!("│   POST /v1/embeddings        - Embeddings (OpenAI API)             │");
    eprintln!("│   GET  /v1/models            - List models                         │");
    eprintln!("│   POST /v1/models/load       - Hot-swap model                      │");
    eprintln!("│   GET  /v1/queue/status      - Queue status                        │");
    eprintln!("│   GET  /health               - Health check                        │");
    if rag_enabled {
        eprintln!("├────────────────────────────────────────────────────────────────────┤");
        eprintln!("│ RAG / Knowledge Base Endpoints (Bedrock-style):                    │");
        eprintln!("│   POST /v1/rag/retrieve            - Retrieve from KB              │");
        eprintln!("│   POST /v1/rag/retrieveAndGenerate - RAG pipeline                  │");
        eprintln!("│   POST /v1/rag/ingest              - Ingest documents              │");
        eprintln!("│   POST /v1/rag/knowledgebases      - List knowledge bases          │");
        eprintln!("│   GET  /v1/rag/knowledgebases/:id  - Get KB details                │");
        eprintln!("│   DEL  /v1/rag/knowledgebases/:id  - Delete KB                     │");
    }
    #[cfg(unix)]
    eprintln!("├────────────────────────────────────────────────────────────────────┤");
    #[cfg(unix)]
    eprintln!("│ Send SIGHUP to reload model from the same path                    │");
    eprintln!("╰────────────────────────────────────────────────────────────────────╯");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(socket_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
