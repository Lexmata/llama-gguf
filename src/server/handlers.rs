//! HTTP request handlers

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream};
use tokio::sync::Mutex;

use crate::engine::ChatTemplate;
use crate::model::{InferenceContext, ModelConfig};
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::{Backend, Model};

use super::types::*;

/// Shared application state
pub struct AppState {
    pub model: Arc<dyn Model>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: ModelConfig,
    pub model_name: String,
    /// Detected chat template for the loaded model
    pub chat_template: ChatTemplate,
    /// Backend for inference (CPU or GPU)
    pub backend: Arc<dyn Backend>,
    /// Mutex to serialize inference requests (single-threaded for now)
    pub inference_lock: Mutex<()>,
}

/// Health check endpoint
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
        context_size: state.config.max_seq_len,
    })
}

/// List models endpoint
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created,
            owned_by: "llama-rs".to_string(),
        }],
    })
}

/// Chat completions endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Acquire inference lock
    let _lock = state.inference_lock.lock().await;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let request_id = format!("chatcmpl-{}", created);

    // Format messages into prompt using the model's chat template
    let prompt = format_chat_messages(&request.messages, &state.chat_template);

    // Create sampler
    let sampler_config = SamplerConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        ..Default::default()
    };

    // Generate response
    match generate_response(
        &state,
        &prompt,
        request.max_tokens,
        sampler_config,
        request.stop.as_deref(),
    )
    .await
    {
        Ok((response_text, prompt_tokens, completion_tokens)) => {
            if request.stream {
                // Streaming response
                let stream = create_chat_stream(
                    request_id,
                    state.model_name.clone(),
                    created,
                    response_text,
                );
                Sse::new(stream).into_response()
            } else {
                // Non-streaming response
                let response = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".to_string(),
                    created,
                    model: state.model_name.clone(),
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: Role::Assistant,
                            content: response_text,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };
                Json(response).into_response()
            }
        }
        Err(e) => {
            let error = ErrorResponse::new(e.to_string(), "server_error");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Text completions endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    let _lock = state.inference_lock.lock().await;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let request_id = format!("cmpl-{}", created);

    let sampler_config = SamplerConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        ..Default::default()
    };

    match generate_response(
        &state,
        &request.prompt,
        request.max_tokens,
        sampler_config,
        request.stop.as_deref(),
    )
    .await
    {
        Ok((response_text, prompt_tokens, completion_tokens)) => {
            let response = CompletionResponse {
                id: request_id,
                object: "text_completion".to_string(),
                created,
                model: state.model_name.clone(),
                choices: vec![CompletionChoice {
                    text: response_text,
                    index: 0,
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(response).into_response()
        }
        Err(e) => {
            let error = ErrorResponse::new(e.to_string(), "server_error");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Format chat messages into a prompt string using the detected chat template
fn format_chat_messages(messages: &[ChatMessage], template: &ChatTemplate) -> String {
    // Extract system prompt and conversation messages
    let mut system_prompt = String::new();
    let mut conversation: Vec<&ChatMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => system_prompt = msg.content.clone(),
            _ => conversation.push(msg),
        }
    }

    let mut prompt = String::new();
    let mut is_first_user = true;

    for msg in &conversation {
        match msg.role {
            Role::User => {
                if is_first_user && !system_prompt.is_empty() {
                    prompt.push_str(&template.format_first_turn(&system_prompt, &msg.content));
                    is_first_user = false;
                } else {
                    prompt.push_str(&template.format_continuation(&msg.content));
                    is_first_user = false;
                }
            }
            Role::Assistant => {
                // Append assistant response as-is (template handles framing)
                prompt.push_str(&msg.content);
            }
            Role::System => {} // Already handled
        }
    }

    // If there were no user messages but we have a system prompt, wrap it
    if is_first_user && !system_prompt.is_empty() {
        prompt.push_str(&template.format_first_turn(&system_prompt, ""));
    }

    prompt
}

/// Generate text response
async fn generate_response(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    sampler_config: SamplerConfig,
    _stop_sequences: Option<&[String]>,
) -> Result<(String, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    // Create a new context for this request using the shared backend
    let mut ctx = InferenceContext::new(&state.config, state.backend.clone());
    let mut sampler = Sampler::new(sampler_config, state.config.vocab_size);

    // Encode prompt
    let prompt_tokens = state.tokenizer.encode(prompt, true)?;
    let prompt_len = prompt_tokens.len();

    let mut all_tokens = prompt_tokens.clone();

    // Process prompt tokens (prefill) — process all but the last token
    // The last token's logits are what we sample from for the first generated token
    if prompt_tokens.len() > 1 {
        for (i, &token) in prompt_tokens[..prompt_tokens.len() - 1].iter().enumerate() {
            if i < state.config.max_seq_len {
                let _ = state.model.forward(&[token], &mut ctx);
            }
        }
    }

    // Get stop patterns from the chat template
    let stop_patterns = state.chat_template.stop_patterns();

    // Generate response tokens
    let mut response_text = String::new();
    let mut completion_tokens = 0;

    for _ in 0..max_tokens {
        let last_token = *all_tokens
            .last()
            .unwrap_or(&state.tokenizer.special_tokens.bos_token_id);

        // Forward pass
        let logits = state.model.forward(&[last_token], &mut ctx)?;

        // Sample next token
        let next_token = sampler.sample(&logits, &all_tokens);

        // Check for EOS
        if next_token == state.tokenizer.special_tokens.eos_token_id {
            break;
        }

        // Decode token
        if let Ok(text) = state.tokenizer.decode(&[next_token]) {
            // Check for stop patterns using the chat template
            let combined = format!("{}{}", response_text, text);
            let should_stop = stop_patterns.iter().any(|p| combined.contains(p));

            if should_stop {
                // Trim stop pattern from response
                for pattern in stop_patterns {
                    if let Some(idx) = combined.find(pattern) {
                        response_text = combined[..idx].to_string();
                        return Ok((
                            response_text.trim().to_string(),
                            prompt_len,
                            completion_tokens,
                        ));
                    }
                }
                break;
            }
            response_text.push_str(&text);
        }

        all_tokens.push(next_token);
        completion_tokens += 1;
    }

    Ok((
        response_text.trim().to_string(),
        prompt_len,
        completion_tokens,
    ))
}

/// Create streaming response for chat completions
fn create_chat_stream(
    request_id: String,
    model: String,
    created: u64,
    response_text: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    // For simplicity, we send the whole response as a single chunk
    // A proper implementation would stream token by token
    let chunks = vec![
        // Initial chunk with role
        ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some(Role::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        },
        // Content chunk
        ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: None,
                    content: Some(response_text),
                },
                finish_reason: None,
            }],
        },
        // Final chunk
        ChatCompletionChunk {
            id: request_id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        },
    ];

    stream::iter(chunks.into_iter().map(|chunk| {
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    }))
}

// =============================================================================
// RAG / Knowledge Base Handlers
// =============================================================================

/// RAG state for knowledge base operations
#[cfg(feature = "rag")]
pub struct RagState {
    /// Knowledge base configurations (name -> config)
    pub knowledge_bases: tokio::sync::RwLock<HashMap<String, crate::rag::KnowledgeBaseConfig>>,
    /// RAG config for database connection
    pub rag_config: crate::rag::RagConfig,
}

#[cfg(feature = "rag")]
impl RagState {
    pub fn new(rag_config: crate::rag::RagConfig) -> Self {
        Self {
            knowledge_bases: tokio::sync::RwLock::new(HashMap::new()),
            rag_config,
        }
    }
}

/// Retrieve from knowledge base (Bedrock-style API)
#[cfg(feature = "rag")]
pub async fn retrieve(
    State(rag_state): State<Arc<RagState>>,
    Json(request): Json<RetrieveRequest>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig, RetrievalConfig};

    // Get or create knowledge base config
    let kb_config = {
        let kbs = rag_state.knowledge_bases.read().await;
        kbs.get(&request.knowledge_base_id)
            .cloned()
            .unwrap_or_else(|| KnowledgeBaseConfig {
                name: request.knowledge_base_id.clone(),
                storage: rag_state.rag_config.clone(),
                ..Default::default()
            })
    };

    // Connect to knowledge base
    let kb = match KnowledgeBase::connect(kb_config).await {
        Ok(kb) => kb,
        Err(e) => {
            let error = ErrorResponse::new(
                format!("Failed to connect to knowledge base: {}", e),
                "knowledge_base_error",
            );
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
        }
    };

    // Build retrieval config
    let mut retrieval_config = RetrievalConfig::default();

    if let Some(ref config) = request.retrieval_configuration
        && let Some(ref vs_config) = config.vector_search_configuration
    {
        retrieval_config.max_results = vs_config.number_of_results;

        // Convert filter if provided
        if let Some(ref filter) = vs_config.filter {
            retrieval_config.filter = convert_filter(filter);
        }
    }

    // Perform retrieval
    match kb.retrieve(&request.query, Some(retrieval_config)).await {
        Ok(response) => {
            let results: Vec<RetrievalResult> = response
                .chunks
                .into_iter()
                .map(|chunk| RetrievalResult {
                    content: RetrievalResultContent {
                        text: chunk.content,
                    },
                    location: RetrievalResultLocation {
                        location_type: "CUSTOM".to_string(),
                        s3_location: None,
                        custom_location: Some(CustomLocation {
                            uri: chunk.source.uri,
                        }),
                    },
                    score: chunk.score,
                    metadata: chunk.metadata,
                })
                .collect();

            Json(RetrieveResponse {
                retrieval_results: results,
                next_token: None,
            })
            .into_response()
        }
        Err(e) => {
            let error = ErrorResponse::new(format!("Retrieval failed: {}", e), "retrieval_error");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Retrieve and generate (RAG pipeline)
#[cfg(feature = "rag")]
pub async fn retrieve_and_generate(
    State((app_state, rag_state)): State<(Arc<AppState>, Arc<RagState>)>,
    Json(request): Json<RetrieveAndGenerateRequest>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig, RetrievalConfig};

    let kb_id = &request
        .retrieve_and_generate_configuration
        .knowledge_base_configuration
        .knowledge_base_id;

    // Get or create knowledge base config
    let kb_config = {
        let kbs = rag_state.knowledge_bases.read().await;
        kbs.get(kb_id)
            .cloned()
            .unwrap_or_else(|| KnowledgeBaseConfig {
                name: kb_id.clone(),
                storage: rag_state.rag_config.clone(),
                ..Default::default()
            })
    };

    // Connect to knowledge base
    let kb = match KnowledgeBase::connect(kb_config).await {
        Ok(kb) => kb,
        Err(e) => {
            let error = ErrorResponse::new(
                format!("Failed to connect to knowledge base: {}", e),
                "knowledge_base_error",
            );
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
        }
    };

    // Build retrieval config
    let mut retrieval_config = RetrievalConfig::default();

    if let Some(ref config) = request
        .retrieve_and_generate_configuration
        .knowledge_base_configuration
        .retrieval_configuration
        && let Some(ref vs_config) = config.vector_search_configuration
    {
        retrieval_config.max_results = vs_config.number_of_results;
    }

    // Get prompt template if provided
    if let Some(ref gen_config) = request
        .retrieve_and_generate_configuration
        .knowledge_base_configuration
        .generation_configuration
        && let Some(ref template) = gen_config.prompt_template
    {
        // Convert Bedrock template format ($query$, $search_results$) to our format ({query}, {context})
        let converted = template
            .text_prompt_template
            .replace("$query$", "{query}")
            .replace("$search_results$", "{context}");
        retrieval_config.prompt_template = Some(converted);
    }

    // Perform retrieval
    let rag_response = match kb
        .retrieve_and_generate(&request.input.text, Some(retrieval_config))
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            let error = ErrorResponse::new(format!("RAG failed: {}", e), "rag_error");
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
        }
    };

    // Get inference config
    let (temperature, top_p, max_tokens) = if let Some(ref gen_config) = request
        .retrieve_and_generate_configuration
        .knowledge_base_configuration
        .generation_configuration
    {
        if let Some(ref inf_config) = gen_config.inference_config {
            if let Some(ref text_config) = inf_config.text_inference_config {
                (
                    text_config.temperature,
                    text_config.top_p,
                    text_config.max_tokens,
                )
            } else {
                (0.7, 0.9, 256)
            }
        } else {
            (0.7, 0.9, 256)
        }
    } else {
        (0.7, 0.9, 256)
    };

    // Generate response using the model
    let _lock = app_state.inference_lock.lock().await;

    let sampler_config = SamplerConfig {
        temperature,
        top_p,
        ..Default::default()
    };

    let generated_text = match generate_response(
        &app_state,
        &rag_response.output,
        max_tokens,
        sampler_config,
        None,
    )
    .await
    {
        Ok((text, _, _)) => text,
        Err(e) => {
            let error = ErrorResponse::new(format!("Generation failed: {}", e), "generation_error");
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
        }
    };

    // Build citations
    let citations: Vec<Citation> = rag_response
        .citations
        .into_iter()
        .map(|c| Citation {
            generated_response_part: None,
            retrieved_references: vec![RetrievedReference {
                content: RetrievalResultContent { text: c.content },
                location: RetrievalResultLocation {
                    location_type: "CUSTOM".to_string(),
                    s3_location: None,
                    custom_location: Some(CustomLocation { uri: c.source.uri }),
                },
                metadata: None,
            }],
        })
        .collect();

    Json(RetrieveAndGenerateResponse {
        output: RetrieveAndGenerateOutput {
            text: generated_text,
        },
        citations,
        session_id: request.session_id,
    })
    .into_response()
}

/// Ingest documents into knowledge base
#[cfg(feature = "rag")]
pub async fn ingest(
    State(rag_state): State<Arc<RagState>>,
    Json(request): Json<IngestRequest>,
) -> Response {
    use crate::rag::{DataSource, KnowledgeBase, KnowledgeBaseConfig};

    // Get or create knowledge base config
    let kb_config = {
        let kbs = rag_state.knowledge_bases.read().await;
        kbs.get(&request.knowledge_base_id)
            .cloned()
            .unwrap_or_else(|| KnowledgeBaseConfig {
                name: request.knowledge_base_id.clone(),
                storage: rag_state.rag_config.clone(),
                ..Default::default()
            })
    };

    // Connect to knowledge base
    let kb = match KnowledgeBase::connect(kb_config).await {
        Ok(kb) => kb,
        Err(e) => {
            let error = ErrorResponse::new(
                format!("Failed to connect to knowledge base: {}", e),
                "knowledge_base_error",
            );
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
        }
    };

    let mut total_docs = 0;
    let mut total_chunks = 0;
    let mut failures = Vec::new();

    for doc in request.documents {
        let source = DataSource::Text {
            content: doc.content.text,
            source_id: doc.document_id.clone(),
            metadata: doc.metadata,
        };

        match kb.ingest(source).await {
            Ok(result) => {
                total_docs += result.documents_processed;
                total_chunks += result.chunks_created;
                for (id, err) in result.failures {
                    failures.push(IngestFailure {
                        document_id: id,
                        error_message: err,
                    });
                }
            }
            Err(e) => {
                failures.push(IngestFailure {
                    document_id: doc.document_id,
                    error_message: e.to_string(),
                });
            }
        }
    }

    Json(IngestResponse {
        documents_ingested: total_docs,
        chunks_created: total_chunks,
        failures,
    })
    .into_response()
}

/// List knowledge bases
#[cfg(feature = "rag")]
pub async fn list_knowledge_bases(
    State(rag_state): State<Arc<RagState>>,
    Json(_request): Json<ListKnowledgeBasesRequest>,
) -> Response {
    let kbs = rag_state.knowledge_bases.read().await;

    let summaries: Vec<KnowledgeBaseSummary> = kbs
        .iter()
        .map(|(id, config)| KnowledgeBaseSummary {
            knowledge_base_id: id.clone(),
            name: config.name.clone(),
            description: config.description.clone(),
            status: "ACTIVE".to_string(),
            updated_at: current_timestamp(),
        })
        .collect();

    Json(ListKnowledgeBasesResponse {
        knowledge_base_summaries: summaries,
        next_token: None,
    })
    .into_response()
}

/// Get knowledge base details
#[cfg(feature = "rag")]
pub async fn get_knowledge_base(
    State(rag_state): State<Arc<RagState>>,
    Path(kb_id): Path<String>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig};

    // Get or create knowledge base config
    let kb_config = {
        let kbs = rag_state.knowledge_bases.read().await;
        kbs.get(&kb_id)
            .cloned()
            .unwrap_or_else(|| KnowledgeBaseConfig {
                name: kb_id.clone(),
                storage: rag_state.rag_config.clone(),
                ..Default::default()
            })
    };

    // Try to connect to get stats
    match KnowledgeBase::connect(kb_config.clone()).await {
        Ok(kb) => match kb.stats().await {
            Ok(stats) => Json(GetKnowledgeBaseResponse {
                knowledge_base: KnowledgeBaseDetail {
                    knowledge_base_id: kb_id,
                    name: stats.name,
                    description: kb_config.description,
                    status: "ACTIVE".to_string(),
                    storage_configuration: StorageConfigurationResponse {
                        storage_type: "PGVECTOR".to_string(),
                        vector_dimension: stats.embedding_dimension,
                    },
                    updated_at: current_timestamp(),
                },
            })
            .into_response(),
            Err(e) => {
                let error = ErrorResponse::new(
                    format!("Failed to get stats: {}", e),
                    "knowledge_base_error",
                );
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
            }
        },
        Err(e) => {
            let error = ErrorResponse::new(format!("Knowledge base not found: {}", e), "not_found");
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    }
}

/// Delete knowledge base
#[cfg(feature = "rag")]
pub async fn delete_knowledge_base(
    State(rag_state): State<Arc<RagState>>,
    Path(kb_id): Path<String>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig};

    // Get knowledge base config
    let kb_config = {
        let mut kbs = rag_state.knowledge_bases.write().await;
        kbs.remove(&kb_id).unwrap_or_else(|| KnowledgeBaseConfig {
            name: kb_id.clone(),
            storage: rag_state.rag_config.clone(),
            ..Default::default()
        })
    };

    // Connect and delete
    match KnowledgeBase::connect(kb_config).await {
        Ok(kb) => match kb.delete().await {
            Ok(_) => Json(serde_json::json!({
                "knowledgeBaseId": kb_id,
                "status": "DELETING"
            }))
            .into_response(),
            Err(e) => {
                let error = ErrorResponse::new(format!("Failed to delete: {}", e), "delete_error");
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
            }
        },
        Err(e) => {
            let error = ErrorResponse::new(format!("Knowledge base not found: {}", e), "not_found");
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    }
}

/// Convert Bedrock filter to MetadataFilter
#[cfg(feature = "rag")]
fn convert_filter(filter: &RetrievalFilter) -> Option<crate::rag::MetadataFilter> {
    use crate::rag::MetadataFilter;

    // Handle AND
    if let Some(ref and_filters) = filter.and_all {
        let converted: Vec<_> = and_filters.iter().filter_map(convert_filter).collect();
        if !converted.is_empty() {
            return Some(MetadataFilter::and(converted));
        }
    }

    // Handle OR
    if let Some(ref or_filters) = filter.or_all {
        let converted: Vec<_> = or_filters.iter().filter_map(convert_filter).collect();
        if !converted.is_empty() {
            return Some(MetadataFilter::or(converted));
        }
    }

    // Handle equals
    if let Some(ref cond) = filter.equals {
        return Some(MetadataFilter::eq(&cond.key, cond.value.clone()));
    }

    // Handle not equals
    if let Some(ref cond) = filter.not_equals {
        return Some(MetadataFilter::ne(&cond.key, cond.value.clone()));
    }

    // Handle greater than
    if let Some(ref cond) = filter.greater_than {
        return Some(MetadataFilter::gt(&cond.key, cond.value.clone()));
    }

    // Handle less than
    if let Some(ref cond) = filter.less_than {
        return Some(MetadataFilter::lt(&cond.key, cond.value.clone()));
    }

    // Handle string contains
    if let Some(ref cond) = filter.string_contains
        && let Some(s) = cond.value.as_str()
    {
        return Some(MetadataFilter::contains(&cond.key, s));
    }

    // Handle starts with
    if let Some(ref cond) = filter.starts_with
        && let Some(s) = cond.value.as_str()
    {
        return Some(MetadataFilter::starts_with(&cond.key, s));
    }

    None
}

/// Get current timestamp as ISO string
#[cfg(feature = "rag")]
fn current_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}Z", now)
}
