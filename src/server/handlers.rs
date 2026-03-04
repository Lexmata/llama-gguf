//! HTTP request handlers

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream};
use tokio::sync::{Mutex, RwLock, Semaphore};

#[cfg(feature = "rag")]
use std::collections::HashMap;
#[cfg(feature = "rag")]
use axum::extract::Path;

use crate::engine::ChatTemplate;
use crate::model::ModelConfig;
use crate::model::embeddings::{EmbeddingConfig, EmbeddingExtractor};
use crate::sampling::{Sampler, SamplerConfig};
use crate::sampling::grammar::{Grammar, GrammarSampler};
use crate::tokenizer::Tokenizer;
use crate::{Backend, Model};

use super::types::*;

// =============================================================================
// Application State
// =============================================================================

/// Shared application state
pub struct AppState {
    /// Model behind RwLock for hot-swapping
    pub model: RwLock<Arc<dyn Model>>,
    pub tokenizer: RwLock<Arc<Tokenizer>>,
    pub config: RwLock<ModelConfig>,
    pub model_name: RwLock<String>,
    pub model_path: RwLock<String>,
    pub chat_template: RwLock<ChatTemplate>,
    pub backend: RwLock<Arc<dyn Backend>>,
    /// Semaphore for concurrency control (replaces inference_lock)
    pub inference_semaphore: Arc<Semaphore>,
    /// Request queue
    pub request_queue: RequestQueue,
}

/// FIFO request queue with configurable depth
pub struct RequestQueue {
    pub max_queue_depth: usize,
    pub max_concurrent: usize,
    active: Mutex<usize>,
    queue_depth: Mutex<usize>,
}

impl RequestQueue {
    pub fn new(max_queue_depth: usize, max_concurrent: usize) -> Self {
        Self {
            max_queue_depth,
            max_concurrent,
            active: Mutex::new(0),
            queue_depth: Mutex::new(0),
        }
    }

    /// Try to enqueue a request. Returns Err if queue is full.
    pub async fn try_enqueue(&self) -> Result<QueueGuard<'_>, ()> {
        let mut depth = self.queue_depth.lock().await;
        if *depth >= self.max_queue_depth {
            return Err(());
        }
        *depth += 1;
        Ok(QueueGuard {
            queue_depth: &self.queue_depth,
            active: &self.active,
            promoted: false,
        })
    }

    pub async fn active_count(&self) -> usize {
        *self.active.lock().await
    }

    pub async fn queued_count(&self) -> usize {
        *self.queue_depth.lock().await
    }
}

pub struct QueueGuard<'a> {
    queue_depth: &'a Mutex<usize>,
    active: &'a Mutex<usize>,
    promoted: bool,
}

impl<'a> QueueGuard<'a> {
    pub async fn promote(&mut self) {
        let mut active = self.active.lock().await;
        *active += 1;
        self.promoted = true;
    }
}

impl<'a> Drop for QueueGuard<'a> {
    fn drop(&mut self) {
        let queue_depth = self.queue_depth;
        let active = self.active;
        let promoted = self.promoted;
        // We need to use try_lock since Drop can't be async
        if let Ok(mut depth) = queue_depth.try_lock() {
            if *depth > 0 {
                *depth -= 1;
            }
        }
        if promoted {
            if let Ok(mut act) = active.try_lock() {
                if *act > 0 {
                    *act -= 1;
                }
            }
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Acquire a queue slot, returning 429 if full, then acquire the inference semaphore.
async fn acquire_inference_slot(
    state: &AppState,
) -> Result<(tokio::sync::OwnedSemaphorePermit, QueueGuard<'_>), Response> {
    let mut guard = state.request_queue.try_enqueue().await.map_err(|_| {
        let error = ErrorResponse::new(
            "Server overloaded: request queue is full",
            "rate_limit_exceeded",
        );
        (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response()
    })?;

    // Wait for a semaphore permit (FIFO by tokio Semaphore fairness)
    let permit = state
        .inference_semaphore
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            let error = ErrorResponse::new("Server shutting down", "server_error");
            (StatusCode::SERVICE_UNAVAILABLE, Json(error)).into_response()
        })?;

    guard.promote().await;
    Ok((permit, guard))
}

// =============================================================================
// Health & Models
// =============================================================================

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let config = state.config.read().await;
    let model_name = state.model_name.read().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        model: model_name.clone(),
        context_size: config.max_seq_len,
    })
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let model_name = state.model_name.read().await;
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: model_name.clone(),
            object: "model".to_string(),
            created: now_secs(),
            owned_by: "llama-gguf".to_string(),
        }],
    })
}

// =============================================================================
// Queue Status
// =============================================================================

pub async fn queue_status(State(state): State<Arc<AppState>>) -> Json<QueueStatusResponse> {
    Json(QueueStatusResponse {
        active_requests: state.request_queue.active_count().await,
        queued_requests: state.request_queue.queued_count().await,
        max_queue_depth: state.request_queue.max_queue_depth,
        max_concurrent: state.request_queue.max_concurrent,
    })
}

// =============================================================================
// Embeddings
// =============================================================================

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Response {
    let (_permit, _guard) = match acquire_inference_slot(&state).await {
        Ok(v) => v,
        Err(r) => return r,
    };

    let texts = match request.input {
        EmbeddingInput::Single(ref s) => vec![s.as_str()],
        EmbeddingInput::Batch(ref v) => v.iter().map(|s| s.as_str()).collect(),
    };

    let model = state.model.read().await;
    let tokenizer = state.tokenizer.read().await;
    let config = state.config.read().await;
    let backend = state.backend.read().await;
    let model_name = state.model_name.read().await;

    let embed_config = EmbeddingConfig::default();
    let extractor = EmbeddingExtractor::new(embed_config, &config);

    let mut results = Vec::with_capacity(texts.len());
    let mut total_prompt_tokens = 0usize;

    for (i, text) in texts.iter().enumerate() {
        let tokens = match tokenizer.encode(text, true) {
            Ok(t) => t,
            Err(e) => {
                let error = ErrorResponse::new(
                    format!("Tokenization failed: {}", e),
                    "invalid_request_error",
                );
                return (StatusCode::BAD_REQUEST, Json(error)).into_response();
            }
        };
        total_prompt_tokens += tokens.len();

        let mut ctx = model.create_context(backend.clone());
        match extractor.embed_text(model.as_ref(), &tokenizer, &mut ctx, text) {
            Ok(embedding) => {
                results.push(EmbeddingData {
                    object: "embedding".to_string(),
                    embedding,
                    index: i,
                });
            }
            Err(e) => {
                let error =
                    ErrorResponse::new(format!("Embedding failed: {}", e), "server_error");
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
            }
        }
    }

    Json(EmbeddingResponse {
        object: "list".to_string(),
        data: results,
        model: model_name.clone(),
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    })
    .into_response()
}

// =============================================================================
// Chat Completions (with function calling)
// =============================================================================

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let (_permit, _guard) = match acquire_inference_slot(&state).await {
        Ok(v) => v,
        Err(r) => return r,
    };

    let created = now_secs();
    let request_id = format!("chatcmpl-{}", created);

    let chat_template = state.chat_template.read().await;
    let model_name = state.model_name.read().await;

    let prompt = format_chat_messages(&request.messages, &chat_template, request.tools.as_deref());

    let sampler_config = SamplerConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        ..Default::default()
    };

    let has_tools = request.tools.is_some();
    let forced_function = match &request.tool_choice {
        Some(ToolChoice::Specific { function, .. }) => Some(function.name.clone()),
        _ => None,
    };

    match generate_response(
        &state,
        &prompt,
        request.max_tokens,
        sampler_config,
        request.stop.as_deref(),
        has_tools,
    )
    .await
    {
        Ok((response_text, prompt_tokens, completion_tokens)) => {
            let (message, finish_reason) =
                if has_tools {
                    match parse_tool_calls(&response_text, forced_function.as_deref()) {
                        Some(tool_calls) => (
                            ChatMessage {
                                role: Role::Assistant,
                                content: String::new(),
                                tool_calls: Some(tool_calls),
                                tool_call_id: None,
                            },
                            "tool_calls".to_string(),
                        ),
                        None => (
                            ChatMessage {
                                role: Role::Assistant,
                                content: response_text.clone(),
                                tool_calls: None,
                                tool_call_id: None,
                            },
                            "stop".to_string(),
                        ),
                    }
                } else {
                    (
                        ChatMessage {
                            role: Role::Assistant,
                            content: response_text.clone(),
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        "stop".to_string(),
                    )
                };

            if request.stream {
                let stream = create_chat_stream(
                    request_id,
                    model_name.clone(),
                    created,
                    response_text,
                    prompt_tokens,
                    completion_tokens,
                );
                Sse::new(stream).into_response()
            } else {
                let response = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message,
                        finish_reason,
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

// =============================================================================
// Text Completions
// =============================================================================

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    let (_permit, _guard) = match acquire_inference_slot(&state).await {
        Ok(v) => v,
        Err(r) => return r,
    };

    let created = now_secs();
    let request_id = format!("cmpl-{}", created);
    let model_name = state.model_name.read().await;

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
        false,
    )
    .await
    {
        Ok((response_text, prompt_tokens, completion_tokens)) => {
            let response = CompletionResponse {
                id: request_id,
                object: "text_completion".to_string(),
                created,
                model: model_name.clone(),
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

// =============================================================================
// Model Hot-Swap
// =============================================================================

pub async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LoadModelRequest>,
) -> Response {
    tracing::info!("Hot-swap: loading model from {}", request.model_path);

    match reload_model_from_path(&state, &request.model_path).await {
        Ok((name, ctx_size)) => {
            Json(LoadModelResponse {
                status: "loaded".to_string(),
                model: name,
                context_size: ctx_size,
            })
            .into_response()
        }
        Err(e) => {
            let error = ErrorResponse::new(format!("Model load failed: {}", e), "server_error");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Reload model, swapping all state atomically.
pub async fn reload_model_from_path(
    state: &AppState,
    model_path: &str,
) -> Result<(String, usize), Box<dyn std::error::Error + Send + Sync>> {
    use crate::engine::{ChatTemplate, Engine};
    use crate::gguf::GgufFile;
    use crate::model::ModelLoader;

    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let chat_template = ChatTemplate::detect(&gguf);
    let loader = ModelLoader::load(model_path)?;
    let model_config = loader.config().clone();
    let model = loader.build_model()?;

    let use_gpu = std::env::var("LLAMA_GPU")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    let backend: Arc<dyn crate::Backend> = if use_gpu {
        Engine::select_gpu_backend(&model)
    } else {
        Arc::new(crate::backend::cpu::CpuBackend::new())
    };

    let name = std::path::Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llama")
        .to_string();
    let ctx_size = model_config.max_seq_len;

    // Swap atomically
    *state.model.write().await = Arc::new(model);
    *state.tokenizer.write().await = Arc::new(tokenizer);
    *state.config.write().await = model_config;
    *state.model_name.write().await = name.clone();
    *state.model_path.write().await = model_path.to_string();
    *state.chat_template.write().await = chat_template;
    *state.backend.write().await = backend;

    tracing::info!("Hot-swap complete: {} (ctx={})", name, ctx_size);
    Ok((name, ctx_size))
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Format chat messages into a prompt string, optionally injecting tool definitions
fn format_chat_messages(
    messages: &[ChatMessage],
    template: &ChatTemplate,
    tools: Option<&[ToolDefinition]>,
) -> String {
    let mut system_prompt = String::new();
    let mut conversation: Vec<&ChatMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => system_prompt = msg.content.clone(),
            _ => conversation.push(msg),
        }
    }

    // Inject tool definitions into the system prompt
    if let Some(tools) = tools {
        if !tools.is_empty() {
            let tools_section = format_tools_for_prompt(tools);
            if system_prompt.is_empty() {
                system_prompt = tools_section;
            } else {
                system_prompt = format!("{}\n\n{}", system_prompt, tools_section);
            }
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
                prompt.push_str(&msg.content);
            }
            Role::Tool => {
                let tool_result = if let Some(ref id) = msg.tool_call_id {
                    format!("[Tool Result (call_id={})]:\n{}", id, msg.content)
                } else {
                    format!("[Tool Result]:\n{}", msg.content)
                };
                prompt.push_str(&template.format_continuation(&tool_result));
            }
            Role::System => {}
        }
    }

    if is_first_user && !system_prompt.is_empty() {
        prompt.push_str(&template.format_first_turn(&system_prompt, ""));
    }

    prompt
}

/// Format tool definitions into a prompt-injectable string
fn format_tools_for_prompt(tools: &[ToolDefinition]) -> String {
    let mut section = String::from(
        "You have access to the following tools. To call a tool, respond with a JSON object in this exact format:\n\
         {\"tool_calls\": [{\"name\": \"function_name\", \"arguments\": {\"arg\": \"value\"}}]}\n\n\
         Available tools:\n",
    );

    for tool in tools {
        section.push_str(&format!("- {}", tool.function.name));
        if let Some(ref desc) = tool.function.description {
            section.push_str(&format!(": {}", desc));
        }
        section.push('\n');
        if let Some(ref params) = tool.function.parameters {
            if let Ok(pretty) = serde_json::to_string_pretty(params) {
                section.push_str(&format!("  Parameters: {}\n", pretty));
            }
        }
    }

    section
}

/// Try to parse tool calls from the model's output.
/// Looks for JSON with a "tool_calls" array, or a single function call object.
fn parse_tool_calls(text: &str, forced_name: Option<&str>) -> Option<Vec<ToolCall>> {
    let trimmed = text.trim();

    // Try to find JSON in the response
    let json_str = extract_json_from_text(trimmed)?;
    let value: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    let mut calls = Vec::new();

    if let Some(arr) = value.get("tool_calls").and_then(|v| v.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            let name = forced_name
                .map(String::from)
                .or_else(|| item.get("name").and_then(|v| v.as_str()).map(String::from))?;
            let args = item
                .get("arguments")
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .unwrap_or_else(|| "{}".to_string());
            calls.push(ToolCall {
                id: format!("call_{}", i),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name,
                    arguments: args,
                },
            });
        }
    } else if value.get("name").is_some() || forced_name.is_some() {
        let name = forced_name
            .map(String::from)
            .or_else(|| value.get("name").and_then(|v| v.as_str()).map(String::from))?;
        let args = value
            .get("arguments")
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_else(|| "{}".to_string());
        calls.push(ToolCall {
            id: "call_0".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name,
                arguments: args,
            },
        });
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Extract the first JSON object from text (handles markdown code blocks, etc.)
fn extract_json_from_text(text: &str) -> Option<String> {
    // Try the whole text first
    if text.starts_with('{') {
        if let Ok(_) = serde_json::from_str::<serde_json::Value>(text) {
            return Some(text.to_string());
        }
    }

    // Look for JSON inside code blocks
    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            let candidate = after[..end].trim();
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }

    // Find first { and matching }
    let mut depth = 0i32;
    let mut start_idx = None;
    for (i, ch) in text.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start_idx = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start_idx {
                        let candidate = &text[s..=i];
                        if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                            return Some(candidate.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

/// Generate text response using the model
async fn generate_response(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    sampler_config: SamplerConfig,
    _stop_sequences: Option<&[String]>,
    use_json_grammar: bool,
) -> Result<(String, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let model = state.model.read().await;
    let tokenizer = state.tokenizer.read().await;
    let config = state.config.read().await;
    let backend = state.backend.read().await;
    let chat_template = state.chat_template.read().await;

    let mut ctx = model.create_context(backend.clone());
    let mut sampler = Sampler::new(sampler_config, config.vocab_size);

    // Optional grammar sampler for structured JSON output
    let mut grammar_sampler = if use_json_grammar {
        let vocab: Vec<String> = (0..config.vocab_size as u32)
            .map(|id| {
                tokenizer
                    .get_token(id)
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        Some(GrammarSampler::new(
            Grammar::Json(crate::sampling::grammar::JsonGrammar {
                allow_any: true,
                ..Default::default()
            }),
            vocab,
        ))
    } else {
        None
    };

    let prompt_tokens = tokenizer.encode(prompt, true)?;
    let prompt_len = prompt_tokens.len();
    let mut all_tokens = prompt_tokens.clone();

    // Prefill
    if prompt_tokens.len() > 1 {
        for (i, &token) in prompt_tokens[..prompt_tokens.len() - 1].iter().enumerate() {
            if i < config.max_seq_len {
                let _ = model.forward(&[token], &mut ctx);
            }
        }
    }

    let stop_patterns = chat_template.stop_patterns();
    let mut response_text = String::new();
    let mut completion_tokens = 0;

    for _ in 0..max_tokens {
        let last_token = *all_tokens
            .last()
            .unwrap_or(&tokenizer.special_tokens.bos_token_id);

        let logits = model.forward(&[last_token], &mut ctx)?;

        // Apply grammar constraint if active
        if let Some(ref gs) = grammar_sampler {
            let mut logit_data = logits.as_f32()?.to_vec();
            gs.apply_mask(&mut logit_data);
            // Sample from masked logits via the tensor
            let masked_logits =
                crate::tensor::Tensor::from_f32(&logit_data, logits.shape().to_vec())
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
            let next_token = sampler.sample(&masked_logits, &all_tokens);

            if next_token == tokenizer.special_tokens.eos_token_id {
                break;
            }

            if let Ok(text) = tokenizer.decode(&[next_token]) {
                if let Some(ref mut gs) = grammar_sampler {
                    gs.record_token(&text);
                }
                let combined = format!("{}{}", response_text, text);
                let should_stop = stop_patterns.iter().any(|p| combined.contains(p));
                if should_stop {
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

            if grammar_sampler.as_ref().map_or(false, |gs| gs.is_complete()) {
                break;
            }
        } else {
            let next_token = sampler.sample(&logits, &all_tokens);

            if next_token == tokenizer.special_tokens.eos_token_id {
                break;
            }

            if let Ok(text) = tokenizer.decode(&[next_token]) {
                let combined = format!("{}{}", response_text, text);
                let should_stop = stop_patterns.iter().any(|p| combined.contains(p));
                if should_stop {
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
    }

    Ok((
        response_text.trim().to_string(),
        prompt_len,
        completion_tokens,
    ))
}

/// Create streaming response for chat completions with usage in the final chunk
fn create_chat_stream(
    request_id: String,
    model: String,
    created: u64,
    response_text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let chunks = vec![
        // Role chunk
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
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
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
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        },
        // Final chunk with finish reason and usage
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
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
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

#[cfg(feature = "rag")]
pub struct RagState {
    pub knowledge_bases: tokio::sync::RwLock<HashMap<String, crate::rag::KnowledgeBaseConfig>>,
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

#[cfg(feature = "rag")]
pub async fn retrieve(
    State(rag_state): State<Arc<RagState>>,
    Json(request): Json<RetrieveRequest>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig, RetrievalConfig};

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

    let mut retrieval_config = RetrievalConfig::default();

    if let Some(ref config) = request.retrieval_configuration
        && let Some(ref vs_config) = config.vector_search_configuration
    {
        retrieval_config.max_results = vs_config.number_of_results;

        if let Some(ref filter) = vs_config.filter {
            retrieval_config.filter = convert_filter(filter);
        }
    }

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

    let mut retrieval_config = RetrievalConfig::default();

    if let Some(ref config) = request
        .retrieve_and_generate_configuration
        .knowledge_base_configuration
        .retrieval_configuration
        && let Some(ref vs_config) = config.vector_search_configuration
    {
        retrieval_config.max_results = vs_config.number_of_results;
    }

    if let Some(ref gen_config) = request
        .retrieve_and_generate_configuration
        .knowledge_base_configuration
        .generation_configuration
        && let Some(ref template) = gen_config.prompt_template
    {
        let converted = template
            .text_prompt_template
            .replace("$query$", "{query}")
            .replace("$search_results$", "{context}");
        retrieval_config.prompt_template = Some(converted);
    }

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

    let (_permit, _guard) = match acquire_inference_slot(&app_state).await {
        Ok(v) => v,
        Err(r) => return r,
    };

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
        false,
    )
    .await
    {
        Ok((text, _, _)) => text,
        Err(e) => {
            let error = ErrorResponse::new(format!("Generation failed: {}", e), "generation_error");
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
        }
    };

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

#[cfg(feature = "rag")]
pub async fn ingest(
    State(rag_state): State<Arc<RagState>>,
    Json(request): Json<IngestRequest>,
) -> Response {
    use crate::rag::{DataSource, KnowledgeBase, KnowledgeBaseConfig};

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

#[cfg(feature = "rag")]
pub async fn get_knowledge_base(
    State(rag_state): State<Arc<RagState>>,
    Path(kb_id): Path<String>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig};

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

#[cfg(feature = "rag")]
pub async fn delete_knowledge_base(
    State(rag_state): State<Arc<RagState>>,
    Path(kb_id): Path<String>,
) -> Response {
    use crate::rag::{KnowledgeBase, KnowledgeBaseConfig};

    let kb_config = {
        let mut kbs = rag_state.knowledge_bases.write().await;
        kbs.remove(&kb_id).unwrap_or_else(|| KnowledgeBaseConfig {
            name: kb_id.clone(),
            storage: rag_state.rag_config.clone(),
            ..Default::default()
        })
    };

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

#[cfg(feature = "rag")]
fn convert_filter(filter: &RetrievalFilter) -> Option<crate::rag::MetadataFilter> {
    use crate::rag::MetadataFilter;

    if let Some(ref and_filters) = filter.and_all {
        let converted: Vec<_> = and_filters.iter().filter_map(convert_filter).collect();
        if !converted.is_empty() {
            return Some(MetadataFilter::and(converted));
        }
    }

    if let Some(ref or_filters) = filter.or_all {
        let converted: Vec<_> = or_filters.iter().filter_map(convert_filter).collect();
        if !converted.is_empty() {
            return Some(MetadataFilter::or(converted));
        }
    }

    if let Some(ref cond) = filter.equals {
        return Some(MetadataFilter::eq(&cond.key, cond.value.clone()));
    }

    if let Some(ref cond) = filter.not_equals {
        return Some(MetadataFilter::ne(&cond.key, cond.value.clone()));
    }

    if let Some(ref cond) = filter.greater_than {
        return Some(MetadataFilter::gt(&cond.key, cond.value.clone()));
    }

    if let Some(ref cond) = filter.less_than {
        return Some(MetadataFilter::lt(&cond.key, cond.value.clone()));
    }

    if let Some(ref cond) = filter.string_contains
        && let Some(s) = cond.value.as_str()
    {
        return Some(MetadataFilter::contains(&cond.key, s));
    }

    if let Some(ref cond) = filter.starts_with
        && let Some(s) = cond.value.as_str()
    {
        return Some(MetadataFilter::starts_with(&cond.key, s));
    }

    None
}

#[cfg(feature = "rag")]
fn current_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}Z", now)
}
