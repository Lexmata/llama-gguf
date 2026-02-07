//! HTTP client for connecting to an OpenAI-compatible inference server.
//!
//! This module provides a `RemoteChatClient` that connects to a remote
//! llama-gguf server (or any OpenAI-compatible API) and provides the same
//! interactive chat experience as the local `ChatEngine`.
//!
//! # Example
//!
//! ```no_run
//! use llama_gguf::client::RemoteChatClient;
//!
//! let mut client = RemoteChatClient::new(
//!     "http://192.168.1.4:8080",
//!     Some("You are a helpful assistant."),
//!     0.7,
//!     512,
//!     0.9,
//! ).unwrap();
//!
//! let response = client.chat_streaming("Hello!").unwrap();
//! println!("Response: {}", response);
//! ```

use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

/// Role in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Request body for `/v1/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: usize,
    top_p: f32,
    stream: bool,
}

/// Non-streaming response from `/v1/chat/completions`.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ResponseChoice>,
}

#[derive(Debug, Deserialize)]
struct ResponseChoice {
    message: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: String,
}

/// A single streaming chunk from the SSE stream.
#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: Option<DeltaContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaContent {
    content: Option<String>,
}

/// Health check response from `/health`.
#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub context_size: usize,
}

/// Error returned by the client.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Server error ({status}): {body}")]
    ServerError { status: u16, body: String },

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// HTTP client for an OpenAI-compatible inference server.
///
/// Maintains conversation history and streams responses token-by-token.
pub struct RemoteChatClient {
    base_url: String,
    client: reqwest::blocking::Client,
    history: Vec<ChatMessage>,
    system_prompt: String,
    temperature: f32,
    max_tokens: usize,
    top_p: f32,
    model_name: String,
}

impl RemoteChatClient {
    /// Create a new client and verify the server is reachable.
    pub fn new(
        server_url: &str,
        system_prompt: Option<&str>,
        temperature: f32,
        max_tokens: usize,
        top_p: f32,
    ) -> Result<Self, ClientError> {
        let base_url = server_url.trim_end_matches('/').to_string();
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()?;

        // Verify server is reachable via health endpoint
        let health_url = format!("{}/health", base_url);
        let model_name = match client.get(&health_url).send() {
            Ok(resp) if resp.status().is_success() => {
                let health: HealthResponse =
                    resp.json().unwrap_or(HealthResponse {
                        status: "ok".to_string(),
                        model: "unknown".to_string(),
                        context_size: 0,
                    });
                eprintln!("Connected to server: {}", base_url);
                eprintln!("Model: {} (context: {})", health.model, health.context_size);
                health.model
            }
            Ok(resp) => {
                eprintln!(
                    "Warning: Server health check returned status {}",
                    resp.status()
                );
                "unknown".to_string()
            }
            Err(e) => {
                return Err(ClientError::ConnectionFailed(format!(
                    "Cannot reach server at {}: {}",
                    base_url, e
                )));
            }
        };

        let system = system_prompt
            .unwrap_or("You are a helpful AI assistant.")
            .to_string();

        Ok(Self {
            base_url,
            client,
            history: Vec::new(),
            system_prompt: system,
            temperature,
            max_tokens,
            top_p,
            model_name,
        })
    }

    /// Get the system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Get the remote model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Clear conversation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Send a message and stream the response, printing tokens to stdout.
    /// Returns the full response text.
    pub fn chat_streaming(&mut self, user_message: &str) -> Result<String, ClientError> {
        // Add user message to history
        self.history.push(ChatMessage {
            role: Role::User,
            content: user_message.to_string(),
        });

        // Build full message list with system prompt
        let mut messages = vec![ChatMessage {
            role: Role::System,
            content: self.system_prompt.clone(),
        }];
        messages.extend(self.history.clone());

        let request = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stream: true,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client.post(&url).json(&request).send()?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().unwrap_or_default();
            return Err(ClientError::ServerError { status, body });
        }

        // Read SSE stream line by line
        let mut full_response = String::new();
        let reader = io::BufReader::new(response);

        for line in reader.lines() {
            let line = line?;

            // SSE lines starting with "data: "
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }

                if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(delta) = &choice.delta {
                            if let Some(content) = &delta.content {
                                print!("{}", content);
                                io::stdout().flush()?;
                                full_response.push_str(content);
                            }
                        }
                        // Check for finish
                        if choice.finish_reason.is_some() {
                            break;
                        }
                    }
                }
            }
        }

        // Add assistant response to history
        self.history.push(ChatMessage {
            role: Role::Assistant,
            content: full_response.clone(),
        });

        Ok(full_response)
    }

    /// Send a message and get the full response (non-streaming).
    pub fn chat(&mut self, user_message: &str) -> Result<String, ClientError> {
        self.history.push(ChatMessage {
            role: Role::User,
            content: user_message.to_string(),
        });

        let mut messages = vec![ChatMessage {
            role: Role::System,
            content: self.system_prompt.clone(),
        }];
        messages.extend(self.history.clone());

        let request = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stream: false,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client.post(&url).json(&request).send()?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().unwrap_or_default();
            return Err(ClientError::ServerError { status, body });
        }

        let resp: ChatCompletionResponse = response.json()?;
        let content = resp
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .map(|m| m.content.clone())
            .unwrap_or_default();

        self.history.push(ChatMessage {
            role: Role::Assistant,
            content: content.clone(),
        });

        Ok(content)
    }
}
