//! High-level inference engine for llama-gguf.
//!
//! Provides [`Engine`] and [`ChatEngine`] for easy model loading and text generation
//! without needing to manually wire together GGUF files, tokenizers, backends, and samplers.
//!
//! # Example
//!
//! ```no_run
//! use llama_gguf::engine::{Engine, EngineConfig};
//!
//! let engine = Engine::load(EngineConfig {
//!     model_path: "model.gguf".into(),
//!     ..Default::default()
//! }).unwrap();
//!
//! let response = engine.generate("What is a tort?", 256).unwrap();
//! println!("{}", response);
//! ```

use std::sync::Arc;

use crate::backend::Backend;
use crate::gguf::GgufFile;
use crate::model::{
    EmbeddingConfig, EmbeddingExtractor, InferenceContext, Model, ModelConfig, ModelLoader,
};
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur during engine operations.
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF error: {0}")]
    Gguf(#[from] crate::gguf::GgufError),

    #[error("Model error: {0}")]
    Model(#[from] crate::model::ModelError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] crate::model::EmbeddingError),

    #[error("Engine error: {0}")]
    Other(String),
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for creating an [`Engine`].
///
/// Can be constructed manually, from a [`Config`](crate::config::Config) TOML file,
/// or with [`Default::default()`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    /// Path to the GGUF model file.
    pub model_path: String,

    /// Temperature for sampling (0.0 = greedy, higher = more random).
    pub temperature: f32,

    /// Top-K sampling: only consider the K most likely tokens (0 = disabled).
    pub top_k: usize,

    /// Top-P (nucleus) sampling: only consider tokens with cumulative probability <= p.
    pub top_p: f32,

    /// Repetition penalty (1.0 = no penalty).
    pub repeat_penalty: f32,

    /// Default maximum tokens to generate.
    pub max_tokens: usize,

    /// Random seed for reproducible generation (None = random).
    pub seed: Option<u64>,

    /// Use GPU acceleration (requires `cuda` feature).
    pub use_gpu: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            max_tokens: 512,
            seed: None,
            use_gpu: false,
        }
    }
}

impl EngineConfig {
    /// Load an `EngineConfig` from a [`Config`](crate::config::Config) TOML file.
    ///
    /// This is a convenience method that loads the full config and extracts
    /// the engine-relevant sections.
    pub fn from_config_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::config::ConfigError> {
        let config = crate::config::Config::from_file(path)?;
        Ok(config.to_engine_config(None))
    }

    /// Load an `EngineConfig` using the full config precedence chain.
    ///
    /// Searches default config file locations, then applies environment
    /// variable overrides.
    pub fn from_config(
        config_path: Option<impl AsRef<std::path::Path>>,
    ) -> Result<Self, crate::config::ConfigError> {
        let config = crate::config::Config::load(config_path)?;
        Ok(config.to_engine_config(None))
    }
}

// ============================================================================
// Chat template detection
// ============================================================================

/// Detected chat template format from the GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    /// `<|user|>\n...<|assistant|>\n` format
    UserAssistant,
    /// ChatML: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
    ChatML,
    /// Llama-2: `[INST] ... [/INST]`
    Llama2,
    /// No template detected, use raw text.
    None,
}

impl ChatTemplate {
    /// Detect chat template from GGUF metadata.
    pub fn detect(gguf: &GgufFile) -> Self {
        if let Some(template) = gguf.data.get_string("tokenizer.chat_template") {
            if template.contains("<|user|>") {
                ChatTemplate::UserAssistant
            } else if template.contains("<|im_start|>") {
                ChatTemplate::ChatML
            } else if template.contains("[INST]") {
                ChatTemplate::Llama2
            } else {
                ChatTemplate::None
            }
        } else {
            ChatTemplate::None
        }
    }

    /// Wrap a raw prompt in the appropriate chat format.
    pub fn wrap_prompt(&self, prompt: &str) -> String {
        // If prompt already contains chat tokens, return as-is
        if prompt.contains("<|user|>")
            || prompt.contains("<|im_start|>")
            || prompt.contains("[INST]")
        {
            return prompt.to_string();
        }

        match self {
            ChatTemplate::UserAssistant => {
                format!("<|user|>\n{}<|assistant|>\n", prompt)
            }
            ChatTemplate::ChatML => {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    prompt
                )
            }
            ChatTemplate::Llama2 => {
                format!("[INST] {} [/INST]", prompt)
            }
            ChatTemplate::None => prompt.to_string(),
        }
    }

    /// Format a chat message with system prompt for the first turn.
    pub fn format_first_turn(&self, system_prompt: &str, user_message: &str) -> String {
        match self {
            ChatTemplate::UserAssistant => {
                format!(
                    "<|system|>\n{}<|user|>\n{}<|assistant|>\n",
                    system_prompt, user_message
                )
            }
            ChatTemplate::ChatML => {
                format!(
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    system_prompt, user_message
                )
            }
            ChatTemplate::Llama2 => {
                format!(
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                    system_prompt, user_message
                )
            }
            ChatTemplate::None => {
                format!("System: {}\n\nUser: {}\n\nAssistant:", system_prompt, user_message)
            }
        }
    }

    /// Format a continuation turn (not the first message).
    pub fn format_continuation(&self, user_message: &str) -> String {
        match self {
            ChatTemplate::UserAssistant => {
                format!("<|user|>\n{}<|assistant|>\n", user_message)
            }
            ChatTemplate::ChatML => {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    user_message
                )
            }
            ChatTemplate::Llama2 => {
                format!(" [INST] {} [/INST]", user_message)
            }
            ChatTemplate::None => {
                format!("\n\nUser: {}\n\nAssistant:", user_message)
            }
        }
    }

    /// Patterns that indicate the model is trying to generate a new user turn
    /// (i.e., the response is complete).
    pub fn stop_patterns(&self) -> &[&str] {
        match self {
            ChatTemplate::UserAssistant => &["<|user|>", "<|end|>"],
            ChatTemplate::ChatML => &["<|im_end|>", "<|im_start|>"],
            ChatTemplate::Llama2 => &["[INST]", "</s>"],
            ChatTemplate::None => &["User:", "\nUser:"],
        }
    }
}

// ============================================================================
// Engine
// ============================================================================

/// High-level inference engine that wraps model loading, tokenization, and generation.
///
/// `Engine` is `Send + Sync` safe for the immutable model and config, but the
/// mutable inference context and sampler are created per-call via internal state.
pub struct Engine {
    gguf: GgufFile,
    model: Box<dyn Model>,
    tokenizer: Tokenizer,
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    sampler_config: SamplerConfig,
    chat_template: ChatTemplate,
    add_bos: bool,
    engine_config: EngineConfig,
}

impl Engine {
    /// Load a model and create an inference engine.
    ///
    /// This opens the GGUF file, loads the tokenizer and model weights,
    /// and selects the appropriate backend (CPU or GPU).
    pub fn load(config: EngineConfig) -> Result<Self, EngineError> {
        if config.model_path.is_empty() {
            return Err(EngineError::Other("model_path is required".into()));
        }

        tracing::info!("Loading model from: {}", config.model_path);

        // Load GGUF file
        let gguf = GgufFile::open(&config.model_path)?;

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        tracing::info!("Vocabulary size: {}", tokenizer.vocab_size);

        // Load model weights
        tracing::info!("Loading model weights...");
        let loader = ModelLoader::load(&config.model_path)?;
        let model_config = loader.config().clone();
        tracing::info!(
            "Model: {} layers, {} heads, {} hidden dim, {} ctx",
            model_config.num_layers,
            model_config.num_heads,
            model_config.hidden_size,
            model_config.max_seq_len,
        );

        let concrete_model = loader.build_model()?;

        // Select backend -- must happen before boxing the model so CUDA can
        // access the concrete LlamaModel type for weight upload.
        let backend: Arc<dyn Backend> = if config.use_gpu {
            #[cfg(feature = "cuda")]
            {
                match crate::backend::cuda::CudaBackend::new() {
                    Ok(mut cuda) => {
                        tracing::info!("Using CUDA backend: {}", cuda.device_name());
                        if let Err(e) = cuda.load_model_weights(&concrete_model) {
                            tracing::warn!(
                                "Failed to load GPU weights ({}), using quantized ops",
                                e
                            );
                        }
                        Arc::new(cuda)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to initialize CUDA ({}), falling back to CPU",
                            e
                        );
                        Arc::new(crate::backend::cpu::CpuBackend::new())
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA not compiled in, falling back to CPU");
                Arc::new(crate::backend::cpu::CpuBackend::new())
            }
        } else {
            Arc::new(crate::backend::cpu::CpuBackend::new())
        };

        // Box the model as a trait object now that GPU weight upload is done
        let model: Box<dyn Model> = Box::new(concrete_model);

        // Detect chat template
        let chat_template = ChatTemplate::detect(&gguf);
        tracing::info!("Chat template: {:?}", chat_template);

        // Check BOS token preference
        let add_bos = gguf
            .data
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(true);

        // Build sampler config
        let sampler_config = SamplerConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
            seed: config.seed,
            ..Default::default()
        };

        tracing::info!("Engine ready");

        Ok(Self {
            gguf,
            model,
            tokenizer,
            config: model_config,
            backend,
            sampler_config,
            chat_template,
            add_bos,
            engine_config: config,
        })
    }

    /// Get the model configuration.
    pub fn model_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the detected chat template.
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Get the GGUF file metadata.
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// Get the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the engine configuration.
    pub fn engine_config(&self) -> &EngineConfig {
        &self.engine_config
    }

    /// Generate text from a prompt.
    ///
    /// The prompt is automatically wrapped with the detected chat template
    /// unless it already contains chat formatting tokens.
    ///
    /// Returns the generated text (not including the prompt).
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, EngineError> {
        let mut ctx = InferenceContext::new(&self.config, self.backend.clone());
        let mut sampler = Sampler::new(self.sampler_config.clone(), self.config.vocab_size);

        // Wrap prompt with chat template
        let formatted = self.chat_template.wrap_prompt(prompt);
        let mut tokens = self.tokenizer.encode(&formatted, self.add_bos)?;

        let mut output = String::new();

        for _ in 0..max_tokens {
            // Check if we hit EOS
            if let Some(&last) = tokens.last() {
                if last == self.tokenizer.special_tokens.eos_token_id {
                    break;
                }
            }

            // Run forward pass
            let input_tokens = if ctx.position == 0 {
                &tokens[..]
            } else {
                &tokens[tokens.len() - 1..]
            };

            let logits = self.model.forward(input_tokens, &mut ctx)?;
            let next_token = sampler.sample(&logits, &tokens);

            // Check for EOS
            if next_token == self.tokenizer.special_tokens.eos_token_id {
                break;
            }

            // Decode token
            if let Ok(text) = self.tokenizer.decode(&[next_token]) {
                // Check for stop patterns
                let combined = format!("{}{}", output, text);
                let stop = self
                    .chat_template
                    .stop_patterns()
                    .iter()
                    .any(|p| combined.contains(p));

                if stop {
                    // Add only the text before the stop pattern
                    for pattern in self.chat_template.stop_patterns() {
                        if let Some(idx) = combined.find(pattern) {
                            output = combined[..idx].to_string();
                            return Ok(output.trim().to_string());
                        }
                    }
                    break;
                }

                output.push_str(&text);
            }

            tokens.push(next_token);
        }

        Ok(output.trim().to_string())
    }

    /// Generate text from a prompt, yielding tokens as they are produced.
    ///
    /// Each item in the returned iterator is a `Result<String, EngineError>` containing
    /// the decoded text of one or more tokens.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> GenerationStream<'_> {
        GenerationStream::new(self, prompt, max_tokens)
    }

    /// Extract embeddings from text using the model.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        let mut ctx = InferenceContext::new(&self.config, self.backend.clone());
        let embed_config = EmbeddingConfig::default();
        let extractor = EmbeddingExtractor::new(embed_config, &self.config);
        let embedding =
            extractor.embed_text(self.model.as_ref(), &self.tokenizer, &mut ctx, text)?;
        Ok(embedding)
    }
}

// ============================================================================
// Streaming generation
// ============================================================================

/// Iterator that yields generated tokens as strings.
///
/// Created by [`Engine::generate_streaming`].
pub struct GenerationStream<'a> {
    engine: &'a Engine,
    ctx: InferenceContext,
    sampler: Sampler,
    tokens: Vec<u32>,
    remaining: usize,
    done: bool,
    accumulated: String,
}

impl<'a> GenerationStream<'a> {
    fn new(engine: &'a Engine, prompt: &str, max_tokens: usize) -> Self {
        let ctx = InferenceContext::new(&engine.config, engine.backend.clone());
        let sampler = Sampler::new(engine.sampler_config.clone(), engine.config.vocab_size);

        let formatted = engine.chat_template.wrap_prompt(prompt);
        let tokens = engine
            .tokenizer
            .encode(&formatted, engine.add_bos)
            .unwrap_or_default();

        Self {
            engine,
            ctx,
            sampler,
            tokens,
            remaining: max_tokens,
            done: false,
            accumulated: String::new(),
        }
    }
}

impl<'a> Iterator for GenerationStream<'a> {
    type Item = Result<String, EngineError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.remaining == 0 {
            return None;
        }

        // Check EOS from last token
        if let Some(&last) = self.tokens.last() {
            if last == self.engine.tokenizer.special_tokens.eos_token_id {
                self.done = true;
                return None;
            }
        }

        // Forward pass
        let input_tokens = if self.ctx.position == 0 {
            &self.tokens[..]
        } else {
            &self.tokens[self.tokens.len() - 1..]
        };

        let logits = match self.engine.model.forward(input_tokens, &mut self.ctx) {
            Ok(l) => l,
            Err(e) => {
                self.done = true;
                return Some(Err(EngineError::Model(e)));
            }
        };

        let next_token = self.sampler.sample(&logits, &self.tokens);

        // Check EOS
        if next_token == self.engine.tokenizer.special_tokens.eos_token_id {
            self.done = true;
            return None;
        }

        // Decode
        match self.engine.tokenizer.decode(&[next_token]) {
            Ok(text) => {
                // Check stop patterns
                let combined = format!("{}{}", self.accumulated, text);
                for pattern in self.engine.chat_template.stop_patterns() {
                    if combined.contains(pattern) {
                        self.done = true;
                        // Return text before the stop pattern if any
                        if let Some(idx) = combined.find(pattern) {
                            let before = &combined[self.accumulated.len()..idx];
                            if !before.is_empty() {
                                return Some(Ok(before.to_string()));
                            }
                        }
                        return None;
                    }
                }

                self.accumulated.push_str(&text);
                self.tokens.push(next_token);
                self.remaining -= 1;
                Some(Ok(text))
            }
            Err(e) => {
                self.tokens.push(next_token);
                self.remaining -= 1;
                Some(Err(EngineError::Tokenizer(e)))
            }
        }
    }
}

// ============================================================================
// ChatEngine
// ============================================================================

/// High-level chat engine that maintains conversation state.
///
/// Wraps an [`Engine`] with conversation history, context management,
/// and automatic chat template formatting.
pub struct ChatEngine {
    engine: Engine,
    system_prompt: String,
    conversation_tokens: Vec<u32>,
    ctx: InferenceContext,
    sampler: Sampler,
    is_first_turn: bool,
}

impl ChatEngine {
    /// Create a new chat engine from a loaded [`Engine`].
    pub fn new(engine: Engine, system_prompt: Option<String>) -> Self {
        let ctx = InferenceContext::new(&engine.config, engine.backend.clone());
        let sampler = Sampler::new(engine.sampler_config.clone(), engine.config.vocab_size);

        Self {
            system_prompt: system_prompt
                .unwrap_or_else(|| "You are a helpful AI assistant.".to_string()),
            conversation_tokens: Vec::new(),
            ctx,
            sampler,
            is_first_turn: true,
            engine,
        }
    }

    /// Get a reference to the underlying engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get the current system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Get the number of tokens in the current conversation context.
    pub fn context_len(&self) -> usize {
        self.conversation_tokens.len()
    }

    /// Send a message and get the full response.
    pub fn chat(&mut self, message: &str) -> Result<String, EngineError> {
        let max_tokens = self.engine.engine_config.max_tokens;

        // Format the message using the chat template
        let formatted = if self.is_first_turn {
            self.engine
                .chat_template
                .format_first_turn(&self.system_prompt, message)
        } else {
            self.engine.chat_template.format_continuation(message)
        };

        // Encode new tokens
        let new_tokens = self
            .engine
            .tokenizer
            .encode(&formatted, self.is_first_turn && self.engine.add_bos)?;

        // Check context length and trim if needed
        self.ensure_context_space(new_tokens.len(), max_tokens);

        // Add new tokens to conversation
        self.conversation_tokens.extend(&new_tokens);

        // Process the new prompt tokens through the model
        let start_pos = self.ctx.position;
        for (i, &token) in new_tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos < self.engine.config.max_seq_len {
                let _ = self.engine.model.forward(&[token], &mut self.ctx);
            }
        }

        // Generate response
        let mut response_text = String::new();

        for _ in 0..max_tokens {
            // Check stop patterns
            let should_stop = self
                .engine
                .chat_template
                .stop_patterns()
                .iter()
                .any(|p| response_text.contains(p));
            if should_stop {
                // Trim the stop pattern from the end
                for pattern in self.engine.chat_template.stop_patterns() {
                    if let Some(idx) = response_text.find(pattern) {
                        response_text.truncate(idx);
                        break;
                    }
                }
                break;
            }

            let last_token = *self
                .conversation_tokens
                .last()
                .unwrap_or(&self.engine.tokenizer.special_tokens.bos_token_id);

            let logits = self.engine.model.forward(&[last_token], &mut self.ctx)?;
            let next_token = self.sampler.sample(&logits, &self.conversation_tokens);

            // Check for EOS
            if next_token == self.engine.tokenizer.special_tokens.eos_token_id {
                break;
            }

            if let Ok(text) = self.engine.tokenizer.decode(&[next_token]) {
                response_text.push_str(&text);
            }

            self.conversation_tokens.push(next_token);
        }

        self.is_first_turn = false;
        Ok(response_text.trim().to_string())
    }

    /// Send a message and stream the response token by token.
    ///
    /// Returns an iterator of `Result<String, EngineError>` where each item is
    /// a decoded token chunk.
    pub fn chat_streaming(
        &mut self,
        message: &str,
    ) -> Result<ChatStream<'_>, EngineError> {
        let max_tokens = self.engine.engine_config.max_tokens;

        // Format the message
        let formatted = if self.is_first_turn {
            self.engine
                .chat_template
                .format_first_turn(&self.system_prompt, message)
        } else {
            self.engine.chat_template.format_continuation(message)
        };

        // Encode new tokens
        let new_tokens = self
            .engine
            .tokenizer
            .encode(&formatted, self.is_first_turn && self.engine.add_bos)?;

        // Ensure context space
        self.ensure_context_space(new_tokens.len(), max_tokens);

        // Add and process new tokens
        self.conversation_tokens.extend(&new_tokens);
        let start_pos = self.ctx.position;
        for (i, &token) in new_tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos < self.engine.config.max_seq_len {
                let _ = self.engine.model.forward(&[token], &mut self.ctx);
            }
        }

        self.is_first_turn = false;

        Ok(ChatStream {
            chat_engine: self,
            remaining: max_tokens,
            done: false,
            accumulated: String::new(),
        })
    }

    /// Clear conversation history and reset context.
    pub fn clear_history(&mut self) {
        self.conversation_tokens.clear();
        self.ctx.reset();
        self.sampler.reset();
        self.is_first_turn = true;
    }

    /// Ensure there's enough space in the context for new tokens + generation.
    fn ensure_context_space(&mut self, new_token_count: usize, max_gen_tokens: usize) {
        let total_len = self.conversation_tokens.len() + new_token_count + max_gen_tokens;

        if total_len > self.engine.config.max_seq_len {
            let excess = total_len - self.engine.config.max_seq_len + 100;

            if excess >= self.conversation_tokens.len() {
                tracing::warn!("Context full, resetting conversation");
                self.conversation_tokens.clear();
                self.ctx.reset();
            } else {
                tracing::info!("Trimming {} tokens from context", excess);
                self.conversation_tokens = self.conversation_tokens[excess..].to_vec();
                self.ctx.kv_cache.shift_left(excess);
                self.ctx.position = self.ctx.position.saturating_sub(excess);
            }
        }
    }
}

// ============================================================================
// Chat streaming
// ============================================================================

/// Iterator that yields chat response tokens as strings.
///
/// Created by [`ChatEngine::chat_streaming`].
pub struct ChatStream<'a> {
    chat_engine: &'a mut ChatEngine,
    remaining: usize,
    done: bool,
    accumulated: String,
}

impl<'a> Iterator for ChatStream<'a> {
    type Item = Result<String, EngineError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.remaining == 0 {
            return None;
        }

        // Check stop patterns on accumulated text
        for pattern in self.chat_engine.engine.chat_template.stop_patterns() {
            if self.accumulated.contains(pattern) {
                self.done = true;
                return None;
            }
        }

        let last_token = *self
            .chat_engine
            .conversation_tokens
            .last()
            .unwrap_or(&self.chat_engine.engine.tokenizer.special_tokens.bos_token_id);

        let logits = match self
            .chat_engine
            .engine
            .model
            .forward(&[last_token], &mut self.chat_engine.ctx)
        {
            Ok(l) => l,
            Err(e) => {
                self.done = true;
                return Some(Err(EngineError::Model(e)));
            }
        };

        let next_token = self
            .chat_engine
            .sampler
            .sample(&logits, &self.chat_engine.conversation_tokens);

        // Check for EOS
        if next_token == self.chat_engine.engine.tokenizer.special_tokens.eos_token_id {
            self.done = true;
            return None;
        }

        match self.chat_engine.engine.tokenizer.decode(&[next_token]) {
            Ok(text) => {
                // Check stop patterns in accumulated + new text
                let combined = format!("{}{}", self.accumulated, text);
                for pattern in self.chat_engine.engine.chat_template.stop_patterns() {
                    if combined.contains(pattern) {
                        self.done = true;
                        if let Some(idx) = combined.find(pattern) {
                            let before = &combined[self.accumulated.len()..idx];
                            self.chat_engine.conversation_tokens.push(next_token);
                            if !before.is_empty() {
                                return Some(Ok(before.to_string()));
                            }
                        }
                        return None;
                    }
                }

                self.accumulated.push_str(&text);
                self.chat_engine.conversation_tokens.push(next_token);
                self.remaining -= 1;
                Some(Ok(text))
            }
            Err(e) => {
                self.chat_engine.conversation_tokens.push(next_token);
                self.remaining -= 1;
                Some(Err(EngineError::Tokenizer(e)))
            }
        }
    }
}
