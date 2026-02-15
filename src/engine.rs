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
    /// Path to the model file (GGUF or ONNX).
    pub model_path: String,

    /// Optional path to a tokenizer file.
    ///
    /// For ONNX models, this defaults to `tokenizer.json` in the same directory.
    /// Can also point to a GGUF file to extract just the tokenizer.
    pub tokenizer_path: Option<String>,

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
            tokenizer_path: None,
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
    /// Detect chat template from model type string (for ONNX models without GGUF metadata).
    pub fn detect_from_model_type(model_type: Option<&str>) -> Self {
        match model_type {
            Some("qwen2" | "qwen") => ChatTemplate::ChatML,
            Some("llama" | "codellama") => ChatTemplate::Llama2,
            Some("mistral" | "mixtral") => ChatTemplate::Llama2,
            _ => ChatTemplate::None,
        }
    }

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
                format!(
                    "System: {}\n\nUser: {}\n\nAssistant:",
                    system_prompt, user_message
                )
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
    gguf: Option<GgufFile>,
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
    /// This opens the model file (GGUF or ONNX), loads the tokenizer and model weights,
    /// and selects the appropriate backend (CPU or GPU).
    ///
    /// Format is auto-detected by file extension:
    /// - `.gguf` -- GGUF format (default)
    /// - `.onnx` -- ONNX format (requires `onnx` feature, companion config.json + tokenizer.json)
    pub fn load(config: EngineConfig) -> Result<Self, EngineError> {
        if config.model_path.is_empty() {
            return Err(EngineError::Other("model_path is required".into()));
        }

        let path = std::path::Path::new(&config.model_path);

        // Detect format by extension
        match path.extension().and_then(|e| e.to_str()) {
            #[cfg(feature = "onnx")]
            Some("onnx") => Self::load_onnx(config),
            #[cfg(not(feature = "onnx"))]
            Some("onnx") => Err(EngineError::Other(
                "ONNX support requires the `onnx` feature. Build with: cargo build --features onnx"
                    .into(),
            )),
            _ => Self::load_gguf(config),
        }
    }

    /// Load from a GGUF model file (existing path).
    fn load_gguf(config: EngineConfig) -> Result<Self, EngineError> {
        tracing::info!("Loading GGUF model from: {}", config.model_path);

        // Load GGUF file
        let gguf = GgufFile::open(&config.model_path)?;

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = if let Some(ref tok_path) = config.tokenizer_path {
            // User-specified tokenizer (could be a tokenizer.json or a GGUF file)
            if tok_path.ends_with(".json") {
                Tokenizer::from_hf_json(tok_path)?
            } else {
                let tok_gguf = GgufFile::open(tok_path)?;
                Tokenizer::from_gguf(&tok_gguf)?
            }
        } else {
            Tokenizer::from_gguf(&gguf)?
        };
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

        // Select backend and model implementation.
        //
        // When CUDA is available we try to create a GpuModelWrapper first.
        // This runs the entire forward pass on GPU with pre-allocated scratch
        // buffers, eliminating ~770 host↔device transfers per token that the
        // standard Backend-trait path incurs.  If GPU-only init fails we
        // fall back to the regular CudaBackend (per-op transfers) or CPU.
        let (backend, model): (Arc<dyn Backend>, Box<dyn Model>) = if config.use_gpu {
            Self::select_gpu_model(concrete_model, &model_config)
        } else {
            (
                Arc::new(crate::backend::cpu::CpuBackend::new()),
                Box::new(concrete_model),
            )
        };

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
            gguf: Some(gguf),
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

    /// Load from an ONNX model file with companion config.json and tokenizer.json.
    #[cfg(feature = "onnx")]
    fn load_onnx(config: EngineConfig) -> Result<Self, EngineError> {
        use crate::onnx::OnnxModelLoader;

        tracing::info!("Loading ONNX model from: {}", config.model_path);

        let model_dir = std::path::Path::new(&config.model_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));

        // Load model via ONNX loader
        let loader = OnnxModelLoader::load(&config.model_path)
            .map_err(|e| EngineError::Other(format!("ONNX load error: {}", e)))?;
        let model_config = loader.config().clone();
        let hf_config = loader.hf_config().clone();

        tracing::info!(
            "Model: {} layers, {} heads, {} hidden dim, {} ctx",
            model_config.num_layers,
            model_config.num_heads,
            model_config.hidden_size,
            model_config.max_seq_len,
        );

        let concrete_model = loader
            .build_model()
            .map_err(|e| EngineError::Other(format!("ONNX model build error: {}", e)))?;

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = if let Some(ref tok_path) = config.tokenizer_path {
            if tok_path.ends_with(".json") {
                Tokenizer::from_hf_json(tok_path)?
            } else {
                let tok_gguf = GgufFile::open(tok_path)?;
                Tokenizer::from_gguf(&tok_gguf)?
            }
        } else {
            // Look for tokenizer.json in the same directory
            let tokenizer_path = model_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                tracing::info!("Using tokenizer.json from: {}", tokenizer_path.display());
                Tokenizer::from_hf_json(&tokenizer_path)?
            } else {
                return Err(EngineError::Other(format!(
                    "No tokenizer found. ONNX models require a tokenizer.json file \
                     in the same directory as the model, or specify --tokenizer <path>. \
                     Looked for: {}",
                    tokenizer_path.display()
                )));
            }
        };
        tracing::info!("Vocabulary size: {}", tokenizer.vocab_size);

        // Select backend
        let backend: Arc<dyn Backend> = if config.use_gpu {
            Self::select_gpu_backend(&concrete_model)
        } else {
            Arc::new(crate::backend::cpu::CpuBackend::new())
        };

        let model: Box<dyn Model> = Box::new(concrete_model);

        // Infer chat template from model type
        let chat_template = ChatTemplate::detect_from_model_type(hf_config.model_type.as_deref());
        tracing::info!("Chat template: {:?}", chat_template);

        // For ONNX models, default to adding BOS
        let add_bos = true;

        let sampler_config = SamplerConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
            seed: config.seed,
            ..Default::default()
        };

        tracing::info!("Engine ready (ONNX)");

        Ok(Self {
            gguf: None,
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

    /// Select the best GPU model + backend combination.
    ///
    /// Tries GPU-only inference first (all computation on GPU, ~386× fewer
    /// host↔device transfers), then falls back to the per-op CudaBackend,
    /// then to CPU.
    #[allow(unused_variables)]
    fn select_gpu_model(
        model: crate::model::LlamaModel,
        config: &ModelConfig,
    ) -> (Arc<dyn Backend>, Box<dyn Model>) {
        // Try full GPU-only inference first (CUDA only)
        #[cfg(feature = "cuda")]
        {
            let architecture = model.architecture();
            match crate::backend::cuda::gpu_only::GpuOnlyInference::from_model(
                &model,
                config.max_seq_len,
            ) {
                Ok(gpu) => {
                    tracing::info!(
                        "Using GPU-only inference (all computation on GPU, minimal transfers)"
                    );
                    let wrapper = crate::backend::cuda::gpu_only::GpuModelWrapper::new(
                        gpu,
                        config.clone(),
                        architecture,
                    );
                    // The Backend is only needed for InferenceContext construction;
                    // computation goes through GpuModelWrapper directly.
                    return (
                        Arc::new(crate::backend::cpu::CpuBackend::new()),
                        Box::new(wrapper),
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "GPU-only inference init failed ({}), trying per-op GPU backend",
                        e
                    );
                }
            }
        }

        // Fall back to per-op GPU backend
        let backend = Self::select_gpu_backend(&model);
        (backend, Box::new(model))
    }

    /// Select the best available GPU backend.
    ///
    /// Priority: CUDA > Metal > DX12 > Vulkan > CPU fallback.
    #[allow(unused_variables)]
    pub fn select_gpu_backend(model: &crate::model::LlamaModel) -> Arc<dyn Backend> {
        // Try CUDA first (NVIDIA GPUs)
        #[cfg(feature = "cuda")]
        {
            match crate::backend::cuda::CudaBackend::new() {
                Ok(mut cuda) => {
                    tracing::info!("Using CUDA backend: {}", cuda.device_name());
                    if let Err(e) = cuda.load_model_weights(model) {
                        tracing::warn!("Failed to load GPU weights ({}), using quantized ops", e);
                    }
                    return Arc::new(cuda);
                }
                Err(e) => {
                    tracing::info!("CUDA not available ({}), trying Metal...", e);
                }
            }
        }

        // Try Metal (native macOS / Apple Silicon)
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match crate::backend::metal::MetalBackend::new() {
                Ok(metal) => {
                    tracing::info!("Using Metal backend: {}", metal.device_name());
                    return Arc::new(metal);
                }
                Err(e) => {
                    tracing::info!("Metal not available ({}), trying DX12...", e);
                }
            }
        }

        // Try DX12 (native Windows GPU compute)
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            match crate::backend::dx12::Dx12Backend::new() {
                Ok(dx12) => {
                    tracing::info!("Using DX12 backend: {}", dx12.device_name());
                    return Arc::new(dx12);
                }
                Err(e) => {
                    tracing::info!("DX12 not available ({}), trying Vulkan...", e);
                }
            }
        }

        // Try Vulkan (cross-platform: AMD, Intel, NVIDIA, etc.)
        #[cfg(feature = "vulkan")]
        {
            match crate::backend::vulkan::VulkanBackend::new() {
                Ok(vk) => {
                    tracing::info!("Using Vulkan backend: {}", vk.device_name());
                    return Arc::new(vk);
                }
                Err(e) => {
                    tracing::warn!("Vulkan not available ({}), falling back to CPU", e);
                }
            }
        }

        // Fallback message when no GPU backend is compiled
        #[cfg(not(any(
            feature = "cuda",
            feature = "vulkan",
            all(feature = "metal", target_os = "macos"),
            all(feature = "dx12", target_os = "windows")
        )))]
        {
            tracing::warn!(
                "No GPU backend compiled. Build with --features cuda, --features metal, --features dx12, or --features vulkan"
            );
        }

        Arc::new(crate::backend::cpu::CpuBackend::new())
    }

    /// Get the model configuration.
    pub fn model_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the detected chat template.
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Get the GGUF file metadata (None for ONNX-loaded models).
    pub fn gguf(&self) -> Option<&GgufFile> {
        self.gguf.as_ref()
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
            if let Some(&last) = tokens.last()
                && last == self.tokenizer.special_tokens.eos_token_id
            {
                break;
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
    pub fn generate_streaming(&self, prompt: &str, max_tokens: usize) -> GenerationStream<'_> {
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
        if let Some(&last) = self.tokens.last()
            && last == self.engine.tokenizer.special_tokens.eos_token_id
        {
            self.done = true;
            return None;
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

        // Batch-prefill: process ALL prompt tokens in a single forward pass.
        // This is dramatically faster than one-at-a-time, especially on GPU
        // backends where kernel launch overhead dominates for tiny batches.
        let eos_id = self.engine.tokenizer.special_tokens.eos_token_id;
        let mut response_text = String::new();

        if new_tokens.is_empty() {
            self.is_first_turn = false;
            return Ok(response_text);
        }

        let prefill_logits = self.engine.model.forward(&new_tokens, &mut self.ctx)?;
        let first_token = self.sampler.sample(&prefill_logits, &self.conversation_tokens);

        if first_token == eos_id {
            self.is_first_turn = false;
            return Ok(response_text);
        }

        if let Ok(text) = self.engine.tokenizer.decode(&[first_token]) {
            response_text.push_str(&text);
        }
        self.conversation_tokens.push(first_token);

        // Autoregressive decode for the remaining tokens
        for _ in 1..max_tokens {
            // Check stop patterns
            let should_stop = self
                .engine
                .chat_template
                .stop_patterns()
                .iter()
                .any(|p| response_text.contains(p));
            if should_stop {
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

            if next_token == eos_id {
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

    /// Send a message and get the full response, with a prefix injected as the
    /// start of the assistant's reply. The prefix tokens are prefilled alongside
    /// the prompt tokens so the model continues from the prefix text. The prefix
    /// is prepended to the returned string.
    ///
    /// This is useful for forcing the model to start with a particular token
    /// sequence (e.g. `{` for JSON output).
    pub fn chat_with_prefix(
        &mut self,
        message: &str,
        prefix: &str,
    ) -> Result<String, EngineError> {
        let max_tokens = self.engine.engine_config.max_tokens;

        let formatted = if self.is_first_turn {
            self.engine
                .chat_template
                .format_first_turn(&self.system_prompt, message)
        } else {
            self.engine.chat_template.format_continuation(message)
        };

        // Append prefix to the formatted prompt so it becomes part of the prefill
        let formatted_with_prefix = format!("{}{}", formatted, prefix);

        let new_tokens = self
            .engine
            .tokenizer
            .encode(&formatted_with_prefix, self.is_first_turn && self.engine.add_bos)?;

        self.ensure_context_space(new_tokens.len(), max_tokens);
        self.conversation_tokens.extend(&new_tokens);

        let eos_id = self.engine.tokenizer.special_tokens.eos_token_id;
        let mut response_text = prefix.to_string();

        if new_tokens.is_empty() {
            self.is_first_turn = false;
            return Ok(response_text);
        }

        let prefill_logits = self.engine.model.forward(&new_tokens, &mut self.ctx)?;
        let first_token = self.sampler.sample(&prefill_logits, &self.conversation_tokens);

        if first_token == eos_id {
            self.is_first_turn = false;
            return Ok(response_text);
        }

        if let Ok(text) = self.engine.tokenizer.decode(&[first_token]) {
            response_text.push_str(&text);
        }
        self.conversation_tokens.push(first_token);

        for _ in 1..max_tokens {
            let should_stop = self
                .engine
                .chat_template
                .stop_patterns()
                .iter()
                .any(|p| response_text.contains(p));
            if should_stop {
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

            if next_token == eos_id {
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
    pub fn chat_streaming(&mut self, message: &str) -> Result<ChatStream<'_>, EngineError> {
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

        // Add new tokens to conversation
        self.conversation_tokens.extend(&new_tokens);

        // Batch-prefill all prompt tokens in a single forward pass.
        let prefill_logits = if !new_tokens.is_empty() {
            Some(self.engine.model.forward(&new_tokens, &mut self.ctx)?)
        } else {
            None
        };

        self.is_first_turn = false;

        Ok(ChatStream {
            chat_engine: self,
            remaining: max_tokens,
            done: false,
            accumulated: String::new(),
            prefill_logits,
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
    /// Logits from the batched prefill pass; consumed on the first `next()` call.
    prefill_logits: Option<crate::tensor::Tensor>,
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

        // On the first call, use the prefill logits (no extra forward pass).
        // On subsequent calls, run the standard single-token decode.
        let logits = if let Some(prefill) = self.prefill_logits.take() {
            prefill
        } else {
            let last_token = *self.chat_engine.conversation_tokens.last().unwrap_or(
                &self
                    .chat_engine
                    .engine
                    .tokenizer
                    .special_tokens
                    .bos_token_id,
            );

            match self
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
            }
        };

        let next_token = self
            .chat_engine
            .sampler
            .sample(&logits, &self.chat_engine.conversation_tokens);

        // Check for EOS
        if next_token
            == self
                .chat_engine
                .engine
                .tokenizer
                .special_tokens
                .eos_token_id
        {
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
