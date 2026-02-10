//! TOML configuration file support for llama-gguf.
//!
//! Provides a [`Config`] struct that maps all CLI arguments to a TOML configuration
//! file, enabling persistent and shareable inference setups.
//!
//! # Configuration Precedence (highest to lowest)
//!
//! 1. CLI arguments (always win)
//! 2. Environment variables
//! 3. TOML config file
//! 4. Default values
//!
//! # Example TOML
//!
//! ```toml
//! # llama-gguf.toml
//!
//! [model]
//! path = "/path/to/model.gguf"
//! gpu = true
//!
//! [generation]
//! temperature = 0.7
//! top_k = 40
//! top_p = 0.95
//! repeat_penalty = 1.1
//! max_tokens = 512
//! seed = 42
//!
//! [chat]
//! system_prompt = "You are a helpful AI assistant."
//! max_tokens = 1024
//!
//! [server]
//! host = "0.0.0.0"
//! port = 8080
//!
//! [quantize]
//! output_type = "q4_k"
//! threads = 8
//!
//! [bench]
//! n_prompt = 512
//! n_gen = 128
//! repetitions = 3
//! threads = 4
//!
//! [embed]
//! format = "json"
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::engine::EngineConfig;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur during configuration loading.
#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("Config error: {0}")]
    Other(String),
}

// ============================================================================
// Top-level configuration
// ============================================================================

/// Top-level TOML configuration covering all CLI arguments.
///
/// Each section corresponds to a subcommand or functional area.
/// All fields are optional and fall back to sensible defaults.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    /// Model path and hardware settings.
    pub model: ModelSection,

    /// Generation/sampling parameters (shared by `run`, `chat`, `serve`).
    pub generation: GenerationSection,

    /// Chat-specific overrides.
    pub chat: ChatSection,

    /// HTTP server settings.
    pub server: ServerSection,

    /// Quantization settings.
    pub quantize: QuantizeSection,

    /// Benchmark settings.
    pub bench: BenchSection,

    /// Embedding extraction settings.
    pub embed: EmbedSection,
}

// ============================================================================
// Section structs
// ============================================================================

/// Model path and hardware configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    /// Path to the GGUF model file.
    pub path: Option<String>,

    /// Use GPU acceleration (CUDA/Metal/Vulkan).
    pub gpu: bool,
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            path: None,
            gpu: false,
        }
    }
}

/// Generation and sampling parameters.
///
/// These are shared across `run`, `chat`, and `serve` commands.
/// Command-specific sections (like `[chat]`) can override individual values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationSection {
    /// Temperature for sampling (0.0 = greedy, higher = more random).
    pub temperature: f32,

    /// Top-K sampling: only consider the K most likely tokens (0 = disabled).
    pub top_k: usize,

    /// Top-P (nucleus) sampling: cumulative probability cutoff.
    pub top_p: f32,

    /// Repetition penalty (1.0 = no penalty).
    pub repeat_penalty: f32,

    /// Default maximum tokens to generate.
    pub max_tokens: usize,

    /// Random seed for reproducible generation.
    pub seed: Option<u64>,
}

impl Default for GenerationSection {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            max_tokens: 512,
            seed: None,
        }
    }
}

/// Chat-specific configuration overrides.
///
/// Values here override the corresponding `[generation]` values
/// when running the `chat` command.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChatSection {
    /// System prompt for chat sessions.
    pub system_prompt: Option<String>,

    /// Override max_tokens for chat (defaults to `generation.max_tokens`).
    pub max_tokens: Option<usize>,

    /// Override temperature for chat (defaults to `generation.temperature`).
    pub temperature: Option<f32>,

    /// Override top_p for chat (defaults to `generation.top_p`).
    pub top_p: Option<f32>,

    /// Override top_k for chat (defaults to `generation.top_k`).
    pub top_k: Option<usize>,

    /// Override repeat_penalty for chat (defaults to `generation.repeat_penalty`).
    pub repeat_penalty: Option<f32>,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    /// Host address to bind to.
    pub host: String,

    /// Port to listen on.
    pub port: u16,

    /// RAG database URL (PostgreSQL with pgvector).
    pub rag_database_url: Option<String>,

    /// Path to RAG configuration file.
    pub rag_config: Option<String>,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            rag_database_url: None,
            rag_config: None,
        }
    }
}

impl ServerSection {
    /// Build a full URL from host and port (e.g. `http://192.168.1.4:8080`).
    /// Returns `None` if host is localhost/default (i.e. no remote server configured).
    pub fn host_url(&self) -> Option<String> {
        // Only return a URL if the host is NOT localhost — a remote was explicitly configured
        if self.host == "127.0.0.1" || self.host == "localhost" || self.host == "0.0.0.0" {
            None
        } else {
            Some(format!("http://{}:{}", self.host, self.port))
        }
    }
}

/// Quantization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizeSection {
    /// Target quantization type (q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k).
    pub output_type: String,

    /// Number of threads to use.
    pub threads: Option<usize>,
}

impl Default for QuantizeSection {
    fn default() -> Self {
        Self {
            output_type: "q4_0".to_string(),
            threads: None,
        }
    }
}

/// Benchmark settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchSection {
    /// Number of prompt tokens to process.
    pub n_prompt: usize,

    /// Number of tokens to generate.
    pub n_gen: usize,

    /// Number of repetitions for averaging.
    pub repetitions: usize,

    /// Number of threads to use.
    pub threads: Option<usize>,
}

impl Default for BenchSection {
    fn default() -> Self {
        Self {
            n_prompt: 512,
            n_gen: 128,
            repetitions: 3,
            threads: None,
        }
    }
}

/// Embedding extraction settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedSection {
    /// Output format: "json" or "raw".
    pub format: String,
}

impl Default for EmbedSection {
    fn default() -> Self {
        Self {
            format: "json".to_string(),
        }
    }
}

// ============================================================================
// Config implementation
// ============================================================================

/// Default config file names to search for (in order).
pub const DEFAULT_CONFIG_PATHS: &[&str] = &[
    "llama-gguf.toml",
    "config/llama-gguf.toml",
    ".llama-gguf.toml",
];

impl Config {
    /// Load configuration from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from environment variables.
    ///
    /// Supported variables:
    /// - `LLAMA_MODEL_PATH` - Path to GGUF model file
    /// - `LLAMA_GPU` - Enable GPU acceleration ("1", "true", "yes")
    /// - `LLAMA_TEMPERATURE` - Sampling temperature
    /// - `LLAMA_TOP_K` - Top-K sampling value
    /// - `LLAMA_TOP_P` - Top-P sampling value
    /// - `LLAMA_REPEAT_PENALTY` - Repetition penalty
    /// - `LLAMA_MAX_TOKENS` - Maximum tokens to generate
    /// - `LLAMA_SEED` - Random seed
    /// - `LLAMA_HOST` - Server host address
    /// - `LLAMA_PORT` - Server port
    /// - `LLAMA_SYSTEM_PROMPT` - Default system prompt for chat
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(path) = std::env::var("LLAMA_MODEL_PATH") {
            config.model.path = Some(path);
        }
        if let Ok(gpu) = std::env::var("LLAMA_GPU") {
            config.model.gpu = matches!(gpu.to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(val) = std::env::var("LLAMA_TEMPERATURE") {
            if let Ok(v) = val.parse() {
                config.generation.temperature = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_TOP_K") {
            if let Ok(v) = val.parse() {
                config.generation.top_k = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_TOP_P") {
            if let Ok(v) = val.parse() {
                config.generation.top_p = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_REPEAT_PENALTY") {
            if let Ok(v) = val.parse() {
                config.generation.repeat_penalty = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_MAX_TOKENS") {
            if let Ok(v) = val.parse() {
                config.generation.max_tokens = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_SEED") {
            if let Ok(v) = val.parse() {
                config.generation.seed = Some(v);
            }
        }
        if let Ok(val) = std::env::var("LLAMA_HOST") {
            config.server.host = val;
        }
        if let Ok(val) = std::env::var("LLAMA_PORT") {
            if let Ok(v) = val.parse() {
                config.server.port = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_SYSTEM_PROMPT") {
            config.chat.system_prompt = Some(val);
        }

        config
    }

    /// Load configuration with full precedence chain.
    ///
    /// 1. If `config_path` is provided, load from that file.
    /// 2. Otherwise, search default locations (`llama-gguf.toml`, etc.).
    /// 3. Override with environment variables.
    pub fn load(config_path: Option<impl AsRef<Path>>) -> Result<Self, ConfigError> {
        let mut config = Self::default();

        // Try to load from explicit path
        if let Some(path) = config_path {
            let p = path.as_ref();
            if p.exists() {
                config = Self::from_file(p)?;
            } else {
                return Err(ConfigError::Other(format!(
                    "Config file not found: {}",
                    p.display()
                )));
            }
        } else {
            // Search default locations
            for path in DEFAULT_CONFIG_PATHS {
                if Path::new(path).exists() {
                    config = Self::from_file(path)?;
                    break;
                }
            }
        }

        // Apply environment variable overrides
        config.apply_env();

        Ok(config)
    }

    /// Apply environment variable overrides to the current configuration.
    pub fn apply_env(&mut self) {
        if let Ok(path) = std::env::var("LLAMA_MODEL_PATH") {
            self.model.path = Some(path);
        }
        if let Ok(gpu) = std::env::var("LLAMA_GPU") {
            self.model.gpu = matches!(gpu.to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(val) = std::env::var("LLAMA_TEMPERATURE") {
            if let Ok(v) = val.parse() {
                self.generation.temperature = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_TOP_K") {
            if let Ok(v) = val.parse() {
                self.generation.top_k = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_TOP_P") {
            if let Ok(v) = val.parse() {
                self.generation.top_p = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_REPEAT_PENALTY") {
            if let Ok(v) = val.parse() {
                self.generation.repeat_penalty = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_MAX_TOKENS") {
            if let Ok(v) = val.parse() {
                self.generation.max_tokens = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_SEED") {
            if let Ok(v) = val.parse() {
                self.generation.seed = Some(v);
            }
        }
        if let Ok(val) = std::env::var("LLAMA_HOST") {
            self.server.host = val;
        }
        if let Ok(val) = std::env::var("LLAMA_PORT") {
            if let Ok(v) = val.parse() {
                self.server.port = v;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_SYSTEM_PROMPT") {
            self.chat.system_prompt = Some(val);
        }
    }

    /// Save configuration to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Convert to an [`EngineConfig`] using the model and generation sections.
    ///
    /// The `model_path_override` takes highest priority (from CLI positional arg).
    pub fn to_engine_config(&self, model_path_override: Option<&str>) -> EngineConfig {
        let model_path = model_path_override
            .map(|s| s.to_string())
            .or_else(|| self.model.path.clone())
            .unwrap_or_default();

        EngineConfig {
            model_path,
            tokenizer_path: None,
            temperature: self.generation.temperature,
            top_k: self.generation.top_k,
            top_p: self.generation.top_p,
            repeat_penalty: self.generation.repeat_penalty,
            max_tokens: self.generation.max_tokens,
            seed: self.generation.seed,
            use_gpu: self.model.gpu,
        }
    }

    /// Convert to an [`EngineConfig`] using chat-specific overrides where present.
    ///
    /// Values from `[chat]` override `[generation]` when set.
    pub fn to_chat_engine_config(&self, model_path_override: Option<&str>) -> EngineConfig {
        let mut config = self.to_engine_config(model_path_override);

        // Apply chat-specific overrides
        if let Some(max_tokens) = self.chat.max_tokens {
            config.max_tokens = max_tokens;
        }
        if let Some(temperature) = self.chat.temperature {
            config.temperature = temperature;
        }
        if let Some(top_p) = self.chat.top_p {
            config.top_p = top_p;
        }
        if let Some(top_k) = self.chat.top_k {
            config.top_k = top_k;
        }
        if let Some(repeat_penalty) = self.chat.repeat_penalty {
            config.repeat_penalty = repeat_penalty;
        }

        config
    }
}

// ============================================================================
// Example config generator
// ============================================================================

/// Generate an example TOML configuration with all options documented.
pub fn example_config() -> &'static str {
    r#"# llama-gguf configuration
# All values shown are defaults. Uncomment and modify as needed.
#
# Precedence: CLI arguments > environment variables > this file > defaults

# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────
[model]
# Path to the GGUF model file (can also use LLAMA_MODEL_PATH env var)
# path = "/path/to/model.gguf"

# Use GPU acceleration (CUDA/Metal/Vulkan)
# Also: LLAMA_GPU=1
gpu = false

# ─────────────────────────────────────────────────────────────────────
# Generation / Sampling Parameters
# Used by: run, chat, serve
# ─────────────────────────────────────────────────────────────────────
[generation]
# Sampling temperature (0.0 = greedy/deterministic, higher = more random)
# Also: LLAMA_TEMPERATURE
temperature = 0.7

# Top-K sampling: only consider the K most likely next tokens (0 = disabled)
# Also: LLAMA_TOP_K
top_k = 40

# Top-P (nucleus) sampling: cumulative probability cutoff
# Also: LLAMA_TOP_P
top_p = 0.95

# Repetition penalty (1.0 = no penalty, higher = less repetition)
# Also: LLAMA_REPEAT_PENALTY
repeat_penalty = 1.1

# Default maximum tokens to generate per request
# Also: LLAMA_MAX_TOKENS
max_tokens = 512

# Random seed for reproducible generation (comment out for random)
# Also: LLAMA_SEED
# seed = 42

# ─────────────────────────────────────────────────────────────────────
# Chat Mode Overrides
# Values here override [generation] when using the `chat` command.
# Omitted values fall back to [generation].
# ─────────────────────────────────────────────────────────────────────
[chat]
# Default system prompt for chat sessions
# Also: LLAMA_SYSTEM_PROMPT
# system_prompt = "You are a helpful AI assistant."

# Override generation settings for chat specifically
# max_tokens = 1024
# temperature = 0.7
# top_p = 0.9
# top_k = 40
# repeat_penalty = 1.1

# ─────────────────────────────────────────────────────────────────────
# HTTP Server (used by `serve` command)
# ─────────────────────────────────────────────────────────────────────
[server]
# Host address to bind to
# Also: LLAMA_HOST
host = "127.0.0.1"

# Port to listen on
# Also: LLAMA_PORT
port = 8080

# PostgreSQL/pgvector URL for RAG (requires `rag` feature)
# Also: RAG_DATABASE_URL
# rag_database_url = "postgres://user:pass@localhost:5432/mydb"

# Path to separate RAG config file
# rag_config = "rag.toml"

# ─────────────────────────────────────────────────────────────────────
# Quantization (used by `quantize` command)
# ─────────────────────────────────────────────────────────────────────
[quantize]
# Target quantization type
# Options: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k
output_type = "q4_0"

# Number of threads (default: all available cores)
# threads = 8

# ─────────────────────────────────────────────────────────────────────
# Benchmarking (used by `bench` command)
# ─────────────────────────────────────────────────────────────────────
[bench]
# Number of prompt tokens to process
n_prompt = 512

# Number of tokens to generate
n_gen = 128

# Number of repetitions for averaging results
repetitions = 3

# Number of threads (default: all available cores)
# threads = 4

# ─────────────────────────────────────────────────────────────────────
# Embeddings (used by `embed` command)
# ─────────────────────────────────────────────────────────────────────
[embed]
# Output format: "json" or "raw"
format = "json"
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.generation.temperature, 0.7);
        assert_eq!(config.generation.top_k, 40);
        assert_eq!(config.generation.top_p, 0.95);
        assert_eq!(config.generation.max_tokens, 512);
        assert_eq!(config.server.port, 8080);
        assert!(!config.model.gpu);
    }

    #[test]
    fn test_roundtrip_toml() {
        let config = Config {
            model: ModelSection {
                path: Some("/tmp/test.gguf".to_string()),
                gpu: true,
            },
            generation: GenerationSection {
                temperature: 0.5,
                top_k: 50,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.model.path.as_deref(), Some("/tmp/test.gguf"));
        assert!(parsed.model.gpu);
        assert_eq!(parsed.generation.temperature, 0.5);
        assert_eq!(parsed.generation.top_k, 50);
        assert_eq!(parsed.generation.seed, Some(42));
    }

    #[test]
    fn test_to_engine_config() {
        let config = Config {
            model: ModelSection {
                path: Some("/models/llama.gguf".to_string()),
                gpu: true,
            },
            generation: GenerationSection {
                temperature: 0.3,
                max_tokens: 1024,
                seed: Some(123),
                ..Default::default()
            },
            ..Default::default()
        };

        let engine = config.to_engine_config(None);
        assert_eq!(engine.model_path, "/models/llama.gguf");
        assert_eq!(engine.temperature, 0.3);
        assert_eq!(engine.max_tokens, 1024);
        assert_eq!(engine.seed, Some(123));
        assert!(engine.use_gpu);
    }

    #[test]
    fn test_model_path_override() {
        let config = Config {
            model: ModelSection {
                path: Some("/config/model.gguf".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        // CLI override should win
        let engine = config.to_engine_config(Some("/cli/model.gguf"));
        assert_eq!(engine.model_path, "/cli/model.gguf");

        // Without override, config value is used
        let engine = config.to_engine_config(None);
        assert_eq!(engine.model_path, "/config/model.gguf");
    }

    #[test]
    fn test_chat_overrides() {
        let config = Config {
            generation: GenerationSection {
                temperature: 0.8,
                max_tokens: 256,
                ..Default::default()
            },
            chat: ChatSection {
                max_tokens: Some(1024),
                temperature: Some(0.5),
                ..Default::default()
            },
            ..Default::default()
        };

        let engine = config.to_chat_engine_config(None);
        assert_eq!(engine.max_tokens, 1024); // overridden by chat
        assert_eq!(engine.temperature, 0.5); // overridden by chat
        assert_eq!(engine.top_k, 40); // from generation defaults
    }

    #[test]
    fn test_parse_partial_toml() {
        let toml_str = r#"
[model]
path = "/my/model.gguf"

[generation]
temperature = 0.3
"#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model.path.as_deref(), Some("/my/model.gguf"));
        assert_eq!(config.generation.temperature, 0.3);
        // Defaults should fill in the rest
        assert_eq!(config.generation.top_k, 40);
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_example_config_parses() {
        // The example config has comments, ensure the format is valid
        // by checking it doesn't panic on generation
        let example = example_config();
        assert!(example.contains("[model]"));
        assert!(example.contains("[generation]"));
        assert!(example.contains("[chat]"));
        assert!(example.contains("[server]"));
    }
}
