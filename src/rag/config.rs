//! RAG configuration with TOML and environment variable support
//!
//! Configuration precedence (highest to lowest):
//! 1. Explicit function arguments
//! 2. Environment variables
//! 3. TOML config file
//! 4. Default values

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for RAG/pgvector connection
///
/// # Example TOML Configuration
///
/// ```toml
/// # rag.toml
/// [database]
/// connection_string = "postgres://user:pass@localhost:5432/mydb"
/// pool_size = 10
///
/// [embeddings]
/// table_name = "embeddings"
/// dimension = 384
///
/// [search]
/// max_results = 5
/// min_similarity = 0.5
/// distance_metric = "cosine"  # cosine, l2, or inner_product
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct RagConfig {
    /// Database configuration
    #[serde(default)]
    pub database: DatabaseConfig,

    /// Embeddings configuration
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,

    /// Search configuration
    #[serde(default)]
    pub search: SearchConfig,
}

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    /// PostgreSQL connection string
    /// Format: postgres://user:password@host:port/database
    #[serde(default)]
    pub connection_string: String,

    /// Connection pool size
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,

    /// Connection timeout in seconds
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
}

/// Embeddings table configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingsConfig {
    /// Name of the embeddings table
    #[serde(default = "default_table_name")]
    pub table_name: String,

    /// Embedding vector dimension
    #[serde(default = "default_embedding_dim")]
    pub dimension: usize,
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Maximum number of results to return
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Minimum similarity score (0.0 - 1.0)
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,

    /// Distance metric for similarity search
    #[serde(default)]
    pub distance_metric: DistanceMetric,
}

fn default_table_name() -> String {
    "embeddings".to_string()
}

fn default_embedding_dim() -> usize {
    384
}

fn default_max_results() -> usize {
    5
}

fn default_min_similarity() -> f32 {
    0.5
}

fn default_pool_size() -> usize {
    10
}

fn default_connect_timeout() -> u64 {
    30
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            connection_string: String::new(),
            pool_size: default_pool_size(),
            connect_timeout_secs: default_connect_timeout(),
        }
    }
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            table_name: default_table_name(),
            dimension: default_embedding_dim(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: default_max_results(),
            min_similarity: default_min_similarity(),
            distance_metric: DistanceMetric::default(),
        }
    }
}

/// Distance metric for vector similarity search
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine similarity (default, best for normalized embeddings)
    #[default]
    Cosine,
    /// L2 (Euclidean) distance
    L2,
    /// Inner product
    InnerProduct,
}

impl DistanceMetric {
    /// Get the pgvector operator for this metric
    pub fn operator(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::L2 => "<->",
            DistanceMetric::InnerProduct => "<#>",
        }
    }

    /// Get the index operator class for this metric
    pub fn index_ops(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "vector_cosine_ops",
            DistanceMetric::L2 => "vector_l2_ops",
            DistanceMetric::InnerProduct => "vector_ip_ops",
        }
    }
}

impl RagConfig {
    /// Create a new RAG configuration with just a connection string
    pub fn new(connection_string: impl Into<String>) -> Self {
        Self {
            database: DatabaseConfig {
                connection_string: connection_string.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Load configuration from a TOML file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, super::RagError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            super::RagError::ConfigError(format!("Failed to read config file: {}", e))
        })?;

        toml::from_str(&content)
            .map_err(|e| super::RagError::ConfigError(format!("Failed to parse TOML: {}", e)))
    }

    /// Load configuration from environment variables
    ///
    /// Supported variables:
    /// - RAG_DATABASE_URL / DATABASE_URL - Connection string
    /// - RAG_POOL_SIZE - Connection pool size
    /// - RAG_TABLE_NAME - Embeddings table name
    /// - RAG_EMBEDDING_DIM - Embedding dimension
    /// - RAG_MAX_RESULTS - Maximum search results
    /// - RAG_MIN_SIMILARITY - Minimum similarity threshold
    /// - RAG_DISTANCE_METRIC - Distance metric (cosine, l2, inner_product)
    pub fn from_env() -> Result<Self, super::RagError> {
        let mut config = Self::default();

        // Database
        if let Ok(url) =
            std::env::var("RAG_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        {
            config.database.connection_string = url;
        }

        if let Ok(size) = std::env::var("RAG_POOL_SIZE") {
            config.database.pool_size = size
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_POOL_SIZE".into()))?;
        }

        // Embeddings
        if let Ok(table) = std::env::var("RAG_TABLE_NAME") {
            config.embeddings.table_name = table;
        }

        if let Ok(dim) = std::env::var("RAG_EMBEDDING_DIM") {
            config.embeddings.dimension = dim
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_EMBEDDING_DIM".into()))?;
        }

        // Search
        if let Ok(max) = std::env::var("RAG_MAX_RESULTS") {
            config.search.max_results = max
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_MAX_RESULTS".into()))?;
        }

        if let Ok(min) = std::env::var("RAG_MIN_SIMILARITY") {
            config.search.min_similarity = min
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_MIN_SIMILARITY".into()))?;
        }

        if let Ok(metric) = std::env::var("RAG_DISTANCE_METRIC") {
            config.search.distance_metric = match metric.to_lowercase().as_str() {
                "cosine" => DistanceMetric::Cosine,
                "l2" | "euclidean" => DistanceMetric::L2,
                "inner_product" | "ip" | "dot" => DistanceMetric::InnerProduct,
                _ => {
                    return Err(super::RagError::ConfigError(format!(
                        "Invalid RAG_DISTANCE_METRIC: {}. Use: cosine, l2, or inner_product",
                        metric
                    )));
                }
            };
        }

        Ok(config)
    }

    /// Load configuration with precedence: file -> env -> defaults
    ///
    /// If a config file path is provided and exists, it's loaded first.
    /// Environment variables override file settings.
    pub fn load(config_path: Option<impl AsRef<Path>>) -> Result<Self, super::RagError> {
        // Start with defaults
        let mut config = Self::default();

        // Try to load from file if provided
        if let Some(path) = config_path {
            if path.as_ref().exists() {
                config = Self::from_file(path)?;
            }
        } else {
            // Try default config locations
            let default_paths = ["rag.toml", "config/rag.toml", ".rag.toml"];

            for path in &default_paths {
                if Path::new(path).exists() {
                    config = Self::from_file(path)?;
                    break;
                }
            }
        }

        // Override with environment variables
        config.apply_env()?;

        Ok(config)
    }

    /// Apply environment variable overrides to existing config
    pub fn apply_env(&mut self) -> Result<(), super::RagError> {
        if let Ok(url) =
            std::env::var("RAG_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        {
            self.database.connection_string = url;
        }

        if let Ok(size) = std::env::var("RAG_POOL_SIZE") {
            self.database.pool_size = size
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_POOL_SIZE".into()))?;
        }

        if let Ok(table) = std::env::var("RAG_TABLE_NAME") {
            self.embeddings.table_name = table;
        }

        if let Ok(dim) = std::env::var("RAG_EMBEDDING_DIM") {
            self.embeddings.dimension = dim
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_EMBEDDING_DIM".into()))?;
        }

        if let Ok(max) = std::env::var("RAG_MAX_RESULTS") {
            self.search.max_results = max
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_MAX_RESULTS".into()))?;
        }

        if let Ok(min) = std::env::var("RAG_MIN_SIMILARITY") {
            self.search.min_similarity = min
                .parse()
                .map_err(|_| super::RagError::ConfigError("Invalid RAG_MIN_SIMILARITY".into()))?;
        }

        if let Ok(metric) = std::env::var("RAG_DISTANCE_METRIC") {
            self.search.distance_metric = match metric.to_lowercase().as_str() {
                "cosine" => DistanceMetric::Cosine,
                "l2" | "euclidean" => DistanceMetric::L2,
                "inner_product" | "ip" | "dot" => DistanceMetric::InnerProduct,
                _ => {
                    return Err(super::RagError::ConfigError(format!(
                        "Invalid RAG_DISTANCE_METRIC: {}",
                        metric
                    )));
                }
            };
        }

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), super::RagError> {
        if self.database.connection_string.is_empty() {
            return Err(super::RagError::ConfigError(
                "Database connection string is required. Set RAG_DATABASE_URL or provide in config file.".into()
            ));
        }

        if self.embeddings.dimension == 0 {
            return Err(super::RagError::ConfigError(
                "Embedding dimension must be greater than 0".into(),
            ));
        }

        if self.search.min_similarity < 0.0 || self.search.min_similarity > 1.0 {
            return Err(super::RagError::ConfigError(
                "min_similarity must be between 0.0 and 1.0".into(),
            ));
        }

        Ok(())
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), super::RagError> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            super::RagError::ConfigError(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path, content).map_err(|e| {
            super::RagError::ConfigError(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    // Builder-style methods for backward compatibility

    /// Set the embeddings table name
    pub fn with_table(mut self, table_name: impl Into<String>) -> Self {
        self.embeddings.table_name = table_name.into();
        self
    }

    /// Set the embedding dimension
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embeddings.dimension = dim;
        self
    }

    /// Set the maximum number of search results
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.search.max_results = max;
        self
    }

    /// Set the minimum similarity threshold
    pub fn with_min_similarity(mut self, min: f32) -> Self {
        self.search.min_similarity = min.clamp(0.0, 1.0);
        self
    }

    /// Set the distance metric
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.search.distance_metric = metric;
        self
    }

    /// Set the connection pool size
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.database.pool_size = size;
        self
    }

    // Accessors for backward compatibility

    pub fn connection_string(&self) -> &str {
        &self.database.connection_string
    }

    pub fn table_name(&self) -> &str {
        &self.embeddings.table_name
    }

    pub fn embedding_dim(&self) -> usize {
        self.embeddings.dimension
    }

    pub fn max_results(&self) -> usize {
        self.search.max_results
    }

    pub fn min_similarity(&self) -> f32 {
        self.search.min_similarity
    }

    pub fn distance_metric(&self) -> DistanceMetric {
        self.search.distance_metric
    }

    pub fn pool_size(&self) -> usize {
        self.database.pool_size
    }
}

/// Generate an example configuration file
pub fn example_config() -> &'static str {
    r#"# RAG Configuration
# This file configures the Retrieval-Augmented Generation system

[database]
# PostgreSQL connection string (required)
# Can also be set via RAG_DATABASE_URL or DATABASE_URL environment variable
connection_string = "postgres://user:password@localhost:5432/mydb"

# Connection pool size
pool_size = 10

# Connection timeout in seconds
connect_timeout_secs = 30

[embeddings]
# Name of the table storing embeddings
table_name = "embeddings"

# Dimension of embedding vectors (must match your embedding model)
# Common values:
#   - 384: all-MiniLM-L6-v2, all-MiniLM-L12-v2
#   - 768: all-mpnet-base-v2, BERT-base
#   - 1024: BERT-large
#   - 1536: OpenAI text-embedding-ada-002
#   - 3072: OpenAI text-embedding-3-large
dimension = 384

[search]
# Maximum number of results to return from similarity search
max_results = 5

# Minimum similarity score (0.0 to 1.0) for results to be included
min_similarity = 0.5

# Distance metric for similarity search
# Options: "cosine" (default), "l2", "inner_product"
distance_metric = "cosine"
"#
}
