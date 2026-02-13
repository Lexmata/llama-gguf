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

    /// Index type: "hnsw" (default), "ivfflat", or "none"
    #[serde(default = "default_index_type_str")]
    pub index_type: String,

    /// HNSW parameter: max number of connections per layer
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: u16,

    /// HNSW parameter: size of the dynamic candidate list for construction
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: u16,

    /// IVF-Flat parameter: number of inverted lists
    #[serde(default = "default_ivfflat_lists")]
    pub ivfflat_lists: u16,
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

    /// Search type: semantic (default) or hybrid
    #[serde(default)]
    pub search_type: SearchType,

    /// Reciprocal Rank Fusion k parameter for hybrid search
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,

    /// Oversampling factor for hybrid search candidate retrieval
    #[serde(default = "default_hybrid_oversampling")]
    pub hybrid_oversampling: u32,

    /// PostgreSQL text search language configuration
    #[serde(default = "default_text_search_language")]
    pub text_search_language: String,
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

fn default_index_type_str() -> String {
    "hnsw".to_string()
}

fn default_hnsw_m() -> u16 {
    16
}

fn default_hnsw_ef_construction() -> u16 {
    64
}

fn default_ivfflat_lists() -> u16 {
    100
}

fn default_rrf_k() -> u32 {
    60
}

fn default_hybrid_oversampling() -> u32 {
    2
}

fn default_text_search_language() -> String {
    "english".to_string()
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
            index_type: default_index_type_str(),
            hnsw_m: default_hnsw_m(),
            hnsw_ef_construction: default_hnsw_ef_construction(),
            ivfflat_lists: default_ivfflat_lists(),
        }
    }
}

impl EmbeddingsConfig {
    /// Construct an `IndexType` from the flat config fields.
    pub fn index_type(&self) -> IndexType {
        match self.index_type.to_lowercase().as_str() {
            "hnsw" => IndexType::Hnsw {
                m: self.hnsw_m,
                ef_construction: self.hnsw_ef_construction,
            },
            "ivfflat" => IndexType::IvfFlat {
                lists: self.ivfflat_lists,
            },
            "none" => IndexType::None,
            _ => IndexType::default(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: default_max_results(),
            min_similarity: default_min_similarity(),
            distance_metric: DistanceMetric::default(),
            search_type: SearchType::default(),
            rrf_k: default_rrf_k(),
            hybrid_oversampling: default_hybrid_oversampling(),
            text_search_language: default_text_search_language(),
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

/// Type of vector index to create on the embeddings table
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexType {
    /// HNSW index (recommended for most workloads)
    Hnsw {
        /// Maximum number of connections per layer (default: 16)
        m: u16,
        /// Size of the dynamic candidate list for construction (default: 64)
        ef_construction: u16,
    },
    /// IVF-Flat index (faster build, good for large datasets)
    IvfFlat {
        /// Number of inverted lists (default: 100)
        lists: u16,
    },
    /// No index (brute-force scan)
    None,
}

impl Default for IndexType {
    fn default() -> Self {
        Self::Hnsw {
            m: 16,
            ef_construction: 64,
        }
    }
}

impl IndexType {
    /// Generate the SQL components needed to create this index.
    ///
    /// Returns `(index_method, ops_class, with_clause)` where:
    /// - `index_method` is the PostgreSQL index method (e.g. `"hnsw"`)
    /// - `ops_class` is the operator class passed in via `ops`
    /// - `with_clause` is the `WITH (...)` parameters string
    ///
    /// For `IndexType::None`, all components are empty strings.
    pub fn index_sql<'a>(&self, ops: &'a str) -> (&'static str, &'a str, String) {
        match self {
            IndexType::Hnsw { m, ef_construction } => (
                "hnsw",
                ops,
                format!("WITH (m = {}, ef_construction = {})", m, ef_construction),
            ),
            IndexType::IvfFlat { lists } => (
                "ivfflat",
                ops,
                format!("WITH (lists = {})", lists),
            ),
            IndexType::None => ("", "", String::new()),
        }
    }
}

/// Type of search to perform
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SearchType {
    /// Pure semantic/vector search
    #[default]
    Semantic,
    /// Hybrid: combine semantic and keyword search
    Hybrid,
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
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| super::RagError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
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
        if let Ok(url) = std::env::var("RAG_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL")) {
            config.database.connection_string = url;
        }
        
        if let Ok(size) = std::env::var("RAG_POOL_SIZE") {
            config.database.pool_size = size.parse().map_err(|_| 
                super::RagError::ConfigError("Invalid RAG_POOL_SIZE".into()))?;
        }
        
        // Embeddings
        if let Ok(table) = std::env::var("RAG_TABLE_NAME") {
            config.embeddings.table_name = table;
        }
        
        if let Ok(dim) = std::env::var("RAG_EMBEDDING_DIM") {
            config.embeddings.dimension = dim.parse().map_err(|_| 
                super::RagError::ConfigError("Invalid RAG_EMBEDDING_DIM".into()))?;
        }
        
        // Search
        if let Ok(max) = std::env::var("RAG_MAX_RESULTS") {
            config.search.max_results = max.parse().map_err(|_|
                super::RagError::ConfigError("Invalid RAG_MAX_RESULTS".into()))?;
        }
        
        if let Ok(min) = std::env::var("RAG_MIN_SIMILARITY") {
            config.search.min_similarity = min.parse().map_err(|_|
                super::RagError::ConfigError("Invalid RAG_MIN_SIMILARITY".into()))?;
        }
        
        if let Ok(metric) = std::env::var("RAG_DISTANCE_METRIC") {
            config.search.distance_metric = match metric.to_lowercase().as_str() {
                "cosine" => DistanceMetric::Cosine,
                "l2" | "euclidean" => DistanceMetric::L2,
                "inner_product" | "ip" | "dot" => DistanceMetric::InnerProduct,
                _ => return Err(super::RagError::ConfigError(
                    format!("Invalid RAG_DISTANCE_METRIC: {}. Use: cosine, l2, or inner_product", metric)
                )),
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
            let default_paths = [
                "rag.toml",
                "config/rag.toml",
                ".rag.toml",
            ];
            
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
        if let Ok(url) = std::env::var("RAG_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL")) {
            self.database.connection_string = url;
        }
        
        if let Ok(size) = std::env::var("RAG_POOL_SIZE") {
            self.database.pool_size = size.parse().map_err(|_| 
                super::RagError::ConfigError("Invalid RAG_POOL_SIZE".into()))?;
        }
        
        if let Ok(table) = std::env::var("RAG_TABLE_NAME") {
            self.embeddings.table_name = table;
        }
        
        if let Ok(dim) = std::env::var("RAG_EMBEDDING_DIM") {
            self.embeddings.dimension = dim.parse().map_err(|_| 
                super::RagError::ConfigError("Invalid RAG_EMBEDDING_DIM".into()))?;
        }
        
        if let Ok(max) = std::env::var("RAG_MAX_RESULTS") {
            self.search.max_results = max.parse().map_err(|_|
                super::RagError::ConfigError("Invalid RAG_MAX_RESULTS".into()))?;
        }
        
        if let Ok(min) = std::env::var("RAG_MIN_SIMILARITY") {
            self.search.min_similarity = min.parse().map_err(|_|
                super::RagError::ConfigError("Invalid RAG_MIN_SIMILARITY".into()))?;
        }
        
        if let Ok(metric) = std::env::var("RAG_DISTANCE_METRIC") {
            self.search.distance_metric = match metric.to_lowercase().as_str() {
                "cosine" => DistanceMetric::Cosine,
                "l2" | "euclidean" => DistanceMetric::L2,
                "inner_product" | "ip" | "dot" => DistanceMetric::InnerProduct,
                _ => return Err(super::RagError::ConfigError(
                    format!("Invalid RAG_DISTANCE_METRIC: {}", metric)
                )),
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
                "Embedding dimension must be greater than 0".into()
            ));
        }
        
        if self.search.min_similarity < -1.0 || self.search.min_similarity > 1.0 {
            return Err(super::RagError::ConfigError(
                "min_similarity must be between -1.0 and 1.0".into()
            ));
        }
        
        Ok(())
    }
    
    /// Save configuration to a TOML file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), super::RagError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| super::RagError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| super::RagError::ConfigError(format!("Failed to write config file: {}", e)))?;
        
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
        self.search.min_similarity = min.clamp(-1.0, 1.0);
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

    /// Get the configured index type
    pub fn index_type(&self) -> IndexType {
        self.embeddings.index_type()
    }

    /// Get the configured search type
    pub fn search_type(&self) -> SearchType {
        self.search.search_type
    }

    /// Get the RRF k parameter for hybrid search
    pub fn rrf_k(&self) -> u32 {
        self.search.rrf_k
    }

    /// Get the hybrid search oversampling factor
    pub fn hybrid_oversampling(&self) -> u32 {
        self.search.hybrid_oversampling
    }

    /// Get the text search language configuration
    pub fn text_search_language(&self) -> &str {
        &self.search.text_search_language
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

# Dimension of embedding vectors (auto-detected from model if not set)
# Common values:
#   - 384: all-MiniLM-L6-v2, all-MiniLM-L12-v2
#   - 768: all-mpnet-base-v2, BERT-base
#   - 2048: LLaMA-7B (hidden_size)
#   - 4096: LLaMA-13B/70B (hidden_size)
dimension = 384

# Index type: "hnsw" (default, recommended), "ivfflat", or "none"
index_type = "hnsw"

# HNSW parameters (only used when index_type = "hnsw")
hnsw_m = 16                  # Max connections per node (higher = better recall, more memory)
hnsw_ef_construction = 64    # Construction search depth (higher = better index, slower build)

# IVFFlat parameters (only used when index_type = "ivfflat")
# ivfflat_lists = 100        # Number of inverted lists

[search]
# Maximum number of results to return from similarity search
max_results = 5

# Minimum similarity score (0.0 to 1.0) for results to be included
min_similarity = 0.5

# Distance metric for similarity search
# Options: "cosine" (default), "l2", "inner_product"
distance_metric = "cosine"

# Search type: "semantic" (vector only) or "hybrid" (vector + keyword with RRF)
search_type = "semantic"

# RRF constant for hybrid search (higher = more weight to lower-ranked results)
rrf_k = 60

# Oversampling factor for hybrid search (fetch N * this many candidates before fusion)
hybrid_oversampling = 2

# PostgreSQL text search configuration language
text_search_language = "english"
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_type_default_is_hnsw() {
        let idx = IndexType::default();
        assert_eq!(
            idx,
            IndexType::Hnsw {
                m: 16,
                ef_construction: 64
            }
        );
    }

    #[test]
    fn test_index_type_hnsw_sql() {
        let idx = IndexType::Hnsw {
            m: 32,
            ef_construction: 128,
        };
        let (method, ops, with_clause) = idx.index_sql("vector_cosine_ops");
        assert_eq!(method, "hnsw");
        assert_eq!(ops, "vector_cosine_ops");
        assert_eq!(with_clause, "WITH (m = 32, ef_construction = 128)");
    }

    #[test]
    fn test_index_type_ivfflat_sql() {
        let idx = IndexType::IvfFlat { lists: 200 };
        let (method, ops, with_clause) = idx.index_sql("vector_l2_ops");
        assert_eq!(method, "ivfflat");
        assert_eq!(ops, "vector_l2_ops");
        assert_eq!(with_clause, "WITH (lists = 200)");
    }

    #[test]
    fn test_search_config_defaults() {
        let sc = SearchConfig::default();
        assert_eq!(sc.search_type, SearchType::Semantic);
        assert_eq!(sc.rrf_k, 60);
        assert_eq!(sc.hybrid_oversampling, 2);
        assert_eq!(sc.text_search_language, "english");
        assert_eq!(sc.max_results, 5);
        assert!((sc.min_similarity - 0.5).abs() < f32::EPSILON);
        assert_eq!(sc.distance_metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = RagConfig::new("postgres://localhost/test")
            .with_table("docs")
            .with_dim(768)
            .with_max_results(10)
            .with_min_similarity(0.7)
            .with_distance_metric(DistanceMetric::L2);

        let toml_str = toml::to_string_pretty(&config).expect("serialize");
        let parsed: RagConfig = toml::from_str(&toml_str).expect("deserialize");

        assert_eq!(parsed.connection_string(), "postgres://localhost/test");
        assert_eq!(parsed.table_name(), "docs");
        assert_eq!(parsed.embedding_dim(), 768);
        assert_eq!(parsed.max_results(), 10);
        assert!((parsed.min_similarity() - 0.7).abs() < f32::EPSILON);
        assert_eq!(parsed.distance_metric(), DistanceMetric::L2);
        assert_eq!(parsed.search_type(), SearchType::Semantic);
        assert_eq!(parsed.rrf_k(), 60);
        assert_eq!(parsed.hybrid_oversampling(), 2);
        assert_eq!(parsed.text_search_language(), "english");
        assert_eq!(
            parsed.index_type(),
            IndexType::Hnsw {
                m: 16,
                ef_construction: 64
            }
        );
    }

    #[test]
    fn test_index_type_serde() {
        let toml_str = r#"
[database]
connection_string = "postgres://localhost/test"

[embeddings]
table_name = "embeddings"
dimension = 384
index_type = "ivfflat"
ivfflat_lists = 200

[search]
search_type = "hybrid"
rrf_k = 30
hybrid_oversampling = 4
text_search_language = "spanish"
"#;

        let config: RagConfig = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(
            config.index_type(),
            IndexType::IvfFlat { lists: 200 }
        );
        assert_eq!(config.search_type(), SearchType::Hybrid);
        assert_eq!(config.rrf_k(), 30);
        assert_eq!(config.hybrid_oversampling(), 4);
        assert_eq!(config.text_search_language(), "spanish");
    }
}
