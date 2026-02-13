//! Knowledge Base - AWS Bedrock-style RAG interface
//!
//! This module provides a high-level Knowledge Base abstraction similar to
//! AWS Bedrock Knowledge Bases, with features like:
//!
//! - Multiple data source types (files, directories, URLs)
//! - Configurable chunking strategies (fixed, semantic, hierarchical)
//! - Hybrid search (semantic + keyword)
//! - Result reranking
//! - Source citations
//! - Retrieve and Generate (RAG) pipeline
//!
//! # Example
//!
//! ```rust,ignore
//! use llama_gguf::rag::{KnowledgeBase, KnowledgeBaseConfig, DataSource, ChunkingStrategy};
//!
//! // Create a knowledge base
//! let kb = KnowledgeBase::create(KnowledgeBaseConfig {
//!     name: "my-kb".into(),
//!     description: Some("My knowledge base".into()),
//!     chunking: ChunkingStrategy::FixedSize { 
//!         max_tokens: 300, 
//!         overlap_percentage: 20 
//!     },
//!     ..Default::default()
//! }).await?;
//!
//! // Ingest data
//! kb.ingest(DataSource::Directory { 
//!     path: "./docs".into(),
//!     pattern: Some("**/*.md".into()),
//! }).await?;
//!
//! // Retrieve and generate
//! let response = kb.retrieve_and_generate(
//!     "What is the main feature?",
//!     RetrievalConfig::default(),
//! ).await?;
//!
//! println!("Answer: {}", response.output);
//! for citation in response.citations {
//!     println!("Source: {} (score: {:.2})", citation.source, citation.score);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::backend::Backend;
use crate::model::LlamaModel;
use crate::tokenizer::Tokenizer;

use super::{
    Document, EmbeddingGenerator, MetadataFilter, NewDocument, RagConfig, RagError, RagResult,
    RagStore, SearchType, TextChunker,
};

// =============================================================================
// Configuration Types
// =============================================================================

/// Knowledge base configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    /// Unique name for the knowledge base
    pub name: String,

    /// Optional description
    #[serde(default)]
    pub description: Option<String>,

    /// RAG store configuration
    #[serde(default)]
    pub storage: RagConfig,

    /// Chunking strategy for documents
    #[serde(default)]
    pub chunking: ChunkingStrategy,

    /// Search/retrieval configuration
    #[serde(default)]
    pub retrieval: RetrievalConfig,

    /// Whether to enable hybrid search (semantic + keyword)
    #[serde(default)]
    pub hybrid_search: bool,

    /// Reranking configuration
    #[serde(default)]
    pub reranking: Option<RerankingConfig>,
}

impl Default for KnowledgeBaseConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            description: None,
            storage: RagConfig::default(),
            chunking: ChunkingStrategy::default(),
            retrieval: RetrievalConfig::default(),
            hybrid_search: false,
            reranking: None,
        }
    }
}

/// Chunking strategy for splitting documents
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChunkingStrategy {
    /// No chunking - store documents as-is
    None,

    /// Fixed size chunks with optional overlap
    FixedSize {
        /// Maximum tokens per chunk (approximate, based on chars/4)
        max_tokens: usize,
        /// Overlap percentage between chunks (0-50)
        overlap_percentage: u8,
    },

    /// Semantic chunking - split on sentence/paragraph boundaries
    Semantic {
        /// Maximum tokens per chunk
        max_tokens: usize,
        /// Buffer size for boundary detection
        buffer_size: usize,
    },

    /// Hierarchical chunking - parent/child relationships
    Hierarchical {
        /// Parent chunk max tokens
        parent_max_tokens: usize,
        /// Child chunk max tokens
        child_max_tokens: usize,
        /// Overlap percentage for child chunks
        child_overlap_percentage: u8,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::FixedSize {
            max_tokens: 300,
            overlap_percentage: 20,
        }
    }
}

/// Search/retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Maximum number of results to retrieve
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Minimum similarity score (0.0-1.0)
    #[serde(default = "default_min_score")]
    pub min_score: f32,

    /// Search type
    #[serde(default)]
    pub search_type: SearchType,

    /// Optional metadata filter
    #[serde(skip)]
    pub filter: Option<MetadataFilter>,

    /// Override prompt template for generation
    #[serde(default)]
    pub prompt_template: Option<String>,
}

fn default_max_results() -> usize {
    5
}
fn default_min_score() -> f32 {
    0.5
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            min_score: 0.5,
            search_type: SearchType::Semantic,
            filter: None,
            prompt_template: None,
        }
    }
}

/// Reranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingConfig {
    /// Number of candidates to fetch before reranking
    pub num_candidates: usize,
    /// Reranking model/method
    pub method: RerankingMethod,
}

/// Reranking methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RerankingMethod {
    /// Simple score-based reranking
    ScoreBased,
    /// Cross-encoder reranking (requires model)
    CrossEncoder { model_path: String },
    /// Reciprocal Rank Fusion for hybrid search
    RRF { k: usize },
}

// =============================================================================
// Data Sources
// =============================================================================

/// Data source for ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataSource {
    /// Single file
    File {
        path: PathBuf,
    },

    /// Directory of files
    Directory {
        path: PathBuf,
        /// Optional glob pattern (e.g., "**/*.md")
        pattern: Option<String>,
        /// Recursive search
        #[serde(default = "default_true")]
        recursive: bool,
    },

    /// Raw text content
    Text {
        content: String,
        /// Source identifier for citations
        source_id: String,
        /// Optional metadata
        metadata: Option<serde_json::Value>,
    },

    /// Web URL (for future implementation)
    Url {
        url: String,
        /// Crawl depth (0 = single page)
        #[serde(default)]
        depth: usize,
    },

    /// S3-style object storage path
    ObjectStorage {
        /// Bucket/container name
        bucket: String,
        /// Object prefix/path
        prefix: String,
        /// Endpoint URL (for S3-compatible services)
        endpoint: Option<String>,
    },
}

fn default_true() -> bool {
    true
}

/// Result of data source ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionResult {
    /// Number of documents processed
    pub documents_processed: usize,
    /// Number of chunks created
    pub chunks_created: usize,
    /// Failed documents (path/id -> error message)
    pub failures: HashMap<String, String>,
    /// Metadata about the ingestion
    pub metadata: IngestionMetadata,
}

/// Metadata about an ingestion job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionMetadata {
    /// Data source identifier
    pub source_id: String,
    /// Timestamp
    pub timestamp: String,
    /// Chunking strategy used
    pub chunking_strategy: String,
    /// Total characters processed
    pub total_characters: usize,
}

// =============================================================================
// Retrieval and Generation
// =============================================================================

/// Retrieved chunk with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    /// Chunk content
    pub content: String,
    /// Similarity/relevance score
    pub score: f32,
    /// Source document/location
    pub source: SourceLocation,
    /// Document metadata
    pub metadata: Option<serde_json::Value>,
}

/// Source location for citations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// Source type (file, url, etc.)
    pub source_type: String,
    /// Source identifier (path, URL, etc.)
    pub uri: String,
    /// Optional location within source
    pub location: Option<TextLocation>,
}

/// Location within a text document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLocation {
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
}

/// Citation in a generated response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Text span in the generated output that this citation supports
    pub generated_text_span: Option<TextLocation>,
    /// Source location
    pub source: SourceLocation,
    /// Relevance score
    pub score: f32,
    /// Retrieved content that supports this citation
    pub content: String,
}

/// Retrieve-only response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResponse {
    /// Retrieved chunks
    pub chunks: Vec<RetrievedChunk>,
    /// Query that was used
    pub query: String,
    /// Next token for pagination (if applicable)
    pub next_token: Option<String>,
}

/// Retrieve and Generate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieveAndGenerateResponse {
    /// Generated output text
    pub output: String,
    /// Citations for the generated content
    pub citations: Vec<Citation>,
    /// Retrieved chunks used for generation
    pub retrieved_chunks: Vec<RetrievedChunk>,
    /// Guardrail action taken (if any)
    pub guardrail_action: Option<String>,
}

// =============================================================================
// Reranking
// =============================================================================

/// Rerank retrieved chunks according to the given reranking configuration.
///
/// - `ScoreBased` — sort by score descending.
/// - `RRF { k }` — sort by score descending (RRF scoring already happened at store level).
/// - `CrossEncoder { model_path }` — log a warning and fall back to score-based sort.
pub fn rerank(mut chunks: Vec<RetrievedChunk>, config: &RerankingConfig) -> Vec<RetrievedChunk> {
    match &config.method {
        RerankingMethod::ScoreBased => {
            chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        RerankingMethod::RRF { k: _ } => {
            // RRF scoring already happened at the store level; just sort by score descending.
            chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        RerankingMethod::CrossEncoder { model_path } => {
            tracing::warn!(
                "CrossEncoder reranking with model '{}' is not yet implemented; falling back to score-based sort",
                model_path
            );
            chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    chunks
}

// =============================================================================
// Knowledge Base Implementation
// =============================================================================

/// Knowledge Base - high-level RAG interface
pub struct KnowledgeBase {
    config: KnowledgeBaseConfig,
    store: RagStore,
    embedding_gen: Option<EmbeddingGenerator>,
}

impl KnowledgeBase {
    /// Create a new knowledge base
    pub async fn create(config: KnowledgeBaseConfig) -> RagResult<Self> {
        let store = RagStore::connect(config.storage.clone()).await?;
        store.create_table().await?;

        Ok(Self {
            config,
            store,
            embedding_gen: None,
        })
    }

    /// Connect to an existing knowledge base
    pub async fn connect(config: KnowledgeBaseConfig) -> RagResult<Self> {
        let store = RagStore::connect(config.storage.clone()).await?;
        Ok(Self {
            config,
            store,
            embedding_gen: None,
        })
    }

    /// Attach an embedding generator to this knowledge base
    pub fn with_embedding_generator(mut self, emb_gen: EmbeddingGenerator) -> Self {
        self.embedding_gen = Some(emb_gen);
        self
    }

    /// Get the knowledge base name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the knowledge base configuration
    pub fn config(&self) -> &KnowledgeBaseConfig {
        &self.config
    }

    /// Ingest data from a source
    pub async fn ingest(&self, source: DataSource) -> RagResult<IngestionResult> {
        let mut result = IngestionResult {
            documents_processed: 0,
            chunks_created: 0,
            failures: HashMap::new(),
            metadata: IngestionMetadata {
                source_id: self.source_id(&source),
                timestamp: chrono_now(),
                chunking_strategy: format!("{:?}", self.config.chunking),
                total_characters: 0,
            },
        };

        match source {
            DataSource::File { path } => {
                self.ingest_file(&path, &mut result).await?;
            }
            DataSource::Directory {
                path,
                pattern,
                recursive,
            } => {
                self.ingest_directory(&path, pattern.as_deref(), recursive, &mut result)
                    .await?;
            }
            DataSource::Text {
                content,
                source_id,
                metadata,
            } => {
                self.ingest_text(&content, &source_id, metadata, &mut result)
                    .await?;
            }
            DataSource::Url { url, depth: _ } => {
                result.failures.insert(
                    url,
                    "URL ingestion not yet implemented".into(),
                );
            }
            DataSource::ObjectStorage {
                bucket,
                prefix,
                endpoint: _,
            } => {
                result.failures.insert(
                    format!("{}:{}", bucket, prefix),
                    "Object storage ingestion not yet implemented".into(),
                );
            }
        }

        Ok(result)
    }

    /// Retrieve relevant chunks for a query
    pub async fn retrieve(
        &self,
        query: &str,
        config: Option<RetrievalConfig>,
    ) -> RagResult<RetrievalResponse> {
        let config = config.unwrap_or_else(|| self.config.retrieval.clone());

        // Generate query embedding
        let query_embedding = self.embed_query(query)?;

        // Dispatch to hybrid or semantic search based on configuration
        let docs = if self.config.storage.search_type() == SearchType::Hybrid {
            self.store
                .search_hybrid(
                    &query_embedding,
                    query,
                    Some(config.max_results),
                    config.filter,
                )
                .await?
        } else {
            self.store
                .search_with_filter(&query_embedding, Some(config.max_results), config.filter)
                .await?
        };

        // Convert to retrieved chunks
        let mut chunks: Vec<RetrievedChunk> = docs
            .into_iter()
            .filter(|d| d.score.unwrap_or(0.0) >= config.min_score)
            .map(|d| self.doc_to_chunk(d))
            .collect();

        // Apply reranking if configured
        if let Some(reranking_config) = &self.config.reranking {
            chunks = rerank(chunks, reranking_config);
        }

        Ok(RetrievalResponse {
            chunks,
            query: query.to_string(),
            next_token: None,
        })
    }

    /// Retrieve and generate a response
    ///
    /// This combines retrieval with LLM generation. In practice, you would
    /// pass the retrieved context to your LLM for generation.
    pub async fn retrieve_and_generate(
        &self,
        query: &str,
        config: Option<RetrievalConfig>,
    ) -> RagResult<RetrieveAndGenerateResponse> {
        let config = config.unwrap_or_else(|| self.config.retrieval.clone());

        // Retrieve relevant chunks
        let retrieval = self.retrieve(query, Some(config.clone())).await?;

        // Build context from retrieved chunks
        let context = self.build_context(&retrieval.chunks);

        // Build prompt
        let prompt = if let Some(template) = &config.prompt_template {
            template
                .replace("{context}", &context)
                .replace("{query}", query)
                .replace("{question}", query)
        } else {
            self.default_prompt(&context, query)
        };

        // Create citations from retrieved chunks
        let citations: Vec<Citation> = retrieval
            .chunks
            .iter()
            .map(|chunk| Citation {
                generated_text_span: None, // Would be filled by actual generation
                source: chunk.source.clone(),
                score: chunk.score,
                content: chunk.content.clone(),
            })
            .collect();

        // Note: Actual LLM generation would happen here
        // For now, return the prompt as output (user should pass to LLM)
        Ok(RetrieveAndGenerateResponse {
            output: prompt,
            citations,
            retrieved_chunks: retrieval.chunks,
            guardrail_action: None,
        })
    }

    /// Sync the knowledge base (re-process all data sources)
    pub async fn sync(&self) -> RagResult<()> {
        // In a full implementation, this would track data sources
        // and re-ingest changed/new documents
        Ok(())
    }

    /// Delete the knowledge base
    pub async fn delete(&self) -> RagResult<()> {
        self.store.clear().await?;
        Ok(())
    }

    /// Get statistics about the knowledge base
    pub async fn stats(&self) -> RagResult<KnowledgeBaseStats> {
        let document_count = self.store.count().await? as usize;
        
        Ok(KnowledgeBaseStats {
            name: self.config.name.clone(),
            document_count,
            embedding_dimension: self.config.storage.embedding_dim(),
            chunking_strategy: format!("{:?}", self.config.chunking),
            hybrid_search_enabled: self.config.hybrid_search,
        })
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    fn source_id(&self, source: &DataSource) -> String {
        match source {
            DataSource::File { path } => path.to_string_lossy().to_string(),
            DataSource::Directory { path, .. } => path.to_string_lossy().to_string(),
            DataSource::Text { source_id, .. } => source_id.clone(),
            DataSource::Url { url, .. } => url.clone(),
            DataSource::ObjectStorage { bucket, prefix, .. } => {
                format!("s3://{}/{}", bucket, prefix)
            }
        }
    }

    async fn ingest_file(
        &self,
        path: &std::path::Path,
        result: &mut IngestionResult,
    ) -> RagResult<()> {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let source_id = path.to_string_lossy().to_string();
                let metadata = serde_json::json!({
                    "source": source_id,
                    "source_type": "file",
                    "filename": path.file_name().map(|n| n.to_string_lossy().to_string()),
                });

                self.ingest_text(&content, &source_id, Some(metadata), result)
                    .await?;
                result.documents_processed += 1;
            }
            Err(e) => {
                result
                    .failures
                    .insert(path.to_string_lossy().to_string(), e.to_string());
            }
        }
        Ok(())
    }

    async fn ingest_directory(
        &self,
        path: &std::path::Path,
        pattern: Option<&str>,
        recursive: bool,
        result: &mut IngestionResult,
    ) -> RagResult<()> {
        let entries = if recursive {
            self.walk_directory_recursive(path, pattern)?
        } else {
            self.walk_directory_flat(path, pattern)?
        };

        for entry in entries {
            self.ingest_file(&entry, result).await?;
        }

        Ok(())
    }

    fn walk_directory_recursive(
        &self,
        path: &std::path::Path,
        pattern: Option<&str>,
    ) -> RagResult<Vec<PathBuf>> {
        let mut files = Vec::new();

        fn visit_dir(dir: &std::path::Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dir(&path, files)?;
                } else if path.is_file() {
                    files.push(path);
                }
            }
            Ok(())
        }

        visit_dir(path, &mut files)
            .map_err(|e| RagError::ConfigError(format!("Failed to read directory: {}", e)))?;

        if let Some(pattern) = pattern {
            files.retain(|f| {
                let path_str = f.to_string_lossy();
                matches_glob_pattern(&path_str, pattern)
            });
        }

        Ok(files)
    }

    fn walk_directory_flat(
        &self,
        path: &std::path::Path,
        pattern: Option<&str>,
    ) -> RagResult<Vec<PathBuf>> {
        let mut files = Vec::new();

        let entries = std::fs::read_dir(path)
            .map_err(|e| RagError::ConfigError(format!("Failed to read directory: {}", e)))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            }
        }

        if let Some(pattern) = pattern {
            files.retain(|f| {
                let path_str = f.to_string_lossy();
                matches_glob_pattern(&path_str, pattern)
            });
        }

        Ok(files)
    }

    async fn ingest_text(
        &self,
        content: &str,
        source_id: &str,
        metadata: Option<serde_json::Value>,
        result: &mut IngestionResult,
    ) -> RagResult<()> {
        result.metadata.total_characters += content.len();

        // Chunk the content
        let chunks = self.chunk_text(content);

        // Create documents for each chunk
        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk_metadata = serde_json::json!({
                "source": source_id,
                "chunk_index": i,
                "total_chunks": chunks.len(),
                "parent": metadata.clone(),
            });

            // Generate embedding (placeholder)
            let embedding = self.embed_text(chunk_text)?;

            let doc = NewDocument {
                content: chunk_text.clone(),
                embedding,
                metadata: Some(chunk_metadata),
            };

            self.store.insert(doc).await?;
            result.chunks_created += 1;
        }

        Ok(())
    }

    fn chunk_text(&self, text: &str) -> Vec<String> {
        match &self.config.chunking {
            ChunkingStrategy::None => vec![text.to_string()],

            ChunkingStrategy::FixedSize {
                max_tokens,
                overlap_percentage,
            } => {
                let char_size = max_tokens * 4; // Approximate chars per token
                let overlap = (char_size * *overlap_percentage as usize) / 100;

                let chunker = TextChunker::new(char_size).with_overlap(overlap);
                chunker.chunk(text)
            }

            ChunkingStrategy::Semantic {
                max_tokens,
                buffer_size: _,
            } => {
                // Semantic chunking: split on sentence boundaries
                let char_size = max_tokens * 4;
                let sentences: Vec<&str> = text
                    .split(['.', '!', '?'])
                    .filter(|s| !s.trim().is_empty())
                    .collect();

                let mut chunks = Vec::new();
                let mut current_chunk = String::new();

                for sentence in sentences {
                    let sentence = sentence.trim().to_string() + ".";

                    if current_chunk.len() + sentence.len() > char_size {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.trim().to_string());
                        }
                        current_chunk = sentence;
                    } else {
                        if !current_chunk.is_empty() {
                            current_chunk.push(' ');
                        }
                        current_chunk.push_str(&sentence);
                    }
                }

                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }

                chunks
            }

            ChunkingStrategy::Hierarchical {
                parent_max_tokens,
                child_max_tokens,
                child_overlap_percentage,
            } => {
                // First create parent chunks
                let parent_char_size = parent_max_tokens * 4;
                let child_char_size = child_max_tokens * 4;
                let child_overlap = (child_char_size * *child_overlap_percentage as usize) / 100;

                let parent_chunker = TextChunker::new(parent_char_size);
                let child_chunker = TextChunker::new(child_char_size).with_overlap(child_overlap);

                let parents = parent_chunker.chunk(text);
                let mut all_chunks = Vec::new();

                for parent in parents {
                    let children = child_chunker.chunk(&parent);
                    all_chunks.extend(children);
                }

                all_chunks
            }
        }
    }

    fn embed_text(&self, text: &str) -> RagResult<Vec<f32>> {
        if let Some(emb) = &self.embedding_gen {
            emb.embed(text)
        } else {
            // Fallback: zero vector (for testing without a model)
            Ok(vec![0.0f32; self.config.storage.embedding_dim()])
        }
    }

    fn embed_query(&self, query: &str) -> RagResult<Vec<f32>> {
        if let Some(emb) = &self.embedding_gen {
            emb.embed(query)
        } else {
            // Fallback: zero vector (for testing without a model)
            Ok(vec![0.0f32; self.config.storage.embedding_dim()])
        }
    }

    fn doc_to_chunk(&self, doc: Document) -> RetrievedChunk {
        let source_uri = doc
            .metadata
            .as_ref()
            .and_then(|m| m.get("source"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        RetrievedChunk {
            content: doc.content,
            score: doc.score.unwrap_or(0.0),
            source: SourceLocation {
                source_type: "document".into(),
                uri: source_uri,
                location: None,
            },
            metadata: doc.metadata,
        }
    }

    fn build_context(&self, chunks: &[RetrievedChunk]) -> String {
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i + 1, c.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn default_prompt(&self, context: &str, query: &str) -> String {
        format!(
            r#"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {query}

Helpful Answer:"#
        )
    }
}

/// Knowledge base statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseStats {
    pub name: String,
    pub document_count: usize,
    pub embedding_dimension: usize,
    pub chunking_strategy: String,
    pub hybrid_search_enabled: bool,
}

/// Get current timestamp as string
fn chrono_now() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", duration.as_secs())
}

/// Check whether a path string matches a glob pattern.
///
/// Uses the `glob` crate's `Pattern` for matching. Returns `false` if the
/// pattern is invalid.
fn matches_glob_pattern(path: &str, pattern: &str) -> bool {
    glob::Pattern::new(pattern)
        .map(|p| p.matches(path))
        .unwrap_or(false)
}

// =============================================================================
// Builder Pattern
// =============================================================================

/// Builder for KnowledgeBaseConfig
pub struct KnowledgeBaseBuilder {
    config: KnowledgeBaseConfig,
    model: Option<Arc<LlamaModel>>,
    tokenizer: Option<Arc<Tokenizer>>,
    backend: Option<Arc<dyn Backend>>,
}

impl KnowledgeBaseBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: KnowledgeBaseConfig {
                name: name.into(),
                ..Default::default()
            },
            model: None,
            tokenizer: None,
            backend: None,
        }
    }

    /// Set the model, tokenizer, and backend for real embedding generation
    pub fn with_model(
        mut self,
        model: Arc<LlamaModel>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        self.model = Some(model);
        self.tokenizer = Some(tokenizer);
        self.backend = Some(backend);
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.config.description = Some(desc.into());
        self
    }

    pub fn storage(mut self, storage: RagConfig) -> Self {
        self.config.storage = storage;
        self
    }

    pub fn chunking(mut self, strategy: ChunkingStrategy) -> Self {
        self.config.chunking = strategy;
        self
    }

    pub fn fixed_size_chunking(mut self, max_tokens: usize, overlap_pct: u8) -> Self {
        self.config.chunking = ChunkingStrategy::FixedSize {
            max_tokens,
            overlap_percentage: overlap_pct.min(50),
        };
        self
    }

    pub fn semantic_chunking(mut self, max_tokens: usize) -> Self {
        self.config.chunking = ChunkingStrategy::Semantic {
            max_tokens,
            buffer_size: 100,
        };
        self
    }

    pub fn hierarchical_chunking(
        mut self,
        parent_tokens: usize,
        child_tokens: usize,
        overlap_pct: u8,
    ) -> Self {
        self.config.chunking = ChunkingStrategy::Hierarchical {
            parent_max_tokens: parent_tokens,
            child_max_tokens: child_tokens,
            child_overlap_percentage: overlap_pct.min(50),
        };
        self
    }

    pub fn retrieval(mut self, retrieval: RetrievalConfig) -> Self {
        self.config.retrieval = retrieval;
        self
    }

    pub fn max_results(mut self, max: usize) -> Self {
        self.config.retrieval.max_results = max;
        self
    }

    pub fn min_score(mut self, min: f32) -> Self {
        self.config.retrieval.min_score = min.clamp(0.0, 1.0);
        self
    }

    pub fn hybrid_search(mut self, enabled: bool) -> Self {
        self.config.hybrid_search = enabled;
        self
    }

    pub fn reranking(mut self, config: RerankingConfig) -> Self {
        self.config.reranking = Some(config);
        self
    }

    pub fn build(self) -> KnowledgeBaseConfig {
        self.config
    }

    pub async fn create(self) -> RagResult<KnowledgeBase> {
        let mut kb = KnowledgeBase::create(self.config).await?;

        // Construct EmbeddingGenerator when model, tokenizer, and backend are all provided
        if let (Some(model), Some(tokenizer), Some(backend)) =
            (self.model, self.tokenizer, self.backend)
        {
            kb.embedding_gen = Some(EmbeddingGenerator::new(model, tokenizer, backend));
        }

        Ok(kb)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_score_based() {
        let low = RetrievedChunk {
            content: "low score chunk".into(),
            score: 0.3,
            source: SourceLocation {
                source_type: "document".into(),
                uri: "low.txt".into(),
                location: None,
            },
            metadata: None,
        };
        let high = RetrievedChunk {
            content: "high score chunk".into(),
            score: 0.9,
            source: SourceLocation {
                source_type: "document".into(),
                uri: "high.txt".into(),
                location: None,
            },
            metadata: None,
        };

        // Feed them in low-first order
        let chunks = vec![low, high];
        let config = RerankingConfig {
            num_candidates: 10,
            method: RerankingMethod::ScoreBased,
        };

        let result = rerank(chunks, &config);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "high score chunk");
        assert_eq!(result[1].content, "low score chunk");
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn test_glob_pattern_matching() {
        // Markdown files match **/*.md
        assert!(matches_glob_pattern("docs/readme.md", "**/*.md"));

        // Rust source files match **/*.rs
        assert!(matches_glob_pattern("src/lib.rs", "**/*.rs"));

        // PNG file does NOT match **/*.md
        assert!(!matches_glob_pattern("image.png", "**/*.md"));

        // Edge cases
        assert!(matches_glob_pattern("a.md", "**/*.md"));
        assert!(!matches_glob_pattern("docs/readme.txt", "**/*.md"));
    }
}
