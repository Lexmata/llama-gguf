//! RAG vector store with pgvector

use deadpool_postgres::{Config, Pool, Runtime};
use pgvector::Vector;
use tokio_postgres::NoTls;
use serde::{Deserialize, Serialize};

use super::{RagConfig, RagError, RagResult};

/// Metadata filter for search queries
/// 
/// Filters can be combined using AND/OR logic and support various comparison operators
/// for JSONB fields in PostgreSQL.
/// 
/// # Example
/// 
/// ```rust,ignore
/// use llama_gguf::rag::MetadataFilter;
/// 
/// // Simple equality filter
/// let filter = MetadataFilter::eq("source", "docs/readme.md");
/// 
/// // Combine filters with AND
/// let filter = MetadataFilter::and(vec![
///     MetadataFilter::eq("type", "documentation"),
///     MetadataFilter::gte("version", 2),
/// ]);
/// 
/// // Complex filter with OR
/// let filter = MetadataFilter::or(vec![
///     MetadataFilter::eq("category", "api"),
///     MetadataFilter::contains("tags", "important"),
/// ]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MetadataFilter {
    /// Exact equality: metadata->>'field' = 'value'
    Eq { field: String, value: serde_json::Value },
    
    /// Not equal: metadata->>'field' != 'value'
    Ne { field: String, value: serde_json::Value },
    
    /// Greater than: (metadata->>'field')::numeric > value
    Gt { field: String, value: serde_json::Value },
    
    /// Greater than or equal: (metadata->>'field')::numeric >= value
    Gte { field: String, value: serde_json::Value },
    
    /// Less than: (metadata->>'field')::numeric < value
    Lt { field: String, value: serde_json::Value },
    
    /// Less than or equal: (metadata->>'field')::numeric <= value
    Lte { field: String, value: serde_json::Value },
    
    /// Field exists: metadata ? 'field'
    Exists { field: String },
    
    /// Field does not exist: NOT (metadata ? 'field')
    NotExists { field: String },
    
    /// String contains (case-insensitive): metadata->>'field' ILIKE '%value%'
    Contains { field: String, value: String },
    
    /// String starts with: metadata->>'field' LIKE 'value%'
    StartsWith { field: String, value: String },
    
    /// String ends with: metadata->>'field' LIKE '%value'
    EndsWith { field: String, value: String },
    
    /// Value in array: metadata->'field' ? 'value' (for array fields)
    InArray { field: String, value: String },
    
    /// Value in list: metadata->>'field' IN ('a', 'b', 'c')
    In { field: String, values: Vec<serde_json::Value> },
    
    /// Value not in list: metadata->>'field' NOT IN ('a', 'b', 'c')
    NotIn { field: String, values: Vec<serde_json::Value> },
    
    /// JSON path exists: metadata @? '$.field[*] ? (@ == "value")'
    JsonPath { path: String },
    
    /// Logical AND of multiple filters
    And { filters: Vec<MetadataFilter> },
    
    /// Logical OR of multiple filters
    Or { filters: Vec<MetadataFilter> },
    
    /// Logical NOT
    Not { filter: Box<MetadataFilter> },
}

impl MetadataFilter {
    // Convenience constructors
    
    /// Create an equality filter
    pub fn eq(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Eq { field: field.into(), value: value.into() }
    }
    
    /// Create a not-equal filter
    pub fn ne(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Ne { field: field.into(), value: value.into() }
    }
    
    /// Create a greater-than filter
    pub fn gt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Gt { field: field.into(), value: value.into() }
    }
    
    /// Create a greater-than-or-equal filter
    pub fn gte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Gte { field: field.into(), value: value.into() }
    }
    
    /// Create a less-than filter
    pub fn lt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Lt { field: field.into(), value: value.into() }
    }
    
    /// Create a less-than-or-equal filter
    pub fn lte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Lte { field: field.into(), value: value.into() }
    }
    
    /// Create an exists filter
    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists { field: field.into() }
    }
    
    /// Create a not-exists filter
    pub fn not_exists(field: impl Into<String>) -> Self {
        Self::NotExists { field: field.into() }
    }
    
    /// Create a contains filter (case-insensitive substring match)
    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Contains { field: field.into(), value: value.into() }
    }
    
    /// Create a starts-with filter
    pub fn starts_with(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::StartsWith { field: field.into(), value: value.into() }
    }
    
    /// Create an ends-with filter
    pub fn ends_with(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::EndsWith { field: field.into(), value: value.into() }
    }
    
    /// Create a filter for checking if value is in a JSON array field
    pub fn in_array(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::InArray { field: field.into(), value: value.into() }
    }
    
    /// Create an IN filter
    pub fn in_values(field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        Self::In { field: field.into(), values }
    }
    
    /// Create a NOT IN filter
    pub fn not_in(field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        Self::NotIn { field: field.into(), values }
    }
    
    /// Create a JSON path filter
    pub fn json_path(path: impl Into<String>) -> Self {
        Self::JsonPath { path: path.into() }
    }
    
    /// Create an AND filter combining multiple filters
    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        Self::And { filters }
    }
    
    /// Create an OR filter combining multiple filters
    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        Self::Or { filters }
    }
    
    /// Create a NOT filter
    pub fn not(filter: MetadataFilter) -> Self {
        Self::Not { filter: Box::new(filter) }
    }
    
    /// Convert the filter to a SQL WHERE clause fragment
    ///
    /// Returns the SQL string and a list of parameter values.
    /// Returns an error if any field name contains invalid characters.
    pub fn to_sql(&self, param_offset: usize) -> RagResult<(String, Vec<String>)> {
        let mut params = Vec::new();
        let sql = self.to_sql_inner(param_offset, &mut params)?;
        Ok((sql, params))
    }

    fn to_sql_inner(&self, param_offset: usize, params: &mut Vec<String>) -> RagResult<String> {
        match self {
            Self::Eq { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(json_value_to_string(value));
                Ok(format!("metadata->>'{}' = ${}", field, param_idx))
            }

            Self::Ne { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(json_value_to_string(value));
                Ok(format!("metadata->>'{}' != ${}", field, param_idx))
            }

            Self::Gt { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(json_value_to_string(value));
                Ok(format!("(metadata->>'{}')::numeric > ${}::numeric", field, param_idx))
            }

            Self::Gte { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(json_value_to_string(value));
                Ok(format!("(metadata->>'{}')::numeric >= ${}::numeric", field, param_idx))
            }

            Self::Lt { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(json_value_to_string(value));
                Ok(format!("(metadata->>'{}')::numeric < ${}::numeric", field, param_idx))
            }

            Self::Lte { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(json_value_to_string(value));
                Ok(format!("(metadata->>'{}')::numeric <= ${}::numeric", field, param_idx))
            }

            Self::Exists { field } => {
                let field = validate_field_name(field)?;
                Ok(format!("metadata ? '{}'", field))
            }

            Self::NotExists { field } => {
                let field = validate_field_name(field)?;
                Ok(format!("NOT (metadata ? '{}')", field))
            }

            Self::Contains { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(format!("%{}%", value));
                Ok(format!("metadata->>'{}' ILIKE ${}", field, param_idx))
            }

            Self::StartsWith { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(format!("{}%", value));
                Ok(format!("metadata->>'{}' LIKE ${}", field, param_idx))
            }

            Self::EndsWith { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(format!("%{}", value));
                Ok(format!("metadata->>'{}' LIKE ${}", field, param_idx))
            }

            Self::InArray { field, value } => {
                let field = validate_field_name(field)?;
                let param_idx = param_offset + params.len() + 1;
                params.push(value.clone());
                Ok(format!("metadata->'{}' ? ${}", field, param_idx))
            }

            Self::In { field, values } => {
                let field = validate_field_name(field)?;
                if values.is_empty() {
                    return Ok("FALSE".to_string());
                }
                let placeholders: Vec<String> = values.iter().enumerate().map(|(i, _v)| {
                    let param_idx = param_offset + params.len() + 1 + i;
                    format!("${}", param_idx)
                }).collect();
                for v in values {
                    params.push(json_value_to_string(v));
                }
                Ok(format!("metadata->>'{}' IN ({})", field, placeholders.join(", ")))
            }

            Self::NotIn { field, values } => {
                let field = validate_field_name(field)?;
                if values.is_empty() {
                    return Ok("TRUE".to_string());
                }
                let placeholders: Vec<String> = values.iter().enumerate().map(|(i, _)| {
                    let param_idx = param_offset + params.len() + 1 + i;
                    format!("${}", param_idx)
                }).collect();
                for v in values {
                    params.push(json_value_to_string(v));
                }
                Ok(format!("metadata->>'{}' NOT IN ({})", field, placeholders.join(", ")))
            }

            Self::JsonPath { path } => {
                Ok(format!("metadata @? '{}'", path.replace('\'', "''")))
            }

            Self::And { filters } => {
                if filters.is_empty() {
                    return Ok("TRUE".to_string());
                }
                let mut parts = Vec::new();
                for f in filters {
                    parts.push(f.to_sql_inner(param_offset + params.len(), params)?);
                }
                Ok(format!("({})", parts.join(" AND ")))
            }

            Self::Or { filters } => {
                if filters.is_empty() {
                    return Ok("FALSE".to_string());
                }
                let mut parts = Vec::new();
                for f in filters {
                    parts.push(f.to_sql_inner(param_offset + params.len(), params)?);
                }
                Ok(format!("({})", parts.join(" OR ")))
            }

            Self::Not { filter } => {
                let inner = filter.to_sql_inner(param_offset + params.len(), params)?;
                Ok(format!("NOT ({})", inner))
            }
        }
    }
    
    /// Parse a filter from a simple string syntax
    /// 
    /// Supported formats:
    /// - `field=value` - equality
    /// - `field!=value` - not equal
    /// - `field>value` - greater than
    /// - `field>=value` - greater than or equal
    /// - `field<value` - less than
    /// - `field<=value` - less than or equal
    /// - `field~value` - contains (case-insensitive)
    /// - `field^value` - starts with
    /// - `field$value` - ends with
    /// - `field?` - exists
    /// - `!field?` - not exists
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        
        // Check for exists/not exists
        if s.ends_with('?') {
            if s.starts_with('!') {
                return Ok(Self::not_exists(&s[1..s.len()-1]));
            }
            return Ok(Self::exists(&s[..s.len()-1]));
        }
        
        // Try to find operator
        let operators = [">=", "<=", "!=", "=", ">", "<", "~", "^", "$"];
        
        for op in &operators {
            if let Some(pos) = s.find(op) {
                let field = s[..pos].trim();
                let value = s[pos + op.len()..].trim();
                
                // Try to parse as number first, then as string
                let json_value: serde_json::Value = if let Ok(n) = value.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else if let Ok(n) = value.parse::<f64>() {
                    serde_json::Number::from_f64(n)
                        .map(serde_json::Value::Number)
                        .unwrap_or_else(|| serde_json::Value::String(value.to_string()))
                } else if value == "true" {
                    serde_json::Value::Bool(true)
                } else if value == "false" {
                    serde_json::Value::Bool(false)
                } else if value == "null" {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(value.to_string())
                };
                
                return Ok(match *op {
                    "=" => Self::eq(field, json_value),
                    "!=" => Self::ne(field, json_value),
                    ">" => Self::gt(field, json_value),
                    ">=" => Self::gte(field, json_value),
                    "<" => Self::lt(field, json_value),
                    "<=" => Self::lte(field, json_value),
                    "~" => Self::contains(field, value),
                    "^" => Self::starts_with(field, value),
                    "$" => Self::ends_with(field, value),
                    _ => unreachable!(),
                });
            }
        }
        
        Err(format!("Invalid filter syntax: {}", s))
    }
    
    /// Parse multiple filters from a string, combining with AND
    /// 
    /// Filters are separated by `;` or newlines
    pub fn parse_many(s: &str) -> Result<Self, String> {
        let filters: Result<Vec<_>, _> = s
            .split([';', '\n'])
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(Self::parse)
            .collect();
        
        let filters = filters?;
        
        if filters.is_empty() {
            return Err("No filters provided".to_string());
        }
        
        if filters.len() == 1 {
            Ok(filters.into_iter().next().unwrap())
        } else {
            Ok(Self::and(filters))
        }
    }
}

/// Validate a field name for safe use in SQL
/// Only allows alphanumeric, underscore, and dot characters
fn validate_field_name(field: &str) -> Result<&str, RagError> {
    if field.is_empty() {
        return Err(RagError::QueryFailed("Empty field name".into()));
    }
    if field.len() > 128 {
        return Err(RagError::QueryFailed("Field name too long".into()));
    }
    if !field.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.') {
        return Err(RagError::QueryFailed(
            format!("Invalid field name '{}': only alphanumeric, underscore, and dot allowed", field)
        ));
    }
    Ok(field)
}

/// Convert a JSON value to a string for use as a SQL parameter
fn json_value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        _ => value.to_string(),
    }
}

/// A document with its embedding stored in the vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier
    pub id: i64,
    /// Text content
    pub content: String,
    /// Optional metadata as JSON
    pub metadata: Option<serde_json::Value>,
    /// Similarity score from search (only populated in search results)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// A document to be inserted (without ID)
#[derive(Debug, Clone)]
pub struct NewDocument {
    /// Text content
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// RAG vector store backed by PostgreSQL + pgvector
pub struct RagStore {
    pool: Pool,
    config: RagConfig,
}

impl RagStore {
    /// Connect to the vector store
    pub async fn connect(config: RagConfig) -> RagResult<Self> {
        // Validate config first
        config.validate()?;
        
        let mut pg_config = Config::new();
        
        // Parse connection string
        let url = url::Url::parse(config.connection_string())
            .map_err(|e| RagError::ConfigError(format!("Invalid connection string: {}", e)))?;
        
        pg_config.host = url.host_str().map(String::from);
        pg_config.port = url.port();
        pg_config.user = if url.username().is_empty() { None } else { Some(url.username().to_string()) };
        pg_config.password = url.password().map(String::from);
        pg_config.dbname = Some(url.path().trim_start_matches('/').to_string());
        
        let pool = pg_config
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        // Test connection
        let client = pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        // Verify pgvector extension is available
        client.query_one("SELECT extversion FROM pg_extension WHERE extname = 'vector'", &[])
            .await
            .map_err(|_| RagError::ConnectionFailed(
                "pgvector extension not installed. Run: CREATE EXTENSION vector;".into()
            ))?;
        
        Ok(Self { pool, config })
    }
    
    /// Connect using configuration loaded from file and/or environment
    pub async fn connect_with_config(config_path: Option<&str>) -> RagResult<Self> {
        let config = RagConfig::load(config_path)?;
        Self::connect(config).await
    }
    
    /// Create the embeddings table if it doesn't exist
    ///
    /// The table schema and indexes are driven by the [`RagConfig`]:
    /// - When `search_type == Hybrid`, a generated `content_tsv` tsvector
    ///   column and a GIN index on it are added for full-text keyword search.
    /// - The vector index type (HNSW or IVFFlat) is determined by
    ///   [`IndexType`](super::IndexType) from the config.
    pub async fn create_table(&self) -> RagResult<()> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

        let tsv_column = if self.config.search_type() == super::SearchType::Hybrid {
            format!(
                ", content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('{}', content)) STORED",
                self.config.text_search_language()
            )
        } else {
            String::new()
        };

        let create_table = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({}) NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(){}
            )
            "#,
            self.config.table_name(),
            self.config.embedding_dim(),
            tsv_column,
        );

        client.execute(&create_table, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

        // Create vector and (optionally) GIN indexes
        self.create_index_inner(&client).await?;

        Ok(())
    }

    /// Recreate the vector (and optional GIN) indexes.
    ///
    /// This is useful after a large bulk insert where you want to drop
    /// and rebuild the index for optimal search performance.
    pub async fn create_index(&self) -> RagResult<()> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        self.create_index_inner(&client).await
    }

    /// Shared helper that creates vector and GIN indexes using an
    /// already-acquired pool client.
    async fn create_index_inner(&self, client: &deadpool_postgres::Object) -> RagResult<()> {
        let index_type = self.config.index_type();
        let ops = self.config.distance_metric().index_ops();
        let (method, ops_class, with_clause) = index_type.index_sql(ops);

        if !method.is_empty() {
            let create_vec_idx = format!(
                "CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} USING {method} (embedding {ops_class}) {with_clause}",
                table = self.config.table_name(),
                method = method,
                ops_class = ops_class,
                with_clause = with_clause,
            );

            // Index creation may fail if table is empty (IVFFlat); that's okay
            let _ = client.execute(&create_vec_idx, &[]).await;
        }

        // GIN index for hybrid text search
        if self.config.search_type() == super::SearchType::Hybrid {
            let create_gin_idx = format!(
                "CREATE INDEX IF NOT EXISTS {}_content_tsv_idx ON {} USING gin (content_tsv)",
                self.config.table_name(),
                self.config.table_name(),
            );
            let _ = client.execute(&create_gin_idx, &[]).await;
        }

        Ok(())
    }

    /// Set the HNSW `ef_search` parameter for the current connection,
    /// controlling the trade-off between search quality and speed.
    ///
    /// Higher values yield more accurate results at the cost of latency.
    /// This is a session-level setting and does not persist across connections.
    pub async fn set_hnsw_ef_search(&self, ef_search: u16) -> RagResult<()> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

        let query = format!("SET hnsw.ef_search = {}", ef_search);
        client.execute(&query, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

        Ok(())
    }
    
    /// Insert a document with its embedding
    pub async fn insert(&self, doc: NewDocument) -> RagResult<i64> {
        if doc.embedding.len() != self.config.embedding_dim() {
            return Err(RagError::DimensionMismatch {
                expected: self.config.embedding_dim(),
                actual: doc.embedding.len(),
            });
        }
        
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let embedding = Vector::from(doc.embedding);
        
        let query = format!(
            "INSERT INTO {} (content, embedding, metadata) VALUES ($1, $2, $3) RETURNING id",
            self.config.table_name()
        );
        
        let row = client.query_one(&query, &[&doc.content, &embedding, &doc.metadata]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.get(0))
    }
    
    /// Insert multiple documents in a batch
    pub async fn insert_batch(&self, docs: Vec<NewDocument>) -> RagResult<Vec<i64>> {
        let mut ids = Vec::with_capacity(docs.len());
        
        // TODO: Use COPY for better performance with large batches
        for doc in docs {
            let id = self.insert(doc).await?;
            ids.push(id);
        }
        
        Ok(ids)
    }
    
    /// Search for similar documents using vector similarity
    pub async fn search(&self, query_embedding: &[f32], limit: Option<usize>) -> RagResult<Vec<Document>> {
        self.search_with_filter(query_embedding, limit, None).await
    }

    /// Search for similar documents with metadata filtering
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llama_gguf::rag::{RagStore, MetadataFilter};
    ///
    /// // Search with a simple filter
    /// let filter = MetadataFilter::eq("type", "documentation");
    /// let results = store.search_with_filter(&embedding, Some(10), Some(filter)).await?;
    ///
    /// // Search with multiple filters
    /// let filter = MetadataFilter::and(vec![
    ///     MetadataFilter::eq("source", "docs"),
    ///     MetadataFilter::gte("version", 2),
    /// ]);
    /// let results = store.search_with_filter(&embedding, Some(5), Some(filter)).await?;
    /// ```
    pub async fn search_with_filter(
        &self,
        query_embedding: &[f32],
        limit: Option<usize>,
        filter: Option<MetadataFilter>,
    ) -> RagResult<Vec<Document>> {
        self.search_vector_inner(query_embedding, limit, filter).await
    }

    /// Core vector-similarity search implementation.
    async fn search_vector_inner(
        &self,
        query_embedding: &[f32],
        limit: Option<usize>,
        filter: Option<MetadataFilter>,
    ) -> RagResult<Vec<Document>> {
        if query_embedding.len() != self.config.embedding_dim() {
            return Err(RagError::DimensionMismatch {
                expected: self.config.embedding_dim(),
                actual: query_embedding.len(),
            });
        }

        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

        let embedding = Vector::from(query_embedding.to_vec());
        let limit = limit.unwrap_or(self.config.max_results()) as i64;
        let operator = self.config.distance_metric().operator();

        // For cosine distance, convert to similarity (1 - distance)
        let score_expr = match self.config.distance_metric() {
            super::DistanceMetric::Cosine => format!("1 - (embedding {} $1)", operator),
            super::DistanceMetric::L2 => format!("1 / (1 + (embedding {} $1))", operator),
            super::DistanceMetric::InnerProduct => format!("-(embedding {} $1)", operator),
        };

        let min_sim = self.config.min_similarity();

        // Build the WHERE clause
        let (filter_clause, filter_params) = if let Some(f) = filter {
            let (sql, params) = f.to_sql(3)?; // Start after $1 (embedding), $2 (min_sim), $3 (limit)
            (format!(" AND {}", sql), params)
        } else {
            (String::new(), Vec::new())
        };

        let query = format!(
            r#"
            SELECT id, content, metadata, {} as score
            FROM {}
            WHERE {} >= $2{}
            ORDER BY embedding {} $1
            LIMIT $3
            "#,
            score_expr,
            self.config.table_name(),
            score_expr,
            filter_clause,
            operator
        );

        // Build params dynamically
        use tokio_postgres::types::ToSql;
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&embedding, &min_sim, &limit];
        let filter_param_refs: Vec<&str> = filter_params.iter().map(|s| s.as_str()).collect();
        for p in &filter_param_refs {
            params.push(p);
        }

        let rows = client.query(&query, &params).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

        let docs = rows.iter().map(|row| {
            Document {
                id: row.get(0),
                content: row.get(1),
                metadata: row.get(2),
                score: Some(row.get(3)),
            }
        }).collect();

        Ok(docs)
    }

    /// Perform a keyword-only search using PostgreSQL full-text search.
    ///
    /// Returns `(id, ts_rank_score)` pairs ordered by relevance.
    /// Requires `search_type = "hybrid"` in the config so that the
    /// `content_tsv` generated column exists.
    pub async fn search_keyword(
        &self,
        query_text: &str,
        limit: usize,
        filter: Option<MetadataFilter>,
    ) -> RagResult<Vec<(i64, f32)>> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

        let lang = self.config.text_search_language();

        let (filter_clause, filter_params) = if let Some(f) = filter {
            let (sql, params) = f.to_sql(2)?; // $1 = query_text, $2 = limit
            (format!(" AND {}", sql), params)
        } else {
            (String::new(), Vec::new())
        };

        let limit_i64 = limit as i64;

        let query = format!(
            r#"
            SELECT id, ts_rank(content_tsv, plainto_tsquery('{lang}', $1)) as rank
            FROM {table}
            WHERE content_tsv @@ plainto_tsquery('{lang}', $1){filter}
            ORDER BY rank DESC
            LIMIT $2
            "#,
            lang = lang,
            table = self.config.table_name(),
            filter = filter_clause,
        );

        use tokio_postgres::types::ToSql;
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&query_text, &limit_i64];
        let filter_param_refs: Vec<&str> = filter_params.iter().map(|s| s.as_str()).collect();
        for p in &filter_param_refs {
            params.push(p);
        }

        let rows = client.query(&query, &params).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

        let results = rows.iter().map(|row| {
            let id: i64 = row.get(0);
            let score: f32 = row.get(1);
            (id, score)
        }).collect();

        Ok(results)
    }

    /// Perform a hybrid search combining vector similarity and keyword
    /// relevance via Reciprocal Rank Fusion (RRF).
    ///
    /// Both the vector and keyword searches are run, then their results
    /// are merged using [`rrf_fuse`] and the final documents are returned
    /// sorted by the fused score.
    pub async fn search_hybrid(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        limit: Option<usize>,
        filter: Option<MetadataFilter>,
    ) -> RagResult<Vec<Document>> {
        let limit = limit.unwrap_or(self.config.max_results());
        let oversampled = limit * self.config.hybrid_oversampling() as usize;

        // Run vector search (oversampled)
        let vec_docs = self.search_vector_inner(query_embedding, Some(oversampled), filter.clone()).await?;
        let vector_results: Vec<(i64, f32)> = vec_docs.iter().map(|d| (d.id, d.score.unwrap_or(0.0))).collect();

        // Run keyword search (oversampled)
        let keyword_results = self.search_keyword(query_text, oversampled, filter).await?;

        // Fuse with RRF
        let fused = rrf_fuse(&vector_results, &keyword_results, self.config.rrf_k(), limit);

        // Build a map of id -> Document from vector results
        let mut doc_map: std::collections::HashMap<i64, Document> = vec_docs.into_iter().map(|d| (d.id, d)).collect();

        // Identify keyword-only IDs that we need to fetch
        let missing_ids: Vec<i64> = fused.iter()
            .filter(|(id, _)| !doc_map.contains_key(id))
            .map(|(id, _)| *id)
            .collect();

        if !missing_ids.is_empty() {
            let client = self.pool.get().await
                .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

            // Fetch missing docs by ID
            let placeholders: Vec<String> = (1..=missing_ids.len()).map(|i| format!("${}", i)).collect();
            let query = format!(
                "SELECT id, content, metadata FROM {} WHERE id IN ({})",
                self.config.table_name(),
                placeholders.join(", ")
            );

            use tokio_postgres::types::ToSql;
            let params: Vec<&(dyn ToSql + Sync)> = missing_ids.iter().map(|id| id as &(dyn ToSql + Sync)).collect();

            let rows = client.query(&query, &params).await
                .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

            for row in &rows {
                let doc = Document {
                    id: row.get(0),
                    content: row.get(1),
                    metadata: row.get(2),
                    score: None,
                };
                doc_map.insert(doc.id, doc);
            }
        }

        // Assemble final results in fused order
        let results = fused.into_iter().filter_map(|(id, score)| {
            doc_map.remove(&id).map(|mut doc| {
                doc.score = Some(score);
                doc
            })
        }).collect();

        Ok(results)
    }

    /// Count documents matching a filter
    pub async fn count_with_filter(&self, filter: Option<MetadataFilter>) -> RagResult<i64> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let (filter_clause, filter_params) = if let Some(f) = filter {
            let (sql, params) = f.to_sql(0)?;
            (format!(" WHERE {}", sql), params)
        } else {
            (String::new(), Vec::new())
        };
        
        let query = format!(
            "SELECT COUNT(*) FROM {}{}",
            self.config.table_name(),
            filter_clause
        );
        
        use tokio_postgres::types::ToSql;
        let filter_param_refs: Vec<&str> = filter_params.iter().map(|s| s.as_str()).collect();
        let params: Vec<&(dyn ToSql + Sync)> = filter_param_refs.iter().map(|p| p as &(dyn ToSql + Sync)).collect();
        
        let row = client.query_one(&query, &params).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.get(0))
    }
    
    /// Delete documents matching a filter
    pub async fn delete_with_filter(&self, filter: MetadataFilter) -> RagResult<u64> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let (filter_sql, filter_params) = filter.to_sql(0)?;
        
        let query = format!(
            "DELETE FROM {} WHERE {}",
            self.config.table_name(),
            filter_sql
        );
        
        use tokio_postgres::types::ToSql;
        let filter_param_refs: Vec<&str> = filter_params.iter().map(|s| s.as_str()).collect();
        let params: Vec<&(dyn ToSql + Sync)> = filter_param_refs.iter().map(|p| p as &(dyn ToSql + Sync)).collect();
        
        let affected = client.execute(&query, &params).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(affected)
    }
    
    /// List unique values for a metadata field
    pub async fn list_metadata_values(&self, field: &str, limit: Option<usize>) -> RagResult<Vec<String>> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let limit = limit.unwrap_or(100) as i64;
        
        let field = validate_field_name(field)?;

        let query = format!(
            "SELECT DISTINCT metadata->>'{}' as val FROM {} WHERE metadata ? '{}' ORDER BY val LIMIT $1",
            field,
            self.config.table_name(),
            field
        );
        
        let rows = client.query(&query, &[&limit]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        let values = rows.iter()
            .filter_map(|row| row.get::<_, Option<String>>(0))
            .collect();
        
        Ok(values)
    }
    
    /// Get a document by ID
    pub async fn get(&self, id: i64) -> RagResult<Option<Document>> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!(
            "SELECT id, content, metadata FROM {} WHERE id = $1",
            self.config.table_name()
        );
        
        let row = client.query_opt(&query, &[&id]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.map(|r| Document {
            id: r.get(0),
            content: r.get(1),
            metadata: r.get(2),
            score: None,
        }))
    }
    
    /// Delete a document by ID
    pub async fn delete(&self, id: i64) -> RagResult<bool> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!("DELETE FROM {} WHERE id = $1", self.config.table_name());
        let affected = client.execute(&query, &[&id]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(affected > 0)
    }
    
    /// Count total documents
    pub async fn count(&self) -> RagResult<i64> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!("SELECT COUNT(*) FROM {}", self.config.table_name());
        let row = client.query_one(&query, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.get(0))
    }
    
    /// Clear all documents from the table
    pub async fn clear(&self) -> RagResult<u64> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!("DELETE FROM {}", self.config.table_name());
        let affected = client.execute(&query, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(affected)
    }
    
    /// Get the configuration
    pub fn config(&self) -> &RagConfig {
        &self.config
    }
}

/// Builder for creating RAG context from search results
pub struct RagContextBuilder {
    docs: Vec<Document>,
    separator: String,
    max_tokens: Option<usize>,
    include_scores: bool,
}

impl RagContextBuilder {
    /// Create a new context builder from search results
    pub fn new(docs: Vec<Document>) -> Self {
        Self {
            docs,
            separator: "\n\n".to_string(),
            max_tokens: None,
            include_scores: false,
        }
    }
    
    /// Set the separator between documents
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
    
    /// Set approximate maximum tokens (characters / 4)
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }
    
    /// Include similarity scores in output
    pub fn with_scores(mut self, include: bool) -> Self {
        self.include_scores = include;
        self
    }
    
    /// Build the context string
    pub fn build(self) -> String {
        let mut parts = Vec::new();
        let mut total_chars = 0;
        let max_chars = self.max_tokens.map(|t| t * 4);
        
        for doc in &self.docs {
            let part = if self.include_scores {
                if let Some(score) = doc.score {
                    format!("[{:.2}] {}", score, doc.content)
                } else {
                    doc.content.clone()
                }
            } else {
                doc.content.clone()
            };
            
            if let Some(max) = max_chars
                && total_chars + part.len() > max {
                    break;
                }
            
            total_chars += part.len() + self.separator.len();
            parts.push(part);
        }
        
        parts.join(&self.separator)
    }
    
    /// Build a prompt with context and question
    pub fn build_prompt(self, question: &str) -> String {
        let context = self.build();
        format!(
            "Use the following context to answer the question.\n\n\
            Context:\n{}\n\n\
            Question: {}\n\n\
            Answer:",
            context,
            question
        )
    }
}

/// Reciprocal Rank Fusion - merge two ranked result lists
///
/// Each input is a list of (id, score) pairs ordered by relevance.
/// Returns fused results sorted by RRF score, limited to `limit`.
pub(crate) fn rrf_fuse(
    vector_results: &[(i64, f32)],
    keyword_results: &[(i64, f32)],
    k: u32,
    limit: usize,
) -> Vec<(i64, f32)> {
    use std::collections::HashMap;

    let mut scores: HashMap<i64, f32> = HashMap::new();

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        *scores.entry(*id).or_default() += 1.0 / (k as f32 + rank as f32 + 1.0);
    }

    for (rank, (id, _)) in keyword_results.iter().enumerate() {
        *scores.entry(*id).or_default() += 1.0 / (k as f32 + rank as f32 + 1.0);
    }

    let mut fused: Vec<(i64, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_field_name_valid() {
        assert!(validate_field_name("source").is_ok());
        assert!(validate_field_name("my_field").is_ok());
        assert!(validate_field_name("field123").is_ok());
        assert!(validate_field_name("a.b").is_ok());
    }

    #[test]
    fn test_validate_field_name_rejects_sql_injection() {
        assert!(validate_field_name("'; DROP TABLE --").is_err());
        assert!(validate_field_name("field; DELETE").is_err());
        assert!(validate_field_name("").is_err());
        assert!(validate_field_name("a\"b").is_err());
    }

    #[test]
    fn test_rrf_fusion_basic() {
        let vector_results = vec![(1i64, 0.95f32), (2, 0.85), (3, 0.75)];
        let keyword_results = vec![(2i64, 0.9f32), (3, 0.8), (4, 0.7)];
        let fused = rrf_fuse(&vector_results, &keyword_results, 60, 3);
        // ID 2 appears in both, should rank highest
        assert_eq!(fused[0].0, 2);
        assert!(fused.iter().all(|(_, score)| *score > 0.0));
        assert!(fused.len() <= 3);
    }

    #[test]
    fn test_rrf_fusion_disjoint() {
        let vector_results = vec![(1i64, 0.9f32)];
        let keyword_results = vec![(2i64, 0.9f32)];
        let fused = rrf_fuse(&vector_results, &keyword_results, 60, 10);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_rrf_fusion_empty() {
        let fused = rrf_fuse(&[], &[], 60, 5);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_metadata_filter_to_sql_validates_fields() {
        let filter = MetadataFilter::eq("valid_field", "value");
        assert!(filter.to_sql(0).is_ok());

        let bad_filter = MetadataFilter::eq("'; DROP TABLE", "value");
        assert!(bad_filter.to_sql(0).is_err());
    }
}
