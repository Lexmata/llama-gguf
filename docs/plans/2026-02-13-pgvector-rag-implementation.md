# pgvector RAG Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the pgvector RAG pipeline fully functional with real embeddings, HNSW indexes, RRF hybrid search, pipelined batch inserts, and reranking.

**Architecture:** Wire the existing `EmbeddingExtractor` into `KnowledgeBase`, enhance `RagStore` with HNSW indexes and hybrid tsvector+pgvector search using RRF fusion, replace O(n) inserts with pipelined batches.

**Tech Stack:** Rust, PostgreSQL + pgvector, tokio-postgres, deadpool-postgres, glob crate

---

### Task 1: Add `glob` dependency to Cargo.toml

**Files:**
- Modify: `Cargo.toml:30` (rag feature) and `Cargo.toml:63` (dependencies)

**Step 1: Add glob dependency**

In `Cargo.toml`, add `glob` as an optional dependency and include it in the `rag` feature:

```toml
# Line 30 - update rag feature to include glob:
rag = ["dep:tokio-postgres", "dep:pgvector", "dep:deadpool-postgres", "dep:tokio", "dep:url", "dep:glob"]

# After line 63 (after url), add:
glob = { version = "0.3", optional = true }
```

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully (downloads glob crate)

**Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: add glob dependency for RAG directory walk"
```

---

### Task 2: Add IndexType enum and search config extensions to config.rs

**Files:**
- Modify: `src/rag/config.rs:67-77` (EmbeddingsConfig), `src/rag/config.rs:80-94` (SearchConfig)
- Test: `src/rag/config.rs` (inline tests)

**Step 1: Write the failing tests**

Add at the bottom of `src/rag/config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_type_default_is_hnsw() {
        let idx = IndexType::default();
        assert!(matches!(idx, IndexType::Hnsw { m: 16, ef_construction: 64 }));
    }

    #[test]
    fn test_index_type_hnsw_sql() {
        let idx = IndexType::Hnsw { m: 16, ef_construction: 64 };
        let (method, ops, with) = idx.index_sql("vector_cosine_ops");
        assert_eq!(method, "hnsw");
        assert_eq!(ops, "vector_cosine_ops");
        assert_eq!(with, "WITH (m = 16, ef_construction = 64)");
    }

    #[test]
    fn test_index_type_ivfflat_sql() {
        let idx = IndexType::IvfFlat { lists: 100 };
        let (method, ops, with) = idx.index_sql("vector_cosine_ops");
        assert_eq!(method, "ivfflat");
        assert_eq!(ops, "vector_cosine_ops");
        assert_eq!(with, "WITH (lists = 100)");
    }

    #[test]
    fn test_search_config_defaults() {
        let config = SearchConfig::default();
        assert_eq!(config.rrf_k, 60);
        assert_eq!(config.hybrid_oversampling, 2);
        assert_eq!(config.text_search_language, "english");
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = RagConfig::new("postgres://user:pass@localhost/db");
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: RagConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.connection_string(), config.connection_string());
    }

    #[test]
    fn test_index_type_serde() {
        let config_str = r#"
[database]
connection_string = "postgres://localhost/test"

[embeddings]
dimension = 384
index_type = "hnsw"
hnsw_m = 32
hnsw_ef_construction = 128

[search]
distance_metric = "cosine"
search_type = "hybrid"
rrf_k = 60
"#;
        let config: RagConfig = toml::from_str(config_str).unwrap();
        assert!(matches!(config.embeddings.index_type, IndexType::Hnsw { m: 32, ef_construction: 128 }));
        assert_eq!(config.search.search_type, SearchType::Hybrid);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --features rag --lib rag::config::tests`
Expected: FAIL — `IndexType` not defined, `index_sql` not defined, new fields not on `SearchConfig`

**Step 3: Implement IndexType and extend configs**

Add `IndexType` enum after `DistanceMetric` in `src/rag/config.rs`:

```rust
/// Index type for vector similarity search
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "index_type", rename_all = "snake_case")]
pub enum IndexType {
    /// HNSW index (default) - better recall, no training step
    Hnsw {
        #[serde(default = "default_hnsw_m", alias = "hnsw_m")]
        m: u16,
        #[serde(default = "default_hnsw_ef_construction", alias = "hnsw_ef_construction")]
        ef_construction: u16,
    },
    /// IVFFlat index - good for very large datasets with memory constraints
    IvfFlat {
        #[serde(default = "default_ivfflat_lists")]
        lists: u16,
    },
    /// No index
    None,
}

fn default_hnsw_m() -> u16 { 16 }
fn default_hnsw_ef_construction() -> u16 { 64 }
fn default_ivfflat_lists() -> u16 { 100 }

impl Default for IndexType {
    fn default() -> Self {
        Self::Hnsw {
            m: default_hnsw_m(),
            ef_construction: default_hnsw_ef_construction(),
        }
    }
}

impl IndexType {
    /// Generate the SQL fragments for CREATE INDEX
    /// Returns (method, operator_class, with_clause)
    pub fn index_sql(&self, ops: &str) -> (&'static str, &str, String) {
        match self {
            Self::Hnsw { m, ef_construction } => {
                ("hnsw", ops, format!("WITH (m = {}, ef_construction = {})", m, ef_construction))
            }
            Self::IvfFlat { lists } => {
                ("ivfflat", ops, format!("WITH (lists = {})", lists))
            }
            Self::None => ("", ops, String::new()),
        }
    }
}
```

Add `index_type` field to `EmbeddingsConfig`:

```rust
pub struct EmbeddingsConfig {
    #[serde(default = "default_table_name")]
    pub table_name: String,
    #[serde(default = "default_embedding_dim")]
    pub dimension: usize,
    /// Index type for vector search
    #[serde(flatten, default)]
    pub index_type: IndexType,
}
```

Update `Default for EmbeddingsConfig`:

```rust
impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            table_name: default_table_name(),
            dimension: default_embedding_dim(),
            index_type: IndexType::default(),
        }
    }
}
```

Move `SearchType` from `knowledge_base.rs` to `config.rs` (it's needed here for config). Add new fields to `SearchConfig`:

```rust
/// Type of search to perform
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SearchType {
    #[default]
    Semantic,
    Hybrid,
}

pub struct SearchConfig {
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
    #[serde(default)]
    pub distance_metric: DistanceMetric,
    /// Search type (semantic or hybrid)
    #[serde(default)]
    pub search_type: SearchType,
    /// RRF constant for hybrid search (higher = more weight to lower-ranked results)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,
    /// Oversampling factor for hybrid search candidates
    #[serde(default = "default_hybrid_oversampling")]
    pub hybrid_oversampling: u32,
    /// PostgreSQL text search configuration language
    #[serde(default = "default_text_search_language")]
    pub text_search_language: String,
}

fn default_rrf_k() -> u32 { 60 }
fn default_hybrid_oversampling() -> u32 { 2 }
fn default_text_search_language() -> String { "english".to_string() }
```

Update `Default for SearchConfig`:

```rust
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
```

Add accessor methods to `RagConfig`:

```rust
pub fn index_type(&self) -> &IndexType {
    &self.embeddings.index_type
}

pub fn search_type(&self) -> SearchType {
    self.search.search_type
}

pub fn rrf_k(&self) -> u32 {
    self.search.rrf_k
}

pub fn hybrid_oversampling(&self) -> u32 {
    self.search.hybrid_oversampling
}

pub fn text_search_language(&self) -> &str {
    &self.search.text_search_language
}
```

Update `example_config()` to include the new fields.

Also update `knowledge_base.rs` to import `SearchType` from `super` (config) instead of defining it locally. Remove the duplicate `SearchType` definition from `knowledge_base.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test --features rag --lib rag::config::tests`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/rag/config.rs src/rag/knowledge_base.rs
git commit -m "feat(rag): add IndexType enum and hybrid search config fields"
```

---

### Task 3: Harden field name validation in store.rs

**Files:**
- Modify: `src/rag/store.rs:417-420` (escape_field function)
- Test: `src/rag/store.rs` (inline tests)

**Step 1: Write the failing tests**

Add or extend the test module at the bottom of `src/rag/store.rs`:

```rust
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
        // Two result sets with overlapping IDs
        let vector_results = vec![(1i64, 0.95f32), (2, 0.85), (3, 0.75)];
        let keyword_results = vec![(2i64, 0.9f32), (3, 0.8), (4, 0.7)];

        let fused = rrf_fuse(&vector_results, &keyword_results, 60, 3);

        // ID 2 appears in both, should rank highest
        assert_eq!(fused[0].0, 2);
        // All results should have positive scores
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
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --features rag --lib rag::store::tests`
Expected: FAIL — `validate_field_name` and `rrf_fuse` not defined

**Step 3: Implement validate_field_name and rrf_fuse**

Replace the `escape_field` function with `validate_field_name`:

```rust
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
```

Update all callers of `escape_field` in the `MetadataFilter::to_sql_inner` method to use `validate_field_name` instead. Since `to_sql_inner` returns a String and doesn't propagate errors, we need to make `to_sql` return `RagResult<(String, Vec<String>)>`. This is a signature change, so update all callers in `store.rs` that use `filter.to_sql()`.

Alternatively, keep `to_sql` infallible and validate field names at filter construction time. The cleaner approach: validate in `to_sql`, make it return `Result`:

```rust
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
        // ... same pattern for all variants
    }
}
```

Add the RRF fusion function:

```rust
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
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --features rag --lib rag::store::tests`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/rag/store.rs
git commit -m "feat(rag): add field validation and RRF fusion function"
```

---

### Task 4: Update RagStore::create_table for HNSW/IVFFlat and tsvector

**Files:**
- Modify: `src/rag/store.rs:507-543` (create_table method)

**Step 1: Update create_table**

Replace the existing `create_table` method:

```rust
pub async fn create_table(&self) -> RagResult<()> {
    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

    let tsvector_col = if self.config.search_type() == super::SearchType::Hybrid {
        format!(
            ",\n                content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('{}', content)) STORED",
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
        tsvector_col,
    );

    client.execute(&create_table, &[]).await
        .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

    // Create vector index
    self.create_vector_index(&client).await?;

    // Create GIN index for tsvector if hybrid search is enabled
    if self.config.search_type() == super::SearchType::Hybrid {
        let gin_index = format!(
            "CREATE INDEX IF NOT EXISTS {}_content_tsv_idx ON {} USING gin (content_tsv)",
            self.config.table_name(),
            self.config.table_name()
        );
        let _ = client.execute(&gin_index, &[]).await;
    }

    Ok(())
}

/// Create or recreate the vector similarity index
pub async fn create_index(&self) -> RagResult<()> {
    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
    self.create_vector_index(&client).await
}

async fn create_vector_index(&self, client: &deadpool_postgres::Object) -> RagResult<()> {
    let index_type = self.config.index_type();
    if matches!(index_type, super::IndexType::None) {
        return Ok(());
    }

    let ops = self.config.distance_metric().index_ops();
    let (method, ops, with_clause) = index_type.index_sql(ops);

    let create_index = format!(
        "CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} USING {method} (embedding {ops}) {with}",
        table = self.config.table_name(),
        method = method,
        ops = ops,
        with = with_clause,
    );

    // Index creation may fail on empty tables (IVFFlat), that's okay
    let _ = client.execute(&create_index, &[]).await;

    Ok(())
}

/// Set HNSW ef_search parameter for the current session
/// Higher values = better recall at the cost of speed
pub async fn set_hnsw_ef_search(&self, ef_search: u16) -> RagResult<()> {
    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

    client.execute(&format!("SET hnsw.ef_search = {}", ef_search), &[]).await
        .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

    Ok(())
}
```

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/rag/store.rs
git commit -m "feat(rag): HNSW and IVFFlat index support in create_table"
```

---

### Task 5: Implement hybrid search in RagStore

**Files:**
- Modify: `src/rag/store.rs:606-679` (search_with_filter) — add hybrid dispatch
- Add new method: `search_keyword` and `search_hybrid`

**Step 1: Add search_keyword and search_hybrid methods**

Add after the existing `search_with_filter` method:

```rust
/// Keyword search using PostgreSQL full-text search
async fn search_keyword(
    &self,
    query_text: &str,
    limit: usize,
    filter: Option<MetadataFilter>,
) -> RagResult<Vec<(i64, f32)>> {
    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

    let lang = self.config.text_search_language();
    let limit_i64 = limit as i64;

    let (filter_clause, filter_params) = if let Some(f) = filter {
        let (sql, params) = f.to_sql(2)?; // After $1 (query) and $2 (limit)
        (format!(" AND {}", sql), params)
    } else {
        (String::new(), Vec::new())
    };

    let query_sql = format!(
        r#"
        SELECT id, ts_rank(content_tsv, plainto_tsquery('{}', $1)) as rank
        FROM {}
        WHERE content_tsv @@ plainto_tsquery('{}', $1){}
        ORDER BY rank DESC
        LIMIT $2
        "#,
        lang,
        self.config.table_name(),
        lang,
        filter_clause,
    );

    use tokio_postgres::types::ToSql;
    let mut params: Vec<&(dyn ToSql + Sync)> = vec![&query_text, &limit_i64];
    let filter_param_refs: Vec<&str> = filter_params.iter().map(|s| s.as_str()).collect();
    for p in &filter_param_refs {
        params.push(p);
    }

    let rows = client.query(&query_sql, &params).await
        .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

    Ok(rows.iter().map(|row| {
        let id: i64 = row.get(0);
        let rank: f32 = row.get(1);
        (id, rank)
    }).collect())
}

/// Hybrid search combining vector similarity and keyword search with RRF
pub async fn search_hybrid(
    &self,
    query_embedding: &[f32],
    query_text: &str,
    limit: Option<usize>,
    filter: Option<MetadataFilter>,
) -> RagResult<Vec<Document>> {
    let limit = limit.unwrap_or(self.config.max_results());
    let oversampled = limit * self.config.hybrid_oversampling() as usize;

    // Run vector search and keyword search
    // Clone filter for second query (if present)
    let filter_clone = filter.clone();

    let vector_docs = self.search_with_filter_inner(
        query_embedding, Some(oversampled), filter, false
    ).await?;

    let keyword_ids = self.search_keyword(query_text, oversampled, filter_clone).await?;

    // Build vector results as (id, score)
    let vector_results: Vec<(i64, f32)> = vector_docs.iter()
        .map(|d| (d.id, d.score.unwrap_or(0.0)))
        .collect();

    // RRF fusion
    let fused = rrf_fuse(&vector_results, &keyword_ids, self.config.rrf_k(), limit);

    // Collect the fused IDs in order and build result set
    let fused_ids: Vec<i64> = fused.iter().map(|(id, _)| *id).collect();
    let fused_scores: std::collections::HashMap<i64, f32> = fused.into_iter().collect();

    // Build result from already-fetched docs + fetch any keyword-only results
    let mut doc_map: std::collections::HashMap<i64, Document> = vector_docs.into_iter()
        .map(|d| (d.id, d))
        .collect();

    // Fetch any docs that came from keyword search only
    let missing_ids: Vec<i64> = fused_ids.iter()
        .filter(|id| !doc_map.contains_key(id))
        .copied()
        .collect();

    for id in missing_ids {
        if let Some(doc) = self.get(id).await? {
            doc_map.insert(id, doc);
        }
    }

    // Assemble in fused order with RRF scores
    let results = fused_ids.iter()
        .filter_map(|id| {
            doc_map.remove(id).map(|mut doc| {
                doc.score = fused_scores.get(id).copied();
                doc
            })
        })
        .collect();

    Ok(results)
}
```

Rename the core of the existing `search_with_filter` to `search_with_filter_inner` (taking an extra `_hybrid: bool` param that's unused for now, just to distinguish). Then make `search_with_filter` dispatch:

```rust
pub async fn search_with_filter(
    &self,
    query_embedding: &[f32],
    limit: Option<usize>,
    filter: Option<MetadataFilter>,
) -> RagResult<Vec<Document>> {
    self.search_with_filter_inner(query_embedding, limit, filter, false).await
}

/// Inner vector-only search implementation
async fn search_with_filter_inner(
    &self,
    query_embedding: &[f32],
    limit: Option<usize>,
    filter: Option<MetadataFilter>,
    _is_hybrid_component: bool,
) -> RagResult<Vec<Document>> {
    // ... existing search_with_filter body, but use to_sql()? with ? operator
}
```

Also derive `Clone` on `MetadataFilter` (it already has `Clone` derived, good).

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/rag/store.rs
git commit -m "feat(rag): implement hybrid search with tsvector and RRF fusion"
```

---

### Task 6: Pipelined batch inserts

**Files:**
- Modify: `src/rag/store.rs:570-581` (insert_batch method)

**Step 1: Replace the O(n) insert_batch with pipelined version**

```rust
/// Insert multiple documents in a batch using pipelined queries
pub async fn insert_batch(&self, docs: Vec<NewDocument>) -> RagResult<Vec<i64>> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }

    // Validate all dimensions up front
    for doc in &docs {
        if doc.embedding.len() != self.config.embedding_dim() {
            return Err(RagError::DimensionMismatch {
                expected: self.config.embedding_dim(),
                actual: doc.embedding.len(),
            });
        }
    }

    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

    let query = format!(
        "INSERT INTO {} (content, embedding, metadata) VALUES ($1, $2, $3) RETURNING id",
        self.config.table_name()
    );

    let statement = client.prepare(&query).await
        .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;

    // Process in batches to control memory pressure
    let batch_size = 100;
    let mut all_ids = Vec::with_capacity(docs.len());

    for chunk in docs.chunks(batch_size) {
        // Build a transaction for each batch
        let transaction = client.transaction().await
            .map_err(|e| RagError::QueryFailed(format!("Failed to start transaction: {}", e)))?;

        let mut batch_ids = Vec::with_capacity(chunk.len());

        for doc in chunk {
            let embedding = pgvector::Vector::from(doc.embedding.clone());
            let row = transaction.query_one(&statement, &[&doc.content, &embedding, &doc.metadata]).await
                .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
            batch_ids.push(row.get::<_, i64>(0));
        }

        transaction.commit().await
            .map_err(|e| RagError::QueryFailed(format!("Failed to commit batch: {}", e)))?;

        all_ids.extend(batch_ids);
    }

    Ok(all_ids)
}
```

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/rag/store.rs
git commit -m "feat(rag): pipelined batch inserts with transaction batching"
```

---

### Task 7: Add upsert and health_check to RagStore

**Files:**
- Modify: `src/rag/store.rs` — add upsert and health_check methods

**Step 1: Add upsert method**

Add after the `insert` method:

```rust
/// Insert or update a document by ID
///
/// If a document with the given ID exists, updates its content, embedding, and metadata.
/// If not, inserts a new document (with auto-generated ID if id is None).
pub async fn upsert(&self, id: Option<i64>, doc: NewDocument) -> RagResult<i64> {
    if doc.embedding.len() != self.config.embedding_dim() {
        return Err(RagError::DimensionMismatch {
            expected: self.config.embedding_dim(),
            actual: doc.embedding.len(),
        });
    }

    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

    let embedding = pgvector::Vector::from(doc.embedding);

    if let Some(id) = id {
        let query = format!(
            r#"INSERT INTO {} (id, content, embedding, metadata)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (id) DO UPDATE SET
                   content = EXCLUDED.content,
                   embedding = EXCLUDED.embedding,
                   metadata = EXCLUDED.metadata
               RETURNING id"#,
            self.config.table_name()
        );

        let row = client.query_one(&query, &[&id, &doc.content, &embedding, &doc.metadata]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        Ok(row.get(0))
    } else {
        // No ID specified, just insert
        let query = format!(
            "INSERT INTO {} (content, embedding, metadata) VALUES ($1, $2, $3) RETURNING id",
            self.config.table_name()
        );
        let row = client.query_one(&query, &[&doc.content, &embedding, &doc.metadata]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        Ok(row.get(0))
    }
}

/// Check connection health
pub async fn health_check(&self) -> RagResult<()> {
    let client = self.pool.get().await
        .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;

    client.query_one("SELECT 1", &[]).await
        .map_err(|e| RagError::ConnectionFailed(format!("Health check failed: {}", e)))?;

    Ok(())
}
```

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/rag/store.rs
git commit -m "feat(rag): add upsert and health_check to RagStore"
```

---

### Task 8: Wire EmbeddingExtractor into rag/embedding.rs

**Files:**
- Modify: `src/rag/embedding.rs:1-80` — replace stubbed EmbeddingGenerator

**Step 1: Rewrite EmbeddingGenerator to use EmbeddingExtractor**

Replace the `EmbeddingGenerator` struct and impl with:

```rust
//! Embedding generation for RAG
//!
//! Wraps the model's EmbeddingExtractor for use in the RAG pipeline.

use std::sync::Arc;

use crate::backend::Backend;
use crate::model::{
    EmbeddingConfig, EmbeddingExtractor, InferenceContext, LlamaModel, Model, ModelConfig,
};
use crate::tokenizer::Tokenizer;

use super::RagResult;
use super::RagError;

/// Embedding generator using a language model
///
/// Delegates to `EmbeddingExtractor` from the model module, which handles
/// tokenization, forward pass, pooling, and normalization.
pub struct EmbeddingGenerator {
    extractor: EmbeddingExtractor,
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    backend: Arc<dyn Backend>,
    model_config: ModelConfig,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator from a loaded model
    pub fn new(
        model: Arc<LlamaModel>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        let model_config = model.config().clone();
        let extractor = EmbeddingExtractor::new(EmbeddingConfig::default(), &model_config);

        Self {
            extractor,
            model,
            tokenizer,
            backend,
            model_config,
        }
    }

    /// Create with custom embedding configuration
    pub fn with_config(
        model: Arc<LlamaModel>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
        config: EmbeddingConfig,
    ) -> Self {
        let model_config = model.config().clone();
        let extractor = EmbeddingExtractor::new(config, &model_config);

        Self {
            extractor,
            model,
            tokenizer,
            backend,
            model_config,
        }
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.extractor.embedding_dim()
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> RagResult<Vec<f32>> {
        let mut ctx = InferenceContext::new(&self.model_config, self.backend.clone());

        self.extractor
            .embed_text(self.model.as_ref(), &self.tokenizer, &mut ctx, text)
            .map_err(|e| RagError::QueryFailed(format!("Embedding generation failed: {}", e)))
    }

    /// Generate embeddings for multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> RagResult<Vec<Vec<f32>>> {
        let mut ctx = InferenceContext::new(&self.model_config, self.backend.clone());

        self.extractor
            .embed_batch(self.model.as_ref(), &self.tokenizer, &mut ctx, texts)
            .map_err(|e| RagError::QueryFailed(format!("Batch embedding failed: {}", e)))
    }
}

/// Simple text chunker for splitting documents
pub struct TextChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    separator: String,
}

// ... keep the existing TextChunker implementation and tests unchanged
```

Keep the existing `TextChunker` impl and its `#[cfg(test)]` module exactly as they are.

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully. The `EmbeddingGenerator::new` signature changed (now takes `Arc<LlamaModel>`, `Arc<Tokenizer>`, `Arc<dyn Backend>` instead of `LlamaModel` and `Arc<dyn Backend>`), which is fine since nothing else in the crate currently constructs one outside tests.

**Step 3: Run existing tests**

Run: `cargo test --features rag --lib rag::embedding::tests`
Expected: TextChunker and l2_normalize tests still pass

**Step 4: Commit**

```bash
git add src/rag/embedding.rs
git commit -m "feat(rag): wire EmbeddingExtractor into EmbeddingGenerator for real embeddings"
```

---

### Task 9: Wire embeddings into KnowledgeBase

**Files:**
- Modify: `src/rag/knowledge_base.rs:383-386` (KnowledgeBase struct), `798-807` (embed methods), `878-969` (builder)

**Step 1: Add model references to KnowledgeBase**

Update the `KnowledgeBase` struct:

```rust
use std::sync::Arc;
use crate::model::{LlamaModel, Model, InferenceContext, EmbeddingConfig, ModelConfig};
use crate::tokenizer::Tokenizer;
use crate::backend::Backend;

pub struct KnowledgeBase {
    config: KnowledgeBaseConfig,
    store: RagStore,
    embedding_gen: Option<super::EmbeddingGenerator>,
}
```

Update `embed_text` and `embed_query`:

```rust
fn embed_text(&self, text: &str) -> RagResult<Vec<f32>> {
    match &self.embedding_gen {
        Some(gen) => gen.embed(text),
        None => {
            // Fallback: zero vector (for testing without a model)
            Ok(vec![0.0f32; self.config.storage.embedding_dim()])
        }
    }
}

fn embed_query(&self, query: &str) -> RagResult<Vec<f32>> {
    self.embed_text(query)
}
```

Update the `retrieve` method to pass query text for hybrid search:

```rust
pub async fn retrieve(
    &self,
    query: &str,
    config: Option<RetrievalConfig>,
) -> RagResult<RetrievalResponse> {
    let config = config.unwrap_or_else(|| self.config.retrieval.clone());

    let query_embedding = self.embed_query(query)?;

    let docs = if self.config.storage.search_type() == super::SearchType::Hybrid {
        self.store
            .search_hybrid(&query_embedding, query, Some(config.max_results), config.filter)
            .await?
    } else {
        self.store
            .search_with_filter(&query_embedding, Some(config.max_results), config.filter)
            .await?
    };

    let chunks = docs
        .into_iter()
        .filter(|d| d.score.unwrap_or(0.0) >= config.min_score)
        .map(|d| self.doc_to_chunk(d))
        .collect();

    Ok(RetrievalResponse {
        chunks,
        query: query.to_string(),
        next_token: None,
    })
}
```

Update `KnowledgeBase::create` and `connect`:

```rust
pub async fn create(config: KnowledgeBaseConfig) -> RagResult<Self> {
    let store = RagStore::connect(config.storage.clone()).await?;
    store.create_table().await?;
    Ok(Self { config, store, embedding_gen: None })
}

pub async fn connect(config: KnowledgeBaseConfig) -> RagResult<Self> {
    let store = RagStore::connect(config.storage.clone()).await?;
    Ok(Self { config, store, embedding_gen: None })
}

/// Set the embedding generator (provides real embeddings from a model)
pub fn with_embedding_generator(mut self, gen: super::EmbeddingGenerator) -> Self {
    self.embedding_gen = Some(gen);
    self
}
```

Update the builder:

```rust
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

    /// Supply the inference model for embedding generation
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

    // ... keep all existing builder methods ...

    pub async fn create(self) -> RagResult<KnowledgeBase> {
        let mut kb = KnowledgeBase::create(self.config).await?;

        if let (Some(model), Some(tokenizer), Some(backend)) = (self.model, self.tokenizer, self.backend) {
            let gen = super::EmbeddingGenerator::new(model, tokenizer, backend);
            kb.embedding_gen = Some(gen);
        }

        Ok(kb)
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/rag/knowledge_base.rs
git commit -m "feat(rag): wire real embeddings into KnowledgeBase via EmbeddingGenerator"
```

---

### Task 10: Implement reranking in KnowledgeBase::retrieve

**Files:**
- Modify: `src/rag/knowledge_base.rs` (retrieve method, add rerank helper)

**Step 1: Write a test for reranking**

Add to the bottom of `knowledge_base.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_score_based() {
        let chunks = vec![
            RetrievedChunk {
                content: "low".into(),
                score: 0.5,
                source: SourceLocation { source_type: "doc".into(), uri: "a".into(), location: None },
                metadata: None,
            },
            RetrievedChunk {
                content: "high".into(),
                score: 0.9,
                source: SourceLocation { source_type: "doc".into(), uri: "b".into(), location: None },
                metadata: None,
            },
        ];

        let config = RerankingConfig {
            num_candidates: 10,
            method: RerankingMethod::ScoreBased,
        };

        let reranked = rerank(chunks, &config);
        assert_eq!(reranked[0].content, "high");
        assert_eq!(reranked[1].content, "low");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --features rag --lib rag::knowledge_base::tests`
Expected: FAIL — `rerank` function not defined

**Step 3: Implement rerank function**

Add before the `KnowledgeBase` impl:

```rust
/// Rerank retrieved chunks based on the configured method
fn rerank(mut chunks: Vec<RetrievedChunk>, config: &RerankingConfig) -> Vec<RetrievedChunk> {
    match &config.method {
        RerankingMethod::ScoreBased => {
            chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        RerankingMethod::RRF { k: _ } => {
            // RRF is handled at the store level during hybrid search
            // Here we just sort by score (which is already the RRF score)
            chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        RerankingMethod::CrossEncoder { model_path } => {
            tracing::warn!(
                "Cross-encoder reranking not implemented (model: {}), falling back to score-based",
                model_path
            );
            chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    chunks
}
```

Wire it into `retrieve()` — after building chunks, before returning:

```rust
// Apply reranking if configured
let chunks = if let Some(ref reranking_config) = self.config.reranking {
    rerank(chunks, reranking_config)
} else {
    chunks
};
```

**Step 4: Run test to verify it passes**

Run: `cargo test --features rag --lib rag::knowledge_base::tests`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/knowledge_base.rs
git commit -m "feat(rag): implement reranking for retrieved chunks"
```

---

### Task 11: Fix glob pattern in directory walk

**Files:**
- Modify: `src/rag/knowledge_base.rs:637-681` (walk_directory_recursive, walk_directory_flat)

**Step 1: Write a test**

```rust
#[test]
fn test_glob_pattern_matching() {
    // Test the glob matching helper
    assert!(matches_glob_pattern("docs/readme.md", "**/*.md"));
    assert!(matches_glob_pattern("src/lib.rs", "**/*.rs"));
    assert!(!matches_glob_pattern("image.png", "**/*.md"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --features rag --lib rag::knowledge_base::tests`
Expected: FAIL — `matches_glob_pattern` not defined

**Step 3: Implement glob matching in directory walk**

Add a helper function:

```rust
/// Check if a path matches a glob pattern
fn matches_glob_pattern(path: &str, pattern: &str) -> bool {
    glob::Pattern::new(pattern)
        .map(|p| p.matches(path))
        .unwrap_or(false)
}
```

Update `walk_directory_recursive`:

```rust
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

    // Apply glob filter if provided
    if let Some(pattern) = pattern {
        files.retain(|f| {
            let path_str = f.to_string_lossy();
            matches_glob_pattern(&path_str, pattern)
        });
    }

    Ok(files)
}
```

Update `walk_directory_flat` similarly:

```rust
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
            if let Some(pattern) = pattern {
                if !matches_glob_pattern(&path.to_string_lossy(), pattern) {
                    continue;
                }
            }
            files.push(path);
        }
    }

    Ok(files)
}
```

**Step 4: Run tests**

Run: `cargo test --features rag --lib rag::knowledge_base::tests`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/rag/knowledge_base.rs
git commit -m "fix(rag): apply glob patterns in directory walk"
```

---

### Task 12: Update mod.rs exports and lib.rs re-exports

**Files:**
- Modify: `src/rag/mod.rs:77-86` — export new types
- Modify: `src/lib.rs:72` — re-export new types

**Step 1: Update rag/mod.rs**

Add new error variant:

```rust
#[error("Embedding error: {0}")]
EmbeddingError(String),
```

The exports should already cover the new types since `store.rs` and `config.rs` use `pub use store::*` and `pub use config::*`. Verify that `IndexType`, `SearchType`, and new store methods are accessible.

**Step 2: Update lib.rs**

Update the rag re-export line to include new public types:

```rust
#[cfg(feature = "rag")]
pub use rag::{
    RagConfig, RagStore, RagError, RagResult, Document, NewDocument, RagContextBuilder, TextChunker,
    // Config types
    IndexType, SearchType, DistanceMetric, DatabaseConfig, EmbeddingsConfig, SearchConfig,
    // Knowledge base
    KnowledgeBase, KnowledgeBaseBuilder, KnowledgeBaseConfig, DataSource, ChunkingStrategy,
    RetrievalConfig, RetrievalResponse, RetrieveAndGenerateResponse, RetrievedChunk,
    Citation, SourceLocation, IngestionResult,
    // Embeddings
    EmbeddingGenerator,
    // Metadata filtering
    MetadataFilter,
};
```

**Step 3: Verify everything compiles**

Run: `cargo check --features rag`
Expected: Compiles successfully

Run: `cargo check --features rag,server`
Expected: Compiles successfully (server handlers still work with updated types)

**Step 4: Run all RAG tests**

Run: `cargo test --features rag --lib rag`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/rag/mod.rs src/lib.rs
git commit -m "feat(rag): update module exports for new pgvector types"
```

---

### Task 13: Update example_config and README

**Files:**
- Modify: `src/rag/config.rs:445-484` (example_config function)

**Step 1: Update example_config**

Replace the `example_config()` function body to include new fields:

```rust
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
#   - 1024: BERT-large
#   - 2048: LLaMA-7B (hidden_size)
#   - 4096: LLaMA-13B/70B (hidden_size)
dimension = 384

# Index type: "hnsw" (default, recommended), "ivfflat", or "none"
index_type = "hnsw"

# HNSW parameters (only used when index_type = "hnsw")
hnsw_m = 16                  # Max connections per node (higher = better recall, more memory)
hnsw_ef_construction = 64    # Construction search depth (higher = better index, slower build)

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
```

**Step 2: Verify it compiles**

Run: `cargo check --features rag`
Expected: Compiles

**Step 3: Commit**

```bash
git add src/rag/config.rs
git commit -m "docs(rag): update example config with HNSW and hybrid search options"
```

---

### Task 14: Final integration verification

**Step 1: Run full test suite**

Run: `cargo test --features rag`
Expected: All tests pass

**Step 2: Run clippy**

Run: `cargo clippy --features rag -- -D warnings`
Expected: No warnings

**Step 3: Build with all feature combinations that include rag**

Run: `cargo check --features rag,server`
Run: `cargo check --features rag,server,cuda`
Expected: All compile successfully

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(rag): address clippy warnings and compilation issues"
```
