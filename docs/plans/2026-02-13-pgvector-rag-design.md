# pgvector RAG Enhancement Design

**Date**: 2026-02-13
**Branch**: feature/pgvector
**Status**: Approved

## Problem

The existing RAG module (`src/rag/`) has solid structure but critical gaps:

1. **Embeddings are fake** — `EmbeddingGenerator` and `KnowledgeBase.embed_text/embed_query` return zero vectors
2. **No HNSW index** — only IVFFlat, which requires training and has worse recall
3. **No hybrid search** — `SearchType::Hybrid` is configured but not implemented
4. **Batch inserts are O(n)** — individual INSERT per document
5. **Reranking is stubbed** — types defined but never used
6. **Directory walk ignores glob patterns**
7. **No connection health checks**

## Approach

Wire & Enhance — connect existing components and fill gaps without rewriting the architecture.

## Design

### 1. Embedding Integration

Connect the existing `EmbeddingExtractor` (from `model/embeddings.rs`) into `KnowledgeBase`.

**Changes:**

- `KnowledgeBase` gains model/tokenizer/backend references via `KnowledgeBaseBuilder::with_model()`
- `embed_text()` and `embed_query()` delegate to `EmbeddingExtractor::embed_text()` instead of returning zeros
- `rag/embedding.rs` `EmbeddingGenerator` wraps `EmbeddingExtractor` internally, removing duplicate pooling/normalization code
- `InferenceContext` is created fresh per embed call (reset between texts)
- Embedding dimension auto-detected from `model.config().hidden_size`; `RagConfig.embedding_dim` becomes a validation check

**Files**: `src/rag/knowledge_base.rs`, `src/rag/embedding.rs`

### 2. HNSW + IVFFlat Index Support

**New type in `config.rs`:**

```rust
pub enum IndexType {
    Hnsw { m: u16, ef_construction: u16 },  // default: m=16, ef=64
    IvfFlat { lists: u16 },
    None,
}
```

**Changes to `store.rs`:**

- `create_table()` creates index based on `IndexType`
- HNSW: `CREATE INDEX ... USING hnsw (embedding {ops}) WITH (m={m}, ef_construction={ef})`
- IVFFlat: `CREATE INDEX ... USING ivfflat (embedding {ops}) WITH (lists={lists})`
- Public `create_index()` method for recreating indexes after bulk inserts
- `set_hnsw_ef_search(ef)` for runtime search quality tuning

**TOML config:**

```toml
[embeddings]
index_type = "hnsw"
hnsw_m = 16
hnsw_ef_construction = 64
```

**Files**: `src/rag/config.rs`, `src/rag/store.rs`

### 3. Hybrid Search with RRF

**Schema additions:**

```sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
CREATE INDEX {table}_content_tsv_idx ON {table} USING gin (content_tsv)
```

The `tsvector` column is auto-generated — no changes to insert logic.

**RRF implementation:**

1. Vector similarity search → top `N * oversampling` results
2. Keyword search via `ts_rank(content_tsv, plainto_tsquery())` → top `N * oversampling` results
3. RRF fusion: `score = Σ 1/(k + rank)` with `k = 60`
4. Return top N by fused score

Two separate SQL queries merged in application code. `search_with_filter()` dispatches to `search_hybrid()` or `search_vector()` based on `SearchType`.

**New config fields:**

```toml
[search]
search_type = "hybrid"
rrf_k = 60
hybrid_oversampling = 2
text_search_language = "english"
```

**Files**: `src/rag/store.rs`, `src/rag/config.rs`

### 4. Batch Inserts

Replace O(n) individual `INSERT` in `insert_batch()`:

- **Standard batches**: Use `tokio-postgres` pipelined queries (multiple statements sent without waiting for individual responses)
- **Large batches (>1000)**: `COPY ... FROM STDIN` with binary format via copy-in API
- `BulkIngestConfig` with `batch_size` (default 100)

**Files**: `src/rag/store.rs`

### 5. Reranking

Implement existing `RerankingMethod` variants:

- `ScoreBased` — re-sort by score
- `RRF { k }` — used automatically with hybrid search
- `CrossEncoder` — return error if configured (requires separate model, out of scope)

Applied as post-processing in `KnowledgeBase::retrieve()`.

**Files**: `src/rag/knowledge_base.rs`

### 6. Fixes

- **Glob patterns**: Apply glob pattern in `walk_directory_recursive/flat` using the `glob` crate
- **Connection health**: Retry with backoff in `RagStore::connect()`, add `health_check()` method
- **Field validation**: `escape_field()` rejects field names with dangerous characters instead of just escaping quotes
- **Upsert**: Add `upsert()` method using `ON CONFLICT (id) DO UPDATE`

**Files**: `src/rag/store.rs`, `src/rag/knowledge_base.rs`, `Cargo.toml`

## Dependencies

New Cargo dependencies (all optional under `rag` feature):
- `glob` — for directory walk glob patterns (if not already transitive)

Existing dependencies remain unchanged:
- `tokio-postgres`, `pgvector`, `deadpool-postgres`, `tokio`, `url`

## Testing

- Unit tests for RRF fusion logic (no database needed)
- Unit tests for field validation
- Integration tests require a PostgreSQL instance with pgvector (document setup in README)
- Existing embedding tests in `model/embeddings.rs` already cover pooling strategies
