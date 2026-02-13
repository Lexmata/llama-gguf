//! Integration tests for RAG/pgvector functionality.
//!
//! These tests require a running PostgreSQL instance with pgvector extension.
//! Set the `RAG_TEST_DATABASE_URL` environment variable to the connection string.
//!
//! Example:
//!   RAG_TEST_DATABASE_URL="postgres://postgres:testpass@localhost:5434/ragtest" \
//!     cargo test --features rag --test rag_integration_test -- --test-threads=1
//!
//! The tests use unique table names and clean up after themselves.

#![cfg(feature = "rag")]

use llama_gguf::rag::{
    DistanceMetric, MetadataFilter, NewDocument, RagConfig, RagContextBuilder, RagStore, SearchType,
};
use serde_json::json;

/// Connection string for the test database.
/// Falls back to a local pgvector Docker container if not set.
fn test_db_url() -> String {
    std::env::var("RAG_TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:testpass@localhost:5434/ragtest".to_string())
}

/// Dimension used for test embeddings.
const DIM: usize = 8;

/// Create a RagConfig pointing at the test database with a unique table name.
fn test_config(table_name: &str) -> RagConfig {
    RagConfig::new(test_db_url())
        .with_table(table_name)
        .with_dim(DIM)
        .with_min_similarity(-1.0) // accept all results in tests
        .with_max_results(100)
}

/// Create a hybrid-search RagConfig with a unique table name.
fn test_config_hybrid(table_name: &str) -> RagConfig {
    let mut config = test_config(table_name);
    config.search.search_type = SearchType::Hybrid;
    config
}

/// Create a config with a specific distance metric.
fn test_config_metric(table_name: &str, metric: DistanceMetric) -> RagConfig {
    test_config(table_name).with_distance_metric(metric)
}

/// Generate a simple normalized embedding vector.
/// `seed` controls direction — different seeds yield different vectors.
fn make_embedding(seed: u32) -> Vec<f32> {
    let raw: Vec<f32> = (0..DIM)
        .map(|i| ((seed as f32 + 1.0) * (i as f32 + 1.0)).sin())
        .collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

/// Helper to create a NewDocument with a given seed and content.
fn make_doc(content: &str, seed: u32, metadata: Option<serde_json::Value>) -> NewDocument {
    NewDocument {
        content: content.to_string(),
        embedding: make_embedding(seed),
        metadata,
    }
}

/// Drop a table if it exists — cleanup helper.
async fn drop_table(store: &RagStore, table_name: &str) {
    // We access the underlying pool through a raw query via the store's own
    // methods. Since RagStore doesn't expose raw queries, we just clear the
    // table. But we also want to DROP on setup — so we create a temporary
    // direct connection.
    let url = test_db_url();
    let (client, connection) = tokio_postgres::connect(&url, tokio_postgres::NoTls)
        .await
        .expect("failed to connect for cleanup");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("cleanup connection error: {}", e);
        }
    });
    let _ = client
        .execute(&format!("DROP TABLE IF EXISTS {} CASCADE", table_name), &[])
        .await;
    // suppress unused variable warning
    let _ = store;
}

/// Drop a table via a direct connection (no store needed).
async fn drop_table_direct(table_name: &str) {
    let url = test_db_url();
    let (client, connection) = tokio_postgres::connect(&url, tokio_postgres::NoTls)
        .await
        .expect("failed to connect for cleanup");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("cleanup connection error: {}", e);
        }
    });
    let _ = client
        .execute(&format!("DROP TABLE IF EXISTS {} CASCADE", table_name), &[])
        .await;
}

// =============================================================================
// Connection and Health Check
// =============================================================================

#[tokio::test]
async fn test_connect_and_health_check() {
    let config = test_config("test_health");
    let store = RagStore::connect(config).await.expect("connect failed");
    store.health_check().await.expect("health check failed");
}

#[tokio::test]
async fn test_connect_bad_url() {
    let config = RagConfig::new("postgres://nobody:badpass@localhost:19999/nonexistent")
        .with_table("bad_test")
        .with_dim(DIM);
    let result = RagStore::connect(config).await;
    assert!(result.is_err(), "expected connection to fail");
}

// =============================================================================
// Table Creation
// =============================================================================

#[tokio::test]
async fn test_create_table_hnsw() {
    let table = "test_create_hnsw";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Verify table exists by inserting a document
    let id = store
        .insert(make_doc("hello", 1, None))
        .await
        .expect("insert");
    assert!(id > 0);

    // Cleanup
    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_create_table_hybrid() {
    let table = "test_create_hybrid";
    drop_table_direct(table).await;

    let config = test_config_hybrid(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Verify the tsvector column exists by inserting and doing keyword search
    let id = store
        .insert(make_doc("PostgreSQL is a relational database", 1, None))
        .await
        .expect("insert");
    assert!(id > 0);

    let keyword_results = store.search_keyword("relational database", 10, None).await;
    assert!(keyword_results.is_ok(), "keyword search should work on hybrid table");

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_create_table_no_index() {
    let table = "test_create_noindex";
    drop_table_direct(table).await;

    let mut config = test_config(table);
    config.embeddings.index_type = "none".to_string();
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let id = store
        .insert(make_doc("no index test", 1, None))
        .await
        .expect("insert");
    assert!(id > 0);

    drop_table(&store, table).await;
}

// =============================================================================
// Insert, Get, Delete
// =============================================================================

#[tokio::test]
async fn test_insert_get_delete() {
    let table = "test_insert_get_delete";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Insert
    let metadata = json!({"source": "test.txt", "page": 1});
    let id = store
        .insert(make_doc("The quick brown fox", 42, Some(metadata.clone())))
        .await
        .expect("insert");
    assert!(id > 0);

    // Get
    let doc = store.get(id).await.expect("get").expect("doc should exist");
    assert_eq!(doc.id, id);
    assert_eq!(doc.content, "The quick brown fox");
    assert_eq!(doc.metadata.unwrap()["source"], "test.txt");

    // Count
    let count = store.count().await.expect("count");
    assert_eq!(count, 1);

    // Delete
    let deleted = store.delete(id).await.expect("delete");
    assert!(deleted);

    // Verify deleted
    let gone = store.get(id).await.expect("get after delete");
    assert!(gone.is_none());

    let count = store.count().await.expect("count after delete");
    assert_eq!(count, 0);

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_insert_dimension_mismatch() {
    let table = "test_dim_mismatch";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let bad_doc = NewDocument {
        content: "wrong dim".to_string(),
        embedding: vec![1.0, 2.0, 3.0], // 3 != DIM (8)
        metadata: None,
    };
    let result = store.insert(bad_doc).await;
    assert!(result.is_err(), "should reject wrong dimension");

    drop_table(&store, table).await;
}

// =============================================================================
// Upsert
// =============================================================================

#[tokio::test]
async fn test_upsert_insert_and_update() {
    let table = "test_upsert";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Upsert without ID → insert
    let id = store
        .upsert(None, make_doc("original content", 1, Some(json!({"v": 1}))))
        .await
        .expect("upsert insert");
    assert!(id > 0);

    let doc = store.get(id).await.expect("get").unwrap();
    assert_eq!(doc.content, "original content");

    // Upsert with existing ID → update
    let same_id = store
        .upsert(
            Some(id),
            make_doc("updated content", 2, Some(json!({"v": 2}))),
        )
        .await
        .expect("upsert update");
    assert_eq!(same_id, id);

    let updated = store.get(id).await.expect("get updated").unwrap();
    assert_eq!(updated.content, "updated content");
    assert_eq!(updated.metadata.unwrap()["v"], 2);

    // Count should still be 1
    assert_eq!(store.count().await.expect("count"), 1);

    drop_table(&store, table).await;
}

// =============================================================================
// Batch Insert
// =============================================================================

#[tokio::test]
async fn test_batch_insert() {
    let table = "test_batch_insert";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let docs: Vec<NewDocument> = (0..150)
        .map(|i| make_doc(&format!("Document number {}", i), i, Some(json!({"idx": i}))))
        .collect();

    let ids = store.insert_batch(docs).await.expect("batch insert");
    assert_eq!(ids.len(), 150);

    // All IDs should be unique
    let mut unique = ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 150);

    let count = store.count().await.expect("count");
    assert_eq!(count, 150);

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_batch_insert_empty() {
    let table = "test_batch_empty";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let ids = store.insert_batch(vec![]).await.expect("empty batch");
    assert!(ids.is_empty());

    drop_table(&store, table).await;
}

// =============================================================================
// Vector Similarity Search (Cosine)
// =============================================================================

#[tokio::test]
async fn test_vector_search_cosine() {
    let table = "test_search_cosine";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Insert documents with distinct embeddings
    store
        .insert(make_doc("Rust programming language", 10, None))
        .await
        .expect("insert 1");
    store
        .insert(make_doc("Python programming language", 20, None))
        .await
        .expect("insert 2");
    store
        .insert(make_doc("JavaScript framework", 30, None))
        .await
        .expect("insert 3");

    // Search with embedding close to seed=10
    let query = make_embedding(10);
    let results = store.search(&query, Some(3)).await.expect("search");

    assert!(!results.is_empty(), "should find results");
    // The first result should be the one with matching embedding (seed=10)
    assert_eq!(results[0].content, "Rust programming language");
    // Score should be close to 1.0 (identical vector)
    assert!(
        results[0].score.unwrap() > 0.99,
        "exact match should have high score, got {}",
        results[0].score.unwrap()
    );

    // All results should have scores
    for doc in &results {
        assert!(doc.score.is_some());
    }

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_search_with_limit() {
    let table = "test_search_limit";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    for i in 0..10 {
        store
            .insert(make_doc(&format!("Doc {}", i), i, None))
            .await
            .expect("insert");
    }

    let results = store
        .search(&make_embedding(0), Some(3))
        .await
        .expect("search");
    assert_eq!(results.len(), 3, "should respect limit");

    drop_table(&store, table).await;
}

// =============================================================================
// Search with Metadata Filters
// =============================================================================

#[tokio::test]
async fn test_search_with_eq_filter() {
    let table = "test_search_eq_filter";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc(
            "Rust docs",
            1,
            Some(json!({"type": "docs", "lang": "rust"})),
        ))
        .await
        .expect("insert");
    store
        .insert(make_doc(
            "Python docs",
            2,
            Some(json!({"type": "docs", "lang": "python"})),
        ))
        .await
        .expect("insert");
    store
        .insert(make_doc(
            "Rust tutorial",
            3,
            Some(json!({"type": "tutorial", "lang": "rust"})),
        ))
        .await
        .expect("insert");

    let filter = MetadataFilter::eq("lang", "rust");
    let results = store
        .search_with_filter(&make_embedding(1), Some(10), Some(filter))
        .await
        .expect("filtered search");

    assert_eq!(results.len(), 2, "should find 2 rust docs");
    for doc in &results {
        assert!(
            doc.content.contains("Rust"),
            "all results should be Rust docs"
        );
    }

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_search_with_and_filter() {
    let table = "test_search_and_filter";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc(
            "Rust docs",
            1,
            Some(json!({"type": "docs", "lang": "rust"})),
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Python docs",
            2,
            Some(json!({"type": "docs", "lang": "python"})),
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Rust tutorial",
            3,
            Some(json!({"type": "tutorial", "lang": "rust"})),
        ))
        .await
        .unwrap();

    let filter = MetadataFilter::and(vec![
        MetadataFilter::eq("lang", "rust"),
        MetadataFilter::eq("type", "docs"),
    ]);
    let results = store
        .search_with_filter(&make_embedding(1), Some(10), Some(filter))
        .await
        .expect("and filter search");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "Rust docs");

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_search_with_exists_filter() {
    let table = "test_search_exists_filter";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc(
            "With tag",
            1,
            Some(json!({"tag": "important"})),
        ))
        .await
        .unwrap();
    store
        .insert(make_doc("Without tag", 2, Some(json!({"other": true}))))
        .await
        .unwrap();
    store.insert(make_doc("No metadata", 3, None)).await.unwrap();

    let filter = MetadataFilter::exists("tag");
    let results = store
        .search_with_filter(&make_embedding(1), Some(10), Some(filter))
        .await
        .expect("exists filter");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "With tag");

    drop_table(&store, table).await;
}

// =============================================================================
// Count and Delete with Filters
// =============================================================================

#[tokio::test]
async fn test_count_and_delete_with_filter() {
    let table = "test_count_delete_filter";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    for i in 0..5 {
        let cat = if i < 3 { "alpha" } else { "beta" };
        store
            .insert(make_doc(
                &format!("Doc {} category {}", i, cat),
                i,
                Some(json!({"category": cat})),
            ))
            .await
            .unwrap();
    }

    // Count with filter
    let alpha_filter = MetadataFilter::eq("category", "alpha");
    let alpha_count = store
        .count_with_filter(Some(alpha_filter))
        .await
        .expect("count filtered");
    assert_eq!(alpha_count, 3);

    // Total count
    let total = store.count().await.expect("total count");
    assert_eq!(total, 5);

    // Delete alpha
    let delete_filter = MetadataFilter::eq("category", "alpha");
    let deleted = store
        .delete_with_filter(delete_filter)
        .await
        .expect("delete filtered");
    assert_eq!(deleted, 3);

    let remaining = store.count().await.expect("remaining count");
    assert_eq!(remaining, 2);

    drop_table(&store, table).await;
}

// =============================================================================
// Clear
// =============================================================================

#[tokio::test]
async fn test_clear() {
    let table = "test_clear";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    for i in 0..5 {
        store
            .insert(make_doc(&format!("Doc {}", i), i, None))
            .await
            .unwrap();
    }
    assert_eq!(store.count().await.unwrap(), 5);

    let cleared = store.clear().await.expect("clear");
    assert_eq!(cleared, 5);
    assert_eq!(store.count().await.unwrap(), 0);

    drop_table(&store, table).await;
}

// =============================================================================
// List Metadata Values
// =============================================================================

#[tokio::test]
async fn test_list_metadata_values() {
    let table = "test_list_metadata";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let categories = ["api", "docs", "tutorial"];
    for (i, cat) in categories.iter().enumerate() {
        store
            .insert(make_doc(
                &format!("{} content", cat),
                i as u32,
                Some(json!({"category": *cat})),
            ))
            .await
            .unwrap();
    }

    let values = store
        .list_metadata_values("category", None)
        .await
        .expect("list values");
    assert_eq!(values.len(), 3);
    assert!(values.contains(&"api".to_string()));
    assert!(values.contains(&"docs".to_string()));
    assert!(values.contains(&"tutorial".to_string()));

    drop_table(&store, table).await;
}

// =============================================================================
// Hybrid Search (Semantic + Keyword with RRF)
// =============================================================================

#[tokio::test]
async fn test_hybrid_search() {
    let table = "test_hybrid_search";
    drop_table_direct(table).await;

    let config = test_config_hybrid(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Insert documents with varying semantic and keyword relevance
    store
        .insert(make_doc(
            "PostgreSQL is the best relational database for production workloads",
            1,
            None,
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Redis is an in-memory data structure store used as a cache",
            2,
            None,
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Database management systems handle transaction processing",
            3,
            None,
        ))
        .await
        .unwrap();

    // Keyword search should find "database" mentions
    let keyword_results = store
        .search_keyword("database", 10, None)
        .await
        .expect("keyword search");
    assert!(
        keyword_results.len() >= 2,
        "keyword search should find at least 2 docs with 'database', found {}",
        keyword_results.len()
    );

    // Hybrid search combines both signals
    let query_emb = make_embedding(1); // close to the PostgreSQL doc
    let hybrid_results = store
        .search_hybrid(&query_emb, "database", Some(5), None)
        .await
        .expect("hybrid search");

    assert!(!hybrid_results.is_empty(), "hybrid search should return results");
    // All results should have fused scores
    for doc in &hybrid_results {
        assert!(doc.score.is_some());
        assert!(doc.score.unwrap() > 0.0);
    }

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_keyword_search_standalone() {
    let table = "test_keyword_standalone";
    drop_table_direct(table).await;

    let config = test_config_hybrid(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc(
            "Rust is a systems programming language focused on safety",
            1,
            None,
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Python is great for machine learning and data science",
            2,
            None,
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "JavaScript runs in the browser and on the server with Node.js",
            3,
            None,
        ))
        .await
        .unwrap();

    let results = store
        .search_keyword("programming language", 10, None)
        .await
        .expect("keyword search");
    // Should find the Rust doc (has both "programming" and "language")
    assert!(!results.is_empty());
    // The Rust doc should rank highest
    let top_id = results[0].0;
    let top_doc = store.get(top_id).await.unwrap().unwrap();
    assert!(
        top_doc.content.contains("Rust"),
        "top keyword result should be the Rust doc"
    );

    drop_table(&store, table).await;
}

// =============================================================================
// Different Distance Metrics
// =============================================================================

#[tokio::test]
async fn test_search_l2_distance() {
    let table = "test_search_l2";
    drop_table_direct(table).await;

    let config = test_config_metric(table, DistanceMetric::L2);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc("close vector", 5, None))
        .await
        .unwrap();
    store
        .insert(make_doc("far vector", 100, None))
        .await
        .unwrap();

    let query = make_embedding(5);
    let results = store.search(&query, Some(2)).await.expect("l2 search");

    assert_eq!(results.len(), 2);
    // The closer vector should be first
    assert_eq!(results[0].content, "close vector");
    assert!(results[0].score.unwrap() > results[1].score.unwrap());

    drop_table(&store, table).await;
}

#[tokio::test]
async fn test_search_inner_product() {
    let table = "test_search_ip";
    drop_table_direct(table).await;

    let config = test_config_metric(table, DistanceMetric::InnerProduct);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc("matching direction", 7, None))
        .await
        .unwrap();
    store
        .insert(make_doc("different direction", 77, None))
        .await
        .unwrap();

    let query = make_embedding(7);
    let results = store.search(&query, Some(2)).await.expect("ip search");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "matching direction");

    drop_table(&store, table).await;
}

// =============================================================================
// HNSW ef_search Parameter
// =============================================================================

#[tokio::test]
async fn test_set_hnsw_ef_search() {
    let table = "test_hnsw_ef";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Should not error
    store
        .set_hnsw_ef_search(200)
        .await
        .expect("set ef_search");

    drop_table(&store, table).await;
}

// =============================================================================
// Context Builder
// =============================================================================

#[tokio::test]
async fn test_context_builder() {
    let table = "test_ctx_builder";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    for i in 0..3 {
        store
            .insert(make_doc(&format!("Context chunk {}", i), i, None))
            .await
            .unwrap();
    }

    let results = store
        .search(&make_embedding(0), Some(3))
        .await
        .expect("search");

    // Build context string
    let context = RagContextBuilder::new(results.clone()).build();
    assert!(context.contains("Context chunk"));

    // Build with scores
    let context_with_scores = RagContextBuilder::new(results.clone())
        .with_scores(true)
        .build();
    assert!(context_with_scores.contains("["));

    // Build prompt
    let prompt = RagContextBuilder::new(results.clone())
        .build_prompt("What are the chunks about?");
    assert!(prompt.contains("Question:"));
    assert!(prompt.contains("Context chunk"));

    // Build with token limit
    let short = RagContextBuilder::new(results)
        .with_max_tokens(5) // ~20 chars
        .build();
    assert!(short.len() < 200, "token limit should truncate");

    drop_table(&store, table).await;
}

// =============================================================================
// Recreate Index
// =============================================================================

#[tokio::test]
async fn test_recreate_index() {
    let table = "test_recreate_idx";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Insert some data
    for i in 0..20 {
        store
            .insert(make_doc(&format!("Indexed doc {}", i), i, None))
            .await
            .unwrap();
    }

    // Recreate index after bulk insert
    store.create_index().await.expect("recreate index");

    // Search should still work
    let results = store
        .search(&make_embedding(0), Some(5))
        .await
        .expect("search after reindex");
    assert_eq!(results.len(), 5);

    drop_table(&store, table).await;
}

// =============================================================================
// MetadataFilter::parse
// =============================================================================

#[tokio::test]
async fn test_filter_parse_against_db() {
    let table = "test_filter_parse";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc(
            "Version 3 doc",
            1,
            Some(json!({"version": "3", "status": "published"})),
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Version 1 doc",
            2,
            Some(json!({"version": "1", "status": "draft"})),
        ))
        .await
        .unwrap();

    // Parse a filter string and use it
    let filter = MetadataFilter::parse("status=published").expect("parse filter");
    let results = store
        .search_with_filter(&make_embedding(1), Some(10), Some(filter))
        .await
        .expect("filtered search");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "Version 3 doc");

    // Parse combined filters
    let filter = MetadataFilter::parse_many("status=published;version=3").expect("parse many");
    let results = store
        .search_with_filter(&make_embedding(1), Some(10), Some(filter))
        .await
        .expect("multi-filter search");
    assert_eq!(results.len(), 1);

    drop_table(&store, table).await;
}

// =============================================================================
// Search Ordering Correctness
// =============================================================================

#[tokio::test]
async fn test_search_ordering_by_similarity() {
    let table = "test_search_ordering";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    // Insert with known seeds — seed 5 should be closest to query seed 5
    for seed in [1u32, 5, 10, 50, 100] {
        store
            .insert(make_doc(&format!("seed_{}", seed), seed, None))
            .await
            .unwrap();
    }

    let query = make_embedding(5);
    let results = store.search(&query, Some(5)).await.expect("search");

    assert_eq!(results.len(), 5);
    // First result should be the exact match
    assert_eq!(results[0].content, "seed_5");

    // Scores should be in descending order
    for i in 1..results.len() {
        assert!(
            results[i - 1].score.unwrap() >= results[i].score.unwrap(),
            "results should be ordered by descending score"
        );
    }

    drop_table(&store, table).await;
}

// =============================================================================
// Hybrid Search with Metadata Filter
// =============================================================================

#[tokio::test]
async fn test_hybrid_search_with_filter() {
    let table = "test_hybrid_filter";
    drop_table_direct(table).await;

    let config = test_config_hybrid(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    store
        .insert(make_doc(
            "Rust programming guide for systems developers",
            1,
            Some(json!({"lang": "rust"})),
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Python programming tutorial for beginners",
            2,
            Some(json!({"lang": "python"})),
        ))
        .await
        .unwrap();
    store
        .insert(make_doc(
            "Rust async programming with tokio",
            3,
            Some(json!({"lang": "rust"})),
        ))
        .await
        .unwrap();

    let filter = MetadataFilter::eq("lang", "rust");
    let results = store
        .search_hybrid(&make_embedding(1), "programming", Some(10), Some(filter))
        .await
        .expect("hybrid search with filter");

    // Should only return Rust docs
    for doc in &results {
        assert!(
            doc.content.contains("Rust"),
            "filtered hybrid results should only contain Rust docs, got: {}",
            doc.content
        );
    }

    drop_table(&store, table).await;
}

// =============================================================================
// Search Dimension Mismatch
// =============================================================================

#[tokio::test]
async fn test_search_dimension_mismatch() {
    let table = "test_search_dim_mismatch";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let wrong_dim_query = vec![1.0f32; DIM + 5];
    let result = store.search(&wrong_dim_query, Some(5)).await;
    assert!(result.is_err(), "should reject wrong query dimension");

    drop_table(&store, table).await;
}

// =============================================================================
// Delete Non-existent Document
// =============================================================================

#[tokio::test]
async fn test_delete_nonexistent() {
    let table = "test_delete_none";
    drop_table_direct(table).await;

    let config = test_config(table);
    let store = RagStore::connect(config).await.expect("connect");
    store.create_table().await.expect("create_table");

    let deleted = store.delete(999999).await.expect("delete nonexistent");
    assert!(!deleted, "deleting nonexistent should return false");

    drop_table(&store, table).await;
}
