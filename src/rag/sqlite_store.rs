//! SQLite-based vector store for single-node RAG setups
//!
//! Provides an alternative to PostgreSQL/pgvector for environments where
//! PostgreSQL is not available. Uses brute-force cosine similarity for
//! small datasets (<100K documents) and optional HNSW for larger ones.

#![cfg(feature = "rag-sqlite")]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use bytemuck;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use super::{RagError, RagResult};

// -----------------------------------------------------------------------------
// Distance functions (pure Rust)
// -----------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn compute_score(
    a: &[f32],
    b: &[f32],
    metric: SqliteDistanceMetric,
) -> f32 {
    match metric {
        SqliteDistanceMetric::Cosine => cosine_similarity(a, b),
        SqliteDistanceMetric::L2 => {
            let d = l2_distance(a, b);
            1.0 / (1.0 + d)
        }
        SqliteDistanceMetric::InnerProduct => inner_product(a, b),
    }
}

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// Document stored in SQLite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqliteDocument {
    pub id: i64,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub score: Option<f32>,
}

/// New document to insert
#[derive(Debug, Clone)]
pub struct SqliteNewDocument {
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// SQLite vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqliteConfig {
    pub path: String,
    pub table_name: String,
    pub dimension: usize,
    pub use_hnsw: bool,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub distance_metric: SqliteDistanceMetric,
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self {
            path: ":memory:".to_string(),
            table_name: "embeddings".to_string(),
            dimension: 384,
            use_hnsw: false,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            distance_metric: SqliteDistanceMetric::default(),
        }
    }
}

impl SqliteConfig {
    /// Create config for in-memory store
    pub fn memory(dimension: usize) -> Self {
        Self {
            path: ":memory:".to_string(),
            table_name: "embeddings".to_string(),
            dimension,
            use_hnsw: false,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            distance_metric: SqliteDistanceMetric::default(),
        }
    }

    /// Create config for file-based store
    pub fn file(path: impl AsRef<Path>, dimension: usize) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            table_name: "embeddings".to_string(),
            dimension,
            use_hnsw: false,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            distance_metric: SqliteDistanceMetric::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum SqliteDistanceMetric {
    #[default]
    Cosine,
    L2,
    InnerProduct,
}

/// Simplified metadata filter for SQLite JSON1 extension
#[derive(Debug, Clone)]
pub enum SqliteMetadataFilter {
    Eq {
        field: String,
        value: serde_json::Value,
    },
    Ne {
        field: String,
        value: serde_json::Value,
    },
    Contains {
        field: String,
        value: String,
    },
    Exists {
        field: String,
    },
    And {
        filters: Vec<SqliteMetadataFilter>,
    },
    Or {
        filters: Vec<SqliteMetadataFilter>,
    },
    Not {
        filter: Box<SqliteMetadataFilter>,
    },
}

impl SqliteMetadataFilter {
    pub fn eq(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Eq {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn ne(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Ne {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Contains {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists {
            field: field.into(),
        }
    }

    pub fn and(filters: Vec<SqliteMetadataFilter>) -> Self {
        Self::And { filters }
    }

    pub fn or(filters: Vec<SqliteMetadataFilter>) -> Self {
        Self::Or { filters }
    }

    pub fn not(filter: SqliteMetadataFilter) -> Self {
        Self::Not {
            filter: Box::new(filter),
        }
    }

    /// Generate SQLite-compatible SQL using json_extract
    pub fn to_sql(&self, param_offset: usize) -> RagResult<(String, Vec<rusqlite::types::Value>)> {
        let mut params = Vec::new();
        let sql = self.to_sql_inner(param_offset, &mut params)?;
        Ok((sql, params))
    }

    fn to_sql_inner(
        &self,
        param_offset: usize,
        params: &mut Vec<rusqlite::types::Value>,
    ) -> RagResult<String> {
        fn json_path(field: &str) -> String {
            format!("json_extract(metadata, '$.{}')", field)
        }

        fn validate_field(field: &str) -> RagResult<&str> {
            if field.is_empty() {
                return Err(RagError::QueryFailed("Empty field name".into()));
            }
            if field.len() > 128 {
                return Err(RagError::QueryFailed("Field name too long".into()));
            }
            if !field
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
            {
                return Err(RagError::QueryFailed(format!(
                    "Invalid field name '{}': only alphanumeric, underscore, and dot allowed",
                    field
                )));
            }
            Ok(field)
        }

        fn json_value_to_rusqlite(v: &serde_json::Value) -> rusqlite::types::Value {
            match v {
                serde_json::Value::String(s) => rusqlite::types::Value::Text(s.clone()),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        rusqlite::types::Value::Integer(i)
                    } else if let Some(f) = n.as_f64() {
                        rusqlite::types::Value::Real(f)
                    } else {
                        rusqlite::types::Value::Text(n.to_string())
                    }
                }
                serde_json::Value::Bool(b) => rusqlite::types::Value::Integer(*b as i64),
                serde_json::Value::Null => rusqlite::types::Value::Null,
                _ => rusqlite::types::Value::Text(v.to_string()),
            }
        }

        match self {
            Self::Eq { field, value } => {
                let field = validate_field(field)?;
                params.push(json_value_to_rusqlite(value));
                Ok(format!("{} = ?{}", json_path(field), param_offset + params.len()))
            }
            Self::Ne { field, value } => {
                let field = validate_field(field)?;
                params.push(json_value_to_rusqlite(value));
                Ok(format!("{} != ?{}", json_path(field), param_offset + params.len()))
            }
            Self::Contains { field, value } => {
                let field = validate_field(field)?;
                params.push(rusqlite::types::Value::Text(format!("%{}%", value)));
                Ok(format!("{} LIKE ?{}", json_path(field), param_offset + params.len()))
            }
            Self::Exists { field } => {
                let field = validate_field(field)?;
                Ok(format!("{} IS NOT NULL", json_path(field)))
            }
            Self::And { filters } => {
                if filters.is_empty() {
                    return Ok("1=1".to_string());
                }
                let mut parts = Vec::new();
                for f in filters {
                    parts.push(f.to_sql_inner(param_offset, params)?);
                }
                Ok(format!("({})", parts.join(" AND ")))
            }
            Self::Or { filters } => {
                if filters.is_empty() {
                    return Ok("1=0".to_string());
                }
                let mut parts = Vec::new();
                for f in filters {
                    parts.push(f.to_sql_inner(param_offset, params)?);
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            Self::Not { filter } => {
                let inner = filter.to_sql_inner(param_offset, params)?;
                Ok(format!("NOT ({})", inner))
            }
        }
    }
}

// -----------------------------------------------------------------------------
// HNSW Index (pure Rust, in-memory)
// -----------------------------------------------------------------------------

struct HnswNode {
    id: i64,
    embedding: Vec<f32>,
    neighbors: Vec<Vec<i64>>,
}

struct HnswIndex {
    nodes: Vec<HnswNode>,
    id_to_idx: HashMap<i64, usize>,
    deleted: HashSet<i64>,
    max_level: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    entry_point: Option<usize>,
    dim: usize,
    metric: SqliteDistanceMetric,
}

impl HnswIndex {
    fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        dim: usize,
        metric: SqliteDistanceMetric,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            deleted: HashSet::new(),
            max_level: 0,
            m,
            ef_construction,
            ef_search,
            entry_point: None,
            dim,
            metric,
        }
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            SqliteDistanceMetric::Cosine => 1.0 - cosine_similarity(a, b),
            SqliteDistanceMetric::L2 => l2_distance(a, b),
            SqliteDistanceMetric::InnerProduct => -inner_product(a, b),
        }
    }

    fn insert(&mut self, id: i64, embedding: Vec<f32>) {
        if embedding.len() != self.dim {
            return;
        }
        if self.id_to_idx.contains_key(&id) {
            self.remove(id);
        }

        let mut rng = StdRng::from_entropy();
        let level = if self.entry_point.is_none() {
            0
        } else {
            (0..=16)
                .take_while(|_| rng.gen_range(0.0..1.0) < 0.5)
                .count()
        };

        let idx = self.nodes.len();
        self.nodes.push(HnswNode {
            id,
            embedding: embedding.clone(),
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
        });
        self.id_to_idx.insert(id, idx);

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_level = level;
            return;
        }

        let ep = self.entry_point.unwrap();
        let mut curr = ep;
        for l in (level + 1..=self.max_level).rev() {
            loop {
                let curr_node = &self.nodes[curr];
                let neighbors = &curr_node.neighbors[l.min(curr_node.neighbors.len().saturating_sub(1))];
                let mut next = curr;
                let mut best_d = self.distance(&self.nodes[curr].embedding, &embedding);
                for &nid in neighbors {
                    let Some(&n) = self.id_to_idx.get(&nid) else { continue };
                    if self.deleted.contains(&self.nodes[n].id) {
                        continue;
                    }
                    let d = self.distance(&self.nodes[n].embedding, &embedding);
                    if d < best_d {
                        best_d = d;
                        next = n;
                    }
                }
                if next == curr {
                    break;
                }
                curr = next;
            }
        }

        for l in (0..=level).rev() {
            let mut candidates: Vec<(usize, f32)> = vec![(curr, self.distance(&self.nodes[curr].embedding, &embedding))];
            let mut visited = HashSet::new();
            visited.insert(curr);

            let mut to_add = self.ef_construction;
            while to_add > 0 {
                let mut changed = false;
                let mut new_candidates = Vec::new();
                for (cand_idx, _) in &candidates.clone() {
                    let cand_node = &self.nodes[*cand_idx];
                    let layer_neighbors = cand_node.neighbors.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
                    for &nid in layer_neighbors {
                        let Some(&n) = self.id_to_idx.get(&nid) else { continue };
                        if self.deleted.contains(&self.nodes[n].id) {
                            continue;
                        }
                        if visited.insert(n) {
                            let d = self.distance(&self.nodes[n].embedding, &embedding);
                            new_candidates.push((n, d));
                            changed = true;
                        }
                    }
                }
                candidates.extend(new_candidates);
                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates.truncate(self.ef_construction);
                if !changed {
                    break;
                }
                to_add = to_add.saturating_sub(1);
            }

            let nearest: Vec<usize> = candidates
                .iter()
                .take(self.m)
                .map(|(i, _)| *i)
                .collect();

            for n in &nearest {
                if let Some(neighbors_at_l) = self.nodes[*n].neighbors.get_mut(l) {
                    neighbors_at_l.push(id);
                    if neighbors_at_l.len() > self.m * 2 {
                        neighbors_at_l.truncate(self.m);
                    }
                }
            }
            self.nodes[idx].neighbors[l] = nearest.iter().map(|&i| self.nodes[i].id).collect();

            if l > 0 {
                curr = *nearest.first().unwrap_or(&curr);
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(idx);
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if query.len() != self.dim || self.entry_point.is_none() {
            return Vec::new();
        }

        let mut curr = self.entry_point.unwrap();
        if self.deleted.contains(&self.nodes[curr].id) {
            return Vec::new();
        }

        for l in (1..=self.max_level).rev() {
            loop {
                let curr_node = &self.nodes[curr];
                let layer = curr_node.neighbors.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut next = curr;
                let mut best_d = self.distance(&curr_node.embedding, query);
                for &nid in layer {
                    if let Some(&n) = self.id_to_idx.get(&nid) {
                        if self.deleted.contains(&nid) {
                            continue;
                        }
                        let d = self.distance(&self.nodes[n].embedding, query);
                        if d < best_d {
                            best_d = d;
                            next = n;
                        }
                    }
                }
                if next == curr {
                    break;
                }
                curr = next;
            }
        }

        let mut candidates: Vec<(usize, f32)> =
            vec![(curr, self.distance(&self.nodes[curr].embedding, query))];
        let mut visited = HashSet::new();
        visited.insert(curr);

        for _ in 0..self.ef_search {
            let mut changed = false;
            let mut new_candidates = Vec::new();
            for (cand_idx, _) in candidates.clone() {
                let cand_node = &self.nodes[cand_idx];
                let layer0 = cand_node.neighbors.get(0).map(|v| v.as_slice()).unwrap_or(&[]);
                for &nid in layer0 {
                    if self.deleted.contains(&nid) {
                        continue;
                    }
                    if let Some(&n) = self.id_to_idx.get(&nid) {
                        if visited.insert(n) {
                            let d = self.distance(&self.nodes[n].embedding, query);
                            new_candidates.push((n, d));
                            changed = true;
                        }
                    }
                }
            }
            candidates.extend(new_candidates);
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(self.ef_search);
            if !changed {
                break;
            }
        }

        candidates
            .into_iter()
            .take(k)
            .filter(|(i, _)| !self.deleted.contains(&self.nodes[*i].id))
            .map(|(i, d)| {
                let score = match self.metric {
                    SqliteDistanceMetric::Cosine => 1.0 - d,
                    SqliteDistanceMetric::L2 => 1.0 / (1.0 + d),
                    SqliteDistanceMetric::InnerProduct => -d,
                };
                (self.nodes[i].id, score)
            })
            .collect()
    }

    fn remove(&mut self, id: i64) {
        self.deleted.insert(id);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.nodes.len() - self.deleted.len()
    }

    fn build_from_store(conn: &Connection, config: &SqliteConfig) -> RagResult<Self> {
        let mut index = Self::new(
            config.hnsw_m,
            config.hnsw_ef_construction,
            config.hnsw_ef_search,
            config.dimension,
            config.distance_metric,
        );

        let query = format!(
            "SELECT id, embedding FROM {}",
            config.table_name
        );
        let mut stmt = conn.prepare(&query).map_err(|e| RagError::QueryFailed(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                ))
            })
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        for row in rows {
            let (id, blob) = row.map_err(|e| RagError::QueryFailed(e.to_string()))?;
            let embedding: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&blob).to_vec();
            if embedding.len() == config.dimension {
                index.insert(id, embedding);
            }
        }

        Ok(index)
    }
}

// -----------------------------------------------------------------------------
// SqliteStore
// -----------------------------------------------------------------------------

pub struct SqliteStore {
    conn: Connection,
    config: SqliteConfig,
    hnsw_index: RefCell<Option<HnswIndex>>,
}

impl SqliteStore {
    /// Open or create a SQLite store
    pub fn open(config: SqliteConfig) -> RagResult<Self> {
        let conn = Connection::open(&config.path)
            .map_err(|e| RagError::ConnectionFailed(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        let mut store = Self {
            conn,
            config,
            hnsw_index: RefCell::new(None),
        };
        store.create_table()?;
        if store.config.use_hnsw {
            store.build_index()?;
        }
        Ok(store)
    }

    /// Open in-memory store
    pub fn open_memory(dimension: usize) -> RagResult<Self> {
        Self::open(SqliteConfig::memory(dimension))
    }

    /// Create the embeddings table
    pub fn create_table(&self) -> RagResult<()> {
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )",
            self.config.table_name
        );
        self.conn
            .execute_batch(&sql)
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Insert a document
    pub fn insert(&mut self, doc: SqliteNewDocument) -> RagResult<i64> {
        if doc.embedding.len() != self.config.dimension {
            return Err(RagError::DimensionMismatch {
                expected: self.config.dimension,
                actual: doc.embedding.len(),
            });
        }

        let metadata_str = doc
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).map_err(|e| RagError::SerializationError(e.to_string())))
            .transpose()?
            .unwrap_or_default();

        let embedding_blob = bytemuck::cast_slice::<f32, u8>(&doc.embedding);

        self.conn
            .execute(
                &format!(
                    "INSERT INTO {} (content, embedding, metadata) VALUES (?1, ?2, ?3)",
                    self.config.table_name
                ),
                params![doc.content, embedding_blob, metadata_str],
            )
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        let id = self.conn.last_insert_rowid();

        if let Some(ref mut idx) = *self.hnsw_index.borrow_mut() {
            idx.insert(id, doc.embedding);
        }

        Ok(id)
    }

    /// Insert batch
    pub fn insert_batch(&mut self, docs: Vec<SqliteNewDocument>) -> RagResult<Vec<i64>> {
        let mut ids = Vec::with_capacity(docs.len());
        for doc in docs {
            ids.push(self.insert(doc)?);
        }
        Ok(ids)
    }

    /// Search by vector similarity (brute-force or HNSW)
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> RagResult<Vec<SqliteDocument>> {
        self.search_with_filter(query_embedding, limit, None)
    }

    /// Search with metadata filter
    pub fn search_with_filter(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<&SqliteMetadataFilter>,
    ) -> RagResult<Vec<SqliteDocument>> {
        if query_embedding.len() != self.config.dimension {
            return Err(RagError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query_embedding.len(),
            });
        }

        if let Some(ref idx) = *self.hnsw_index.borrow() {
            let results = idx.search(query_embedding, limit);
            let ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
            if ids.is_empty() {
                return Ok(Vec::new());
            }
            return self.fetch_documents_by_ids(&ids, &results, filter);
        }

        self.brute_force_search(query_embedding, limit, filter)
    }

    fn fetch_documents_by_ids(
        &self,
        ids: &[i64],
        scores: &[(i64, f32)],
        filter: Option<&SqliteMetadataFilter>,
    ) -> RagResult<Vec<SqliteDocument>> {
        let score_map: HashMap<i64, f32> = scores.iter().copied().collect();
        let placeholders = ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");

        let (filter_clause, filter_params) = if let Some(f) = filter {
            let (sql, params) = f.to_sql(ids.len())?;
            (format!(" AND ({})", sql), params)
        } else {
            (String::new(), Vec::new())
        };

        let query = format!(
            "SELECT id, content, metadata FROM {} WHERE id IN ({}){}",
            self.config.table_name,
            placeholders,
            filter_clause
        );

        let mut stmt = self
            .conn
            .prepare(&query)
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        let param_iter = ids
            .iter()
            .map(|x| x as &dyn rusqlite::ToSql)
            .chain(filter_params.iter().map(|x| x as &dyn rusqlite::ToSql));

        let rows = stmt
            .query_map(rusqlite::params_from_iter(param_iter), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        let mut docs = Vec::new();
        for row in rows {
            let (id, content, metadata_str) = row.map_err(|e| RagError::QueryFailed(e.to_string()))?;
            let metadata = metadata_str.and_then(|s| serde_json::from_str(s.as_str()).ok());
            let score = score_map.get(&id).copied();
            docs.push(SqliteDocument {
                id,
                content,
                metadata,
                score,
            });
        }

        docs.sort_by(|a, b| {
            let sa = a.score.unwrap_or(0.0);
            let sb = b.score.unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(docs)
    }

    fn brute_force_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<&SqliteMetadataFilter>,
    ) -> RagResult<Vec<SqliteDocument>> {
        let (filter_clause, filter_params) = if let Some(f) = filter {
            let (sql, params) = f.to_sql(0)?;
            (format!(" WHERE {}", sql), params)
        } else {
            (String::new(), Vec::new())
        };

        let query = format!(
            "SELECT id, content, embedding, metadata FROM {}{}",
            self.config.table_name,
            filter_clause
        );

        let mut stmt = self
            .conn
            .prepare(&query)
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        let param_iter = filter_params
            .iter()
            .map(|p| p as &dyn rusqlite::ToSql);
        let rows = stmt
            .query_map(rusqlite::params_from_iter(param_iter), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        let mut scored: Vec<(i64, String, Option<serde_json::Value>, f32)> = Vec::new();
        for row in rows {
            let (id, content, embedding_blob, metadata_str) =
                row.map_err(|e| RagError::QueryFailed(e.to_string()))?;
            let metadata = metadata_str.and_then(|s| serde_json::from_str(s.as_str()).ok());
            let embedding: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&embedding_blob).to_vec();
            let score = compute_score(query_embedding, &embedding, self.config.distance_metric);
            scored.push((id, content, metadata, score));
        }

        scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored
            .into_iter()
            .map(|(id, content, metadata, score)| SqliteDocument {
                id,
                content,
                metadata,
                score: Some(score),
            })
            .collect())
    }

    /// Get document by ID
    pub fn get(&self, id: i64) -> RagResult<Option<SqliteDocument>> {
        let query = format!(
            "SELECT id, content, metadata FROM {} WHERE id = ?1",
            self.config.table_name
        );
        let mut stmt = self
            .conn
            .prepare(&query)
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;

        if let Some(row) = rows.next() {
            let (doc_id, content, metadata_str) = row.map_err(|e| RagError::QueryFailed(e.to_string()))?;
            let metadata = metadata_str.and_then(|s| serde_json::from_str(s.as_str()).ok());
            Ok(Some(SqliteDocument {
                id: doc_id,
                content,
                metadata,
                score: None,
            }))
        } else {
            Ok(None)
        }
    }

    /// Delete by ID
    pub fn delete(&self, id: i64) -> RagResult<bool> {
        let query = format!("DELETE FROM {} WHERE id = ?1", self.config.table_name);
        let affected = self
            .conn
            .execute(&query, params![id])
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;
        if affected > 0 {
            if let Some(ref mut idx) = *self.hnsw_index.borrow_mut() {
                idx.remove(id);
            }
        }
        Ok(affected > 0)
    }

    /// Count documents
    pub fn count(&self) -> RagResult<i64> {
        let query = format!("SELECT COUNT(*) FROM {}", self.config.table_name);
        let count: i64 = self
            .conn
            .query_row(&query, [], |row| row.get(0))
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;
        Ok(count)
    }

    /// Clear all documents
    pub fn clear(&mut self) -> RagResult<u64> {
        let query = format!("DELETE FROM {}", self.config.table_name);
        let affected = self
            .conn
            .execute(&query, [])
            .map_err(|e| RagError::QueryFailed(e.to_string()))?;
        if let Some(ref mut idx) = *self.hnsw_index.borrow_mut() {
            *idx = HnswIndex::new(
                self.config.hnsw_m,
                self.config.hnsw_ef_construction,
                self.config.hnsw_ef_search,
                self.config.dimension,
                self.config.distance_metric,
            );
        }
        Ok(affected as u64)
    }

    /// Build HNSW index from current data
    pub fn build_index(&mut self) -> RagResult<()> {
        let index = HnswIndex::build_from_store(&self.conn, &self.config)?;
        *self.hnsw_index.borrow_mut() = Some(index);
        Ok(())
    }

    /// Rebuild HNSW index
    pub fn rebuild_index(&mut self) -> RagResult<()> {
        *self.hnsw_index.borrow_mut() = None;
        self.build_index()
    }

    /// Get config
    pub fn config(&self) -> &SqliteConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(dim: usize, seed: i32) -> Vec<f32> {
        (0..dim).map(|i| (seed as f32 + i as f32) * 0.1).collect()
    }

    #[test]
    fn test_sqlite_store_basic() {
        let mut store = SqliteStore::open_memory(4).unwrap();
        store.create_table().unwrap();

        let doc = SqliteNewDocument {
            content: "hello world".to_string(),
            embedding: make_embedding(4, 1),
            metadata: None,
        };
        let id = store.insert(doc).unwrap();
        assert!(id > 0);

        let results = store.search(&make_embedding(4, 1), 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "hello world");
        assert!(results[0].score.unwrap() > 0.9);

        let got = store.get(id).unwrap().unwrap();
        assert_eq!(got.content, "hello world");

        let deleted = store.delete(id).unwrap();
        assert!(deleted);
        assert!(store.get(id).unwrap().is_none());
    }

    #[test]
    fn test_sqlite_store_batch_insert() {
        let mut store = SqliteStore::open_memory(8).unwrap();
        store.create_table().unwrap();

        let docs: Vec<SqliteNewDocument> = (0..100)
            .map(|i| SqliteNewDocument {
                content: format!("doc {}", i),
                embedding: make_embedding(8, i),
                metadata: None,
            })
            .collect();
        let ids = store.insert_batch(docs).unwrap();
        assert_eq!(ids.len(), 100);

        let count = store.count().unwrap();
        assert_eq!(count, 100);

        let results = store.search(&make_embedding(8, 42), 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score.unwrap() > 0.0);
    }

    #[test]
    fn test_sqlite_cosine_similarity() {
        let mut store = SqliteStore::open_memory(3).unwrap();
        store.create_table().unwrap();

        store
            .insert(SqliteNewDocument {
                content: "unit x".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
                metadata: None,
            })
            .unwrap();
        store
            .insert(SqliteNewDocument {
                content: "unit y".to_string(),
                embedding: vec![0.0, 1.0, 0.0],
                metadata: None,
            })
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "unit x");
        assert!(results[0].score.unwrap() > 0.99);

        let results = store.search(&[0.0, 1.0, 0.0], 2).unwrap();
        assert_eq!(results[0].content, "unit y");
    }

    #[test]
    fn test_sqlite_metadata_filter() {
        let mut store = SqliteStore::open_memory(4).unwrap();
        store.create_table().unwrap();

        store
            .insert(SqliteNewDocument {
                content: "doc a".to_string(),
                embedding: make_embedding(4, 1),
                metadata: Some(serde_json::json!({"type": "a", "x": 1})),
            })
            .unwrap();
        store
            .insert(SqliteNewDocument {
                content: "doc b".to_string(),
                embedding: make_embedding(4, 2),
                metadata: Some(serde_json::json!({"type": "b", "x": 2})),
            })
            .unwrap();

        let filter = SqliteMetadataFilter::eq("type", "a");
        let results = store
            .search_with_filter(&make_embedding(4, 1), 10, Some(&filter))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "doc a");
    }

    #[test]
    fn test_sqlite_hnsw_basic() {
        let mut config = SqliteConfig::memory(4);
        config.use_hnsw = true;
        config.hnsw_m = 4;
        config.hnsw_ef_construction = 10;
        config.hnsw_ef_search = 5;

        let mut store = SqliteStore::open(config).unwrap();

        for i in 0..20 {
            store
                .insert(SqliteNewDocument {
                    content: format!("doc {}", i),
                    embedding: make_embedding(4, i),
                    metadata: None,
                })
                .unwrap();
        }

        let results = store.search(&make_embedding(4, 7), 3).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score.unwrap() > 0.0);
    }

    #[test]
    fn test_sqlite_config_defaults() {
        let config = SqliteConfig::default();
        assert_eq!(config.path, ":memory:");
        assert_eq!(config.table_name, "embeddings");
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_ef_construction, 200);
        assert_eq!(config.hnsw_ef_search, 50);

        let config = SqliteConfig::memory(256);
        assert_eq!(config.dimension, 256);
        assert!(!config.use_hnsw);
    }
}
