//! Incremental re-indexing for RAG document sources
//!
//! Tracks document content hashes to detect changes and only
//! re-index modified or new documents.

#![cfg(any(feature = "rag", feature = "rag-sqlite"))]

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{RagError, RagResult};

/// Tracks document sources and their content hashes for incremental re-indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentTracker {
    /// Map of source URI -> tracked document info
    entries: HashMap<String, TrackedDocument>,
    /// Path to the tracker state file
    #[serde(skip)]
    state_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedDocument {
    pub source_uri: String,
    pub content_hash: String,
    pub chunk_ids: Vec<i64>,
    pub last_indexed: String, // ISO 8601 timestamp (or Unix seconds as string)
    pub file_modified: Option<String>,
    pub file_size: Option<u64>,
}

/// Result of a sync operation
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub errors: Vec<(String, String)>,
}

/// Change detected in a source
#[derive(Debug, Clone)]
pub enum SourceChange {
    New { uri: String, content: String },
    Modified {
        uri: String,
        content: String,
        old_chunk_ids: Vec<i64>,
    },
    Deleted { uri: String, old_chunk_ids: Vec<i64> },
    Unchanged { uri: String },
}

impl DocumentTracker {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            state_path: None,
        }
    }

    pub fn with_state_file(path: impl AsRef<Path>) -> Self {
        Self {
            entries: HashMap::new(),
            state_path: Some(path.as_ref().to_path_buf()),
        }
    }

    /// Load tracker state from file
    pub fn load(&mut self) -> RagResult<()> {
        let path = match &self.state_path {
            Some(p) => p,
            None => return Ok(()),
        };
        let content = std::fs::read_to_string(path)
            .map_err(|e| RagError::ConfigError(format!("Failed to read tracker state: {}", e)))?;
        let loaded: DocumentTracker = serde_json::from_str(&content)
            .map_err(|e| RagError::SerializationError(format!("Failed to parse tracker state: {}", e)))?;
        self.entries = loaded.entries;
        Ok(())
    }

    /// Save tracker state to file
    pub fn save(&self) -> RagResult<()> {
        let path = match &self.state_path {
            Some(p) => p,
            None => return Ok(()),
        };
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| RagError::SerializationError(format!("Failed to serialize tracker state: {}", e)))?;
        std::fs::write(path, content)
            .map_err(|e| RagError::ConfigError(format!("Failed to write tracker state: {}", e)))?;
        Ok(())
    }

    /// Compute content hash (uses DefaultHasher for fast deterministic hashing)
    pub fn content_hash(content: &str) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Detect changes in a directory
    pub fn detect_changes(
        &self,
        dir: &Path,
        pattern: Option<&str>,
        recursive: bool,
    ) -> Vec<SourceChange> {
        let files = collect_files(dir, pattern, recursive);
        let mut changes = Vec::new();

        let mut seen_uris: std::collections::HashSet<String> = std::collections::HashSet::new();

        for path in &files {
            let uri = path.to_string_lossy().to_string();
            seen_uris.insert(uri.clone());

            let change = match std::fs::read_to_string(path) {
                Ok(content) => self.detect_file_change_with_content(path, &content),
                Err(_) => continue, // Skip unreadable files
            };
            changes.push(change);
        }

        // Detect deleted: tracked sources no longer present in directory
        for (tracked_uri, doc) in &self.entries {
            if !seen_uris.contains(tracked_uri) {
                changes.push(SourceChange::Deleted {
                    uri: tracked_uri.clone(),
                    old_chunk_ids: doc.chunk_ids.clone(),
                });
            }
        }

        changes
    }

    /// Detect changes for a single file
    pub fn detect_file_change(&self, path: &Path) -> SourceChange {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        self.detect_file_change_with_content(path, &content)
    }

    fn detect_file_change_with_content(&self, path: &Path, content: &str) -> SourceChange {
        let uri = path.to_string_lossy().to_string();
        let current_hash = Self::content_hash(content);

        match self.entries.get(&uri) {
            Some(doc) => {
                if doc.content_hash == current_hash {
                    SourceChange::Unchanged { uri }
                } else {
                    SourceChange::Modified {
                        uri,
                        content: content.to_string(),
                        old_chunk_ids: doc.chunk_ids.clone(),
                    }
                }
            }
            None => SourceChange::New {
                uri,
                content: content.to_string(),
            },
        }
    }

    /// Record that a document was indexed
    pub fn record_indexed(&mut self, source_uri: &str, content_hash: &str, chunk_ids: Vec<i64>) {
        let last_indexed = iso8601_now();
        let entry = TrackedDocument {
            source_uri: source_uri.to_string(),
            content_hash: content_hash.to_string(),
            chunk_ids,
            last_indexed,
            file_modified: None,
            file_size: None,
        };
        self.entries.insert(source_uri.to_string(), entry);
    }

    /// Record that a document was deleted
    pub fn record_deleted(&mut self, source_uri: &str) {
        self.entries.remove(source_uri);
    }

    /// Get tracked document info
    pub fn get(&self, source_uri: &str) -> Option<&TrackedDocument> {
        self.entries.get(source_uri)
    }

    /// Get all tracked source URIs
    pub fn tracked_sources(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Get chunk IDs for a source
    pub fn chunk_ids(&self, source_uri: &str) -> Option<&[i64]> {
        self.entries.get(source_uri).map(|d| d.chunk_ids.as_slice())
    }

    /// Number of tracked documents
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for DocumentTracker {
    fn default() -> Self {
        Self::new()
    }
}

fn collect_files(dir: &Path, pattern: Option<&str>, recursive: bool) -> Vec<PathBuf> {
    let mut files = Vec::new();

    fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>, pattern: Option<&str>, recursive: bool) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && recursive {
                visit_dir(&path, files, pattern, recursive)?;
            } else if path.is_file() {
                let path_str = path.to_string_lossy();
                let matches = match pattern {
                    Some(p) => glob::Pattern::new(p).map(|pat| pat.matches(&path_str)).unwrap_or(false),
                    None => true,
                };
                if matches {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    let _ = visit_dir(dir, &mut files, pattern, recursive);
    files
}

fn iso8601_now() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    format!("{}", secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_content_hash_deterministic() {
        let content = "hello world";
        let h1 = DocumentTracker::content_hash(content);
        let h2 = DocumentTracker::content_hash(content);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_different() {
        let h1 = DocumentTracker::content_hash("hello");
        let h2 = DocumentTracker::content_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_document_tracker_basic() {
        let mut tracker = DocumentTracker::new();
        assert!(tracker.is_empty());

        tracker.record_indexed("file:///a.txt", "abc123", vec![1, 2, 3]);
        assert_eq!(tracker.len(), 1);

        let doc = tracker.get("file:///a.txt").unwrap();
        assert_eq!(doc.content_hash, "abc123");
        assert_eq!(doc.chunk_ids, vec![1, 2, 3]);

        assert_eq!(tracker.chunk_ids("file:///a.txt"), Some([1i64, 2, 3].as_slice()));

        tracker.record_deleted("file:///a.txt");
        assert!(tracker.is_empty());
        assert!(tracker.get("file:///a.txt").is_none());
    }

    #[test]
    fn test_detect_changes_with_tempdir() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path();

        // Create two files
        let f1 = dir.join("a.txt");
        let f2 = dir.join("b.txt");
        std::fs::write(&f1, "content a").unwrap();
        std::fs::write(&f2, "content b").unwrap();

        let mut tracker = DocumentTracker::new();
        let changes = tracker.detect_changes(dir, None, false);

        // Both should be New
        let new_count = changes.iter().filter(|c| matches!(c, SourceChange::New { .. })).count();
        assert_eq!(new_count, 2, "expected 2 new files, got {:?}", changes);

        // Record both files as indexed
        tracker.record_indexed(&f1.to_string_lossy(), &DocumentTracker::content_hash("content a"), vec![1]);
        tracker.record_indexed(&f2.to_string_lossy(), &DocumentTracker::content_hash("content b"), vec![2]);

        // Modify a.txt
        std::fs::write(&f1, "content a modified").unwrap();

        let changes2 = tracker.detect_changes(dir, None, false);
        let modified = changes2.iter().find(|c| matches!(c, SourceChange::Modified { .. }));
        assert!(modified.is_some(), "expected modified file, got {:?}", changes2);

        let unchanged = changes2.iter().find(|c| matches!(c, SourceChange::Unchanged { .. }));
        assert!(unchanged.is_some(), "expected unchanged file (b.txt), got {:?}", changes2);
    }

    #[test]
    fn test_sync_result_defaults() {
        let result = SyncResult::default();
        assert_eq!(result.added, 0);
        assert_eq!(result.updated, 0);
        assert_eq!(result.deleted, 0);
        assert_eq!(result.unchanged, 0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_tracker_save_load() {
        let temp = TempDir::new().unwrap();
        let state_path = temp.path().join("tracker.json");

        let mut tracker = DocumentTracker::with_state_file(&state_path);
        tracker.record_indexed("file:///x.txt", "hash1", vec![10, 20]);
        tracker.record_indexed("file:///y.txt", "hash2", vec![30]);
        tracker.save().unwrap();

        let mut loaded = DocumentTracker::with_state_file(&state_path);
        loaded.load().unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("file:///x.txt").unwrap().content_hash, "hash1");
        assert_eq!(loaded.get("file:///x.txt").unwrap().chunk_ids, vec![10, 20]);
        assert_eq!(loaded.get("file:///y.txt").unwrap().chunk_ids, vec![30]);
    }
}
