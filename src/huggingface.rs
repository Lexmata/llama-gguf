//! HuggingFace Hub integration for downloading GGUF models
//!
//! This module provides functionality to:
//! - List GGUF files in a HuggingFace repository
//! - Download models with progress indication
//! - Manage a local model cache

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// Error types for HuggingFace operations
#[derive(thiserror::Error, Debug)]
pub enum HfError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Repository not found: {0}")]
    RepoNotFound(String),

    #[error("File not found in repository: {0}")]
    FileNotFound(String),

    #[error("Invalid repository format. Expected 'owner/repo' or full URL")]
    InvalidRepoFormat,

    #[error("No GGUF files found in repository")]
    NoGgufFiles,
}

pub type HfResult<T> = Result<T, HfError>;

/// File information from HuggingFace API
#[derive(Debug, Clone, Deserialize)]
pub struct HfFileInfo {
    /// File path/name
    pub path: String,
    /// File type ("file" or "directory")
    #[serde(rename = "type")]
    pub file_type: Option<String>,
    /// File size (for non-LFS files)
    pub size: Option<u64>,
    /// LFS info (for large files)
    pub lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LfsInfo {
    pub size: u64,
}

impl HfFileInfo {
    /// Get the filename (last component of path)
    pub fn filename(&self) -> &str {
        &self.path
    }
}

impl HfFileInfo {
    /// Get the actual file size (from LFS if available, otherwise direct size)
    pub fn file_size(&self) -> Option<u64> {
        self.lfs.as_ref().map(|lfs| lfs.size).or(self.size)
    }

    /// Check if this is a file (not a directory)
    pub fn is_file(&self) -> bool {
        self.file_type.as_deref() == Some("file")
    }
}

/// Repository information
#[derive(Debug, Clone, Deserialize)]
pub struct RepoInfo {
    pub id: String,
    #[serde(rename = "modelId")]
    pub model_id: Option<String>,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
}

/// HuggingFace Hub client
pub struct HfClient {
    /// Base URL for HuggingFace Hub API
    api_base: String,
    /// Base URL for file downloads
    download_base: String,
    /// HTTP client
    client: reqwest::blocking::Client,
    /// Local cache directory
    cache_dir: PathBuf,
}

impl Default for HfClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HfClient {
    /// Create a new HuggingFace client with default settings
    pub fn new() -> Self {
        let cache_dir = Self::default_cache_dir();
        Self {
            api_base: "https://huggingface.co/api".to_string(),
            download_base: "https://huggingface.co".to_string(),
            client: reqwest::blocking::Client::builder()
                .user_agent("llama-rs/0.1.0")
                .build()
                .expect("Failed to create HTTP client"),
            cache_dir,
        }
    }

    /// Create a client with a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            ..Self::new()
        }
    }

    /// Get the default cache directory
    pub fn default_cache_dir() -> PathBuf {
        if let Some(proj_dirs) = directories::ProjectDirs::from("com", "llama-rs", "llama-rs") {
            proj_dirs.cache_dir().join("models")
        } else {
            // Fallback to home directory
            dirs_fallback()
                .join(".cache")
                .join("llama-rs")
                .join("models")
        }
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Parse a repository identifier (handles "owner/repo" and full URLs)
    pub fn parse_repo_id(repo: &str) -> HfResult<String> {
        // Handle full URLs
        if repo.starts_with("https://huggingface.co/") {
            let path = repo.strip_prefix("https://huggingface.co/").unwrap();
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() >= 2 {
                return Ok(format!("{}/{}", parts[0], parts[1]));
            }
        }

        // Handle owner/repo format
        if repo.contains('/') && !repo.contains("://") {
            let parts: Vec<&str> = repo.split('/').collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                return Ok(repo.to_string());
            }
        }

        Err(HfError::InvalidRepoFormat)
    }

    /// Get repository information
    pub fn repo_info(&self, repo_id: &str) -> HfResult<RepoInfo> {
        let url = format!("{}/models/{}", self.api_base, repo_id);
        let response = self.client.get(&url).send()?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(HfError::RepoNotFound(repo_id.to_string()));
        }

        let info: RepoInfo = response.json()?;
        Ok(info)
    }

    /// List all files in a repository
    pub fn list_files(&self, repo_id: &str) -> HfResult<Vec<HfFileInfo>> {
        let url = format!("{}/models/{}/tree/main", self.api_base, repo_id);
        let response = self.client.get(&url).send()?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(HfError::RepoNotFound(repo_id.to_string()));
        }

        let files: Vec<HfFileInfo> = response.json()?;
        Ok(files)
    }

    /// List only GGUF files in a repository
    pub fn list_gguf_files(&self, repo_id: &str) -> HfResult<Vec<HfFileInfo>> {
        let all_files = self.list_files(repo_id)?;
        let gguf_files: Vec<_> = all_files
            .into_iter()
            .filter(|f| f.is_file() && f.path.ends_with(".gguf"))
            .collect();

        if gguf_files.is_empty() {
            return Err(HfError::NoGgufFiles);
        }

        Ok(gguf_files)
    }

    /// Get the local path where a file would be cached
    pub fn get_cached_path(&self, repo_id: &str, filename: &str) -> PathBuf {
        let safe_repo = repo_id.replace('/', "--");
        self.cache_dir.join(&safe_repo).join(filename)
    }

    /// Check if a file is already cached
    pub fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        self.get_cached_path(repo_id, filename).exists()
    }

    /// Download a file from a repository with progress indication
    pub fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        show_progress: bool,
    ) -> HfResult<PathBuf> {
        let cached_path = self.get_cached_path(repo_id, filename);

        // Check if already downloaded
        if cached_path.exists() {
            if show_progress {
                println!("File already cached: {}", cached_path.display());
            }
            return Ok(cached_path);
        }

        // Create cache directory
        if let Some(parent) = cached_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Build download URL
        let url = format!(
            "{}/{}/resolve/main/{}",
            self.download_base, repo_id, filename
        );

        if show_progress {
            println!("Downloading from: {}", url);
        }

        // Start download
        let response = self.client.get(&url).send()?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(HfError::FileNotFound(filename.to_string()));
        }

        let total_size = response.content_length();

        // Create progress bar
        let progress_bar = if show_progress {
            let pb = if let Some(size) = total_size {
                let pb = ProgressBar::new(size);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb
            } else {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {bytes}")
                        .unwrap(),
                );
                pb
            };
            Some(pb)
        } else {
            None
        };

        // Download to temporary file first
        let temp_path = cached_path.with_extension("tmp");
        let mut file = File::create(&temp_path)?;

        let mut downloaded: u64 = 0;
        let mut buffer = [0u8; 8192];

        use std::io::Read;
        let mut reader = response;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            if let Some(ref pb) = progress_bar {
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = progress_bar {
            pb.finish_with_message("Download complete");
        }

        // Rename temp file to final path
        fs::rename(&temp_path, &cached_path)?;

        if show_progress {
            println!("Saved to: {}", cached_path.display());
        }

        Ok(cached_path)
    }

    /// Search for models on HuggingFace Hub
    pub fn search_models(&self, query: &str, limit: usize) -> HfResult<Vec<RepoInfo>> {
        let url = format!(
            "{}/models?search={}&filter=gguf&limit={}",
            self.api_base, query, limit
        );
        let response = self.client.get(&url).send()?;
        let models: Vec<RepoInfo> = response.json()?;
        Ok(models)
    }

    /// Clear the model cache
    pub fn clear_cache(&self) -> HfResult<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Get the total size of the cache in bytes
    pub fn cache_size(&self) -> HfResult<u64> {
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        let mut total = 0u64;
        for entry in walkdir(&self.cache_dir)? {
            if entry.is_file() {
                total += entry.metadata()?.len();
            }
        }
        Ok(total)
    }

    /// List all cached models
    pub fn list_cached(&self) -> HfResult<Vec<(String, PathBuf)>> {
        let mut cached = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(cached);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let repo_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.replace("--", "/"))
                    .unwrap_or_default();

                for file_entry in fs::read_dir(&path)? {
                    let file_entry = file_entry?;
                    let file_path = file_entry.path();

                    if file_path.extension().is_some_and(|ext| ext == "gguf") {
                        cached.push((repo_name.clone(), file_path));
                    }
                }
            }
        }

        Ok(cached)
    }
}

/// Helper to walk directory recursively
fn walkdir(path: &Path) -> io::Result<Vec<PathBuf>> {
    let mut entries = Vec::new();

    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                entries.extend(walkdir(&path)?);
            } else {
                entries.push(path);
            }
        }
    }

    Ok(entries)
}

/// Fallback for getting home directory
fn dirs_fallback() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Format bytes into human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_repo_id() {
        assert_eq!(
            HfClient::parse_repo_id("Qwen/Qwen2.5-0.5B-Instruct-GGUF").unwrap(),
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        );

        assert_eq!(
            HfClient::parse_repo_id("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF")
                .unwrap(),
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        );

        assert!(HfClient::parse_repo_id("invalid").is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.46 KB");
        assert_eq!(format_bytes(1_500_000), "1.43 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.40 GB");
    }
}
