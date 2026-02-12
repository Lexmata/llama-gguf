//! Prompt caching and prefix sharing
//!
//! This module provides mechanisms to cache and reuse KV cache entries
//! for common prompt prefixes, enabling faster inference for:
//! - System prompts that are reused across conversations
//! - Common instruction prefixes
//! - RAG context that's shared across queries

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::tensor::Tensor;

/// Unique identifier for a cached prefix
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixId(pub u64);

impl PrefixId {
    /// Create a prefix ID from tokens
    pub fn from_tokens(tokens: &[u32]) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        PrefixId(hasher.finish())
    }
}

/// Cached KV state for a prefix
#[derive(Debug, Clone)]
pub struct CachedPrefix {
    /// The tokens that make up this prefix
    pub tokens: Vec<u32>,
    /// Cached key tensors per layer
    pub k_cache: Vec<Tensor>,
    /// Cached value tensors per layer  
    pub v_cache: Vec<Tensor>,
    /// Number of tokens cached
    pub seq_len: usize,
    /// Reference count (for LRU eviction)
    pub ref_count: usize,
    /// Last access time
    pub last_access: std::time::Instant,
}

impl CachedPrefix {
    /// Create a new cached prefix
    pub fn new(tokens: Vec<u32>, k_cache: Vec<Tensor>, v_cache: Vec<Tensor>) -> Self {
        let seq_len = tokens.len();
        Self {
            tokens,
            k_cache,
            v_cache,
            seq_len,
            ref_count: 0,
            last_access: std::time::Instant::now(),
        }
    }

    /// Memory size in bytes
    pub fn memory_size(&self) -> usize {
        let k_size: usize = self.k_cache.iter().map(|t| t.data().len()).sum();
        let v_size: usize = self.v_cache.iter().map(|t| t.data().len()).sum();
        k_size + v_size + self.tokens.len() * 4
    }
}

/// Prompt cache configuration
#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    /// Maximum number of cached prefixes
    pub max_entries: usize,
    /// Maximum total memory for cache (bytes)
    pub max_memory: usize,
    /// Minimum prefix length to cache
    pub min_prefix_len: usize,
    /// Enable automatic caching of system prompts
    pub cache_system_prompts: bool,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            max_memory: 1024 * 1024 * 1024, // 1 GB
            min_prefix_len: 32,
            cache_system_prompts: true,
        }
    }
}

/// Prompt cache for prefix sharing
pub struct PromptCache {
    /// Configuration
    config: PromptCacheConfig,
    /// Cached prefixes by ID
    entries: HashMap<PrefixId, CachedPrefix>,
    /// Current memory usage
    memory_used: usize,
}

impl PromptCache {
    /// Create a new prompt cache
    pub fn new(config: PromptCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            memory_used: 0,
        }
    }

    /// Cache a prefix
    pub fn cache_prefix(
        &mut self,
        tokens: &[u32],
        k_cache: Vec<Tensor>,
        v_cache: Vec<Tensor>,
    ) -> PrefixId {
        let id = PrefixId::from_tokens(tokens);

        // Check if already cached
        if self.entries.contains_key(&id) {
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.ref_count += 1;
                entry.last_access = std::time::Instant::now();
            }
            return id;
        }

        // Check if prefix is long enough
        if tokens.len() < self.config.min_prefix_len {
            return id;
        }

        let prefix = CachedPrefix::new(tokens.to_vec(), k_cache, v_cache);
        let size = prefix.memory_size();

        // Evict if necessary
        while self.memory_used + size > self.config.max_memory
            || self.entries.len() >= self.config.max_entries
        {
            if !self.evict_lru() {
                break;
            }
        }

        self.memory_used += size;
        self.entries.insert(id.clone(), prefix);

        id
    }

    /// Get a cached prefix
    pub fn get_prefix(&mut self, id: &PrefixId) -> Option<&CachedPrefix> {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.ref_count += 1;
            entry.last_access = std::time::Instant::now();
            Some(entry)
        } else {
            None
        }
    }

    /// Find the longest matching prefix
    pub fn find_matching_prefix(&mut self, tokens: &[u32]) -> Option<(PrefixId, usize)> {
        let mut best_match: Option<(PrefixId, usize)> = None;

        for (id, entry) in &self.entries {
            // Check if this prefix matches the start of tokens
            if tokens.len() >= entry.tokens.len()
                && tokens[..entry.tokens.len()] == entry.tokens[..]
            {
                let match_len = entry.tokens.len();
                if best_match.is_none() || match_len > best_match.as_ref().unwrap().1 {
                    best_match = Some((id.clone(), match_len));
                }
            }
        }

        // Update access time for matched entry
        if let Some((ref id, _)) = best_match
            && let Some(entry) = self.entries.get_mut(id)
        {
            entry.last_access = std::time::Instant::now();
            entry.ref_count += 1;
        }

        best_match
    }

    /// Remove a prefix from cache
    pub fn remove_prefix(&mut self, id: &PrefixId) {
        if let Some(entry) = self.entries.remove(id) {
            self.memory_used = self.memory_used.saturating_sub(entry.memory_size());
        }
    }

    /// Clear all cached prefixes
    pub fn clear(&mut self) {
        self.entries.clear();
        self.memory_used = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> PromptCacheStats {
        PromptCacheStats {
            num_entries: self.entries.len(),
            memory_used: self.memory_used,
            total_tokens_cached: self.entries.values().map(|e| e.seq_len).sum(),
        }
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) -> bool {
        // Find LRU entry (oldest last_access with ref_count == 0)
        let lru_id = self
            .entries
            .iter()
            .filter(|(_, e)| e.ref_count == 0)
            .min_by_key(|(_, e)| e.last_access)
            .map(|(id, _)| id.clone());

        if let Some(id) = lru_id {
            self.remove_prefix(&id);
            true
        } else {
            false
        }
    }

    /// Decrease reference count for a prefix
    pub fn release_prefix(&mut self, id: &PrefixId) {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.ref_count = entry.ref_count.saturating_sub(1);
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct PromptCacheStats {
    /// Number of cached prefixes
    pub num_entries: usize,
    /// Memory used in bytes
    pub memory_used: usize,
    /// Total tokens cached
    pub total_tokens_cached: usize,
}

/// Helper to manage prefix sharing in inference
pub struct PrefixSharing {
    /// The prompt cache
    cache: PromptCache,
    /// Active prefix ID for current session
    active_prefix: Option<PrefixId>,
}

impl PrefixSharing {
    /// Create a new prefix sharing manager
    pub fn new(config: PromptCacheConfig) -> Self {
        Self {
            cache: PromptCache::new(config),
            active_prefix: None,
        }
    }

    /// Try to restore cached prefix into KV cache
    ///
    /// Returns the number of tokens restored (0 if no match)
    pub fn try_restore(
        &mut self,
        tokens: &[u32],
        k_cache: &mut [Tensor],
        v_cache: &mut [Tensor],
    ) -> usize {
        // Find matching prefix
        let (id, match_len) = match self.cache.find_matching_prefix(tokens) {
            Some(m) => m,
            None => return 0,
        };

        // Get cached data
        let prefix = match self.cache.get_prefix(&id) {
            Some(p) => p,
            None => return 0,
        };

        // Copy cached KV to current cache
        for (layer_idx, (cached_k, cached_v)) in
            prefix.k_cache.iter().zip(prefix.v_cache.iter()).enumerate()
        {
            if layer_idx < k_cache.len() {
                // Copy cached data
                let k_src = cached_k.data();
                let v_src = cached_v.data();

                if let Some(k_dst) = k_cache[layer_idx].data_mut() {
                    let copy_len = k_src.len().min(k_dst.len());
                    k_dst[..copy_len].copy_from_slice(&k_src[..copy_len]);
                }

                if let Some(v_dst) = v_cache[layer_idx].data_mut() {
                    let copy_len = v_src.len().min(v_dst.len());
                    v_dst[..copy_len].copy_from_slice(&v_src[..copy_len]);
                }
            }
        }

        self.active_prefix = Some(id);
        match_len
    }

    /// Save current KV cache as a prefix
    pub fn save_prefix(
        &mut self,
        tokens: &[u32],
        k_cache: &[Tensor],
        v_cache: &[Tensor],
    ) -> PrefixId {
        // Clone the cache tensors
        let k_cloned: Vec<Tensor> = k_cache.to_vec();
        let v_cloned: Vec<Tensor> = v_cache.to_vec();

        let id = self.cache.cache_prefix(tokens, k_cloned, v_cloned);
        self.active_prefix = Some(id.clone());
        id
    }

    /// Release the active prefix
    pub fn release_active(&mut self) {
        if let Some(id) = self.active_prefix.take() {
            self.cache.release_prefix(&id);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> PromptCacheStats {
        self.cache.stats()
    }

    /// Clear all cached prefixes
    pub fn clear(&mut self) {
        self.active_prefix = None;
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn test_prefix_id() {
        let tokens1 = vec![1, 2, 3, 4];
        let tokens2 = vec![1, 2, 3, 4];
        let tokens3 = vec![1, 2, 3, 5];

        let id1 = PrefixId::from_tokens(&tokens1);
        let id2 = PrefixId::from_tokens(&tokens2);
        let id3 = PrefixId::from_tokens(&tokens3);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_prompt_cache() {
        let config = PromptCacheConfig {
            min_prefix_len: 2,
            ..Default::default()
        };
        let mut cache = PromptCache::new(config);

        let tokens = vec![1, 2, 3, 4, 5];
        let k = vec![Tensor::zeros(vec![4, 4], DType::F32)];
        let v = vec![Tensor::zeros(vec![4, 4], DType::F32)];

        let id = cache.cache_prefix(&tokens, k, v);

        assert!(cache.get_prefix(&id).is_some());
        assert_eq!(cache.stats().num_entries, 1);
    }

    #[test]
    fn test_find_matching_prefix() {
        let config = PromptCacheConfig {
            min_prefix_len: 2,
            ..Default::default()
        };
        let mut cache = PromptCache::new(config);

        let prefix = vec![1, 2, 3];
        let k = vec![Tensor::zeros(vec![4, 4], DType::F32)];
        let v = vec![Tensor::zeros(vec![4, 4], DType::F32)];

        cache.cache_prefix(&prefix, k, v);

        // Should match
        let query = vec![1, 2, 3, 4, 5];
        let result = cache.find_matching_prefix(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, 3);

        // Should not match
        let query2 = vec![1, 2, 4, 5];
        let result2 = cache.find_matching_prefix(&query2);
        assert!(result2.is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let config = PromptCacheConfig {
            max_entries: 2,
            min_prefix_len: 1,
            ..Default::default()
        };
        let mut cache = PromptCache::new(config);

        // Add 3 entries, should evict one
        for i in 0..3 {
            let tokens = vec![i];
            let k = vec![Tensor::zeros(vec![4, 4], DType::F32)];
            let v = vec![Tensor::zeros(vec![4, 4], DType::F32)];
            cache.cache_prefix(&tokens, k, v);
        }

        assert!(cache.stats().num_entries <= 2);
    }
}
