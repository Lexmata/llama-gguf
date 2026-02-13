//! Embedding generation for RAG

use std::sync::Arc;

use crate::backend::Backend;
use crate::model::{EmbeddingConfig, EmbeddingExtractor, InferenceContext, LlamaModel};
use crate::tokenizer::Tokenizer;

use super::{RagError, RagResult};

/// Embedding generator using a language model
///
/// Wraps [`EmbeddingExtractor`] to provide a convenient interface for
/// generating embeddings within the RAG pipeline. For best results, use a
/// dedicated embedding model rather than a generative LLM.
pub struct EmbeddingGenerator {
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    backend: Arc<dyn Backend>,
    extractor: EmbeddingExtractor,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator with default [`EmbeddingConfig`]
    pub fn new(model: Arc<LlamaModel>, tokenizer: Arc<Tokenizer>, backend: Arc<dyn Backend>) -> Self {
        let config = EmbeddingConfig::default();
        let extractor = EmbeddingExtractor::new(config, model.config());
        Self {
            model,
            tokenizer,
            backend,
            extractor,
        }
    }

    /// Create a new embedding generator with a custom [`EmbeddingConfig`]
    pub fn with_config(
        model: Arc<LlamaModel>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
        config: EmbeddingConfig,
    ) -> Self {
        let extractor = EmbeddingExtractor::new(config, model.config());
        Self {
            model,
            tokenizer,
            backend,
            extractor,
        }
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.extractor.embedding_dim()
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> RagResult<Vec<f32>> {
        let mut ctx = InferenceContext::new(self.model.config(), Arc::clone(&self.backend));
        self.extractor
            .embed_text(self.model.as_ref(), &self.tokenizer, &mut ctx, text)
            .map_err(|e| RagError::EmbeddingError(e.to_string()))
    }

    /// Generate embeddings for multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> RagResult<Vec<Vec<f32>>> {
        let mut ctx = InferenceContext::new(self.model.config(), Arc::clone(&self.backend));
        self.extractor
            .embed_batch(self.model.as_ref(), &self.tokenizer, &mut ctx, texts)
            .map_err(|e| RagError::EmbeddingError(e.to_string()))
    }

    /// L2-normalize a vector (kept for tests and potential future use)
    #[allow(dead_code)]
    fn l2_normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }
}

/// Simple text chunker for splitting documents
pub struct TextChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    separator: String,
}

impl Default for TextChunker {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            chunk_overlap: 50,
            separator: " ".to_string(),
        }
    }
}

impl TextChunker {
    /// Create a new chunker with specified size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }
    
    /// Set the overlap between chunks
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }
    
    /// Set the word separator
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
    
    /// Split text into chunks
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split(&self.separator).collect();
        let mut chunks = Vec::new();
        
        let mut i = 0;
        while i < words.len() {
            let mut chunk_words = Vec::new();
            let mut char_count = 0;
            
            // Build chunk up to size limit
            for j in i..words.len() {
                let word = words[j];
                let word_len = word.len() + if chunk_words.is_empty() { 0 } else { 1 };
                
                if char_count + word_len > self.chunk_size && !chunk_words.is_empty() {
                    break;
                }
                
                chunk_words.push(word);
                char_count += word_len;
            }
            
            if !chunk_words.is_empty() {
                chunks.push(chunk_words.join(&self.separator));
                
                // Move forward, accounting for overlap
                let advance = chunk_words.len().saturating_sub(self.chunk_overlap / 10);
                i += advance.max(1);
            } else {
                i += 1;
            }
        }
        
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunker() {
        let chunker = TextChunker::new(50).with_overlap(10);
        let text = "This is a test sentence. It has multiple words. We want to chunk it.";
        let chunks = chunker.chunk(text);
        
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 60); // Allow some slack for word boundaries
        }
    }
    
    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = EmbeddingGenerator::l2_normalize(&v);
        
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
}
