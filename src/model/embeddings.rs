//! Embedding extraction API
//!
//! This module provides functionality to extract embeddings from models,
//! useful for:
//! - Semantic similarity/search
//! - Retrieval-Augmented Generation (RAG)
//! - Clustering and classification
//! - Vector databases

use crate::model::{InferenceContext, Model, ModelConfig};
use crate::tokenizer::Tokenizer;

/// Embedding extraction configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Which layer to extract embeddings from (-1 = last layer)
    pub layer: i32,
    /// Pooling strategy for sequence embeddings
    pub pooling: PoolingStrategy,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Maximum sequence length
    pub max_length: usize,
    /// Truncation strategy
    pub truncation: TruncationStrategy,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            layer: -1,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            max_length: 512,
            truncation: TruncationStrategy::Right,
        }
    }
}

/// Pooling strategy for combining token embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the last token's embedding (common for decoder models)
    Last,
    /// Use the first token's embedding (CLS token style)
    First,
    /// Average all token embeddings
    Mean,
    /// Max pooling across tokens
    Max,
    /// Weighted mean based on attention
    WeightedMean,
}

/// Truncation strategy for long sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Truncate from the right (keep beginning)
    Right,
    /// Truncate from the left (keep end)
    Left,
    /// Keep both ends, truncate middle
    Middle,
}

/// Embedding extractor for a model
pub struct EmbeddingExtractor {
    /// Embedding configuration
    config: EmbeddingConfig,
    /// Model hidden dimension
    hidden_dim: usize,
}

impl EmbeddingExtractor {
    /// Create a new embedding extractor
    pub fn new(config: EmbeddingConfig, model_config: &ModelConfig) -> Self {
        Self {
            config,
            hidden_dim: model_config.hidden_size,
        }
    }

    /// Extract embedding for a single text
    pub fn embed_text(
        &self,
        model: &dyn Model,
        tokenizer: &Tokenizer,
        ctx: &mut InferenceContext,
        text: &str,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Tokenize (without BOS token for embeddings)
        let tokens = tokenizer.encode(text, false)?;

        // Truncate if needed
        let tokens = self.truncate_tokens(&tokens);

        // Get embeddings for all tokens
        let embeddings = self.get_token_embeddings(model, ctx, &tokens)?;

        // Pool to single embedding
        let pooled = self.pool_embeddings(&embeddings, tokens.len());

        // Normalize if requested
        if self.config.normalize {
            Ok(self.normalize_embedding(&pooled))
        } else {
            Ok(pooled)
        }
    }

    /// Extract embeddings for multiple texts (batched)
    pub fn embed_batch(
        &self,
        model: &dyn Model,
        tokenizer: &Tokenizer,
        ctx: &mut InferenceContext,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            // Reset context for each text
            ctx.reset();
            let embedding = self.embed_text(model, tokenizer, ctx, text)?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Truncate tokens based on configuration
    fn truncate_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        if tokens.len() <= self.config.max_length {
            return tokens.to_vec();
        }

        match self.config.truncation {
            TruncationStrategy::Right => tokens[..self.config.max_length].to_vec(),
            TruncationStrategy::Left => tokens[tokens.len() - self.config.max_length..].to_vec(),
            TruncationStrategy::Middle => {
                let half = self.config.max_length / 2;
                let mut truncated = tokens[..half].to_vec();
                truncated.extend_from_slice(&tokens[tokens.len() - half..]);
                truncated
            }
        }
    }

    /// Get embeddings for each token
    fn get_token_embeddings(
        &self,
        model: &dyn Model,
        ctx: &mut InferenceContext,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut embeddings = Vec::with_capacity(tokens.len());

        // Process tokens one at a time and capture hidden states
        for token in tokens {
            let logits = model.forward(&[*token], ctx)?;

            // For now, use logits as a proxy for hidden state
            // A full implementation would capture the actual hidden states
            // from the model's internal layers
            let logits_data = logits.as_f32()?;

            // Create embedding from the beginning of logits (approximation)
            let dim = self.hidden_dim.min(logits_data.len());
            embeddings.push(logits_data[..dim].to_vec());
        }

        Ok(embeddings)
    }

    /// Pool token embeddings into a single embedding
    fn pool_embeddings(&self, embeddings: &[Vec<f32>], _seq_len: usize) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; self.hidden_dim];
        }

        let dim = embeddings[0].len();

        match self.config.pooling {
            PoolingStrategy::Last => embeddings.last().cloned().unwrap_or_else(|| vec![0.0; dim]),
            PoolingStrategy::First => embeddings
                .first()
                .cloned()
                .unwrap_or_else(|| vec![0.0; dim]),
            PoolingStrategy::Mean => {
                let mut mean = vec![0.0f32; dim];
                for emb in embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        mean[i] += v;
                    }
                }
                let n = embeddings.len() as f32;
                for v in &mut mean {
                    *v /= n;
                }
                mean
            }
            PoolingStrategy::Max => {
                let mut max = vec![f32::NEG_INFINITY; dim];
                for emb in embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        max[i] = max[i].max(v);
                    }
                }
                max
            }
            PoolingStrategy::WeightedMean => {
                // Simple linear weighting - later tokens get more weight
                let mut weighted = vec![0.0f32; dim];
                let mut total_weight = 0.0f32;

                for (pos, emb) in embeddings.iter().enumerate() {
                    let weight = (pos + 1) as f32;
                    total_weight += weight;
                    for (i, &v) in emb.iter().enumerate() {
                        weighted[i] += v * weight;
                    }
                }

                for v in &mut weighted {
                    *v /= total_weight;
                }
                weighted
            }
        }
    }

    /// L2 normalize an embedding
    fn normalize_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Embedding-specific error type
#[derive(thiserror::Error, Debug)]
pub enum EmbeddingError {
    #[error("Tokenization error: {0}")]
    Tokenization(#[from] crate::tokenizer::TokenizerError),

    #[error("Model error: {0}")]
    Model(#[from] crate::model::ModelError),

    #[error("Tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),

    #[error("Empty input")]
    EmptyInput,
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute Euclidean distance between two embeddings
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute dot product between two embeddings
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Find k nearest neighbors from a set of embeddings
pub fn find_nearest(query: &[f32], embeddings: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(query, emb)))
        .collect();

    // Sort by similarity (descending)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scores.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.layer, -1);
        assert!(config.normalize);
        assert_eq!(config.pooling, PoolingStrategy::Mean);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_find_nearest() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0], // Most similar
            vec![0.0, 1.0], // Orthogonal
            vec![0.7, 0.7], // Somewhat similar
        ];

        let nearest = find_nearest(&query, &embeddings, 2);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 0); // First embedding is most similar
    }

    #[test]
    fn test_normalize() {
        let extractor = EmbeddingExtractor {
            config: EmbeddingConfig::default(),
            hidden_dim: 3,
        };

        let embedding = vec![3.0, 4.0, 0.0];
        let normalized = extractor.normalize_embedding(&embedding);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pooling_mean() {
        let extractor = EmbeddingExtractor {
            config: EmbeddingConfig {
                pooling: PoolingStrategy::Mean,
                ..Default::default()
            },
            hidden_dim: 2,
        };

        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let pooled = extractor.pool_embeddings(&embeddings, 2);
        assert!((pooled[0] - 0.5).abs() < 0.001);
        assert!((pooled[1] - 0.5).abs() < 0.001);
    }
}
