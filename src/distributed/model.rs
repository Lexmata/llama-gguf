//! Distributed model that implements the `Model` trait
//!
//! Wraps the pipeline executor to provide transparent distributed inference
//! through the same `Model` interface used by `Engine`.

use tokio::runtime::Handle;

use crate::backend::Backend;
use crate::model::{
    Architecture, InferenceContext, Model, ModelConfig, ModelError, ModelResult,
};
use crate::model::layers::{Linear, RMSNorm};
use crate::tensor::{DType, Tensor};

use super::pipeline::PipelineExecutor;

/// A model that distributes transformer layers across remote shards
/// while keeping embedding and output projection on the coordinator.
///
/// Implements the `Model` trait so it can be used transparently with `Engine`.
pub struct DistributedModel {
    /// Token embedding table (kept on coordinator)
    token_embedding: Tensor,
    /// Final RMS normalization (kept on coordinator)
    norm: RMSNorm,
    /// Output projection to vocab logits (kept on coordinator)
    output: Linear,
    /// Pipeline executor that chains forward passes across shards
    pipeline: tokio::sync::Mutex<PipelineExecutor>,
    /// Model configuration
    config: ModelConfig,
    /// Model architecture
    architecture: Architecture,
}

// SAFETY: The pipeline mutex ensures exclusive access during forward passes.
// The other fields are immutable after construction.
unsafe impl Send for DistributedModel {}
unsafe impl Sync for DistributedModel {}

impl DistributedModel {
    /// Create a new distributed model.
    pub fn new(
        token_embedding: Tensor,
        norm: RMSNorm,
        output: Linear,
        pipeline: PipelineExecutor,
        config: ModelConfig,
        architecture: Architecture,
    ) -> Self {
        Self {
            token_embedding,
            norm,
            output,
            pipeline: tokio::sync::Mutex::new(pipeline),
            config,
            architecture,
        }
    }

    /// Dequantize the embedding table to f32 for token lookup.
    fn dequantize_embeddings(&self, backend: &dyn Backend) -> ModelResult<Vec<f32>> {
        if self.token_embedding.dtype() == DType::F32 {
            return Ok(self.token_embedding.as_f32()?.to_vec());
        }

        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let mut out = Tensor::zeros(vec![vocab_size, hidden_size], DType::F32);
        backend.dequantize(&self.token_embedding, &mut out)?;
        Ok(out.as_f32()?.to_vec())
    }

    /// Compute logits from hidden state.
    fn compute_logits(&self, hidden: &Tensor, backend: &dyn Backend) -> ModelResult<Tensor> {
        let mut normed = Tensor::zeros(hidden.shape().to_vec(), DType::F32);
        self.norm.forward(hidden, &mut normed, backend)?;

        let mut logits = Tensor::zeros(vec![self.config.vocab_size], DType::F32);
        self.output.forward(&normed, &mut logits, backend)?;

        Ok(logits)
    }
}

impl Model for DistributedModel {
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor> {
        let backend = ctx.backend.as_ref();

        let new_pos = ctx.position + tokens.len();
        if new_pos > self.config.max_seq_len {
            return Err(ModelError::ContextLengthExceeded {
                current: new_pos,
                max: self.config.max_seq_len,
            });
        }

        let embedding_data = self.dequantize_embeddings(backend)?;
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // We need to call async pipeline.forward() from sync Model::forward().
        // Use the tokio runtime handle from the current async context.
        let handle = Handle::try_current().map_err(|_| {
            ModelError::ConfigError(
                "distributed model requires a tokio runtime".into(),
            )
        })?;

        let mut hidden_buf = vec![0.0f32; hidden_size];

        for (token_offset, &token) in tokens.iter().enumerate() {
            let current_pos = ctx.position + token_offset;
            let token_idx = token as usize;

            if token_idx >= vocab_size {
                return Err(ModelError::InvalidMetadata {
                    key: "token".into(),
                    message: format!("Token ID {} exceeds vocab size {}", token, vocab_size),
                });
            }

            // Look up embedding
            let src = token_idx * hidden_size;
            hidden_buf.copy_from_slice(&embedding_data[src..src + hidden_size]);
            let hidden = Tensor::from_f32(&hidden_buf, vec![hidden_size])?;

            // Send through the distributed pipeline (async -> sync bridge)
            let pipeline_result = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    let mut pipeline = self.pipeline.lock().await;
                    pipeline.forward(&hidden, current_pos).await
                })
            });

            let output_hidden = pipeline_result.map_err(|e| {
                ModelError::ConfigError(format!("distributed forward failed: {}", e))
            })?;

            if token_offset + 1 == tokens.len() {
                ctx.position = new_pos;
                ctx.kv_cache.seq_len = new_pos;
                return self.compute_logits(&output_hidden, backend);
            }
        }

        Err(ModelError::ConfigError("No tokens to process".into()))
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn architecture(&self) -> Architecture {
        self.architecture
    }
}
