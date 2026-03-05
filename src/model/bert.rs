//! BERT encoder-only model implementation
//!
//! Supports BERT, ModernBert, NomicBert, JinaBert, and similar
//! encoder-only transformers for embedding generation.

use crate::backend::Backend;
use crate::tensor::{DType, Tensor};

use super::config::ModelConfig;
use super::error::{ModelError, ModelResult};
use super::layers::{Linear, NormLayer};
use super::{Architecture, InferenceContext, Model};

/// BERT encoder-only transformer model
pub struct BertModel {
    config: ModelConfig,
    token_embedding: Tensor,
    position_embedding: Option<Tensor>,
    token_type_embedding: Option<Tensor>,
    embed_norm: Option<NormLayer>,
    layers: Vec<BertLayer>,
    architecture: Architecture,
}

/// A single BERT encoder layer
pub struct BertLayer {
    pub attn_norm: NormLayer,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_norm: NormLayer,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl BertLayer {
    /// Bidirectional self-attention forward pass (no causal mask)
    pub fn forward(
        &self,
        hiddens: &[Tensor],
        backend: &dyn Backend,
    ) -> ModelResult<Vec<Tensor>> {
        let seq_len = hiddens.len();
        let hidden_size = hiddens[0].shape()[0];

        // Project Q, K, V for all positions
        let mut all_q = Vec::with_capacity(seq_len);
        let mut all_k = Vec::with_capacity(seq_len);
        let mut all_v = Vec::with_capacity(seq_len);

        for h in hiddens {
            let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
            self.attn_norm.forward(h, &mut normed, backend)?;

            let q_size = self.num_heads * self.head_dim;
            let k_size = self.num_heads * self.head_dim;
            let v_size = self.num_heads * self.head_dim;

            let mut q = Tensor::zeros(vec![q_size], DType::F32);
            let mut k = Tensor::zeros(vec![k_size], DType::F32);
            let mut v = Tensor::zeros(vec![v_size], DType::F32);

            self.wq.forward(&normed, &mut q, backend)?;
            self.wk.forward(&normed, &mut k, backend)?;
            self.wv.forward(&normed, &mut v, backend)?;

            all_q.push(q);
            all_k.push(k);
            all_v.push(v);
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Compute bidirectional attention for each position
        let mut attn_outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let q_data = all_q[i].as_f32()?;
            let mut out = vec![0.0f32; self.num_heads * self.head_dim];

            for head in 0..self.num_heads {
                let q_offset = head * self.head_dim;
                let q_head = &q_data[q_offset..q_offset + self.head_dim];

                // Compute attention scores against ALL positions (bidirectional)
                let mut scores = vec![0.0f32; seq_len];
                for j in 0..seq_len {
                    let k_data = all_k[j].as_f32()?;
                    let k_head = &k_data[q_offset..q_offset + self.head_dim];
                    let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                    scores[j] = dot * scale;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Weighted sum of values
                for j in 0..seq_len {
                    let v_data = all_v[j].as_f32()?;
                    let v_head = &v_data[q_offset..q_offset + self.head_dim];
                    for d in 0..self.head_dim {
                        out[q_offset + d] += scores[j] * v_head[d];
                    }
                }
            }

            let attn_flat = Tensor::from_f32(&out, vec![self.num_heads * self.head_dim])?;
            let mut projected = Tensor::zeros(vec![hidden_size], DType::F32);
            self.wo.forward(&attn_flat, &mut projected, backend)?;

            // Residual
            let proj_data = projected.as_f32_mut()?;
            let h_data = hiddens[i].as_f32()?;
            for (p, &h) in proj_data.iter_mut().zip(h_data.iter()) {
                *p += h;
            }

            attn_outputs.push(projected);
        }

        // FFN with residual for each position
        let mut outputs = Vec::with_capacity(seq_len);
        for h in &attn_outputs {
            let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
            self.ffn_norm.forward(h, &mut normed, backend)?;

            let intermediate_size = self.ffn_up.out_features;
            let mut up = Tensor::zeros(vec![intermediate_size], DType::F32);
            self.ffn_up.forward(&normed, &mut up, backend)?;

            // GELU activation
            {
                let data = up.as_f32_mut()?;
                for v in data.iter_mut() {
                    let x = *v;
                    *v = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh());
                }
            }

            let mut down = Tensor::zeros(vec![hidden_size], DType::F32);
            self.ffn_down.forward(&up, &mut down, backend)?;

            // Residual
            let d = down.as_f32_mut()?;
            let h_data = h.as_f32()?;
            for (dd, &hv) in d.iter_mut().zip(h_data.iter()) {
                *dd += hv;
            }

            outputs.push(down);
        }

        Ok(outputs)
    }
}

impl BertModel {
    pub fn new(
        config: ModelConfig,
        token_embedding: Tensor,
        position_embedding: Option<Tensor>,
        token_type_embedding: Option<Tensor>,
        embed_norm: Option<NormLayer>,
        layers: Vec<BertLayer>,
        architecture: Architecture,
    ) -> ModelResult<Self> {
        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            token_type_embedding,
            embed_norm,
            layers,
            architecture,
        })
    }

    /// Encode tokens and return hidden states for all positions.
    /// Returns the hidden state of the [CLS] token (first position) by default.
    pub fn encode(&self, tokens: &[u32], backend: &dyn Backend) -> ModelResult<Vec<Tensor>> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // Build embeddings
        let emb_data = if self.token_embedding.dtype() == DType::F32 {
            std::borrow::Cow::Borrowed(self.token_embedding.as_f32()?)
        } else {
            let numel = self.token_embedding.numel();
            let mut dequant = Tensor::zeros(vec![numel], DType::F32);
            backend.dequantize(&self.token_embedding, &mut dequant)?;
            std::borrow::Cow::Owned(dequant.as_f32()?.to_vec())
        };

        let pos_data = self
            .position_embedding
            .as_ref()
            .map(|p| p.as_f32())
            .transpose()?;
        let type_data = self
            .token_type_embedding
            .as_ref()
            .map(|t| t.as_f32())
            .transpose()?;

        let mut hiddens: Vec<Tensor> = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            let idx = token as usize;
            if idx >= vocab_size {
                return Err(ModelError::InvalidMetadata {
                    key: "token".into(),
                    message: format!("Token {} >= vocab_size {}", token, vocab_size),
                });
            }

            let src = idx * hidden_size;
            let mut h = emb_data[src..src + hidden_size].to_vec();

            // Add position embedding
            if let Some(ref pos) = pos_data {
                let pos_src = i * hidden_size;
                if pos_src + hidden_size <= pos.len() {
                    for (j, val) in h.iter_mut().enumerate() {
                        *val += pos[pos_src + j];
                    }
                }
            }

            // Add token type embedding (segment 0 for now)
            if let Some(ref type_emb) = type_data {
                for (j, val) in h.iter_mut().enumerate().take(hidden_size) {
                    if j < type_emb.len() {
                        *val += type_emb[j]; // segment 0
                    }
                }
            }

            let mut t = Tensor::from_f32(&h, vec![hidden_size])?;

            // Apply embedding normalization if present
            if let Some(ref norm) = self.embed_norm {
                let mut normed = Tensor::zeros(vec![hidden_size], DType::F32);
                norm.forward(&t, &mut normed, backend)?;
                t = normed;
            }

            hiddens.push(t);
        }

        // Run through encoder layers
        for layer in &self.layers {
            hiddens = layer.forward(&hiddens, backend)?;
        }

        Ok(hiddens)
    }
}

impl Model for BertModel {
    fn forward(&self, tokens: &[u32], ctx: &mut InferenceContext) -> ModelResult<Tensor> {
        let backend = ctx.backend.as_ref();
        let hiddens = self.encode(tokens, backend)?;

        // Return the CLS token (position 0) hidden state
        // For actual embedding use, callers should use encode() directly
        if hiddens.is_empty() {
            return Err(ModelError::ConfigError("Empty input".into()));
        }
        Ok(hiddens[0].clone())
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn architecture(&self) -> Architecture {
        self.architecture
    }
}

impl std::fmt::Debug for BertModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BertModel")
            .field("architecture", &self.architecture)
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl std::fmt::Debug for BertLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BertLayer")
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}
