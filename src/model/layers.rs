//! Neural network layer building blocks
//!
//! This module provides common layer types used in transformer models.

use crate::backend::{Backend, BackendResult};
use crate::tensor::{DType, Tensor};

use super::error::{ModelError, ModelResult};

/// Linear (fully connected) layer: y = x @ W + b
///
/// GGUF convention: weight is stored as [in_features, out_features]
/// This is transposed from the typical PyTorch convention [out_features, in_features]
#[derive(Debug)]
pub struct Linear {
    /// Weight matrix [in_features, out_features] (GGUF convention)
    pub weight: Tensor,
    /// Optional bias [out_features]
    pub bias: Option<Tensor>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> ModelResult<Self> {
        if weight.ndim() != 2 {
            return Err(ModelError::ConfigError("Linear weight must be 2D".into()));
        }

        // GGUF convention: [in_features, out_features]
        let in_features = weight.shape()[0];
        let out_features = weight.shape()[1];

        if let Some(ref b) = bias
            && b.shape() != [out_features]
        {
            return Err(ModelError::TensorShapeMismatch {
                name: "bias".into(),
                expected: vec![out_features],
                got: b.shape().to_vec(),
            });
        }

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass: y = x @ W + b
    pub fn forward(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        backend: &dyn Backend,
    ) -> BackendResult<()> {
        // For quantized weights, use vec_mat_q (x @ W)
        if self.weight.dtype().is_quantized() {
            backend.vec_mat_q(x, &self.weight, out)?;
        } else {
            backend.vec_mat(x, &self.weight, out)?;
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let mut temp = Tensor::zeros(out.shape().to_vec(), DType::F32);
            backend.add(out, bias, &mut temp)?;
            // Copy temp back to out
            let out_data = out.as_f32_mut()?;
            let temp_data = temp.as_f32()?;
            out_data.copy_from_slice(temp_data);
        }

        Ok(())
    }

    /// Forward pass without bias: y = x @ W
    /// This is useful when bias needs to be applied after another operation (e.g., RoPE)
    pub fn forward_no_bias(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        backend: &dyn Backend,
    ) -> BackendResult<()> {
        if self.weight.dtype().is_quantized() {
            backend.vec_mat_q(x, &self.weight, out)?;
        } else {
            backend.vec_mat(x, &self.weight, out)?;
        }
        Ok(())
    }

    /// Apply bias to output tensor (if bias exists)
    pub fn apply_bias(&self, out: &mut Tensor, backend: &dyn Backend) -> BackendResult<()> {
        if let Some(ref bias) = self.bias {
            let mut temp = Tensor::zeros(out.shape().to_vec(), DType::F32);
            backend.add(out, bias, &mut temp)?;
            let out_data = out.as_f32_mut()?;
            let temp_data = temp.as_f32()?;
            out_data.copy_from_slice(temp_data);
        }
        Ok(())
    }
}

/// RMS Normalization layer
#[derive(Debug)]
pub struct RMSNorm {
    /// Learned scale parameter [hidden_size]
    pub weight: Tensor,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Hidden dimension
    pub hidden_size: usize,
}

impl RMSNorm {
    /// Create a new RMS normalization layer
    pub fn new(weight: Tensor, eps: f32) -> ModelResult<Self> {
        if weight.ndim() != 1 {
            return Err(ModelError::ConfigError("RMSNorm weight must be 1D".into()));
        }

        let hidden_size = weight.shape()[0];

        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }

    /// Forward pass: out = x / rms(x) * weight
    pub fn forward(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        backend: &dyn Backend,
    ) -> BackendResult<()> {
        backend.rms_norm(x, &self.weight, self.eps, out)
    }
}

/// Self-attention layer with Grouped Query Attention support
#[derive(Debug)]
pub struct Attention {
    /// Query projection [num_heads * head_dim, hidden_size]
    pub wq: Linear,
    /// Key projection [num_kv_heads * head_dim, hidden_size]
    pub wk: Linear,
    /// Value projection [num_kv_heads * head_dim, hidden_size]
    pub wv: Linear,
    /// Output projection [hidden_size, num_heads * head_dim]
    pub wo: Linear,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Attention scale factor (1 / sqrt(head_dim))
    pub scale: f32,
    /// Whether to use NeoX-style RoPE (Qwen2) or normal style (LLaMA)
    pub use_neox_rope: bool,
}

impl Attention {
    /// Create a new attention layer
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::with_rope_type(wq, wk, wv, wo, num_heads, num_kv_heads, head_dim, false)
    }

    /// Create a new attention layer with explicit RoPE type
    #[allow(clippy::too_many_arguments)]
    pub fn with_rope_type(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        use_neox_rope: bool,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            use_neox_rope,
        }
    }

    /// Forward pass with KV cache
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, hidden_size] or [hidden_size] for single token
    /// * `k_cache` - Key cache [num_kv_heads, max_seq_len, head_dim]
    /// * `v_cache` - Value cache [num_kv_heads, max_seq_len, head_dim]
    /// * `pos` - Current position in sequence
    /// * `freq_base` - RoPE frequency base
    /// * `freq_scale` - RoPE frequency scale
    /// * `backend` - Computation backend
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
        backend: &dyn Backend,
    ) -> ModelResult<Tensor> {
        let hidden_size = x.shape().last().copied().unwrap_or(0);
        let seq_len = if x.ndim() == 1 { 1 } else { x.shape()[0] };

        // Project Q, K, V
        let mut q = Tensor::zeros(vec![self.num_heads * self.head_dim], DType::F32);
        let mut k = Tensor::zeros(vec![self.num_kv_heads * self.head_dim], DType::F32);
        let mut v = Tensor::zeros(vec![self.num_kv_heads * self.head_dim], DType::F32);

        // For simplicity, handle single token case (most common during generation)
        let x_vec = if x.ndim() == 2 {
            // Take last token for now
            let x_data = x.as_f32()?;
            let start = (seq_len - 1) * hidden_size;
            Tensor::from_f32(&x_data[start..start + hidden_size], vec![hidden_size])?
        } else {
            x.clone()
        };

        // For Qwen2: apply bias BEFORE RoPE
        // This is because Qwen2's biases encode positional information that should be rotated
        // V doesn't use RoPE, so bias order doesn't matter for it
        self.wq.forward(&x_vec, &mut q, backend)?;
        self.wk.forward(&x_vec, &mut k, backend)?;
        self.wv.forward(&x_vec, &mut v, backend)?;

        // Reshape to [num_heads, 1, head_dim] for RoPE
        let mut q_reshaped = q.reshape(vec![self.num_heads, 1, self.head_dim])?;
        let mut k_reshaped = k.reshape(vec![self.num_kv_heads, 1, self.head_dim])?;
        let v_reshaped = v.reshape(vec![self.num_kv_heads, 1, self.head_dim])?;

        // Apply RoPE to current Q and K
        backend.rope(
            &mut q_reshaped,
            &mut k_reshaped,
            pos,
            freq_base,
            freq_scale,
            self.use_neox_rope,
        )?;

        // Get cache dimensions before mutable borrow
        let max_seq_len = k_cache.shape()[1];
        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;

        // Write current K, V to cache at position `pos`
        // k_cache shape: [num_kv_heads, max_seq_len, head_dim]
        // We need to write k_reshaped [num_kv_heads, 1, head_dim] to position pos
        {
            let k_cache_data = k_cache.as_f32_mut()?;
            let k_new_data = k_reshaped.as_f32()?;

            for h in 0..num_kv_heads {
                let cache_offset = h * max_seq_len * head_dim + pos * head_dim;
                let new_offset = h * head_dim;

                k_cache_data[cache_offset..cache_offset + head_dim]
                    .copy_from_slice(&k_new_data[new_offset..new_offset + head_dim]);
            }
        }

        {
            let v_cache_data = v_cache.as_f32_mut()?;
            let v_new_data = v_reshaped.as_f32()?;

            for h in 0..num_kv_heads {
                let cache_offset = h * max_seq_len * head_dim + pos * head_dim;
                let new_offset = h * head_dim;

                v_cache_data[cache_offset..cache_offset + head_dim]
                    .copy_from_slice(&v_new_data[new_offset..new_offset + head_dim]);
            }
        }

        // Build K and V tensors from cache for attention
        // We need [num_kv_heads, kv_len, head_dim] where kv_len = pos + 1
        let kv_len = pos + 1;
        let mut k_for_attn = Tensor::zeros(vec![num_kv_heads, kv_len, head_dim], DType::F32);
        let mut v_for_attn = Tensor::zeros(vec![num_kv_heads, kv_len, head_dim], DType::F32);

        {
            let k_cache_data = k_cache.as_f32()?;
            let k_attn_data = k_for_attn.as_f32_mut()?;

            for h in 0..num_kv_heads {
                for p in 0..kv_len {
                    let cache_offset = h * max_seq_len * head_dim + p * head_dim;
                    let attn_offset = h * kv_len * head_dim + p * head_dim;

                    k_attn_data[attn_offset..attn_offset + head_dim]
                        .copy_from_slice(&k_cache_data[cache_offset..cache_offset + head_dim]);
                }
            }
        }

        {
            let v_cache_data = v_cache.as_f32()?;
            let v_attn_data = v_for_attn.as_f32_mut()?;

            for h in 0..num_kv_heads {
                for p in 0..kv_len {
                    let cache_offset = h * max_seq_len * head_dim + p * head_dim;
                    let attn_offset = h * kv_len * head_dim + p * head_dim;

                    v_attn_data[attn_offset..attn_offset + head_dim]
                        .copy_from_slice(&v_cache_data[cache_offset..cache_offset + head_dim]);
                }
            }
        }

        // Compute attention using full cached K, V
        let mut attn_out = Tensor::zeros(vec![self.num_heads, 1, self.head_dim], DType::F32);
        backend.attention(
            &q_reshaped,
            &k_for_attn,
            &v_for_attn,
            &mut attn_out,
            self.scale,
        )?;

        // Reshape back to [hidden_size]
        let attn_out_flat = attn_out.reshape(vec![self.num_heads * self.head_dim])?;

        // Output projection
        let mut out = Tensor::zeros(vec![hidden_size], DType::F32);
        self.wo.forward(&attn_out_flat, &mut out, backend)?;

        Ok(out)
    }
}

/// Feed-forward network (MLP) layer
#[derive(Debug)]
pub struct FeedForward {
    /// Gate projection [intermediate_size, hidden_size]
    pub w_gate: Linear,
    /// Up projection [intermediate_size, hidden_size]
    pub w_up: Linear,
    /// Down projection [hidden_size, intermediate_size]
    pub w_down: Linear,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate dimension
    pub intermediate_size: usize,
}

impl FeedForward {
    /// Create a new feed-forward layer
    pub fn new(w_gate: Linear, w_up: Linear, w_down: Linear) -> Self {
        let hidden_size = w_down.out_features;
        let intermediate_size = w_gate.out_features;

        Self {
            w_gate,
            w_up,
            w_down,
            hidden_size,
            intermediate_size,
        }
    }

    /// Forward pass: out = down(silu(gate(x)) * up(x))
    pub fn forward(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        backend: &dyn Backend,
    ) -> BackendResult<()> {
        let mut gate = Tensor::zeros(vec![self.intermediate_size], DType::F32);
        let mut up = Tensor::zeros(vec![self.intermediate_size], DType::F32);
        let mut gate_silu = Tensor::zeros(vec![self.intermediate_size], DType::F32);
        let mut intermediate = Tensor::zeros(vec![self.intermediate_size], DType::F32);

        // Compute gate and up projections
        self.w_gate.forward(x, &mut gate, backend)?;
        self.w_up.forward(x, &mut up, backend)?;

        // Apply SiLU to gate
        backend.silu(&gate, &mut gate_silu)?;

        // Multiply gate_silu * up
        backend.mul(&gate_silu, &up, &mut intermediate)?;

        // Down projection
        self.w_down.forward(&intermediate, out, backend)?;

        Ok(())
    }
}

/// Single transformer layer (decoder block)
#[derive(Debug)]
pub struct TransformerLayer {
    /// Attention normalization
    pub attn_norm: RMSNorm,
    /// Self-attention
    pub attention: Attention,
    /// FFN normalization
    pub ffn_norm: RMSNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Layer index
    pub layer_idx: usize,
}

impl TransformerLayer {
    /// Forward pass with residual connections
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
        backend: &dyn Backend,
    ) -> ModelResult<Tensor> {
        let hidden_size = x.shape().last().copied().unwrap_or(0);

        // Attention with residual
        let mut norm_out = Tensor::zeros(x.shape().to_vec(), DType::F32);
        self.attn_norm.forward(x, &mut norm_out, backend)?;

        let attn_out = self.attention.forward(
            &norm_out, k_cache, v_cache, pos, freq_base, freq_scale, backend,
        )?;

        // Residual connection for attention
        let mut h = Tensor::zeros(vec![hidden_size], DType::F32);
        let x_flat = if x.ndim() == 2 {
            let x_data = x.as_f32()?;
            let seq_len = x.shape()[0];
            let start = (seq_len - 1) * hidden_size;
            Tensor::from_f32(&x_data[start..start + hidden_size], vec![hidden_size])?
        } else {
            x.clone()
        };
        backend.add(&x_flat, &attn_out, &mut h)?;

        // FFN with residual
        let mut ffn_norm_out = Tensor::zeros(vec![hidden_size], DType::F32);
        self.ffn_norm.forward(&h, &mut ffn_norm_out, backend)?;

        let mut ffn_out = Tensor::zeros(vec![hidden_size], DType::F32);
        self.ffn.forward(&ffn_norm_out, &mut ffn_out, backend)?;

        // Residual connection for FFN
        let mut out = Tensor::zeros(vec![hidden_size], DType::F32);
        backend.add(&h, &ffn_out, &mut out)?;

        Ok(out)
    }
}
