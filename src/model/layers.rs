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
    /// Query projection
    pub wq: Linear,
    /// Key projection
    pub wk: Linear,
    /// Value projection
    pub wv: Linear,
    /// Output projection
    pub wo: Linear,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head for traditional models (hidden_size / num_heads)
    pub head_dim: usize,
    /// Per-head key/query dimension (may differ from head_dim)
    pub key_length: usize,
    /// Per-head value dimension (may differ from head_dim)
    pub value_length: usize,
    /// Number of RoPE dimensions per head (may be less than key_length)
    pub rope_dims: usize,
    /// Attention scale factor (1 / sqrt(key_length))
    pub scale: f32,
    /// Whether to use NeoX-style RoPE (Qwen2) or normal style (LLaMA)
    pub use_neox_rope: bool,
    /// Whether Q projection includes an attention gate
    pub has_attention_gate: bool,
    /// Optional per-head Q normalization (Qwen3)
    pub q_norm: Option<RMSNorm>,
    /// Optional per-head K normalization (Qwen3)
    pub k_norm: Option<RMSNorm>,
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
            key_length: head_dim,
            value_length: head_dim,
            rope_dims: head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            use_neox_rope,
            has_attention_gate: false,
            q_norm: None,
            k_norm: None,
        }
    }

    /// Create attention with explicit key/value/rope dimensions (for Qwen3Next etc.)
    #[allow(clippy::too_many_arguments)]
    pub fn with_kv_dims(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        rope_dims: usize,
        use_neox_rope: bool,
        has_attention_gate: bool,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            key_length,
            value_length,
            rope_dims,
            scale: 1.0 / (key_length as f32).sqrt(),
            use_neox_rope,
            has_attention_gate,
            q_norm: None,
            k_norm: None,
        }
    }

    /// Set QK normalization layers (used by Qwen3/Qwen3Moe)
    pub fn set_qk_norms(&mut self, q_norm: RMSNorm, k_norm: RMSNorm) {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
    }

    /// Forward pass with KV cache
    ///
    /// Supports standard attention and Qwen3Next-style attention with:
    /// - Separate nope/rope dimensions (partial RoPE application)
    /// - Attention gating (Q includes gate component)
    /// - Different key_length/value_length from head_dim
    ///
    /// KV caches: `[num_kv_heads, max_seq_len, key_length]`
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
        let kl = self.key_length;
        let vl = self.value_length;

        let x_vec = if x.ndim() == 2 {
            let x_data = x.as_f32()?;
            let start = (seq_len - 1) * hidden_size;
            Tensor::from_f32(&x_data[start..start + hidden_size], vec![hidden_size])?
        } else {
            x.clone()
        };

        // Determine Q output size: with gate, Q outputs Q proper + gate
        let q_out_size = self.wq.out_features;
        let mut q_raw = Tensor::zeros(vec![q_out_size], DType::F32);
        let mut k = Tensor::zeros(vec![self.num_kv_heads * kl], DType::F32);
        let mut v = Tensor::zeros(vec![self.num_kv_heads * vl], DType::F32);

        self.wq.forward(&x_vec, &mut q_raw, backend)?;
        self.wk.forward(&x_vec, &mut k, backend)?;
        self.wv.forward(&x_vec, &mut v, backend)?;

        // Split Q into q_proper and gate if attention gating is active
        // Per head in Q output: [q_proper(kl) | gate(vl)]
        let (q_proper_data, gate_data) = if self.has_attention_gate {
            let raw = q_raw.as_f32()?;
            let q_proper_len = self.num_heads * kl;
            let gate_len = self.num_heads * vl;
            if raw.len() >= q_proper_len + gate_len {
                // Per-head split: each head has (kl + vl) contiguous elements
                let per_head_total = kl + vl;
                let mut q_buf = vec![0.0f32; q_proper_len];
                let mut g_buf = vec![0.0f32; gate_len];
                for h in 0..self.num_heads {
                    let src = h * per_head_total;
                    let q_dst = h * kl;
                    let g_dst = h * vl;
                    q_buf[q_dst..q_dst + kl].copy_from_slice(&raw[src..src + kl]);
                    g_buf[g_dst..g_dst + vl].copy_from_slice(&raw[src + kl..src + kl + vl]);
                }
                (q_buf, Some(g_buf))
            } else {
                (raw.to_vec(), None)
            }
        } else {
            (q_raw.as_f32()?.to_vec(), None)
        };

        let q_flat = Tensor::from_f32(&q_proper_data, vec![self.num_heads * kl])?;

        // Reshape for per-head operations
        let mut q_reshaped = q_flat.reshape(vec![self.num_heads, 1, kl])?;
        let mut k_reshaped = k.reshape(vec![self.num_kv_heads, 1, kl])?;
        let v_reshaped = v.reshape(vec![self.num_kv_heads, 1, vl])?;

        // Per-head QK normalization before RoPE
        if let Some(ref q_norm) = self.q_norm {
            let q_data = q_reshaped.as_f32()?.to_vec();
            let q_out = q_reshaped.as_f32_mut()?;
            let norm_w = q_norm.weight.as_f32()?;
            let norm_dim = norm_w.len();
            for h in 0..self.num_heads {
                let offset = h * kl;
                let head_slice = &q_data[offset..offset + kl];
                let ss: f32 = head_slice[..norm_dim].iter().map(|x| x * x).sum::<f32>()
                    / norm_dim as f32;
                let rms = (ss + q_norm.eps).sqrt();
                for d in 0..norm_dim.min(kl) {
                    q_out[offset + d] = head_slice[d] / rms * norm_w[d];
                }
            }
        }
        if let Some(ref k_norm) = self.k_norm {
            let k_data = k_reshaped.as_f32()?.to_vec();
            let k_out = k_reshaped.as_f32_mut()?;
            let norm_w = k_norm.weight.as_f32()?;
            let norm_dim = norm_w.len();
            for h in 0..self.num_kv_heads {
                let offset = h * kl;
                let head_slice = &k_data[offset..offset + kl];
                let ss: f32 = head_slice[..norm_dim].iter().map(|x| x * x).sum::<f32>()
                    / norm_dim as f32;
                let rms = (ss + k_norm.eps).sqrt();
                for d in 0..norm_dim.min(kl) {
                    k_out[offset + d] = head_slice[d] / rms * norm_w[d];
                }
            }
        }

        // Partial RoPE: rotate only the LAST `rope_dims` dimensions per head.
        // Per-head layout: [nope(kl - rope_dims) | rope(rope_dims)]
        // This matches llama.cpp's Qwen3Next: Qnope = first dims, Qrope = last dims.
        if self.rope_dims > 0 && self.rope_dims < kl {
            let nope_dims = kl - self.rope_dims;
            let q_data = q_reshaped.as_f32()?.to_vec();
            let k_data = k_reshaped.as_f32()?.to_vec();

            let mut q_rope = vec![0.0f32; self.num_heads * self.rope_dims];
            let mut k_rope = vec![0.0f32; self.num_kv_heads * self.rope_dims];

            for h in 0..self.num_heads {
                let src = h * kl + nope_dims;
                let dst = h * self.rope_dims;
                q_rope[dst..dst + self.rope_dims]
                    .copy_from_slice(&q_data[src..src + self.rope_dims]);
            }
            for h in 0..self.num_kv_heads {
                let src = h * kl + nope_dims;
                let dst = h * self.rope_dims;
                k_rope[dst..dst + self.rope_dims]
                    .copy_from_slice(&k_data[src..src + self.rope_dims]);
            }

            let mut q_rope_t =
                Tensor::from_f32(&q_rope, vec![self.num_heads, 1, self.rope_dims])?;
            let mut k_rope_t =
                Tensor::from_f32(&k_rope, vec![self.num_kv_heads, 1, self.rope_dims])?;

            backend.rope(
                &mut q_rope_t,
                &mut k_rope_t,
                pos,
                freq_base,
                freq_scale,
                self.use_neox_rope,
            )?;

            let q_rope_out = q_rope_t.as_f32()?;
            let k_rope_out = k_rope_t.as_f32()?;
            let q_out = q_reshaped.as_f32_mut()?;
            let k_out = k_reshaped.as_f32_mut()?;

            for h in 0..self.num_heads {
                let dst = h * kl + nope_dims;
                let src = h * self.rope_dims;
                q_out[dst..dst + self.rope_dims]
                    .copy_from_slice(&q_rope_out[src..src + self.rope_dims]);
            }
            for h in 0..self.num_kv_heads {
                let dst = h * kl + nope_dims;
                let src = h * self.rope_dims;
                k_out[dst..dst + self.rope_dims]
                    .copy_from_slice(&k_rope_out[src..src + self.rope_dims]);
            }
        } else {
            // Standard: RoPE on all dimensions
            backend.rope(
                &mut q_reshaped,
                &mut k_reshaped,
                pos,
                freq_base,
                freq_scale,
                self.use_neox_rope,
            )?;
        }

        let max_seq_len = k_cache.shape()[1];
        let num_kv_heads = self.num_kv_heads;

        // Write K, V to cache
        {
            let k_cache_data = k_cache.as_f32_mut()?;
            let k_new_data = k_reshaped.as_f32()?;
            for h in 0..num_kv_heads {
                let cache_offset = h * max_seq_len * kl + pos * kl;
                let new_offset = h * kl;
                k_cache_data[cache_offset..cache_offset + kl]
                    .copy_from_slice(&k_new_data[new_offset..new_offset + kl]);
            }
        }
        {
            let v_cache_data = v_cache.as_f32_mut()?;
            let v_new_data = v_reshaped.as_f32()?;
            for h in 0..num_kv_heads {
                let cache_offset = h * max_seq_len * vl + pos * vl;
                let new_offset = h * vl;
                v_cache_data[cache_offset..cache_offset + vl]
                    .copy_from_slice(&v_new_data[new_offset..new_offset + vl]);
            }
        }

        // Compute attention
        let kv_len = pos + 1;
        let mut attn_out = Tensor::zeros(vec![self.num_heads, 1, vl], DType::F32);
        backend.attention_cached(
            &q_reshaped,
            k_cache,
            v_cache,
            &mut attn_out,
            self.scale,
            kv_len,
        )?;

        // Apply attention gate: output = sigmoid(gate) * attn_output
        let attn_flat = if let Some(ref gate) = gate_data {
            let attn_data = attn_out.as_f32()?;
            let total = self.num_heads * vl;
            let mut gated = vec![0.0f32; total];
            for i in 0..total {
                let sigmoid_g = 1.0 / (1.0 + (-gate[i]).exp());
                gated[i] = sigmoid_g * attn_data[i];
            }
            Tensor::from_f32(&gated, vec![total])?
        } else {
            attn_out.reshape(vec![self.num_heads * vl])?
        };

        // Output projection
        let mut out = Tensor::zeros(vec![hidden_size], DType::F32);
        self.wo.forward(&attn_flat, &mut out, backend)?;

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

/// FFN variant: either dense FFN or Mixture of Experts
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum FfnLayer {
    /// Standard dense feed-forward network
    Dense(FeedForward),
    /// Mixture of Experts
    Moe(super::moe::MoeLayer),
}

/// Attention variant: full softmax attention or delta-net recurrent.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum AttentionLayer {
    /// Standard multi-head softmax attention (with optional gating, partial RoPE)
    FullAttention(Attention),
    /// Gated DeltaNet linear attention (SSM/recurrent)
    DeltaNet(Box<super::deltanet::DeltaNetLayer>),
}

/// Single transformer layer (decoder block)
#[derive(Debug)]
pub struct TransformerLayer {
    /// Attention normalization
    pub attn_norm: RMSNorm,
    /// Attention: either full softmax or delta-net recurrent
    pub attn_layer: AttentionLayer,
    /// Optional post-attention normalization (Qwen3Next)
    pub post_attn_norm: Option<RMSNorm>,
    /// FFN normalization
    pub ffn_norm: RMSNorm,
    /// Feed-forward network (dense or MoE)
    pub ffn_layer: FfnLayer,
    /// Layer index
    pub layer_idx: usize,
}

impl TransformerLayer {
    /// Get the full attention layer if this is not a delta-net layer
    pub fn attention(&self) -> Option<&Attention> {
        match &self.attn_layer {
            AttentionLayer::FullAttention(attn) => Some(attn),
            AttentionLayer::DeltaNet(_) => None,
        }
    }

    /// Get the dense FFN layer if this is not an MoE layer
    pub fn ffn(&self) -> Option<&FeedForward> {
        match &self.ffn_layer {
            FfnLayer::Dense(ffn) => Some(ffn),
            FfnLayer::Moe(_) => None,
        }
    }

    /// Get the MoE layer if this is not a dense FFN layer
    pub fn moe(&self) -> Option<&super::moe::MoeLayer> {
        match &self.ffn_layer {
            FfnLayer::Dense(_) => None,
            FfnLayer::Moe(moe) => Some(moe),
        }
    }

    /// Whether this is a recurrent (delta-net) layer
    pub fn is_recurrent(&self) -> bool {
        matches!(&self.attn_layer, AttentionLayer::DeltaNet(_))
    }

    /// Forward pass with residual connections.
    /// `recurrent_state` is used only for DeltaNet layers.
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
        recurrent_state: Option<&mut super::deltanet::DeltaNetState>,
    ) -> ModelResult<Tensor> {
        let hidden_size = x.shape().last().copied().unwrap_or(0);

        // Attention normalization
        let mut norm_out = Tensor::zeros(x.shape().to_vec(), DType::F32);
        self.attn_norm.forward(x, &mut norm_out, backend)?;

        // Run attention (full or delta-net)
        let attn_out = match &self.attn_layer {
            AttentionLayer::FullAttention(attn) => {
                attn.forward(&norm_out, k_cache, v_cache, pos, freq_base, freq_scale, backend)?
            }
            AttentionLayer::DeltaNet(dn) => {
                let state = recurrent_state.ok_or_else(|| {
                    ModelError::ConfigError(
                        "DeltaNet layer requires recurrent state".into(),
                    )
                })?;
                dn.forward(&norm_out, state, backend)?
            }
        };

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

        // FFN normalization: use post_attn_norm if present (Qwen3Next),
        // otherwise use ffn_norm (standard models)
        let mut ffn_norm_out = Tensor::zeros(vec![hidden_size], DType::F32);
        if let Some(ref pan) = self.post_attn_norm {
            pan.forward(&h, &mut ffn_norm_out, backend)?;
        } else {
            self.ffn_norm.forward(&h, &mut ffn_norm_out, backend)?;
        }

        let ffn_out = match &self.ffn_layer {
            FfnLayer::Dense(ffn) => {
                let mut out = Tensor::zeros(vec![hidden_size], DType::F32);
                ffn.forward(&ffn_norm_out, &mut out, backend)?;
                out
            }
            FfnLayer::Moe(moe) => moe.forward(&ffn_norm_out, backend)?,
        };

        // Residual connection for FFN
        let mut out = Tensor::zeros(vec![hidden_size], DType::F32);
        backend.add(&h, &ffn_out, &mut out)?;

        Ok(out)
    }
}
