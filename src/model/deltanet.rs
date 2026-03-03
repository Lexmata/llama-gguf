//! Gated DeltaNet (linear attention with delta rule) for Qwen3Next recurrent layers.
//!
//! Implements the autoregressive path of the delta rule following llama.cpp's
//! `build_delta_net_autoregressive` and `build_qwen3next_linear_attn`.
//!
//! The state update per value head:
//!   S_t = S_{t-1} * exp(gate) + beta * (v - S_{t-1}^T @ k) ⊗ k^T
//!   output = S_t @ q
//!
//! Where gate = softplus(alpha + dt_bias) * ssm_a (negative → decay).

use crate::backend::Backend;
use crate::tensor::{DType, Tensor};

use super::error::ModelResult;
use super::layers::{Linear, RMSNorm};

/// Configuration for a DeltaNet layer, derived from GGUF SSM metadata.
#[derive(Debug, Clone)]
pub struct DeltaNetConfig {
    pub d_inner: usize,
    pub d_state: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
    pub conv_kernel: usize,
    pub qkv_dim: usize,
}

/// Gated DeltaNet layer for recurrent (non-attention) layers.
pub struct DeltaNetLayer {
    pub config: DeltaNetConfig,
    /// Combined QKV projection [hidden_size, qkv_dim]
    pub attn_qkv: Linear,
    /// Output gate projection [hidden_size, d_inner]
    pub attn_gate: Linear,
    /// Beta + Alpha projection [hidden_size, 2 * num_v_heads]
    pub ssm_ba: Linear,
    /// 1D convolution kernel [conv_kernel, qkv_dim]
    pub ssm_conv1d_weight: Tensor,
    /// Decay multiplier per value head [num_v_heads] (negative values → state decays)
    pub ssm_a: Tensor,
    /// Decay bias per value head [num_v_heads]
    pub ssm_dt_bias: Tensor,
    /// Per-head output RMS normalization [head_v_dim]
    pub ssm_norm: RMSNorm,
    /// Output projection [d_inner, hidden_size]
    pub ssm_out: Linear,
}

impl std::fmt::Debug for DeltaNetLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaNetLayer")
            .field("config", &self.config)
            .finish()
    }
}

/// Per-layer recurrent state for DeltaNet.
#[derive(Debug, Clone)]
pub struct DeltaNetState {
    /// Convolution ring buffer: last (kernel_size - 1) QKV vectors.
    /// Layout: [(kernel_size - 1), qkv_dim] stored as [(kernel_size-1) * qkv_dim]
    pub conv_state: Vec<f32>,
    /// SSM state matrices: one [head_v_dim, head_k_dim] per value head.
    /// Flat: [num_v_heads * head_v_dim * head_k_dim]
    pub ssm_state: Vec<f32>,
}

/// Recurrent state for all layers in a Qwen3Next model.
#[derive(Debug, Clone)]
pub struct RecurrentState {
    pub states: Vec<Option<DeltaNetState>>,
}

impl RecurrentState {
    pub fn new(num_layers: usize, is_recurrent: &[bool], config: &DeltaNetConfig) -> Self {
        let states = (0..num_layers)
            .map(|i| {
                if i < is_recurrent.len() && is_recurrent[i] {
                    let conv_len = (config.conv_kernel - 1) * config.qkv_dim;
                    let ssm_len = config.num_v_heads * config.head_v_dim * config.head_k_dim;
                    Some(DeltaNetState {
                        conv_state: vec![0.0; conv_len],
                        ssm_state: vec![0.0; ssm_len],
                    })
                } else {
                    None
                }
            })
            .collect();
        Self { states }
    }

    pub fn reset(&mut self) {
        for ds in self.states.iter_mut().flatten() {
            ds.conv_state.fill(0.0);
            ds.ssm_state.fill(0.0);
        }
    }
}

impl DeltaNetLayer {
    /// Forward pass for a single token (autoregressive decode).
    ///
    /// x: [hidden_size]
    /// state: mutable DeltaNetState for this layer
    /// Returns: [hidden_size]
    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut DeltaNetState,
        backend: &dyn Backend,
    ) -> ModelResult<Tensor> {
        let cfg = &self.config;
        let hidden_size = x.shape().last().copied().unwrap_or(0);

        // 1. Project to QKV space and gate (z)
        let mut qkv = Tensor::zeros(vec![cfg.qkv_dim], DType::F32);
        self.attn_qkv.forward(x, &mut qkv, backend)?;

        let mut z_raw = Tensor::zeros(vec![cfg.d_inner], DType::F32);
        self.attn_gate.forward(x, &mut z_raw, backend)?;

        // 2. Project to beta/alpha
        let mut ba_raw = Tensor::zeros(vec![cfg.num_v_heads * 2], DType::F32);
        self.ssm_ba.forward(x, &mut ba_raw, backend)?;

        let qkv_data = qkv.as_f32()?.to_vec();
        let ba_data = ba_raw.as_f32()?;

        // Split beta and alpha from ssm_ba output
        // ssm_ba output layout: [ba_new_dim, num_k_heads] flattened
        // ba_new_dim = 2 * (num_v_heads / num_k_heads)
        let kv_ratio = cfg.num_v_heads / cfg.num_k_heads.max(1);
        let ba_per_group = 2 * kv_ratio;

        let mut beta = vec![0.0f32; cfg.num_v_heads];
        let mut alpha = vec![0.0f32; cfg.num_v_heads];

        for kh in 0..cfg.num_k_heads {
            let group_offset = kh * ba_per_group;
            for r in 0..kv_ratio {
                let vh = kh * kv_ratio + r;
                beta[vh] = sigmoid(ba_data[group_offset + r]);
                alpha[vh] = ba_data[group_offset + kv_ratio + r];
            }
        }

        // 3. Compute gate (decay): gate = softplus(alpha + dt_bias) * ssm_a
        let ssm_a_data = self.ssm_a.as_f32()?;
        let dt_bias_data = self.ssm_dt_bias.as_f32()?;
        let mut gate = vec![0.0f32; cfg.num_v_heads];
        for h in 0..cfg.num_v_heads {
            gate[h] = softplus(alpha[h] + dt_bias_data[h]) * ssm_a_data[h];
        }

        // 4. Causal 1D convolution on QKV
        let conv_out = self.apply_conv1d(&qkv_data, state)?;

        // 5. Apply SiLU to the convolution output
        let mut conv_silu: Vec<f32> = conv_out.iter().map(|&x| silu(x)).collect();

        // 6. Split into Q, K, V and apply L2 normalization
        let q_dim = cfg.num_k_heads * cfg.head_k_dim;
        let k_dim = cfg.num_k_heads * cfg.head_k_dim;

        let (q_raw, rest) = conv_silu.split_at_mut(q_dim);
        let (k_raw, v_raw) = rest.split_at_mut(k_dim);

        let l2_eps = 1e-6_f32;

        // L2-normalize Q and K per head
        for h in 0..cfg.num_k_heads {
            let offset = h * cfg.head_k_dim;
            l2_normalize_inplace(&mut q_raw[offset..offset + cfg.head_k_dim], l2_eps);
            l2_normalize_inplace(&mut k_raw[offset..offset + cfg.head_k_dim], l2_eps);
        }

        // Scale Q by 1/sqrt(head_k_dim) as in llama.cpp's build_delta_net_autoregressive
        let q_scale = 1.0 / (cfg.head_k_dim as f32).sqrt();
        for q in q_raw.iter_mut() {
            *q *= q_scale;
        }

        // 7. Repeat-interleave Q and K if num_k_heads != num_v_heads
        let q_expanded: Vec<f32>;
        let k_expanded: Vec<f32>;
        if cfg.num_k_heads != cfg.num_v_heads {
            q_expanded = repeat_interleave(q_raw, cfg.num_k_heads, cfg.head_k_dim, kv_ratio);
            k_expanded = repeat_interleave(k_raw, cfg.num_k_heads, cfg.head_k_dim, kv_ratio);
        } else {
            q_expanded = q_raw.to_vec();
            k_expanded = k_raw.to_vec();
        }

        // 8. Delta rule update per value head
        let mut output = vec![0.0f32; cfg.d_inner];

        for vh in 0..cfg.num_v_heads {
            let s_offset = vh * cfg.head_v_dim * cfg.head_k_dim;
            let v_offset = vh * cfg.head_v_dim;
            let q_offset = vh * cfg.head_k_dim;
            let k_offset = vh * cfg.head_k_dim;
            let o_offset = vh * cfg.head_v_dim;

            let s = &mut state.ssm_state[s_offset..s_offset + cfg.head_v_dim * cfg.head_k_dim];

            // Decay state: s = s * exp(gate[vh])
            let decay = gate[vh].exp().min(1e10);
            for x in s.iter_mut() {
                *x *= decay;
            }

            // sk = s^T @ k  → [head_v_dim]
            let mut sk = vec![0.0f32; cfg.head_v_dim];
            #[allow(clippy::needless_range_loop)]
            for vi in 0..cfg.head_v_dim {
                let row_start = vi * cfg.head_k_dim;
                let mut dot = 0.0f32;
                for ki in 0..cfg.head_k_dim {
                    dot += s[row_start + ki] * k_expanded[k_offset + ki];
                }
                sk[vi] = dot;
            }

            // delta = (v - sk) * beta[vh]  → [head_v_dim]
            let b = beta[vh];
            let mut delta = vec![0.0f32; cfg.head_v_dim];
            #[allow(clippy::needless_range_loop)]
            for vi in 0..cfg.head_v_dim {
                delta[vi] = (v_raw[v_offset + vi] - sk[vi]) * b;
            }

            // State update: s += delta @ k^T  → outer product
            #[allow(clippy::needless_range_loop)]
            for vi in 0..cfg.head_v_dim {
                let row_start = vi * cfg.head_k_dim;
                for ki in 0..cfg.head_k_dim {
                    s[row_start + ki] += delta[vi] * k_expanded[k_offset + ki];
                }
            }

            // Output: o = s @ q  → [head_v_dim]
            for vi in 0..cfg.head_v_dim {
                let row_start = vi * cfg.head_k_dim;
                let mut dot = 0.0f32;
                for ki in 0..cfg.head_k_dim {
                    dot += s[row_start + ki] * q_expanded[q_offset + ki];
                }
                output[o_offset + vi] = dot;
            }
        }

        // 9. Gated normalization: result = rms_norm(output) * silu(z)
        let norm_w = self.ssm_norm.weight.as_f32()?;
        let norm_eps = self.ssm_norm.eps;
        let z_data = z_raw.as_f32()?;

        for vh in 0..cfg.num_v_heads {
            let offset = vh * cfg.head_v_dim;
            let ss: f32 = output[offset..offset + cfg.head_v_dim]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                / cfg.head_v_dim as f32;
            let rms = (ss + norm_eps).sqrt();
            for d in 0..cfg.head_v_dim {
                let normed = output[offset + d] / rms * norm_w[d % norm_w.len()];
                output[offset + d] = normed * silu(z_data[offset + d]);
            }
        }

        // 10. Output projection
        let output_tensor = Tensor::from_f32(&output, vec![cfg.d_inner])?;
        let mut result = Tensor::zeros(vec![hidden_size], DType::F32);
        self.ssm_out.forward(&output_tensor, &mut result, backend)?;

        Ok(result)
    }

    /// Apply 1D causal depthwise convolution using the ring buffer state.
    ///
    /// State holds the last (kernel_size - 1) inputs. The convolution uses
    /// the state positions for kernel taps 0..ks-2, and the current input qkv
    /// for the final tap (ks-1). The state is updated AFTER the convolution.
    ///
    /// Conv weight layout (GGML): [kernel_size, channels] → data[ch * kernel_size + ki]
    fn apply_conv1d(
        &self,
        qkv: &[f32],
        state: &mut DeltaNetState,
    ) -> ModelResult<Vec<f32>> {
        let cfg = &self.config;
        let channels = cfg.qkv_dim;
        let ks = cfg.conv_kernel;
        let buf_len = ks - 1;

        let conv_w = self.ssm_conv1d_weight.as_f32()?;

        // Depthwise 1D convolution: out[ch] = sum_k(input[k][ch] * weight[ch][k])
        // State holds positions [t-(ks-1), ..., t-1], current qkv is position t
        let mut out = vec![0.0f32; channels];

        for ch in 0..channels {
            let mut sum = 0.0f32;
            for ki in 0..buf_len {
                sum += state.conv_state[ki * channels + ch] * conv_w[ch * ks + ki];
            }
            sum += qkv[ch] * conv_w[ch * ks + (ks - 1)];
            out[ch] = sum;
        }

        // Update state: shift left and append current qkv
        if buf_len > 1 {
            state
                .conv_state
                .copy_within(channels..buf_len * channels, 0);
        }
        let last_start = (buf_len - 1) * channels;
        state.conv_state[last_start..last_start + channels].copy_from_slice(qkv);

        Ok(out)
    }
}

/// Repeat-interleave: expand [num_k_heads, head_dim] to [num_v_heads, head_dim]
/// by repeating each head `repeat` times.
fn repeat_interleave(data: &[f32], num_heads: usize, head_dim: usize, repeat: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * repeat * head_dim];
    for h in 0..num_heads {
        let src = h * head_dim;
        for r in 0..repeat {
            let dst = (h * repeat + r) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn l2_normalize_inplace(v: &mut [f32], eps: f32) {
    let sum_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = (sum_sq + eps).sqrt();
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-4);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.6931).abs() < 1e-3);
        assert!((softplus(25.0) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize_inplace(&mut v, 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-4);
        assert!((v[1] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_repeat_interleave() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2 heads, dim=2
        let out = repeat_interleave(&data, 2, 2, 3); // repeat 3x → 6 heads
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
    }
}
