//! Mamba v1 selective state space model (SSM) layer.
//!
//! Implements the Mamba architecture as described in "Mamba: Linear-Time Sequence
//! Modeling with Selective State Spaces". This is distinct from DeltaNet (Qwen3Next).
//!
//! Flow per token:
//! 1. Input projection: x -> (x_ssm, z) via ssm_in [hidden, 2*d_inner]
//! 2. Causal 1D convolution on x_ssm
//! 3. SiLU activation
//! 4. Selective scan: dt, B, C from x_db projection; state update; output
//! 5. Gate: y = y * silu(z)
//! 6. Output projection: ssm_out

use crate::backend::Backend;
use crate::tensor::{DType, Tensor};

use super::error::ModelResult;
use super::layers::{Linear, RMSNorm};

/// Configuration for a Mamba layer.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub conv_kernel: usize,
}

/// Per-layer recurrent state for Mamba.
#[derive(Debug, Clone)]
pub struct MambaState {
    /// Convolution ring buffer: last (kernel_size - 1) vectors of size d_inner.
    pub conv_state: Vec<f32>,
    /// SSM state: [d_inner * d_state], layout s[d * d_state + n]
    pub ssm_state: Vec<f32>,
}

impl MambaState {
    pub fn new(config: &MambaConfig) -> Self {
        let conv_len = (config.conv_kernel - 1) * config.d_inner;
        let ssm_len = config.d_inner * config.d_state;
        Self {
            conv_state: vec![0.0; conv_len],
            ssm_state: vec![0.0; ssm_len],
        }
    }

    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
    }
}

/// Mamba v1 SSM layer.
pub struct MambaLayer {
    /// Input projection [hidden, 2*d_inner] -> (x_ssm, z)
    pub ssm_in: Linear,
    /// 1D convolution kernel [conv_kernel, d_inner]
    pub ssm_conv1d_weight: Tensor,
    /// Optional conv bias [d_inner]
    pub ssm_conv1d_bias: Option<Tensor>,
    /// x_db projection [d_inner, dt_rank + 2*d_state]
    pub ssm_x: Linear,
    /// dt projection [dt_rank, d_inner]
    pub ssm_dt: Linear,
    /// dt bias [d_inner]
    pub ssm_dt_bias: Tensor,
    /// A parameter [d_inner, d_state], stored as [d_state, d_inner] in GGUF
    pub ssm_a: Tensor,
    /// D skip connection [d_inner]
    pub ssm_d: Option<Tensor>,
    /// Optional output norm (Mamba typically doesn't use this)
    pub ssm_norm: Option<RMSNorm>,
    /// Output projection [d_inner, hidden]
    pub ssm_out: Linear,
    pub config: MambaConfig,
}

impl std::fmt::Debug for MambaLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MambaLayer")
            .field("config", &self.config)
            .finish()
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

impl MambaLayer {
    /// Forward pass for a single token (autoregressive decode).
    ///
    /// x: [hidden_size]
    /// state: mutable MambaState for this layer
    /// Returns: [hidden_size]
    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut MambaState,
        backend: &dyn Backend,
    ) -> ModelResult<Tensor> {
        let cfg = &self.config;
        let hidden_size = x.shape().last().copied().unwrap_or(0);
        let d_inner = cfg.d_inner;
        let d_state = cfg.d_state;
        let dt_rank = cfg.dt_rank;

        // 1. Input projection: [hidden] -> [2*d_inner]
        let mut in_proj = Tensor::zeros(vec![2 * d_inner], DType::F32);
        self.ssm_in.forward(x, &mut in_proj, backend)?;

        let in_data = in_proj.as_f32()?;
        let (x_ssm, z) = (
            in_data[0..d_inner].to_vec(),
            in_data[d_inner..2 * d_inner].to_vec(),
        );

        // 2. Causal 1D convolution
        let x_ssm = self.apply_conv1d(&x_ssm, state)?;

        // 3. SiLU activation
        let x_ssm: Vec<f32> = x_ssm.iter().map(|&v| silu(v)).collect();

        // 4. x_db = x_ssm @ ssm_x -> [dt_rank + 2*d_state]
        let x_db = self.compute_x_db(&x_ssm, backend)?;

        // Split x_db into dt_raw [dt_rank], B [d_state], C [d_state]
        let dt_raw: Vec<f32> = x_db[0..dt_rank].to_vec();
        let b: Vec<f32> = x_db[dt_rank..dt_rank + d_state].to_vec();
        let c: Vec<f32> = x_db[dt_rank + d_state..dt_rank + 2 * d_state].to_vec();

        // dt = softplus(dt_raw @ ssm_dt + ssm_dt_bias) [d_inner]
        let dt = self.compute_dt(&dt_raw, backend)?;

        // 5. Selective scan
        let a_data = self.ssm_a.as_f32()?;
        let d_data = self
            .ssm_d
            .as_ref()
            .map(|t| t.as_f32().unwrap())
            .unwrap_or(&[]);

        let mut y = vec![0.0f32; d_inner];
        for d in 0..d_inner {
            let mut out_d = 0.0f32;
            for n in 0..d_state {
                // A[d, n]: ssm_a is [d_state, d_inner], so A[d,n] = a_data[n * d_inner + d]
                let a_dn = a_data[n * d_inner + d];
                let decay = (a_dn * dt[d]).exp().min(1e10);
                let idx = d * d_state + n;
                state.ssm_state[idx] =
                    decay * state.ssm_state[idx] + b[n] * dt[d] * x_ssm[d];
                out_d += c[n] * state.ssm_state[idx];
            }
            // Skip connection D
            if !d_data.is_empty() {
                out_d += d_data[d] * x_ssm[d];
            }
            y[d] = out_d;
        }

        // 6. Gate: y = y * silu(z)
        for d in 0..d_inner {
            y[d] *= silu(z[d]);
        }

        // 7. Output projection
        let y_tensor = Tensor::from_f32(&y, vec![d_inner])?;
        let mut result = Tensor::zeros(vec![hidden_size], DType::F32);
        self.ssm_out.forward(&y_tensor, &mut result, backend)?;

        Ok(result)
    }

    fn apply_conv1d(
        &self,
        x: &[f32],
        state: &mut MambaState,
    ) -> ModelResult<Vec<f32>> {
        let cfg = &self.config;
        let d_inner = cfg.d_inner;
        let ks = cfg.conv_kernel;
        let buf_len = ks - 1;

        let conv_w = self.ssm_conv1d_weight.as_f32()?;
        let conv_b = self
            .ssm_conv1d_bias
            .as_ref()
            .map(|t| t.as_f32().unwrap())
            .unwrap_or(&[]);

        // Depthwise 1D causal convolution
        // weight layout: [kernel_size, d_inner] -> conv_w[ki * d_inner + ch]
        let mut out = vec![0.0f32; d_inner];
        for ch in 0..d_inner {
            let mut sum = 0.0f32;
            for ki in 0..buf_len {
                sum += state.conv_state[ki * d_inner + ch] * conv_w[ki * d_inner + ch];
            }
            sum += x[ch] * conv_w[(ks - 1) * d_inner + ch];
            out[ch] = sum + conv_b.get(ch).copied().unwrap_or(0.0);
        }

        // Update state
        if buf_len > 1 {
            state
                .conv_state
                .copy_within(d_inner..buf_len * d_inner, 0);
        }
        let last_start = (buf_len - 1) * d_inner;
        state.conv_state[last_start..last_start + d_inner].copy_from_slice(x);

        Ok(out)
    }

    fn compute_x_db(&self, x_ssm: &[f32], backend: &dyn Backend) -> ModelResult<Vec<f32>> {
        let x_t = Tensor::from_f32(x_ssm, vec![self.config.d_inner])?;
        let mut out = Tensor::zeros(
            vec![self.config.dt_rank + 2 * self.config.d_state],
            DType::F32,
        );
        self.ssm_x.forward(&x_t, &mut out, backend)?;
        Ok(out.as_f32()?.to_vec())
    }

    fn compute_dt(&self, dt_raw: &[f32], backend: &dyn Backend) -> ModelResult<Vec<f32>> {
        let dt_raw_t = Tensor::from_f32(dt_raw, vec![self.config.dt_rank])?;
        let mut dt_proj = Tensor::zeros(vec![self.config.d_inner], DType::F32);
        self.ssm_dt.forward(&dt_raw_t, &mut dt_proj, backend)?;

        let dt_proj_data = dt_proj.as_f32()?;
        let bias_data = self.ssm_dt_bias.as_f32()?;
        let dt: Vec<f32> = dt_proj_data
            .iter()
            .zip(bias_data.iter())
            .map(|(&p, &b)| softplus(p + b))
            .collect();
        Ok(dt)
    }
}
