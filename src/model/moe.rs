//! Mixture of Experts (MoE) support
//!
//! MoE models use a gating/router network to select which expert FFN layers
//! to use for each token. This enables sparse computation where only a subset
//! of experts are activated, allowing for larger model capacity without
//! proportional compute increase.
//!
//! Supports architectures like:
//! - Mixtral (top-2 routing)
//! - Qwen MoE
//! - DeepSeek MoE

use crate::backend::Backend;
use crate::tensor::{DType, Tensor};

/// MoE configuration
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to route to per token (top-k)
    pub num_experts_per_token: usize,
    /// Hidden dimension of experts
    pub expert_hidden_dim: usize,
    /// Whether to use shared experts (in addition to routed experts)
    pub num_shared_experts: usize,
    /// Auxiliary load balancing loss coefficient
    pub aux_loss_coef: f32,
    /// Whether to normalize router logits
    pub normalize_router_logits: bool,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_token: 2,
            expert_hidden_dim: 14336,
            num_shared_experts: 0,
            aux_loss_coef: 0.01,
            normalize_router_logits: true,
        }
    }
}

impl MoeConfig {
    /// Create a Mixtral-style config (8 experts, top-2 routing)
    pub fn mixtral() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_token: 2,
            expert_hidden_dim: 14336,
            num_shared_experts: 0,
            aux_loss_coef: 0.01,
            normalize_router_logits: true,
        }
    }

    /// Create a DeepSeek-style config with shared experts
    pub fn deepseek(num_experts: usize, num_shared: usize) -> Self {
        Self {
            num_experts,
            num_experts_per_token: 2,
            expert_hidden_dim: 11008,
            num_shared_experts: num_shared,
            aux_loss_coef: 0.01,
            normalize_router_logits: true,
        }
    }
}

/// Expert selection result
#[derive(Debug, Clone)]
pub struct ExpertSelection {
    /// Selected expert indices for each token [batch_size, num_experts_per_token]
    pub indices: Vec<Vec<usize>>,
    /// Routing weights for each selected expert [batch_size, num_experts_per_token]
    pub weights: Vec<Vec<f32>>,
}

/// Router/Gating network for MoE
#[derive(Debug)]
pub struct MoeRouter {
    /// Router weights [num_experts, hidden_dim]
    pub weight: Tensor,
    /// Number of experts
    num_experts: usize,
    /// Number of experts to select per token
    top_k: usize,
    /// Whether to normalize logits
    normalize: bool,
}

impl MoeRouter {
    /// Create a new router
    pub fn new(hidden_dim: usize, num_experts: usize, top_k: usize, normalize: bool) -> Self {
        let weight = Tensor::zeros(vec![num_experts, hidden_dim], DType::F32);
        Self {
            weight,
            num_experts,
            top_k,
            normalize,
        }
    }

    /// Create from existing weight tensor
    pub fn from_weight(weight: Tensor, top_k: usize, normalize: bool) -> Self {
        let num_experts = weight.shape()[0];
        Self {
            weight,
            num_experts,
            top_k,
            normalize,
        }
    }

    /// Route tokens to experts
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch_size, hidden_dim] or [hidden_dim]
    ///
    /// # Returns
    /// Expert selection with indices and weights
    pub fn route(
        &self,
        hidden_states: &Tensor,
    ) -> Result<ExpertSelection, crate::backend::BackendError> {
        let h_data = hidden_states.as_f32()?;
        let w_data = self.weight.as_f32()?;

        let hidden_dim = self.weight.shape()[1];
        let h_shape = hidden_states.shape();

        // Handle both batched and unbatched inputs
        let (batch_size, h_offset_stride) = if h_shape.len() == 1 {
            (1, 0)
        } else {
            (h_shape[0], hidden_dim)
        };

        let mut all_indices = Vec::with_capacity(batch_size);
        let mut all_weights = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Compute router logits: hidden @ weight.T
            let h_offset = b * h_offset_stride;
            let mut logits = vec![0.0f32; self.num_experts];

            for e in 0..self.num_experts {
                let mut sum = 0.0f32;
                for d in 0..hidden_dim {
                    sum += h_data[h_offset + d] * w_data[e * hidden_dim + d];
                }
                logits[e] = sum;
            }

            // Optionally normalize logits
            if self.normalize {
                let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                for l in &mut logits {
                    *l -= max_logit;
                }
            }

            // Find top-k experts
            let mut indexed_logits: Vec<(usize, f32)> =
                logits.iter().cloned().enumerate().collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_indices: Vec<usize> = indexed_logits[..self.top_k]
                .iter()
                .map(|(i, _)| *i)
                .collect();
            let top_k_logits: Vec<f32> = indexed_logits[..self.top_k]
                .iter()
                .map(|(_, l)| *l)
                .collect();

            // Softmax over top-k to get routing weights
            let max_val = top_k_logits
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = top_k_logits.iter().map(|&l| (l - max_val).exp()).sum();
            let weights: Vec<f32> = top_k_logits
                .iter()
                .map(|&l| (l - max_val).exp() / exp_sum)
                .collect();

            all_indices.push(top_k_indices);
            all_weights.push(weights);
        }

        Ok(ExpertSelection {
            indices: all_indices,
            weights: all_weights,
        })
    }
}

/// Single MoE expert (feed-forward network)
#[derive(Debug)]
pub struct MoeExpert {
    /// Gate projection [hidden_dim, intermediate_dim]
    pub gate_proj: Tensor,
    /// Up projection [hidden_dim, intermediate_dim]
    pub up_proj: Tensor,
    /// Down projection [intermediate_dim, hidden_dim]
    pub down_proj: Tensor,
}

impl MoeExpert {
    /// Create a new expert
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            gate_proj: Tensor::zeros(vec![intermediate_dim, hidden_dim], DType::F32),
            up_proj: Tensor::zeros(vec![intermediate_dim, hidden_dim], DType::F32),
            down_proj: Tensor::zeros(vec![hidden_dim, intermediate_dim], DType::F32),
        }
    }

    /// Forward pass through expert (SwiGLU activation)
    ///
    /// output = down_proj(silu(gate_proj(x)) * up_proj(x))
    pub fn forward(
        &self,
        x: &Tensor,
        backend: &dyn Backend,
    ) -> Result<Tensor, crate::backend::BackendError> {
        let hidden_dim = self.down_proj.shape()[0];
        let intermediate_dim = self.gate_proj.shape()[0];

        // Gate projection
        let mut gate_out = Tensor::zeros(vec![intermediate_dim], DType::F32);
        backend.matvec(&self.gate_proj, x, &mut gate_out)?;

        // Apply SiLU to gate
        let mut gate_silu = Tensor::zeros(vec![intermediate_dim], DType::F32);
        backend.silu(&gate_out, &mut gate_silu)?;

        // Up projection
        let mut up_out = Tensor::zeros(vec![intermediate_dim], DType::F32);
        backend.matvec(&self.up_proj, x, &mut up_out)?;

        // Element-wise multiply
        let mut intermediate = Tensor::zeros(vec![intermediate_dim], DType::F32);
        backend.mul(&gate_silu, &up_out, &mut intermediate)?;

        // Down projection
        let mut output = Tensor::zeros(vec![hidden_dim], DType::F32);
        backend.matvec(&self.down_proj, &intermediate, &mut output)?;

        Ok(output)
    }
}

/// Mixture of Experts layer
#[derive(Debug)]
pub struct MoeLayer {
    /// Configuration
    config: MoeConfig,
    /// Router/gating network
    pub router: MoeRouter,
    /// Expert networks
    pub experts: Vec<MoeExpert>,
    /// Shared experts (if any)
    pub shared_experts: Vec<MoeExpert>,
}

impl MoeLayer {
    /// Create a new MoE layer
    pub fn new(hidden_dim: usize, config: MoeConfig) -> Self {
        let router = MoeRouter::new(
            hidden_dim,
            config.num_experts,
            config.num_experts_per_token,
            config.normalize_router_logits,
        );

        let experts = (0..config.num_experts)
            .map(|_| MoeExpert::new(hidden_dim, config.expert_hidden_dim))
            .collect();

        let shared_experts = (0..config.num_shared_experts)
            .map(|_| MoeExpert::new(hidden_dim, config.expert_hidden_dim))
            .collect();

        Self {
            config,
            router,
            experts,
            shared_experts,
        }
    }

    /// Forward pass through MoE layer
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch_size, hidden_dim] or [hidden_dim]
    /// * `backend` - Backend for computation
    ///
    /// # Returns
    /// Output tensor with same shape as input
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        backend: &dyn Backend,
    ) -> Result<Tensor, crate::backend::BackendError> {
        let h_shape = hidden_states.shape();
        let hidden_dim = *h_shape.last().unwrap_or(&0);

        // Route tokens to experts
        let selection = self.router.route(hidden_states)?;

        // Process each token
        let h_data = hidden_states.as_f32()?;
        let batch_size = if h_shape.len() == 1 { 1 } else { h_shape[0] };

        let mut output_data = vec![0.0f32; batch_size * hidden_dim];

        for (b, (indices, weights)) in selection
            .indices
            .iter()
            .zip(selection.weights.iter())
            .enumerate()
        {
            // Get input for this token
            let h_offset = b * hidden_dim;
            let token_input = if h_shape.len() == 1 {
                hidden_states.clone()
            } else {
                Tensor::from_f32(&h_data[h_offset..h_offset + hidden_dim], vec![hidden_dim])?
            };

            // Compute weighted sum of expert outputs
            for (&expert_idx, &weight) in indices.iter().zip(weights.iter()) {
                let expert_output = self.experts[expert_idx].forward(&token_input, backend)?;
                let expert_data = expert_output.as_f32()?;

                for d in 0..hidden_dim {
                    output_data[b * hidden_dim + d] += weight * expert_data[d];
                }
            }

            // Add shared expert contributions (if any)
            for shared_expert in &self.shared_experts {
                let shared_output = shared_expert.forward(&token_input, backend)?;
                let shared_data = shared_output.as_f32()?;

                // Shared experts contribute equally
                let shared_weight = 1.0 / (self.shared_experts.len() as f32 + 1.0);
                for d in 0..hidden_dim {
                    output_data[b * hidden_dim + d] += shared_weight * shared_data[d];
                }
            }
        }

        if h_shape.len() == 1 {
            Ok(Tensor::from_f32(&output_data, vec![hidden_dim])?)
        } else {
            Ok(Tensor::from_f32(
                &output_data,
                vec![batch_size, hidden_dim],
            )?)
        }
    }

    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    /// Get number of experts per token
    pub fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token
    }
}

/// Statistics from MoE routing (for load balancing analysis)
#[derive(Debug, Clone, Default)]
pub struct MoeStats {
    /// Number of tokens routed
    pub total_tokens: usize,
    /// Count per expert
    pub expert_counts: Vec<usize>,
    /// Total weight per expert
    pub expert_weights: Vec<f32>,
}

impl MoeStats {
    /// Create new stats for given number of experts
    pub fn new(num_experts: usize) -> Self {
        Self {
            total_tokens: 0,
            expert_counts: vec![0; num_experts],
            expert_weights: vec![0.0; num_experts],
        }
    }

    /// Record routing decision
    pub fn record(&mut self, selection: &ExpertSelection) {
        for (indices, weights) in selection.indices.iter().zip(selection.weights.iter()) {
            self.total_tokens += 1;
            for (&idx, &weight) in indices.iter().zip(weights.iter()) {
                self.expert_counts[idx] += 1;
                self.expert_weights[idx] += weight;
            }
        }
    }

    /// Get load balance factor (1.0 = perfectly balanced)
    pub fn load_balance_factor(&self) -> f32 {
        if self.total_tokens == 0 {
            return 1.0;
        }

        let n = self.expert_counts.len() as f32;
        let ideal = self.total_tokens as f32 / n;

        let variance: f32 = self
            .expert_counts
            .iter()
            .map(|&c| (c as f32 - ideal).powi(2))
            .sum::<f32>()
            / n;

        1.0 / (1.0 + variance / ideal.powi(2))
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.expert_counts.fill(0);
        self.expert_weights.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    #[test]
    fn test_moe_config_default() {
        let config = MoeConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.num_experts_per_token, 2);
    }

    #[test]
    fn test_moe_config_mixtral() {
        let config = MoeConfig::mixtral();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.num_experts_per_token, 2);
    }

    #[test]
    fn test_moe_router() {
        let router = MoeRouter::new(64, 4, 2, true);
        let hidden = Tensor::from_f32(&vec![0.1f32; 64], vec![64]).unwrap();

        let selection = router.route(&hidden).unwrap();
        assert_eq!(selection.indices.len(), 1);
        assert_eq!(selection.indices[0].len(), 2);
        assert_eq!(selection.weights[0].len(), 2);

        // Weights should sum to ~1.0
        let weight_sum: f32 = selection.weights[0].iter().sum();
        assert!((weight_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_moe_expert() {
        let backend = CpuBackend::new();
        let expert = MoeExpert::new(64, 256);
        let input = Tensor::from_f32(&vec![0.1f32; 64], vec![64]).unwrap();

        let output = expert.forward(&input, &backend).unwrap();
        assert_eq!(output.shape(), &[64]);
    }

    #[test]
    fn test_moe_layer() {
        let backend = CpuBackend::new();
        let config = MoeConfig {
            num_experts: 4,
            num_experts_per_token: 2,
            expert_hidden_dim: 128,
            num_shared_experts: 0,
            aux_loss_coef: 0.01,
            normalize_router_logits: true,
        };

        let layer = MoeLayer::new(64, config);
        let input = Tensor::from_f32(&vec![0.1f32; 64], vec![64]).unwrap();

        let output = layer.forward(&input, &backend).unwrap();
        assert_eq!(output.shape(), &[64]);
    }

    #[test]
    fn test_moe_stats() {
        let mut stats = MoeStats::new(4);

        let selection = ExpertSelection {
            indices: vec![vec![0, 1], vec![1, 2]],
            weights: vec![vec![0.6, 0.4], vec![0.7, 0.3]],
        };

        stats.record(&selection);

        assert_eq!(stats.total_tokens, 2);
        assert_eq!(stats.expert_counts[0], 1);
        assert_eq!(stats.expert_counts[1], 2);
        assert_eq!(stats.expert_counts[2], 1);
        assert_eq!(stats.expert_counts[3], 0);
    }
}
