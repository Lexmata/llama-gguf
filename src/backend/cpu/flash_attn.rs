//! Flash Attention implementation for CPU
//!
//! This module implements Flash Attention, a memory-efficient attention algorithm
//! that uses tiling to reduce memory usage from O(n²) to O(n).
//!
//! Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
//! https://arxiv.org/abs/2205.14135

use crate::backend::{BackendError, BackendResult};
use crate::tensor::{DType, Tensor};
use rayon::prelude::*;

/// Block size for Flash Attention tiling
/// Smaller blocks use less memory but have more overhead
const BLOCK_SIZE: usize = 64;

/// Flash Attention configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for query tiling
    pub block_q: usize,
    /// Block size for key/value tiling
    pub block_kv: usize,
    /// Whether to use causal masking
    pub causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_q: BLOCK_SIZE,
            block_kv: BLOCK_SIZE,
            causal: true,
        }
    }
}

/// Compute Flash Attention
///
/// This implementation uses the tiled algorithm from the Flash Attention paper:
/// 1. Split Q into blocks of size Bq
/// 2. For each Q block, iterate over K,V blocks of size Bkv
/// 3. Use online softmax to accumulate attention weights
/// 4. Never materialize the full n×n attention matrix
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
    scale: f32,
    causal: bool,
) -> BackendResult<()> {
    // Validate shapes - expect [num_heads, seq_len, head_dim] or [batch, num_heads, seq_len, head_dim]
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() < 3 || k_shape.len() < 3 || v_shape.len() < 3 {
        return Err(BackendError::InvalidArgument(
            "Flash attention requires at least 3D tensors".into(),
        ));
    }

    // Handle both 3D [num_heads, seq_len, head_dim] and 4D [batch, num_heads, seq_len, head_dim]
    let (num_heads, seq_len_q, head_dim) = if q_shape.len() == 3 {
        (q_shape[0], q_shape[1], q_shape[2])
    } else {
        (q_shape[1], q_shape[2], q_shape[3])
    };

    let seq_len_kv = if k_shape.len() == 3 {
        k_shape[1]
    } else {
        k_shape[2]
    };

    let num_kv_heads = if k_shape.len() == 3 {
        k_shape[0]
    } else {
        k_shape[1]
    };

    // Check dtypes
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        return Err(BackendError::InvalidArgument(
            "Flash attention requires F32 tensors".into(),
        ));
    }

    let q_data = q.as_f32()?;
    let k_data = k.as_f32()?;
    let v_data = v.as_f32()?;
    let out_data = out.as_f32_mut()?;

    // Compute number of heads per group for GQA
    let heads_per_group = num_heads / num_kv_heads;

    // Process each head in parallel and collect results
    let results: Vec<(usize, Vec<f32>)> = (0..num_heads)
        .into_par_iter()
        .map(|head| {
            let kv_head = head / heads_per_group;

            // Offsets into data arrays
            let q_head_offset = head * seq_len_q * head_dim;
            let k_head_offset = kv_head * seq_len_kv * head_dim;
            let v_head_offset = kv_head * seq_len_kv * head_dim;

            let mut head_output = vec![0.0f32; seq_len_q * head_dim];

            // Process each query position using Flash Attention algorithm
            for q_pos in 0..seq_len_q {
                let q_offset = q_head_offset + q_pos * head_dim;

                // Online softmax accumulators
                let mut max_score = f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;
                let mut output = vec![0.0f32; head_dim];

                // Determine KV range based on causal masking
                let kv_end = if causal {
                    (q_pos + 1).min(seq_len_kv)
                } else {
                    seq_len_kv
                };

                // Process KV in blocks
                let block_size = BLOCK_SIZE.min(kv_end.max(1));
                let num_blocks = kv_end.div_ceil(block_size);

                for block_idx in 0..num_blocks {
                    let block_start = block_idx * block_size;
                    let block_end = (block_start + block_size).min(kv_end);

                    // Compute attention scores for this block
                    for kv_pos in block_start..block_end {
                        let k_offset = k_head_offset + kv_pos * head_dim;

                        // Compute Q·K score
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_data[q_offset + d] * k_data[k_offset + d];
                        }
                        score *= scale;

                        // Online softmax update
                        let v_offset = v_head_offset + kv_pos * head_dim;

                        if score > max_score {
                            // New maximum - rescale previous accumulations
                            let rescale = (max_score - score).exp();
                            for out_val in output.iter_mut().take(head_dim) {
                                *out_val *= rescale;
                            }
                            sum_exp *= rescale;
                            max_score = score;
                        }

                        // Add contribution from this KV position
                        let exp_score = (score - max_score).exp();
                        sum_exp += exp_score;

                        for d in 0..head_dim {
                            output[d] += exp_score * v_data[v_offset + d];
                        }
                    }
                }

                // Normalize output by softmax denominator
                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                let out_pos_offset = q_pos * head_dim;
                for d in 0..head_dim {
                    head_output[out_pos_offset + d] = output[d] * inv_sum;
                }
            }

            (head, head_output)
        })
        .collect();

    // Copy results to output tensor
    for (head, head_output) in results {
        let out_head_offset = head * seq_len_q * head_dim;
        for (i, &val) in head_output.iter().enumerate() {
            out_data[out_head_offset + i] = val;
        }
    }

    Ok(())
}

/// Flash Attention with explicit block sizes
pub fn flash_attention_blocked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
    scale: f32,
    config: &FlashAttentionConfig,
) -> BackendResult<()> {
    // For now, delegate to the standard implementation
    // A full implementation would use the specified block sizes
    flash_attention(q, k, v, out, scale, config.causal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_basic() {
        // Create small tensors for testing
        // Shape: [num_heads=2, seq_len=4, head_dim=8]
        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let v_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();

        let q = Tensor::from_f32(&q_data, vec![2, 4, 8]).unwrap();
        let k = Tensor::from_f32(&k_data, vec![2, 4, 8]).unwrap();
        let v = Tensor::from_f32(&v_data, vec![2, 4, 8]).unwrap();
        let mut out = Tensor::zeros(vec![2, 4, 8], DType::F32);

        let scale = 1.0 / (8.0f32).sqrt();
        let result = flash_attention(&q, &k, &v, &mut out, scale, true);

        assert!(result.is_ok());

        // Check output is not all zeros
        let out_data = out.as_f32().unwrap();
        let sum: f32 = out_data.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_flash_attention_causal() {
        // Test that causal masking works
        // Position 0 should only attend to position 0
        let q_data = vec![1.0f32; 8];
        let k_data = vec![1.0f32; 24]; // 3 positions
        let v_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // v[0]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // v[1]
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // v[2]
        ];

        let q = Tensor::from_f32(&q_data, vec![1, 1, 8]).unwrap();
        let k = Tensor::from_f32(&k_data, vec![1, 3, 8]).unwrap();
        let v = Tensor::from_f32(&v_data, vec![1, 3, 8]).unwrap();
        let mut out = Tensor::zeros(vec![1, 1, 8], DType::F32);

        let scale = 1.0 / (8.0f32).sqrt();
        flash_attention(&q, &k, &v, &mut out, scale, true).unwrap();

        // With causal masking at position 0, output should be v[0]
        let out_data = out.as_f32().unwrap();
        assert!((out_data[0] - 1.0).abs() < 0.01);
        assert!((out_data[1]).abs() < 0.01);
    }

    #[test]
    fn test_flash_attention_non_causal() {
        // Test non-causal attention
        let q_data = vec![1.0f32; 8];
        let k_data = vec![1.0f32; 16]; // 2 positions
        let v_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // v[0]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // v[1]
        ];

        let q = Tensor::from_f32(&q_data, vec![1, 1, 8]).unwrap();
        let k = Tensor::from_f32(&k_data, vec![1, 2, 8]).unwrap();
        let v = Tensor::from_f32(&v_data, vec![1, 2, 8]).unwrap();
        let mut out = Tensor::zeros(vec![1, 1, 8], DType::F32);

        let scale = 1.0 / (8.0f32).sqrt();
        flash_attention(&q, &k, &v, &mut out, scale, false).unwrap();

        // With non-causal attention, output should be average of v[0] and v[1]
        // (since K values are equal, attention weights should be equal)
        let out_data = out.as_f32().unwrap();
        assert!((out_data[0] - 0.5).abs() < 0.01);
        assert!((out_data[1] - 0.5).abs() < 0.01);
    }
}
