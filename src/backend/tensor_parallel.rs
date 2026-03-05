//! Tensor parallelism for multi-GPU inference
//!
//! Splits model layers across multiple GPUs on a single node.
//! Attention heads are split across GPUs (each GPU handles a subset of heads).
//! FFN matrices are split column-wise (gate/up) or row-wise (down).
//! All-reduce is used to synchronize after each layer.

use crate::backend::{BackendError, BackendResult};
use crate::tensor::{compute_strides, Tensor};
pub use crate::tensor::DType;

/// Trait for tensor parallel communication primitives
pub trait TensorParallel: Send + Sync {
    /// Number of devices (world size)
    fn world_size(&self) -> usize;

    /// This device's rank (0-indexed)
    fn rank(&self) -> usize;

    /// All-reduce sum: sum tensor across all devices in-place
    fn all_reduce_sum(&self, tensor: &mut Tensor) -> BackendResult<()>;

    /// All-gather: gather local tensors from all devices into output
    /// output is world_size * local_size
    fn all_gather(&self, local: &Tensor, output: &mut Tensor) -> BackendResult<()>;

    /// Scatter: split input across devices, each gets 1/world_size
    fn scatter(&self, input: &Tensor, output: &mut Tensor) -> BackendResult<()>;

    /// Barrier: synchronize all devices
    fn barrier(&self) -> BackendResult<()>;
}

/// Tensor parallelism configuration
#[derive(Debug, Clone)]
pub struct TPConfig {
    /// Number of GPUs to use
    pub num_devices: usize,
    /// Device IDs (e.g., [0, 1] for 2 GPUs)
    pub device_ids: Vec<usize>,
}

impl Default for TPConfig {
    fn default() -> Self {
        Self {
            num_devices: 1,
            device_ids: vec![0],
        }
    }
}

/// How a model is sharded across devices
#[derive(Debug, Clone)]
pub struct ShardingPlan {
    /// Number of attention heads per device
    pub heads_per_device: usize,
    /// Number of KV heads per device
    pub kv_heads_per_device: usize,
    /// FFN intermediate size per device
    pub ffn_dim_per_device: usize,
    /// Total number of heads
    pub total_heads: usize,
    /// Total number of KV heads
    pub total_kv_heads: usize,
    /// Total FFN intermediate size
    pub total_ffn_dim: usize,
}

impl ShardingPlan {
    /// Create a sharding plan from model config
    pub fn from_config(
        num_heads: usize,
        num_kv_heads: usize,
        ffn_dim: usize,
        world_size: usize,
    ) -> Result<Self, String> {
        // Validate divisibility
        if num_heads % world_size != 0 {
            return Err(format!(
                "num_heads ({}) must be divisible by world_size ({})",
                num_heads, world_size
            ));
        }
        if num_kv_heads % world_size != 0 {
            return Err(format!(
                "num_kv_heads ({}) must be divisible by world_size ({})",
                num_kv_heads, world_size
            ));
        }
        if ffn_dim % world_size != 0 {
            return Err(format!(
                "ffn_dim ({}) must be divisible by world_size ({})",
                ffn_dim, world_size
            ));
        }

        Ok(Self {
            heads_per_device: num_heads / world_size,
            kv_heads_per_device: num_kv_heads / world_size,
            ffn_dim_per_device: ffn_dim / world_size,
            total_heads: num_heads,
            total_kv_heads: num_kv_heads,
            total_ffn_dim: ffn_dim,
        })
    }
}

/// Split a weight tensor along a dimension for tensor parallelism
///
/// For a [out, in] matrix split along dim=0 with world_size=2:
/// - rank 0 gets [0..out/2, :]
/// - rank 1 gets [out/2..out, :]
///
/// Supports F32, F16, BF16, F64. Quantized types are not supported.
pub fn shard_weight(
    weight: &Tensor,
    dim: usize,
    rank: usize,
    world_size: usize,
) -> Result<Tensor, BackendError> {
    let shape = weight.shape();
    if dim >= shape.len() {
        return Err(BackendError::InvalidArgument(format!(
            "dim {} out of range for shape {:?}",
            dim, shape
        )));
    }
    if rank >= world_size {
        return Err(BackendError::InvalidArgument(format!(
            "rank {} must be < world_size {}",
            rank, world_size
        )));
    }
    let dim_size = shape[dim];
    if dim_size % world_size != 0 {
        return Err(BackendError::InvalidArgument(format!(
            "shape[{}] ({}) must be divisible by world_size ({})",
            dim, dim_size, world_size
        )));
    }
    if !weight.is_contiguous() {
        return Err(BackendError::InvalidArgument(
            "weight must be contiguous for sharding".into(),
        ));
    }
    if weight.dtype().is_quantized() {
        return Err(BackendError::Unsupported(
            "shard_weight does not support quantized tensors".into(),
        ));
    }

    let chunk_size = dim_size / world_size;
    let start_idx = rank * chunk_size;

    // Build output shape: replace shape[dim] with chunk_size
    let mut out_shape = shape.to_vec();
    out_shape[dim] = chunk_size;

    let out_numel: usize = out_shape.iter().product();
    let elem_size = weight.dtype().size_for_elements(1);
    let out_bytes = weight.dtype().size_for_elements(out_numel);
    let mut out_data = vec![0u8; out_bytes];

    let in_strides = weight.strides();
    let in_data = weight.data();

    // Iterate over all output indices and copy from input
    for out_linear in 0..out_numel {
        // Decode output linear index to multi-index
        let mut out_idx = vec![0; out_shape.len()];
        let mut rem = out_linear;
        for d in (0..out_shape.len()).rev() {
            out_idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        // Map to input index (offset in split dimension)
        let mut in_idx = out_idx.clone();
        in_idx[dim] += start_idx;
        // Compute input linear index
        let in_linear: usize = in_idx.iter().zip(in_strides.iter()).map(|(i, s)| i * s).sum();
        let src_off = in_linear * elem_size;
        let dst_off = out_linear * elem_size;
        out_data[dst_off..dst_off + elem_size]
            .copy_from_slice(&in_data[src_off..src_off + elem_size]);
    }

    Tensor::new(out_data, out_shape, weight.dtype())
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

/// Merge sharded weight tensors back (inverse of shard_weight)
///
/// Concatenates shards along the given dimension.
/// Supports F32, F16, BF16, F64. Quantized types are not supported.
pub fn merge_shards(shards: &[Tensor], dim: usize) -> Result<Tensor, BackendError> {
    if shards.is_empty() {
        return Err(BackendError::InvalidArgument(
            "merge_shards requires at least one shard".into(),
        ));
    }
    let dtype = shards[0].dtype();
    if dtype.is_quantized() {
        return Err(BackendError::Unsupported(
            "merge_shards does not support quantized tensors".into(),
        ));
    }
    for s in shards {
        if s.dtype() != dtype {
            return Err(BackendError::InvalidArgument(
                "all shards must have the same dtype".into(),
            ));
        }
        if !s.is_contiguous() {
            return Err(BackendError::InvalidArgument(
                "all shards must be contiguous".into(),
            ));
        }
    }

    let first_shape = shards[0].shape();
    if dim >= first_shape.len() {
        return Err(BackendError::InvalidArgument(format!(
            "dim {} out of range for shape {:?}",
            dim, first_shape
        )));
    }

    // Build merged shape: sum shards along dim
    let mut merged_shape = first_shape.to_vec();
    merged_shape[dim] = 0;
    for s in shards {
        let s_shape = s.shape();
        if s_shape.len() != merged_shape.len() {
            return Err(BackendError::InvalidArgument(
                "all shards must have same number of dimensions".into(),
            ));
        }
        for (i, (m, &ss)) in merged_shape.iter_mut().zip(s_shape.iter()).enumerate() {
            if i == dim {
                *m += ss;
            } else if *m != ss {
                return Err(BackendError::InvalidArgument(format!(
                    "shard shape mismatch at dim {}: expected {}, got {}",
                    i, m, ss
                )));
            }
        }
    }

    let merged_numel: usize = merged_shape.iter().product();
    let elem_size = dtype.size_for_elements(1);
    let merged_bytes = dtype.size_for_elements(merged_numel);
    let mut merged_data = vec![0u8; merged_bytes];
    let merged_strides = compute_strides(&merged_shape);

    let mut offset_along_dim = 0;
    for shard in shards {
        let shard_shape = shard.shape();
        let shard_size = shard_shape[dim];
        let shard_numel: usize = shard_shape.iter().product();
        let shard_data = shard.data();

        for shard_linear in 0..shard_numel {
            let mut shard_idx = vec![0; shard_shape.len()];
            let mut rem = shard_linear;
            for d in (0..shard_shape.len()).rev() {
                shard_idx[d] = rem % shard_shape[d];
                rem /= shard_shape[d];
            }
            let mut merged_idx = shard_idx.clone();
            merged_idx[dim] += offset_along_dim;
            let merged_linear: usize = merged_idx
                .iter()
                .zip(merged_strides.iter())
                .map(|(i, s)| i * s)
                .sum();
            let src_off = shard_linear * elem_size;
            let dst_off = merged_linear * elem_size;
            merged_data[dst_off..dst_off + elem_size]
                .copy_from_slice(&shard_data[src_off..src_off + elem_size]);
        }
        offset_along_dim += shard_size;
    }

    Tensor::new(merged_data, merged_shape, dtype)
        .map_err(|e| BackendError::OperationFailed(format!("{}", e)))
}

/// No-op tensor parallel for single device (world_size=1)
pub struct SingleDeviceTP;

impl TensorParallel for SingleDeviceTP {
    fn world_size(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        0
    }
    fn all_reduce_sum(&self, _tensor: &mut Tensor) -> BackendResult<()> {
        Ok(())
    }
    fn all_gather(&self, local: &Tensor, output: &mut Tensor) -> BackendResult<()> {
        let local_data = local.data();
        let out_data = output
            .data_mut()
            .ok_or_else(|| BackendError::InvalidArgument("output must be mutable".into()))?;
        let copy_len = local_data.len().min(out_data.len());
        out_data[..copy_len].copy_from_slice(&local_data[..copy_len]);
        Ok(())
    }
    fn scatter(&self, input: &Tensor, output: &mut Tensor) -> BackendResult<()> {
        let input_data = input.data();
        let out_data = output
            .data_mut()
            .ok_or_else(|| BackendError::InvalidArgument("output must be mutable".into()))?;
        let copy_len = input_data.len().min(out_data.len());
        out_data[..copy_len].copy_from_slice(&input_data[..copy_len]);
        Ok(())
    }
    fn barrier(&self) -> BackendResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharding_plan_valid() {
        let plan = ShardingPlan::from_config(32, 8, 11008, 2).unwrap();
        assert_eq!(plan.heads_per_device, 16);
        assert_eq!(plan.kv_heads_per_device, 4);
        assert_eq!(plan.ffn_dim_per_device, 5504);
        assert_eq!(plan.total_heads, 32);
        assert_eq!(plan.total_kv_heads, 8);
        assert_eq!(plan.total_ffn_dim, 11008);
    }

    #[test]
    fn test_sharding_plan_invalid() {
        // num_heads not divisible
        assert!(ShardingPlan::from_config(31, 8, 11008, 2).is_err());
        // num_kv_heads not divisible
        assert!(ShardingPlan::from_config(32, 7, 11008, 2).is_err());
        // ffn_dim not divisible
        assert!(ShardingPlan::from_config(32, 8, 11007, 2).is_err());
    }

    #[test]
    fn test_shard_weight() {
        // [8, 4] tensor, split along dim=0 into 2 shards of [4, 4]
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let weight = Tensor::from_f32(&data, vec![8, 4]).unwrap();

        let shard0 = shard_weight(&weight, 0, 0, 2).unwrap();
        assert_eq!(shard0.shape(), &[4, 4]);
        let s0 = shard0.as_f32().unwrap();
        assert_eq!(s0, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        let shard1 = shard_weight(&weight, 0, 1, 2).unwrap();
        assert_eq!(shard1.shape(), &[4, 4]);
        let s1 = shard1.as_f32().unwrap();
        assert_eq!(s1, &[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]);
    }

    #[test]
    fn test_merge_shards() {
        let shard0 = Tensor::from_f32(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            vec![4, 4],
        )
        .unwrap();
        let shard1 = Tensor::from_f32(
            &[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
            vec![4, 4],
        )
        .unwrap();

        let merged = merge_shards(&[shard0, shard1], 0).unwrap();
        assert_eq!(merged.shape(), &[8, 4]);
        let m = merged.as_f32().unwrap();
        let expected: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(m, expected.as_slice());
    }

    #[test]
    fn test_single_device_tp() {
        let tp = SingleDeviceTP;
        assert_eq!(tp.world_size(), 1);
        assert_eq!(tp.rank(), 0);

        let mut tensor = Tensor::from_f32(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        tp.all_reduce_sum(&mut tensor).unwrap();
        assert_eq!(tensor.as_f32().unwrap(), &[1.0, 2.0, 3.0]);

        let local = Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap();
        let mut output = Tensor::zeros(vec![2], DType::F32);
        tp.all_gather(&local, &mut output).unwrap();
        assert_eq!(output.as_f32().unwrap(), &[1.0, 2.0]);

        let input = Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap();
        let mut out = Tensor::zeros(vec![2], DType::F32);
        tp.scatter(&input, &mut out).unwrap();
        assert_eq!(out.as_f32().unwrap(), &[1.0, 2.0]);

        tp.barrier().unwrap();
    }
}
