//! Metal compute operation implementations.
//!
//! Each operation follows the pattern:
//! 1. Create Metal buffers from tensor data (shared memory on Apple Silicon)
//! 2. Dispatch compute shader
//! 3. Read back output data
//!
//! On Apple Silicon, buffers use shared storage mode which means CPU and GPU
//! see the same memory -- no explicit copies are needed.

use metal::MTLSize;

use crate::backend::metal::context::MetalContext;
use crate::backend::{BackendError, BackendResult};
use crate::tensor::Tensor;

/// Element-wise addition: out = a + b
pub fn add(ctx: &MetalContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let n_i32 = n as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "add",
        &[&a_buf, &b_buf, &out_buf],
        Some((param_bytes, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Element-wise multiplication: out = a * b
pub fn mul(ctx: &MetalContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let n_i32 = n as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "mul",
        &[&a_buf, &b_buf, &out_buf],
        Some((param_bytes, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Scale: out = a * scalar
pub fn scale(
    ctx: &MetalContext,
    a: &Tensor,
    scalar: f32,
    out: &mut Tensor,
) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    // ScaleParams: { n: i32, scalar: f32 }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScaleParams {
        n: i32,
        scalar: f32,
    }

    let params = ScaleParams {
        n: n as i32,
        scalar,
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "scale",
        &[&a_buf, &out_buf],
        Some((param_bytes, 2)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// SiLU activation: out = x * sigmoid(x)
pub fn silu(ctx: &MetalContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let n_i32 = n as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "silu",
        &[&x_buf, &out_buf],
        Some((param_bytes, 2)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// GELU activation
pub fn gelu(ctx: &MetalContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let n_i32 = n as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "gelu",
        &[&x_buf, &out_buf],
        Some((param_bytes, 2)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Softmax along last dimension (multi-pass with partial reduction)
pub fn softmax(ctx: &MetalContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    // For softmax, use CPU if the size is small (GPU overhead dominates)
    if n < 1024 {
        return Err(BackendError::Unsupported(
            "Small softmax: use CPU".to_string(),
        ));
    }

    let x_buf = ctx.create_buffer_with_data(x_data)?;

    // Pass 1: Find max using partial reduction
    let tg_size: u64 = 256;
    let num_threadgroups = ((n as u64) + tg_size - 1) / tg_size;
    let partial_max_buf = ctx.create_output_buffer(num_threadgroups as usize)?;

    let n_i32 = n as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    ctx.dispatch_threadgroups(
        "softmax_max",
        &[&x_buf, &partial_max_buf],
        Some((param_bytes, 2)),
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    )?;

    // Read partial maxes and find global max on CPU
    let partial_maxes = ctx.read_buffer(&partial_max_buf, num_threadgroups as usize);
    let max_val = partial_maxes
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Pass 2: exp(x - max) per element
    let out_buf = ctx.create_output_buffer(n)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SoftmaxExpParams {
        n: i32,
        max_value: f32,
    }

    let exp_params = SoftmaxExpParams {
        n: n as i32,
        max_value: max_val,
    };
    let exp_param_bytes = bytemuck::bytes_of(&exp_params);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "softmax_exp",
        &[&x_buf, &out_buf],
        Some((exp_param_bytes, 2)),
        grid,
        tg,
    )?;

    // Read exp values and compute sum on CPU
    let exp_values = ctx.read_buffer(&out_buf, n);
    let sum_val: f32 = exp_values.iter().sum();

    // Write normalized values
    let inv_sum = 1.0 / sum_val;
    for (i, v) in exp_values.iter().enumerate() {
        out_data[i] = v * inv_sum;
    }

    Ok(())
}

/// RMS normalization: out = x / rms(x) * weight (2-pass with partial reduction)
pub fn rms_norm(
    ctx: &MetalContext,
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    out: &mut Tensor,
) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let w_data = weight.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let w_buf = ctx.create_buffer_with_data(w_data)?;

    // Pass 1: Partial sum of squares
    let tg_size: u64 = 256;
    let num_threadgroups = ((n as u64) + tg_size - 1) / tg_size;
    let partial_sum_buf = ctx.create_output_buffer(num_threadgroups as usize)?;

    let n_i32 = n as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    ctx.dispatch_threadgroups(
        "rms_norm_sum",
        &[&x_buf, &partial_sum_buf],
        Some((param_bytes, 2)),
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    )?;

    // Sum partial results on CPU
    let partial_sums = ctx.read_buffer(&partial_sum_buf, num_threadgroups as usize);
    let sum_sq: f32 = partial_sums.iter().sum();

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    // Pass 2: Normalize and scale on GPU
    let out_buf = ctx.create_output_buffer(n)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RmsNormScaleParams {
        n: i32,
        rms_inv: f32,
    }

    let scale_params = RmsNormScaleParams {
        n: n as i32,
        rms_inv,
    };
    let scale_param_bytes = bytemuck::bytes_of(&scale_params);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "rms_norm_scale",
        &[&x_buf, &w_buf, &out_buf],
        Some((scale_param_bytes, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Vector-matrix multiply: out = a @ b where a is [k], b is [k, n] (GGUF column-major), out is [n]
pub fn vec_mat(
    ctx: &MetalContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let k = b.shape()[0];
    let n = b.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct VecMatParams {
        k: i32,
        n: i32,
    }

    let params = VecMatParams {
        k: k as i32,
        n: n as i32,
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let (grid, tg) = MetalContext::sizes_1d(n);
    ctx.dispatch(
        "vec_mat",
        &[&a_buf, &b_buf, &out_buf],
        Some((param_bytes, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Rotary Position Embedding (RoPE)
pub fn rope(
    ctx: &MetalContext,
    q: &mut Tensor,
    k: &mut Tensor,
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
    use_neox: bool,
) -> BackendResult<()> {
    let q_shape = q.shape().to_vec();
    let k_shape = k.shape().to_vec();

    // Handle 2D [num_heads, head_dim] or 3D [num_heads, 1, head_dim]
    let (num_q_heads, head_dim) = if q_shape.len() == 2 {
        (q_shape[0], q_shape[1])
    } else if q_shape.len() == 3 && q_shape[1] == 1 {
        (q_shape[0], q_shape[2])
    } else {
        return Err(BackendError::Unsupported(format!(
            "RoPE unsupported shape: {:?}",
            q_shape
        )));
    };

    let num_k_heads = if k_shape.len() == 2 {
        k_shape[0]
    } else if k_shape.len() == 3 && k_shape[1] == 1 {
        k_shape[0]
    } else {
        return Err(BackendError::Unsupported(format!(
            "RoPE unsupported k shape: {:?}",
            k_shape
        )));
    };

    let q_data = q.as_f32_mut()?;
    let k_data = k.as_f32_mut()?;

    // Upload q and k as read-write buffers (RoPE modifies in place)
    let q_buf = ctx.create_buffer_with_data(q_data)?;
    let k_buf = ctx.create_buffer_with_data(k_data)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RopeParams {
        num_q_heads: i32,
        num_k_heads: i32,
        head_dim: i32,
        position: i32,
        freq_base: f32,
        freq_scale: f32,
        use_neox: i32,
    }

    let params = RopeParams {
        num_q_heads: num_q_heads as i32,
        num_k_heads: num_k_heads as i32,
        head_dim: head_dim as i32,
        position: pos as i32,
        freq_base,
        freq_scale,
        use_neox: if use_neox { 1 } else { 0 },
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let max_heads = num_q_heads.max(num_k_heads);

    // Dispatch: one threadgroup per head, threads per threadgroup = head_dim/2
    let threadgroup_count = MTLSize::new(max_heads as u64, 1, 1);
    let threadgroup_size = MTLSize::new(64.min(head_dim as u64 / 2), 1, 1);

    ctx.dispatch_threadgroups(
        "rope",
        &[&q_buf, &k_buf],
        Some((param_bytes, 2)),
        threadgroup_count,
        threadgroup_size,
    )?;

    // Read back modified q and k
    let q_result = ctx.read_buffer(&q_buf, q_data.len());
    let k_result = ctx.read_buffer(&k_buf, k_data.len());

    q_data.copy_from_slice(&q_result);
    k_data.copy_from_slice(&k_result);

    Ok(())
}
