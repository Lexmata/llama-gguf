//! Vulkan compute operation implementations.
//!
//! Each operation follows the pattern:
//! 1. Upload input data to GPU buffers
//! 2. Dispatch compute shader
//! 3. Read back output data
//! 4. Free temporary buffers

use crate::backend::vulkan::context::VulkanContext;
use crate::backend::{BackendError, BackendResult};
use crate::tensor::Tensor;

/// Element-wise addition: out = a + b
pub fn add(ctx: &VulkanContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let push = (n as i32).to_le_bytes();
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    ctx.dispatch("add", &[&a_buf, &b_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(b_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Element-wise multiplication: out = a * b
pub fn mul(ctx: &VulkanContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let push = (n as i32).to_le_bytes();
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    ctx.dispatch("mul", &[&a_buf, &b_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(b_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Scale: out = a * scalar
pub fn scale(
    ctx: &VulkanContext,
    a: &Tensor,
    scalar: f32,
    out: &mut Tensor,
) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    // Push constants: [n: i32, scalar: f32]
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&(n as i32).to_le_bytes());
    push.extend_from_slice(&scalar.to_le_bytes());

    let wg = VulkanContext::workgroup_count_1d(n, 256);
    ctx.dispatch("scale", &[&a_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// SiLU activation: out = x * sigmoid(x)
pub fn silu(ctx: &VulkanContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let push = (n as i32).to_le_bytes();
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    ctx.dispatch("silu", &[&x_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(x_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// GELU activation
pub fn gelu(ctx: &VulkanContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let push = (n as i32).to_le_bytes();
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    ctx.dispatch("gelu", &[&x_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(x_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Softmax along last dimension (3-pass implementation with partial reduction)
pub fn softmax(ctx: &VulkanContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    // For softmax, we do it on CPU if the size is small (GPU overhead dominates)
    if n < 1024 {
        return Err(BackendError::Unsupported(
            "Small softmax: use CPU".to_string(),
        ));
    }

    let x_buf = ctx.create_buffer_with_data(x_data)?;

    // Pass 1: Find max using partial reduction
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    let num_workgroups = wg.0 as usize;
    let partial_max_buf = ctx.create_output_buffer(num_workgroups)?;

    let push_n = (n as i32).to_le_bytes();
    ctx.dispatch("softmax_max", &[&x_buf, &partial_max_buf], &push_n, wg)?;

    // Read partial maxes and find global max on CPU
    let partial_maxes = ctx.read_buffer(&partial_max_buf)?;
    let max_val = partial_maxes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    ctx.free_buffer(partial_max_buf);

    // Pass 2: exp(x - max) per element
    let out_buf = ctx.create_output_buffer(n)?;
    let mut push2 = Vec::with_capacity(8);
    push2.extend_from_slice(&(n as i32).to_le_bytes());
    push2.extend_from_slice(&max_val.to_le_bytes());
    ctx.dispatch("softmax_exp", &[&x_buf, &out_buf], &push2, wg)?;

    // Read exp values and compute sum on CPU
    let exp_values = ctx.read_buffer(&out_buf)?;
    let sum_val: f32 = exp_values.iter().sum();
    ctx.free_buffer(x_buf);

    // Write normalized values back
    // We already have exp values, just divide by sum on CPU (avoids another GPU round trip)
    let inv_sum = 1.0 / sum_val;
    for (i, v) in exp_values.iter().enumerate() {
        out_data[i] = v * inv_sum;
    }

    ctx.free_buffer(out_buf);

    Ok(())
}

/// RMS normalization: out = x / rms(x) * weight (2-pass with partial reduction)
pub fn rms_norm(
    ctx: &VulkanContext,
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

    // Pass 1: Partial sum of squares (each workgroup writes its partial sum)
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    let num_workgroups = wg.0 as usize;
    let partial_sum_buf = ctx.create_output_buffer(num_workgroups)?;

    let push_n = (n as i32).to_le_bytes();
    ctx.dispatch("rms_norm_sum", &[&x_buf, &partial_sum_buf], &push_n, wg)?;

    // Sum partial results on CPU
    let partial_sums = ctx.read_buffer(&partial_sum_buf)?;
    let sum_sq: f32 = partial_sums.iter().sum();
    ctx.free_buffer(partial_sum_buf);

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    // Pass 2: Normalize and scale on GPU
    let out_buf = ctx.create_output_buffer(n)?;
    let mut push2 = Vec::with_capacity(8);
    push2.extend_from_slice(&(n as i32).to_le_bytes());
    push2.extend_from_slice(&rms_inv.to_le_bytes());
    ctx.dispatch("rms_norm_scale", &[&x_buf, &w_buf, &out_buf], &push2, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(x_buf);
    ctx.free_buffer(w_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Vector-matrix multiply: out = a @ b where a is [k], b is [k, n] (GGUF column-major), out is [n]
pub fn vec_mat(ctx: &VulkanContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let k = b.shape()[0];
    let n = b.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&(k as i32).to_le_bytes());
    push.extend_from_slice(&(n as i32).to_le_bytes());

    let wg = VulkanContext::workgroup_count_1d(n, 256);
    ctx.dispatch("vec_mat", &[&a_buf, &b_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(b_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Rotary Position Embedding (RoPE)
pub fn rope(
    ctx: &VulkanContext,
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

    // Push constants: [num_q_heads, num_k_heads, head_dim, position, freq_base, freq_scale, use_neox]
    let mut push = Vec::with_capacity(28);
    push.extend_from_slice(&(num_q_heads as i32).to_le_bytes());
    push.extend_from_slice(&(num_k_heads as i32).to_le_bytes());
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    push.extend_from_slice(&(pos as i32).to_le_bytes());
    push.extend_from_slice(&freq_base.to_le_bytes());
    push.extend_from_slice(&freq_scale.to_le_bytes());
    push.extend_from_slice(&(if use_neox { 1i32 } else { 0i32 }).to_le_bytes());

    let max_heads = num_q_heads.max(num_k_heads);
    let wg = (max_heads as u32, 1, 1);
    ctx.dispatch("rope", &[&q_buf, &k_buf], &push, wg)?;

    // Read back modified q and k
    let q_result = ctx.read_buffer(&q_buf)?;
    let k_result = ctx.read_buffer(&k_buf)?;

    q_data.copy_from_slice(&q_result);
    k_data.copy_from_slice(&k_result);

    ctx.free_buffer(q_buf);
    ctx.free_buffer(k_buf);

    Ok(())
}
