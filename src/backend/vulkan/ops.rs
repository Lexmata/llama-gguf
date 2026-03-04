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
pub fn scale(ctx: &VulkanContext, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
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

/// Softmax along last dimension (3-pass GPU implementation with partial reduction)
pub fn softmax(ctx: &VulkanContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;

    // Pass 1: Find max using partial reduction
    let wg = VulkanContext::workgroup_count_1d(n, 256);
    let num_workgroups = wg.0 as usize;
    let partial_max_buf = ctx.create_output_buffer(num_workgroups)?;

    let push_n = (n as i32).to_le_bytes();
    ctx.dispatch("softmax_max", &[&x_buf, &partial_max_buf], &push_n, wg)?;

    let partial_maxes = ctx.read_buffer(&partial_max_buf)?;
    let max_val = partial_maxes
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    ctx.free_buffer(partial_max_buf);

    // Pass 2: exp(x - max) per element, also compute partial sums
    let out_buf = ctx.create_output_buffer(n)?;
    let mut push2 = Vec::with_capacity(8);
    push2.extend_from_slice(&(n as i32).to_le_bytes());
    push2.extend_from_slice(&max_val.to_le_bytes());
    ctx.dispatch("softmax_exp", &[&x_buf, &out_buf], &push2, wg)?;
    ctx.free_buffer(x_buf);

    // Read exp values to compute sum on CPU (sum reduction is cheap)
    let exp_values = ctx.read_buffer(&out_buf)?;
    let sum_val: f32 = exp_values.iter().sum();
    let inv_sum = 1.0 / sum_val;

    // Pass 3: divide by sum on GPU
    let mut push3 = Vec::with_capacity(8);
    push3.extend_from_slice(&(n as i32).to_le_bytes());
    push3.extend_from_slice(&inv_sum.to_le_bytes());
    ctx.dispatch("softmax_div", &[&out_buf], &push3, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

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

/// Dequantize quantized tensor data on GPU
pub fn dequantize(
    ctx: &VulkanContext,
    src: &crate::tensor::Tensor,
    out: &mut crate::tensor::Tensor,
) -> BackendResult<()> {
    use crate::tensor::DType;

    let raw_data = src.data();
    let out_data = out.as_f32_mut()?;
    let num_elements = out_data.len();

    let (shader, block_size, type_size) = match src.dtype() {
        DType::Q8_0 => ("dequant_q8_0", 32usize, 34usize),
        DType::Q4K => ("dequant_q4_k", 256usize, 144usize),
        DType::Q6K => ("dequant_q6_k", 256usize, 210usize),
        other => {
            return Err(BackendError::Unsupported(format!(
                "GPU dequantize not supported for {:?}",
                other
            )));
        }
    };

    let num_blocks = num_elements / block_size;
    debug_assert_eq!(raw_data.len(), num_blocks * type_size);

    let raw_buf = ctx.create_buffer_with_bytes(raw_data, raw_data.len() as u64)?;
    let out_buf = ctx.create_output_buffer(num_elements)?;

    let push = (num_blocks as i32).to_le_bytes();
    let wg = (num_blocks as u32, 1, 1);
    ctx.dispatch(shader, &[&raw_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(raw_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Matrix multiply: out = a @ b where a is [m, k], b is [k, n] (row-major), out is [m, n]
pub fn matmul(ctx: &VulkanContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m * n)?;

    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&(m as i32).to_le_bytes());
    push.extend_from_slice(&(k as i32).to_le_bytes());
    push.extend_from_slice(&(n as i32).to_le_bytes());

    let wg_x = ((n + 15) / 16) as u32;
    let wg_y = ((m + 15) / 16) as u32;
    let wg = (wg_x, wg_y, 1);
    ctx.dispatch("matmul", &[&a_buf, &b_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(b_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Matrix-vector multiply: out = a @ b where a is [m, n] (row-major), b is [n], out is [m]
pub fn matvec(ctx: &VulkanContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let m = a.shape()[0];
    let n = a.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m)?;

    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&(m as i32).to_le_bytes());
    push.extend_from_slice(&(n as i32).to_le_bytes());

    let wg = VulkanContext::workgroup_count_1d(m, 256);
    ctx.dispatch("matvec", &[&a_buf, &b_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(b_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Quantized vector-matrix multiply (2-pass GPU: dequant + vec_mat)
pub fn vec_mat_q(
    ctx: &VulkanContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> BackendResult<()> {
    use crate::tensor::DType;

    if b.dtype() == DType::F32 {
        return vec_mat(ctx, a, b, out);
    }

    let (block_size, type_size) = match b.dtype() {
        DType::Q8_0 => (32usize, 34usize),
        DType::Q4K => (256usize, 144usize),
        DType::Q6K => (256usize, 210usize),
        other => {
            return Err(BackendError::Unsupported(format!(
                "GPU vec_mat_q not supported for {:?}",
                other
            )));
        }
    };

    let k_dim = b.shape()[0];
    let n_dim = b.shape()[1];
    let num_elements = k_dim * n_dim;
    let num_blocks = num_elements / block_size;

    let raw_data = b.data();
    debug_assert_eq!(raw_data.len(), num_blocks * type_size);

    // Pass 1: Dequantize to temp GPU buffer
    let raw_buf = ctx.create_buffer_with_bytes(raw_data, raw_data.len() as u64)?;
    let dequant_buf = ctx.create_output_buffer(num_elements)?;

    let shader = match b.dtype() {
        DType::Q8_0 => "dequant_q8_0",
        DType::Q4K => "dequant_q4_k",
        DType::Q6K => "dequant_q6_k",
        _ => unreachable!(),
    };

    let push = (num_blocks as i32).to_le_bytes();
    let wg = (num_blocks as u32, 1, 1);
    ctx.dispatch(shader, &[&raw_buf, &dequant_buf], &push, wg)?;
    ctx.free_buffer(raw_buf);

    // Pass 2: vec_mat using the dequantized f32 buffer
    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let out_buf = ctx.create_output_buffer(n_dim)?;

    let mut push2 = Vec::with_capacity(8);
    push2.extend_from_slice(&(k_dim as i32).to_le_bytes());
    push2.extend_from_slice(&(n_dim as i32).to_le_bytes());

    let wg2 = VulkanContext::workgroup_count_1d(n_dim, 256);
    ctx.dispatch("vec_mat", &[&a_buf, &dequant_buf, &out_buf], &push2, wg2)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(a_buf);
    ctx.free_buffer(dequant_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Quantized matrix-vector multiply (2-pass GPU: dequant + matvec)
pub fn matvec_q(
    ctx: &VulkanContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> BackendResult<()> {
    use crate::tensor::DType;

    if a.dtype() == DType::F32 {
        return matvec(ctx, a, b, out);
    }

    let (block_size, type_size) = match a.dtype() {
        DType::Q8_0 => (32usize, 34usize),
        DType::Q4K => (256usize, 144usize),
        DType::Q6K => (256usize, 210usize),
        other => {
            return Err(BackendError::Unsupported(format!(
                "GPU matvec_q not supported for {:?}",
                other
            )));
        }
    };

    let m_dim = a.shape()[0];
    let n_dim = a.shape()[1];
    let num_elements = m_dim * n_dim;
    let num_blocks = num_elements / block_size;

    let raw_data = a.data();
    debug_assert_eq!(raw_data.len(), num_blocks * type_size);

    // Pass 1: Dequantize
    let raw_buf = ctx.create_buffer_with_bytes(raw_data, raw_data.len() as u64)?;
    let dequant_buf = ctx.create_output_buffer(num_elements)?;

    let shader = match a.dtype() {
        DType::Q8_0 => "dequant_q8_0",
        DType::Q4K => "dequant_q4_k",
        DType::Q6K => "dequant_q6_k",
        _ => unreachable!(),
    };

    let push = (num_blocks as i32).to_le_bytes();
    let wg = (num_blocks as u32, 1, 1);
    ctx.dispatch(shader, &[&raw_buf, &dequant_buf], &push, wg)?;
    ctx.free_buffer(raw_buf);

    // Pass 2: matvec using dequantized f32 buffer
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m_dim)?;

    let mut push2 = Vec::with_capacity(8);
    push2.extend_from_slice(&(m_dim as i32).to_le_bytes());
    push2.extend_from_slice(&(n_dim as i32).to_le_bytes());

    let wg2 = VulkanContext::workgroup_count_1d(m_dim, 256);
    ctx.dispatch("matvec", &[&dequant_buf, &b_buf, &out_buf], &push2, wg2)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(b_buf);
    ctx.free_buffer(dequant_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Rotary Position Embedding (RoPE)
/// Multi-head attention with causal masking and GQA (online softmax, full GPU)
/// Q: [num_heads, seq_len, head_dim], K: [num_kv_heads, kv_len, head_dim]
/// V: [num_kv_heads, kv_len, head_dim], Out: [num_heads, seq_len, head_dim]
pub fn attention(
    ctx: &VulkanContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
    scale: f32,
) -> BackendResult<()> {
    let q_data = q.as_f32()?;
    let k_data = k.as_f32()?;
    let v_data = v.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let num_heads = q.shape()[0];
    let seq_len = q.shape()[1];
    let head_dim = q.shape()[2];
    let num_kv_heads = k.shape()[0];
    let kv_len = k.shape()[1];

    let q_buf = ctx.create_buffer_with_data(q_data)?;
    let k_buf = ctx.create_buffer_with_data(k_data)?;
    let v_buf = ctx.create_buffer_with_data(v_data)?;
    let out_buf = ctx.create_output_buffer(num_heads * seq_len * head_dim)?;

    let mut push = Vec::with_capacity(24);
    push.extend_from_slice(&(num_heads as i32).to_le_bytes());
    push.extend_from_slice(&(num_kv_heads as i32).to_le_bytes());
    push.extend_from_slice(&(seq_len as i32).to_le_bytes());
    push.extend_from_slice(&(kv_len as i32).to_le_bytes());
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    push.extend_from_slice(&scale.to_le_bytes());

    let wg = (num_heads as u32, seq_len as u32, 1);
    ctx.dispatch("attention", &[&q_buf, &k_buf, &v_buf, &out_buf], &push, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(q_buf);
    ctx.free_buffer(k_buf);
    ctx.free_buffer(v_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

/// Cached attention for single-token generation with GQA (online softmax, full GPU)
/// Q: [num_heads, 1, head_dim], K_cache: [num_kv_heads, max_seq_len, head_dim]
/// V_cache: [num_kv_heads, max_seq_len, head_dim], Out: [num_heads, 1, head_dim]
pub fn attention_cached(
    ctx: &VulkanContext,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &mut Tensor,
    scale: f32,
    kv_len: usize,
) -> BackendResult<()> {
    let q_data = q.as_f32()?;
    let k_data = k_cache.as_f32()?;
    let v_data = v_cache.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let num_heads = q.shape()[0];
    let head_dim = q.shape()[q.ndim() - 1];
    let num_kv_heads = k_cache.shape()[0];
    let max_seq_len = k_cache.shape()[1];

    let q_buf = ctx.create_buffer_with_data(q_data)?;
    let k_buf = ctx.create_buffer_with_data(k_data)?;
    let v_buf = ctx.create_buffer_with_data(v_data)?;
    let out_buf = ctx.create_output_buffer(num_heads * head_dim)?;

    let mut push = Vec::with_capacity(24);
    push.extend_from_slice(&(num_heads as i32).to_le_bytes());
    push.extend_from_slice(&(num_kv_heads as i32).to_le_bytes());
    push.extend_from_slice(&(kv_len as i32).to_le_bytes());
    push.extend_from_slice(&(max_seq_len as i32).to_le_bytes());
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    push.extend_from_slice(&scale.to_le_bytes());

    let wg = (num_heads as u32, 1, 1);
    ctx.dispatch(
        "attention_cached",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        &push,
        wg,
    )?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    ctx.free_buffer(q_buf);
    ctx.free_buffer(k_buf);
    ctx.free_buffer(v_buf);
    ctx.free_buffer(out_buf);

    Ok(())
}

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

    // Handle 1D [total], 2D [num_heads, head_dim], or 3D [num_heads, seq_len, head_dim]
    let (num_q_heads, head_dim) = match q_shape.len() {
        2 => (q_shape[0], q_shape[1]),
        3 => (q_shape[0], q_shape[2]),
        _ => {
            return Err(BackendError::Unsupported(format!(
                "RoPE unsupported shape: {:?}",
                q_shape
            )));
        }
    };

    let num_k_heads = match k_shape.len() {
        2 => k_shape[0],
        3 => k_shape[0],
        _ => {
            return Err(BackendError::Unsupported(format!(
                "RoPE unsupported k shape: {:?}",
                k_shape
            )));
        }
    };

    let q_data = q.as_f32_mut()?;
    let k_data = k.as_f32_mut()?;

    let q_buf = ctx.create_buffer_with_data(q_data)?;
    let k_buf = ctx.create_buffer_with_data(k_data)?;

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

    let q_result = ctx.read_buffer(&q_buf)?;
    let k_result = ctx.read_buffer(&k_buf)?;

    q_data.copy_from_slice(&q_result);
    k_data.copy_from_slice(&k_result);

    ctx.free_buffer(q_buf);
    ctx.free_buffer(k_buf);

    Ok(())
}
