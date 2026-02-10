//! DX12 compute operation implementations.
//!
//! Each operation follows the pattern:
//! 1. Upload input data to GPU buffers
//! 2. Dispatch compute shader
//! 3. Read back output data
//! 4. (buffers are dropped automatically via RAII)

use crate::backend::dx12::context::Dx12Context;
use crate::backend::{BackendError, BackendResult};
use crate::tensor::Tensor;

/// Element-wise addition: out = a + b
pub fn add(ctx: &Dx12Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let constants = [n as u32];
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    ctx.dispatch("add", &[&a_buf, &b_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Element-wise multiplication: out = a * b
pub fn mul(ctx: &Dx12Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let constants = [n as u32];
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    ctx.dispatch("mul", &[&a_buf, &b_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Scale: out = a * scalar
pub fn scale(ctx: &Dx12Context, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = a_data.len();

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    // Root constants: [n: u32, scalar: f32 as u32 bits]
    let constants = [n as u32, scalar.to_bits()];
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    ctx.dispatch("scale", &[&a_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// SiLU activation: out = x * sigmoid(x)
pub fn silu(ctx: &Dx12Context, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let constants = [n as u32];
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    ctx.dispatch("silu", &[&x_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// GELU activation
pub fn gelu(ctx: &Dx12Context, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let constants = [n as u32];
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    ctx.dispatch("gelu", &[&x_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Softmax along last dimension (3-pass implementation with partial reduction)
pub fn softmax(ctx: &Dx12Context, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
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
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    let num_workgroups = wg.0 as usize;
    let partial_max_buf = ctx.create_output_buffer(num_workgroups)?;

    let constants_n = [n as u32];
    ctx.dispatch("softmax_max", &[&x_buf, &partial_max_buf], &constants_n, wg)?;

    // Read partial maxes and find global max on CPU
    let partial_maxes = ctx.read_buffer(&partial_max_buf)?;
    let max_val = partial_maxes
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Pass 2: exp(x - max) per element
    let out_buf = ctx.create_output_buffer(n)?;
    let constants_exp = [n as u32, max_val.to_bits()];
    ctx.dispatch("softmax_exp", &[&x_buf, &out_buf], &constants_exp, wg)?;

    // Read exp values and compute sum on CPU
    let exp_values = ctx.read_buffer(&out_buf)?;
    let sum_val: f32 = exp_values.iter().sum();

    // Write normalized values back (divide on CPU to avoid another GPU round trip)
    let inv_sum = 1.0 / sum_val;
    for (i, v) in exp_values.iter().enumerate() {
        out_data[i] = v * inv_sum;
    }

    Ok(())
}

/// RMS normalization: out = x / rms(x) * weight (2-pass with partial reduction)
pub fn rms_norm(
    ctx: &Dx12Context,
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
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    let num_workgroups = wg.0 as usize;
    let partial_sum_buf = ctx.create_output_buffer(num_workgroups)?;

    let constants_n = [n as u32];
    ctx.dispatch(
        "rms_norm_sum",
        &[&x_buf, &partial_sum_buf],
        &constants_n,
        wg,
    )?;

    // Sum partial results on CPU
    let partial_sums = ctx.read_buffer(&partial_sum_buf)?;
    let sum_sq: f32 = partial_sums.iter().sum();

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    // Pass 2: Normalize and scale on GPU
    let out_buf = ctx.create_output_buffer(n)?;
    let constants_scale = [n as u32, rms_inv.to_bits()];
    ctx.dispatch(
        "rms_norm_scale",
        &[&x_buf, &w_buf, &out_buf],
        &constants_scale,
        wg,
    )?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Vector-matrix multiply: out = a @ b where a is [k], b is [k, n] (GGUF column-major), out is [n]
pub fn vec_mat(ctx: &Dx12Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let k = b.shape()[0];
    let n = b.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(n)?;

    let constants = [k as u32, n as u32];
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    ctx.dispatch("vec_mat", &[&a_buf, &b_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Rotary Position Embedding (RoPE)
pub fn rope(
    ctx: &Dx12Context,
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

    // Root constants: [num_q_heads, num_k_heads, head_dim, position, freq_base, freq_scale, use_neox]
    let constants = [
        num_q_heads as u32,
        num_k_heads as u32,
        head_dim as u32,
        pos as u32,
        freq_base.to_bits(),
        freq_scale.to_bits(),
        if use_neox { 1u32 } else { 0u32 },
    ];

    let max_heads = num_q_heads.max(num_k_heads);
    let wg = (max_heads as u32, 1, 1);
    ctx.dispatch("rope", &[&q_buf, &k_buf], &constants, wg)?;

    // Read back modified q and k
    let q_result = ctx.read_buffer(&q_buf)?;
    let k_result = ctx.read_buffer(&k_buf)?;

    q_data.copy_from_slice(&q_result);
    k_data.copy_from_slice(&k_result);

    Ok(())
}
