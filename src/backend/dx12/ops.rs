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

/// Softmax along last dimension (3-pass GPU implementation with partial reduction)
pub fn softmax(ctx: &Dx12Context, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

    let x_buf = ctx.create_buffer_with_data(x_data)?;

    // Pass 1: Find max using partial reduction
    let wg = Dx12Context::workgroup_count_1d(n, 256);
    let num_workgroups = wg.0 as usize;
    let partial_max_buf = ctx.create_output_buffer(num_workgroups)?;

    let constants_n = [n as u32];
    ctx.dispatch("softmax_max", &[&x_buf, &partial_max_buf], &constants_n, wg)?;

    let partial_maxes = ctx.read_buffer(&partial_max_buf)?;
    let max_val = partial_maxes
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Pass 2: exp(x - max) per element
    let out_buf = ctx.create_output_buffer(n)?;
    let constants_exp = [n as u32, max_val.to_bits()];
    ctx.dispatch("softmax_exp", &[&x_buf, &out_buf], &constants_exp, wg)?;

    // Compute sum on CPU (sum reduction is cheap)
    let exp_values = ctx.read_buffer(&out_buf)?;
    let sum_val: f32 = exp_values.iter().sum();
    let inv_sum = 1.0 / sum_val;

    // Pass 3: divide by sum on GPU
    let constants_div = [n as u32, inv_sum.to_bits()];
    ctx.dispatch("softmax_div", &[&out_buf], &constants_div, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

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

/// Dequantize quantized tensor data on GPU
pub fn dequantize(
    ctx: &Dx12Context,
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

    let raw_buf = ctx.create_buffer_with_raw_bytes(raw_data)?;
    let out_buf = ctx.create_output_buffer(num_elements)?;

    let constants = [num_blocks as u32];
    let wg = (num_blocks as u32, 1, 1);
    ctx.dispatch(shader, &[&raw_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Matrix multiply: out = a @ b where a is [m, k], b is [k, n] (row-major), out is [m, n]
pub fn matmul(ctx: &Dx12Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m * n)?;

    let constants = [m as u32, k as u32, n as u32];
    let wg_x = ((n + 15) / 16) as u32;
    let wg_y = ((m + 15) / 16) as u32;
    let wg = (wg_x, wg_y, 1);
    ctx.dispatch("matmul", &[&a_buf, &b_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Matrix-vector multiply: out = a @ b where a is [m, n] (row-major), b is [n], out is [m]
pub fn matvec(ctx: &Dx12Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let m = a.shape()[0];
    let n = a.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m)?;

    let constants = [m as u32, n as u32];
    let wg = Dx12Context::workgroup_count_1d(m, 256);
    ctx.dispatch("matvec", &[&a_buf, &b_buf, &out_buf], &constants, wg)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Quantized vector-matrix multiply (2-pass GPU: dequant + vec_mat)
pub fn vec_mat_q(
    ctx: &Dx12Context,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> BackendResult<()> {
    use crate::tensor::DType;

    if b.dtype() == DType::F32 {
        return vec_mat(ctx, a, b, out);
    }

    let (block_size, type_size, shader) = match b.dtype() {
        DType::Q8_0 => (32usize, 34usize, "dequant_q8_0"),
        DType::Q4K => (256usize, 144usize, "dequant_q4_k"),
        DType::Q6K => (256usize, 210usize, "dequant_q6_k"),
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
    let _ = type_size;

    let raw_buf = ctx.create_buffer_with_raw_bytes(b.data())?;
    let dequant_buf = ctx.create_output_buffer(num_elements)?;

    let constants = [num_blocks as u32];
    let wg = (num_blocks as u32, 1, 1);
    ctx.dispatch(shader, &[&raw_buf, &dequant_buf], &constants, wg)?;

    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let out_buf = ctx.create_output_buffer(n_dim)?;

    let constants2 = [k_dim as u32, n_dim as u32];
    let wg2 = Dx12Context::workgroup_count_1d(n_dim, 256);
    ctx.dispatch("vec_mat", &[&a_buf, &dequant_buf, &out_buf], &constants2, wg2)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Quantized matrix-vector multiply (2-pass GPU: dequant + matvec)
pub fn matvec_q(
    ctx: &Dx12Context,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> BackendResult<()> {
    use crate::tensor::DType;

    if a.dtype() == DType::F32 {
        return matvec(ctx, a, b, out);
    }

    let (block_size, type_size, shader) = match a.dtype() {
        DType::Q8_0 => (32usize, 34usize, "dequant_q8_0"),
        DType::Q4K => (256usize, 144usize, "dequant_q4_k"),
        DType::Q6K => (256usize, 210usize, "dequant_q6_k"),
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
    let _ = type_size;

    let raw_buf = ctx.create_buffer_with_raw_bytes(a.data())?;
    let dequant_buf = ctx.create_output_buffer(num_elements)?;

    let constants = [num_blocks as u32];
    let wg = (num_blocks as u32, 1, 1);
    ctx.dispatch(shader, &[&raw_buf, &dequant_buf], &constants, wg)?;

    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m_dim)?;

    let constants2 = [m_dim as u32, n_dim as u32];
    let wg2 = Dx12Context::workgroup_count_1d(m_dim, 256);
    ctx.dispatch("matvec", &[&dequant_buf, &b_buf, &out_buf], &constants2, wg2)?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Multi-head attention with causal masking and GQA (online softmax, full GPU)
pub fn attention(
    ctx: &Dx12Context,
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

    let constants = [
        num_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        kv_len as u32,
        head_dim as u32,
        scale.to_bits(),
    ];
    let wg = (num_heads as u32, seq_len as u32, 1);
    ctx.dispatch(
        "attention",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        &constants,
        wg,
    )?;

    let result = ctx.read_buffer(&out_buf)?;
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Cached attention for single-token generation with GQA (online softmax, full GPU)
pub fn attention_cached(
    ctx: &Dx12Context,
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

    let constants = [
        num_heads as u32,
        num_kv_heads as u32,
        kv_len as u32,
        max_seq_len as u32,
        head_dim as u32,
        scale.to_bits(),
    ];
    let wg = (num_heads as u32, 1, 1);
    ctx.dispatch(
        "attention_cached",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        &constants,
        wg,
    )?;

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

    // Handle 2D [num_heads, head_dim] or 3D [num_heads, seq_len, head_dim]
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

    // Upload q and k as read-write buffers (RoPE modifies in place, needs readback)
    let q_buf = ctx.create_readwrite_buffer(q_data)?;
    let k_buf = ctx.create_readwrite_buffer(k_data)?;

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
