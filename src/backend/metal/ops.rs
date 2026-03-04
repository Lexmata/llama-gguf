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
pub fn scale(ctx: &MetalContext, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
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

/// Softmax along last dimension (3-pass GPU implementation with partial reduction)
pub fn softmax(ctx: &MetalContext, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let x_data = x.as_f32()?;
    let out_data = out.as_f32_mut()?;
    let n = x_data.len();

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

    // Compute sum on CPU (sum reduction is cheap)
    let exp_values = ctx.read_buffer(&out_buf, n);
    let sum_val: f32 = exp_values.iter().sum();
    let inv_sum = 1.0 / sum_val;

    // Pass 3: divide by sum on GPU
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SoftmaxDivParams {
        n: i32,
        inv_sum: f32,
    }

    let div_params = SoftmaxDivParams {
        n: n as i32,
        inv_sum,
    };
    let div_param_bytes = bytemuck::bytes_of(&div_params);

    ctx.dispatch(
        "softmax_div",
        &[&out_buf],
        Some((div_param_bytes, 1)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n);
    out_data.copy_from_slice(&result);

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
pub fn vec_mat(ctx: &MetalContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
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

/// Dequantize quantized tensor data on GPU
pub fn dequantize(
    ctx: &MetalContext,
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

    let n_i32 = num_blocks as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);

    let threadgroup_count = MTLSize::new(num_blocks as u64, 1, 1);
    let threadgroup_size = MTLSize::new(block_size as u64, 1, 1);

    ctx.dispatch_threadgroups(
        shader,
        &[&raw_buf, &out_buf],
        Some((param_bytes, 2)),
        threadgroup_count,
        threadgroup_size,
    )?;

    let result = ctx.read_buffer(&out_buf, num_elements);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Matrix multiply: out = a @ b where a is [m, k], b is [k, n] (row-major), out is [m, n]
pub fn matmul(ctx: &MetalContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m * n)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatmulParams {
        m: i32,
        k: i32,
        n: i32,
    }

    let params = MatmulParams {
        m: m as i32,
        k: k as i32,
        n: n as i32,
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let threadgroup_count = MTLSize::new(
        ((n + 15) / 16) as u64,
        ((m + 15) / 16) as u64,
        1,
    );
    let threadgroup_size = MTLSize::new(16, 16, 1);

    ctx.dispatch_threadgroups(
        "matmul",
        &[&a_buf, &b_buf, &out_buf],
        Some((param_bytes, 3)),
        threadgroup_count,
        threadgroup_size,
    )?;

    let result = ctx.read_buffer(&out_buf, m * n);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Matrix-vector multiply: out = a @ b where a is [m, n] (row-major), b is [n], out is [m]
pub fn matvec(ctx: &MetalContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let m = a.shape()[0];
    let n = a.shape()[1];

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatvecParams {
        m: i32,
        n: i32,
    }

    let params = MatvecParams {
        m: m as i32,
        n: n as i32,
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let (grid, tg) = MetalContext::sizes_1d(m);
    ctx.dispatch(
        "matvec",
        &[&a_buf, &b_buf, &out_buf],
        Some((param_bytes, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, m);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Quantized vector-matrix multiply (2-pass GPU: dequant + vec_mat)
pub fn vec_mat_q(
    ctx: &MetalContext,
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

    let raw_data = b.data();
    debug_assert_eq!(raw_data.len(), num_blocks * type_size);

    // Pass 1: Dequantize
    let raw_buf = ctx.create_buffer_with_raw_bytes(raw_data)?;
    let dequant_buf = ctx.create_output_buffer(num_elements)?;

    let n_i32 = num_blocks as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);
    ctx.dispatch_threadgroups(
        shader,
        &[&raw_buf, &dequant_buf],
        Some((param_bytes, 2)),
        MTLSize::new(num_blocks as u64, 1, 1),
        MTLSize::new(block_size as u64, 1, 1),
    )?;

    // Pass 2: vec_mat
    let a_data = a.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let a_buf = ctx.create_buffer_with_data(a_data)?;
    let out_buf = ctx.create_output_buffer(n_dim)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct VecMatParams {
        k: i32,
        n: i32,
    }

    let params = VecMatParams {
        k: k_dim as i32,
        n: n_dim as i32,
    };
    let param_bytes2 = bytemuck::bytes_of(&params);
    let (grid, tg) = MetalContext::sizes_1d(n_dim);
    ctx.dispatch(
        "vec_mat",
        &[&a_buf, &dequant_buf, &out_buf],
        Some((param_bytes2, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, n_dim);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Quantized matrix-vector multiply (2-pass GPU: dequant + matvec)
pub fn matvec_q(
    ctx: &MetalContext,
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

    let raw_data = a.data();
    debug_assert_eq!(raw_data.len(), num_blocks * type_size);

    // Pass 1: Dequantize
    let raw_buf = ctx.create_buffer_with_raw_bytes(raw_data)?;
    let dequant_buf = ctx.create_output_buffer(num_elements)?;

    let n_i32 = num_blocks as i32;
    let param_bytes = bytemuck::bytes_of(&n_i32);
    ctx.dispatch_threadgroups(
        shader,
        &[&raw_buf, &dequant_buf],
        Some((param_bytes, 2)),
        MTLSize::new(num_blocks as u64, 1, 1),
        MTLSize::new(block_size as u64, 1, 1),
    )?;

    // Pass 2: matvec
    let b_data = b.as_f32()?;
    let out_data = out.as_f32_mut()?;

    let b_buf = ctx.create_buffer_with_data(b_data)?;
    let out_buf = ctx.create_output_buffer(m_dim)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatvecParams {
        m: i32,
        n: i32,
    }

    let params = MatvecParams {
        m: m_dim as i32,
        n: n_dim as i32,
    };
    let param_bytes2 = bytemuck::bytes_of(&params);
    let (grid, tg) = MetalContext::sizes_1d(m_dim);
    ctx.dispatch(
        "matvec",
        &[&dequant_buf, &b_buf, &out_buf],
        Some((param_bytes2, 3)),
        grid,
        tg,
    )?;

    let result = ctx.read_buffer(&out_buf, m_dim);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Multi-head attention with causal masking and GQA (online softmax, full GPU)
pub fn attention(
    ctx: &MetalContext,
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

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttentionParams {
        num_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        kv_len: i32,
        head_dim: i32,
        scale: f32,
    }

    let params = AttentionParams {
        num_heads: num_heads as i32,
        num_kv_heads: num_kv_heads as i32,
        seq_len: seq_len as i32,
        kv_len: kv_len as i32,
        head_dim: head_dim as i32,
        scale,
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let threadgroup_count = MTLSize::new(num_heads as u64, seq_len as u64, 1);
    let threadgroup_size = MTLSize::new(256, 1, 1);

    ctx.dispatch_threadgroups(
        "attention",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        Some((param_bytes, 4)),
        threadgroup_count,
        threadgroup_size,
    )?;

    let result = ctx.read_buffer(&out_buf, num_heads * seq_len * head_dim);
    out_data.copy_from_slice(&result);

    Ok(())
}

/// Cached attention for single-token generation with GQA (online softmax, full GPU)
pub fn attention_cached(
    ctx: &MetalContext,
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

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttentionCachedParams {
        num_heads: i32,
        num_kv_heads: i32,
        kv_len: i32,
        max_seq_len: i32,
        head_dim: i32,
        scale: f32,
    }

    let params = AttentionCachedParams {
        num_heads: num_heads as i32,
        num_kv_heads: num_kv_heads as i32,
        kv_len: kv_len as i32,
        max_seq_len: max_seq_len as i32,
        head_dim: head_dim as i32,
        scale,
    };
    let param_bytes = bytemuck::bytes_of(&params);

    let threadgroup_count = MTLSize::new(num_heads as u64, 1, 1);
    let threadgroup_size = MTLSize::new(256, 1, 1);

    ctx.dispatch_threadgroups(
        "attention_cached",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        Some((param_bytes, 4)),
        threadgroup_count,
        threadgroup_size,
    )?;

    let result = ctx.read_buffer(&out_buf, num_heads * head_dim);
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
