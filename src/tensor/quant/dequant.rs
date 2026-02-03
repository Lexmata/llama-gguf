//! Dequantization functions for converting quantized blocks to f32
//!
//! These functions match llama.cpp's dequantization implementations exactly.

use super::blocks::*;
use half::f16;

// =============================================================================
// Basic Quantization Dequantization (32 elements per block)
// =============================================================================

/// Dequantize Q4_0 block to f32
///
/// Q4_0 stores 32 4-bit values with a single f16 scale factor.
/// Values are stored as unsigned 0-15, then shifted by -8 to get signed -8..7.
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();

    for i in 0..16 {
        let byte = block.qs[i];
        // Low 4 bits (first half of block)
        let lo = (byte & 0x0F) as i32 - 8;
        // High 4 bits (second half of block)
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;

        output[i] = lo as f32 * d;
        output[i + 16] = hi as f32 * d;
    }
}

/// Dequantize Q4_1 block to f32
///
/// Q4_1 stores 32 4-bit values with scale (d) and minimum (m).
/// Values are stored as unsigned 0-15, then: value * d + m
pub fn dequantize_q4_1(block: &BlockQ4_1, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let m = block.m.to_f32();

    for i in 0..16 {
        let byte = block.qs[i];
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;

        output[i] = lo * d + m;
        output[i + 16] = hi * d + m;
    }
}

/// Dequantize Q5_0 block to f32
///
/// Q5_0 stores 32 5-bit values. The low 4 bits are in qs[], and
/// the high bit for each value is packed in qh (32 bits for 32 values).
pub fn dequantize_q5_0(block: &BlockQ5_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let qh = u32::from_le_bytes(block.qh);

    for i in 0..16 {
        let byte = block.qs[i];
        // Low 4 bits from qs
        let lo4 = (byte & 0x0F) as i32;
        let hi4 = ((byte >> 4) & 0x0F) as i32;

        // High bits from qh
        let lo5 = ((qh >> i) & 1) as i32;
        let hi5 = ((qh >> (i + 16)) & 1) as i32;

        // Combine to get 5-bit value, shift by -16 for signed
        let lo = (lo4 | (lo5 << 4)) - 16;
        let hi = (hi4 | (hi5 << 4)) - 16;

        output[i] = lo as f32 * d;
        output[i + 16] = hi as f32 * d;
    }
}

/// Dequantize Q5_1 block to f32
///
/// Q5_1 stores 32 5-bit values with scale (d) and minimum (m).
pub fn dequantize_q5_1(block: &BlockQ5_1, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let m = block.m.to_f32();
    let qh = u32::from_le_bytes(block.qh);

    for i in 0..16 {
        let byte = block.qs[i];
        let lo4 = (byte & 0x0F) as u32;
        let hi4 = ((byte >> 4) & 0x0F) as u32;

        let lo5 = (qh >> i) & 1;
        let hi5 = (qh >> (i + 16)) & 1;

        let lo = lo4 | (lo5 << 4);
        let hi = hi4 | (hi5 << 4);

        output[i] = lo as f32 * d + m;
        output[i + 16] = hi as f32 * d + m;
    }
}

/// Dequantize Q8_0 block to f32
///
/// Q8_0 stores 32 signed 8-bit values with a single f16 scale factor.
pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();

    for (o, &q) in output.iter_mut().zip(block.qs.iter()) {
        *o = q as f32 * d;
    }
}

/// Dequantize Q8_1 block to f32
///
/// Q8_1 stores 32 signed 8-bit values with a f32 scale factor.
pub fn dequantize_q8_1(block: &BlockQ8_1, output: &mut [f32; 32]) {
    let d = block.d;

    for (o, &q) in output.iter_mut().zip(block.qs.iter()) {
        *o = q as f32 * d;
    }
}

// =============================================================================
// K-Quant Dequantization (256 elements per block)
// =============================================================================

/// Dequantize Q2_K block to f32 (256 elements)
///
/// Q2_K stores 256 2-bit values in groups of 32, with 4-bit scales and mins.
pub fn dequantize_q2_k(block: &BlockQ2K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    // Process 16 groups of 16 values (2 sub-blocks of 128 each)
    for i in 0..16 {
        // Each scale byte encodes two 4-bit values: scale and min
        let sc_m = block.scales[i];
        let scale = (sc_m & 0x0F) as f32;
        let min = ((sc_m >> 4) & 0x0F) as f32;

        let d_scale = d * scale;
        let d_min = dmin * min;

        // Each group of 16 values uses 4 bytes (16 * 2 bits = 32 bits = 4 bytes)
        let offset = i * 16;
        let qs_offset = i * 4;

        for j in 0..4 {
            let byte = block.qs[qs_offset + j];
            // Unpack 4 2-bit values from each byte
            for k in 0..4 {
                let q = ((byte >> (k * 2)) & 0x03) as f32;
                output[offset + j * 4 + k] = d_scale * q - d_min;
            }
        }
    }
}

/// Dequantize Q3_K block to f32 (256 elements)
///
/// Q3_K stores 256 3-bit values. Low 2 bits in qs, high bit in hmask.
pub fn dequantize_q3_k(block: &BlockQ3K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();

    // Decode the 12-byte scales (16 6-bit scales encoded)
    let mut scales = [0i8; 16];
    for i in 0..4 {
        let b0 = block.scales[i * 3] as i32;
        let b1 = block.scales[i * 3 + 1] as i32;
        let b2 = block.scales[i * 3 + 2] as i32;

        scales[i * 4] = ((b0 & 0x3F) as i8).wrapping_sub(32);
        scales[i * 4 + 1] = (((b0 >> 6) | ((b1 & 0x0F) << 2)) as i8).wrapping_sub(32);
        scales[i * 4 + 2] = (((b1 >> 4) | ((b2 & 0x03) << 4)) as i8).wrapping_sub(32);
        scales[i * 4 + 3] = ((b2 >> 2) as i8).wrapping_sub(32);
    }

    // Process 16 groups of 16 values
    for i in 0..16 {
        let scale = d * scales[i] as f32;
        let offset = i * 16;

        for j in 0..16 {
            let qs_idx = offset + j;
            let qs_byte_idx = qs_idx / 4;
            let qs_shift = (qs_idx % 4) * 2;

            // Low 2 bits from qs
            let lo2 = ((block.qs[qs_byte_idx] >> qs_shift) & 0x03) as i32;

            // High bit from hmask
            let hmask_idx = qs_idx / 8;
            let hmask_shift = qs_idx % 8;
            let hi1 = ((block.hmask[hmask_idx] >> hmask_shift) & 0x01) as i32;

            // Combine to 3-bit value and convert to signed
            let q = (lo2 | (hi1 << 2)) - 4;
            output[offset + j] = scale * q as f32;
        }
    }
}

/// Dequantize Q4_K block to f32 (256 elements)
///
/// Q4_K stores 256 4-bit values in 8 groups of 32, with 6-bit scales and mins.
pub fn dequantize_q4_k(block: &BlockQ4K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    // Decode 8 groups of (scale, min) from 12 bytes using get_scale_min_k4 logic
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    // First 4 groups (j < 4): scales in lower 6 bits of bytes 0-3, mins in lower 6 bits of bytes 4-7
    for j in 0..4 {
        scales[j] = block.scales[j] & 0x3F;
        mins[j] = block.scales[j + 4] & 0x3F;
    }

    // Last 4 groups (j >= 4): high bits from earlier bytes
    for j in 4..8 {
        scales[j] = (block.scales[j + 4] & 0x0F) | ((block.scales[j - 4] >> 6) << 4);
        mins[j] = ((block.scales[j + 4] >> 4) & 0x0F) | ((block.scales[j] >> 6) << 4);
    }

    // Process 256 values in 4 groups of 64
    // Each group of 64 uses two scale/min pairs
    // First 32 use low nibbles with scale[is], next 32 use high nibbles with scale[is+1]
    let mut out_idx = 0;
    let mut qs_ptr = 0;
    let mut is = 0;

    for _ in 0..4 {
        // Get scales and mins for this group of 64
        let d1 = d * scales[is] as f32;
        let m1 = dmin * mins[is] as f32;
        let d2 = d * scales[is + 1] as f32;
        let m2 = dmin * mins[is + 1] as f32;

        // First 32 values: low nibbles
        for l in 0..32 {
            let q = (block.qs[qs_ptr + l] & 0x0F) as f32;
            output[out_idx] = d1 * q - m1;
            out_idx += 1;
        }

        // Next 32 values: high nibbles
        for l in 0..32 {
            let q = ((block.qs[qs_ptr + l] >> 4) & 0x0F) as f32;
            output[out_idx] = d2 * q - m2;
            out_idx += 1;
        }

        qs_ptr += 32;
        is += 2;
    }
}

/// Dequantize Q5_K block to f32 (256 elements)
///
/// Q5_K stores 256 5-bit values. Low 4 bits in qs, high bit in qh.
/// Layout matches llama.cpp's dequantize_row_q5_K.
pub fn dequantize_q5_k(block: &BlockQ5K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    // Decode scales and mins (same as Q4_K using get_scale_min_k4 logic)
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for j in 0..4 {
        scales[j] = block.scales[j] & 0x3F;
        mins[j] = block.scales[j + 4] & 0x3F;
    }

    for j in 4..8 {
        scales[j] = (block.scales[j + 4] & 0x0F) | ((block.scales[j - 4] >> 6) << 4);
        mins[j] = ((block.scales[j + 4] >> 4) & 0x0F) | ((block.scales[j] >> 6) << 4);
    }

    // Process 256 values in 4 groups of 64
    let mut out_idx = 0;
    let mut ql_ptr = 0;
    let mut is = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for _ in 0..4 {
        // Get scales and mins for this group of 64
        let d1 = d * scales[is] as f32;
        let m1 = dmin * mins[is] as f32;
        let d2 = d * scales[is + 1] as f32;
        let m2 = dmin * mins[is + 1] as f32;

        // First 32 values: low nibbles with u1 high bit
        for l in 0..32 {
            let lo4 = (block.qs[ql_ptr + l] & 0x0F) as f32;
            let hi5 = if block.qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            output[out_idx] = d1 * (lo4 + hi5) - m1;
            out_idx += 1;
        }

        // Next 32 values: high nibbles with u2 high bit
        for l in 0..32 {
            let hi4 = ((block.qs[ql_ptr + l] >> 4) & 0x0F) as f32;
            let hi5 = if block.qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            output[out_idx] = d2 * (hi4 + hi5) - m2;
            out_idx += 1;
        }

        ql_ptr += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantize Q6_K block to f32 (256 elements)
///
/// Q6_K stores 256 6-bit values using complex interleaved packing.
/// Layout matches llama.cpp's dequantize_row_q6_K.
pub fn dequantize_q6_k(block: &BlockQ6K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();

    // Process 256 elements in two groups of 128
    // Each group uses 64 bytes from ql, 32 bytes from qh, 8 scales
    for n in 0..2 {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let sc_base = n * 8;
        let out_base = n * 128;

        for l in 0..32 {
            let is = l / 16;  // Scale group index (0 or 1)

            // Extract 4 quantized values using interleaved pattern
            let q1 = ((block.ql[ql_base + l] & 0x0F) | ((block.qh[qh_base + l] & 0x03) << 4)) as i32 - 32;
            let q2 = ((block.ql[ql_base + l + 32] & 0x0F) | (((block.qh[qh_base + l] >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((block.ql[ql_base + l] >> 4) | (((block.qh[qh_base + l] >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((block.ql[ql_base + l + 32] >> 4) | (((block.qh[qh_base + l] >> 6) & 0x03) << 4)) as i32 - 32;

            // Apply scales with correct interleaved pattern
            output[out_base + l] = d * block.scales[sc_base + is] as f32 * q1 as f32;
            output[out_base + l + 32] = d * block.scales[sc_base + is + 2] as f32 * q2 as f32;
            output[out_base + l + 64] = d * block.scales[sc_base + is + 4] as f32 * q3 as f32;
            output[out_base + l + 96] = d * block.scales[sc_base + is + 6] as f32 * q4 as f32;
        }
    }
}

/// Dequantize Q8_K block to f32 (256 elements)
///
/// Q8_K stores 256 8-bit signed values with a single f32 scale.
pub fn dequantize_q8_k(block: &BlockQ8K, output: &mut [f32; 256]) {
    let d = block.d;

    for (i, &q) in block.qs.iter().enumerate() {
        output[i] = q as f32 * d;
    }
}

// =============================================================================
// Quantization Functions (for roundtrip testing)
// =============================================================================

/// Quantize f32 values to Q4_0 block
pub fn quantize_q4_0(input: &[f32; 32]) -> BlockQ4_0 {
    // Find max absolute value
    let mut amax = 0.0f32;
    for &x in input.iter() {
        amax = amax.max(x.abs());
    }

    // Scale factor: map [-amax, amax] to [-8, 7]
    let d = amax / 7.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];
    for i in 0..16 {
        // Quantize and shift to [0, 15] range
        let lo = ((input[i] * id).round() as i32).clamp(-8, 7) + 8;
        let hi = ((input[i + 16] * id).round() as i32).clamp(-8, 7) + 8;
        qs[i] = (lo as u8) | ((hi as u8) << 4);
    }

    BlockQ4_0 {
        d: f16::from_f32(d),
        qs,
    }
}

/// Quantize f32 values to Q4_1 block
pub fn quantize_q4_1(input: &[f32; 32]) -> BlockQ4_1 {
    // Find min and max values
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &x in input.iter() {
        min_val = min_val.min(x);
        max_val = max_val.max(x);
    }

    // Scale and minimum
    let d = (max_val - min_val) / 15.0;
    let m = min_val;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];
    for i in 0..16 {
        let lo = (((input[i] - m) * id).round() as i32).clamp(0, 15);
        let hi = (((input[i + 16] - m) * id).round() as i32).clamp(0, 15);
        qs[i] = (lo as u8) | ((hi as u8) << 4);
    }

    BlockQ4_1 {
        d: f16::from_f32(d),
        m: f16::from_f32(m),
        qs,
    }
}

/// Quantize f32 values to Q8_0 block
pub fn quantize_q8_0(input: &[f32; 32]) -> BlockQ8_0 {
    // Find max absolute value
    let mut amax = 0.0f32;
    for &x in input.iter() {
        amax = amax.max(x.abs());
    }

    // Scale factor: map to [-127, 127]
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (input[i] * id).round().clamp(-127.0, 127.0) as i8;
    }

    BlockQ8_0 {
        d: f16::from_f32(d),
        qs,
    }
}

// =============================================================================
// Batch Dequantization (for full tensors)
// =============================================================================

/// Dequantize a slice of Q4_0 blocks to f32
pub fn dequantize_q4_0_blocks(blocks: &[BlockQ4_0], output: &mut [f32]) {
    assert_eq!(
        output.len(),
        blocks.len() * 32,
        "Output slice must have space for all dequantized values"
    );

    for (i, block) in blocks.iter().enumerate() {
        let start = i * 32;
        let out_slice: &mut [f32; 32] = (&mut output[start..start + 32]).try_into().unwrap();
        dequantize_q4_0(block, out_slice);
    }
}

/// Dequantize a slice of Q8_0 blocks to f32
pub fn dequantize_q8_0_blocks(blocks: &[BlockQ8_0], output: &mut [f32]) {
    assert_eq!(
        output.len(),
        blocks.len() * 32,
        "Output slice must have space for all dequantized values"
    );

    for (i, block) in blocks.iter().enumerate() {
        let start = i * 32;
        let out_slice: &mut [f32; 32] = (&mut output[start..start + 32]).try_into().unwrap();
        dequantize_q8_0(block, out_slice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);

        let block = quantize_q4_0(&original);
        let mut decoded = [0.0f32; 32];
        dequantize_q4_0(&block, &mut decoded);

        // Check values are close (quantization has some error)
        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!(
                (o - d).abs() < 0.15,
                "Q4_0 roundtrip error too large: original={}, decoded={}",
                o,
                d
            );
        }
    }

    #[test]
    fn test_q4_1_roundtrip() {
        let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1 + 1.0);

        let block = quantize_q4_1(&original);
        let mut decoded = [0.0f32; 32];
        dequantize_q4_1(&block, &mut decoded);

        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!(
                (o - d).abs() < 0.15,
                "Q4_1 roundtrip error too large: original={}, decoded={}",
                o,
                d
            );
        }
    }

    #[test]
    fn test_q8_0_roundtrip() {
        let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);

        let block = quantize_q8_0(&original);
        let mut decoded = [0.0f32; 32];
        dequantize_q8_0(&block, &mut decoded);

        // Q8 should be very close
        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!(
                (o - d).abs() < 0.02,
                "Q8_0 roundtrip error too large: original={}, decoded={}",
                o,
                d
            );
        }
    }

    #[test]
    fn test_q4_0_zeros() {
        let original = [0.0f32; 32];
        let block = quantize_q4_0(&original);
        let mut decoded = [0.0f32; 32];
        dequantize_q4_0(&block, &mut decoded);

        for d in decoded.iter() {
            assert_eq!(*d, 0.0);
        }
    }

    #[test]
    fn test_q8_0_zeros() {
        let original = [0.0f32; 32];
        let block = quantize_q8_0(&original);
        let mut decoded = [0.0f32; 32];
        dequantize_q8_0(&block, &mut decoded);

        for d in decoded.iter() {
            assert_eq!(*d, 0.0);
        }
    }

    #[test]
    fn test_batch_dequantize_q4_0() {
        let input1: [f32; 32] = std::array::from_fn(|i| i as f32);
        let input2: [f32; 32] = std::array::from_fn(|i| (i + 32) as f32);

        let blocks = [quantize_q4_0(&input1), quantize_q4_0(&input2)];
        let mut output = vec![0.0f32; 64];

        dequantize_q4_0_blocks(&blocks, &mut output);

        // First 32 should be approximately 0-31
        // Second 32 should be approximately 32-63
        assert!(output[0] >= -1.0 && output[0] <= 1.0);
        assert!(output[31] >= 30.0);
        assert!(output[32] >= 30.0);
        assert!(output[63] >= 60.0);
    }
}
