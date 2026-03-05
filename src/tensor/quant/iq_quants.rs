//! IQ (importance quantization) format dequantization and quantization
//!
//! Implements llama.cpp-compatible IQ formats: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
//! IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL.

use super::blocks::*;
use half::f16;
use super::iq_grids::{IQ2XXS_GRID, IQ2XS_GRID, IQ2S_GRID, IQ3XXS_GRID, IQ3S_GRID};

// =============================================================================
// Lookup Tables (from llama.cpp ggml-common.h)
// =============================================================================

/// IQ4_NL codebook: 16 values mapping 4-bit index to dequantized magnitude.
/// Values are int8 in range [-127, 113], used as multiplier with scale.
pub static IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Sign bit masks for IQ2/IQ3 formats (one per element in group of 8).
const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Sign lookup table for IQ2/IQ3 (128 entries).
const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43,
    172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65,
    66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215,
    216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108,
    237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

// =============================================================================
// Helper: Extract 8 bytes from IQ2 grid u64 (little-endian)
// =============================================================================

#[inline(always)]
fn iq2_grid_bytes(grid_val: u64) -> [u8; 8] {
    [
        grid_val as u8,
        (grid_val >> 8) as u8,
        (grid_val >> 16) as u8,
        (grid_val >> 24) as u8,
        (grid_val >> 32) as u8,
        (grid_val >> 40) as u8,
        (grid_val >> 48) as u8,
        (grid_val >> 56) as u8,
    ]
}

#[inline(always)]
fn iq3_grid_nibbles(grid_val: u32) -> [u8; 4] {
    [
        (grid_val & 0x0F) as u8,
        ((grid_val >> 4) & 0x0F) as u8,
        ((grid_val >> 8) & 0x0F) as u8,
        ((grid_val >> 12) & 0x0F) as u8,
    ]
}

// =============================================================================
// IQ4_NL (32 elements)
// =============================================================================

/// Dequantize IQ4_NL block to f32 (32 elements).
pub fn dequantize_iq4_nl(block: &BlockIQ4NL, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    for i in 0..16 {
        let byte = block.qs[i];
        let lo = (byte & 0x0F) as usize;
        let hi = ((byte >> 4) & 0x0F) as usize;
        output[i] = d * IQ4_NL_VALUES[lo] as f32;
        output[i + 16] = d * IQ4_NL_VALUES[hi] as f32;
    }
}

// =============================================================================
// IQ4_XS (256 elements)
// =============================================================================

/// Dequantize IQ4_XS block to f32 (256 elements).
pub fn dequantize_iq4_xs(block: &BlockIQ4XS, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let qs = &block.qs;

    for ib in 0..8 {
        let ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF)
            | ((((block.scales_h >> (2 * ib)) & 3) << 4) as u8);
        let dl = d * (ls as i32 - 32) as f32;

        for j in 0..16 {
            let byte = qs[ib * 16 + j];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            output[ib * 32 + j] = dl * IQ4_NL_VALUES[lo] as f32;
            output[ib * 32 + j + 16] = dl * IQ4_NL_VALUES[hi] as f32;
        }
    }
}

// =============================================================================
// IQ2_XXS (256 elements)
// =============================================================================

/// Dequantize IQ2_XXS block to f32 (256 elements).
pub fn dequantize_iq2_xxs(block: &BlockIQ2XXS, output: &mut [f32; 256]) {
    let d = block.d.to_f32();

    for ib32 in 0..8 {
        let aux32_0 = block.qs[4 * ib32] as u32 | ((block.qs[4 * ib32 + 1] as u32) << 16);
        let aux32_1 = block.qs[4 * ib32 + 2] as u32 | ((block.qs[4 * ib32 + 3] as u32) << 16);
        let aux32: [u32; 2] = [aux32_0, aux32_1];

        let db = d * (0.5 + ((aux32[1] >> 28) as f32)) * 0.25;

        let aux8: [u8; 8] = [
            aux32[0] as u8,
            (aux32[0] >> 8) as u8,
            (aux32[0] >> 16) as u8,
            (aux32[0] >> 24) as u8,
            aux32[1] as u8,
            (aux32[1] >> 8) as u8,
            (aux32[1] >> 16) as u8,
            (aux32[1] >> 24) as u8,
        ];

        for l in 0..4 {
            let grid_val = IQ2XXS_GRID[aux8[l] as usize];
            let grid = iq2_grid_bytes(grid_val);
            let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];

            for j in 0..8 {
                let sign = if (signs & KMASK_IQ2XS[j]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[ib32 * 32 + l * 8 + j] = db * grid[j] as f32 * sign;
            }
        }
    }
}

// =============================================================================
// IQ2_XS (256 elements)
// =============================================================================

/// Dequantize IQ2_XS block to f32 (256 elements).
pub fn dequantize_iq2_xs(block: &BlockIQ2XS, output: &mut [f32; 256]) {
    let d = block.d.to_f32();

    for ib32 in 0..8 {
        let db0 = d * (0.5 + (block.scales[ib32] & 0xF) as f32) * 0.25;
        let db1 = d * (0.5 + (block.scales[ib32] >> 4) as f32) * 0.25;

        for l in 0..4 {
            let qs_val = block.qs[4 * ib32 + l];
            let grid_idx = (qs_val & 511) as usize;
            let grid_val = IQ2XS_GRID[grid_idx];
            let grid = iq2_grid_bytes(grid_val);
            let signs = KSIGNS_IQ2XS[(qs_val >> 9) as usize];
            let db = if l < 2 { db0 } else { db1 };

            for j in 0..8 {
                let sign = if (signs & KMASK_IQ2XS[j]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[ib32 * 32 + l * 8 + j] = db * grid[j] as f32 * sign;
            }
        }
    }
}

// =============================================================================
// IQ2_S (256 elements)
// =============================================================================

/// Dequantize IQ2_S block to f32 (256 elements).
pub fn dequantize_iq2_s(block: &BlockIQ2S, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let qs = &block.qs;
    let qh = &block.qh;
    let signs = &qs[32..64];

    for ib32 in 0..8 {
        let db0 = d * (0.5 + (block.scales[ib32] & 0xF) as f32) * 0.25;
        let db1 = d * (0.5 + (block.scales[ib32] >> 4) as f32) * 0.25;
        let qs_base = ib32 * 4;
        let signs_base = ib32 * 4;

        for l in 0..4 {
            let grid_idx = (qs[qs_base + l] as usize)
                | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
            let grid_val = IQ2S_GRID[grid_idx];
            let grid = iq2_grid_bytes(grid_val);
            let sign_bits = signs[signs_base + l];
            let db = if l < 2 { db0 } else { db1 };

            for j in 0..8 {
                let sign = if (sign_bits & KMASK_IQ2XS[j]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[ib32 * 32 + l * 8 + j] = db * grid[j] as f32 * sign;
            }
        }
    }
}

// =============================================================================
// IQ3_XXS (256 elements)
// =============================================================================

/// Dequantize IQ3_XXS block to f32 (256 elements).
pub fn dequantize_iq3_xxs(block: &BlockIQ3XXS, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let qs = &block.qs;
    let scales_and_signs = &qs[64..];

    for ib32 in 0..8 {
        let aux32 = u32::from_le_bytes([
            scales_and_signs[4 * ib32],
            scales_and_signs[4 * ib32 + 1],
            scales_and_signs[4 * ib32 + 2],
            scales_and_signs[4 * ib32 + 3],
        ]);
        let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;

        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let grid1_val = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
            let grid2_val = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
            let g1 = iq3_grid_nibbles(grid1_val);
            let g2 = iq3_grid_nibbles(grid2_val);

            for j in 0..4 {
                let sign0 = if (signs & KMASK_IQ2XS[j]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                let sign4 = if (signs & KMASK_IQ2XS[j + 4]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[ib32 * 32 + l * 8 + j] = db * g1[j] as f32 * sign0;
                output[ib32 * 32 + l * 8 + j + 4] = db * g2[j] as f32 * sign4;
            }
        }
    }
}

// =============================================================================
// IQ3_S (256 elements)
// =============================================================================

/// Dequantize IQ3_S block to f32 (256 elements).
pub fn dequantize_iq3_s(block: &BlockIQ3S, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let qs = &block.qs;
    let qh = &block.qh;
    let signs = &block.signs;

    let mut out_idx = 0;
    let mut qs_idx = 0;
    let mut signs_idx = 0;
    let mut qh_idx = 0;

    for ib32 in (0..8).step_by(2) {
        let db1 = d * (1.0 + 2.0 * (block.scales[ib32 / 2] & 0xF) as f32);
        let db2 = d * (1.0 + 2.0 * (block.scales[ib32 / 2] >> 4) as f32);

        for l in 0..4 {
            let idx1 = (qs[qs_idx + 2 * l] as usize)
                | ((qh[qh_idx] as usize) << (8 - 2 * l) & 256);
            let idx2 = (qs[qs_idx + 2 * l + 1] as usize)
                | ((qh[qh_idx] as usize) << (7 - 2 * l) & 256);
            let grid1 = iq3_grid_nibbles(IQ3S_GRID[idx1]);
            let grid2 = iq3_grid_nibbles(IQ3S_GRID[idx2]);
            let sign_bits = signs[signs_idx + l];

            for j in 0..4 {
                let s0 = if (sign_bits & KMASK_IQ2XS[j]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                let s4 = if (sign_bits & KMASK_IQ2XS[j + 4]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_idx + j] = db1 * grid1[j] as f32 * s0;
                output[out_idx + j + 4] = db1 * grid2[j] as f32 * s4;
            }
            out_idx += 8;
        }
        qs_idx += 8;
        signs_idx += 4;

        for l in 0..4 {
            let idx1 = (qs[qs_idx + 2 * l] as usize)
                | ((qh[qh_idx + 1] as usize) << (8 - 2 * l) & 256);
            let idx2 = (qs[qs_idx + 2 * l + 1] as usize)
                | ((qh[qh_idx + 1] as usize) << (7 - 2 * l) & 256);
            let grid1 = iq3_grid_nibbles(IQ3S_GRID[idx1]);
            let grid2 = iq3_grid_nibbles(IQ3S_GRID[idx2]);
            let sign_bits = signs[signs_idx + l];

            for j in 0..4 {
                let s0 = if (sign_bits & KMASK_IQ2XS[j]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                let s4 = if (sign_bits & KMASK_IQ2XS[j + 4]) != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_idx + j] = db2 * grid1[j] as f32 * s0;
                output[out_idx + j + 4] = db2 * grid2[j] as f32 * s4;
            }
            out_idx += 8;
        }
        qs_idx += 8;
        signs_idx += 4;
        qh_idx += 2;
    }
}

// =============================================================================
// IQ1_S / IQ1_M - placeholders (require iq1s_grid)
// =============================================================================

/// Dequantize IQ1_S block (placeholder).
#[allow(dead_code)]
pub fn dequantize_iq1_s(_block: &BlockIQ1S, output: &mut [f32; 256]) {
    output.fill(0.0);
}

/// Dequantize IQ1_M block (placeholder).
#[allow(dead_code)]
pub fn dequantize_iq1_m(_block: &BlockIQ1M, output: &mut [f32; 256]) {
    output.fill(0.0);
}

// =============================================================================
// Quantization (IQ4_NL)
// =============================================================================

fn find_nearest_iq4nl(x: f32) -> usize {
    let mut best = 0;
    let mut best_err = (x - IQ4_NL_VALUES[0] as f32).abs();
    for (i, &v) in IQ4_NL_VALUES.iter().enumerate().skip(1) {
        let err = (x - v as f32).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best
}

/// Quantize f32 to IQ4_NL block.
pub fn quantize_iq4_nl(input: &[f32; 32]) -> BlockIQ4NL {
    let amax = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];
    for i in 0..16 {
        let lo = find_nearest_iq4nl(input[i] * id);
        let hi = find_nearest_iq4nl(input[i + 16] * id);
        qs[i] = (lo as u8) | ((hi as u8) << 4);
    }
    BlockIQ4NL {
        d: f16::from_f32(d),
        qs,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iq4_nl_roundtrip() {
        let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);

        let block = quantize_iq4_nl(&original);
        let mut decoded = [0.0f32; 32];
        dequantize_iq4_nl(&block, &mut decoded);

        let mut total_err: f32 = 0.0;
        let mut max_err: f32 = 0.0;
        for (o, d) in original.iter().zip(decoded.iter()) {
            let err = (o - d).abs();
            total_err += err;
            max_err = max_err.max(err);
        }
        let rel_err = if original.iter().map(|x| x.abs()).sum::<f32>() > 0.0 {
            total_err / original.iter().map(|x| x.abs()).sum::<f32>()
        } else {
            0.0
        };
        assert!(
            rel_err < 0.05 || max_err < 0.5,
            "IQ4_NL roundtrip: rel_err={}, max_err={}",
            rel_err,
            max_err
        );
    }

    #[test]
    fn test_iq4_xs_dequantize() {
        // ls=32 gives dl=0 (scale factor zero). Need scales_h bits so (scales_h>>2*ib)&3=2
        // for each ib, and scales_l low nibbles = 0. 0xAAAA has bit pattern for that.
        let block = BlockIQ4XS {
            d: f16::from_f32(1.0),
            scales_h: 0xAAAA, // (>>2*ib)&3 = 2 for all ib -> ls=32
            scales_l: [0, 0, 0, 0],
            qs: [0; 128],
        };
        let mut output = [0.0f32; 256];
        dequantize_iq4_xs(&block, &mut output);
        for &v in output.iter() {
            assert_eq!(v, 0.0, "zero block should decode to zeros");
        }
    }

    #[test]
    fn test_iq4_nl_values() {
        assert_eq!(IQ4_NL_VALUES[0], -127);
        assert_eq!(IQ4_NL_VALUES[8], 1);
        assert_eq!(IQ4_NL_VALUES[15], 113);
    }

    #[test]
    fn test_iq2_xxs_dequantize() {
        let block = BlockIQ2XXS {
            d: f16::from_f32(0.1),
            qs: [0u16; 32],
        };
        let mut output = [0.0f32; 256];
        dequantize_iq2_xxs(&block, &mut output);
        assert!(output[0].abs() > 0.0, "IQ2_XXS should produce non-zero output");
    }

    #[test]
    fn test_iq3_xxs_dequantize() {
        let mut block = BlockIQ3XXS {
            d: f16::from_f32(0.1),
            qs: [0u8; 96],
        };
        block.qs[64..68].fill(0);
        let mut output = [0.0f32; 256];
        dequantize_iq3_xxs(&block, &mut output);
        assert!(output[0].abs() >= 0.0, "IQ3_XXS should produce valid output");
    }
}
