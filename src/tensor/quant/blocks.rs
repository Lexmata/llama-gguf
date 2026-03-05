//! Quantization block structures matching llama.cpp exactly

use bytemuck::{Pod, Zeroable};
use half::f16;

// Basic Quantization Blocks (32 elements per block)

/// Q4_0: 4-bit quantization, 32 elements per block, 18 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    pub d: f16,       // scale
    pub qs: [u8; 16], // 32 x 4-bit values packed
}
impl BlockQ4_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 18;
}

/// Q4_1: 4-bit with min, 32 elements, 20 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; 16],
}
impl BlockQ4_1 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 20;
}

/// Q5_0: 5-bit quantization, 22 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_0 {
    pub d: f16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}
impl BlockQ5_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 22;
}

/// Q5_1: 5-bit with min, 24 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_1 {
    pub d: f16,
    pub m: f16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}
impl BlockQ5_1 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 24;
}

/// Q8_0: 8-bit quantization, 34 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; 32],
}
impl BlockQ8_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 34;
}

/// Q8_1: 8-bit with sum, 36 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_1 {
    pub d: f32,
    pub qs: [i8; 32],
}
impl BlockQ8_1 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 36;
}

// K-Quants (256 elements per block)

/// Q2_K: 2-bit, 84 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ2K {
    pub scales: [u8; 16],
    pub qs: [u8; 64],
    pub d: f16,
    pub dmin: f16,
}
impl BlockQ2K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 84;
}

/// Q3_K: 3-bit, 110 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: [u8; 12],
    pub d: f16,
}
impl BlockQ3K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 110;
}

/// Q4_K: 4-bit, 144 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}
impl BlockQ4K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 144;
}

/// Q5_K: 5-bit, 176 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}
impl BlockQ5K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 176;
}

/// Q6_K: 6-bit, 210 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: f16,
}
impl BlockQ6K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 210;
}

/// Q8_K: 8-bit K-quant, 292 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}
impl BlockQ8K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 292;
}

// IQ (importance-weighted) quantization blocks

/// IQ1_S: 1-bit importance quantization, 256 elements, 50 bytes
/// Super block with 8 sub-blocks of 32 elements each
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ1S {
    pub d: f16,           // super block scale
    pub qs: [u8; 32],     // 8 groups of 4 bytes: indices into grid
    pub qh: [u16; 8],     // high bits + signs
}
impl BlockIQ1S {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 50; // 2 + 32 + 16 = 50
}

/// IQ1_M: 1-bit importance quantization (mixture), 256 elements, 56 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ1M {
    pub qs: [u8; 32],      // 8 groups of 4 bytes: indices into grid
    pub qh: [u8; 16],      // high bits
    pub scales: [u8; 8],   // per-sub-block scales (packed)
}
impl BlockIQ1M {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 56; // 32 + 16 + 8 = 56
}

/// IQ2_XXS: 2-bit importance quantization (extra extra small), 256 elements, 66 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ2XXS {
    pub d: f16,            // super block scale
    pub qs: [u16; 32],     // 8 groups: each has 4 u16 grid indices
}
impl BlockIQ2XXS {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 66; // 2 + 64 = 66
}

/// IQ2_XS: 2-bit importance quantization (extra small), 256 elements, 74 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ2XS {
    pub d: f16,            // super block scale
    pub qs: [u16; 32],     // 8 groups of 4 u16 grid indices
    pub scales: [u8; 8],   // per-sub-block scales (4-bit packed)
}
impl BlockIQ2XS {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 74; // 2 + 64 + 8 = 74
}

/// IQ2_S: 2-bit importance quantization (small), 256 elements, 82 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ2S {
    pub d: f16,            // super block scale
    pub qs: [u8; 64],      // quantized values
    pub qh: [u8; 8],       // high bits
    pub scales: [u8; 8],   // per-sub-block scales
}
impl BlockIQ2S {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 82; // 2 + 64 + 8 + 8 = 82
}

/// IQ3_XXS: 3-bit importance quantization (extra extra small), 256 elements, 98 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ3XXS {
    pub d: f16,            // super block scale
    pub qs: [u8; 96],      // 3-bit quantized values (256 * 3 / 8 = 96) + high bits packed in
}
impl BlockIQ3XXS {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 98; // 2 + 96 = 98
}

/// IQ3_S: 3-bit importance quantization (small), 256 elements, 110 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ3S {
    pub d: f16,            // super block scale
    pub qs: [u8; 64],      // 2-bit values
    pub qh: [u8; 32],      // high bits
    pub signs: [u8; 8],    // sign bits
    pub scales: [u8; 4],   // per-sub-block scales
}
impl BlockIQ3S {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 110; // 2 + 64 + 32 + 8 + 4 = 110
}

/// IQ4_XS: 4-bit importance quantization (extra small), 256 elements, 136 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ4XS {
    pub d: f16,            // super block scale
    pub scales_h: u16,     // high bits of per-sub-block scales
    pub scales_l: [u8; 4], // low bits of per-sub-block scales (256/64 = 4)
    pub qs: [u8; 128],     // 4-bit quantized values (256/2 = 128)
}
impl BlockIQ4XS {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 136; // 2 + 2 + 4 + 128 = 136
}

/// IQ4_NL: 4-bit non-linear quantization, 32 elements, 18 bytes
/// Uses a non-linear lookup table for dequantization
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ4NL {
    pub d: f16,            // scale
    pub qs: [u8; 16],      // 4-bit quantized values (32/2 = 16)
}
impl BlockIQ4NL {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 18; // 2 + 16 = 18
}

// Compile-time size assertions
const _: () = {
    assert!(std::mem::size_of::<BlockQ4_0>() == BlockQ4_0::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ4_1>() == BlockQ4_1::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ5_0>() == BlockQ5_0::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ5_1>() == BlockQ5_1::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ8_0>() == BlockQ8_0::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ8_1>() == BlockQ8_1::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ2K>() == BlockQ2K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ3K>() == BlockQ3K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ4K>() == BlockQ4K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ5K>() == BlockQ5K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ6K>() == BlockQ6K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ8K>() == BlockQ8K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ1S>() == BlockIQ1S::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ1M>() == BlockIQ1M::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ2XXS>() == BlockIQ2XXS::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ2XS>() == BlockIQ2XS::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ2S>() == BlockIQ2S::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ3XXS>() == BlockIQ3XXS::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ3S>() == BlockIQ3S::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ4XS>() == BlockIQ4XS::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockIQ4NL>() == BlockIQ4NL::TYPE_SIZE);
};
