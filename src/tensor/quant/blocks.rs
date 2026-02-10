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
};
