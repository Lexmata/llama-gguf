//! Quantized KV cache for reduced VRAM usage
//!
//! Stores KV cache entries in INT8 or FP8 format instead of F32,
//! achieving ~2x memory reduction with minimal quality impact.

#[allow(unused_imports)]
use crate::tensor::{DType, Tensor};

/// KV cache storage format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KVCacheFormat {
    /// Full precision (default)
    F32,
    /// INT8 with per-head symmetric quantization: value = scale * int8_value
    Int8,
    /// FP8 E4M3 format (4 exponent, 3 mantissa bits) - good for inference
    Fp8E4M3,
    /// FP8 E5M2 format (5 exponent, 2 mantissa bits) - wider range, less precision
    Fp8E5M2,
}

impl KVCacheFormat {
    /// Bytes per element for this format
    const fn bytes_per_element(&self) -> usize {
        match self {
            KVCacheFormat::F32 => 4,
            KVCacheFormat::Int8 | KVCacheFormat::Fp8E4M3 | KVCacheFormat::Fp8E5M2 => 1,
        }
    }

    /// Whether this format uses per-head scales (INT8 only)
    const fn uses_scales(&self) -> bool {
        matches!(self, KVCacheFormat::Int8)
    }
}

/// Quantized KV cache that stores K/V in reduced precision
pub struct QuantizedKVCache {
    /// Quantized key data per layer - raw bytes in the chosen format
    pub k_data: Vec<Vec<u8>>,
    /// Quantized value data per layer
    pub v_data: Vec<Vec<u8>>,
    /// Per-head scale factors for INT8 (one per head per position per layer)
    /// Layout: [layer][head * max_seq_len + pos]
    pub k_scales: Vec<Vec<f32>>,
    pub v_scales: Vec<Vec<f32>>,
    /// Storage format
    pub format: KVCacheFormat,
    /// Current sequence length
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

impl QuantizedKVCache {
    /// Create a new quantized KV cache with the given dimensions and format
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        format: KVCacheFormat,
    ) -> Self {
        let elements_per_layer = num_kv_heads * max_seq_len * head_dim;
        let bytes_per_element = format.bytes_per_element();
        let layer_bytes = elements_per_layer * bytes_per_element;

        let k_data: Vec<Vec<u8>> = (0..num_layers)
            .map(|_| vec![0u8; layer_bytes])
            .collect();
        let v_data: Vec<Vec<u8>> = (0..num_layers)
            .map(|_| vec![0u8; layer_bytes])
            .collect();

        let scales_per_layer = if format.uses_scales() {
            num_kv_heads * max_seq_len
        } else {
            0
        };

        let k_scales: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; scales_per_layer])
            .collect();
        let v_scales: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; scales_per_layer])
            .collect();

        Self {
            k_data,
            v_data,
            k_scales,
            v_scales,
            format,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            num_layers,
        }
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for k in &mut self.k_data {
            k.fill(0);
        }
        for v in &mut self.v_data {
            v.fill(0);
        }
        for s in &mut self.k_scales {
            s.fill(0.0);
        }
        for s in &mut self.v_scales {
            s.fill(0.0);
        }
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.seq_len)
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let data_bytes: usize = self.k_data.iter().map(|v| v.len()).sum::<usize>()
            + self.v_data.iter().map(|v| v.len()).sum::<usize>();
        let scale_bytes: usize = self.k_scales.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.v_scales.iter().map(|v| v.len() * 4).sum::<usize>();
        data_bytes + scale_bytes
    }

    /// Write quantized K/V for one position
    ///
    /// `k_data` and `v_data` are `[num_kv_heads * head_dim]` each
    pub fn write_kv(
        &mut self,
        layer: usize,
        pos: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) {
        assert!(layer < self.num_layers);
        assert!(pos < self.max_seq_len);
        assert_eq!(k_data.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(v_data.len(), self.num_kv_heads * self.head_dim);

        let k_layer = &mut self.k_data[layer];
        let v_layer = &mut self.v_data[layer];

        for head in 0..self.num_kv_heads {
            let head_start = head * self.head_dim;
            let head_end = head_start + self.head_dim;
            let k_head = &k_data[head_start..head_end];
            let v_head = &v_data[head_start..head_end];

            let k_offset = (head * self.max_seq_len + pos) * self.head_dim
                * self.format.bytes_per_element();
            let v_offset = (head * self.max_seq_len + pos) * self.head_dim
                * self.format.bytes_per_element();

            match self.format {
                KVCacheFormat::F32 => {
                    for (i, &val) in k_head.iter().enumerate() {
                        let bytes = val.to_le_bytes();
                        k_layer[k_offset + i * 4..k_offset + (i + 1) * 4]
                            .copy_from_slice(&bytes);
                    }
                    for (i, &val) in v_head.iter().enumerate() {
                        let bytes = val.to_le_bytes();
                        v_layer[v_offset + i * 4..v_offset + (i + 1) * 4]
                            .copy_from_slice(&bytes);
                    }
                }
                KVCacheFormat::Int8 => {
                    let (k_quant, k_scale) = quantize_int8(k_head);
                    let (v_quant, v_scale) = quantize_int8(v_head);

                    let scale_idx = head * self.max_seq_len + pos;
                    self.k_scales[layer][scale_idx] = k_scale;
                    self.v_scales[layer][scale_idx] = v_scale;

                    for (i, &q) in k_quant.iter().enumerate() {
                        k_layer[k_offset + i] = q as u8;
                    }
                    for (i, &q) in v_quant.iter().enumerate() {
                        v_layer[v_offset + i] = q as u8;
                    }
                }
                KVCacheFormat::Fp8E4M3 => {
                    for (i, &val) in k_head.iter().enumerate() {
                        k_layer[k_offset + i] = quantize_fp8_e4m3(val);
                    }
                    for (i, &val) in v_head.iter().enumerate() {
                        v_layer[v_offset + i] = quantize_fp8_e4m3(val);
                    }
                }
                KVCacheFormat::Fp8E5M2 => {
                    for (i, &val) in k_head.iter().enumerate() {
                        k_layer[k_offset + i] = quantize_fp8_e5m2(val);
                    }
                    for (i, &val) in v_head.iter().enumerate() {
                        v_layer[v_offset + i] = quantize_fp8_e5m2(val);
                    }
                }
            }
        }
    }

    /// Dequantize and return key for one head at one position
    pub fn read_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        self.read_k_range(layer, head, pos, pos + 1)
    }

    /// Dequantize and return value for one head at one position
    pub fn read_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        self.read_v_range(layer, head, pos, pos + 1)
    }

    /// Dequantize key range for one head
    ///
    /// Returns `[end_pos - start_pos, head_dim]` as flat vec
    pub fn read_k_range(
        &self,
        layer: usize,
        head: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Vec<f32> {
        let k_layer = &self.k_data[layer];
        let bpe = self.format.bytes_per_element();
        let mut result = Vec::with_capacity((end_pos - start_pos) * self.head_dim);

        for pos in start_pos..end_pos {
            let offset = (head * self.max_seq_len + pos) * self.head_dim * bpe;

            for d in 0..self.head_dim {
                let val = match self.format {
                    KVCacheFormat::F32 => {
                        let byte_offset = offset + d * 4;
                        f32::from_le_bytes(
                            k_layer[byte_offset..byte_offset + 4]
                                .try_into()
                                .unwrap(),
                        )
                    }
                    KVCacheFormat::Int8 => {
                        let scale_idx = head * self.max_seq_len + pos;
                        let scale = self.k_scales[layer][scale_idx];
                        let q = k_layer[offset + d] as i8;
                        dequantize_int8(&[q], scale)[0]
                    }
                    KVCacheFormat::Fp8E4M3 => dequantize_fp8_e4m3(k_layer[offset + d]),
                    KVCacheFormat::Fp8E5M2 => dequantize_fp8_e5m2(k_layer[offset + d]),
                };
                result.push(val);
            }
        }
        result
    }

    /// Dequantize value range for one head
    ///
    /// Returns `[end_pos - start_pos, head_dim]` as flat vec
    pub fn read_v_range(
        &self,
        layer: usize,
        head: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Vec<f32> {
        let v_layer = &self.v_data[layer];
        let bpe = self.format.bytes_per_element();
        let mut result = Vec::with_capacity((end_pos - start_pos) * self.head_dim);

        for pos in start_pos..end_pos {
            let offset = (head * self.max_seq_len + pos) * self.head_dim * bpe;

            for d in 0..self.head_dim {
                let val = match self.format {
                    KVCacheFormat::F32 => {
                        let byte_offset = offset + d * 4;
                        f32::from_le_bytes(
                            v_layer[byte_offset..byte_offset + 4]
                                .try_into()
                                .unwrap(),
                        )
                    }
                    KVCacheFormat::Int8 => {
                        let scale_idx = head * self.max_seq_len + pos;
                        let scale = self.v_scales[layer][scale_idx];
                        let q = v_layer[offset + d] as i8;
                        dequantize_int8(&[q], scale)[0]
                    }
                    KVCacheFormat::Fp8E4M3 => dequantize_fp8_e4m3(v_layer[offset + d]),
                    KVCacheFormat::Fp8E5M2 => dequantize_fp8_e5m2(v_layer[offset + d]),
                };
                result.push(val);
            }
        }
        result
    }

    /// Shift cache left by `amount` positions (for sliding window)
    pub fn shift_left(&mut self, amount: usize) {
        if amount == 0 || amount >= self.seq_len {
            self.reset();
            return;
        }

        let new_len = self.seq_len - amount;
        let bpe = self.format.bytes_per_element();

        for layer_idx in 0..self.num_layers {
            let k_layer = &mut self.k_data[layer_idx];
            let v_layer = &mut self.v_data[layer_idx];

            for head in 0..self.num_kv_heads {
                for pos in 0..new_len {
                    let src_pos = pos + amount;
                    let src_offset = (head * self.max_seq_len + src_pos) * self.head_dim * bpe;
                    let dst_offset = (head * self.max_seq_len + pos) * self.head_dim * bpe;
                    let block_len = self.head_dim * bpe;

                    k_layer.copy_within(src_offset..src_offset + block_len, dst_offset);
                    v_layer.copy_within(src_offset..src_offset + block_len, dst_offset);
                }
            }

            if self.format.uses_scales() {
                let k_scales = &mut self.k_scales[layer_idx];
                let v_scales = &mut self.v_scales[layer_idx];

                for head in 0..self.num_kv_heads {
                    for pos in 0..new_len {
                        let src_idx = head * self.max_seq_len + (pos + amount);
                        let dst_idx = head * self.max_seq_len + pos;
                        k_scales[dst_idx] = k_scales[src_idx];
                        v_scales[dst_idx] = v_scales[src_idx];
                    }
                }
            }
        }

        self.seq_len = new_len;
    }

    /// Truncate cache to a specific length
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.seq_len {
            self.seq_len = new_len;
        }
    }
}

// --- Internal helpers ---

/// Symmetric INT8 quantization: scale = max(|x|) / 127
fn quantize_int8(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);

    let scale = if max_abs > 1e-10 {
        max_abs / 127.0
    } else {
        1.0
    };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| {
            let q = (x / scale).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect();

    (quantized, scale)
}

/// Dequantize INT8: value = scale * int8_value
fn dequantize_int8(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&q| (q as f32) * scale).collect()
}

/// Convert f32 to FP8 E4M3 (1 sign + 4 exp + 3 mantissa, bias 7)
fn quantize_fp8_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0xFF;
    }
    if value.is_infinite() {
        return if value > 0.0 { 0x7F } else { 0xFF };
    }
    if value == 0.0 {
        return 0x00;
    }

    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut mantissa = bits & 0x7F_FFFF;
    if exponent != -127 {
        mantissa |= 0x800_000;
    }

    let e4m3_exp = exponent + 7;

    if e4m3_exp > 15 {
        return (sign << 7) | 0x7E;
    }
    if (e4m3_exp > -3) && (e4m3_exp <= 0) {
        let shift_bits = (3 + e4m3_exp) as u32;
        let mask = 0x7u32 >> (0i32.saturating_sub(e4m3_exp) as u32);
        let e4m3_mantissa = ((mantissa >> (24 - shift_bits)) & mask) as u8;
        return (sign << 7) | e4m3_mantissa;
    }
    if e4m3_exp <= -3 {
        return sign << 7;
    }

    let e4m3_mantissa = ((mantissa >> 20) & 0x7) as u8;
    (sign << 7) | ((e4m3_exp as u8) << 3) | e4m3_mantissa
}

/// Convert FP8 E4M3 back to f32
fn dequantize_fp8_e4m3(value: u8) -> f32 {
    let bits = value;
    // Zero: S.0000.000
    if (bits & 0x7F) == 0 {
        return 0.0;
    }
    // NaN: S.1111.111
    if (bits & 0x7F) == 0x7F {
        return f32::NAN;
    }

    let sign = (bits >> 7) & 1;
    let e4m3_exp = (bits >> 3) & 0xF;
    let e4m3_mantissa = bits & 0x7;
    let exponent = (e4m3_exp as i32) - 7;
    let float_exp = (exponent + 127) as u32;

    let result = if e4m3_exp > 0 {
        (sign as u32) << 31 | float_exp << 23 | (e4m3_mantissa as u32) << 20
    } else {
        match e4m3_mantissa {
            m if m >= 4 => (sign as u32) << 31 | float_exp << 23 | ((m & 3) as u32) << 21,
            m if m > 1 => (sign as u32) << 31 | (float_exp - 1) << 23 | ((m & 1) as u32) << 22,
            1 => (sign as u32) << 31 | (float_exp - 2) << 23,
            _ => return f32::NAN,
        }
    };

    f32::from_bits(result)
}

/// Convert f32 to FP8 E5M2 (1 sign + 5 exp + 2 mantissa, bias 15)
fn quantize_fp8_e5m2(value: f32) -> u8 {
    if value.is_nan() {
        return 0xFF;
    }
    if value.is_infinite() {
        return if value > 0.0 { 0x7C } else { 0xFC };
    }
    if value == 0.0 {
        return 0x00;
    }

    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut mantissa = bits & 0x7F_FFFF;
    if exponent != -127 {
        mantissa |= 0x800_000;
    }

    let e5m2_exp = exponent + 15;

    if e5m2_exp > 31 {
        return (sign << 7) | 0x7C;
    }
    if (e5m2_exp >= -1) && (e5m2_exp <= 0) {
        let shift_bits = (2 + e5m2_exp) as u32;
        let mask = 0x3u32 >> (0i32.saturating_sub(e5m2_exp) as u32);
        let e5m2_mantissa = ((mantissa >> (24 - shift_bits)) & mask) as u8;
        return (sign << 7) | e5m2_mantissa;
    }
    if e5m2_exp < -1 {
        return sign << 7;
    }

    let e5m2_mantissa = ((mantissa >> 21) & 0x3) as u8;
    (sign << 7) | ((e5m2_exp as u8) << 2) | e5m2_mantissa
}

/// Convert FP8 E5M2 back to f32
fn dequantize_fp8_e5m2(value: u8) -> f32 {
    let bits = value;
    // Zero: S.00000.00
    if (bits & 0x7F) == 0 {
        return 0.0;
    }
    // Inf: S.11111.00
    if (bits & 0x7F) == 0x7C {
        return if (bits >> 7) != 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    // NaN: S.11111.{01,10,11}
    if (bits & 0x7F) >= 0x7D {
        return f32::NAN;
    }

    let sign = (bits >> 7) & 1;
    let e5m2_exp = (bits >> 2) & 0x1F;
    let e5m2_mantissa = bits & 0x3;
    let exponent = (e5m2_exp as i32) - 15;
    let float_exp = (exponent + 127) as u32;

    let result = if e5m2_exp > 0 {
        (sign as u32) << 31 | float_exp << 23 | (e5m2_mantissa as u32) << 21
    } else {
        match e5m2_mantissa {
            m if m >= 2 => (sign as u32) << 31 | float_exp << 23 | ((m & 1) as u32) << 22,
            1 => (sign as u32) << 31 | (float_exp - 1) << 23,
            _ => return f32::NAN,
        }
    };

    f32::from_bits(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
        let (quantized, scale) = quantize_int8(&data);
        let dequantized = dequantize_int8(&quantized, scale);
        for (orig, dec) in data.iter().zip(dequantized.iter()) {
            let rel_err = if orig.abs() > 1e-6 {
                (orig - dec).abs() / orig.abs()
            } else {
                (orig - dec).abs()
            };
            assert!(rel_err < 0.02, "orig={orig}, dec={dec}, rel_err={rel_err}");
        }
    }

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            0.0136719,
            448.0,
            2f32.powi(-6),
            2f32.powi(-9),
        ];
        for &val in &values {
            let q = quantize_fp8_e4m3(val);
            let d = dequantize_fp8_e4m3(q);
            if val == 0.0 {
                assert_eq!(d, 0.0, "zero roundtrip");
            } else if val.abs() < 1e-5 {
                assert!(d.abs() < 0.01, "small value {val} -> {d}");
            } else {
                let rel_err = (val - d).abs() / val.abs();
                assert!(rel_err < 0.05, "val={val}, d={d}, rel_err={rel_err}");
            }
        }
    }

    #[test]
    fn test_fp8_e5m2_roundtrip() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            57344.0,
            2f32.powi(-14),
            1.52588e-5,
        ];
        for &val in &values {
            let q = quantize_fp8_e5m2(val);
            let d = dequantize_fp8_e5m2(q);
            if val == 0.0 {
                assert_eq!(d, 0.0, "zero roundtrip");
            } else if val.abs() < 1e-5 {
                assert!(d.abs() < 0.01, "small value {val} -> {d}");
            } else {
                let rel_err = (val - d).abs() / val.abs();
                assert!(rel_err < 0.1, "val={val}, d={d}, rel_err={rel_err}");
            }
        }
    }

    #[test]
    fn test_quantized_kv_cache_basic() {
        let num_layers = 2;
        let num_kv_heads = 4;
        let max_seq_len = 16;
        let head_dim = 64;

        for format in [
            KVCacheFormat::Int8,
            KVCacheFormat::Fp8E4M3,
            KVCacheFormat::Fp8E5M2,
        ] {
            let mut cache =
                QuantizedKVCache::new(num_layers, num_kv_heads, max_seq_len, head_dim, format);

            let k_data: Vec<f32> = (0..num_kv_heads * head_dim)
                .map(|i| (i as f32) * 0.01 - 1.0)
                .collect();
            let v_data: Vec<f32> = (0..num_kv_heads * head_dim)
                .map(|i| (i as f32) * 0.02 - 0.5)
                .collect();

            cache.write_kv(0, 0, &k_data, &v_data);
            cache.seq_len = 1;

            let read_k = cache.read_k(0, 0, 0);
            let read_v = cache.read_v(0, 0, 0);

            assert_eq!(read_k.len(), head_dim);
            assert_eq!(read_v.len(), head_dim);

            let orig_k_head = &k_data[0..head_dim];
            let orig_v_head = &v_data[0..head_dim];

            let tol = match format {
                KVCacheFormat::Int8 => 0.15,
                KVCacheFormat::Fp8E4M3 | KVCacheFormat::Fp8E5M2 => 0.25,
                _ => 0.01,
            };
            for (a, b) in orig_k_head.iter().zip(read_k.iter()) {
                let rel_err = if a.abs() > 1e-6 {
                    (a - b).abs() / a.abs()
                } else {
                    (a - b).abs()
                };
                assert!(rel_err < tol, "k: orig={a}, read={b}");
            }
            for (a, b) in orig_v_head.iter().zip(read_v.iter()) {
                let rel_err = if a.abs() > 1e-6 {
                    (a - b).abs() / a.abs()
                } else {
                    (a - b).abs()
                };
                assert!(rel_err < tol, "v: orig={a}, read={b}");
            }
        }
    }

    #[test]
    fn test_memory_savings() {
        let num_layers = 4;
        let num_kv_heads = 32;
        let max_seq_len = 2048;
        let head_dim = 128;

        let f32_size = num_layers * 2 * (num_kv_heads * max_seq_len * head_dim * 4);

        let int8_cache =
            QuantizedKVCache::new(num_layers, num_kv_heads, max_seq_len, head_dim, KVCacheFormat::Int8);
        let fp8_cache =
            QuantizedKVCache::new(num_layers, num_kv_heads, max_seq_len, head_dim, KVCacheFormat::Fp8E4M3);

        let int8_size = int8_cache.memory_usage();
        let fp8_size = fp8_cache.memory_usage();

        assert!(int8_size < f32_size / 2 + f32_size / 4);
        assert!(fp8_size < f32_size / 2 + f32_size / 4);
    }

    #[test]
    fn test_shift_left() {
        let num_layers = 1;
        let num_kv_heads = 2;
        let max_seq_len = 8;
        let head_dim = 4;

        let mut cache = QuantizedKVCache::new(
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            KVCacheFormat::Int8,
        );

        for pos in 0..5 {
            let k_data: Vec<f32> = (0..num_kv_heads * head_dim)
                .map(|_| pos as f32)
                .collect();
            let v_data = k_data.clone();
            cache.write_kv(0, pos, &k_data, &v_data);
        }
        cache.seq_len = 5;

        cache.shift_left(2);

        assert_eq!(cache.seq_len, 3);

        for (i, pos) in (2..5).enumerate() {
            let read_k = cache.read_k(0, 0, i);
            let expected: Vec<f32> = (0..head_dim).map(|_| pos as f32).collect();
            for (a, b) in read_k.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 0.01, "pos {i}: expected {b}, got {a}");
            }
        }
    }
}
