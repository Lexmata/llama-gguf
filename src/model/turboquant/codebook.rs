//! Lloyd-Max optimal scalar quantizer for TurboQuant
//!
//! After random Hadamard rotation, each coordinate of a unit-norm vector
//! follows approximately N(0, 1/d). The codebook centroids are precomputed
//! Lloyd-Max solutions for N(0, 1) scaled by 1/sqrt(d).

/// Precomputed Lloyd-Max codebook for scalar quantization.
///
/// Centroids and decision boundaries are stored pre-scaled by `1/sqrt(d)`
/// so that quantize/dequantize operate directly on rotated coordinates.
#[derive(Debug, Clone)]
pub struct Codebook {
    centroids: Vec<f32>,
    boundaries: Vec<f32>,
    bits: u8,
    dim: usize,
}

/// Lloyd-Max centroids for N(0,1) at various bit widths.
/// These are the standard optimal quantizer reconstruction levels.
const LLOYD_MAX_1BIT: &[f32] = &[-0.7978845608, 0.7978845608];

const LLOYD_MAX_2BIT: &[f32] = &[-1.510_232_6, -0.452_842_7, 0.452_842_7, 1.510_232_6];

const LLOYD_MAX_3BIT: &[f32] = &[
    -2.152_164_5,
    -1.344_183_8,
    -0.756_130_3,
    -0.245_340_4,
    0.245_340_4,
    0.756_130_3,
    1.344_183_8,
    2.152_164_5,
];

/// Decision boundaries for N(0,1) Lloyd-Max quantizer (midpoints between centroids).
const BOUNDARIES_1BIT: &[f32] = &[0.0];

const BOUNDARIES_2BIT: &[f32] = &[-0.981_537_65, 0.0, 0.981_537_65];

const BOUNDARIES_3BIT: &[f32] = &[
    -1.748_174_15,
    -1.050_157_05,
    -0.500_735_35,
    0.0,
    0.500_735_35,
    1.050_157_05,
    1.748_174_15,
];

impl Codebook {
    /// Build a codebook for the given head dimension and bit width.
    ///
    /// Centroids are scaled by `1/sqrt(dim)` to match the distribution of
    /// rotated unit-norm vectors.
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!((1..=3).contains(&bits), "supported bit widths: 1, 2, 3");
        let inv_sqrt_d = 1.0 / (dim as f32).sqrt();

        let (raw_centroids, raw_boundaries) = match bits {
            1 => (LLOYD_MAX_1BIT, BOUNDARIES_1BIT),
            2 => (LLOYD_MAX_2BIT, BOUNDARIES_2BIT),
            3 => (LLOYD_MAX_3BIT, BOUNDARIES_3BIT),
            _ => unreachable!(),
        };

        let centroids: Vec<f32> = raw_centroids.iter().map(|&c| c * inv_sqrt_d).collect();
        let boundaries: Vec<f32> = raw_boundaries.iter().map(|&b| b * inv_sqrt_d).collect();

        Self {
            centroids,
            boundaries,
            bits,
            dim,
        }
    }

    /// Quantize a single scalar to its nearest centroid index.
    #[inline]
    pub fn quantize(&self, val: f32) -> u8 {
        // Binary search through boundaries. For 2-8 entries this is
        // faster than a linear scan due to branch prediction.
        let mut idx = 0u8;
        for &b in &self.boundaries {
            if val >= b {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Look up the centroid value for a quantization index.
    #[inline]
    pub fn dequantize(&self, idx: u8) -> f32 {
        self.centroids[idx as usize]
    }

    /// Number of bits per element.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of centroids (2^bits).
    #[inline]
    pub fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// The head dimension this codebook was built for.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw centroid slice (useful for CUDA upload).
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Raw boundary slice.
    pub fn boundaries(&self) -> &[f32] {
        &self.boundaries
    }

    /// Quantize a full vector, packing indices into bytes.
    ///
    /// For b=1: 8 values per byte. For b=2: 4 values per byte. For b=3: packed
    /// sequentially (3 bits each, so 8 values = 3 bytes with padding).
    pub fn quantize_vector(&self, data: &[f32], out: &mut Vec<u8>) {
        out.clear();
        match self.bits {
            1 => {
                for chunk in data.chunks(8) {
                    let mut byte = 0u8;
                    for (i, &val) in chunk.iter().enumerate() {
                        byte |= self.quantize(val) << i;
                    }
                    out.push(byte);
                }
            }
            2 => {
                for chunk in data.chunks(4) {
                    let mut byte = 0u8;
                    for (i, &val) in chunk.iter().enumerate() {
                        byte |= self.quantize(val) << (i * 2);
                    }
                    out.push(byte);
                }
            }
            3 => {
                // Pack 3-bit indices: every 8 indices = 3 bytes (24 bits)
                for chunk in data.chunks(8) {
                    let mut bits_acc: u32 = 0;
                    for (i, &val) in chunk.iter().enumerate() {
                        bits_acc |= (self.quantize(val) as u32) << (i * 3);
                    }
                    out.push((bits_acc & 0xFF) as u8);
                    out.push(((bits_acc >> 8) & 0xFF) as u8);
                    out.push(((bits_acc >> 16) & 0xFF) as u8);
                }
            }
            _ => unreachable!(),
        }
    }

    /// Dequantize a packed vector into f32 values.
    pub fn dequantize_vector(&self, packed: &[u8], count: usize, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(count);
        match self.bits {
            1 => {
                let mut remaining = count;
                for &byte in packed {
                    let n = remaining.min(8);
                    for i in 0..n {
                        let idx = (byte >> i) & 1;
                        out.push(self.dequantize(idx));
                    }
                    remaining -= n;
                }
            }
            2 => {
                let mut remaining = count;
                for &byte in packed {
                    let n = remaining.min(4);
                    for i in 0..n {
                        let idx = (byte >> (i * 2)) & 0x3;
                        out.push(self.dequantize(idx));
                    }
                    remaining -= n;
                }
            }
            3 => {
                let mut remaining = count;
                for triple in packed.chunks(3) {
                    let bits_acc = (triple[0] as u32)
                        | ((triple.get(1).copied().unwrap_or(0) as u32) << 8)
                        | ((triple.get(2).copied().unwrap_or(0) as u32) << 16);
                    let n = remaining.min(8);
                    for i in 0..n {
                        let idx = ((bits_acc >> (i * 3)) & 0x7) as u8;
                        out.push(self.dequantize(idx));
                    }
                    remaining -= n;
                }
            }
            _ => unreachable!(),
        }
    }

    /// Compute dot product between a full-precision query (in rotated space)
    /// and a packed quantized key vector, without dequantizing the full key.
    ///
    /// This is the fast path: `sum_i query[i] * centroid[key_idx[i]]`.
    pub fn dot_with_packed(&self, query: &[f32], packed: &[u8], count: usize) -> f32 {
        let mut sum = 0.0f32;
        match self.bits {
            1 => {
                let mut pos = 0;
                for &byte in packed {
                    let n = (count - pos).min(8);
                    for i in 0..n {
                        let idx = (byte >> i) & 1;
                        sum += query[pos + i] * self.centroids[idx as usize];
                    }
                    pos += n;
                }
            }
            2 => {
                let mut pos = 0;
                for &byte in packed {
                    let n = (count - pos).min(4);
                    for i in 0..n {
                        let idx = (byte >> (i * 2)) & 0x3;
                        sum += query[pos + i] * self.centroids[idx as usize];
                    }
                    pos += n;
                }
            }
            3 => {
                let mut pos = 0;
                for triple in packed.chunks(3) {
                    let bits_acc = (triple[0] as u32)
                        | ((triple.get(1).copied().unwrap_or(0) as u32) << 8)
                        | ((triple.get(2).copied().unwrap_or(0) as u32) << 16);
                    let n = (count - pos).min(8);
                    for i in 0..n {
                        let idx = ((bits_acc >> (i * 3)) & 0x7) as u8;
                        sum += query[pos + i] * self.centroids[idx as usize];
                    }
                    pos += n;
                }
            }
            _ => unreachable!(),
        }
        sum
    }

    /// Bytes required to store `count` packed indices.
    pub fn packed_bytes(&self, count: usize) -> usize {
        match self.bits {
            1 => (count + 7) / 8,
            2 => (count + 3) / 4,
            3 => ((count + 7) / 8) * 3,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_1bit_roundtrip() {
        let cb = Codebook::new(128, 1);
        assert_eq!(cb.num_centroids(), 2);
        assert_eq!(cb.quantize(-0.1), 0);
        assert_eq!(cb.quantize(0.1), 1);
        let c0 = cb.dequantize(0);
        let c1 = cb.dequantize(1);
        assert!(c0 < 0.0);
        assert!(c1 > 0.0);
        assert!((c0 + c1).abs() < 1e-6, "symmetric centroids");
    }

    #[test]
    fn test_codebook_2bit_ordering() {
        let cb = Codebook::new(128, 2);
        assert_eq!(cb.num_centroids(), 4);
        for i in 0..3 {
            assert!(
                cb.dequantize(i) < cb.dequantize(i + 1),
                "centroids must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_vector_quantize_roundtrip() {
        let cb = Codebook::new(128, 2);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.001).collect();
        let mut packed = Vec::new();
        cb.quantize_vector(&data, &mut packed);
        let mut deq = Vec::new();
        cb.dequantize_vector(&packed, 128, &mut deq);
        assert_eq!(deq.len(), 128);
        for (&orig, &dec) in data.iter().zip(deq.iter()) {
            assert!(
                (orig - dec).abs() < 0.1,
                "orig={orig}, dec={dec}"
            );
        }
    }

    #[test]
    fn test_dot_with_packed_consistency() {
        let cb = Codebook::new(64, 2);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.005).collect();
        let query: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let mut packed = Vec::new();
        cb.quantize_vector(&data, &mut packed);
        let mut deq = Vec::new();
        cb.dequantize_vector(&packed, 64, &mut deq);

        let direct_dot: f32 = query.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();
        let fast_dot = cb.dot_with_packed(&query, &packed, 64);
        assert!(
            (direct_dot - fast_dot).abs() < 1e-5,
            "direct={direct_dot}, fast={fast_dot}"
        );
    }

    #[test]
    fn test_3bit_packing() {
        let cb = Codebook::new(16, 3);
        let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.02).collect();
        let mut packed = Vec::new();
        cb.quantize_vector(&data, &mut packed);
        assert_eq!(packed.len(), cb.packed_bytes(16));
        let mut deq = Vec::new();
        cb.dequantize_vector(&packed, 16, &mut deq);
        assert_eq!(deq.len(), 16);
    }
}
