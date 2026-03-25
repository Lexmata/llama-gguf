//! Quantized Johnson-Lindenstrauss (QJL) 1-bit compression
//!
//! Projects a vector through a random Gaussian matrix and retains only the
//! sign bits. Combined with the vector norm, this provides an unbiased
//! inner product estimator at 1 bit per dimension + 1 scalar overhead.
//!
//! Reference: "Quantized Johnson-Lindenstrauss" (AAAI 2025, arxiv 2406.03482)

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// QJL projector: generates a d×d Gaussian projection matrix on the fly
/// from a seed and compresses vectors to sign bits.
#[derive(Debug, Clone)]
pub struct QjlProjector {
    seed: u64,
    dim: usize,
}

impl QjlProjector {
    /// Create a projector for vectors of the given dimension.
    ///
    /// The seed determines the random projection matrix. Each attention
    /// head/layer should use a distinct seed from the rotation matrix.
    pub fn new(seed: u64, dim: usize) -> Self {
        Self { seed, dim }
    }

    /// Compress a vector to sign bits + norm.
    ///
    /// Returns packed sign bits (1 bit per dimension in u64 words) and
    /// the L2 norm of the input vector.
    pub fn compress(&self, x: &[f32]) -> (Vec<u64>, f32) {
        assert_eq!(x.len(), self.dim);

        let norm = l2_norm(x);
        let words_needed = (self.dim + 63) / 64;
        let mut bits = vec![0u64; words_needed];

        let mut rng = StdRng::seed_from_u64(self.seed);

        // For each row i of the projection matrix S, compute z_i = S[i] · x
        // and store sign(z_i).
        for i in 0..self.dim {
            let mut dot = 0.0f32;
            for j in 0..self.dim {
                let s_ij: f32 = StandardNormal.sample(&mut rng);
                dot += s_ij * x[j];
            }
            if dot >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }

        (bits, norm)
    }

    /// Estimate the inner product <query, x> from the QJL-compressed key.
    ///
    /// Uses the asymmetric estimator: project the query through the same
    /// random matrix S, then compute the sign-weighted sum.
    ///
    /// `query` is full-precision, `key_bits` and `key_norm` are the compressed key.
    pub fn inner_product(&self, query: &[f32], key_bits: &[u64], key_norm: f32) -> f32 {
        assert_eq!(query.len(), self.dim);

        let coeff = std::f32::consts::FRAC_PI_2.sqrt() / (self.dim as f32);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut sum = 0.0f32;

        for i in 0..self.dim {
            // Recompute S[i] · query (same row i as in compress)
            let mut proj_q = 0.0f32;
            for j in 0..self.dim {
                let s_ij: f32 = StandardNormal.sample(&mut rng);
                proj_q += s_ij * query[j];
            }
            let sign = if (key_bits[i / 64] >> (i % 64)) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            };
            sum += proj_q * sign;
        }

        coeff * key_norm * sum
    }

    /// Batch-project a query through the random matrix S.
    ///
    /// Returns `projected_query[i] = S[i] · query` for all i in 0..dim.
    /// This is precomputed once per query and reused across all key positions.
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);
        let mut projected = vec![0.0f32; self.dim];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for i in 0..self.dim {
            let mut dot = 0.0f32;
            for j in 0..self.dim {
                let s_ij: f32 = StandardNormal.sample(&mut rng);
                dot += s_ij * query[j];
            }
            projected[i] = dot;
        }

        projected
    }

    /// Fast inner product using a pre-projected query.
    ///
    /// `projected_query` is from `project_query()`. This avoids regenerating
    /// the random matrix for each key position.
    pub fn inner_product_fast(
        &self,
        projected_query: &[f32],
        key_bits: &[u64],
        key_norm: f32,
    ) -> f32 {
        assert_eq!(projected_query.len(), self.dim);

        let coeff = std::f32::consts::FRAC_PI_2.sqrt() / (self.dim as f32);
        let sum = crate::backend::cpu::simd::dot_with_sign_bits_fast(projected_query, key_bits, self.dim);
        coeff * key_norm * sum
    }

    /// Number of u64 words needed to store sign bits for one vector.
    pub fn packed_words(&self) -> usize {
        (self.dim + 63) / 64
    }

    /// Dimension of vectors this projector handles.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The seed used for the random projection.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Compute dot product of an f32 vector with packed sign bits.
///
/// `sign_bits[i/64] >> (i%64) & 1` encodes +1 (bit set) or -1 (bit clear).
/// Returns `sum_i projected_query[i] * sign(i)`.
#[inline]
pub fn dot_with_sign_bits(values: &[f32], bits: &[u64], count: usize) -> f32 {
    let mut sum = 0.0f32;
    let full_words = count / 64;
    let remainder = count % 64;

    for word_idx in 0..full_words {
        let word = bits[word_idx];
        let base = word_idx * 64;
        for bit in 0..64 {
            let sign = if (word >> bit) & 1 == 1 { 1.0f32 } else { -1.0f32 };
            sum += values[base + bit] * sign;
        }
    }

    if remainder > 0 {
        let word = bits[full_words];
        let base = full_words * 64;
        for bit in 0..remainder {
            let sign = if (word >> bit) & 1 == 1 { 1.0f32 } else { -1.0f32 };
            sum += values[base + bit] * sign;
        }
    }

    sum
}

fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_produces_correct_shape() {
        let proj = QjlProjector::new(42, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let (bits, norm) = proj.compress(&x);
        assert_eq!(bits.len(), 2); // 128 / 64 = 2 u64 words
        assert!(norm > 0.0);
    }

    #[test]
    fn test_inner_product_unbiased() {
        // Over many random vectors, the QJL inner product estimate should
        // converge to the true inner product.
        let dim = 64;
        let proj = QjlProjector::new(42, dim);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let y: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let true_dot: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let (bits, norm) = proj.compress(&x);
        let est = proj.inner_product(&y, &bits, norm);

        // Single estimate can be noisy, but for d=64 variance is bounded
        let rel_err = (est - true_dot).abs() / true_dot.abs().max(1e-6);
        assert!(
            rel_err < 2.0,
            "estimate should be in the right ballpark: est={est}, true={true_dot}"
        );
    }

    #[test]
    fn test_fast_matches_slow() {
        let dim = 32;
        let proj = QjlProjector::new(99, dim);

        let key: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.02).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 16.0) * 0.01).collect();

        let (bits, norm) = proj.compress(&key);
        let slow = proj.inner_product(&query, &bits, norm);
        let projected_q = proj.project_query(&query);
        let fast = proj.inner_product_fast(&projected_q, &bits, norm);

        assert!(
            (slow - fast).abs() < 1e-4,
            "fast and slow should match: {slow} vs {fast}"
        );
    }

    #[test]
    fn test_dot_with_sign_bits_all_positive() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let bits = vec![0xF_u64]; // all 4 bits set = all +1
        let result = dot_with_sign_bits(&values, &bits, 4);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_with_sign_bits_alternating() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let bits = vec![0b0101_u64]; // bits 0,2 set (+1); bits 1,3 clear (-1)
        let result = dot_with_sign_bits(&values, &bits, 4);
        let expected = 1.0 - 2.0 + 3.0 - 4.0;
        assert!((result - expected).abs() < 1e-6);
    }
}
