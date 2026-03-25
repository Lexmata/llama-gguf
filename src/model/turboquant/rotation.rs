//! Randomized Hadamard Transform for TurboQuant preconditioning
//!
//! Applies a random sign flip followed by a normalized Walsh-Hadamard transform.
//! This is O(d log d) vs O(d^2) for a dense random orthogonal matrix, and
//! produces the same statistical properties needed by TurboQuant: after rotation,
//! coordinates of a unit-norm vector follow approximately N(0, 1/d).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Randomized Hadamard rotation: diagonal sign-flip D followed by
/// the normalized Walsh-Hadamard transform H.
///
/// The combined transform `(1/sqrt(d)) * H * D` is orthogonal and
/// deterministic given a seed. Each attention head/layer uses a
/// distinct seed so rotation matrices are independent.
///
/// For non-power-of-2 dimensions, the input is zero-padded to the next
/// power of 2 internally so the WHT butterfly can run. The extra
/// dimensions are discarded on output (forward) or zeroed on input
/// (inverse), preserving the mathematical properties.
#[derive(Debug, Clone)]
pub struct HadamardRotation {
    signs: Vec<f32>,
    /// The original (caller-visible) dimension.
    dim: usize,
    /// The internal padded dimension (always a power of 2).
    padded_dim: usize,
}

fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

impl HadamardRotation {
    /// Create a rotation for the given dimension and seed.
    ///
    /// The dimension does not need to be a power of 2 — non-power-of-2
    /// dimensions are handled by zero-padding internally.
    pub fn new(seed: u64, dim: usize) -> Self {
        assert!(dim >= 2, "dim must be >= 2");

        let padded_dim = next_power_of_two(dim);

        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rng.r#gen::<bool>() { 1.0 } else { -1.0 })
            .collect();

        Self { signs, dim, padded_dim }
    }

    /// Forward rotation: out = (1/sqrt(d_pad)) * H * D * pad(x)
    ///
    /// Input `x` has `dim` elements (the original dimension).
    /// Output `out` has `padded_dim` elements — the full rotated vector
    /// must be preserved for lossless inverse.
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.padded_dim);

        for i in 0..self.dim {
            out[i] = x[i] * self.signs[i];
        }
        for i in self.dim..self.padded_dim {
            out[i] = 0.0;
        }

        fast_walsh_hadamard(out);

        let norm = 1.0 / (self.padded_dim as f32).sqrt();
        for v in out.iter_mut() {
            *v *= norm;
        }
    }

    /// Inverse rotation: recovers the original `dim`-element vector from a
    /// `padded_dim`-element rotated vector.
    pub fn rotate_inverse(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.padded_dim);
        assert_eq!(out.len(), self.dim);

        let mut buf = vec![0.0f32; self.padded_dim];
        let scale = (self.padded_dim as f32).sqrt();
        for (b, &v) in buf.iter_mut().zip(x.iter()) {
            *b = v * scale;
        }

        fast_walsh_hadamard(&mut buf);
        let inv_d = 1.0 / self.padded_dim as f32;
        for i in 0..self.dim {
            out[i] = buf[i] * inv_d * self.signs[i];
        }
    }

    /// Dimension of the transform (caller-visible, may be non-power-of-2).
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The internal padded dimension (always power of 2).
    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// The random sign vector (useful for CUDA upload).
    pub fn signs(&self) -> &[f32] {
        &self.signs
    }
}

/// In-place unnormalized Walsh-Hadamard transform (butterfly network).
///
/// After this, each output element is a sum/difference of all inputs.
/// Caller must apply `1/sqrt(d)` normalization for an orthogonal transform.
fn fast_walsh_hadamard(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut half = 1;
    while half < n {
        for block_start in (0..n).step_by(half * 2) {
            for i in 0..half {
                let a = data[block_start + i];
                let b = data[block_start + i + half];
                data[block_start + i] = a + b;
                data[block_start + i + half] = a - b;
            }
        }
        half *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let rot = HadamardRotation::new(42, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
        let mut rotated = vec![0.0f32; rot.padded_dim()];
        let mut recovered = vec![0.0f32; 128];
        rot.rotate(&x, &mut rotated);
        rot.rotate_inverse(&rotated, &mut recovered);
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "roundtrip failed: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_norm_preservation() {
        let rot = HadamardRotation::new(123, 64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.64).collect();
        let orig_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut rotated = vec![0.0f32; rot.padded_dim()];
        rot.rotate(&x, &mut rotated);
        let rot_norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-3,
            "norm not preserved: {orig_norm} vs {rot_norm}"
        );
    }

    #[test]
    fn test_deterministic() {
        let rot1 = HadamardRotation::new(999, 32);
        let rot2 = HadamardRotation::new(999, 32);
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut out1 = vec![0.0f32; rot1.padded_dim()];
        let mut out2 = vec![0.0f32; rot2.padded_dim()];
        rot1.rotate(&x, &mut out1);
        rot2.rotate(&x, &mut out2);
        assert_eq!(out1, out2, "same seed must produce identical results");
    }

    #[test]
    fn test_different_seeds_differ() {
        let rot1 = HadamardRotation::new(1, 32);
        let rot2 = HadamardRotation::new(2, 32);
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut out1 = vec![0.0f32; rot1.padded_dim()];
        let mut out2 = vec![0.0f32; rot2.padded_dim()];
        rot1.rotate(&x, &mut out1);
        rot2.rotate(&x, &mut out2);
        assert_ne!(out1, out2, "different seeds should produce different rotations");
    }

    #[test]
    fn test_non_power_of_two_roundtrip() {
        for dim in [3, 5, 7, 10, 13, 17, 33, 65, 80, 96, 100] {
            let rot = HadamardRotation::new(42, dim);
            assert_eq!(rot.dim(), dim);
            assert!(rot.padded_dim().is_power_of_two());
            assert!(rot.padded_dim() >= dim);

            let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let mut rotated = vec![0.0f32; rot.padded_dim()];
            let mut recovered = vec![0.0f32; dim];
            rot.rotate(&x, &mut rotated);
            rot.rotate_inverse(&rotated, &mut recovered);
            for (a, b) in x.iter().zip(recovered.iter()) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "roundtrip failed for dim={dim}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_non_power_of_two_norm_preservation() {
        let rot = HadamardRotation::new(123, 80);
        let x: Vec<f32> = (0..80).map(|i| (i as f32) * 0.02 - 0.8).collect();
        let orig_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut rotated = vec![0.0f32; rot.padded_dim()];
        rot.rotate(&x, &mut rotated);
        let rot_norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-3,
            "norm not preserved for dim=80: {orig_norm} vs {rot_norm}"
        );
    }
}
