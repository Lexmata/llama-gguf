//! TurboQuant top-level quantize/dequantize and attention scoring
//!
//! Combines codebook quantization (PolarQuant) with optional QJL correction
//! to implement both TurboQuant_MSE and TurboQuant_prod variants.

use super::codebook::Codebook;
use super::qjl::QjlProjector;
use super::rotation::HadamardRotation;

/// Configuration for a single TurboQuant layer/head.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    pub bits: u8,
    pub use_qjl: bool,
    pub dim: usize,
}

/// Complete TurboQuant engine for one attention head.
///
/// Holds the rotation, codebook, and optional QJL projector
/// needed to compress/decompress a single head's K or V cache entries.
///
/// All internal operations happen in `padded_dim` space (power of 2) while
/// the caller-visible dimension is `dim` (which may be non-power-of-2).
#[derive(Debug, Clone)]
pub struct TurboQuantEngine {
    rotation: HadamardRotation,
    codebook: Codebook,
    qjl: Option<QjlProjector>,
    dim: usize,
    padded_dim: usize,
}

/// Compressed representation of a single key or value vector.
pub struct CompressedVector {
    /// Packed codebook indices (PolarQuant component)
    pub packed_indices: Vec<u8>,
    /// QJL sign bits for the quantization residual (only for TurboQuant_prod)
    pub qjl_bits: Option<Vec<u64>>,
    /// Norm of the residual (only for TurboQuant_prod)
    pub residual_norm: Option<f32>,
}

impl TurboQuantEngine {
    /// Create a new engine for the given head config.
    ///
    /// `rotation_seed` and `qjl_seed` must differ; typically derived from
    /// `layer_idx * n_heads + head_idx` with different base offsets.
    pub fn new(config: &TurboQuantConfig, rotation_seed: u64, qjl_seed: u64) -> Self {
        let rotation = HadamardRotation::new(rotation_seed, config.dim);
        let padded_dim = rotation.padded_dim();
        let codebook = Codebook::new(padded_dim, config.bits);
        let qjl = if config.use_qjl {
            Some(QjlProjector::new(qjl_seed, padded_dim))
        } else {
            None
        };

        Self {
            rotation,
            codebook,
            qjl,
            dim: config.dim,
            padded_dim,
        }
    }

    /// Compress a key/value vector.
    ///
    /// 1. Rotate: x' = R * x
    /// 2. Scalar quantize x' → packed indices
    /// 3. If QJL enabled: compute residual r = x' - dequant(x'), compress with QJL
    pub fn compress(&self, x: &[f32], rotated_buf: &mut Vec<f32>, deq_buf: &mut Vec<f32>) -> CompressedVector {
        rotated_buf.resize(self.padded_dim, 0.0);
        self.rotation.rotate(x, rotated_buf);

        let mut packed_indices = Vec::new();
        self.codebook.quantize_vector(rotated_buf, &mut packed_indices);

        let (qjl_bits, residual_norm) = if let Some(ref proj) = self.qjl {
            deq_buf.clear();
            self.codebook
                .dequantize_vector(&packed_indices, self.padded_dim, deq_buf);

            let residual: Vec<f32> = rotated_buf
                .iter()
                .zip(deq_buf.iter())
                .map(|(&r, &d)| r - d)
                .collect();

            let (bits, norm) = proj.compress(&residual);
            (Some(bits), Some(norm))
        } else {
            (None, None)
        };

        CompressedVector {
            packed_indices,
            qjl_bits,
            residual_norm,
        }
    }

    /// Compute the attention score <query, key> where key is compressed.
    ///
    /// **TurboQuant_MSE** (no QJL): returns `codebook_dot(R*query, packed_key)`
    /// **TurboQuant_prod** (with QJL): adds the QJL correction term for unbiased estimation
    pub fn attention_score(
        &self,
        query: &[f32],
        compressed: &CompressedVector,
        rotated_query_buf: &mut Vec<f32>,
    ) -> f32 {
        rotated_query_buf.resize(self.padded_dim, 0.0);
        self.rotation.rotate(query, rotated_query_buf);

        let polar_score = self
            .codebook
            .dot_with_packed(rotated_query_buf, &compressed.packed_indices, self.padded_dim);

        if let (Some(proj), Some(bits), Some(norm)) =
            (&self.qjl, &compressed.qjl_bits, compressed.residual_norm)
        {
            let qjl_correction = proj.inner_product(rotated_query_buf, bits, norm);
            polar_score + qjl_correction
        } else {
            polar_score
        }
    }

    /// Batch attention: score one query against multiple compressed keys.
    ///
    /// For TurboQuant_prod, pre-projects the query through QJL once.
    pub fn attention_scores(
        &self,
        query: &[f32],
        keys: &[CompressedVector],
        rotated_query_buf: &mut Vec<f32>,
        scores: &mut Vec<f32>,
    ) {
        rotated_query_buf.resize(self.padded_dim, 0.0);
        self.rotation.rotate(query, rotated_query_buf);

        let projected_query = self.qjl.as_ref().map(|proj| proj.project_query(rotated_query_buf));

        scores.clear();
        scores.reserve(keys.len());

        for key in keys {
            let polar_score = self
                .codebook
                .dot_with_packed(rotated_query_buf, &key.packed_indices, self.padded_dim);

            let score = if let (Some(proj_q), Some(proj), Some(bits), Some(norm)) = (
                &projected_query,
                &self.qjl,
                &key.qjl_bits,
                key.residual_norm,
            ) {
                let _ = proj;
                let qjl_correction = proj.inner_product_fast(proj_q, bits, norm);
                polar_score + qjl_correction
            } else {
                polar_score
            };

            scores.push(score);
        }
    }

    /// Bytes per compressed vector entry.
    pub fn bytes_per_entry(&self) -> usize {
        let codebook_bytes = self.codebook.packed_bytes(self.padded_dim);
        if self.qjl.is_some() {
            let qjl_words = (self.padded_dim + 63) / 64;
            codebook_bytes + qjl_words * 8 + 4 // +4 for residual norm f32
        } else {
            codebook_bytes
        }
    }

    /// Bits per element in the KV cache (relative to original dimension).
    pub fn bits_per_element(&self) -> f32 {
        let total_bits = self.bytes_per_entry() as f32 * 8.0;
        total_bits / self.dim as f32
    }

    /// The padded dimension used internally for rotation/quantization.
    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Access the internal rotation (for CUDA upload of sign vectors).
    pub fn rotation(&self) -> &HadamardRotation {
        &self.rotation
    }

    /// Access the internal codebook (for CUDA upload of centroids).
    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    /// Access the QJL projector if present.
    pub fn qjl(&self) -> Option<&QjlProjector> {
        self.qjl.as_ref()
    }
}

/// Compute softmax in-place over a slice of attention scores.
pub fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    let inv_sum = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize, bits: u8, use_qjl: bool) -> TurboQuantConfig {
        TurboQuantConfig { bits, use_qjl, dim }
    }

    #[test]
    fn test_mse_compress_decompress() {
        let config = make_config(64, 2, false);
        let engine = TurboQuantEngine::new(&config, 100, 200);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let mut rot_buf = Vec::new();
        let mut deq_buf = Vec::new();

        let compressed = engine.compress(&x, &mut rot_buf, &mut deq_buf);
        assert!(compressed.qjl_bits.is_none());
        assert!(compressed.residual_norm.is_none());
    }

    #[test]
    fn test_prod_compress_has_qjl() {
        let config = make_config(64, 2, true);
        let engine = TurboQuantEngine::new(&config, 100, 200);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let mut rot_buf = Vec::new();
        let mut deq_buf = Vec::new();

        let compressed = engine.compress(&x, &mut rot_buf, &mut deq_buf);
        assert!(compressed.qjl_bits.is_some());
        assert!(compressed.residual_norm.is_some());
    }

    #[test]
    fn test_attention_score_mse_direction() {
        let config = make_config(64, 2, false);
        let engine = TurboQuantEngine::new(&config, 100, 200);

        let q: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        let k_similar: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 + 0.001).collect();
        let k_opposite: Vec<f32> = (0..64).map(|i| -(i as f32) * 0.02).collect();

        let mut rot_buf = Vec::new();
        let mut deq_buf = Vec::new();
        let mut rot_q_buf = Vec::new();

        let c_similar = engine.compress(&k_similar, &mut rot_buf, &mut deq_buf);
        let c_opposite = engine.compress(&k_opposite, &mut rot_buf, &mut deq_buf);

        let s1 = engine.attention_score(&q, &c_similar, &mut rot_q_buf);
        let s2 = engine.attention_score(&q, &c_opposite, &mut rot_q_buf);

        assert!(
            s1 > s2,
            "similar key should score higher: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_batch_scores_match_individual() {
        let config = make_config(32, 2, false);
        let engine = TurboQuantEngine::new(&config, 100, 200);

        let q: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let keys_raw: Vec<Vec<f32>> = (0..5)
            .map(|k| (0..32).map(|i| ((i + k) as f32) * 0.05).collect())
            .collect();

        let mut rot_buf = Vec::new();
        let mut deq_buf = Vec::new();
        let mut rot_q_buf = Vec::new();

        let compressed: Vec<CompressedVector> = keys_raw
            .iter()
            .map(|k| engine.compress(k, &mut rot_buf, &mut deq_buf))
            .collect();

        let individual: Vec<f32> = compressed
            .iter()
            .map(|c| engine.attention_score(&q, c, &mut rot_q_buf))
            .collect();

        let mut batch = Vec::new();
        engine.attention_scores(&q, &compressed, &mut rot_q_buf, &mut batch);

        for (a, b) in individual.iter().zip(batch.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "batch should match individual: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_softmax() {
        let mut scores = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_bytes_per_entry() {
        let config = make_config(128, 2, false);
        let engine = TurboQuantEngine::new(&config, 1, 2);
        let pd = engine.padded_dim(); // 128 is already pow2
        assert_eq!(pd, 128);
        assert_eq!(engine.bytes_per_entry(), pd / 4);

        let config_qjl = make_config(128, 2, true);
        let engine_qjl = TurboQuantEngine::new(&config_qjl, 1, 2);
        let expected = pd / 4 + (pd + 63) / 64 * 8 + 4;
        assert_eq!(engine_qjl.bytes_per_entry(), expected);
    }

    #[test]
    fn test_non_power_of_two_dim() {
        let config = make_config(80, 2, false);
        let engine = TurboQuantEngine::new(&config, 100, 200);
        assert_eq!(engine.dim(), 80);
        assert_eq!(engine.padded_dim(), 128);

        let x: Vec<f32> = (0..80).map(|i| (i as f32 - 40.0) * 0.01).collect();
        let mut rot_buf = Vec::new();
        let mut deq_buf = Vec::new();
        let mut rot_q_buf = Vec::new();

        let compressed = engine.compress(&x, &mut rot_buf, &mut deq_buf);
        let score = engine.attention_score(&x, &compressed, &mut rot_q_buf);
        assert!(score > 0.0, "self-attention score should be positive");
    }
}
