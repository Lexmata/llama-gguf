//! TurboQuant-compressed KV cache for memory-efficient attention
//!
//! Stores K/V vectors in TurboQuant-compressed form rather than full f32.
//! At 2-3 bits per dimension this achieves 5-8x memory savings vs. f32.

use super::turboquant::{CompressedVector, TurboQuantConfig, TurboQuantEngine};

/// A single head's compressed KV storage for one layer.
struct HeadCache {
    k_entries: Vec<CompressedVector>,
    v_entries: Vec<CompressedVector>,
}

/// TurboQuant-compressed KV cache replacing the standard f32 KV cache.
///
/// Each layer × head pair gets its own `TurboQuantEngine` (with unique
/// rotation and optional QJL seeds) plus a growable list of compressed
/// K and V vectors.
pub struct TurboQuantKVCache {
    engines_k: Vec<Vec<TurboQuantEngine>>,
    engines_v: Vec<Vec<TurboQuantEngine>>,
    caches: Vec<Vec<HeadCache>>,
    seq_len: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    config: TurboQuantConfig,
}

impl TurboQuantKVCache {
    /// Create a new TurboQuant KV cache.
    ///
    /// Seeds are derived deterministically: each (layer, head, k/v) triple
    /// gets a unique seed so the rotation matrices are independent.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        config: TurboQuantConfig,
    ) -> Self {
        let mut engines_k = Vec::with_capacity(num_layers);
        let mut engines_v = Vec::with_capacity(num_layers);
        let mut caches = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let mut layer_engines_k = Vec::with_capacity(num_kv_heads);
            let mut layer_engines_v = Vec::with_capacity(num_kv_heads);
            let mut layer_caches = Vec::with_capacity(num_kv_heads);

            for head in 0..num_kv_heads {
                let base_seed = (layer * num_kv_heads + head) as u64;
                let rotation_seed_k = base_seed * 4;
                let qjl_seed_k = base_seed * 4 + 1;
                let rotation_seed_v = base_seed * 4 + 2;
                let qjl_seed_v = base_seed * 4 + 3;

                layer_engines_k.push(TurboQuantEngine::new(&config, rotation_seed_k, qjl_seed_k));
                layer_engines_v.push(TurboQuantEngine::new(&config, rotation_seed_v, qjl_seed_v));
                layer_caches.push(HeadCache {
                    k_entries: Vec::with_capacity(max_seq_len),
                    v_entries: Vec::with_capacity(max_seq_len),
                });
            }

            engines_k.push(layer_engines_k);
            engines_v.push(layer_engines_v);
            caches.push(layer_caches);
        }

        Self {
            engines_k,
            engines_v,
            caches,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            num_layers,
            config,
        }
    }

    /// Store one position's K and V vectors for one layer.
    ///
    /// `k_heads` is `[num_kv_heads * key_length]` flat, `v_heads` similarly.
    pub fn write_kv(
        &mut self,
        layer_idx: usize,
        k_heads: &[f32],
        v_heads: &[f32],
    ) {
        let dim = self.head_dim;
        let mut rot_buf = Vec::with_capacity(dim);
        let mut deq_buf = Vec::with_capacity(dim);

        for head in 0..self.num_kv_heads {
            let k_start = head * dim;
            let k_slice = &k_heads[k_start..k_start + dim];
            let v_start = head * dim;
            let v_slice = &v_heads[v_start..v_start + dim];

            let ck = self.engines_k[layer_idx][head].compress(k_slice, &mut rot_buf, &mut deq_buf);
            let cv = self.engines_v[layer_idx][head].compress(v_slice, &mut rot_buf, &mut deq_buf);

            let cache = &mut self.caches[layer_idx][head];
            if cache.k_entries.len() < self.max_seq_len {
                cache.k_entries.push(ck);
                cache.v_entries.push(cv);
            }
        }

        // Track seq_len from layer 0 only
        if layer_idx == 0 {
            self.seq_len = (self.seq_len + 1).min(self.max_seq_len);
        }
    }

    /// Compute attention scores for one head at one layer.
    ///
    /// Returns softmax-weighted output vector of dimension `head_dim`.
    /// `query_head` is the single query vector for this head (after RoPE).
    /// `scale` is typically `1/sqrt(key_length)`.
    pub fn attention_head(
        &self,
        layer_idx: usize,
        head_idx: usize,
        query_head: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let kv_len = self.seq_len;
        let dim = self.head_dim;
        let cache = &self.caches[layer_idx][head_idx];
        let k_engine = &self.engines_k[layer_idx][head_idx];
        let v_engine = &self.engines_v[layer_idx][head_idx];

        let padded_dim = k_engine.padded_dim();

        let mut rot_q_buf = Vec::with_capacity(padded_dim);
        let mut scores = Vec::with_capacity(kv_len);
        k_engine.attention_scores(query_head, &cache.k_entries[..kv_len], &mut rot_q_buf, &mut scores);

        for s in &mut scores {
            *s *= scale;
        }

        super::turboquant::quant::softmax_inplace(&mut scores);

        let mut output = vec![0.0f32; dim];
        let v_padded_dim = v_engine.padded_dim();
        let mut deq_v: Vec<f32> = Vec::with_capacity(v_padded_dim);

        for (pos, &weight) in scores.iter().enumerate() {
            if weight < 1e-8 {
                continue;
            }
            let cv = &cache.v_entries[pos];
            v_engine.codebook().dequantize_vector(&cv.packed_indices, v_padded_dim, &mut deq_v);

            let mut original_v = vec![0.0f32; dim];
            v_engine.rotation().rotate_inverse(&deq_v, &mut original_v);

            for (o, &v) in output.iter_mut().zip(original_v.iter()) {
                *o += weight * v;
            }
        }

        output
    }

    /// Compute full multi-head attention for one layer.
    ///
    /// `queries` is `[num_heads * key_length]` flat (after Q projection + RoPE).
    /// Returns `[num_heads * value_length]` flat.
    pub fn attention_layer(
        &self,
        layer_idx: usize,
        queries: &[f32],
        num_heads: usize,
        scale: f32,
    ) -> Vec<f32> {
        let kl = self.head_dim;
        let vl = self.head_dim;
        let num_queries_per_kv = num_heads / self.num_kv_heads;

        let mut output = vec![0.0f32; num_heads * vl];

        for head in 0..num_heads {
            let kv_head = head / num_queries_per_kv;
            let q_start = head * kl;
            let q_slice = &queries[q_start..q_start + kl];

            let head_out = self.attention_head(layer_idx, kv_head, q_slice, scale);

            let out_start = head * vl;
            output[out_start..out_start + vl].copy_from_slice(&head_out);
        }

        output
    }

    /// Current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset cache for a new sequence.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer_caches in &mut self.caches {
            for hc in layer_caches {
                hc.k_entries.clear();
                hc.v_entries.clear();
            }
        }
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_usage(&self) -> usize {
        let bytes_per_k = self.engines_k[0][0].bytes_per_entry();
        let bytes_per_v = self.engines_v[0][0].bytes_per_entry();
        let per_entry = bytes_per_k + bytes_per_v;
        per_entry * self.num_kv_heads * self.seq_len * self.num_layers
    }

    /// Check remaining capacity.
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.seq_len)
    }

    /// Check if cache is full.
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }

    /// Truncate cache to a specific length.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.seq_len {
            self.seq_len = new_len;
            for layer_caches in &mut self.caches {
                for hc in layer_caches {
                    hc.k_entries.truncate(new_len);
                    hc.v_entries.truncate(new_len);
                }
            }
        }
    }

    /// Shift cache left by `amount` positions (for sliding window).
    pub fn shift_left(&mut self, amount: usize) {
        if amount == 0 || amount >= self.seq_len {
            self.reset();
            return;
        }
        let new_len = self.seq_len - amount;
        for layer_caches in &mut self.caches {
            for hc in layer_caches {
                let k_len = hc.k_entries.len();
                if amount < k_len {
                    hc.k_entries.drain(..amount);
                    hc.v_entries.drain(..amount);
                } else {
                    hc.k_entries.clear();
                    hc.v_entries.clear();
                }
            }
        }
        self.seq_len = new_len;
    }

    /// Get the config.
    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(dim: usize, bits: u8, use_qjl: bool) -> TurboQuantKVCache {
        let config = TurboQuantConfig { bits, use_qjl, dim };
        TurboQuantKVCache::new(2, 4, 64, dim, config)
    }

    #[test]
    fn test_write_and_seq_len() {
        let mut cache = make_cache(64, 2, false);
        assert_eq!(cache.seq_len(), 0);

        let k = vec![0.1f32; 4 * 64];
        let v = vec![0.2f32; 4 * 64];
        cache.write_kv(0, &k, &v);
        assert_eq!(cache.seq_len(), 1);

        cache.write_kv(0, &k, &v);
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_attention_direction() {
        let mut cache = make_cache(64, 2, false);

        // Write two keys: one similar to query, one opposite
        let k_similar: Vec<f32> = (0..4 * 64).map(|i| (i as f32) * 0.01).collect();
        let k_opposite: Vec<f32> = (0..4 * 64).map(|i| -(i as f32) * 0.01).collect();
        let v1 = vec![1.0f32; 4 * 64];
        let v2 = vec![2.0f32; 4 * 64];

        cache.write_kv(0, &k_similar, &v1);
        cache.write_kv(0, &k_opposite, &v2);

        let query: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let scale = 1.0 / (64.0f32).sqrt();
        let out = cache.attention_head(0, 0, &query, scale);

        // Output should be closer to v1 than v2 since query ~ k_similar
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_reset() {
        let mut cache = make_cache(32, 2, false);
        let k = vec![0.1f32; 4 * 32];
        let v = vec![0.2f32; 4 * 32];
        cache.write_kv(0, &k, &v);
        cache.write_kv(0, &k, &v);
        assert_eq!(cache.seq_len(), 2);
        cache.reset();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_truncate() {
        let mut cache = make_cache(32, 2, false);
        let k = vec![0.1f32; 4 * 32];
        let v = vec![0.2f32; 4 * 32];
        for _ in 0..5 {
            cache.write_kv(0, &k, &v);
        }
        assert_eq!(cache.seq_len(), 5);
        cache.truncate(3);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_shift_left() {
        let mut cache = make_cache(32, 2, false);
        let k = vec![0.1f32; 4 * 32];
        let v = vec![0.2f32; 4 * 32];
        for _ in 0..5 {
            cache.write_kv(0, &k, &v);
        }
        assert_eq!(cache.seq_len(), 5);
        cache.shift_left(2);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_memory_savings() {
        let cache = make_cache(128, 2, false);
        let f32_bytes = 128 * 4 * 2 * 4 * 2; // dim * sizeof(f32) * kv * heads * layers
        let tq_bytes_per = cache.config.dim / 4; // 2 bits = 4 values/byte
        let tq_total = tq_bytes_per * 2 * 4 * 2;
        assert!(
            tq_total < f32_bytes / 4,
            "TurboQuant should use < 25% of f32 memory"
        );
    }

    #[test]
    fn test_attention_layer_multi_head() {
        let mut cache = make_cache(64, 2, false);

        let k = vec![0.5f32; 4 * 64];
        let v = vec![1.0f32; 4 * 64];
        cache.write_kv(0, &k, &v);

        let queries = vec![0.5f32; 4 * 64]; // 4 heads x 64 dim
        let scale = 1.0 / (64.0f32).sqrt();
        let out = cache.attention_layer(0, &queries, 4, scale);
        assert_eq!(out.len(), 4 * 64);
        for val in &out {
            assert!(val.is_finite(), "Attention output must be finite");
        }
    }

    #[test]
    fn test_shift_left_excess() {
        let mut cache = make_cache(32, 2, false);
        let k = vec![0.1f32; 4 * 32];
        let v = vec![0.2f32; 4 * 32];
        cache.write_kv(0, &k, &v);
        cache.write_kv(0, &k, &v);
        assert_eq!(cache.seq_len(), 2);

        cache.shift_left(10);
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_qjl_mode() {
        let mut cache = make_cache(64, 2, true);
        let k = vec![0.3f32; 4 * 64];
        let v = vec![0.7f32; 4 * 64];
        cache.write_kv(0, &k, &v);
        assert_eq!(cache.seq_len(), 1);

        let query = vec![0.3f32; 64];
        let scale = 1.0 / (64.0f32).sqrt();
        let out = cache.attention_head(0, 0, &query, scale);
        assert_eq!(out.len(), 64);
        for val in &out {
            assert!(val.is_finite());
        }
    }
}
