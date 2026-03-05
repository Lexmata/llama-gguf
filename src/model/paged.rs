//! Paged attention for efficient KV cache memory management
//!
//! Implements vLLM-style paged attention where the KV cache is divided into
//! fixed-size blocks that are allocated on demand and can be shared across
//! sequences (for prefix caching with copy-on-write).

use std::sync::atomic::{AtomicUsize, Ordering};

/// Physical block ID
pub type BlockId = usize;

/// Block size in tokens (each block stores this many KV entries per head)
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Manages a pool of physical blocks.
pub struct PageAllocator {
    /// Total number of physical blocks
    num_blocks: usize,
    /// Free block IDs (stack-based for O(1) alloc/free)
    free_blocks: Vec<BlockId>,
    /// Reference count per block (for copy-on-write)
    ref_counts: Vec<AtomicUsize>,
}

impl PageAllocator {
    /// Create a new page allocator with all blocks initially free.
    pub fn new(num_blocks: usize) -> Self {
        let free_blocks: Vec<BlockId> = (0..num_blocks).collect();
        let ref_counts: Vec<AtomicUsize> = (0..num_blocks).map(|_| AtomicUsize::new(0)).collect();
        Self {
            num_blocks,
            free_blocks,
            ref_counts,
        }
    }

    /// Allocate a block from the free list. Returns None if no blocks available.
    pub fn allocate(&mut self) -> Option<BlockId> {
        let block_id = self.free_blocks.pop()?;
        self.ref_counts[block_id].store(1, Ordering::SeqCst);
        Some(block_id)
    }

    /// Free a block. Decrements ref count; pushes to free list when it reaches zero.
    pub fn free(&mut self, block_id: BlockId) {
        if block_id >= self.num_blocks {
            return;
        }
        let prev = self.ref_counts[block_id].fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.free_blocks.push(block_id);
        }
    }

    /// Increment reference count for copy-on-write sharing.
    pub fn increment_ref(&self, block_id: BlockId) {
        if block_id < self.num_blocks {
            self.ref_counts[block_id].fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Get the current reference count of a block.
    pub fn ref_count(&self, block_id: BlockId) -> usize {
        if block_id >= self.num_blocks {
            return 0;
        }
        self.ref_counts[block_id].load(Ordering::SeqCst)
    }

    /// Number of free blocks available.
    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of blocks currently in use (ref count > 0).
    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }
}

/// Per-sequence mapping from logical to physical blocks.
pub struct BlockTable {
    /// Logical block index -> physical BlockId
    entries: Vec<Option<BlockId>>,
    /// Number of tokens stored
    num_tokens: usize,
    /// Block size
    block_size: usize,
}

impl BlockTable {
    /// Create an empty block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Number of allocated blocks.
    pub fn num_blocks(&self) -> usize {
        self.entries.len()
    }

    /// Number of tokens stored.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Map logical block index to physical block ID.
    pub fn logical_to_physical(&self, logical_idx: usize) -> Option<BlockId> {
        self.entries.get(logical_idx).and_then(|e| *e)
    }

    /// Add a new block mapping.
    pub fn append_block(&mut self, block_id: BlockId) {
        self.entries.push(Some(block_id));
    }

    /// Convert token position to (logical_block_idx, offset_within_block).
    pub fn token_to_block(&self, token_pos: usize) -> (usize, usize) {
        if self.block_size == 0 {
            return (0, 0);
        }
        let logical_block_idx = token_pos / self.block_size;
        let offset_within_block = token_pos % self.block_size;
        (logical_block_idx, offset_within_block)
    }

    /// Update the token count.
    pub fn set_num_tokens(&mut self, n: usize) {
        self.num_tokens = n;
    }
}

/// The main memory pool that holds all KV data.
pub struct PagedKVPool {
    /// Physical KV data per layer: flat storage [block_id][head][offset][dim]
    /// Layout: block_id * block_stride + head * (block_size * head_dim) + offset * head_dim + d
    k_pool: Vec<Vec<f32>>,
    v_pool: Vec<Vec<f32>>,
    /// Page allocator
    allocator: PageAllocator,
    /// Configuration
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_blocks: usize,
}

impl PagedKVPool {
    /// Size of one block in floats (all heads).
    fn block_stride(&self) -> usize {
        self.num_kv_heads * self.block_size * self.head_dim
    }

    /// Offset for a single (block_id, offset, head) position.
    fn block_offset(&self, block_id: BlockId, offset: usize, head: usize) -> usize {
        block_id * self.block_stride() + head * (self.block_size * self.head_dim) + offset * self.head_dim
    }

    /// Create a new paged KV pool.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> Self {
        let block_stride = num_kv_heads * block_size * head_dim;
        let layer_size = num_blocks * block_stride;

        let k_pool: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0; layer_size])
            .collect();
        let v_pool: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0; layer_size])
            .collect();

        Self {
            k_pool,
            v_pool,
            allocator: PageAllocator::new(num_blocks),
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
        }
    }

    /// Allocate N blocks from the pool.
    pub fn allocate_blocks(&mut self, count: usize) -> Option<Vec<BlockId>> {
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let block_id = self.allocator.allocate()?;
            blocks.push(block_id);
        }
        Some(blocks)
    }

    /// Free blocks back to the pool.
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) {
        for &block_id in block_ids {
            self.allocator.free(block_id);
        }
    }

    /// Write one KV position. k and v must have length head_dim.
    pub fn write_kv(
        &mut self,
        layer: usize,
        block_id: BlockId,
        offset: usize,
        head: usize,
        k: &[f32],
        v: &[f32],
    ) {
        if layer >= self.num_layers
            || head >= self.num_kv_heads
            || offset >= self.block_size
            || k.len() != self.head_dim
            || v.len() != self.head_dim
        {
            return;
        }
        let base = self.block_offset(block_id, offset, head);
        self.k_pool[layer][base..base + self.head_dim].copy_from_slice(k);
        self.v_pool[layer][base..base + self.head_dim].copy_from_slice(v);
    }

    /// Read K for one position. Returns slice of length head_dim.
    pub fn read_k(
        &self,
        layer: usize,
        block_id: BlockId,
        offset: usize,
        head: usize,
    ) -> &[f32] {
        if layer >= self.num_layers
            || head >= self.num_kv_heads
            || offset >= self.block_size
        {
            return &[];
        }
        let base = self.block_offset(block_id, offset, head);
        &self.k_pool[layer][base..base + self.head_dim]
    }

    /// Read V for one position. Returns slice of length head_dim.
    pub fn read_v(
        &self,
        layer: usize,
        block_id: BlockId,
        offset: usize,
        head: usize,
    ) -> &[f32] {
        if layer >= self.num_layers
            || head >= self.num_kv_heads
            || offset >= self.block_size
        {
            return &[];
        }
        let base = self.block_offset(block_id, offset, head);
        &self.v_pool[layer][base..base + self.head_dim]
    }

    /// Copy-on-write: copy all layer data from src block to dst block.
    pub fn copy_block(&mut self, src: BlockId, dst: BlockId) {
        let block_stride = self.block_stride();
        let src_base = src * block_stride;
        let dst_base = dst * block_stride;
        for layer in 0..self.num_layers {
            let src_slice = self.k_pool[layer][src_base..src_base + block_stride].to_vec();
            self.k_pool[layer][dst_base..dst_base + block_stride].copy_from_slice(&src_slice);
            let src_slice = self.v_pool[layer][src_base..src_base + block_stride].to_vec();
            self.v_pool[layer][dst_base..dst_base + block_stride].copy_from_slice(&src_slice);
        }
    }

    /// Total pool memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let floats_per_layer = self.num_blocks * self.block_stride();
        let total_floats = floats_per_layer * self.num_layers * 2; // K and V
        total_floats * std::mem::size_of::<f32>()
    }

    /// Number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free()
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Expose allocator for PagedSequence (needs allocate/free/increment_ref).
    #[allow(dead_code)]
    pub(crate) fn allocator_mut(&mut self) -> &mut PageAllocator {
        &mut self.allocator
    }

    /// Expose allocator for ref counting.
    #[allow(dead_code)]
    pub(crate) fn allocator(&self) -> &PageAllocator {
        &self.allocator
    }
}

/// Per-sequence state for paged attention.
pub struct PagedSequence {
    /// Block table mapping logical blocks to physical
    pub block_table: BlockTable,
    /// Sequence ID
    pub seq_id: usize,
    /// Current token count
    pub num_tokens: usize,
}

impl PagedSequence {
    /// Create a new paged sequence.
    pub fn new(seq_id: usize, block_size: usize) -> Self {
        Self {
            block_table: BlockTable::new(block_size),
            seq_id,
            num_tokens: 0,
        }
    }

    /// Append a KV entry for one (layer, head). Allocates a new block if needed.
    pub fn append_token(
        &mut self,
        pool: &mut PagedKVPool,
        layer: usize,
        head: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), &'static str> {
        let (logical_block_idx, offset_within_block) =
            self.block_table.token_to_block(self.num_tokens);

        // Allocate new block if needed
        while logical_block_idx >= self.block_table.num_blocks() {
            let blocks = pool
                .allocate_blocks(1)
                .ok_or("No free blocks in pool")?;
            let block_id = blocks[0];
            self.block_table.append_block(block_id);
        }

        let block_id = self
            .block_table
            .logical_to_physical(logical_block_idx)
            .ok_or("Missing block mapping")?;

        pool.write_kv(layer, block_id, offset_within_block, head, k, v);

        Ok(())
    }

    /// Advance to the next token position after writing all (layer, head) for the current token.
    pub fn advance_token(&mut self) {
        self.num_tokens += 1;
        self.block_table.set_num_tokens(self.num_tokens);
    }

    /// Gather all K/V for the given layer and head into contiguous buffers for attention.
    /// Returns (k_buf, v_buf) each of size num_tokens * head_dim.
    pub fn get_kv_for_attention(
        &self,
        pool: &PagedKVPool,
        layer: usize,
        head: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let num_tokens = self.num_tokens;
        let head_dim = pool.head_dim;

        let mut k_buf = vec![0.0; num_tokens * head_dim];
        let mut v_buf = vec![0.0; num_tokens * head_dim];

        for token_pos in 0..num_tokens {
            let (logical_block_idx, offset) = self.block_table.token_to_block(token_pos);
            if let Some(block_id) = self.block_table.logical_to_physical(logical_block_idx) {
                let k_slice = pool.read_k(layer, block_id, offset, head);
                let v_slice = pool.read_v(layer, block_id, offset, head);
                if k_slice.len() == head_dim && v_slice.len() == head_dim {
                    k_buf[token_pos * head_dim..(token_pos + 1) * head_dim]
                        .copy_from_slice(k_slice);
                    v_buf[token_pos * head_dim..(token_pos + 1) * head_dim]
                        .copy_from_slice(v_slice);
                }
            }
        }

        (k_buf, v_buf)
    }

}

impl BlockTable {
    /// Clear all block mappings (caller must free physical blocks separately).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.num_tokens = 0;
    }
}

impl PagedSequence {
    /// Release all blocks back to the pool.
    pub fn free(&mut self, pool: &mut PagedKVPool) {
        let block_ids: Vec<BlockId> = (0..self.block_table.num_blocks())
            .filter_map(|i| self.block_table.logical_to_physical(i))
            .collect();
        pool.free_blocks(&block_ids);
        self.block_table.clear();
        self.num_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_allocator_basic() {
        let mut alloc = PageAllocator::new(4);
        assert_eq!(alloc.num_free(), 4);
        assert_eq!(alloc.num_used(), 0);

        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        assert_eq!(alloc.num_free(), 2);
        assert_eq!(alloc.num_used(), 2);
        assert_eq!(alloc.ref_count(b0), 1);
        assert_eq!(alloc.ref_count(b1), 1);

        alloc.increment_ref(b0);
        assert_eq!(alloc.ref_count(b0), 2);

        alloc.free(b0);
        assert_eq!(alloc.ref_count(b0), 1);
        assert_eq!(alloc.num_free(), 2);

        alloc.free(b0);
        assert_eq!(alloc.ref_count(b0), 0);
        assert_eq!(alloc.num_free(), 3);

        alloc.free(b1);
        assert_eq!(alloc.num_free(), 4);
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(16);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);

        table.append_block(5);
        table.append_block(7);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.logical_to_physical(0), Some(5));
        assert_eq!(table.logical_to_physical(1), Some(7));
        assert_eq!(table.logical_to_physical(2), None);

        assert_eq!(table.token_to_block(0), (0, 0));
        assert_eq!(table.token_to_block(15), (0, 15));
        assert_eq!(table.token_to_block(16), (1, 0));
        assert_eq!(table.token_to_block(31), (1, 15));

        table.set_num_tokens(20);
        assert_eq!(table.num_tokens(), 20);
    }

    #[test]
    fn test_paged_kv_pool() {
        let mut pool = PagedKVPool::new(2, 4, 8, 16, 10);
        assert_eq!(pool.num_free_blocks(), 10);
        assert_eq!(pool.total_blocks(), 10);

        let blocks = pool.allocate_blocks(2).unwrap();
        let b0 = blocks[0];
        let b1 = blocks[1];

        let k: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..8).map(|i| (i + 10) as f32).collect();

        pool.write_kv(0, b0, 0, 0, &k, &v);
        pool.write_kv(0, b0, 1, 1, &k, &v);

        let read_k = pool.read_k(0, b0, 0, 0);
        let read_v = pool.read_v(0, b0, 0, 0);
        assert_eq!(read_k, &k[..]);
        assert_eq!(read_v, &v[..]);

        pool.free_blocks(&[b0, b1]);
        assert_eq!(pool.num_free_blocks(), 10);
        assert!(pool.memory_usage() > 0);
    }

    #[test]
    fn test_paged_sequence() {
        let mut pool = PagedKVPool::new(1, 1, 4, 8, 16);
        let mut seq = PagedSequence::new(0, 8);

        let k: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        seq.append_token(&mut pool, 0, 0, &k, &v).unwrap();
        seq.advance_token();

        let k2: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let v2: Vec<f32> = vec![50.0, 60.0, 70.0, 80.0];
        seq.append_token(&mut pool, 0, 0, &k2, &v2).unwrap();
        seq.advance_token();

        assert_eq!(seq.num_tokens, 2);

        let (gathered_k, gathered_v) = seq.get_kv_for_attention(&pool, 0, 0);
        assert_eq!(gathered_k[0..4], k[..]);
        assert_eq!(gathered_v[0..4], v[..]);
        assert_eq!(gathered_k[4..8], k2[..]);
        assert_eq!(gathered_v[4..8], v2[..]);

        seq.free(&mut pool);
        assert_eq!(pool.num_free_blocks(), 16);
    }

    #[test]
    fn test_copy_on_write() {
        let mut pool = PagedKVPool::new(1, 1, 4, 8, 16);
        let blocks = pool.allocate_blocks(2).unwrap();
        let src = blocks[0];
        let dst = blocks[1];

        let k: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        pool.write_kv(0, src, 0, 0, &k, &v);

        pool.copy_block(src, dst);

        let read_k = pool.read_k(0, dst, 0, 0);
        let read_v = pool.read_v(0, dst, 0, 0);
        assert_eq!(read_k, &k[..]);
        assert_eq!(read_v, &v[..]);

        pool.allocator_mut().increment_ref(src);
        assert_eq!(pool.allocator().ref_count(src), 2);

        pool.free_blocks(&[src, dst]);
    }

    #[test]
    fn test_memory_fragmentation() {
        let mut pool = PagedKVPool::new(1, 1, 4, 8, 10);
        let mut allocated = Vec::new();

        for _ in 0..10 {
            let blocks = pool.allocate_blocks(1).unwrap();
            allocated.push(blocks[0]);
        }
        assert_eq!(pool.num_free_blocks(), 0);
        assert!(pool.allocate_blocks(1).is_none());

        pool.free_blocks(&allocated[0..5]);
        assert_eq!(pool.num_free_blocks(), 5);

        let blocks = pool.allocate_blocks(5).unwrap();
        assert_eq!(pool.num_free_blocks(), 0);

        pool.free_blocks(&allocated[5..10]);
        pool.free_blocks(&blocks);
        assert_eq!(pool.num_free_blocks(), 10);
    }
}
