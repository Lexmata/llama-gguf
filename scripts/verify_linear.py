#!/usr/bin/env python3
"""
Verify the linear layer (y = x @ W + b) computation against gguf data.
We'll compute manually and compare with what llama-cpp produces.
"""

import numpy as np
from gguf import GGUFReader
import struct

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def dequantize_q5_0_block(block_data):
    """Dequantize a single Q5_0 block (32 elements from 22 bytes)"""
    # Q5_0: 2 bytes f16 scale + 4 bytes high bits + 16 bytes low bits
    scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
    
    # High bits: 4 bytes = 32 bits
    qh = np.frombuffer(block_data[2:6], dtype=np.uint8)
    qh_bits = np.unpackbits(qh, bitorder='little')[:32]
    
    # Low bits: 16 bytes = 32 4-bit values
    ql = np.frombuffer(block_data[6:22], dtype=np.uint8)
    
    result = np.zeros(32, dtype=np.float32)
    for i in range(32):
        # Get low 4 bits
        if i < 16:
            lo = ql[i] & 0x0F
        else:
            lo = (ql[i - 16] >> 4) & 0x0F
        
        # Get high bit
        hi = qh_bits[i]
        
        # Combine to get 5-bit value, then dequantize
        q = (hi << 4) | lo
        result[i] = float(scale) * (q - 16)
    
    return result

def main():
    print("Loading GGUF file...")
    reader = GGUFReader(MODEL_PATH)
    
    # Build tensor dictionary
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t
    
    # Get embedding for token 28 ('=')
    # Token embedding is Q5_0 with shape [896, 151936]
    # In GGUF row-major convention, this means [vocab, hidden] = [151936, 896]
    
    token_id = 28
    hidden_size = 896
    
    emb_tensor = tensors["token_embd.weight"]
    print(f"token_embd.weight: type={emb_tensor.tensor_type}, shape={emb_tensor.shape}")
    
    # Embedding is stored as [vocab_size, hidden_size] in row-major order
    # Each row is hidden_size elements = 896 elements
    # Q5_0 block is 32 elements, so each row is 896/32 = 28 blocks
    # Each block is 22 bytes, so each row is 28 * 22 = 616 bytes
    
    blocks_per_row = hidden_size // 32
    bytes_per_row = blocks_per_row * 22
    
    row_start = token_id * bytes_per_row
    row_data = bytes(emb_tensor.data[row_start:row_start + bytes_per_row])
    
    # Dequantize this row
    embedding = np.zeros(hidden_size, dtype=np.float32)
    for block_idx in range(blocks_per_row):
        block_start = block_idx * 22
        block_data = row_data[block_start:block_start + 22]
        embedding[block_idx * 32:(block_idx + 1) * 32] = dequantize_q5_0_block(block_data)
    
    print(f"\nEmbedding for token {token_id}:")
    print(f"  min={embedding.min():.6f}, max={embedding.max():.6f}")
    print(f"  first 10: {embedding[:10]}")
    
    # Now let's verify this matches our Rust output
    print("\n=== Rust embedding for comparison ===")
    print("  first 10: [0.0055236816, -0.016571045, -0.016571045, 0.019332886, 0.019332886, ...]")
    
    # Check if they match
    rust_first5 = [0.0055236816, -0.016571045, -0.016571045, 0.019332886, 0.019332886]
    if np.allclose(embedding[:5], rust_first5, rtol=1e-4):
        print("  MATCH! Embedding values match Rust.")
    else:
        print(f"  MISMATCH! Python: {embedding[:5]}")
        
    # Now let's check the attention norm weights
    norm_tensor = tensors["blk.0.attn_norm.weight"]
    print(f"\nblk.0.attn_norm.weight: type={norm_tensor.tensor_type}, shape={norm_tensor.shape}")
    norm_weight = np.frombuffer(bytes(norm_tensor.data), dtype=np.float32)
    print(f"  min={norm_weight.min():.6f}, max={norm_weight.max():.6f}")
    print(f"  first 10: {norm_weight[:10]}")
    
    # Compute RMSNorm manually
    eps = 1e-6
    rms = np.sqrt(np.mean(embedding ** 2) + eps)
    normed = (embedding / rms) * norm_weight
    
    print(f"\nAfter RMSNorm:")
    print(f"  min={normed.min():.6f}, max={normed.max():.6f}")
    print(f"  first 10: {normed[:10]}")
    
    # Rust output for comparison
    print("\n=== Rust after RMSNorm for comparison ===")
    print("  first 10: [-0.033233173, -0.09551787, 0.077177264, 0.16835451, ...]")
    
    rust_norm_first4 = [-0.033233173, -0.09551787, 0.077177264, 0.16835451]
    if np.allclose(normed[:4], rust_norm_first4, rtol=1e-2):
        print("  Values approximately match Rust!")
    else:
        print(f"  MISMATCH! Python: {normed[:4]}")
        
    # Now let's check Q bias
    bq_tensor = tensors["blk.0.attn_q.bias"]
    print(f"\nblk.0.attn_q.bias: type={bq_tensor.tensor_type}, shape={bq_tensor.shape}")
    bq = np.frombuffer(bytes(bq_tensor.data), dtype=np.float32)
    print(f"  min={bq.min():.2f}, max={bq.max():.2f}")
    print(f"  first 10: {bq[:10]}")

if __name__ == "__main__":
    main()
