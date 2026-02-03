#!/usr/bin/env python3
"""
Verify how GGUF stores embeddings - row-major or column-major?
"""

import numpy as np
from gguf import GGUFReader

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def dequantize_q5_0_block(block_data):
    """Dequantize a single Q5_0 block (32 elements from 22 bytes)"""
    scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
    qh = np.frombuffer(block_data[2:6], dtype=np.uint8)
    qh_bits = np.unpackbits(qh, bitorder='little')[:32]
    ql = np.frombuffer(block_data[6:22], dtype=np.uint8)
    
    result = np.zeros(32, dtype=np.float32)
    for i in range(32):
        if i < 16:
            lo = ql[i] & 0x0F
        else:
            lo = (ql[i - 16] >> 4) & 0x0F
        hi = qh_bits[i]
        q = (hi << 4) | lo
        result[i] = float(scale) * (q - 16)
    return result

def main():
    reader = GGUFReader(MODEL_PATH)
    
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t
    
    emb_tensor = tensors["token_embd.weight"]
    print(f"token_embd.weight:")
    print(f"  GGUF type: {emb_tensor.tensor_type}")
    print(f"  GGUF shape: {emb_tensor.shape}")  # This shows [896, 151936]
    print(f"  Data size: {len(emb_tensor.data)} bytes")
    
    hidden_size = 896
    vocab_size = 151936
    block_size = 32
    bytes_per_block = 22
    
    # Q5_0 compresses 32 elements into 22 bytes
    # Total elements = hidden_size * vocab_size = 136,134,656
    # Total blocks = 136,134,656 / 32 = 4,254,208
    # Total bytes = 4,254,208 * 22 = 93,592,576
    total_elements = hidden_size * vocab_size
    total_blocks = total_elements // block_size
    expected_bytes = total_blocks * bytes_per_block
    print(f"  Expected bytes: {expected_bytes}")
    
    # In GGUF, the shape [896, 151936] with GGML convention means:
    # - First dim (896) is the row count (hidden_size)
    # - Second dim (151936) is the column count (vocab_size)
    # But GGML uses row-major storage, so data is stored as [rows, cols] = [hidden, vocab]
    # 
    # WAIT - that would mean each "row" is a vocab entry, and we have 896 such rows.
    # That doesn't match typical embedding tables!
    #
    # Let me check what makes sense:
    # - Embeddings should be [vocab_size, hidden_size] in numpy terms
    # - Each token maps to a hidden_size vector
    #
    # If GGUF shape is [896, 151936], let's see both interpretations:
    
    print("\n=== Testing Interpretation 1: Row-major [896, 151936] ===")
    print("Each row is 151936 elements, there are 896 rows")
    print("This would mean token embedding is scattered across rows")
    
    # Row 0 would be elements 0..151935
    # Token 28 would be at row r, column 28, i.e. element r*151936 + 28
    
    print("\n=== Testing Interpretation 2: Transposed [151936, 896] ===")
    print("Each row is 896 elements (hidden_size), there are 151936 rows (vocab)")
    print("This is the standard embedding layout")
    
    # In this case, token 28's embedding is at rows starting at 28*896
    
    # Let's try interpretation 2 (standard embedding layout)
    # Each token's embedding is hidden_size elements = 896 elements
    # = 28 Q5_0 blocks = 28 * 22 = 616 bytes per token
    
    token_id = 28
    blocks_per_token = hidden_size // block_size  # 896 / 32 = 28
    bytes_per_token = blocks_per_token * bytes_per_block  # 28 * 22 = 616
    
    # Token 28's data starts at byte 28 * 616 = 17248
    token_offset = token_id * bytes_per_token
    token_data = bytes(emb_tensor.data[token_offset:token_offset + bytes_per_token])
    
    # Dequantize
    embedding_v2 = np.zeros(hidden_size, dtype=np.float32)
    for b in range(blocks_per_token):
        block_data = token_data[b * bytes_per_block:(b + 1) * bytes_per_block]
        embedding_v2[b * block_size:(b + 1) * block_size] = dequantize_q5_0_block(block_data)
    
    print(f"\nInterpretation 2 - Token {token_id} embedding:")
    print(f"  min={embedding_v2.min():.6f}, max={embedding_v2.max():.6f}")
    print(f"  first 10: {embedding_v2[:10]}")
    
    # Compare with Rust
    rust_first10 = [0.0055236816, -0.016571045, -0.016571045, 0.019332886, 0.019332886,
                   -0.008285522, 0.008285522, -0.019332886, 0.0055236816, -0.0055236816]
    
    print(f"\n  Rust first 10: {rust_first10}")
    
    if np.allclose(embedding_v2[:10], rust_first10, rtol=1e-4):
        print("  MATCH! Interpretation 2 is correct.")
    else:
        print("  MISMATCH!")
        
    # What if the data is quantized in blocks along a different axis?
    # Let's try treating it as column-major where each column is a token
    
    print("\n=== Testing Interpretation 3: Column-major data layout ===")
    # In Q5_0, data is organized in blocks of 32 elements
    # If layout is column-major [896, 151936], each column has 896 elements
    # So first 896 elements (28 blocks) are token 0, next 896 are token 1, etc.
    # This is actually the same as interpretation 2!
    
    # Actually, let me check if maybe the blocks are interleaved differently
    # What if blocks span across tokens?
    
    # Total data = 93,592,576 bytes
    # Total blocks = 4,254,208
    # If blocks are stored per-token: token 0 gets blocks 0-27, token 1 gets 28-55, etc.
    # Token 28 would get blocks 28*28 to 28*28+27 = 784 to 811
    
    # But wait - let me double check my Python dequantization
    print("\n=== Checking Q5_0 dequantization ===")
    # Read first block
    first_block = bytes(emb_tensor.data[:22])
    first_values = dequantize_q5_0_block(first_block)
    print(f"First block (32 values): {first_values[:10]}...")
    
    # These should be the first 32 values of token 0's embedding
    # If Rust token 0 first 5 are: [-0.010192871, 0.040771484, 0.010192871, -0.0, -0.028030396]
    rust_token0_first5 = [-0.010192871, 0.040771484, 0.010192871, -0.0, -0.028030396]
    
    print(f"Rust token 0 first 5: {rust_token0_first5}")
    if np.allclose(first_values[:5], rust_token0_first5, rtol=1e-3):
        print("  MATCH! First block is token 0's embedding.")
    else:
        print("  MISMATCH!")

if __name__ == "__main__":
    main()
