#!/usr/bin/env python3
"""Verify embedding values by manually dequantizing"""

import numpy as np
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

# Q4_K dequantization is complex, but let's try to understand the layout first
# by looking at the raw data shape

for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        print(f"=== token_embd.weight ===")
        print(f"Shape (GGUF ne): {tensor.shape}")  # [896, 151936]
        print(f"Type: {tensor.tensor_type}")  # 6 = Q4_K
        print(f"Data shape: {tensor.data.shape}")  # (151936, 616)
        
        # The raw data shape (151936, 616) suggests:
        # - 151936 rows, one per vocab token
        # - 616 bytes per row = packed hidden dimension
        
        # For Q4_K with 896 hidden dim:
        # - Q4_K uses super-blocks of 256 values
        # - 896 / 256 = 3.5, so need to understand how GGUF handles this
        
        # Actually, looking at bytes: 616 bytes
        # Q4_K block size 256: 144 bytes per 256 values
        # 896 values = ?
        # 
        # Let me check if maybe they pad to 1024:
        # 1024 values = 4 blocks * 144 = 576 bytes (not 616)
        #
        # Or maybe the format is different for sub-super-block sizes
        
        # Let's compare with output.weight (Q8_0) which we know works:
        # output.weight: shape (151936, 952) where 952 = 28 Q8_0 blocks * 34 bytes
        # 28 * 32 = 896 elements
        
        # For Q4_K with 616 bytes per row:
        # If we assume similar structure (row = vocab entry's hidden dim weights)
        # then we need to figure out how 896 floats pack into 616 bytes
        
        # Q4_K: 4 bits per value = 0.5 bytes per value
        # 896 * 0.5 = 448 bytes just for quants
        # Plus scales, mins, etc = ~616 bytes seems reasonable
        
        print("\nLayout analysis:")
        print("  Row count (151936) = vocab_size")
        print("  Each row = packed hidden dim (896 values)")
        print("  emb[token] should come from data[token, :]")
        print("  After dequantization: emb[token * 896 : (token+1) * 896]")
        
        # Let's manually extract embedding for token 28 by finding its row
        row_28 = tensor.data[28]
        print(f"\nToken 28 raw data: {len(row_28)} bytes")
        print(f"First 10 bytes: {list(row_28[:10])}")
        
        # Save for comparison - but we can't easily dequantize Q4_K in Python
        # without the proper implementation
        np.save('/tmp/token_embd_row28_raw.npy', row_28)
        print("Saved raw bytes to /tmp/token_embd_row28_raw.npy")
        
        break

# Alternative: use llama-cpp to get the embedding and save it
print("\n=== Getting reference embedding from llama-cpp ===")
from llama_cpp import Llama

llm = Llama(model_path=model_path, n_ctx=512, verbose=False, embedding=True)

# Try to get embedding for token 28
# Note: llama-cpp's embed() function works with text, not tokens
# We need a different approach

# Let's see if we can access internal state
# Actually, let's just save our computed embedding stats and compare

print("\n=== Comparing embedding statistics ===")
print("Our implementation embedding for token 28:")
print("  First 5: [0.0055, -0.0166, -0.0166, 0.0193, 0.0193]")
print("  Sum: -0.230461")
print("  (values from debug_layer0.rs)")

# These values seem reasonable for an embedding
# The question is whether they match what llama-cpp uses internally

# Let me try a different approach: compute logits with identity hidden state
# If we pass all 1s through final layer norm and output, we should get weight sums
# This might help verify the matrix multiplication
