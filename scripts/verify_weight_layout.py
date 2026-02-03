#!/usr/bin/env python3
"""Verify weight matrix layouts in GGUF"""

import numpy as np
from gguf import GGUFReader, GGUFValueType
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

print("=== GGUF Tensor Info ===")
for tensor in reader.tensors:
    if tensor.name in ['token_embd.weight', 'output.weight', 'output_norm.weight', 'blk.0.attn_q.weight']:
        print(f"\n{tensor.name}:")
        print(f"  Shape (ne): {tensor.shape}")
        print(f"  Type: {tensor.tensor_type}")
        print(f"  Data shape: {tensor.data.shape}")
        
        # For output.weight specifically
        if tensor.name == 'output.weight':
            # The shape is [896, 151936] in GGUF
            # Let's check if the actual number of elements matches
            hidden_size = 896
            vocab_size = 151936
            expected_elements = hidden_size * vocab_size
            print(f"  Expected elements (896 x 151936): {expected_elements}")

# Now let's use llama-cpp to get the embedding and output weights
# and verify the matrix multiplication
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, embedding=True)

# Get embedding for token 28 ("=")
# Unfortunately llama-cpp doesn't expose raw weights easily
# But we can verify by checking the prediction matches

# Let's manually dequantize and verify a small portion
for tensor in reader.tensors:
    if tensor.name == 'output.weight':
        print(f"\n=== output.weight details ===")
        print(f"  Type: {tensor.tensor_type} (Q8_0 = 8)")
        # Q8_0 has block size 32
        # Each block: 32 int8 quants + 1 fp16 scale = 34 bytes
        
        # Shape is [896, 151936]
        # In GGUF, this means:
        # - ne[0] = 896 (first dimension)
        # - ne[1] = 151936 (second dimension)
        # - The data is stored column-major
        # - For each column j, elements are data[j*896:(j+1)*896]
        
        # Q8_0: block_size = 32, so 896/32 = 28 blocks per column
        # Each block = 32 + 2 = 34 bytes
        # Total = 28 * 34 * 151936 bytes
        blocks_per_column = 896 // 32  # 28
        bytes_per_block = 32 + 2  # 34
        expected_bytes = blocks_per_column * bytes_per_block * 151936
        print(f"  Expected raw bytes: {expected_bytes}")
        print(f"  Actual raw data len: {len(tensor.data.tobytes())}")
        
        # Actually the raw data shape gives us the answer
        # data.shape = (151936, 952) for Q8_0
        # 952 = 28 blocks * 34 bytes per block = 952 bytes per row
        # This confirms the layout

print("\n=== Verifying matrix multiplication order ===")
# The key question: for logits computation
# Option A: logits[j] = sum_i(hidden[i] * W[i + j*hidden_size])  -- j indexes vocab
# Option B: logits[j] = sum_i(hidden[i] * W[j + i*vocab_size])  -- i indexes vocab

# If shape is [896, 151936]:
# - ne[0]=896 is the inner dimension
# - ne[1]=151936 is the outer dimension  
# - Column j contains elements W[0,j], W[1,j], ..., W[895,j]
# - These are stored contiguously at W[j*896 : (j+1)*896]
#
# So for hidden @ W where hidden is [896] and W is [896, 151936]:
# logits[j] = sum_i(hidden[i] * W[i,j]) = sum_i(hidden[i] * data[i + j*896])
#
# This is exactly what vec_mat does. Let's verify.

print("\nAccording to GGUF spec:")
print("  - ne[0] is the fastest-varying dimension (contiguous)")
print("  - For shape [896, 151936], each column of 896 elements is contiguous")
print("  - output.weight[i, j] = data[i + j * 896]")
print("  - logits[j] = sum_i(hidden[i] * output.weight[i, j])")
print("  - This matches vec_mat implementation")

# Double check by comparing small values
print("\n=== Checking first few dequantized values ===")
for tensor in reader.tensors:
    if tensor.name == 'output.weight':
        # Q8_0 dequantization
        data = tensor.data
        print(f"  Raw data dtype: {data.dtype}")
        print(f"  Raw data shape: {data.shape}")
        
        # For Q8_0, the data is stored as:
        # For each block of 32 values:
        #   - 1 x fp16 scale (2 bytes)
        #   - 32 x int8 quants (32 bytes)
        # Total = 34 bytes per block
        # But the shape (151936, 952) suggests different layout
        # 952 / 34 = 28 blocks per "row" (where row = one vocab entry's weights)
        # 28 * 32 = 896 elements = hidden_size  -- This matches!
        
        # So the layout is:
        # data[j, :] contains the 896 weights for vocab token j, packed as Q8_0 blocks
        # This is ROW-major with respect to vocab!
        
        print(f"\n  IMPORTANT: The shape (151936, 952) means:")
        print(f"    - 151936 rows (one per vocab token)")
        print(f"    - 952 bytes per row = 28 Q8_0 blocks * 34 bytes/block")
        print(f"    - 28 blocks * 32 elements/block = 896 elements")
        print(f"    - So data[vocab_idx, :] = packed Q8_0 for hidden weights of that vocab token")
        print(f"\n  This means the weight is stored as [vocab_size, hidden_size] in memory!")
        print(f"  But GGUF reports shape as [896, 151936]...")
        print(f"  The GGUF shape [ne0, ne1] = [896, 151936] might mean:")
        print(f"    - GGUF convention: ne[0] is inner dimension")
        print(f"    - But physical storage could be [ne1, packed_ne0]")
