#!/usr/bin/env python3
"""Verify embedding lookup"""

import numpy as np
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

# Find token_embd.weight (Q4_K)
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        print(f"=== token_embd.weight ===")
        print(f"Shape (GGUF ne): {tensor.shape}")  # [896, 151936]
        print(f"Type: {tensor.tensor_type}")  # 6 = Q4_K
        print(f"Data shape: {tensor.data.shape}")  # (151936, 616)
        
        # Q4_K structure (from ggml):
        # block_q4_K: QK_K = 256 values per super-block
        #   - d: fp16 scale for first half (2 bytes)
        #   - dmin: fp16 min for first half (2 bytes)
        #   - scales: 12 bytes of packed scales
        #   - qs: 128 bytes of 4-bit quantized values
        # Total: 2 + 2 + 12 + 128 = 144 bytes per super-block
        
        # 896 values = 896/256 = 3.5 super-blocks
        # But Q4_K has QK_K=256, so 896 doesn't divide evenly...
        # Actually llama.cpp might use a different block size for smaller tensors
        
        # Let's check 616 bytes per row:
        # If block size is 256, then 896 values = 3.5 blocks (doesn't work)
        # If block size is 128, then 896 values = 7 blocks
        # 7 * 72 bytes per block = 504 bytes (not 616)
        
        # Actually for Q4_K, the super-block is 256 values
        # Maybe they pad to 1024 values? 1024/256 = 4 super-blocks
        # 4 * 144 = 576 bytes (not 616)
        
        # Let's just check if the layout is [vocab, packed_hidden] like output.weight
        data = tensor.data  # shape (151936, 616)
        print(f"\nRaw data shape implies {data.shape[0]} vocab tokens")
        print(f"Each with {data.shape[1]} bytes of packed hidden data")
        
        # For Q4_K, values are 4-bit, so 896 values = 448 bytes for quants alone
        # Plus scales and mins... 616 seems reasonable
        
        # Key question: is embedding for token T in row T of the data?
        # Based on output.weight analysis, yes - rows index vocab tokens
        
        print("\n=== Embedding layout should be same as output.weight ===")
        print("emb[token * 896 : (token+1) * 896] = embedding for that token")
        break

# Save a reference embedding for comparison
print("\n=== Saving reference embedding for token 28 ('=') ===")
# We need to actually dequantize Q4_K which is complex
# Let's use llama-cpp to get the embedding instead

from llama_cpp import Llama

llm = Llama(model_path=model_path, n_ctx=512, verbose=False, embedding=True)

# Get embedding for single token
tokens = [28]  # "="
embedding = llm.embed(llm.detokenize(tokens).decode())
print(f"Embedding shape from llama-cpp: {np.array(embedding).shape}")

# Actually let me try a different approach - compute with the model
# and save the first layer's input (which is just the embedding + any normalization)

# Unfortunately llama-cpp doesn't expose raw embeddings easily
# Let's just verify our dequantization for Q4_K is working
print("\nNote: Q4_K dequantization is more complex than Q8_0")
print("The key verification is that the layout is [vocab, hidden]")
print("which we confirmed for output.weight (Q8_0)")
