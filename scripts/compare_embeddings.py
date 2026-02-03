#!/usr/bin/env python3
"""Compare embeddings and intermediate values with llama-cpp-python."""

import numpy as np

# Get embeddings from the GGUF file
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

# Find embedding tensor
for tensor in reader.tensors:
    if tensor.name == "token_embd.weight":
        print(f"Embedding tensor: {tensor.name}")
        print(f"  Shape: {tensor.shape}")  # Should be [896, 151936]
        print(f"  Type: {tensor.tensor_type}")
        
        # The GGUF stores in [hidden_size, vocab_size] but we need to access per token
        # For quantized data, we need to dequantize
        
        # Try to get the raw data shape
        data = tensor.data
        print(f"  Raw data shape: {data.shape}")
        break

# Compare logits from llama-cpp-python
from llama_cpp import Llama
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

# Single token test
tokens = [16, 10, 16, 28]  # "1+1="
print(f"\nTokens: {tokens}")

llm.reset()
llm.eval(tokens)
logits = np.array(llm._scores[-1])

print(f"\nllama-cpp-python logits:")
print(f"  Shape: {logits.shape}")
print(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}")
print(f"  Token 17 logit: {logits[17]:.4f}")

# Get top 10 predictions
top_idx = np.argsort(logits)[::-1][:10]
print(f"\n  Top 10 predictions:")
for rank, idx in enumerate(top_idx):
    print(f"    {rank+1}. Token {idx}: {logits[idx]:.4f}")

# Check if there's a pattern with the top tokens
print(f"\n  Tokens 16-19 logits: {logits[16:20]}")
print(f"  These correspond to: 1, 2, 3, 4 (numeric tokens)")
