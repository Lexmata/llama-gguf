#!/usr/bin/env python3
"""Compare layer 0 outputs between llama-cpp-python internals and our computation."""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

# Try to get access to internal state
# Note: llama-cpp-python doesn't expose internal hidden states directly
# But we can try to understand what's happening

# Test with single token first
tokens = [28]  # "="
print(f"Input tokens: {tokens}")

llm.reset()
llm.eval(tokens)
logits_single = np.array(llm._scores[-1])

print(f"\nSingle token '=' (token 28) logits:")
print(f"  min: {logits_single.min():.4f}, max: {logits_single.max():.4f}")
print(f"  Token 17 ('2') logit: {logits_single[17]:.4f}")
print(f"  First 10: {logits_single[:10]}")

# Get top 5
top_idx = np.argsort(logits_single)[::-1][:5]
print(f"  Top 5: {[(idx, logits_single[idx]) for idx in top_idx]}")

# Compare with our Rust output for single token
# From our earlier tests:
# Single token "=" gives Token 17 rank around 3880 in Rust
# But llama-cpp should be different

# Also check the embedding itself
# We can do this by using the tokenizer
print("\n=== Token Embedding Check ===")
from gguf import GGUFReader
reader = GGUFReader(model_path)

# Find embedding tensor
for tensor in reader.tensors:
    if tensor.name == "token_embd.weight":
        print(f"Embedding tensor shape: {tensor.shape}")
        print(f"Embedding tensor type: {tensor.tensor_type}")
        
        # The shape should be [hidden_size, vocab_size] = [896, 151936]
        # Check if we can access the data
        data = tensor.data
        print(f"Data shape: {data.shape}")
        
        # For token 28, the embedding should be at column 28
        # In GGUF column-major, this means indices [28*896 : (28+1)*896]
        # But the quantized data needs dequantization
        break
