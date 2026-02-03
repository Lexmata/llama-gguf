#!/usr/bin/env python3
"""Extract logits from llama-cpp-python for comparison."""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

print("Loading model with logits_all=True...")
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

# Test prompt
tokens = [16, 10, 16, 28]  # "1+1="
print(f"Input tokens: {tokens}")

# Evaluate
llm.reset()
llm.eval(tokens)

# Get logits for the last token
logits = np.array(llm._scores[-1])

print(f"\nLogits shape: {logits.shape}")
print(f"Logits min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")

# Token 17 is "2"
print(f"\nToken 17 ('2') logit: {logits[17]:.6f}")

# Get top 10 predictions
top_indices = np.argsort(logits)[::-1][:10]
print("\nTop 10 predictions:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Token {idx}: logit = {logits[idx]:.4f}")

# Find rank of token 17
rank_17 = np.where(np.argsort(logits)[::-1] == 17)[0][0] + 1
print(f"\nToken 17 ('2') rank: {rank_17}")

# Save logits to file for comparison
np.save('/tmp/llama_cpp_logits.npy', logits)
print(f"\nSaved logits to /tmp/llama_cpp_logits.npy")

# Also print some specific logits for comparison
print("\nFirst 20 logits:")
print(logits[:20])
print("\nLogits at indices 15-20 (around token 17):")
print(logits[15:21])
