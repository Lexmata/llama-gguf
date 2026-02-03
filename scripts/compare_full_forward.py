#!/usr/bin/env python3
"""Compare full forward pass between llama-cpp-python and our implementation"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Load model with logits_all=True to get logits for all positions
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

# Test "1+1="
tokens = [16, 10, 16, 28]  # "1+1="
print(f"=== Testing tokens: {tokens} ===")
print("Expected: '2' (token 17) should be top prediction after '='")

llm.reset()
llm.eval(tokens)

# Get logits for the last position (after "=")
logits = np.array(llm._scores[-1])

print(f"\nLogits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

# Top predictions
top_idx = np.argsort(logits)[::-1][:10]
print("\nTop 10 predictions:")
for idx in top_idx:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  Token {idx} ({repr(tok)}): {logits[idx]:.4f}")

# Check token 17
token_17_rank = int(np.sum(logits > logits[17])) + 1
print(f"\nToken 17 ('2') logit: {logits[17]:.4f}, rank: {token_17_rank}")

# Also check numeric tokens
print("\nNumeric tokens:")
for i in range(15, 21):
    tok = llm.detokenize([i]).decode('utf-8', errors='replace')
    rank = int(np.sum(logits > logits[i])) + 1
    print(f"  Token {i} ('{tok}'): {logits[i]:.4f}, rank {rank}")

# Save logits for comparison
np.save('/tmp/llama_cpp_logits_1plus1.npy', logits)
print("\nSaved logits to /tmp/llama_cpp_logits_1plus1.npy")

# Also test single token for comparison
print("\n=== Single token '=' (28) ===")
llm.reset()
llm.eval([28])
logits_single = np.array(llm._scores[-1])
print(f"Logits stats: min={logits_single.min():.4f}, max={logits_single.max():.4f}, mean={logits_single.mean():.4f}")

top_idx = np.argsort(logits_single)[::-1][:5]
print("Top 5 predictions:")
for idx in top_idx:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  Token {idx} ({repr(tok)}): {logits_single[idx]:.4f}")
